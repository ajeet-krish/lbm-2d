"""Physics-Informed Neural Network (PINN) for 2D incompressible flow.

Two architectures:
  1. PINN:           (x, y) -> (u, v, p)  -- single-case, fixed Re
  2. ParametricPINN: (x, y, p1, ...) -> (u, v, p)  -- parametric surrogate
                                  with frozen Fourier feature embedding

Input coordinates are normalized to [-1, 1]. For ParametricPINN, extra
parameters (Re, geometry dims) are concatenated after the Fourier features
and independently normalized to [0, 1].

The Fourier feature layer lifts spatial coordinates into a high-frequency
space, mitigating the spectral bias of tanh MLPs (Tancik et al. 2020).
The B matrix is frozen (sampled once from N(0, sigma^2)).

Torch-only. MPS backend recommended for Apple Silicon.
"""

import math
import torch
import torch.nn as nn


class FourierFeatureLayer(nn.Module):
    """Random Fourier feature mapping for spatial coordinates (frozen B).

    Maps (x, y) -> [cos(2*pi*B_0*x), sin(2*pi*B_0*x),
                    cos(2*pi*B_1*x), sin(2*pi*B_1*x),
                    cos(2*pi*B_0*y), sin(2*pi*B_0*y), ...]
    where B is a frozen random matrix drawn from N(0, sigma^2). Applying this
    before the MLP lifts coordinates into a high-frequency space, which
    breaks the spectral bias of tanh networks and lets them represent thin
    boundary layers.

    Only spatial coordinates are Fourier-encoded. Physical parameters are
    concatenated after the Fourier features by the calling network.
    """

    def __init__(self, in_dim: int = 2, n_freqs: int = 128, sigma: float = 5.0):
        super().__init__()
        # B is a buffer (frozen, not a trainable parameter)
        B = torch.randn(n_freqs, in_dim) * sigma
        self.register_buffer("B", B)
        self.in_dim = in_dim
        self.n_freqs = n_freqs
        self.sigma = sigma
        self.out_dim = 2 * n_freqs * in_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode spatial coordinates.

        Args:
            x: (N, in_dim) normalized spatial coordinates.

        Returns:
            (N, 2*n_freqs*in_dim) Fourier features.
        """
        feats = []
        two_pi = 2.0 * math.pi
        for d in range(self.in_dim):
            # (N, n_freqs): 2*pi * x_d * B[:, d]
            proj = two_pi * x[:, d:d + 1] @ self.B[:, d:d + 1].T
            feats.append(torch.cos(proj))
            feats.append(torch.sin(proj))
        return torch.cat(feats, dim=1)


class MultiScaleFourierLayer(nn.Module):
    """Multi-scale random Fourier feature mapping for spatial coordinates.

    Concatenates multiple single-scale Fourier feature bands with different
    frequency scalings (sigma values). This gives the network access to
    low-frequency (bulk flow), medium-frequency (vortex), and high-frequency
    (boundary layer) representation simultaneously, mitigating the spectral
    bias of tanh MLPs more effectively than a single band.

    Input: (N, in_dim) spatial coordinates.
    Output: (N, 2 * n_freqs * in_dim * n_scales) multi-scale Fourier features.

    Args:
        in_dim:    Spatial input dimension (default 2).
        n_freqs:   Frequencies per scale band.
        sigmas:    List of standard-deviations for each scale band.
    """

    def __init__(self, in_dim: int = 2, n_freqs: int = 128,
                 sigmas: tuple = (1.0, 5.0, 20.0)):
        super().__init__()
        self.in_dim = in_dim
        self.n_freqs = n_freqs
        self.sigmas = tuple(sigmas)
        self.out_dim = 2 * n_freqs * in_dim * len(self.sigmas)
        # One frozen random matrix per scale band.
        self.Bs = nn.ParameterList([
            nn.Parameter(torch.randn(n_freqs, in_dim) * s, requires_grad=False)
            for s in self.sigmas
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode spatial coordinates at multiple frequency scales."""
        feats = []
        two_pi = 2.0 * math.pi
        for B in self.Bs:
            band = []
            for d in range(self.in_dim):
                proj = two_pi * x[:, d:d + 1] @ B[:, d:d + 1].T
                band.append(torch.cos(proj))
                band.append(torch.sin(proj))
            feats.append(torch.cat(band, dim=1))
        return torch.cat(feats, dim=1)


class PINN(nn.Module):
    """Steady-state PINN: (x, y) -> (u, v, p)."""

    def __init__(self, hidden: int = 64, n_layers: int = 8):
        super().__init__()
        layers = [nn.Linear(2, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, 3))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            xy: (N, 2) normalized coordinates in [-1, 1].

        Returns:
            (N, 3) tensor of [u, v, p].
        """
        return self.net(xy)


class ParametricPINN(nn.Module):
    """Parametric PINN with Fourier feature embedding: (x, y, p1, ...) -> (u, v, p).

    Spatial coordinates (x, y) are passed through a frozen random Fourier
    feature layer (single-scale or multi-scale) before the MLP; physical
    parameters are concatenated after the Fourier features.

    Input dimension = 2 (spatial) + n_params (physical parameters).
    MLP input dimension = 2*n_freqs*2 (Fourier features) + n_params.
    Parameters should be pre-normalized to [0, 1] before passing to forward().

    Args:
        n_params:  Number of physical parameters (Re, geometry dims, etc.).
        hidden:    MLP hidden width.
        n_layers:  Number of MLP hidden layers.
        n_freqs:   Number of Fourier frequencies (m). Default 128.
        sigma:     Fourier feature frequency scale. Default 5.0.
        sigmas:    If set, use multi-scale Fourier features with these bands.
                   Overrides sigma. Example: (1.0, 5.0, 20.0).
    """

    def __init__(self, n_params: int = 1, hidden: int = 256, n_layers: int = 8,
                 n_freqs: int = 128, sigma: float = 5.0, sigmas=None):
        super().__init__()
        self.n_params = n_params
        self.n_freqs = n_freqs
        self.sigma = sigma if sigmas is None else sigmas
        self.sigmas = sigmas

        if sigmas is not None:
            self.fourier = MultiScaleFourierLayer(
                in_dim=2, n_freqs=n_freqs, sigmas=sigmas)
        else:
            self.fourier = FourierFeatureLayer(
                in_dim=2, n_freqs=n_freqs, sigma=sigma)

        in_dim = self.fourier.out_dim + n_params
        layers = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, 3))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (N, 2+n_params) tensor. First 2 columns are normalized
               coordinates [-1, 1], remaining columns are normalized
               physical parameters [0, 1].

        Returns:
            (N, 3) tensor of [u, v, p].
        """
        spatial = x[:, :2]
        params = x[:, 2:]
        fourier_feats = self.fourier(spatial)
        inp = torch.cat([fourier_feats, params], dim=1)
        return self.net(inp)


def predict(model, xy: torch.Tensor):
    """Split output into u, v, p."""
    out = model(xy)
    return out[:, 0], out[:, 1], out[:, 2]
