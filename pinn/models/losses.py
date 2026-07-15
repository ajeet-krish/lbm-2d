"""Loss functions for the PINN: PDE residual, boundary conditions, data.

All derivatives are computed via torch.autograd from the network output.

PDE: Steady incompressible Navier-Stokes (lattice units)
  Continuity:  du/dx + dv/dy = 0
  x-momentum:  u du/dx + v du/dy + dp/dx - (1/Re)(d2u/dx2 + d2u/dy2) = 0
  y-momentum:  u dv/dx + v dv/dy + dp/dy - (1/Re)(d2v/dx2 + d2v/dy2) = 0

Boundary conditions vary by case:
  Cylinder:  inflow + outlet + walls + cylinder surface (no-slip)
  Cavity:    all walls (no-slip) + moving lid (top wall, u = u_lid)

Data loss: MSE on sparse sensor subsamples of solver u, v (and optionally p).
"""

import torch
import torch.nn.functional as F


def _grad(y: torch.Tensor, x: torch.Tensor, create_graph: bool = True):
    """Compute dy/dx with shape matching x. Returns (N, dx_cols)."""
    if y.dim() == 1:
        y = y.unsqueeze(1)
    return torch.autograd.grad(
        y, x,
        grad_outputs=torch.ones_like(y),
        create_graph=create_graph,
        retain_graph=True,
    )[0]


# --------------------------------------------------------------------------
# PDE residual
# --------------------------------------------------------------------------
def pde_loss(model, collocation_xy: torch.Tensor, re: float, u_inflow: float):
    """PDE residual at collocation points for fixed-Re case.

    Args:
        model:  PINN (N, 2) -> (N, 3) or ParametricPINN (N, 2+n) -> (N, 3).
        collocation_xy: (Nc, D) coords (+ optional params), requires_grad=True.
        re: Reynolds number (scalar, used for nu = 1/re).
        u_inflow: Inflow velocity (lattice units, typically 0.1).
    """
    xy = collocation_xy.clone().detach().requires_grad_(True)
    out = model(xy)
    u, v, p = out[:, 0], out[:, 1], out[:, 2]

    du = _grad(u, xy)
    dv = _grad(v, xy)
    dp = _grad(p, xy)

    du_dx, du_dy = du[:, 0], du[:, 1]
    dv_dx, dv_dy = dv[:, 0], dv[:, 1]
    dp_dx, dp_dy = dp[:, 0], dp[:, 1]

    d2u = _grad(du_dx, xy)
    d2v = _grad(dv_dx, xy)
    d2u_dx2, d2u_dy2 = d2u[:, 0], d2u[:, 1]
    d2v_dx2, d2v_dy2 = d2v[:, 0], d2v[:, 1]

    nu = 1.0 / re

    f_continuity = du_dx + dv_dy
    f_x = u * du_dx + v * du_dy + dp_dx - nu * (d2u_dx2 + d2u_dy2)
    f_y = u * dv_dx + v * dv_dy + dp_dy - nu * (d2v_dx2 + d2v_dy2)

    return (F.mse_loss(f_continuity, torch.zeros_like(f_continuity))
            + F.mse_loss(f_x, torch.zeros_like(f_x))
            + F.mse_loss(f_y, torch.zeros_like(f_y)))


def pde_loss_multi_re(model, colloc_by_re: dict, device):
    """PDE residual averaged across multiple Reynolds numbers.

    Args:
        model: ParametricPINN (N, 2+n_params) -> (N, 3).
        colloc_by_re: dict mapping re_value -> (Nc, 2+n_params) normalized
            coords with Re already appended as the last column.
        device: torch device.

    Returns:
        scalar PDE loss (mean of per-Re losses).
    """
    total = 0.0
    for re, xy_np in colloc_by_re.items():
        inp = torch.from_numpy(xy_np).float().to(device).clone().detach().requires_grad_(True)
        out = model(inp)
        u, v, p = out[:, 0], out[:, 1], out[:, 2]

        du = _grad(u, inp)
        dv = _grad(v, inp)
        dp = _grad(p, inp)
        du_dx, du_dy = du[:, 0], du[:, 1]
        dv_dx, dv_dy = dv[:, 0], dv[:, 1]
        dp_dx, dp_dy = dp[:, 0], dp[:, 1]

        d2u = _grad(du_dx, inp)
        d2v = _grad(dv_dx, inp)
        d2u_dx2, d2u_dy2 = d2u[:, 0], d2u[:, 1]
        d2v_dx2, d2v_dy2 = d2v[:, 0], d2v[:, 1]

        nu = 1.0 / re
        f_cont = du_dx + dv_dy
        f_x = u * du_dx + v * du_dy + dp_dx - nu * (d2u_dx2 + d2u_dy2)
        f_y = u * dv_dx + v * dv_dy + dp_dy - nu * (d2v_dx2 + d2v_dy2)

        L = (F.mse_loss(f_cont, torch.zeros_like(f_cont))
             + F.mse_loss(f_x, torch.zeros_like(f_x))
             + F.mse_loss(f_y, torch.zeros_like(f_y)))
        total = total + L
    return total / len(colloc_by_re)


# --------------------------------------------------------------------------
# Cavity physics: combined PDE + pressure-Poisson residuals (single forward)
# --------------------------------------------------------------------------
def cavity_physics_loss(model, colloc_xy: torch.Tensor, re: float):
    """Combined steady-NS PDE residual + pressure-Poisson residual (single Re).

    Returns (L_pde, L_pp) from a single forward pass. The pressure-Poisson
    term Laplacian(p) + div(u·grad u) couples p to the velocity field. NOTE:
    this is a consequence of momentum + continuity, so it acts as an auxiliary
    regularizer; pressure variation must also be driven by data supervision.
    """
    xy = colloc_xy.clone().detach().requires_grad_(True)
    out = model(xy)
    u, v, p = out[:, 0], out[:, 1], out[:, 2]

    du = _grad(u, xy); dv = _grad(v, xy); dp = _grad(p, xy)
    u_x, u_y = du[:, 0], du[:, 1]
    v_x, v_y = dv[:, 0], dv[:, 1]
    p_x, p_y = dp[:, 0], dp[:, 1]

    d2u = _grad(u_x, xy); d2v = _grad(v_x, xy)
    u_xx, u_yy = d2u[:, 0], d2u[:, 1]
    v_xx, v_yy = d2v[:, 0], d2v[:, 1]

    nu = 1.0 / re
    f_cont = u_x + v_y
    f_x = u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
    f_y = u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)
    L_pde = (F.mse_loss(f_cont, torch.zeros_like(f_cont))
             + F.mse_loss(f_x, torch.zeros_like(f_x))
             + F.mse_loss(f_y, torch.zeros_like(f_y)))

    # Pressure-Poisson: Laplacian(p) + div(u·grad u) = 0
    T_x = u * u_x + v * u_y
    T_y = u * v_x + v * v_y
    T_x_x = _grad(T_x, xy)[:, 0]
    T_y_y = _grad(T_y, xy)[:, 1]
    div_T = T_x_x + T_y_y
    p_xx = _grad(p_x, xy)[:, 0]
    p_yy = _grad(p_y, xy)[:, 1]
    f_pp = (p_xx + p_yy) + div_T
    L_pp = F.mse_loss(f_pp, torch.zeros_like(f_pp))

    return L_pde, L_pp


def cavity_physics_loss_multi_re(model, colloc_by_re: dict, device):
    """Combined PDE + pressure-Poisson residuals across multiple Re (one fwd/Re).

    Returns (L_pde, L_pp) averaged over Re.
    """
    L_pde_tot = 0.0
    L_pp_tot = 0.0
    for re, xy_np in colloc_by_re.items():
        inp = torch.from_numpy(xy_np).float().to(device).clone().detach().requires_grad_(True)
        out = model(inp)
        u, v, p = out[:, 0], out[:, 1], out[:, 2]

        du = _grad(u, inp); dv = _grad(v, inp); dp = _grad(p, inp)
        u_x, u_y = du[:, 0], du[:, 1]
        v_x, v_y = dv[:, 0], dv[:, 1]
        p_x, p_y = dp[:, 0], dp[:, 1]

        d2u = _grad(u_x, inp); d2v = _grad(v_x, inp)
        u_xx, u_yy = d2u[:, 0], d2u[:, 1]
        v_xx, v_yy = d2v[:, 0], d2v[:, 1]

        nu = 1.0 / re
        f_cont = u_x + v_y
        f_x = u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
        f_y = u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)
        L_pde = (F.mse_loss(f_cont, torch.zeros_like(f_cont))
                 + F.mse_loss(f_x, torch.zeros_like(f_x))
                 + F.mse_loss(f_y, torch.zeros_like(f_y)))

        T_x = u * u_x + v * u_y
        T_y = u * v_x + v * v_y
        T_x_x = _grad(T_x, inp)[:, 0]
        T_y_y = _grad(T_y, inp)[:, 1]
        div_T = T_x_x + T_y_y
        p_xx = _grad(p_x, inp)[:, 0]
        p_yy = _grad(p_y, inp)[:, 1]
        f_pp = (p_xx + p_yy) + div_T
        L_pp = F.mse_loss(f_pp, torch.zeros_like(f_pp))

        L_pde_tot = L_pde_tot + L_pde
        L_pp_tot = L_pp_tot + L_pp
    return L_pde_tot / len(colloc_by_re), L_pp_tot / len(colloc_by_re)


# --------------------------------------------------------------------------
# Boundary conditions: Cavity (all no-slip walls + moving lid)
# --------------------------------------------------------------------------
def bc_loss_cavity_walls(model, n: int, device, re_norm=None):
    """No-slip on bottom, left, right walls: u = 0, v = 0.

    Args:
        re_norm: If not None, augment input with Re parameter (for ParametricPINN).
    """
    # Bottom wall: y_norm = -1
    x_bot = torch.linspace(-1.0, 1.0, n)
    y_bot = torch.full_like(x_bot, -1.0)
    xy_bot = torch.stack([x_bot, y_bot], dim=1)
    # Left wall: x_norm = -1
    y_left = torch.linspace(-1.0, 1.0, n)
    x_left = torch.full_like(y_left, -1.0)
    xy_left = torch.stack([x_left, y_left], dim=1)
    # Right wall: x_norm = +1
    y_right = torch.linspace(-1.0, 1.0, n)
    x_right = torch.full_like(y_right, 1.0)
    xy_right = torch.stack([x_right, y_right], dim=1)

    xy = torch.cat([xy_bot, xy_left, xy_right], dim=0)
    if re_norm is not None:
        re_col = torch.full((xy.shape[0], 1), re_norm)
        xy = torch.cat([xy, re_col], dim=1)
    xy = _on_device(xy, device)
    u, v, _ = _split(model(xy))
    return F.mse_loss(u, torch.zeros_like(u)) + F.mse_loss(v, torch.zeros_like(v))


def bc_loss_cavity_lid(model, n: int, u_lid: float, device, re_norm=None):
    """Moving lid at y_norm = +1: u = u_lid, v = 0.

    Args:
        re_norm: If not None, augment input with Re parameter (for ParametricPINN).
    """
    x_vals = torch.linspace(-1.0, 1.0, n)
    y_vals = torch.full_like(x_vals, 1.0)
    xy = torch.stack([x_vals, y_vals], dim=1)
    if re_norm is not None:
        re_col = torch.full((xy.shape[0], 1), re_norm)
        xy = torch.cat([xy, re_col], dim=1)
    xy = _on_device(xy, device)
    u, v, _ = _split(model(xy))
    return (F.mse_loss(u, torch.full_like(u, u_lid))
            + F.mse_loss(v, torch.zeros_like(v)))


def bc_loss_cavity(model, n: int, u_lid: float, device, re_norm=None):
    """Combined cavity BC: walls (no-slip) + lid (moving).

    Args:
        re_norm: If not None, augment input with Re parameter (for ParametricPINN).
    """
    return (bc_loss_cavity_walls(model, n, device, re_norm)
            + bc_loss_cavity_lid(model, n, u_lid, device, re_norm))


def bc_loss_cavity_multi_re(model, n: int, u_lid: float, device):
    """Cavity BC loss: lid velocity is independent of Re (u_lid = const)."""
    return bc_loss_cavity(model, n, u_lid, device)


# --------------------------------------------------------------------------
# Boundary conditions: Cylinder (inflow + outlet + walls + surface)
# --------------------------------------------------------------------------
def _on_device(t: torch.Tensor, device: torch.device):
    return t.to(device, dtype=torch.float32)


def bc_loss_inflow(model, n: int, u_inflow: float, device):
    """Inflow BC at x_norm = -1: u = u_inflow, v = 0."""
    y_vals = torch.linspace(-1.0, 1.0, n)
    x_vals = torch.full_like(y_vals, -1.0)
    xy = _on_device(torch.stack([x_vals, y_vals], dim=1), device)
    u, v, _ = _split(model(xy))
    return F.mse_loss(u, torch.full_like(u, u_inflow)) + F.mse_loss(v, torch.zeros_like(v))


def bc_loss_outlet(model, n: int, device):
    """Outlet BC at x_norm = +1: du/dx = 0, dv/dx = 0 (zero gradient)."""
    y_vals = torch.linspace(-1.0, 1.0, n)
    x_vals = torch.full_like(y_vals, 1.0)
    xy = _on_device(torch.stack([x_vals, y_vals], dim=1), device).requires_grad_(True)
    u, v, _ = _split(model(xy))
    du = _grad(u, xy)
    dv = _grad(v, xy)
    return F.mse_loss(du[:, 0], torch.zeros_like(du[:, 0])) + \
           F.mse_loss(dv[:, 0], torch.zeros_like(dv[:, 0]))


def bc_loss_walls(model, n: int, device):
    """Wall BC at y_norm = +/-1: u = 0, v = 0."""
    x_vals = torch.linspace(-1.0, 1.0, n)
    y_bot = torch.full_like(x_vals, -1.0)
    y_top = torch.full_like(x_vals, 1.0)
    xy_bot = torch.stack([x_vals, y_bot], dim=1)
    xy_top = torch.stack([x_vals, y_top], dim=1)
    xy = _on_device(torch.cat([xy_bot, xy_top], dim=0), device)
    u, v, _ = _split(model(xy))
    return F.mse_loss(u, torch.zeros_like(u)) + F.mse_loss(v, torch.zeros_like(v))


def bc_loss_cylinder(model, cx_norm: float, cy_norm: float, r_norm_x: float,
                     r_norm_y: float, n: int, device):
    """No-slip BC on cylinder surface: u = 0, v = 0."""
    theta = torch.linspace(0.0, 2.0 * 3.14159265, n)
    x = cx_norm + r_norm_x * torch.cos(theta)
    y = cy_norm + r_norm_y * torch.sin(theta)
    xy = _on_device(torch.stack([x, y], dim=1), device)
    u, v, _ = _split(model(xy))
    return F.mse_loss(u, torch.zeros_like(u)) + F.mse_loss(v, torch.zeros_like(v))


# --------------------------------------------------------------------------
# Data loss
# --------------------------------------------------------------------------
def data_loss(model, coords: torch.Tensor, u_target: torch.Tensor,
              v_target: torch.Tensor):
    """MSE on sparse sensor subsamples of the solver field (u, v only)."""
    u, v, _ = _split(model(coords))
    return F.mse_loss(u, u_target) + F.mse_loss(v, v_target)


def data_loss_full(model, coords: torch.Tensor, u_target: torch.Tensor,
                   v_target: torch.Tensor, p_target: torch.Tensor):
    """MSE on sparse sensor subsamples including pressure (u, v, p)."""
    u, v, p = _split(model(coords))
    # Weight u and v equally with p (pressure scale is comparable after L2 norm)
    return (F.mse_loss(u, u_target) + F.mse_loss(v, v_target)
            + 1.0 * F.mse_loss(p, p_target))


# --------------------------------------------------------------------------
# Total losses
# --------------------------------------------------------------------------
def total_loss(model, colloc_xy, coords, u_target, v_target,
               re, u_inflow, cx_norm, cy_norm, r_norm_x, r_norm_y,
               device, w_pde=1.0, w_data=1.0, w_bc=1.0):
    """Combined hybrid loss for cylinder case: PDE + data + BCs."""
    L_pde = pde_loss(model, colloc_xy, re, u_inflow)
    L_data = data_loss(model, coords, u_target, v_target)
    L_bc = (bc_loss_inflow(model, 200, u_inflow, device)
            + bc_loss_outlet(model, 200, device)
            + bc_loss_walls(model, 200, device)
            + bc_loss_cylinder(model, cx_norm, cy_norm, r_norm_x, r_norm_y, 400, device))
    return w_pde * L_pde + w_data * L_data + w_bc * L_bc, L_pde, L_data, L_bc


def total_loss_cavity(model, colloc_xy, coords, u_target, v_target, p_target,
                       re, u_lid, device, w_pde=1.0, w_data=1.0, w_bc=1.0,
                       w_pp=1.0, w_p=1.0, re_norm=None):
    """Combined hybrid loss for cavity case: PDE + Poisson + data + BCs.

    Returns (L_total, L_pde, L_pp, L_data, L_bc).
    """
    L_pde, L_pp = cavity_physics_loss(model, colloc_xy, re)
    u_pred, v_pred, p_pred = _split(model(coords))
    L_data = (F.mse_loss(u_pred, u_target) + F.mse_loss(v_pred, v_target)
              + w_p * F.mse_loss(p_pred, p_target))
    L_bc = bc_loss_cavity(model, 200, u_lid, device, re_norm=re_norm)
    L = w_pde * L_pde + w_pp * L_pp + w_data * L_data + w_bc * L_bc
    return L, L_pde, L_pp, L_data, L_bc


def total_loss_cavity_multi_re(model, colloc_by_re, sens_by_re, u_lid,
                                device, w_pde=1.0, w_data=1.0, w_bc=1.0,
                                w_pp=1.0, w_p=1.0):
    """Combined hybrid loss for parametric cavity: PDE + Poisson + data + BCs.

    Args:
        model: ParametricPINN.
        colloc_by_re: dict {re_val: (Nc, 3) numpy coords (x, y, re_norm)}.
        sens_by_re: dict {re_val: {'coords': (Ns,3), 'u': (Ns,), 'v': (Ns,), 'p': (Ns,)}}.
        u_lid: lid velocity (scalar, same for all Re).
        device: torch device.

    Returns (L_total, L_pde, L_pp, L_data, L_bc).
    """
    L_pde, L_pp = cavity_physics_loss_multi_re(model, colloc_by_re, device)

    L_data = 0.0
    for re, sens in sens_by_re.items():
        coords_t = torch.from_numpy(sens["coords"]).float().to(device)
        u_t = torch.from_numpy(sens["u"]).float().to(device)
        v_t = torch.from_numpy(sens["v"]).float().to(device)
        p_t = torch.from_numpy(sens["p"]).float().to(device)
        u_pred, v_pred, p_pred = _split(model(coords_t))
        L_data = (L_data
                  + F.mse_loss(u_pred, u_t) + F.mse_loss(v_pred, v_t)
                  + w_p * F.mse_loss(p_pred, p_t))
    L_data = L_data / len(sens_by_re)

    # BCs are Re-independent; use re_norm=0.0 (Re=100) as representative
    L_bc = bc_loss_cavity(model, 200, u_lid, device, re_norm=0.0)
    L = w_pde * L_pde + w_pp * L_pp + w_data * L_data + w_bc * L_bc
    return L, L_pde, L_pp, L_data, L_bc


def _split(out: torch.Tensor):
    """Split (N, 3) output into u, v, p."""
    return out[:, 0], out[:, 1], out[:, 2]
