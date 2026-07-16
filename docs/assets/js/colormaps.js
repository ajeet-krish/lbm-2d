// ==========================================================================
// LBM-2D: Color maps for canvas flow visualization
// Builds 256-entry LUTs for viridis, turbo, RdBu (diverging), and jet.
// Pure JS, no dependencies.
// ==========================================================================

const Colormaps = (function () {
    'use strict';

    // Viridis control points (RGB 0-255), interpolated to a 256 LUT.
    const VIRIDIS_STOPS = [
        [0.00, [68, 1, 84]],
        [0.10, [72, 35, 116]],
        [0.20, [64, 67, 135]],
        [0.30, [52, 94, 141]],
        [0.40, [41, 121, 142]],
        [0.50, [32, 146, 140]],
        [0.60, [34, 168, 132]],
        [0.70, [68, 191, 112]],
        [0.80, [121, 209, 81]],
        [0.90, [189, 223, 38]],
        [1.00, [253, 231, 37]],
    ];

    function buildLUT(stops, n) {
        const lut = new Uint8Array(n * 3);
        for (let i = 0; i < n; i++) {
            const t = i / (n - 1);
            let k = 0;
            while (k < stops.length - 2 && t > stops[k + 1][0]) k++;
            const [t0, c0] = stops[k];
            const [t1, c1] = stops[k + 1];
            const f = (t - t0) / (t1 - t0 || 1);
            lut[i * 3 + 0] = Math.round(c0[0] + (c1[0] - c0[0]) * f);
            lut[i * 3 + 1] = Math.round(c0[1] + (c1[1] - c0[1]) * f);
            lut[i * 3 + 2] = Math.round(c0[2] + (c1[2] - c0[2]) * f);
        }
        return lut;
    }

    // Analytic jet colormap (classic blue->cyan->green->yellow->red).
    function jetRGB(t) {
        t = Math.max(0, Math.min(1, t));
        const r = Math.max(0, Math.min(1, 1.5 - Math.abs(4 * t - 3)));
        const g = Math.max(0, Math.min(1, 1.5 - Math.abs(4 * t - 2)));
        const b = Math.max(0, Math.min(1, 1.5 - Math.abs(4 * t - 1)));
        return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
    }

    // Analytic RdBu diverging (blue -> white -> red).
    function rdbuRGB(t) {
        t = Math.max(0, Math.min(1, t));
        const blue = [5, 113, 176];
        const white = [247, 247, 247];
        const red = [213, 62, 79];
        let c;
        if (t < 0.5) {
            const f = t / 0.5;
            c = [blue, white];
            return [
                Math.round(blue[0] + (white[0] - blue[0]) * f),
                Math.round(blue[1] + (white[1] - blue[1]) * f),
                Math.round(blue[2] + (white[2] - blue[2]) * f),
            ];
        } else {
            const f = (t - 0.5) / 0.5;
            return [
                Math.round(white[0] + (red[0] - white[0]) * f),
                Math.round(white[1] + (red[1] - white[1]) * f),
                Math.round(white[2] + (red[2] - white[2]) * f),
            ];
        }
    }

    function buildUniform(fn, n) {
        const lut = new Uint8Array(n * 3);
        for (let i = 0; i < n; i++) {
            const [r, g, b] = fn(i / (n - 1));
            lut[i * 3 + 0] = r;
            lut[i * 3 + 1] = g;
            lut[i * 3 + 2] = b;
        }
        return lut;
    }

    // Turbo colormap (Google, 2019) -- perceptually-improved rainbow.
    // Control points sampled from the published 256-entry turbo LUT.
    const TURBO_STOPS = [
        [0.00, [48, 18, 59]],
        [0.05, [70, 33, 114]],
        [0.10, [78, 42, 154]],
        [0.15, [69, 53, 188]],
        [0.20, [54, 66, 208]],
        [0.25, [41, 81, 217]],
        [0.30, [31, 97, 219]],
        [0.35, [25, 113, 217]],
        [0.40, [23, 130, 211]],
        [0.45, [27, 146, 201]],
        [0.50, [36, 162, 187]],
        [0.55, [52, 177, 170]],
        [0.60, [74, 191, 149]],
        [0.65, [102, 203, 124]],
        [0.70, [134, 213, 96]],
        [0.75, [167, 220, 67]],
        [0.80, [200, 223, 42]],
        [0.85, [230, 220, 30]],
        [0.90, [251, 200, 39]],
        [0.95, [253, 170, 43]],
        [1.00, [231, 105, 28]],
    ];

    const N = 256;
    const LUTS = {
        viridis: buildLUT(VIRIDIS_STOPS, N),
        turbo: buildLUT(TURBO_STOPS, N),
        jet: buildUniform(jetRGB, N),
        rdbu: buildUniform(rdbuRGB, N),
    };

    // Sample a LUT at normalized value t in [0, 1].
    function sample(name, t) {
        const lut = LUTS[name] || LUTS.viridis;
        t = Math.max(0, Math.min(1, t));
        const idx = Math.min(N - 1, Math.floor(t * (N - 1))) * 3;
        return [lut[idx], lut[idx + 1], lut[idx + 2]];
    }

    // Fill a provided ImageData with a field using the given colormap.
    // field: Float32Array of length nx*ny (row-major, y up). vmin/vmax define range.
    // flipY: if true, row 0 is at the bottom (matplotlib origin='lower').
    function paintField(imgData, field, nx, ny, vmin, vmax, cmap, flipY) {
        const data = imgData.data;
        const range = (vmax - vmin) || 1e-9;
        const lut = LUTS[cmap] || LUTS.viridis;
        for (let j = 0; j < ny; j++) {
            const srcRow = flipY ? (ny - 1 - j) : j;
            for (let i = 0; i < nx; i++) {
                const val = field[srcRow * nx + i];
                let t = (val - vmin) / range;
                t = t < 0 ? 0 : (t > 1 ? 1 : t);
                const li = Math.min(N - 1, Math.floor(t * (N - 1))) * 3;
                const di = (j * nx + i) * 4;
                data[di] = lut[li];
                data[di + 1] = lut[li + 1];
                data[di + 2] = lut[li + 2];
                data[di + 3] = 255;
            }
        }
    }

    return { LUTS, sample, paintField };
})();

// Expose globally for non-module scripts.
if (typeof window !== 'undefined') {
    window.Colormaps = Colormaps;
}
