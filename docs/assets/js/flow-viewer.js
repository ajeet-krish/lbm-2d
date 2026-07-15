// ==========================================================================
// LBM-2D: Interactive Flow Viewer (canvas-based LBM time-series engine)
// Streams compact float16 binary frame data exported by pinn/export_web_data.py
// and renders velocity-magnitude contours with overlaid streamlines directly
// on a <canvas>. No external dependencies (uses Colormaps global from
// colormaps.js). This viewer renders the C++ LBM solver's evolution from rest
// to steady state; the PINN surrogate is handled by pinn-inference.js.
// ==========================================================================

(function () {
    'use strict';

    const MAGIC = 0x4C424D31;

    // --- float16 -> float32 -------------------------------------------------
    function float16ToFloat32(h) {
        const s = (h & 0x8000) >> 15;
        const e = (h & 0x7C00) >> 10;
        const f = h & 0x03FF;
        if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
        if (e === 0x1F) return f ? NaN : (s ? -Infinity : Infinity);
        return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
    }

    // --- gzip decompress via browser DecompressionStream -------------------
    async function gunzip(ab) {
        if (typeof DecompressionStream === 'undefined') return ab;
        try {
            const ds = new DecompressionStream('gzip');
            const stream = new Blob([ab]).stream().pipeThrough(ds);
            return await new Response(stream).arrayBuffer();
        } catch (e) {
            return ab;
        }
    }

    async function fetchBinary(url) {
        // Prefer gzipped copy; fall back to raw.
        let buf = null;
        try {
            const r = await fetch(url + '.gz');
            if (r.ok) buf = await gunzip(await r.arrayBuffer());
        } catch (e) { /* ignore */ }
        if (!buf) {
            const r2 = await fetch(url);
            buf = await r2.arrayBuffer();
        }
        return buf;
    }

    function parseBinary(buf) {
        const dv = new DataView(buf);
        const magic = dv.getUint32(0, true);
        if (magic !== MAGIC) throw new Error('Bad magic in binary flow data');
        const n_frames = dv.getUint32(4, true);
        const nx = dv.getUint32(8, true);
        const ny = dv.getUint32(12, true);
        const n_chan = dv.getUint32(16, true);
        const dtype = dv.getUint32(20, true);
        const off = 24;
        let data;
        if (dtype === 1) {
            const u16 = new Uint16Array(buf, off);
            data = new Float32Array(u16.length);
            for (let i = 0; i < u16.length; i++) data[i] = float16ToFloat32(u16[i]);
        } else {
            data = new Float32Array(buf, off);
        }
        return { n_frames, nx, ny, n_chan, data };
    }

    // --- bilinear sample ----------------------------------------------------
    function sample(arr, nx, ny, x, y) {
        if (x < 0) x = 0; else if (x > nx - 1) x = nx - 1;
        if (y < 0) y = 0; else if (y > ny - 1) y = ny - 1;
        const x0 = Math.floor(x), y0 = Math.floor(y);
        const x1 = Math.min(nx - 1, x0 + 1), y1 = Math.min(ny - 1, y0 + 1);
        const fx = x - x0, fy = y - y0;
        const v00 = arr[y0 * nx + x0], v10 = arr[y0 * nx + x1];
        const v01 = arr[y1 * nx + x0], v11 = arr[y1 * nx + x1];
        return (v00 * (1 - fx) + v10 * fx) * (1 - fy) + (v01 * (1 - fx) + v11 * fx) * fy;
    }

    // --- streamline tracing (RK4, constant arc-length) ----------------------
    function traceStreamline(u, v, nx, ny, obs, x0, y0, dir, maxSteps, stepLen) {
        const pts = [];
        let x = x0, y = y0;
        for (let i = 0; i < maxSteps; i++) {
            if (x < 0 || x > nx - 1 || y < 0 || y > ny - 1) break;
            const ix = Math.floor(y) * nx + Math.floor(x);
            if (obs && obs[ix] > 0.5) break;
            const ux = sample(u, nx, ny, x, y), vy = sample(v, nx, ny, x, y);
            const sp = Math.hypot(ux, vy);
            if (sp < 1e-5) break;
            pts.push({ x, y });
            const inv = dir * stepLen / sp;
            const k1x = ux, k1y = vy;
            const ax = x + inv * 0.5 * k1x, ay = y + inv * 0.5 * k1y;
            const k2x = sample(u, nx, ny, ax, ay), k2y = sample(v, nx, ny, ax, ay);
            const bx = x + inv * 0.5 * k2x, by = y + inv * 0.5 * k2y;
            const k3x = sample(u, nx, ny, bx, by), k3y = sample(v, nx, ny, bx, by);
            const cx = x + inv * k3x, cy = y + inv * k3y;
            const k4x = sample(u, nx, ny, cx, cy), k4y = sample(v, nx, ny, cx, cy);
            x += inv * (k1x + 2 * k2x + 2 * k3x + k4x) / 6;
            y += inv * (k1y + 2 * k2y + 2 * k3y + k4y) / 6;
        }
        return pts;
    }

    function buildStreamlines(u, v, nx, ny, obs, nSeeds) {
        const lines = [];
        const step = Math.max(4, Math.floor(nx / nSeeds));
        for (let j = step / 2; j < ny; j += step) {
            for (let i = step / 2; i < nx; i += step) {
                if (obs && obs[Math.floor(j) * nx + Math.floor(i)] > 0.5) continue;
                const back = traceStreamline(u, v, nx, ny, obs, i, j, -1, 90, 1.0);
                const fwd = traceStreamline(u, v, nx, ny, obs, i, j, 1, 90, 1.0);
                const line = back.reverse().concat(fwd);
                if (line.length > 2) lines.push(line);
            }
        }
        return lines;
    }

    // ======================================================================
    // FlowViewer -- renders the C++ LBM time-series as an animated canvas.
    // ======================================================================
    class FlowViewer {
        constructor(canvas, dataDir, opts) {
            opts = opts || {};
            this.canvas = canvas;
            this.ctx = canvas.getContext('2d');
            this.dataDir = dataDir || 'assets/data/cavity';
            // filePrefix selects which binary set to load: 'lbm' (solver
            // evolution) or 'pinn_temporal' (trained time-parametric surrogate).
            this.filePrefix = opts.filePrefix || 'lbm';
            // cmap: per-case color scheme. Every plot AND animation for a
            // given case must share this colormap for visual consistency.
            this.cmap = opts.cmap || 'viridis';
            this.cache = {};            // re -> { lbm, vmax }
            this._missing = {};         // re -> true if file failed to load
            this.re = 100;
            this.currentFrame = 0;
            this.playing = false;
            this.showStreamlines = true;
            this.frameInterval = 90;    // ms between animation frames
            this.onFrameChange = null;
            this.onStatus = null;
            this._last = 0;
            this._imgData = null;
            this._scratch = null;
            this._loopBound = this._loop.bind(this);
            requestAnimationFrame(this._loopBound);
        }

        // ---- data loading ----
        async load(re) {
            if (this.cache[re]) return this.cache[re];
            if (this._missing[re]) return null;
            this._status('Loading Re=' + re + ' data...');
            try {
                const buf = await fetchBinary(`${this.dataDir}/${this.filePrefix}_re${re}.bin`);
                const lbm = parseBinary(buf);
                const entry = { lbm, vmax: this._computeVmax(lbm) };
                this.cache[re] = entry;
                this._missing[re] = false;
                this._status('');
                return entry;
            } catch (e) {
                this._missing[re] = true;
                this._status('');
                this._showMissing();
                return null;
            }
        }

        _showMissing() {
            if (!this.ctx) return;
            const w = this.canvas.width, h = this.canvas.height;
            this.ctx.clearRect(0, 0, w, h);
            this.ctx.fillStyle = 'rgba(13,17,23,0.85)';
            this.ctx.fillRect(0, 0, w, h);
            this.ctx.fillStyle = '#8b949e';
            this.ctx.font = '11px "JetBrains Mono", monospace';
            this.ctx.textAlign = 'center';
            this.ctx.fillText('PINN surrogate training in progress', w / 2, h / 2 - 6);
            this.ctx.fillStyle = '#586069';
            this.ctx.fillText('LBM baseline shown at left', w / 2, h / 2 + 10);
            this.ctx.textAlign = 'left';
        }

        _computeVmax(lbm) {
            const { nx, ny, n_chan, data, n_frames } = lbm;
            const n = nx * ny;
            let vmax = 1e-9;
            for (let f = 0; f < n_frames; f++) {
                const uOff = (f * n_chan) * n;
                const vOff = (f * n_chan + 1) * n;
                for (let k = 0; k < n; k++) {
                    const vm = Math.hypot(data[uOff + k], data[vOff + k]);
                    if (vm > vmax) vmax = vm;
                }
            }
            return vmax;
        }

        // ---- channel accessors ----
        _chan(src, frame, ch) {
            const { nx, ny, n_chan, data } = src;
            const off = ((frame * n_chan) + ch) * nx * ny;
            return data.subarray(off, off + nx * ny);
        }

        _ensureScratch(n) {
            if (!this._scratch || this._scratch.length !== n) this._scratch = new Float32Array(n);
            return this._scratch;
        }

        // ---- rendering (velocity magnitude only) ----
        _render() {
            const entry = this.cache[this.re];
            if (!entry) return;
            const src = entry.lbm;
            const { nx, ny } = src;
            const frame = this.currentFrame;
            const scr = this._ensureScratch(nx * ny);
            const u = this._chan(src, frame, 0), v = this._chan(src, frame, 1);
            for (let k = 0; k < nx * ny; k++) scr[k] = Math.hypot(u[k], v[k]);
            this._paint(this.ctx, scr, nx, ny, 0, entry.vmax, this.cmap);
            if (this.showStreamlines) {
                const obs = src.n_chan >= 5 ? this._chan(src, frame, 4) : null;
                this._drawStreamlines(this.ctx, u, v, nx, ny, obs, entry.vmax);
            }
        }

        _paint(ctx, arr, nx, ny, min, max, cmap) {
            if (!this._imgData || this._imgData.width !== nx || this._imgData.height !== ny) {
                this._imgData = ctx.createImageData(nx, ny);
            }
            Colormaps.paintField(this._imgData, arr, nx, ny, min, max, cmap, true);
            ctx.putImageData(this._imgData, 0, 0);
        }

        _drawStreamlines(ctx, u, v, nx, ny, obs, vmax) {
            if (!this.showStreamlines) return;
            const lines = buildStreamlines(u, v, nx, ny, obs, 13);
            ctx.lineJoin = 'round';
            ctx.lineCap = 'round';
            ctx.lineWidth = 1.3;
            for (const line of lines) {
                for (let i = 1; i < line.length; i++) {
                    const a = line[i - 1], b = line[i];
                    const sp = Math.hypot(
                        sample(u, nx, ny, b.x, b.y), sample(v, nx, ny, b.x, b.y));
                    const t = Math.min(1, sp / (vmax || 1));
                    const [r, g, bl] = Colormaps.sample(this.cmap, t);
                    ctx.strokeStyle = `rgba(${r},${g},${bl},0.92)`;
                    // Transform data-y (y=0 = bottom) to canvas-y (y=0 = top) to
                    // match the flipped contour rendered by paintField(flipY=true).
                    ctx.beginPath();
                    ctx.moveTo(a.x, ny - 1 - a.y);
                    ctx.lineTo(b.x, ny - 1 - b.y);
                    ctx.stroke();
                }
            }
        }

        // ---- animation loop ----
        _loop(ts) {
            // Always reschedule first so the loop survives the window between
            // construction and the first cache load (otherwise Play does nothing).
            requestAnimationFrame(this._loopBound);
            if (!this.cache[this.re]) return;
            if (!this.playing) { this._render(); return; }
            const dt = ts - this._last;
            if (dt < this.frameInterval) return;
            this._last = ts;
            this.currentFrame = (this.currentFrame + 1) % this.cache[this.re].lbm.n_frames;
            if (this.onFrameChange) this.onFrameChange(this.currentFrame);
            this._render();
        }

        // ---- public controls ----
        async setRe(re) {
            this.re = re;
            this.currentFrame = 0;
            const entry = await this.load(re);
            if (!entry) {
                if (this.onFrameChange) this.onFrameChange(0);
                return;
            }
            this._resetCanvasSize();
            this._render();
            if (this.onFrameChange) this.onFrameChange(this.currentFrame);
        }

        setFrame(i) {
            if (!this.cache[this.re]) return;
            this.currentFrame = Math.max(0, Math.min(this.cache[this.re].lbm.n_frames - 1, i));
            this._render();
        }
        play() { this.playing = true; this._last = 0; }
        pause() { this.playing = false; }
        togglePlay() { this.playing = !this.playing; if (this.playing) this._last = 0; return this.playing; }

        _resetCanvasSize() {
            const src = this.cache[this.re];
            if (!src) return;
            this.canvas.width = src.lbm.nx;
            this.canvas.height = src.lbm.ny;
        }

        async init(re) {
            await this.setRe(re);
        }

        _status(msg) { if (this.onStatus) this.onStatus(msg); }
    }

    window.FlowViewer = FlowViewer;
    // Helpers reused by the Live PINN module.
    window.FlowStreamlines = { buildStreamlines, sample, traceStreamline };
    window.FlowData = { parseBinary, fetchBinary, float16ToFloat32 };
})();
