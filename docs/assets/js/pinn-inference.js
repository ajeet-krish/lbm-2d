// ==========================================================================
// LBM-2D: PINN Surrogate
// Drives the "PINN Prediction" section. By default renders precomputed dense
// Re-sweep predictions (instant, robust, works on any static host). If
// onnxruntime-web is available it upgrades to true in-browser ONNX inference of
// the trained ParametricPINN, so the Reynolds slider runs the actual network.
//
// Velocity-magnitude only (the primary deliverable). Cases without a trained
// surrogate show a "training in progress" placeholder until the temporal PINN
// (Phase 6.8) is deployed.
// ==========================================================================

class PinnSurrogate {
    constructor(canvas, opts) {
        opts = opts || {};
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.dataDir = opts.dataDir || 'assets/data/cavity';
        this.modelDir = opts.modelDir || 'assets/js/vendor/onnxruntime-web/dist';
        this.nx = 96;
        this.ny = 96;
        this.sweep = null;
        this.reList = null;
        this.session = null;
        this.ortReady = false;
        this.ortLoading = false;
        this.reNormMin = 100;
        this.reNormMax = 1000;
        this.gridX = null;
        this.gridY = null;
        this.lastInferMs = null;
        this.onStatus = null;
        this._img = null;
    }

    _status(m) { if (this.onStatus) this.onStatus(m); }

    async loadSweep() {
        try {
            const buf = await window.FlowData.fetchBinary(`${this.dataDir}/pinn_sweep.bin`);
            this.sweep = window.FlowData.parseBinary(buf);
            this.reList = [];
            for (let r = 100; r <= 400; r += 15) this.reList.push(r);
            if (this.reList.length !== this.sweep.n_frames) {
                this.reList = [];
                for (let i = 0; i < this.sweep.n_frames; i++)
                    this.reList.push(100 + (400 - 100) * i / (this.sweep.n_frames - 1));
            }
            return true;
        } catch (e) {
            this.sweep = null;
            this.reList = null;
            return false;
        }
    }

    async loadGrid() {
        try {
            const buf = await window.FlowData.fetchBinary(`${this.dataDir}/pinn_grid.bin`);
            const g = window.FlowData.parseBinary(buf);
            const n = this.nx * this.ny;
            this.gridX = g.data.subarray(0, n);
            this.gridY = g.data.subarray(n, 2 * n);
            return true;
        } catch (e) {
            this.gridX = null;
            this.gridY = null;
            return false;
        }
    }

    async init() {
        const hasSweep = await this.loadSweep();
        await this.loadGrid();
        if (hasSweep) {
            this._tryLoadOrt();
        } else {
            this._showPlaceholder();
        }
    }

    _showPlaceholder() {
        if (!this.ctx) return;
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.fillStyle = 'rgba(13,17,23,0.85)';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.fillStyle = '#8b949e';
        this.ctx.font = '11px "JetBrains Mono", monospace';
        this.ctx.textAlign = 'center';
        this.ctx.fillText('PINN surrogate training in progress',
            this.canvas.width / 2, this.canvas.height / 2 - 6);
        this.ctx.fillStyle = '#586069';
        this.ctx.fillText('LBM baseline shown at left',
            this.canvas.width / 2, this.canvas.height / 2 + 10);
        this.ctx.textAlign = 'left';
    }

    async _tryLoadOrt() {
        if (this.ortLoading || this.ortReady) return;
        this.ortLoading = true;
        try {
            if (typeof ort === 'undefined') {
                await this._injectScript(`${this.modelDir}/ort.min.js`);
            }
            if (typeof ort !== 'undefined') {
                ort.env.wasm.wasmPaths = this.modelDir + '/';
                ort.env.wasm.numThreads = 1; // single-thread: no COOP/COEP headers needed
                this._status('Loading PINN ONNX model...');
                this.session = await ort.InferenceSession.create(`${this.dataDir}/pinn_model.onnx`);
                this.ortReady = true;
                this._status('Live ONNX inference ready');
            }
        } catch (e) {
            this.ortReady = false;
            this._status('(using precomputed surrogate predictions)');
        }
        this.ortLoading = false;
    }

    _injectScript(src) {
        return new Promise((res, rej) => {
            const s = document.createElement('script');
            s.src = src;
            s.onload = res;
            s.onerror = rej;
            document.head.appendChild(s);
        });
    }

    _sweepField(re, ch) {
        const rel = (re - 100) / (400 - 100);
        const nf = this.sweep.n_frames;
        const t = Math.max(0, Math.min(1, rel)) * (nf - 1);
        const i0 = Math.floor(t), i1 = Math.min(nf - 1, i0 + 1), f = t - i0;
        const n = this.nx * this.ny;
        const off0 = ((i0 * this.sweep.n_chan) + ch) * n;
        const off1 = ((i1 * this.sweep.n_chan) + ch) * n;
        const d = this.sweep.data;
        const out = new Float32Array(n);
        for (let k = 0; k < n; k++) out[k] = d[off0 + k] * (1 - f) + d[off1 + k] * f;
        return out;
    }

    async predict(re) {
        if (this.ortReady && this.session) return await this._predictLive(re);
        return {
            u: this._sweepField(re, 0),
            v: this._sweepField(re, 1),
        };
    }

    async _predictLive(re) {
        const t0 = performance.now();
        const reN = (re - this.reNormMin) / (this.reNormMax - this.reNormMin);
        const n = this.nx * this.ny;
        const inp = new Float32Array(n * 3);
        for (let k = 0; k < n; k++) {
            inp[k * 3] = this.gridX[k];
            inp[k * 3 + 1] = this.gridY[k];
            inp[k * 3 + 2] = reN;
        }
        const tensor = new ort.Tensor('float32', inp, [n, 3]);
        const out = await this.session.run({ input: tensor });
        const res = out.output.data;
        const u = new Float32Array(n), v = new Float32Array(n);
        for (let k = 0; k < n; k++) { u[k] = res[k * 3]; v[k] = res[k * 3 + 1]; }
        this.lastInferMs = performance.now() - t0;
        return { u, v };
    }

    render(re) {
        if (!this.sweep) { this._showPlaceholder(); return Promise.resolve(null); }
        return this.predict(re).then(({ u, v }) => {
            const n = this.nx * this.ny;
            const arr = new Float32Array(n);
            let vmax = 1e-9;
            for (let k = 0; k < n; k++) { const m = Math.hypot(u[k], v[k]); arr[k] = m; if (m > vmax) vmax = m; }
            if (!this._img || this._img.width !== this.nx) this._img = this.ctx.createImageData(this.nx, this.ny);
            Colormaps.paintField(this._img, arr, this.nx, this.ny, 0, vmax, 'viridis', true);
            this.ctx.putImageData(this._img, 0, 0);

            if (window.FlowStreamlines) {
                const lines = window.FlowStreamlines.buildStreamlines(u, v, this.nx, this.ny, null, 13);
                this.ctx.lineWidth = 1.1;
                this.ctx.lineJoin = 'round';
                for (const line of lines) {
                    for (let i = 1; i < line.length; i++) {
                        const b = line[i];
                        const sp = Math.hypot(
                            window.FlowStreamlines.sample(u, this.nx, this.ny, b.x, b.y),
                            window.FlowStreamlines.sample(v, this.nx, this.ny, b.x, b.y));
                        const t = Math.min(1, sp / (vmax || 1));
                        const [r, g, bl] = Colormaps.sample('viridis', t);
                        this.ctx.strokeStyle = `rgba(${r},${g},${bl},0.55)`;
                        // Flip data-y to canvas-y to match the flipped contour.
                        this.ctx.beginPath();
                        this.ctx.moveTo(line[i - 1].x, this.ny - 1 - line[i - 1].y);
                        this.ctx.lineTo(b.x, this.ny - 1 - b.y);
                        this.ctx.stroke();
                    }
                }
            }
            return { inferMs: this.lastInferMs };
        });
    }
}

window.PinnSurrogate = PinnSurrogate;
