// ==========================================================================
// LBM-2D: PINN Surrogate (live temporal ONNX inference)
// Drives the "PINN Prediction" section of each case page. Loads the trained
// time-parametric ParametricPINN (exported to ONNX) and runs real in-browser
// inference at ANY Reynolds number in the trained range, animating the full
// transient evolution frame-by-frame.
//
// Model signature:
//   input  : [N, 4]  (x, y, Re_n, t_n)   -- normalized coords, Re, time
//   output : [N, 3]  (u, v, p)
// where Re_n in [0,1] over [RE_MIN, RE_MAX] and t_n in [0,1] over the frame set.
//
// Velocity-magnitude (with speed-colored streamlines) is rendered using the
// case colormap passed via opts.cmap (turbo for cavity). The static 3-panel
// error-delta plot keeps its own diverging colormap (RdBu) and is unaffected.
// ==========================================================================

class PinnSurrogate {
    constructor(canvas, opts) {
        opts = opts || {};
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.dataDir = opts.dataDir || 'assets/data/cavity';
        this.modelDir = opts.modelDir || 'assets/js/vendor/onnxruntime-web/dist';
        this.cmap = opts.cmap || 'turbo';
        this.nx = 96;
        this.ny = 96;
        this.nFrames = 51;
        this.reMin = 100;
        this.reMax = 1000;
        this.gridX = null;
        this.gridY = null;
        this.session = null;
        this.ortReady = false;
        this.ortLoading = false;
        this.cache = {};          // key "re:frame" -> { u, v, vmax }
        this.re = 100;
        this.frame = 0;
        this.playing = false;
        this._last = 0;
        this._img = null;
        this.onFrameChange = null;
        this.onStatus = null;
        this._inferToken = 0;    // guards against out-of-order async inference
    }

    _status(m) { if (this.onStatus) this.onStatus(m); }

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
        const ok = await this.loadGrid();
        if (!ok) { this._showError('grid data missing'); return; }
        this._tryLoadOrt();
    }

    _showError(msg) {
        if (!this.ctx) return;
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.fillStyle = 'rgba(13,17,23,0.85)';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.fillStyle = '#8b949e';
        this.ctx.font = '11px "JetBrains Mono", monospace';
        this.ctx.textAlign = 'center';
        this.ctx.fillText('PINN surrogate unavailable', this.canvas.width / 2, this.canvas.height / 2);
        this.ctx.fillStyle = '#586069';
        this.ctx.fillText(msg, this.canvas.width / 2, this.canvas.height / 2 + 16);
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
                this.session = await ort.InferenceSession.create(
                    `${this.dataDir}/pinn_temporal_model.onnx`);
                this.ortReady = true;
                this._status('Live ONNX inference ready');
                this._renderCurrent();
            } else {
                this._showError('ONNX runtime failed to load');
            }
        } catch (e) {
            this.ortReady = false;
            this._showError('model load failed: ' + (e && e.message ? e.message : e));
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

    _key(re, frame) { return re + ':' + frame; }

    async _infer(re, frame) {
        const key = this._key(re, frame);
        if (this.cache[key]) return this.cache[key];
        if (!this.ortReady || !this.session) return null;
        const token = ++this._inferToken;
        const t0 = performance.now();
        const reN = (re - this.reMin) / (this.reMax - this.reMin);
        const tN = this.nFrames > 1 ? frame / (this.nFrames - 1) : 0.0;
        const n = this.nx * this.ny;
        const inp = new Float32Array(n * 4);
        for (let k = 0; k < n; k++) {
            inp[k * 4] = this.gridX[k];
            inp[k * 4 + 1] = this.gridY[k];
            inp[k * 4 + 2] = reN;
            inp[k * 4 + 3] = tN;
        }
        const tensor = new ort.Tensor('float32', inp, [n, 4]);
        const out = await this.session.run({ input: tensor });
        if (token !== this._inferToken) return null; // superseded
        const res = out.output.data;
        const u = new Float32Array(n), v = new Float32Array(n);
        let vmax = 1e-9;
        for (let k = 0; k < n; k++) {
            const uu = res[k * 3], vv = res[k * 3 + 1];
            u[k] = uu; v[k] = vv;
            const m = Math.hypot(uu, vv);
            if (m > vmax) vmax = m;
        }
        const result = { u, v, vmax, inferMs: performance.now() - t0 };
        this.cache[key] = result;
        return result;
    }

    async _renderCurrent() {
        const data = await this._infer(this.re, this.frame);
        if (!data) return;
        const { u, v, vmax } = data;
        const n = this.nx * this.ny;
        const arr = new Float32Array(n);
        for (let k = 0; k < n; k++) arr[k] = Math.hypot(u[k], v[k]);
        if (!this._img || this._img.width !== this.nx) {
            this._img = this.ctx.createImageData(this.nx, this.ny);
        }
        Colormaps.paintField(this._img, arr, this.nx, this.ny, 0, vmax, this.cmap, true);
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
                    const [r, g, bl] = Colormaps.sample(this.cmap, t);
                    this.ctx.strokeStyle = `rgba(${r},${g},${bl},0.6)`;
                    // Flip data-y to canvas-y (data y=0 is the bottom wall).
                    this.ctx.beginPath();
                    this.ctx.moveTo(line[i - 1].x, this.ny - 1 - line[i - 1].y);
                    this.ctx.lineTo(b.x, this.ny - 1 - b.y);
                    this.ctx.stroke();
                }
            }
        }
        if (data.inferMs != null && this.onStatus) {
            this.onStatus('Live ONNX inference ready (' + data.inferMs.toFixed(1) + ' ms/frame)');
        }
    }

    setRe(re) {
        this.re = re;
        return this._renderCurrent();
    }

    setFrame(f) {
        this.frame = Math.max(0, Math.min(this.nFrames - 1, f | 0));
        if (this.onFrameChange) this.onFrameChange(this.frame);
        return this._renderCurrent();
    }

    play() { this.playing = true; this._last = 0; this._loop(); }
    pause() { this.playing = false; }
    togglePlay() { this.playing = !this.playing; if (this.playing) { this._last = 0; this._loop(); } return this.playing; }

    _loop() {
        if (!this.playing) return;
        const now = performance.now();
        if (now - this._last >= 60) {
            this._last = now;
            this.frame = (this.frame + 1) % this.nFrames;
            if (this.onFrameChange) this.onFrameChange(this.frame);
            this._renderCurrent();
        }
        requestAnimationFrame(() => this._loop());
    }
}

window.PinnSurrogate = PinnSurrogate;
