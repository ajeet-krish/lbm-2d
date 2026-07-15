// ==========================================================================
// LBM-2D: Shared Flow Viewer Init
// Wires the LBM Evolution + PINN Prediction sections on each case page to the
// FlowViewer (LBM time-series) and PinnSurrogate (parametric surrogate) classes.
// Each page calls initFlowViewerSections() with its own case configuration.
// ==========================================================================

function initFlowViewerSections(opts) {
    opts = opts || {};
    var dataDir = opts.dataDir || 'assets/data/cavity';
    var modelDir = opts.modelDir || 'assets/js/vendor/onnxruntime-web/dist';
    var lbmConfigs = opts.lbmConfigs || [{ label: '100', file: '100' }];

    var lbmCanvas = document.getElementById('lbmCanvas');
    var pinnCanvas = document.getElementById('pinnCanvas');
    if (!lbmCanvas) return;

    var viewer = new FlowViewer(lbmCanvas, dataDir);
    var surrogate = pinnCanvas
        ? new PinnSurrogate(pinnCanvas, { dataDir: dataDir, modelDir: modelDir })
        : null;

    var statusEl = document.getElementById('fvStatus');
    viewer.onStatus = function (m) { if (statusEl) statusEl.textContent = m; };
    if (surrogate) surrogate.onStatus = function (m) { if (statusEl) statusEl.textContent = m; };

    viewer.onFrameChange = function (f) {
        var lbl = document.getElementById('lbmFrameLabel');
        if (lbl && viewer.cache[viewer.re]) {
            lbl.textContent = 'Frame ' + f + ' / ' + (viewer.cache[viewer.re].lbm.n_frames - 1);
            var scr = document.getElementById('lbmScrubber');
            if (scr) { scr.max = viewer.cache[viewer.re].lbm.n_frames - 1; scr.value = f; }
        }
    };

    // Build LBM parameter (Re / config) buttons.
    var reGroup = document.getElementById('lbmReGroup');
    if (reGroup) {
        var grp = reGroup.querySelector('.fv-group');
        if (grp) grp.innerHTML = '<span class="fv-label-txt">' + (opts.paramLabel || 'Re') + '</span>';
        lbmConfigs.forEach(function (c, i) {
            var btn = document.createElement('button');
            btn.className = 'fv-btn fv-re' + (i === 0 ? ' active' : '');
            btn.setAttribute('data-file', c.file);
            btn.textContent = c.label;
            btn.addEventListener('click', function () {
                reGroup.querySelectorAll('.fv-re').forEach(function (b) { b.classList.remove('active'); });
                btn.classList.add('active');
                viewer.setRe(c.file);
            });
            if (grp) grp.appendChild(btn);
        });
    }

    // LBM play / pause
    var playBtn = document.getElementById('lbmPlay');
    if (playBtn) {
        playBtn.addEventListener('click', function () {
            var playing = viewer.togglePlay();
            this.innerHTML = playing ? '&#10074;&#10074; Pause' : '&#9654; Play';
        });
    }

    // LBM scrubber
    var scrubber = document.getElementById('lbmScrubber');
    if (scrubber) {
        scrubber.addEventListener('input', function () {
            viewer.pause();
            if (playBtn) playBtn.innerHTML = '&#9654; Play';
            viewer.setFrame(parseInt(this.value, 10));
        });
    }

    // PINN slider (only if a trained surrogate is available for this case).
    var pinnControls = document.getElementById('pinnControls');
    if (surrogate && opts.pinn && pinnControls) {
        var p = opts.pinn;
        var labelTxt = document.createElement('span');
        labelTxt.className = 'fv-label-txt';
        labelTxt.innerHTML = p.label + ' = <strong id="pinnRe">' + p.default + '</strong>';
        var slider = document.createElement('input');
        slider.type = 'range';
        slider.id = 'pinnSlider';
        slider.className = 'fv-slider';
        slider.min = p.min; slider.max = p.max; slider.step = p.step || 1; slider.value = p.default;
        var time = document.createElement('span');
        time.className = 'fv-frame';
        time.id = 'pinnInferTime';
        pinnControls.appendChild(labelTxt);
        pinnControls.appendChild(slider);
        pinnControls.appendChild(time);

        slider.addEventListener('input', function () {
            var val = parseInt(this.value, 10);
            var rl = document.getElementById('pinnRe');
            if (rl) rl.textContent = val;
            surrogate.render(val).then(function (r) {
                if (r && r.inferMs != null) {
                    time.textContent = 'ONNX inference: ' + r.inferMs.toFixed(1) + ' ms';
                } else {
                    time.textContent = 'surrogate prediction';
                }
            });
        });
    } else if (pinnControls && opts.pinnNote) {
        var note = document.createElement('span');
        note.className = 'fv-label-txt';
        note.textContent = opts.pinnNote;
        pinnControls.appendChild(note);
    }

    // Boot
    viewer.init(lbmConfigs[0].file).then(function () {
        viewer.play();
        if (playBtn) playBtn.innerHTML = '&#10074;&#10074; Pause';
        // Preload remaining LBM configs in the background.
        lbmConfigs.slice(1).forEach(function (c) { viewer.load(c.file); });
        if (surrogate) surrogate.init();
    });
}

if (typeof window !== 'undefined') {
    window.initFlowViewerSections = initFlowViewerSections;
}
