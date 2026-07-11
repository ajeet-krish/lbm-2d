/**
 * LBM-2D Interactive Simulation Viewer
 * Canvas-based animation player for pre-computed simulation data.
 * Loads velocity field JSON + force coefficient JSON.
 */
function initViewer(containerId, simDataUrl, forceDataUrl) {
    const container = document.getElementById(containerId);
    if (!container) return null;

    container.style.position = 'relative';
    container.style.background = '#0a0e14';
    container.style.overflow = 'hidden';

    const canvas = document.createElement('canvas');
    canvas.style.width = '100%';
    canvas.style.height = '100%';
    canvas.style.display = 'block';
    container.appendChild(canvas);

    const ctx = canvas.getContext('2d');

    // State
    let simData = null;
    let forceData = null;
    let currentFrame = 0;
    let isPlaying = true;
    let animId = null;
    let speed = 1;              // multiplier (0.5, 1, 2, 4)
    let lastTimestamp = 0;
    let frameInterval = 100;    // ms between frames at 1x
    let onFrameChange = null;   // callback(frameNum, cd, cl)

    // UI container (outside canvas, below it)
    const controls = document.createElement('div');
    controls.style.cssText =
        'display:flex;align-items:center;gap:8px;padding:6px 12px;' +
        'background:#161b22;border:1px solid #21262d;border-top:none;' +
        'border-radius:0 0 6px 6px;font-family:JetBrains Mono,monospace;font-size:11px;';

    // --- Play/Pause ---
    const playBtn = document.createElement('button');
    playBtn.textContent = '\u25A0';
    playBtn.title = 'Play / Pause';
    playBtn.style.cssText =
        'background:transparent;border:1px solid #30363d;color:#c9d1d9;' +
        'padding:2px 10px;cursor:pointer;font-family:inherit;font-size:12px;border-radius:4px;' +
        'line-height:1.4;';
    playBtn.addEventListener('click', () => togglePlay());
    controls.appendChild(playBtn);

    // --- Step Backward ---
    const prevBtn = document.createElement('button');
    prevBtn.textContent = '\u23EE';
    prevBtn.title = 'Previous frame';
    prevBtn.style.cssText =
        'background:transparent;border:1px solid #30363d;color:#c9d1d9;' +
        'padding:2px 8px;cursor:pointer;font-family:inherit;font-size:12px;border-radius:4px;' +
        'line-height:1.4;';
    prevBtn.addEventListener('click', () => { currentFrame = Math.max(0, currentFrame - 1); slider.value = currentFrame; render(); });
    controls.appendChild(prevBtn);

    // --- Step Forward ---
    const nextBtn = document.createElement('button');
    nextBtn.textContent = '\u23ED';
    nextBtn.title = 'Next frame';
    nextBtn.style.cssText =
        'background:transparent;border:1px solid #30363d;color:#c9d1d9;' +
        'padding:2px 8px;cursor:pointer;font-family:inherit;font-size:12px;border-radius:4px;' +
        'line-height:1.4;';
    nextBtn.addEventListener('click', () => { currentFrame = Math.min(simData ? simData.frames.length - 1 : 0, currentFrame + 1); slider.value = currentFrame; render(); });
    controls.appendChild(nextBtn);

    // --- Frame slider ---
    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = 0;
    slider.max = 0;
    slider.value = 0;
    slider.style.cssText = 'flex:1;accent-color:#00d4ff;height:4px;cursor:pointer;min-width:60px;';
    slider.addEventListener('input', () => {
        currentFrame = parseInt(slider.value);
        render();
    });
    controls.appendChild(slider);

    // --- Speed buttons ---
    const speeds = [0.5, 1, 2, 4];
    const speedBtns = [];
    const speedContainer = document.createElement('div');
    speedContainer.style.cssText = 'display:flex;gap:2px;';
    speeds.forEach((s) => {
        const btn = document.createElement('button');
        btn.textContent = s + 'x';
        btn.style.cssText =
            'background:transparent;border:1px solid #30363d;color:#8b949e;' +
            'padding:2px 6px;cursor:pointer;font-family:inherit;font-size:10px;border-radius:3px;' +
            'line-height:1.4;' + (s === speed ? 'background:#21262d;color:#c9d1d9;' : '');
        btn.addEventListener('click', () => {
            speed = s;
            speedBtns.forEach((b) => { b.style.background = 'transparent'; b.style.color = '#8b949e'; });
            btn.style.background = '#21262d';
            btn.style.color = '#c9d1d9';
        });
        speedBtns.push(btn);
        speedContainer.appendChild(btn);
    });
    controls.appendChild(speedContainer);

    container.parentNode.insertBefore(controls, container.nextSibling);

    // --- Resize ---
    function resize() {
        const rect = container.getBoundingClientRect();
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;
        ctx.scale(dpr, dpr);
        render();
    }

    // --- Jet colormap ---
    function jetColor(t) {
        t = Math.max(0, Math.min(1, t));
        let r, g, b;
        if (t < 0.125) { r = 0; g = 0; b = 0.5 + t * 4; }
        else if (t < 0.375) { r = 0; g = (t - 0.125) * 4; b = 1; }
        else if (t < 0.625) { r = (t - 0.375) * 4; g = 1; b = 1 - (t - 0.375) * 4; }
        else if (t < 0.875) { r = 1; g = 1 - (t - 0.625) * 4; b = 0; }
        else { r = 1 - (t - 0.875) * 2; g = 0; b = 0; }
        return { r: Math.round(r * 255), g: Math.round(g * 255), b: Math.round(b * 255) };
    }

    // --- Render ---
    function render() {
        if (!simData || !simData.frames || simData.frames.length === 0) return;

        const frame = simData.frames[currentFrame];
        if (!frame) return;

        const vel = frame.velocity;
        const nx = frame.nx || simData.meta.nx;
        const ny = frame.ny || simData.meta.ny;

        const rect = container.getBoundingClientRect();
        const w = rect.width;
        const h = rect.height;

        // Find max velocity for scaling
        let vmax = 0;
        for (let i = 0; i < vel.length; i++) { if (vel[i] > vmax) vmax = vel[i]; }
        if (vmax < 0.01) vmax = 0.01;

        const cellW = w / nx;
        const cellH = h / ny;

        // Draw velocity field
        for (let y = 0; y < ny; y++) {
            for (let x = 0; x < nx; x++) {
                const val = vel[y * nx + x] / vmax;
                const c = jetColor(val);
                ctx.fillStyle = 'rgb(' + c.r + ',' + c.g + ',' + c.b + ')';
                ctx.fillRect(x * cellW, y * cellH, cellW + 0.5, cellH + 0.5);
            }
        }

        // Draw cylinder location
        const cx = nx * 0.25;
        const cy = ny * 0.5;
        const radius = ny * 0.1;
        ctx.beginPath();
        ctx.arc(cx * cellW, cy * cellH, radius * cellW, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(255,255,255,0.15)';
        ctx.fill();
        ctx.strokeStyle = 'rgba(255,255,255,0.5)';
        ctx.lineWidth = 1;
        ctx.stroke();

        // HUD: step number (top left)
        ctx.fillStyle = 'rgba(0, 212, 255, 0.85)';
        ctx.font = '12px JetBrains Mono, monospace';
        ctx.fillText('Step ' + frame.frame, 10, 20);

        // HUD: Cd / Cl (bottom left)
        const stepNum = frame.frame;
        let cdVal = '--';
        let clVal = '--';
        if (forceData && forceData.data && forceData.data.length > 0) {
            // Find nearest step in force data
            let nearest = forceData.data[0];
            for (let i = 0; i < forceData.data.length; i++) {
                if (Math.abs(forceData.data[i].step - stepNum) < Math.abs(nearest.step - stepNum)) {
                    nearest = forceData.data[i];
                }
            }
            cdVal = nearest.cd.toFixed(4);
            clVal = nearest.cl.toFixed(4);
        }

        ctx.font = '12px JetBrains Mono, monospace';

        const hudX = 10;
        const hudY = h - 36;
        ctx.fillStyle = 'rgba(10, 14, 20, 0.65)';
        ctx.fillRect(hudX - 4, hudY - 14, 200, 40);
        ctx.fillStyle = '#ff6b35';
        ctx.fillText('Cd = ' + cdVal, hudX, hudY);
        ctx.fillStyle = '#00d4ff';
        ctx.fillText('Cl = ' + clVal, hudX, hudY + 18);

        // Callback
        if (onFrameChange) onFrameChange(stepNum, cdVal, clVal);
    }

    // --- Animation loop ---
    function animate(timestamp) {
        if (!isPlaying || !simData) return;

        const elapsed = timestamp - lastTimestamp;
        const effectiveInterval = frameInterval / speed;

        if (elapsed >= effectiveInterval) {
            currentFrame = (currentFrame + 1) % simData.frames.length;
            slider.value = currentFrame;
            render();
            lastTimestamp = timestamp;
        }

        animId = requestAnimationFrame(animate);
    }

    function togglePlay() {
        isPlaying = !isPlaying;
        playBtn.textContent = isPlaying ? '\u25A0' : '\u25B6';
        if (isPlaying) {
            lastTimestamp = performance.now();
            animId = requestAnimationFrame(animate);
        } else {
            if (animId) cancelAnimationFrame(animId);
        }
    }

    // --- Load data ---
    function loadData(simUrl, forceUrl) {
        const simPromise = fetch(simUrl).then(function(r) { if (!r.ok) throw Error('Network'); return r.json(); });
        const forcePromise = forceUrl ? fetch(forceUrl).then(function(r) { if (!r.ok) throw Error('Network'); return r.json(); }) : Promise.resolve(null);

        Promise.all([simPromise, forcePromise])
            .then(function(results) {
                simData = results[0];
                forceData = results[1];
                slider.max = simData.frames.length - 1;
                slider.value = 0;
                currentFrame = 0;
                render();
                if (isPlaying) {
                    lastTimestamp = performance.now();
                    if (animId) cancelAnimationFrame(animId);
                    animId = requestAnimationFrame(animate);
                }
            })
            .catch(function() {
                container.innerHTML =
                    '<div style="display:flex;align-items:center;justify-content:center;height:100%;' +
                    'color:#8b949e;font-size:12px;text-align:center;padding:20px;">' +
                    'Simulation data not found.<br>' +
                    '<span style="display:block;margin-top:8px;font-size:10px;color:#ff6b35;">' +
                    'Run python3 scripts/postprocess.py output/reX --json --every 1</span></div>';
            });
    }

    // --- Public API ---
    function load(re) {
        const simUrl = 'assets/data/sim_re' + re + '.json';
        const forceUrl = 'assets/data/force_re' + re + '.json';
        loadData(simUrl, forceUrl);
    }

    // --- Init ---
    window.addEventListener('resize', resize);
    resize();

    if (simDataUrl) loadData(simDataUrl, forceDataUrl);

    return {
        load: load,
        play: function() { if (!isPlaying) togglePlay(); },
        pause: function() { if (isPlaying) togglePlay(); },
        goToFrame: function(f) {
            currentFrame = Math.min(f, simData ? simData.frames.length - 1 : 0);
            slider.value = currentFrame;
            render();
        },
        onFrameChange: function(cb) { onFrameChange = cb; },
        destroy: function() {
            if (animId) cancelAnimationFrame(animId);
            window.removeEventListener('resize', resize);
        },
    };
}
