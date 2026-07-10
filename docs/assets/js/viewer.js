/**
 * LBM-2D Interactive Simulation Viewer
 *
 * Canvas-based player for pre-computed simulation data.
 * Reads JSON files exported by scripts/postprocess.py
 *
 * Usage:
 *   initViewer('containerId', 'assets/data/sim_re100.json');
 */
function initViewer(containerId, dataUrl) {
  const container = document.getElementById(containerId);
  if (!container) return;

  // --- Layout ---
  container.style.position = 'relative';
  container.style.background = '#0a0e14';
  container.style.overflow = 'hidden';

  const canvas = document.createElement('canvas');
  canvas.style.width = '100%';
  canvas.style.height = '100%';
  canvas.style.display = 'block';
  container.appendChild(canvas);

  const ctx = canvas.getContext('2d');

  // --- State ---
  let simData = null;
  let currentFrame = 0;
  let isPlaying = true;
  let animId = null;
  let lastTime = 0;

  // --- UI Controls ---
  const controls = document.createElement('div');
  controls.style.cssText =
    'display:flex;align-items:center;gap:10px;padding:8px 12px;' +
    'background:#161b22;border:1px solid #21262d;border-top:none;' +
    'border-radius:0 0 6px 6px;font-family:JetBrains Mono,monospace;';

  const playBtn = document.createElement('button');
  playBtn.textContent = '\u25A0';
  playBtn.style.cssText =
    'background:transparent;border:1px solid #30363d;color:#c9d1d9;' +
    'padding:4px 10px;cursor:pointer;font-family:inherit;font-size:11px;border-radius:4px;';
  playBtn.addEventListener('click', () => {
    isPlaying = !isPlaying;
    playBtn.textContent = isPlaying ? '\u25A0' : '\u25B6';
    if (isPlaying) animate();
  });
  controls.appendChild(playBtn);

  const slider = document.createElement('input');
  slider.type = 'range';
  slider.min = 0;
  slider.max = 0;
  slider.value = 0;
  slider.style.cssText = 'flex:1;accent-color:#00d4ff;height:4px;cursor:pointer;';
  slider.addEventListener('input', () => {
    currentFrame = parseInt(slider.value);
    render();
  });
  controls.appendChild(slider);

  const frameLabel = document.createElement('span');
  frameLabel.style.cssText = 'font-size:10px;color:#8b949e;min-width:70px;text-align:center;';
  controls.appendChild(frameLabel);

  container.parentNode.insertBefore(controls, container.nextSibling);

  // --- Resize ---
  function resize() {
    const rect = container.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    canvas.style.width = rect.width + 'px';
    canvas.style.height = rect.height + 'px';
    render();
  }

  // --- Color mapping (Jet colormap) ---
  function jetColor(t) {
    // t in [0, 1]
    let r, g, b;
    if (t < 0.125) {
      r = 0; g = 0; b = 0.5 + t * 4;
    } else if (t < 0.375) {
      r = 0; g = (t - 0.125) * 4; b = 1;
    } else if (t < 0.625) {
      r = (t - 0.375) * 4; g = 1; b = 1 - (t - 0.375) * 4;
    } else if (t < 0.875) {
      r = 1; g = 1 - (t - 0.625) * 4; b = 0;
    } else {
      r = 1 - (t - 0.875) * 2; g = 0; b = 0;
    }
    return { r: Math.round(r * 255), g: Math.round(g * 255), b: Math.round(b * 255) };
  }

  // --- Render ---
  function render() {
    if (!simData || !simData.frames || simData.frames.length === 0) return;

    const frame = simData.frames[currentFrame];
    const vel = frame.velocity;
    const nx = frame.nx || 100;
    const ny = frame.ny || 38;

    const rect = container.getBoundingClientRect();
    const w = rect.width;
    const h = rect.height;

    // Clear
    ctx.fillStyle = '#0a0e14';
    ctx.fillRect(0, 0, w, h);

    // Find max velocity for scaling
    let vmax = 0;
    for (let i = 0; i < vel.length; i++) {
      if (vel[i] > vmax) vmax = vel[i];
    }
    if (vmax < 0.01) vmax = 0.01;

    const cellW = w / nx;
    const cellH = h / ny;

    // Draw velocity field
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++) {
        const val = vel[y * nx + x] / vmax;
        const c = jetColor(Math.max(0, Math.min(1, val)));
        ctx.fillStyle = `rgb(${c.r},${c.g},${c.b})`;
        ctx.fillRect(x * cellW, y * cellH, cellW + 0.5, cellH + 0.5);
      }
    }

    // Draw cylinder location (approximate)
    const cx = nx * 0.25;
    const cy = ny * 0.5;
    const radius = ny * 0.1;
    ctx.beginPath();
    ctx.arc(cx * cellW, cy * cellH, radius * cellW, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(255,255,255,0.3)';
    ctx.fill();
    ctx.strokeStyle = 'rgba(255,255,255,0.6)';
    ctx.lineWidth = 1;
    ctx.stroke();

    // HUD
    ctx.fillStyle = 'rgba(0, 212, 255, 0.8)';
    ctx.font = '11px JetBrains Mono, monospace';
    ctx.fillText(`Frame ${frame.frame}`, 10, 20);

    frameLabel.textContent = `Frame ${frame.frame} / ${simData.frames.length - 1}`;
  }

  // --- Animation loop ---
  function animate() {
    if (!isPlaying || !simData) return;

    currentFrame = (currentFrame + 1) % simData.frames.length;
    slider.value = currentFrame;
    render();

    animId = requestAnimationFrame(animate);
  }

  // --- Load data ---
  function loadData(url) {
    fetch(url)
      .then(r => {
        if (!r.ok) throw new Error('Network error');
        return r.json();
      })
      .then(data => {
        simData = data;
        slider.max = data.frames.length - 1;
        slider.value = 0;
        currentFrame = 0;
        render();
        if (isPlaying) animate();
      })
      .catch(() => {
        container.innerHTML =
          '<div style="display:flex;align-items:center;justify-content:center;height:100%;' +
          'color:#8b949e;font-size:12px;text-align:center;padding:20px;">' +
          'Simulation data not found.<br>' +
          '<span style="display:block;margin-top:8px;font-size:10px;color:#ff6b35;">' +
          'Run: python3 scripts/postprocess.py output/re100 --json --every 5</span></div>';
      });
  }

  // --- Init ---
  window.addEventListener('resize', resize);
  resize();

  if (dataUrl) loadData(dataUrl);

  return {
    load: loadData,
    play: () => { isPlaying = true; playBtn.textContent = '\u25A0'; animate(); },
    pause: () => { isPlaying = false; playBtn.textContent = '\u25B6'; if (animId) cancelAnimationFrame(animId); },
    goToFrame: (f) => { currentFrame = Math.min(f, simData ? simData.frames.length - 1 : 0); slider.value = currentFrame; render(); },
    destroy: () => { if (animId) cancelAnimationFrame(animId); window.removeEventListener('resize', resize); },
  };
}
