// ==========================================================================
// LBM-2D WASM Real-Time Rendering Engine
// Velocity contour with dense pathline particle visualization
// ==========================================================================

(function() {
    'use strict';

    const U_INFLOW = 0.1;
    const PARTICLE_COUNT = 800;
    const TRAIL_LENGTH = 60;
    const PARTICLE_DT = 0.4;
    const MAX_STEPS_PER_FRAME = 30;

    // ------------------------------------------------------------------
    // Jet colormap with enhanced contrast
    // ------------------------------------------------------------------
    function buildJetColormap() {
        const t = new Uint8Array(256 * 4);
        for (let i = 0; i < 256; i++) {
            const x = i / 255.0;
            let r, g, b;
            if (x < 0.125)       { r=0; g=0; b=0.5+x*4.0; }
            else if (x < 0.375)  { r=0; g=(x-0.125)*4.0; b=1; }
            else if (x < 0.625)  { r=(x-0.375)*4.0; g=1; b=1-(x-0.375)*4.0; }
            else if (x < 0.875)  { r=1; g=1-(x-0.625)*4.0; b=0; }
            else                  { r=1-(x-0.875)*4.0; g=0; b=0; }
            t[i*4+0] = Math.max(0,Math.min(255,Math.round(r*255)));
            t[i*4+1] = Math.max(0,Math.min(255,Math.round(g*255)));
            t[i*4+2] = Math.max(0,Math.min(255,Math.round(b*255)));
            t[i*4+3] = 255;
        }
        return t;
    }

    // ------------------------------------------------------------------
    // Simulator
    // ------------------------------------------------------------------
    function Simulator(canvas, hudEls) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.hudCd = hudEls.hudCd || null;
        this.hudCl = hudEls.hudCl || null;
        this.hudSteps = hudEls.hudSteps || null;

        this.nx = Module._wasm_get_nx();
        this.ny = Module._wasm_get_ny();

        this.colormap = buildJetColormap();
        this.running = false;
        this.animId = null;
        this.speedMul = 1.0;
        this.stepCount = 0;

        // Seed particles across a wide inlet band
        this.particles = [];
        for (let i = 0; i < PARTICLE_COUNT; i++) {
            this.particles.push(this._createParticle());
        }

        Module._wasm_init(this.nx, this.ny, U_INFLOW, 0.74);
    }

    Simulator.prototype._createParticle = function() {
        // Wide vertical spread, staggered inlet positions
        const sx = 2 + Math.random() * 20;
        const sy = Math.random() * (this.ny - 1);
        return { x: sx, y: sy, trail: [] };
    };

    // ------------------------------------------------------------------
    // Bilinear velocity interpolation + advection
    // ------------------------------------------------------------------
    Simulator.prototype._advect = function(uArr, vArr, obstArr) {
        const nx = this.nx, ny = this.ny;
        for (let i = 0; i < this.particles.length; i++) {
            const p = this.particles[i];
            const fx = p.x, fy = p.y;
            const ix = Math.floor(fx), iy = Math.floor(fy);
            const dx = fx - ix, dy = fy - iy;

            let uVal = 0, vVal = 0;
            if (ix >= 0 && ix < nx-1 && iy >= 0 && iy < ny-1) {
                const i00 = iy*nx+ix, i10 = iy*nx+ix+1;
                const i01 = (iy+1)*nx+ix, i11 = (iy+1)*nx+ix+1;
                uVal = uArr[i00]*(1-dx)*(1-dy) + uArr[i10]*dx*(1-dy)
                     + uArr[i01]*(1-dx)*dy + uArr[i11]*dx*dy;
                vVal = vArr[i00]*(1-dx)*(1-dy) + vArr[i10]*dx*(1-dy)
                     + vArr[i01]*(1-dx)*dy + vArr[i11]*dx*dy;
            }

            p.x += uVal * PARTICLE_DT * this.speedMul;
            p.y += vVal * PARTICLE_DT * this.speedMul;

            // Periodic y
            if (p.y < 0) p.y += ny;
            if (p.y >= ny) p.y -= ny;

            // Exit domain or obstacle collision -> reseed
            if (p.x < 0 || p.x >= nx) {
                const fresh = this._createParticle();
                p.x = fresh.x; p.y = fresh.y; p.trail = [];
                continue;
            }

            const gx = Math.round(p.x), gy = Math.round(p.y);
            if (gx >= 0 && gx < nx && gy >= 0 && gy < ny && obstArr[gy*nx+gx]) {
                const fresh = this._createParticle();
                p.x = fresh.x; p.y = fresh.y; p.trail = [];
                continue;
            }

            p.trail.push({x:p.x, y:p.y});
            if (p.trail.length > TRAIL_LENGTH) p.trail.shift();
        }

        // Periodically inject fresh particles for density
        const targetAlive = Math.floor(PARTICLE_COUNT * 0.95);
        let aliveCount = 0;
        for (let i = 0; i < this.particles.length; i++) {
            if (this.particles[i].trail.length > 0) aliveCount++;
        }
    };

    // ------------------------------------------------------------------
    // Main render
    // ------------------------------------------------------------------
    Simulator.prototype._render = function() {
        if (!this.running) return;

        const uPtr = Module._wasm_get_u_ptr();
        const vPtr = Module._wasm_get_v_ptr();
        const oPtr = Module._wasm_get_obstacle_ptr();
        const uArr = new Float64Array(Module.HEAPF64.buffer, uPtr, this.nx*this.ny);
        const vArr = new Float64Array(Module.HEAPF64.buffer, vPtr, this.nx*this.ny);
        const oArr = new Int32Array(Module.HEAP32.buffer, oPtr, this.nx*this.ny);

        // Run WASM steps
        const steps = Math.max(1, Math.min(MAX_STEPS_PER_FRAME,
            Math.round(MAX_STEPS_PER_FRAME * this.speedMul)));
        for (let s = 0; s < steps; s++) {
            Module._wasm_step();
            this.stepCount++;
        }

        // Advect particles
        this._advect(uArr, vArr, oArr);

        // Build velocity contour pixel buffer
        const imgData = this.ctx.createImageData(this.nx, this.ny);
        const pix = imgData.data;
        const cmap = this.colormap;

        for (let y = 0; y < this.ny; y++) {
            for (let x = 0; x < this.nx; x++) {
                const idx = y*this.nx + x;
                const pi = idx * 4;
                if (oArr[idx]) {
                    pix[pi+0]=10; pix[pi+1]=14; pix[pi+2]=20; pix[pi+3]=255;
                } else {
                    const mag = Math.sqrt(uArr[idx]*uArr[idx] + vArr[idx]*vArr[idx]);
                    // Normalize so that u_inflow=0.1 maps to ~65% of colormap
                    const t = Math.min(1.0, mag / 0.14);
                    const ci = Math.floor(t * 255);
                    pix[pi+0]=cmap[ci*4+0]; pix[pi+1]=cmap[ci*4+1];
                    pix[pi+2]=cmap[ci*4+2]; pix[pi+3]=255;
                }
            }
        }

        // Obstacle outline (white)
        for (let y = 1; y < this.ny-1; y++) {
            for (let x = 1; x < this.nx-1; x++) {
                const idx = y*this.nx + x;
                if (!oArr[idx]) continue;
                for (let dy = -1; dy <= 1; dy++) {
                    for (let dx = -1; dx <= 1; dx++) {
                        if (dx===0 && dy===0) continue;
                        if (!oArr[(y+dy)*this.nx + (x+dx)]) {
                            const pi = idx*4;
                            pix[pi+0]=240; pix[pi+1]=240; pix[pi+2]=240; pix[pi+3]=255;
                            dy=2; break;
                        }
                    }
                }
            }
        }

        // Render at fixed 6x pixel scale
        const scale = 6;
        const dw = this.nx * scale;
        const dh = this.ny * scale;
        if (this.canvas.width !== dw || this.canvas.height !== dh) {
            this.canvas.width = dw;
            this.canvas.height = dh;
        }

        const tempC = document.createElement('canvas');
        tempC.width = this.nx; tempC.height = this.ny;
        const tc = tempC.getContext('2d');
        tc.putImageData(imgData, 0, 0);

        this.ctx.save();
        this.ctx.setTransform(1,0,0,1,0,0);
        this.ctx.imageSmoothingEnabled = false;
        this.ctx.drawImage(tempC, 0, 0, dw, dh);

        // Draw particles as cyan-tinted streaks (scaled)
        this.ctx.setTransform(scale, 0, 0, scale, 0, 0);
        this.ctx.lineWidth = 1.0;
        for (let i = 0; i < this.particles.length; i++) {
            const trail = this.particles[i].trail;
            if (trail.length < 3) continue;
            this.ctx.beginPath();
            this.ctx.moveTo(trail[0].x, trail[0].y);
            for (let j = 1; j < trail.length; j++)
                this.ctx.lineTo(trail[j].x, trail[j].y);
            this.ctx.strokeStyle = 'rgba(0, 212, 255, 0.35)';
            this.ctx.stroke();
        }
        // Particle heads (brighter)
        for (let i = 0; i < this.particles.length; i++) {
            const trail = this.particles[i].trail;
            if (trail.length < 3) continue;
            const h = trail[trail.length-1];
            this.ctx.beginPath();
            this.ctx.arc(h.x, h.y, 0.8, 0, Math.PI*2);
            this.ctx.fillStyle = 'rgba(0, 230, 255, 0.6)';
            this.ctx.fill();
        }
        this.ctx.restore();

        // HUD
        if (this.hudCd) this.hudCd.textContent = 'Cd: ' + Module._wasm_get_cd().toFixed(4);
        if (this.hudCl) this.hudCl.textContent = 'Cl: ' + Module._wasm_get_cl().toFixed(4);
        if (this.hudSteps) this.hudSteps.textContent = 'Steps: ' + this.stepCount;

        this.animId = requestAnimationFrame(this._render.bind(this));
    };

    // ------------------------------------------------------------------
    // Public API
    // ------------------------------------------------------------------
    Simulator.prototype.play = function() {
        if (this.running) return;
        this.running = true;
        this.animId = requestAnimationFrame(this._render.bind(this));
    };

    Simulator.prototype.pause = function() {
        this.running = false;
        if (this.animId) { cancelAnimationFrame(this.animId); this.animId = null; }
    };

    Simulator.prototype.togglePlay = function() {
        if (this.running) this.pause(); else this.play();
    };

    Simulator.prototype.isRunning = function() { return this.running; };
    Simulator.prototype.setSpeed = function(s) { this.speedMul = s; };

    Simulator.prototype.setShape = function(type, cx, cy, size, extra) {
        Module._wasm_set_shape(type, cx, cy, size, extra || 0);
        this.stepCount = 0;
    };

    Simulator.prototype.setRe = function(re, refLen) {
        const nu = U_INFLOW * refLen / re;
        const tau = 0.5 + 3.0 * nu;
        Module._wasm_init(this.nx, this.ny, U_INFLOW, tau);
        this.stepCount = 0;
    };

    Simulator.prototype.setAoA = function(series, aoa) {
        Module._wasm_set_shape(4, 33, this.ny/2, 30, series);
        this.stepCount = 0;
    };

    Simulator.prototype.reset = function() {
        Module._wasm_init(this.nx, this.ny, U_INFLOW, 0.74);
        this.stepCount = 0;
    };

    window.WasmSim = {
        init: function(canvas, hudEls) {
            return new Simulator(canvas, hudEls || {});
        },
    };

})();
