#pragma once
#include "lbm.hpp"
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <limits>

// ==========================================================================
// Block-Structured Adaptive Mesh Refinement (AMR) for D2Q9 LBM
// ==========================================================================
// Architecture:
//   AMRGrid manages levels 0..N. Level 0 is a single block covering the
//   full domain at base resolution. Level l has blocks at 2^l resolution.
//   All levels advance at the same dt.
//
//   Each AMRBlock stores its own f, f_next, obstacle mask, and body force.
//   Ghost nodes at coarse-fine boundaries are filled via bilinear
//   prolongation from the parent level.
// ==========================================================================

// ------------------------------------------------------------------
// AMRBlock: a single rectangular grid block at a given refinement level
// ------------------------------------------------------------------
struct AMRBlock {
    int nx, ny;             // number of fluid nodes (excluding ghost layer)
    int x0, y0;             // offset in base-level (level 0) coordinates
    int level;              // refinement level (0 = base)
    double dx;              // grid spacing = 1.0 / (1 << level)

    std::vector<double> f;          // distribution [total_nodes * 9]
    std::vector<double> f_next;     // streaming buffer
    std::vector<uint8_t> obstacle;  // 1 = solid node
    std::vector<double> fx_body;    // cumulative drag
    std::vector<double> fy_body;    // cumulative lift

    BounceBackGeometry bb_geom;     // geometry for interpolated bounce-back

    static constexpr int GHOST = 1; // ghost cell width on each side

    int total_nx() const { return nx + 2 * GHOST; }
    int total_ny() const { return ny + 2 * GHOST; }
    int total_nodes() const { return total_nx() * total_ny(); }

    // Map from logical (x, y) within block (0..nx-1, 0..ny-1) to
    // array index including ghost layer.
    int idx(int x, int y) const {
        return (y + GHOST) * total_nx() + (x + GHOST);
    }

    // Map from ghost-aware coordinates (gx, gy) to array index.
    int gidx(int gx, int gy) const {
        return gy * total_nx() + gx;
    }

    AMRBlock() : nx(0), ny(0), x0(0), y0(0), level(0), dx(1.0) {}

    AMRBlock(int nx_, int ny_, int x0_, int y0_, int level_)
        : nx(nx_), ny(ny_), x0(x0_), y0(y0_), level(level_)
        , dx(1.0 / static_cast<double>(1 << level_))
    {
        int n = total_nodes() * 9;
        f.assign(n, 0.0);
        f_next.assign(n, 0.0);
        obstacle.assign(total_nodes(), 0);
        fx_body.assign(total_nodes(), 0.0);
        fy_body.assign(total_nodes(), 0.0);
    }

    void reset_forces() {
        std::fill(fx_body.begin(), fx_body.end(), 0.0);
        std::fill(fy_body.begin(), fy_body.end(), 0.0);
    }

    // Initialize all fluid nodes with equilibrium at (u0, v0)
    void init_equilibrium(double rho0, double u0, double v0) {
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                int i = idx(x, y);
                if (obstacle[i]) continue;
                double* f_node = &f[i * 9];
                for (int d = 0; d < 9; ++d) {
                    f_node[d] = compute_equilibrium(d, rho0, u0, v0);
                }
            }
        }
    }
};

// ------------------------------------------------------------------
// AMRGrid: manages the block hierarchy across refinement levels
// ------------------------------------------------------------------
struct AMRGrid {
    int base_nx, base_ny;                // level 0 dimensions
    std::vector<std::vector<AMRBlock>> levels;  // levels[level] = list of blocks

    // Regridding parameters
    double refine_thresh;                // sensor threshold for refinement
    double coarsen_thresh;               // sensor threshold for coarsening
    int n_regrid;                        // regrid every N_regrid steps
    int regrid_counter;                  // steps since last regrid
    bool skip_restriction;               // debug: skip restriction step

    AMRGrid(int nx, int ny)
        : base_nx(nx), base_ny(ny)
        , refine_thresh(0.005)
        , coarsen_thresh(0.0005)
        , n_regrid(100)
        , regrid_counter(0)
        , skip_restriction(false)
    {
        // Level 0: single block covering the full domain
        levels.resize(1);
        levels[0].emplace_back(nx, ny, 0, 0, 0);
    }

    // Accessors
    int n_levels() const { return static_cast<int>(levels.size()); }

    // ------------------------------------------------------------------
    // Prolongation: bilinear interpolation from coarse parent to fine
    // ghost nodes.  For each fine block, fills the single ghost layer
    // from the coarse level.
    //
    // Coordinate system:
    //   Fine node (fx, fy) in fine-local coords maps to parent coords:
    //     px = fine.x0 + fx * fine.dx
    //     py = fine.y0 + fy * fine.dx
    //   Ghost layer is at fx = -1 or fx = fine.nx (and similarly for y),
    //   spanning px = fine.x0 - fine.dx or fine.x0 + fine.nx * fine.dx.
    // ------------------------------------------------------------------
    void apply_prolongation() {
        if (n_levels() < 2) return;

        for (int lev = 1; lev < n_levels(); ++lev) {
            for (AMRBlock& fine : levels[lev]) {
                for (const AMRBlock& parent : levels[lev - 1]) {
                    // Iterate over fine ghost nodes
                    for (int ly = -1; ly <= fine.ny; ++ly) {
                        for (int lx = -1; lx <= fine.nx; ++lx) {
                            // Skip interior (fluid) nodes
                            if (lx >= 0 && lx < fine.nx &&
                                ly >= 0 && ly < fine.ny) continue;

                            // Fine ghost node -> parent coordinates
                            double px = static_cast<double>(fine.x0) + lx * fine.dx;
                            double py = static_cast<double>(fine.y0) + ly * fine.dx;

                            // Check if within parent block
                            if (px < parent.x0 || px >= parent.x0 + parent.nx - 1) continue;
                            if (py < parent.y0 || py >= parent.y0 + parent.ny - 1) continue;

                            // Nearest 4 coarse nodes for bilinear interpolation
                            int cx0 = static_cast<int>(std::floor(px));
                            int cy0 = static_cast<int>(std::floor(py));
                            int cx1 = std::min(cx0 + 1, parent.nx - 1);
                            int cy1 = std::min(cy0 + 1, parent.ny - 1);

                            double tx = px - cx0;
                            double ty = py - cy0;

                            int gix = lx + fine.GHOST;
                            int giy = ly + fine.GHOST;
                            if (gix < 0 || gix >= fine.total_nx()) continue;
                            if (giy < 0 || giy >= fine.total_ny()) continue;

                            for (int d = 0; d < 9; ++d) {
                                double f00 = parent.f[parent.idx(cx0, cy0) * 9 + d];
                                double f10 = parent.f[parent.idx(cx1, cy0) * 9 + d];
                                double f01 = parent.f[parent.idx(cx0, cy1) * 9 + d];
                                double f11 = parent.f[parent.idx(cx1, cy1) * 9 + d];
                                double f0 = f00 + tx * (f10 - f00);
                                double f1 = f01 + tx * (f11 - f01);
                                fine.f[fine.gidx(gix, giy) * 9 + d] = f0 + ty * (f1 - f0);
                            }
                        }
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Restriction: inject the fine node value centered on each coarse
    // node back onto the parent level.  For coarse node at parent
    // coordinates (pcx, pcy), the corresponding fine node is at
    // fine-local position flx = (pcx - fine.x0) / fine.dx.
    //
    // To avoid boundary mismatch, we restrict only coarse nodes that
    // are not touching the fine block boundary.
    // ------------------------------------------------------------------
    void apply_restriction() {
        if (n_levels() < 2) return;

        for (int lev = 1; lev < n_levels(); ++lev) {
            for (const AMRBlock& fine : levels[lev]) {
                AMRBlock& parent = levels[lev - 1][0]; // single parent block

                // Interior coarse nodes covered by fine block (skip 1-cell margin)
                int margin = 1; // skip boundary to avoid interface mismatch
                int c0 = static_cast<int>(std::ceil(
                    (fine.x0 + margin * fine.dx - parent.x0) / 1.0));
                int r0 = static_cast<int>(std::ceil(
                    (fine.y0 + margin * fine.dx - parent.y0) / 1.0));
                int c1 = static_cast<int>(std::floor(
                    (fine.x0 + (fine.nx - margin) * fine.dx - parent.x0) / 1.0));
                int r1 = static_cast<int>(std::floor(
                    (fine.y0 + (fine.ny - margin) * fine.dx - parent.y0) / 1.0));

                for (int pcy = r0; pcy <= r1; ++pcy) {
                    for (int pcx = c0; pcx <= c1; ++pcx) {
                        if (pcx < 0 || pcx >= parent.nx) continue;
                        if (pcy < 0 || pcy >= parent.ny) continue;

                        double px = static_cast<double>(parent.x0 + pcx);
                        double py = static_cast<double>(parent.y0 + pcy);

                        int flx = static_cast<int>((px - fine.x0) / fine.dx + 0.5);
                        int fly = static_cast<int>((py - fine.y0) / fine.dx + 0.5);
                        if (flx < 0 || flx >= fine.nx) continue;
                        if (fly < 0 || fly >= fine.ny) continue;

                        int ci = fine.idx(flx, fly);
                        if (fine.obstacle[ci]) continue;

                        double* pf = &parent.f[parent.idx(pcx, pcy) * 9];
                        const double* ff = &fine.f[ci * 9];
                        for (int d = 0; d < 9; ++d) pf[d] = ff[d];
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Refinement sensor: |grad u| * dx on a given level
    // ------------------------------------------------------------------
    std::vector<double> compute_sensor(int level) {
        std::vector<double> sensor(base_nx * base_ny, 0.0);

        for (const AMRBlock& blk : levels[level]) {
            double ds = blk.dx;
            for (int y = 1; y < blk.ny - 1; ++y) {
                for (int x = 1; x < blk.nx - 1; ++x) {
                    int ci = blk.idx(x, y);
                    if (blk.obstacle[ci]) continue;

                    double rho_c, u_c, v_c;
                    compute_macros(&blk.f[ci * 9], rho_c, u_c, v_c);

                    // Central differences for velocity gradients
                    auto grad = [&](int dx, int dy, double& du, double& dv) {
                        int ni = blk.idx(x + dx, y);
                        double ru, uu, vu;
                        compute_macros(&blk.f[ni * 9], ru, uu, vu);
                        ni = blk.idx(x - dx, y);
                        double rl, ul, vl;
                        compute_macros(&blk.f[ni * 9], rl, ul, vl);
                        du = (uu - ul) / (2.0 * ds * dx);

                        ni = blk.idx(x, y + dy);
                        compute_macros(&blk.f[ni * 9], ru, uu, vu);
                        ni = blk.idx(x, y - dy);
                        compute_macros(&blk.f[ni * 9], rl, ul, vl);
                        dv = (vu - vl) / (2.0 * ds * dy);
                    };

                    double du_dx, du_dy, dv_dx, dv_dy;
                    grad(1, 1, du_dx, dv_dx);
                    // Use y-difference for du_dy, dv_dy
                    double ru, uu, vu;
                    int ni = blk.idx(x, y + 1);
                    compute_macros(&blk.f[ni * 9], ru, uu, vu);
                    ni = blk.idx(x, y - 1);
                    double rl, ul, vl;
                    compute_macros(&blk.f[ni * 9], rl, ul, vl);
                    du_dy = (uu - ul) / (2.0 * ds);
                    ni = blk.idx(x, y + 1);
                    compute_macros(&blk.f[ni * 9], ru, uu, vu);
                    ni = blk.idx(x, y - 1);
                    compute_macros(&blk.f[ni * 9], rl, ul, vl);
                    dv_dy = (vu - vl) / (2.0 * ds);

                    double grad_mag = std::sqrt(du_dx*du_dx + du_dy*du_dy +
                                                dv_dx*dv_dx + dv_dy*dv_dy);
                    double val = grad_mag * ds;

                    // Map to base grid coordinates
                    int bx = blk.x0 + x;
                    int by = blk.y0 + y;
                    if (bx >= 0 && bx < base_nx && by >= 0 && by < base_ny) {
                        sensor[by * base_nx + bx] = std::max(sensor[by * base_nx + bx], val);
                    }
                }
            }
        }
        return sensor;
    }

    // ------------------------------------------------------------------
    // Block clustering: tile-based generation of refined blocks
    // ------------------------------------------------------------------
    std::vector<AMRBlock> cluster_blocks(const std::vector<int>& tagged,
                                          int level, int x0, int y0, int w, int h) {
        constexpr int TILE_SIZE = 16;
        int nt_x = (w + TILE_SIZE - 1) / TILE_SIZE;
        int nt_y = (h + TILE_SIZE - 1) / TILE_SIZE;
        std::vector<uint8_t> tile_marked(nt_x * nt_y, 0);

        for (int ty = 0; ty < nt_y; ++ty) {
            for (int tx = 0; tx < nt_x; ++tx) {
                bool any_tagged = false;
                for (int dy = 0; dy < TILE_SIZE && !any_tagged; ++dy) {
                    for (int dx = 0; dx < TILE_SIZE && !any_tagged; ++dx) {
                        int gx = x0 + tx * TILE_SIZE + dx;
                        int gy = y0 + ty * TILE_SIZE + dy;
                        if (gx >= 0 && gx < base_nx && gy >= 0 && gy < base_ny) {
                            if (tagged[gy * base_nx + gx]) any_tagged = true;
                        }
                    }
                }
                tile_marked[ty * nt_x + tx] = any_tagged ? 1 : 0;
            }
        }

        // Merge adjacent marked tiles into rectangles (simple greedy)
        std::vector<AMRBlock> blocks;
        std::vector<uint8_t> used(nt_x * nt_y, 0);

        for (int ty = 0; ty < nt_y; ++ty) {
            for (int tx = 0; tx < nt_x; ++tx) {
                if (!tile_marked[ty * nt_x + tx] || used[ty * nt_x + tx]) continue;

                // Expand horizontally
                int tx2 = tx;
                while (tx2 + 1 < nt_x && tile_marked[ty * nt_x + tx2 + 1] &&
                       !used[ty * nt_x + tx2 + 1]) ++tx2;

                // Expand vertically
                int ty2 = ty;
                bool valid;
                do {
                    ++ty2;
                    valid = true;
                    for (int t = tx; t <= tx2 && valid; ++t) {
                        if (ty2 >= nt_y || !tile_marked[ty2 * nt_x + t] ||
                            used[ty2 * nt_x + t]) valid = false;
                    }
                } while (valid);
                --ty2;

                // Mark tiles as used
                for (int tty = ty; tty <= ty2; ++tty) {
                    for (int ttx = tx; ttx <= tx2; ++ttx) {
                        used[tty * nt_x + ttx] = 1;
                    }
                }

                int bx0 = x0 + tx * TILE_SIZE;
                int by0 = y0 + ty * TILE_SIZE;
                int bx1 = std::min(base_nx, x0 + (tx2 + 1) * TILE_SIZE);
                int by1 = std::min(base_ny, y0 + (ty2 + 1) * TILE_SIZE);

                // Expand by ghost cells
                int stride = 1 << level;
                bx0 = std::max(0, bx0 - stride);
                by0 = std::max(0, by0 - stride);
                bx1 = std::min(base_nx, bx1 + stride);
                by1 = std::min(base_ny, by1 + stride);

                int blk_nx = bx1 - bx0;
                int blk_ny = by1 - by0;
                if (blk_nx >= 4 && blk_ny >= 4) {
                    blocks.emplace_back(blk_nx, blk_ny, bx0, by0, level);
                }
            }
        }

        return blocks;
    }

    // ------------------------------------------------------------------
    // Regridding: rebuild block hierarchy based on refinement sensor
    // ------------------------------------------------------------------
    void regrid(int max_levels = 2) {
        // Compute sensor on the finest existing level
        int finest = std::min(n_levels() - 1, max_levels - 1);
        auto sensor = compute_sensor(finest);

        // Tag cells for refinement
        std::vector<int> tagged(base_nx * base_ny, 0);
        for (int i = 0; i < base_nx * base_ny; ++i) {
            if (sensor[i] > refine_thresh) tagged[i] = 1;
        }

        // Build new level 1 blocks
        std::vector<AMRBlock> new_blocks;
        if (!levels[0].empty()) {
            const AMRBlock& base = levels[0][0];
            new_blocks = cluster_blocks(tagged, 1, 0, 0, base_nx, base_ny);
        }

        // Preserve level 0
        std::vector<AMRBlock> old_level0 = std::move(levels[0]);
        levels.clear();
        levels.push_back(std::move(old_level0));
        if (!new_blocks.empty()) {
            levels.push_back(std::move(new_blocks));
        }

        // Interpolate f from old hierarchy to new blocks
        // (For simplicity, initialize new blocks from parent level 0)
        if (n_levels() >= 2) {
            for (AMRBlock& blk : levels[1]) {
                blk.init_equilibrium(1.0, 0.1, 0.0);
            }
        }

        regrid_counter = 0;
    }

    // ------------------------------------------------------------------
    // Time step: sequential coarse-then-fine with restriction
    //
    // Standard AMR approach for LBM:
    //   1. Collide + stream on coarse level (level 0)
    //   2. Prolongation (coarse -> fine ghost cells)
    //   3. Collide + stream on fine levels
    //   4. Restriction (fine -> coarse, overwrite covered nodes)
    //   5. Boundary conditions (on coarse)
    //   6. Force extraction (on coarse)
    // ------------------------------------------------------------------
    void execute_amr_step(double tau, double u_inflow) {
        double cs_sq = g_use_les ? (g_cs * g_cs) : 0.0;
        MRTParams mrt = MRTParams::from_tau(tau);
        int n_lvls = n_levels();

        // --- Helper lambda for collide+stream on a block ---
        auto collide_block = [&](AMRBlock& blk) {
            for (int y = 0; y < blk.ny; ++y) {
                for (int x = 0; x < blk.nx; ++x) {
                    int ci = blk.idx(x, y);
                    if (blk.obstacle[ci]) continue;
                    double rho, u, v;
                    compute_macros(&blk.f[ci * 9], rho, u, v);
                    mrt_collide(&blk.f[ci * 9], rho, u, v, mrt, cs_sq, tau);
                }
            }
        };

        auto stream_block = [&](AMRBlock& blk, bool is_base) {
            int TNX = blk.total_nx();
            int TNY = blk.total_ny();
            std::fill(blk.f_next.begin(), blk.f_next.end(), 0.0);
            bool periodic_y = (g_case != CaseType::CAVITY);
            bool use_bouzidi = blk.bb_geom.is_valid() && is_base;

            for (int y = 0; y < blk.ny; ++y) {
                for (int x = 0; x < blk.nx; ++x) {
                    int ci = blk.idx(x, y);
                    if (blk.obstacle[ci]) continue;
                    double* f_node = &blk.f[ci * 9];
                    for (int d = 0; d < 9; ++d) {
                        int nx_ = x + cx[d];
                        int ny_ = y + cy[d];
                        if (is_base) {
                            if (periodic_y) {
                                if (ny_ < 0) ny_ += blk.ny;
                                if (ny_ >= blk.ny) ny_ -= blk.ny;
                            } else {
                                if (ny_ < 0 || ny_ >= blk.ny) continue;
                            }
                            if (nx_ < 0 || nx_ >= blk.nx) continue;
                            int gx = nx_ + blk.GHOST;
                            int gy = ny_ + blk.GHOST;
                            int target = gy * TNX + gx;
                            if (blk.obstacle[target]) {
                                if (use_bouzidi) {
                                    double q = blk.bb_geom.compute_q(
                                        static_cast<double>(x),
                                        static_cast<double>(y), d);
                                    int bb = bounce_back[d];
                                    if (q < 0.5) {
                                        int sx = x - cx[d], sy = y - cy[d];
                                        double fsrc = 0.0;
                                        if (sx >= 0 && sx < blk.nx && sy >= 0 && sy < blk.ny) {
                                            fsrc = blk.f[blk.idx(sx, sy) * 9 + d];
                                        }
                                        blk.f_next[ci * 9 + bb] = 2.0 * q * f_node[d]
                                            + (1.0 - 2.0 * q) * fsrc;
                                    } else {
                                        double inv2q = 1.0 / (2.0 * q);
                                        double fopp = blk.f[ci * 9 + bb];
                                        blk.f_next[ci * 9 + bb] = inv2q * f_node[d]
                                            + (1.0 - inv2q) * fopp;
                                    }
                                } else {
                                    int bb = bounce_back[d];
                                    blk.f_next[ci * 9 + bb] = f_node[d];
                                }
                            } else {
                                blk.f_next[target * 9 + d] = f_node[d];
                            }
                        } else {
                            int gx = nx_ + blk.GHOST;
                            int gy = ny_ + blk.GHOST;
                            if (gx >= 0 && gx < TNX && gy >= 0 && gy < TNY) {
                                int target = gy * TNX + gx;
                                if (blk.obstacle[target]) {
                                    int bb = bounce_back[d];
                                    blk.f_next[ci * 9 + bb] = f_node[d];
                                } else {
                                    blk.f_next[target * 9 + d] = f_node[d];
                                }
                            } else {
                                int bb = bounce_back[d];
                                blk.f_next[ci * 9 + bb] = f_node[d];
                            }
                        }
                    }
                }
            }
            blk.f.swap(blk.f_next);
        };

        // 1. Coarse advance (level 0)
        if (n_lvls >= 1) {
            collide_block(levels[0][0]);
            stream_block(levels[0][0], true);
        }

        // 2. Prolongation: fill fine ghost nodes from coarse
        apply_prolongation();

        // 3. Fine advance (levels >= 1)
        for (int lev = 1; lev < n_lvls; ++lev) {
            for (AMRBlock& blk : levels[lev]) {
                collide_block(blk);
                stream_block(blk, false);
            }
        }

        // 4. Restriction: inject fine values onto coarse
        if (!skip_restriction) apply_restriction();

        // 5. Boundary conditions on level 0
        if (n_lvls >= 1) {
            AMRBlock& base = levels[0][0];
            if (g_case == CaseType::CAVITY) {
                enforce_lid_on_block(base, u_inflow);
            } else if (g_case == CaseType::STEP) {
                enforce_step_inflow_on_block(base, NY / 3, NY - 1 - NY / 3, u_inflow);
                enforce_outflow_on_block(base);
            } else if (g_case != CaseType::RIBS) {
                enforce_inflow_on_block(base, u_inflow);
                enforce_outflow_on_block(base);
            }
        }

        // 6. Force extraction on level 0
        if (g_case != CaseType::CAVITY && n_lvls >= 1) {
            AMRBlock& base = levels[0][0];
            base.reset_forces();
            for (int y = 0; y < base.ny; ++y) {
                for (int x = 0; x < base.nx; ++x) {
                    int ci = base.idx(x, y);
                    if (base.obstacle[ci]) continue;
                    for (int d = 0; d < 9; ++d) {
                        int nx_ = x + cx[d];
                        int ny_ = y + cy[d];
                        if (g_case == CaseType::CAVITY) {
                            if (ny_ < 0 || ny_ >= base.ny) continue;
                        } else {
                            if (ny_ < 0) ny_ += base.ny;
                            if (ny_ >= base.ny) ny_ -= base.ny;
                        }
                        if (g_case == CaseType::RIBS) {
                            if (nx_ < 0) nx_ += base.nx;
                            if (nx_ >= base.nx) nx_ -= base.nx;
                        } else {
                            if (nx_ < 0 || nx_ >= base.nx) continue;
                        }
                        int ti = base.idx(nx_, ny_);
                        if (base.obstacle[ti]) {
                            int bb = bounce_back[d];
                            double f_boundary = base.f[ci * 9 + bb];
                            base.fx_body[ci] += cx[d] * 2.0 * f_boundary;
                            base.fy_body[ci] += cy[d] * 2.0 * f_boundary;
                        }
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // BC enforcements on a single block (internal helpers)
    // ------------------------------------------------------------------
    void enforce_inflow_on_block(AMRBlock& blk, double u_inflow) {
        for (int y = 0; y < blk.ny; ++y) {
            int ci = blk.idx(0, y);
            if (blk.obstacle[ci]) continue;
            double* f_node = &blk.f[ci * 9];
            double rho = (f_node[0] + f_node[2] + f_node[4]
                        + 2.0 * (f_node[3] + f_node[6] + f_node[7]))
                        / (1.0 - u_inflow);
            f_node[1] = f_node[3] + (2.0 / 3.0) * rho * u_inflow;
            f_node[5] = f_node[7] + (1.0 / 6.0) * rho * u_inflow;
            f_node[8] = f_node[6] + (1.0 / 6.0) * rho * u_inflow;
        }
    }

    void enforce_outflow_on_block(AMRBlock& blk) {
        for (int y = 0; y < blk.ny; ++y) {
            int ci = blk.idx(blk.nx - 1, y);
            int ci_in = blk.idx(blk.nx - 2, y);
            double* f_node = &blk.f[ci * 9];
            const double* f_in = &blk.f[ci_in * 9];
            for (int d = 0; d < 9; ++d) f_node[d] = f_in[d];
        }
    }

    void enforce_lid_on_block(AMRBlock& blk, double u_lid) {
        for (int x = 0; x < blk.nx; ++x) {
            int ci = blk.idx(x, blk.ny - 1);
            double* f_node = &blk.f[ci * 9];
            double rho, u, v;
            compute_macros(f_node, rho, u, v);
            for (int d = 0; d < 9; ++d) {
                f_node[d] = compute_equilibrium(d, rho, u_lid, 0.0);
            }
        }
    }

    void enforce_step_inflow_on_block(AMRBlock& blk, int h_step, int h_inlet, double u_max) {
        for (int y = h_step; y < blk.ny; ++y) {
            int ci = blk.idx(0, y);
            if (blk.obstacle[ci]) continue;
            double* f_node = &blk.f[ci * 9];
            double yy = static_cast<double>(y - h_step);
            double h = static_cast<double>(h_inlet);
            double u_local = u_max * 4.0 * yy * (h - yy) / (h * h);
            double rho = (f_node[0] + f_node[2] + f_node[4]
                        + 2.0 * (f_node[3] + f_node[6] + f_node[7]))
                        / (1.0 - u_local);
            f_node[1] = f_node[3] + (2.0 / 3.0) * rho * u_local;
            f_node[5] = f_node[7] + (1.0 / 6.0) * rho * u_local;
            f_node[8] = f_node[6] + (1.0 / 6.0) * rho * u_local;
        }
    }

    // ------------------------------------------------------------------
    // JSON frame export for AMR grid
    // ------------------------------------------------------------------
    void save_json_frame(int step, const std::string& output_dir) {
        int NX = base_nx, NY = base_ny;
        int ds = std::max(1, NX / 100);
        int nx_ds = (NX + ds - 1) / ds;
        int ny_ds = (NY + ds - 1) / ds;

        std::string dir = output_dir + "/frames";
        std::filesystem::create_directories(dir);
        std::string filename = dir + "/frame_" + std::to_string(step) + ".json";
        std::ofstream out(filename);
        out.precision(6);
        out << std::fixed;

        int n_ds = nx_ds * ny_ds;
        std::vector<double> vel_arr(n_ds, 0.0);
        std::vector<double> u_arr(n_ds, 0.0);
        std::vector<double> v_arr(n_ds, 0.0);
        std::vector<double> rho_arr(n_ds, 0.0);
        std::vector<int> obst_arr(n_ds, 0);

        // Sample from finest available level
        const AMRBlock* sample_block = &levels[0][0];
        int sample_level = 0;
        for (int lev = n_levels() - 1; lev >= 0; --lev) {
            if (!levels[lev].empty()) {
                sample_block = &levels[lev][0];
                sample_level = lev;
                break;
            }
        }

        int idx2 = 0;
        for (int y = 0; y < NY; y += ds) {
            for (int x = 0; x < NX; x += ds) {
                // Find which block contains (x, y), preferring finer levels
                bool found = false;
                for (int lev = n_levels() - 1; lev >= 0; --lev) {
                    for (const AMRBlock& blk : levels[lev]) {
                        if (x >= blk.x0 && x < blk.x0 + blk.nx &&
                            y >= blk.y0 && y < blk.y0 + blk.ny) {
                            int bx = x - blk.x0;
                            int by = y - blk.y0;
                            int ci = blk.idx(bx, by);
                            if (blk.obstacle[ci]) {
                                obst_arr[idx2] = 1;
                            } else {
                                double r, uu, vv;
                                compute_macros(&blk.f[ci * 9], r, uu, vv);
                                if (r < 1e-12) { uu = 0.0; vv = 0.0; }
                                vel_arr[idx2] = std::sqrt(uu*uu + vv*vv);
                                u_arr[idx2] = uu;
                                v_arr[idx2] = vv;
                                rho_arr[idx2] = r;
                            }
                            found = true;
                            break;
                        }
                    }
                    if (found) break;
                }
                if (!found) {
                    obst_arr[idx2] = 1;
                }
                ++idx2;
            }
        }

        out << "{\"nx\":" << nx_ds << ",\"ny\":" << ny_ds << ",\"velocity\":[";
        for (int i = 0; i < n_ds; ++i) {
            if (i > 0) out << ",";
            out << vel_arr[i];
        }
        out << "],\"u\":[";
        for (int i = 0; i < n_ds; ++i) {
            if (i > 0) out << ",";
            out << u_arr[i];
        }
        out << "],\"v\":[";
        for (int i = 0; i < n_ds; ++i) {
            if (i > 0) out << ",";
            out << v_arr[i];
        }
        out << "],\"rho\":[";
        for (int i = 0; i < n_ds; ++i) {
            if (i > 0) out << ",";
            out << rho_arr[i];
        }
        out << "],\"p\":[";
        for (int i = 0; i < n_ds; ++i) {
            if (i > 0) out << ",";
            out << (rho_arr[i] / 3.0);
        }
        out << "],\"obstacle\":[";
        for (int i = 0; i < n_ds; ++i) {
            if (i > 0) out << ",";
            out << obst_arr[i];
        }
        out << "]}";
    }
};
