#pragma once
#include "lbm_types.hpp"
#include "geometry.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <cmath>
#include <cstdlib>

// ==========================================================================
// D2Q9 LATTICE BOLTZMANN METHOD -- Core solver
// ==========================================================================

// ------------------------------------------------------------------
// Zou/He velocity inlet: enforce u = u_inflow, v = 0 at x = 0
// ------------------------------------------------------------------
inline void enforce_inflow(LBMCapabilities& sys, double u_inflow) {
    for (int y = 0; y < NY; ++y) {
        int idx = node_index(0, y);
        double* f_node = &sys.f[idx * 9];

        double rho = (f_node[0] + f_node[2] + f_node[4]
                    + 2.0 * (f_node[3] + f_node[6] + f_node[7]))
                    / (1.0 - u_inflow);

        f_node[1] = f_node[3] + (2.0 / 3.0) * rho * u_inflow;
        f_node[5] = f_node[7] + (1.0 / 6.0) * rho * u_inflow;
        f_node[8] = f_node[6] + (1.0 / 6.0) * rho * u_inflow;
    }
}

// ------------------------------------------------------------------
// Parabolic velocity inlet for backward-facing step
// Enforces u(y) as a parabola from y = h_step to y = NY-1, with max u_max
// ------------------------------------------------------------------
inline void enforce_step_inflow(LBMCapabilities& sys, int h_step, int h_inlet, double u_max) {
    for (int y = h_step; y < NY; ++y) {
        if (sys.obstacle[node_index(0, y)]) continue;
        int idx = node_index(0, y);
        double* f_node = &sys.f[idx * 9];

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
// Convective outlet: zero-gradient at x = NX-1
// ------------------------------------------------------------------
inline void enforce_outflow(LBMCapabilities& sys) {
    for (int y = 0; y < NY; ++y) {
        int idx = node_index(NX - 1, y);
        int idx_in = node_index(NX - 2, y);
        double* f_node = &sys.f[idx * 9];
        const double* f_in = &sys.f[idx_in * 9];
        for (int i = 0; i < 9; ++i) {
            f_node[i] = f_in[i];
        }
    }
}

// ------------------------------------------------------------------
// Cavity walls: mark bottom, left, right boundaries as obstacles
// (Top wall is handled by enforce_lid() instead)
// ------------------------------------------------------------------
inline void place_walls(LBMCapabilities& sys) {
    for (int x = 0; x < NX; ++x) {
        sys.obstacle[node_index(x, 0)] = true;
    }
    for (int y = 0; y < NY; ++y) {
        sys.obstacle[node_index(0, y)] = true;
        sys.obstacle[node_index(NX - 1, y)] = true;
    }
}

// ------------------------------------------------------------------
// Moving lid: enforce u = u_lid, v = 0 at y = NY-1
// Sets equilibrium distribution at top wall nodes.
// ------------------------------------------------------------------
inline void enforce_lid(LBMCapabilities& sys, double u_lid) {
    for (int x = 0; x < NX; ++x) {
        int idx = node_index(x, NY - 1);
        double* f_node = &sys.f[idx * 9];
        double rho, u, v;
        compute_macros(f_node, rho, u, v);
        for (int i = 0; i < 9; ++i) {
            f_node[i] = compute_equilibrium(i, rho, u_lid, 0.0);
        }
    }
}

// ------------------------------------------------------------------
// MRT collision (d'Humieres 2002, D2Q9)
// Forward transform: f -> moment space
// Relax moments independently
// Inverse transform: moment space -> f
// ------------------------------------------------------------------
inline void mrt_collide(double* f_node, double rho, double u, double v,
                        const MRTParams& mrt) {
    double f0 = f_node[0], f1 = f_node[1], f2 = f_node[2];
    double f3 = f_node[3], f4 = f_node[4];
    double f5 = f_node[5], f6 = f_node[6], f7 = f_node[7], f8 = f_node[8];

    // Forward transform: compute non-conserved moments
    double e   = -4*f0 - f1 - f2 - f3 - f4 + 2*(f5 + f6 + f7 + f8);
    double eps =  4*f0 - 2*(f1 + f2 + f3 + f4) + (f5 + f6 + f7 + f8);
    double qx  = -2*f1 + 2*f3 + f5 - f6 - f7 + f8;
    double qy  = -2*f2 + 2*f4 + f5 + f6 - f7 - f8;
    double pxx =  f1 - f2 + f3 - f4;
    double pxy =  f5 - f6 + f7 - f8;

    double jx = rho * u;
    double jy = rho * v;
    double usq = u*u + v*v;

    // Equilibrium moments (non-conserved)
    double e_eq   = -2*rho + 3*rho*usq;
    double eps_eq =  rho - 3*rho*usq;
    double qx_eq  = -jx;
    double qy_eq  = -jy;
    double pxx_eq = rho*(u*u - v*v);
    double pxy_eq = rho*u*v;

    // Relax non-conserved moments
    e   -= mrt.s_bulk  * (e   - e_eq);
    eps -= mrt.s_bulk  * (eps - eps_eq);
    qx  -= mrt.s_normal * (qx  - qx_eq);
    qy  -= mrt.s_normal * (qy  - qy_eq);
    pxx -= mrt.s_shear * (pxx - pxx_eq);
    pxy -= mrt.s_shear * (pxy - pxy_eq);

    // Inverse transform: construct f_i = M_inv[i] . m
    f_node[0] = (1.0/9.0)*rho + (-1.0/9.0)*e + (1.0/9.0)*eps;
    f_node[1] = (1.0/9.0)*rho + (-1.0/36.0)*e + (-1.0/18.0)*eps
              + (1.0/6.0)*jx + (-1.0/6.0)*qx + (1.0/4.0)*pxx;
    f_node[2] = (1.0/9.0)*rho + (-1.0/36.0)*e + (-1.0/18.0)*eps
              + (1.0/6.0)*jy + (-1.0/6.0)*qy + (-1.0/4.0)*pxx;
    f_node[3] = (1.0/9.0)*rho + (-1.0/36.0)*e + (-1.0/18.0)*eps
              + (-1.0/6.0)*jx + (1.0/6.0)*qx + (1.0/4.0)*pxx;
    f_node[4] = (1.0/9.0)*rho + (-1.0/36.0)*e + (-1.0/18.0)*eps
              + (-1.0/6.0)*jy + (1.0/6.0)*qy + (-1.0/4.0)*pxx;
    f_node[5] = (1.0/9.0)*rho + (1.0/18.0)*e + (1.0/36.0)*eps
              + (1.0/6.0)*jx + (1.0/12.0)*qx
              + (1.0/6.0)*jy + (1.0/12.0)*qy
              + (1.0/4.0)*pxy;
    f_node[6] = (1.0/9.0)*rho + (1.0/18.0)*e + (1.0/36.0)*eps
              + (-1.0/6.0)*jx + (-1.0/12.0)*qx
              + (1.0/6.0)*jy + (1.0/12.0)*qy
              + (-1.0/4.0)*pxy;
    f_node[7] = (1.0/9.0)*rho + (1.0/18.0)*e + (1.0/36.0)*eps
              + (-1.0/6.0)*jx + (-1.0/12.0)*qx
              + (-1.0/6.0)*jy + (-1.0/12.0)*qy
              + (1.0/4.0)*pxy;
    f_node[8] = (1.0/9.0)*rho + (1.0/18.0)*e + (1.0/36.0)*eps
              + (1.0/6.0)*jx + (1.0/12.0)*qx
              + (-1.0/6.0)*jy + (-1.0/12.0)*qy
              + (-1.0/4.0)*pxy;
}

// ------------------------------------------------------------------
// Bouzidi interpolated bounce-back for a single boundary link
// Called during streaming when fluid->solid link is detected.
// Reads post-collision sys.f, writes to sys.f_next.
// ------------------------------------------------------------------
inline void apply_bouzidi_bb(LBMCapabilities& sys, int x, int y, int i,
                              const double* f_node, int node_idx) {
    const BounceBackGeometry& geom = sys.bb_geom;
    double q = geom.compute_q(static_cast<double>(x), static_cast<double>(y), i);
    int bb = bounce_back[i];

    if (q < 0.5) {
        // Interpolate between x_f and x_f - e_i
        int src_x = x - cx[i];
        int src_y = y - cy[i];
        if (src_x >= 0 && src_x < NX && src_y >= 0 && src_y < NY) {
            double f_i_src = sys.f[node_index(src_x, src_y) * 9 + i];
            sys.f_next[node_idx * 9 + bb] = 2.0 * q * f_node[i]
                                           + (1.0 - 2.0 * q) * f_i_src;
        } else {
            sys.f_next[node_idx * 9 + bb] = f_node[i];
        }
    } else {
        // Interpolate between bounce-back and forward at x_f
        double inv2q = 1.0 / (2.0 * q);
        double f_opp = sys.f[node_idx * 9 + bb];
        sys.f_next[node_idx * 9 + bb] = inv2q * f_node[i]
                                       + (1.0 - inv2q) * f_opp;
    }
}

// ------------------------------------------------------------------
// One complete time step: collide + stream + boundaries + forces
// ------------------------------------------------------------------
inline void execute_time_step(LBMCapabilities& sys, double tau, double u_inflow) {
    int n_nodes = NX * NY;

    // --- Collision ---
    if (g_collision == CollisionType::MRT) {
        MRTParams mrt = MRTParams::from_tau(tau);
        #pragma omp parallel for collapse(2)
        for (int y = 0; y < NY; ++y) {
            for (int x = 0; x < NX; ++x) {
                int node_idx = node_index(x, y);
                if (sys.obstacle[node_idx]) continue;
                double* f_node = &sys.f[node_idx * 9];
                double rho, u, v;
                compute_macros(f_node, rho, u, v);
                mrt_collide(f_node, rho, u, v, mrt);
            }
        }
    } else {
        // BGK fallback
        #pragma omp parallel for collapse(2)
        for (int y = 0; y < NY; ++y) {
            for (int x = 0; x < NX; ++x) {
                int node_idx = node_index(x, y);
                if (sys.obstacle[node_idx]) continue;
                double* f_node = &sys.f[node_idx * 9];
                double rho, u, v;
                compute_macros(f_node, rho, u, v);
                for (int i = 0; i < 9; ++i) {
                    double feq = compute_equilibrium(i, rho, u, v);
                    f_node[i] -= (1.0 / tau) * (f_node[i] - feq);
                }
            }
        }
    }

    // --- Body force post-collision (Guo forcing term, for both MRT and BGK) ---
    if (sys.body_force_x != 0.0) {
        #pragma omp parallel for collapse(2)
        for (int y = 0; y < NY; ++y) {
            for (int x = 0; x < NX; ++x) {
                int node_idx = node_index(x, y);
                if (sys.obstacle[node_idx]) continue;
                double* f_node = &sys.f[node_idx * 9];
                double Fscale = (1.0 - 1.0 / (2.0 * tau)) * 3.0 * sys.body_force_x;
                for (int i = 0; i < 9; ++i) {
                    f_node[i] += weights[i] * cx[i] * Fscale;
                }
            }
        }
    }

    // Zero out f_next for obstacle nodes
    #pragma omp parallel for
    for (int n = 0; n < n_nodes; ++n) {
        for (int i = 0; i < 9; ++i) {
            sys.f_next[n * 9 + i] = 0.0;
        }
    }

    // --- Streaming ---
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {
            int node_idx = node_index(x, y);
            if (sys.obstacle[node_idx]) continue;

            double* f_node = &sys.f[node_idx * 9];

            for (int i = 0; i < 9; ++i) {
                int next_x = x + cx[i];
                int next_y = y + cy[i];
                bool use_interp = sys.bb_geom.is_valid();

                if (g_case == CaseType::CAVITY) {
                    // Cavity: bounce-back on walls
                    if (next_x < 0 || next_x >= NX || next_y < 0 || next_y >= NY) {
                        sys.f_next[node_idx * 9 + bounce_back[i]] = f_node[i];
                    } else {
                        int target_node = node_index(next_x, next_y);
                        if (sys.obstacle[target_node]) {
                            if (use_interp) {
                                apply_bouzidi_bb(sys, x, y, i, f_node, node_idx);
                            } else {
                                sys.f_next[node_idx * 9 + bounce_back[i]] = f_node[i];
                            }
                        } else {
                            sys.f_next[target_node * 9 + i] = f_node[i];
                        }
                    }
                } else if (g_case == CaseType::RIBS) {
                    // Ribbed channel: periodic in x and y
                    if (next_x < 0) next_x += NX;
                    if (next_x >= NX) next_x -= NX;
                    if (next_y < 0) next_y += NY;
                    if (next_y >= NY) next_y -= NY;
                    int target_node = node_index(next_x, next_y);
                    if (sys.obstacle[target_node]) {
                        sys.f_next[node_idx * 9 + bounce_back[i]] = f_node[i];
                    } else {
                        sys.f_next[target_node * 9 + i] = f_node[i];
                    }
                } else {
                    // Default: periodic in y, convective outlet at x
                    if (next_y < 0) next_y += NY;
                    if (next_y >= NY) next_y -= NY;

                    if (next_x >= 0 && next_x < NX) {
                        int target_node = node_index(next_x, next_y);
                        if (sys.obstacle[target_node]) {
                            if (use_interp) {
                                apply_bouzidi_bb(sys, x, y, i, f_node, node_idx);
                            } else {
                                sys.f_next[node_idx * 9 + bounce_back[i]] = f_node[i];
                            }
                        } else {
                            sys.f_next[target_node * 9 + i] = f_node[i];
                        }
                    }
                }
            }
        }
    }

    // Swap buffers
    sys.f.swap(sys.f_next);

    // --- Boundary conditions ---
    if (g_case == CaseType::CAVITY) {
        enforce_lid(sys, u_inflow);
    } else if (g_case == CaseType::STEP) {
        enforce_step_inflow(sys, NY / 3, NY - 1 - NY / 3, u_inflow);
        enforce_outflow(sys);
    } else if (g_case != CaseType::RIBS) {
        enforce_inflow(sys, u_inflow);
        enforce_outflow(sys);
    }
    // RIBS has periodic x/y -- no BC enforcement needed

    // --- Force extraction (all non-cavity cases) ---
    if (g_case != CaseType::CAVITY) {
        sys.reset_forces();
        for (int y = 0; y < NY; ++y) {
            for (int x = 0; x < NX; ++x) {
                int node_idx = node_index(x, y);
                if (sys.obstacle[node_idx]) continue;

                for (int i = 0; i < 9; ++i) {
                    int nx = x + cx[i];
                    int ny = y + cy[i];
                    if (g_case == CaseType::RIBS) {
                        // Periodic in x and y for ribbed channel
                        if (ny < 0) ny += NY;
                        if (ny >= NY) ny -= NY;
                        if (nx < 0) nx += NX;
                        if (nx >= NX) nx -= NX;
                    } else {
                        if (ny < 0) ny += NY;
                        if (ny >= NY) ny -= NY;
                        if (nx < 0 || nx >= NX) continue;
                    }

                    int target_idx = node_index(nx, ny);
                    if (sys.obstacle[target_idx]) {
                        double f_boundary = sys.f[node_idx * 9 + bounce_back[i]];
                        sys.fx_cyl[node_idx] += cx[i] * 2.0 * f_boundary;
                        sys.fy_cyl[node_idx] += cy[i] * 2.0 * f_boundary;
                    }
                }
            }
        }
    }
}

// ------------------------------------------------------------------
// JSON helper: write a double array with 6-digit precision
// ------------------------------------------------------------------
inline void write_json_double_array(std::ofstream& out, const double* data, int count) {
    for (int i = 0; i < count; ++i) {
        if (i > 0) out << ",";
        out.precision(6);
        out << std::fixed << data[i];
    }
}

// ------------------------------------------------------------------
// JSON frame export: velocity field, downsampled for web
// Writes output_dir/frames/frame_{step:04d}.json
// ------------------------------------------------------------------
inline void save_json_frame(const LBMCapabilities& sys, int step, const std::string& output_dir) {
    int ds = std::max(1, NX / 100);                // downsample factor
    int nx_ds = (NX + ds - 1) / ds;                // ceil division
    int ny_ds = (NY + ds - 1) / ds;

    std::string dir = output_dir + "/frames";
    std::string mkdir_cmd = "mkdir -p " + dir;
    ::system(mkdir_cmd.c_str());

    std::string filename = dir + "/frame_" + std::to_string(step) + ".json";
    std::ofstream out(filename);
    out.precision(6);
    out << std::fixed;

    // Write velocity magnitude + components
    out << "{\"nx\":" << nx_ds << ",\"ny\":" << ny_ds << ",\"velocity\":[";
    bool first = true;
    for (int y = 0; y < NY; y += ds) {
        for (int x = 0; x < NX; x += ds) {
            int idx = node_index(x, y);
            if (sys.obstacle[idx]) {
                if (!first) out << ",";
                out << "0";
                first = false;
                continue;
            }
            double rho, u, v;
            compute_macros(&sys.f[idx * 9], rho, u, v);
            if (rho < 1e-12) { u = 0.0; v = 0.0; }
            double vel = std::sqrt(u * u + v * v);
            if (!first) out << ",";
            out << vel;
            first = false;
        }
    }

    out << "],\"u\":[";
    first = true;
    for (int y = 0; y < NY; y += ds) {
        for (int x = 0; x < NX; x += ds) {
            int idx = node_index(x, y);
            if (sys.obstacle[idx]) {
                if (!first) out << ",";
                out << "0";
                first = false;
                continue;
            }
            double rho, u, v;
            compute_macros(&sys.f[idx * 9], rho, u, v);
            if (rho < 1e-12) { u = 0.0; v = 0.0; }
            if (!first) out << ",";
            out << u;
            first = false;
        }
    }

    out << "],\"v\":[";
    first = true;
    for (int y = 0; y < NY; y += ds) {
        for (int x = 0; x < NX; x += ds) {
            int idx = node_index(x, y);
            if (sys.obstacle[idx]) {
                if (!first) out << ",";
                out << "0";
                first = false;
                continue;
            }
            double rho, u, v;
            compute_macros(&sys.f[idx * 9], rho, u, v);
            if (rho < 1e-12) { u = 0.0; v = 0.0; }
            if (!first) out << ",";
            out << v;
            first = false;
        }
    }

    out << "],\"rho\":[";
    first = true;
    for (int y = 0; y < NY; y += ds) {
        for (int x = 0; x < NX; x += ds) {
            int idx = node_index(x, y);
            if (sys.obstacle[idx]) {
                if (!first) out << ",";
                out << "0";
                first = false;
                continue;
            }
            double rho, u, v;
            compute_macros(&sys.f[idx * 9], rho, u, v);
            if (!first) out << ",";
            out << rho;
            first = false;
        }
    }

    out << "],\"obstacle\":[";
    first = true;
    for (int y = 0; y < NY; y += ds) {
        for (int x = 0; x < NX; x += ds) {
            int idx = node_index(x, y);
            if (!first) out << ",";
            out << (sys.obstacle[idx] ? 1 : 0);
            first = false;
        }
    }

    out << "]}";
    out.close();
}

// ------------------------------------------------------------------
// Forces JSONL export: append one line per step
// ------------------------------------------------------------------
inline void save_forces_jsonl(const std::string& output_dir, int step, double cd, double cl) {
    std::string filename = output_dir + "/forces.jsonl";
    std::ofstream out(filename, std::ios::app);
    out.precision(6);
    out << std::fixed;
    out << "{\"step\":" << step << ",\"cd\":" << cd << ",\"cl\":" << cl << "}\n";
}

// ------------------------------------------------------------------
// Metadata JSON export: simulation parameters
// ------------------------------------------------------------------
inline void save_meta_json(const std::string& output_dir, double re, double tau,
                            double u_inflow, double length_scale, const std::string& shape_type,
                            int nx, int ny) {
    std::string filename = output_dir + "/meta.json";
    std::ofstream out(filename);
    out.precision(6);
    out << std::fixed;
    out << "{\n";
    out << "  \"nx\": " << nx << ",\n";
    out << "  \"ny\": " << ny << ",\n";
    out << "  \"re\": " << re << ",\n";
    out << "  \"tau\": " << tau << ",\n";
    out << "  \"u_inflow\": " << u_inflow << ",\n";
    out << "  \"length_scale\": " << length_scale << ",\n";
    out << "  \"shape_type\": \"" << shape_type << "\"\n";
    out << "}\n";
}

// ------------------------------------------------------------------
// VTK export (structured points, ASCII) -- legacy, use --vtk flag
// ------------------------------------------------------------------
inline void save_vtk_frame(const LBMCapabilities& sys, int frame, const std::string& output_dir = "output") {
    std::string filename = output_dir + "/frame_" + std::to_string(frame) + ".vtk";
    std::ofstream out(filename);

    out << "# vtk DataFile Version 3.0\nLBM Fluid Grid\nASCII\nDATASET STRUCTURED_POINTS\n";
    out << "DIMENSIONS " << NX << " " << NY << " 1\n";
    out << "ORIGIN 0 0 0\nSPACING 1 1 1\n";
    out << "POINT_DATA " << NX * NY << "\n";

    // Velocity magnitude
    out << "SCALARS VelocityMagnitude double 1\nLOOKUP_TABLE default\n";
    for (int y = 0; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {
            int idx = node_index(x, y);
            if (sys.obstacle[idx]) {
                out << "0.0\n";
            } else {
                double rho, u, v;
                compute_macros(&sys.f[idx * 9], rho, u, v);
                out << std::sqrt(u * u + v * v) << "\n";
            }
        }
    }

    // Velocity vector field
    out << "VECTORS Velocity double\n";
    for (int y = 0; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {
            int idx = node_index(x, y);
            if (sys.obstacle[idx]) {
                out << "0 0 0\n";
            } else {
                double rho, u, v;
                compute_macros(&sys.f[idx * 9], rho, u, v);
                out << u << " " << v << " 0\n";
            }
        }
    }

    // Density field
    out << "SCALARS Density double 1\nLOOKUP_TABLE default\n";
    for (int y = 0; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {
            int idx = node_index(x, y);
            if (sys.obstacle[idx]) {
                out << "0.0\n";
            } else {
                double rho, u, v;
                compute_macros(&sys.f[idx * 9], rho, u, v);
                out << rho << "\n";
            }
        }
    }

    // Drag coefficient field (obstacle boundary nodes)
    out << "SCALARS DragForce double 1\nLOOKUP_TABLE default\n";
    for (int y = 0; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {
            int idx = node_index(x, y);
            if (sys.fx_cyl[idx] != 0.0 || sys.fy_cyl[idx] != 0.0) {
                out << sys.fx_cyl[idx] << "\n";
            } else {
                out << "0.0\n";
            }
        }
    }
}

// ------------------------------------------------------------------
// Place cylinder obstacle in the domain
// ------------------------------------------------------------------
inline void place_cylinder(LBMCapabilities& sys, int cx_cyl, int cy_cyl, int radius) {
    sys.bb_geom.cx = static_cast<double>(cx_cyl);
    sys.bb_geom.cy = static_cast<double>(cy_cyl);
    sys.bb_geom.radius = static_cast<double>(radius);
    for (int y = 0; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {
            double dx = static_cast<double>(x - cx_cyl);
            double dy = static_cast<double>(y - cy_cyl);
            if (std::sqrt(dx * dx + dy * dy) < radius) {
                sys.obstacle[node_index(x, y)] = true;
            }
        }
    }
}

// ------------------------------------------------------------------
// Place arbitrary polygon obstacle in the domain
// poly: closed polygon vertices in grid coordinates
// Also sets bb_geom for interpolated bounce-back
// ------------------------------------------------------------------
inline void place_polygon(LBMCapabilities& sys,
    const std::vector<std::pair<double,double>>& poly)
{
    sys.bb_geom.poly_vertices = poly;
    sys.bb_geom.is_polygon = true;
    for (int y = 0; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {
            if (point_in_polygon(static_cast<double>(x),
                                 static_cast<double>(y), poly)) {
                sys.obstacle[node_index(x, y)] = true;
            }
        }
    }
}
