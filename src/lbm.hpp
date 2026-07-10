#pragma once
#include "lbm_types.hpp"
#include "geometry.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <cmath>

// ==========================================================================
// D2Q9 LATTICE BOLTZMANN METHOD -- Core solver
// ==========================================================================

// ------------------------------------------------------------------
// Zou/He velocity inlet: enforce u = u_inflow, v = 0 at x = 0 (cylinder case)
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
// Convective outlet: zero-gradient at x = NX-1 (cylinder case)
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
    // Bottom wall (y = 0)
    for (int x = 0; x < NX; ++x) {
        sys.obstacle[node_index(x, 0)] = true;
    }
    // Left and right walls
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
// One complete time step: collide + stream + boundaries + forces
// ------------------------------------------------------------------
inline void execute_time_step(LBMCapabilities& sys, double tau, double u_inflow) {
    int n_nodes = NX * NY;

    // --- Collision ---
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

                if (g_case == CaseType::CAVITY) {
                    // --- Cavity: bounce-back on walls ---
                    // Check if streaming would leave the domain
                    if (next_x < 0 || next_x >= NX || next_y < 0 || next_y >= NY) {
                        // Out of bounds -- bounce-back
                        sys.f_next[node_idx * 9 + bounce_back[i]] = f_node[i];
                    } else {
                        int target_node = node_index(next_x, next_y);
                        if (sys.obstacle[target_node]) {
                            sys.f_next[node_idx * 9 + bounce_back[i]] = f_node[i];
                        } else {
                            sys.f_next[target_node * 9 + i] = f_node[i];
                        }
                    }
                } else {
                    // --- Cylinder: periodic in y ---
                    if (next_y < 0) next_y += NY;
                    if (next_y >= NY) next_y -= NY;

                    if (next_x >= 0 && next_x < NX) {
                        int target_node = node_index(next_x, next_y);
                        if (sys.obstacle[target_node]) {
                            sys.f_next[node_idx * 9 + bounce_back[i]] = f_node[i];
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
    if (g_case == CaseType::CYLINDER) {
        enforce_inflow(sys, u_inflow);
        enforce_outflow(sys);
    } else {
        enforce_lid(sys, u_inflow);
    }

    // --- Force extraction (cylinder case only) ---
    if (g_case == CaseType::CYLINDER) {
        sys.reset_forces();
        for (int y = 0; y < NY; ++y) {
            for (int x = 0; x < NX; ++x) {
                int node_idx = node_index(x, y);
                if (sys.obstacle[node_idx]) continue;

                for (int i = 0; i < 9; ++i) {
                    int nx = x + cx[i];
                    int ny = y + cy[i];
                    if (ny < 0) ny += NY;
                    if (ny >= NY) ny -= NY;
                    if (nx < 0 || nx >= NX) continue;

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
// VTK export (structured points, ASCII)
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

    // Drag coefficient field (cylinder boundary nodes)
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
// ------------------------------------------------------------------
inline void place_polygon(LBMCapabilities& sys,
    const std::vector<std::pair<double,double>>& poly)
{
    for (int y = 0; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {
            if (point_in_polygon(static_cast<double>(x),
                                 static_cast<double>(y), poly)) {
                sys.obstacle[node_index(x, y)] = true;
            }
        }
    }
}
