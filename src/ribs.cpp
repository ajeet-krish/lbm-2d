#include "lbm.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include <cstdlib>

// ==========================================================================
// LBM-2D: Ribbed Channel Flow Entry Point
// ==========================================================================
// Usage:
//   ./build/LBM_Ribs                          (default: Re_H=100, 50000 steps)
//   ./build/LBM_Ribs 200                      (Re_H=200)
//   ./build/LBM_Ribs 100 80000                (Re_H=100, 80000 steps)
//
// Periodic in x and y, driven by body force.
// Validation: friction factor vs Webb 1971 (qualitative for laminar Re).
//
// Re_H = u_bulk * D_h / nu
//   u_bulk = (2/3) * u_max
//   D_h = 2 * NY
// ==========================================================================

struct RibParams {
    double tau;
    double u_max;
    double body_force_accel;  // acceleration (F/rho)
    int num_steps;
    int save_interval;
    int h_rib;
    int pitch;
    int width;
};

RibParams compute_params(double Re_H, int steps = -1) {
    double u_max = 0.1;
    double u_bulk = (2.0 / 3.0) * u_max;
    double length_scale = 2.0 * NY;
    double nu = u_bulk * length_scale / Re_H;
    double tau = 0.5 + 3.0 * nu;

    int h_rib = NY / 20;
    if (h_rib < 2) h_rib = 2;
    int pitch = 10 * h_rib;
    int width = h_rib;

    // Acceleration needed for Poiseuille: G = 8 * nu * u_max / H^2
    double G = 8.0 * nu * u_max / static_cast<double>(NY * NY);

    int num_steps = (steps > 0) ? steps : std::max(30000, static_cast<int>(20.0 * NX / u_bulk));
    int save_interval = num_steps / 50;

    return {tau, u_max, G, num_steps, save_interval, h_rib, pitch, width};
}

int main(int argc, char* argv[]) {
    std::cout << "==============================================" << std::endl;
    std::cout << " LBM-2D: Ribbed Channel Flow" << std::endl;
    std::cout << " D2Q9 | MRT | OpenMP | Cache-Optimized" << std::endl;
    std::cout << "==============================================" << std::endl;

    double Re = 100.0;
    int steps = -1;
    bool save_vtk = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--vtk") {
            save_vtk = true;
        } else if (arg != "--vtk") {
            if (i == 1) Re = std::stod(arg);
            else if (i == 2) steps = std::stoi(arg);
        }
    }

    g_case = CaseType::RIBS;

    // Override grid for ribbed channel
    NX = 400;
    NY = 200;

    auto params = compute_params(Re, steps);

    std::cout << "Re_H = " << Re
              << "  tau = " << params.tau
              << "  steps = " << params.num_steps
              << "  u_max = " << params.u_max
              << "  h_rib = " << params.h_rib
              << "  pitch = " << params.pitch
              << "  body_force = " << params.body_force_accel
              << std::endl;

    LBMCapabilities system;
    system.body_force_x = params.body_force_accel;

    // Place obstacles: bottom wall, top wall, ribs
    for (int y = 0; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {
            // Bottom wall
            if (y == 0) {
                system.obstacle[node_index(x, y)] = true;
            }
            // Top wall
            if (y == NY - 1) {
                system.obstacle[node_index(x, y)] = true;
            }
            // Ribs at x = pitch, 2*pitch, ...
            for (int r = 1; r * params.pitch < NX; ++r) {
                int rx = r * params.pitch;
                if (x >= rx && x < rx + params.width && y < params.h_rib) {
                    system.obstacle[node_index(x, y)] = true;
                }
            }
        }
    }

    // Initialize with equilibrium at rest
    for (int n = 0; n < NX * NY; ++n) {
        double* f_node = &system.f[n * 9];
        for (int i = 0; i < 9; ++i) {
            f_node[i] = compute_equilibrium(i, 1.0, 0.0, 0.0);
        }
    }

    // Output directory
    std::string subdir = "output/ribs_re" + std::to_string(static_cast<int>(Re));
    std::string mkdir_cmd = "mkdir -p " + subdir + "/frames";
    ::system(mkdir_cmd.c_str());

    save_meta_json(subdir, Re, params.tau, params.u_max,
                   static_cast<double>(2 * NY), "ribbed-channel", NX, NY);

    for (int step = 0; step <= params.num_steps; ++step) {
        execute_time_step(system, params.tau, params.u_max);

        // Force extraction on all obstacles
        double fx_total = 0.0, fy_total = 0.0;
        for (int n = 0; n < NX * NY; ++n) {
            fx_total += system.fx_cyl[n];
            fy_total += system.fy_cyl[n];
        }

        save_forces_jsonl(subdir, step, fx_total, fy_total);

        if (step % params.save_interval == 0) {
            save_json_frame(system, step, subdir);
            if (save_vtk) {
                save_vtk_frame(system, step, subdir);
            }
        }

        if (step % 2000 == 0) {
            // Compute bulk velocity
            double u_sum = 0.0;
            int count = 0;
            for (int y = 1; y < NY - 1; ++y) {
                for (int x = 0; x < NX; ++x) {
                    int idx = node_index(x, y);
                    if (system.obstacle[idx]) continue;
                    double rho, u, v;
                    compute_macros(&system.f[idx * 9], rho, u, v);
                    u_sum += u;
                    count++;
                }
            }
            double u_bulk = u_sum / count;
            std::cout << "  step " << std::setw(6) << step
                      << "  u_bulk = " << std::fixed << std::setprecision(6) << u_bulk
                      << std::endl;
        }
    }

    // Compute bulk velocity at final state
    double u_sum = 0.0;
    int fluid_count = 0;
    for (int y = 1; y < NY - 1; ++y) {
        for (int x = 0; x < NX; ++x) {
            int idx = node_index(x, y);
            if (system.obstacle[idx]) continue;
            double rho, u, v;
            compute_macros(&system.f[idx * 9], rho, u, v);
            u_sum += u;
            fluid_count++;
        }
    }
    double u_bulk = u_sum / fluid_count;

    // Friction factor: f = 2 * D_h * tau_wall / (rho * u_bulk^2)
    // tau_wall from force balance: tau_wall * 2*NX = body_force * rho * NX * NY
    // f = 2 * (2*NY) * (body_force * rho * NY / 2) / (rho * u_bulk^2)
    // Simplifies: f = 2 * body_force * NY * NY / (u_bulk^2)
    double body_force = params.body_force_accel;
    double friction_factor = 2.0 * body_force * static_cast<double>(NY * NY)
                             / (u_bulk * u_bulk);

    // Blasius laminar: f_smooth = 64 / Re_H
    double f_smooth = 64.0 / Re;

    // Compute reattachment length Xr/h between ribs
    // Scan the first inter-rib region for u = 0 crossing near wall
    double h_rib_d = static_cast<double>(params.h_rib);
    double xr_h = -1.0;
    int rib_start = params.pitch;
    int rib_end = params.pitch + params.width;
    for (int scan_y = 1; scan_y <= params.h_rib * 2; ++scan_y) {
        for (int x = rib_end + 1; x < 2 * params.pitch; ++x) {
            int idx = node_index(x, scan_y);
            if (system.obstacle[idx]) continue;
            double rho, u, v;
            compute_macros(&system.f[idx * 9], rho, u, v);
            if (u > 1e-6) {
                xr_h = static_cast<double>(x - rib_end) / h_rib_d;
                break;
            }
        }
        if (xr_h > 0.0) break;
    }

    // Update meta.json with friction and reattachment
    {
        std::string fname = subdir + "/meta.json";
        std::ofstream out(fname, std::ios::app);
        out.seekp(-2, std::ios::end);  // before closing }
        out << ",\n  \"u_bulk\": " << u_bulk
            << ",\n  \"friction_factor\": " << friction_factor
            << ",\n  \"f_smooth\": " << f_smooth
            << ",\n  \"xr_h\": " << (xr_h > 0.0 ? xr_h : -1.0)
            << ",\n  \"f_ratio\": " << (f_smooth > 0.0 ? friction_factor / f_smooth : -1.0)
            << "\n}\n";
    }

    std::cout << "==============================================" << std::endl;
    std::cout << " Ribbed channel simulation complete." << std::endl;
    std::cout << "  Re_H = " << Re << std::endl;
    std::cout << "  u_bulk = " << std::fixed << std::setprecision(6) << u_bulk << std::endl;
    std::cout << "  Friction factor f = " << std::setprecision(4) << friction_factor << std::endl;
    std::cout << "  Smooth channel f_smooth = " << f_smooth << std::endl;
    std::cout << "  f / f_smooth = " << (f_smooth > 0.0 ? friction_factor / f_smooth : -1.0) << std::endl;
    std::cout << "  Xr/h = " << (xr_h > 0.0 ? xr_h : -1.0) << std::endl;
    std::cout << "  Steps = " << params.num_steps << std::endl;
    std::cout << "==============================================" << std::endl;

    return 0;
}
