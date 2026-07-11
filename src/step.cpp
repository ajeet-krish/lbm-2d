#include "lbm.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include <cstdlib>

// ==========================================================================
// LBM-2D Backward-Facing Step Entry Point
// ==========================================================================
// Usage:
//   ./build/LBM_Step                          (default: Re_H=100, 30000 steps)
//   ./build/LBM_Step 200                      (Re_H=200)
//   ./build/LBM_Step 100 40000                (Re_H=100, 40000 steps)
//
// Validation: Xr/H reattachment length vs Armaly et al. 1983
//   Re_H=100  -> Xr/H ~ 3
//   Re_H=200  -> Xr/H ~ 6
//   Re_H=400  -> Xr/H ~ 8-10
//
// Re_H = u_mean * D_h / nu
//   u_mean = (2/3) * u_max (parabolic profile)
//   D_h = 2 * H_inlet     (step hydraulic diameter)
//   tau = 0.5 + 3 * nu
// ==========================================================================

struct StepParams {
    double tau;
    double u_max;
    int num_steps;
    int save_interval;
    double length_scale;
    int h_step;
    int h_inlet;
};

StepParams compute_params(double Re_H, int steps = -1) {
    int h_step = NY / 3;                     // step height
    int h_inlet = NY - 1 - h_step;           // inlet height (fluid portion)
    double u_max = 0.1;                      // parabolic peak
    double u_mean = (2.0 / 3.0) * u_max;     // mean velocity for parabola
    double length_scale = 2.0 * h_inlet;     // hydraulic diameter
    double nu = u_mean * length_scale / Re_H;
    double tau = 0.5 + 3.0 * nu;

    int num_steps = (steps > 0) ? steps : std::max(10000, static_cast<int>(15.0 * NX / u_mean));
    int save_interval = num_steps / 50;

    return {tau, u_max, num_steps, save_interval, length_scale, h_step, h_inlet};
}

int main(int argc, char* argv[]) {
    std::cout << "==============================================" << std::endl;
    std::cout << " LBM-2D: Backward-Facing Step" << std::endl;
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
            if (i == 1 || (i == 2 && argc > 2 && std::string(argv[1]).find("--") != 0)) {
                if (i == 1) Re = std::stod(arg);
                else if (i == 2) steps = std::stoi(arg);
            }
        }
    }

    // Set globals
    g_case = CaseType::STEP;


    auto params = compute_params(Re, steps);
    LBMCapabilities system;

    // Place step obstacle
    // y from 0 to h_step-1 is solid for x < NX/4 (the step itself)
    // y = 0 is solid everywhere (bottom wall)
    // y = NY-1 is solid (top wall)
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
            // Step: solid for y < h_step, x < NX/4
            if (x < NX / 4 && y < params.h_step) {
                system.obstacle[node_index(x, y)] = true;
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
    std::string subdir = "output/step_re" + std::to_string(static_cast<int>(Re));
    std::string mkdir_cmd = "mkdir -p " + subdir + "/frames";
    ::system(mkdir_cmd.c_str());

    // Write metadata
    save_meta_json(subdir, Re, params.tau, params.u_max,
                   params.length_scale, "backward-facing-step", NX, NY);

    std::cout << "Re_H = " << Re
              << "  tau = " << params.tau
              << "  steps = " << params.num_steps
              << "  u_max = " << params.u_max
              << "  h_step = " << params.h_step
              << "  h_inlet = " << params.h_inlet
              << "  D_h = " << params.length_scale
              << "  collision = " << (g_collision == CollisionType::MRT ? "MRT" : "BGK")
              << std::endl;

    for (int step = 0; step <= params.num_steps; ++step) {
        execute_time_step(system, params.tau, params.u_max);

        // Save force history (drag on all obstacles = step + walls)
        double fx_total = 0.0, fy_total = 0.0;
        for (int n = 0; n < NX * NY; ++n) {
            fx_total += system.fx_cyl[n];
            fy_total += system.fy_cyl[n];
        }

        save_forces_jsonl(subdir, step, fx_total, fy_total);

        // Save frames at intervals
        if (step % params.save_interval == 0) {
            save_json_frame(system, step, subdir);
            if (save_vtk) {
                save_vtk_frame(system, step, subdir);
            }
        }

        if (step % 1000 == 0) {
            std::cout << "  step " << std::setw(6) << step
                      << std::endl;
        }
    }

    // Compute reattachment length Xr/H
    int step_x0 = NX / 4;
    double H = static_cast<double>(params.h_step);
    int reattach_x = -1;
    for (int scan_y = 1; scan_y <= 10; ++scan_y) {
        for (int x = step_x0 + 1; x < NX; ++x) {
            int idx = node_index(x, scan_y);
            if (system.obstacle[idx]) continue;
            double rho, u, v;
            compute_macros(&system.f[idx * 9], rho, u, v);
            if (u > 1.0e-6) {
                reattach_x = x;
                break;
            }
        }
        if (reattach_x > 0) break;
    }

    double XrH = (reattach_x > 0)
        ? static_cast<double>(reattach_x - step_x0) / H
        : -1.0;

    std::cout << "==============================================" << std::endl;
    std::cout << " Step simulation complete." << std::endl;
    std::cout << "  Re_H = " << Re << std::endl;
    std::cout << "  Xr/H = " << std::fixed << std::setprecision(2) << XrH << std::endl;
    std::cout << "  Armaly 1983: Xr/H ~ 3 at Re=100, ~6 at Re=200, ~9 at Re=400" << std::endl;
    std::cout << "  Steps = " << params.num_steps << std::endl;
    std::cout << "==============================================" << std::endl;

    return 0;
}
