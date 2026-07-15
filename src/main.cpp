#include "lbm.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <string>
#include <random>
#include <fstream>

// ==========================================================================
// LBM-2D Solver Entry Point -- Cylinder Flow
// ==========================================================================
// Usage:
//   ./build/LBM_Engine                      (default: Re=100, 30000 steps)
//   ./build/LBM_Engine 200                  (Re=200)
//   ./build/LBM_Engine 100 20000            (Re=100, 20000 steps)
//   ./build/LBM_Engine 100 20000 --vtk      (include legacy VTK output)
//
// Reynolds: Re = u_inflow * D / nu,  D = 2 * radius = NY/5 = 30
//   nu = (tau - 0.5) / 3
//   tau = 0.5 + 3 * u_inflow * D / Re
// ==========================================================================

struct SimParams {
    double tau;
    double u_inflow;
    int num_steps;
    int save_interval;
    double length_scale;
};

SimParams compute_params(double Re, int steps = -1) {
    double u_inflow = 0.1;
    double length_scale = static_cast<double>(NY) / 5.0;  // D = 2 * NY/10 = NY/5
    double nu = u_inflow * length_scale / Re;
    double tau = 0.5 + 3.0 * nu;

    int num_steps = (steps > 0) ? steps
        : std::max(4000, static_cast<int>(10.0 * NX / u_inflow));
    int save_interval = num_steps / 50;

    return {tau, u_inflow, num_steps, save_interval, length_scale};
}

struct ForceHistory {
    std::vector<double> t;
    std::vector<double> cd;
    std::vector<double> cl;
};

ForceHistory run_simulation(double Re, int steps, bool save_vtk) {
    auto params = compute_params(Re, steps);
    LBMCapabilities system;

    int cx_cyl = NX / 4;
    int cy_cyl = NY / 2 + 1;
    int radius = NY / 10;
    place_cylinder(system, cx_cyl, cy_cyl, radius);

    // Initialize with equilibrium
    for (int n = 0; n < NX * NY; ++n) {
        double* f_node = &system.f[n * 9];
        for (int i = 0; i < 9; ++i) {
            f_node[i] = compute_equilibrium(i, 1.0, params.u_inflow, 0.0);
        }
    }

    // Perturbation to trigger shedding
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> pert_dist(-1e-4, 1e-4);
    for (int x = cx_cyl + 5; x < std::min(NX, cx_cyl + 60); ++x) {
        for (int y = 0; y < NY; ++y) {
            int n = node_index(x, y);
            if (system.obstacle[n]) continue;
            double* f_node = &system.f[n * 9];
            double v_pert = pert_dist(rng);
            double rho, u, v;
            compute_macros(f_node, rho, u, v);
            for (int i = 0; i < 9; ++i) {
                f_node[i] = compute_equilibrium(i, rho, u, v + v_pert);
            }
        }
    }

    // Output directory
    std::string subdir = "output/cylinder/re" + std::to_string(static_cast<int>(Re));
    std::filesystem::create_directories(subdir + "/frames");

    // Write metadata
    save_meta_json(subdir, Re, params.tau, params.u_inflow,
                   params.length_scale, "cylinder", NX, NY);

    ForceHistory history;

    std::cout << "Re = " << Re
              << "  tau = " << params.tau
              << "  steps = " << params.num_steps
              << "  u_in = " << params.u_inflow
              << "  collision = " << (g_collision == CollisionType::MRT ? "MRT" : "BGK")
              << (g_use_les ? "  LES(Cs=" + std::to_string(g_cs) + ")" : "")
              << std::endl;

    for (int step = 0; step <= params.num_steps; ++step) {
        execute_time_step(system, params.tau, params.u_inflow);

        double fx_total = 0.0, fy_total = 0.0;
        for (int n = 0; n < NX * NY; ++n) {
            fx_total += system.fx_body[n];
            fy_total += system.fy_body[n];
        }

        double cd = 2.0 * fx_total / (params.length_scale
                    * params.u_inflow * params.u_inflow);
        double cl = 2.0 * fy_total / (params.length_scale
                    * params.u_inflow * params.u_inflow);

        history.t.push_back(static_cast<double>(step));
        history.cd.push_back(cd);
        history.cl.push_back(cl);

        // Save force history (every step)
        save_forces_jsonl(subdir, step, cd, cl);

        // Save frames at intervals
        if (step % params.save_interval == 0) {
            save_json_frame(system, step, subdir);
            if (save_vtk) {
                save_vtk_frame(system, step, subdir);
            }
        }

        if (step % 500 == 0) {
            std::cout << "  step " << std::setw(6) << step
                      << "  Cd = " << std::fixed << std::setprecision(4) << cd
                      << "  Cl = " << std::fixed << std::setprecision(4) << cl
                      << std::endl;
        }
    }

    return history;
}

int main(int argc, char* argv[]) {
    std::cout << "==============================================" << std::endl;
    std::cout << " LBM-2D: High-Performance Lattice Boltzmann CFD" << std::endl;
    std::cout << " D2Q9 | MRT | OpenMP | Cache-Optimized" << std::endl;
    std::cout << "==============================================" << std::endl;

    double Re = 100.0;
    int steps = -1;
    bool save_vtk = false;
    int positional_idx = 1;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--vtk") {
            save_vtk = true;
        } else if (arg == "--use-les") {
            g_use_les = true;
        } else if (arg == "--cs" && i + 1 < argc) {
            g_cs = std::stod(argv[++i]);
        } else if (arg == "--nx" && i + 1 < argc) {
            NX = std::stoi(argv[++i]);
        } else if (arg == "--ny" && i + 1 < argc) {
            NY = std::stoi(argv[++i]);
        } else if (arg.find("--") != 0) {
            if (positional_idx == 1) Re = std::stod(arg);
            else if (positional_idx == 2) steps = std::stoi(arg);
            ++positional_idx;
        }
    }

    // Auto-LES: enable Smagorinsky when tau < 0.55 (high Re stability)
    {
        double nu = 0.1 * (NY / 5.0) / Re;
        double tau_check = 0.5 + 3.0 * nu;
        if (tau_check < 0.55 && !g_use_les) {
            g_use_les = true;
            std::cout << "Auto-LES: tau=" << tau_check << " < 0.55, enabling Smagorinsky LES (Cs="
                      << g_cs << ")" << std::endl;
        }
    }

    auto history = run_simulation(Re, steps, save_vtk);

    // Statistics from latter half
    double cd_sum = 0.0, cl_max = 0.0, cl_min = 0.0;
    int n_late = history.cd.size() * 0.5;
    for (int i = n_late; i < static_cast<int>(history.cd.size()); ++i) {
        cd_sum += history.cd[i];
        if (history.cl[i] > cl_max) cl_max = history.cl[i];
        if (history.cl[i] < cl_min) cl_min = history.cl[i];
    }
    double cd_mean = cd_sum / (history.cd.size() - n_late);
    double cl_amplitude = (cl_max - cl_min) / 2.0;

    std::cout << "==============================================" << std::endl;
    std::cout << " RESULTS" << std::endl;
    std::cout << "  Mean Cd = " << cd_mean << std::endl;
    std::cout << "  Cl amplitude = " << cl_amplitude << std::endl;
    std::cout << "  Steps completed = " << history.t.size() << std::endl;
    std::cout << "==============================================" << std::endl;

    return 0;
}
