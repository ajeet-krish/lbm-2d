#include "lbm.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include <filesystem>

// ==========================================================================
// LBM-2D: Building Downwash Flow
// ==========================================================================
// Usage:
//   ./build/LBM_Downwash                          (default: Re_H=100)
//   ./build/LBM_Downwash 200                      (Re_H=200)
//
// Tall building upstream, low-rise downstream.
// Buildings scaled up: h_tall=120, h_low=45, w_bldg=45, gap=45.
// Maintains height ratio ~2.67 (120/45).
// Re_H = u_ref * H_tall / nu
// Validation: Cp distribution vs Hunt 1984 (qualitative)
// ==========================================================================

struct DownwashParams {
    double tau;
    double u_ref;
    int num_steps;
    int save_interval;
    int h_tall;
    int h_low;
    int w_bldg;
    int gap;
    int tall_x0;
    int low_x0;
};

DownwashParams compute_params(double Re_H, int steps = -1) {
    int h_tall = 120;
    int h_low  = 45;
    int w_bldg = 45;
    int gap = h_low;  // gap = low-rise height
    double u_ref = 0.1;
    double length_scale = static_cast<double>(h_tall);
    double nu = u_ref * length_scale / Re_H;
    double tau = 0.5 + 3.0 * nu;

    int tall_x0 = NX / 4 - w_bldg / 2;
    int low_x0  = tall_x0 + w_bldg + gap;

    int num_steps = (steps > 0) ? steps
        : std::max(20000, static_cast<int>(20.0 * NX / u_ref));
    int save_interval = num_steps / 50;

    return {tau, u_ref, num_steps, save_interval,
            h_tall, h_low, w_bldg, gap, tall_x0, low_x0};
}

int main(int argc, char* argv[]) {
    std::cout << "==============================================" << std::endl;
    std::cout << " LBM-2D: Building Downwash" << std::endl;
    std::cout << " D2Q9 | MRT | OpenMP | Cache-Optimized" << std::endl;
    std::cout << "==============================================" << std::endl;

    double Re = 100.0;
    int steps = -1;
    bool save_vtk = false;

    int positional_idx = 1;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--vtk") { save_vtk = true;
        } else if (arg == "--use-les") { g_use_les = true;
        } else if (arg == "--cs" && i + 1 < argc) { g_cs = std::stod(argv[++i]);
        } else if (arg.find("--") != 0) {
            if (positional_idx == 1) Re = std::stod(arg);
            else if (positional_idx == 2) steps = std::stoi(arg);
            ++positional_idx;
        }
    }

    g_case = CaseType::DOWNWASH;

    auto params = compute_params(Re, steps);
    LBMCapabilities system;

    for (int y = 0; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {
            if (y == 0) system.obstacle[node_index(x, y)] = true;
            if (y == NY - 1) system.obstacle[node_index(x, y)] = true;
            // Tall building (upstream)
            if (x >= params.tall_x0 && x < params.tall_x0 + params.w_bldg
                && y < params.h_tall)
                system.obstacle[node_index(x, y)] = true;
            // Low-rise building (downstream)
            if (x >= params.low_x0 && x < params.low_x0 + params.w_bldg
                && y < params.h_low)
                system.obstacle[node_index(x, y)] = true;
        }
    }

    for (int n = 0; n < NX * NY; ++n) {
        double* f_node = &system.f[n * 9];
        for (int i = 0; i < 9; ++i) {
            f_node[i] = compute_equilibrium(i, 1.0, params.u_ref, 0.0);
        }
    }

    std::string subdir = "output/urban/downwash_re" + std::to_string(static_cast<int>(Re));
    std::filesystem::create_directories(subdir + "/frames");

    save_meta_json(subdir, Re, params.tau, params.u_ref,
                   static_cast<double>(params.h_tall), "building-downwash", NX, NY);

    std::cout << "Re_H = " << Re
              << "  tau = " << params.tau
              << "  steps = " << params.num_steps
              << "  h_tall = " << params.h_tall
              << "  h_low = " << params.h_low
              << "  w_bldg = " << params.w_bldg
              << "  gap = " << params.gap
              << (g_use_les ? "  LES(Cs=" + std::to_string(g_cs) + ")" : "")
              << std::endl;

    for (int step = 0; step <= params.num_steps; ++step) {
        execute_time_step(system, params.tau, params.u_ref);

        double fx_total = 0.0, fy_total = 0.0;
        for (int n = 0; n < NX * NY; ++n) {
            fx_total += system.fx_body[n];
            fy_total += system.fy_body[n];
        }

        save_forces_jsonl(subdir, step, fx_total, fy_total);

        if (step % params.save_interval == 0) {
            save_json_frame(system, step, subdir);
            if (save_vtk) save_vtk_frame(system, step, subdir);
        }

        if (step % 2000 == 0) {
            std::cout << "  step " << std::setw(6) << step
                      << "  Fx = " << std::fixed << std::setprecision(4) << fx_total
                      << std::endl;
        }
    }

    std::cout << "==============================================" << std::endl;
    std::cout << " Downwash simulation complete." << std::endl;
    std::cout << "  Re_H = " << Re << std::endl;
    std::cout << "  h_tall = " << params.h_tall << "  h_low = " << params.h_low << std::endl;
    std::cout << "  Steps = " << params.num_steps << std::endl;
    std::cout << "==============================================" << std::endl;

    return 0;
}
