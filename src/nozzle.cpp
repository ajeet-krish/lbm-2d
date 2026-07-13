#include "lbm.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include <filesystem>

// ==========================================================================
// LBM-2D: Converging-Diverging Nozzle
// ==========================================================================
// Usage:
//   ./build/LBM_Nozzle 100 0.25               (Re=100, area ratio=0.25)
//   ./build/LBM_Nozzle 500 0.5                (Re=500, area ratio=0.5)
//   ./build/LBM_Nozzle 1000 0.25 --use-les    (Re=1000 with LES)
//
// Smooth cosine-profile converging-diverging nozzle.
// Validation: Bernoulli equation for subsonic inviscid flow.
//   v(x) = Q / A(x)
//   p(x) = p0 - 0.5 * rho * v(x)^2
//
// Parameters:
//   area_ratio = A_throat / A_inlet (0.25 = 4:1 contraction)
//   Re = u_ref * D_throat / nu
//
// Geometry: sinusoidal wall profile
//   y_wall(x) = (NY/2) * [1 - (1 - ar) * sin^2(pi * x / L_nozzle)]
//   for x in [x_start, x_start + L_nozzle]
// ==========================================================================

struct NozzleParams {
    double tau;
    double u_ref;
    int num_steps;
    int save_interval;
    double length_scale;
    double area_ratio;
    int x_start;
    int x_throat;
    int x_end;
    int y_center;
};

NozzleParams compute_params(double Re, double ar, int steps = -1) {
    double u_ref = (Re > 300) ? 0.05 : 0.1;
    int y_center = NY / 2;
    int throat_half = NY / 6;
    double length_scale = static_cast<double>(2 * throat_half);
    double nu = u_ref * length_scale / Re;
    double tau = 0.5 + 3.0 * nu;

    int nozzle_len = NX / 2;
    int x_start = NX / 8;
    int x_throat = x_start + nozzle_len / 2;
    int x_end = x_start + nozzle_len;

    int num_steps = (steps > 0) ? steps
        : std::max(4000, static_cast<int>(10.0 * NX / u_ref));
    int save_interval = num_steps / 50;

    return {tau, u_ref, num_steps, save_interval, length_scale, ar,
            x_start, x_throat, x_end, y_center};
}

void place_nozzle_walls(LBMCapabilities& system, const NozzleParams& p) {
    double ar = p.area_ratio;
    int half_inlet = NY / 4;

    for (int y = 0; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {
            if (y == 0 || y == NY - 1) {
                system.obstacle[node_index(x, y)] = true;
                continue;
            }
            if (x < p.x_start || x >= p.x_end) continue;

            double t = static_cast<double>(x - p.x_start)
                       / static_cast<double>(p.x_end - p.x_start);
            double wall_offset = half_inlet * (1.0 - (1.0 - ar)
                                 * std::sin(M_PI * t) * std::sin(M_PI * t));
            int wall_y = static_cast<int>(wall_offset);

            if (y < wall_y || y >= NY - wall_y) {
                system.obstacle[node_index(x, y)] = true;
            }
        }
    }
}

int main(int argc, char* argv[]) {
    std::cout << "==============================================" << std::endl;
    std::cout << " LBM-2D: Converging-Diverging Nozzle" << std::endl;
    std::cout << " D2Q9 | MRT | OpenMP | Cache-Optimized" << std::endl;
    std::cout << "==============================================" << std::endl;

    double Re = 100.0;
    double area_ratio = 0.25;
    int steps = -1;

    int positional_idx = 1;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--use-les") { g_use_les = true;
        } else if (arg == "--cs" && i + 1 < argc) { g_cs = std::stod(argv[++i]);
        } else if (arg.find("--") != 0) {
            if (positional_idx == 1) Re = std::stod(arg);
            else if (positional_idx == 2) area_ratio = std::stod(arg);
            else if (positional_idx == 3) steps = std::stoi(arg);
            ++positional_idx;
        }
    }

    g_case = CaseType::NOZZLE;

    auto params = compute_params(Re, area_ratio, steps);
    LBMCapabilities system;

    if (params.tau < 0.55 && !g_use_les) {
        g_use_les = true;
        std::cout << "  Auto-LES enabled (tau=" << params.tau << " < 0.55)" << std::endl;
    }

    place_nozzle_walls(system, params);

    for (int n = 0; n < NX * NY; ++n) {
        double* f_node = &system.f[n * 9];
        for (int i = 0; i < 9; ++i) {
            f_node[i] = compute_equilibrium(i, 1.0, params.u_ref, 0.0);
        }
    }

    std::string subdir = "output/nozzle/re" + std::to_string(static_cast<int>(Re))
                         + "_ar" + std::to_string(static_cast<int>(area_ratio * 100));
    std::filesystem::create_directories(subdir + "/frames");

    save_meta_json(subdir, Re, params.tau, params.u_ref,
                   params.length_scale, "nozzle", NX, NY);

    std::cout << "Re = " << Re
              << "  area_ratio = " << area_ratio
              << "  tau = " << params.tau
              << "  steps = " << params.num_steps
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
        }

        if (step % 500 == 0) {
            std::cout << "  step " << std::setw(6) << step
                      << "  Fx = " << std::fixed << std::setprecision(4) << fx_total
                      << std::endl;
        }
    }

    std::cout << "==============================================" << std::endl;
    std::cout << " Nozzle simulation complete." << std::endl;
    std::cout << "  Re = " << Re << "  AR = " << area_ratio << std::endl;
    std::cout << "  Steps = " << params.num_steps << std::endl;
    std::cout << "==============================================" << std::endl;

    return 0;
}
