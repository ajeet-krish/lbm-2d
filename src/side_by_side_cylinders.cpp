#include "lbm.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include <filesystem>
#include <random>

// ==========================================================================
// LBM-2D: Side-by-Side Cylinders (Interference)
// ==========================================================================
// Usage:
//   ./build/LBM_SideBySide 100 3             (Re=100, spacing S/D=3)
//   ./build/LBM_SideBySide 100 5             (Re=100, spacing S/D=5)
//   ./build/LBM_SideBySide 100 2 20000       (Re=100, S/D=2, 20000 steps)
//
// Two identical cylinders placed side-by-side perpendicular to the flow.
// The gap between cylinders creates interference effects:
//   - S/D < 1.5: Single combined wake (shared vortex street)
//   - S/D ~ 1.5-3.5: Biased (deflected) gap flow, flip-flopping
//   - S/D > 3.5: Independent vortex shedding
//
// This is DIFFERENT from tandem cylinders (series). Side-by-side
// produces parallel interference with gap flow acceleration.
//
// Validation: Zdravkovich 1977, Kim & Durbin 1988
//
// Parameters:
//   Re = u_inflow * D / nu,  D = 2 * NY/10 = 60
//   S/D = center-to-center spacing / diameter
//   S = center-to-center distance
// ==========================================================================

struct SideBySideParams {
    double tau;
    double u_inflow;
    int num_steps;
    int save_interval;
    double length_scale;
    int cx1, cx2, cy;
    int radius;
    double sd_ratio;
};

SideBySideParams compute_params(double Re, double sd_ratio, int steps = -1) {
    double u_inflow = 0.1;
    int radius = NY / 10;
    int D = 2 * radius;
    double length_scale = static_cast<double>(D);
    double nu = u_inflow * length_scale / Re;
    double tau = 0.5 + 3.0 * nu;

    int spacing = static_cast<int>(sd_ratio * D);
    int cy = NY / 2;
    int cx1 = NX / 4 - spacing / 2;
    int cx2 = cx1 + spacing;

    int num_steps = (steps > 0) ? steps
        : std::max(4000, static_cast<int>(10.0 * NX / u_inflow));
    int save_interval = num_steps / 50;

    return {tau, u_inflow, num_steps, save_interval, length_scale,
            cx1, cx2, cy, radius, sd_ratio};
}

int main(int argc, char* argv[]) {
    std::cout << "==============================================" << std::endl;
    std::cout << " LBM-2D: Side-by-Side Cylinders" << std::endl;
    std::cout << " D2Q9 | MRT | OpenMP | Cache-Optimized" << std::endl;
    std::cout << "==============================================" << std::endl;

    double Re = 100.0;
    double sd_ratio = 3.0;
    int steps = -1;

    int positional_idx = 1;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--use-les") { g_use_les = true;
        } else if (arg == "--cs" && i + 1 < argc) { g_cs = std::stod(argv[++i]);
        } else if (arg.find("--") != 0) {
            if (positional_idx == 1) Re = std::stod(arg);
            else if (positional_idx == 2) sd_ratio = std::stod(arg);
            else if (positional_idx == 3) steps = std::stoi(arg);
            ++positional_idx;
        }
    }

    g_case = CaseType::SIDE_BY_SIDE;

    auto params = compute_params(Re, sd_ratio, steps);
    LBMCapabilities system;

    place_cylinder(system, params.cx1, params.cy, params.radius);
    place_cylinder(system, params.cx2, params.cy, params.radius);

    for (int n = 0; n < NX * NY; ++n) {
        double* f_node = &system.f[n * 9];
        for (int i = 0; i < 9; ++i) {
            f_node[i] = compute_equilibrium(i, 1.0, params.u_inflow, 0.0);
        }
    }

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> pert_dist(-1e-4, 1e-4);
    int x_start = std::min(params.cx1, params.cx2) + 5;
    int x_end = std::min(NX, std::max(params.cx1, params.cx2) + 60);
    for (int x = x_start; x < x_end; ++x) {
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

    int sd_int = static_cast<int>(sd_ratio * 10);
    std::string subdir = "output/side_by_side/re"
                         + std::to_string(static_cast<int>(Re))
                         + "_sd" + std::to_string(sd_int);
    std::filesystem::create_directories(subdir + "/frames");

    save_meta_json(subdir, Re, params.tau, params.u_inflow,
                   params.length_scale, "side-by-side", NX, NY);

    std::cout << "Re = " << Re
              << "  S/D = " << sd_ratio
              << "  tau = " << params.tau
              << "  steps = " << params.num_steps
              << "  D = " << 2 * params.radius
              << "  spacing = " << (params.cx2 - params.cx1)
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

        save_forces_jsonl(subdir, step, cd, cl);

        if (step % params.save_interval == 0) {
            save_json_frame(system, step, subdir);
        }

        if (step % 500 == 0) {
            std::cout << "  step " << std::setw(6) << step
                      << "  Cd = " << std::fixed << std::setprecision(4) << cd
                      << "  Cl = " << std::fixed << std::setprecision(4) << cl
                      << std::endl;
        }
    }

    std::cout << "==============================================" << std::endl;
    std::cout << " Side-by-side cylinders simulation complete." << std::endl;
    std::cout << "  Re = " << Re << "  S/D = " << sd_ratio << std::endl;
    std::cout << "  Steps = " << params.num_steps << std::endl;
    std::cout << "==============================================" << std::endl;

    return 0;
}
