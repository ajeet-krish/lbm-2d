#include "lbm.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include <filesystem>
#include <random>

// ==========================================================================
// LBM-2D: Cylinder Near a Wall (Ground Effect)
// ==========================================================================
// Usage:
//   ./build/LBM_CylinderNearWall 100 20       (Re=100, gap=20 cells)
//   ./build/LBM_CylinderNearWall 200 40       (Re=200, gap=40 cells)
//   ./build/LBM_CylinderNearWall 100 10 20000 (Re=100, gap=10, 20000 steps)
//
// A cylinder placed near a flat wall. The wall gap (distance from
// cylinder bottom to wall) modifies the wake structure and drag.
// At small gaps, the wake is suppressed; at large gaps, it approaches
// free-space behavior.
//
// This is relevant for:
//   - Landing gear proximity to runway during rollout
//   -翼身 junction flows
//   - Offshore structure foundations near seabed
//   - Heat exchanger tube clearance
//
// Parameters:
//   Re = u_inflow * D / nu,  D = 2 * NY/10 = 60
//   gap = cells between cylinder bottom and wall (default 20)
//   wall at y=0
// ==========================================================================

struct GapParams {
    double tau;
    double u_inflow;
    int num_steps;
    int save_interval;
    double length_scale;
    int cx_cyl, cy_cyl, radius;
    int gap;
};

GapParams compute_params(double Re, int gap, int steps = -1) {
    double u_inflow = 0.1;
    int radius = NY / 10;
    int D = 2 * radius;
    double length_scale = static_cast<double>(D);
    double nu = u_inflow * length_scale / Re;
    double tau = 0.5 + 3.0 * nu;

    int cx_cyl = NX / 4;
    int cy_cyl = radius + gap;

    int num_steps = (steps > 0) ? steps
        : std::max(4000, static_cast<int>(10.0 * NX / u_inflow));
    int save_interval = num_steps / 50;

    return {tau, u_inflow, num_steps, save_interval, length_scale,
            cx_cyl, cy_cyl, radius, gap};
}

int main(int argc, char* argv[]) {
    std::cout << "==============================================" << std::endl;
    std::cout << " LBM-2D: Cylinder Near a Wall" << std::endl;
    std::cout << " D2Q9 | MRT | OpenMP | Cache-Optimized" << std::endl;
    std::cout << "==============================================" << std::endl;

    double Re = 100.0;
    int gap = 20;
    int steps = -1;

    int positional_idx = 1;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--use-les") { g_use_les = true;
        } else if (arg == "--cs" && i + 1 < argc) { g_cs = std::stod(argv[++i]);
        } else if (arg.find("--") != 0) {
            if (positional_idx == 1) Re = std::stod(arg);
            else if (positional_idx == 2) gap = std::stoi(arg);
            else if (positional_idx == 3) steps = std::stoi(arg);
            ++positional_idx;
        }
    }

    g_case = CaseType::CYLINDER_NEAR_WALL;

    auto params = compute_params(Re, gap, steps);
    LBMCapabilities system;

    place_cylinder(system, params.cx_cyl, params.cy_cyl, params.radius);

    // Add physical wall at y=0 (bottom boundary)
    for (int x = 0; x < NX; ++x) {
        system.obstacle[node_index(x, 0)] = true;
    }

    for (int n = 0; n < NX * NY; ++n) {
        double* f_node = &system.f[n * 9];
        for (int i = 0; i < 9; ++i) {
            f_node[i] = compute_equilibrium(i, 1.0, params.u_inflow, 0.0);
        }
    }

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> pert_dist(-1e-4, 1e-4);
    for (int x = params.cx_cyl + 5; x < std::min(NX, params.cx_cyl + 60); ++x) {
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

    std::string subdir = "output/cylinder_near_wall/re"
                         + std::to_string(static_cast<int>(Re))
                         + "_gap" + std::to_string(gap);
    std::filesystem::create_directories(subdir + "/frames");

    save_meta_json(subdir, Re, params.tau, params.u_inflow,
                   params.length_scale, "cylinder-near-wall", NX, NY);

    std::cout << "Re = " << Re
              << "  gap = " << gap << " cells"
              << "  tau = " << params.tau
              << "  steps = " << params.num_steps
              << "  D = " << 2 * params.radius
              << "  cy = " << params.cy_cyl
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
    std::cout << " Cylinder near wall simulation complete." << std::endl;
    std::cout << "  Re = " << Re << "  gap = " << gap << std::endl;
    std::cout << "  Steps = " << params.num_steps << std::endl;
    std::cout << "==============================================" << std::endl;

    return 0;
}
