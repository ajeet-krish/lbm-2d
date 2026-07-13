#include "lbm.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include <filesystem>

// ==========================================================================
// LBM-2D: Square Cylinder Flow (ERCOFTAC Case 043)
// ==========================================================================
// Usage:
//   ./build/LBM_SquareCylinder                      (default: Re=200)
//   ./build/LBM_SquareCylinder 200                  (Re=200)
//   ./build/LBM_SquareCylinder 1000 20000           (Re=1000, 20000 steps)
//
// Sharp-edge separation at front corners (fixed separation points).
// St ~ 0.13 at high Re (vs ~0.16 for circular cylinder).
// Validation: Lyn et al. (1995) J. Fluid Mech. Vol. 304, pp. 285-319.
// ==========================================================================

struct SquareCylParams {
    double tau;
    double u_inflow;
    int num_steps;
    int save_interval;
    double length_scale;
    int side;           // side length of square
    int cx, cy;         // center position
};

SquareCylParams compute_params(double Re, int steps = -1) {
    double u_inflow = 0.1;
    int side = NY / 10;                     // side length = 30 at NY=300
    double length_scale = static_cast<double>(side);
    double nu = u_inflow * length_scale / Re;
    double tau = 0.5 + 3.0 * nu;

    int cx = NX / 4;
    int cy = NY / 2;

    int num_steps = (steps > 0) ? steps
        : std::max(4000, static_cast<int>(10.0 * NX / u_inflow));
    int save_interval = num_steps / 50;

    return {tau, u_inflow, num_steps, save_interval, length_scale, side, cx, cy};
}

int main(int argc, char* argv[]) {
    std::cout << "==============================================" << std::endl;
    std::cout << " LBM-2D: Square Cylinder (ERCOFTAC 043)" << std::endl;
    std::cout << " D2Q9 | MRT | OpenMP | Cache-Optimized" << std::endl;
    std::cout << "==============================================" << std::endl;

    double Re = 200.0;
    int steps = -1;

    int positional_idx = 1;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--use-les") { g_use_les = true;
        } else if (arg == "--cs" && i + 1 < argc) { g_cs = std::stod(argv[++i]);
        } else if (arg.find("--") != 0) {
            if (positional_idx == 1) Re = std::stod(arg);
            else if (positional_idx == 2) steps = std::stoi(arg);
            ++positional_idx;
        }
    }

    g_case = CaseType::SQUARE_CYLINDER;

    auto params = compute_params(Re, steps);
    LBMCapabilities system;

    int half = params.side / 2;

    // Square polygon expanded by 0.5 to include boundary cells in ray-casting
    // (point_in_polygon uses strict <, so interior-only points miss the outer shell)
    double expand = 0.5;
    std::vector<std::pair<double,double>> sq_poly = {
        {static_cast<double>(params.cx - half - expand), static_cast<double>(params.cy - half - expand)},
        {static_cast<double>(params.cx + half + expand), static_cast<double>(params.cy - half - expand)},
        {static_cast<double>(params.cx + half + expand), static_cast<double>(params.cy + half + expand)},
        {static_cast<double>(params.cx - half - expand), static_cast<double>(params.cy + half + expand)}
    };

    place_polygon(system, sq_poly);

    for (int n = 0; n < NX * NY; ++n) {
        double* f_node = &system.f[n * 9];
        for (int i = 0; i < 9; ++i) {
            f_node[i] = compute_equilibrium(i, 1.0, params.u_inflow, 0.0);
        }
    }

    std::string subdir = "output/square_cylinder/re" + std::to_string(static_cast<int>(Re));
    std::filesystem::create_directories(subdir + "/frames");

    save_meta_json(subdir, Re, params.tau, params.u_inflow,
                   params.length_scale, "square-cylinder", NX, NY);

    std::cout << "Re = " << Re
              << "  tau = " << params.tau
              << "  steps = " << params.num_steps
              << "  side = " << params.side
              << (g_use_les ? "  LES(Cs=" + std::to_string(g_cs) + ")" : "")
              << std::endl;

    double cd_sum = 0.0, cl_max = 0.0, cl_min = 0.0;
    int n_report = 0;

    for (int step = 0; step <= params.num_steps; ++step) {
        execute_time_step(system, params.tau, params.u_inflow);

        double fx_total = 0.0, fy_total = 0.0;
        for (int n = 0; n < NX * NY; ++n) {
            fx_total += system.fx_body[n];
            fy_total += system.fy_body[n];
        }

        double cd = 2.0 * fx_total / (params.length_scale * params.u_inflow * params.u_inflow);
        double cl = 2.0 * fy_total / (params.length_scale * params.u_inflow * params.u_inflow);

        save_forces_jsonl(subdir, step, cd, cl);

        if (step % params.save_interval == 0) {
            save_json_frame(system, step, subdir);
        }

        if (step > params.num_steps / 2) {
            cd_sum += cd;
            if (cl > cl_max) cl_max = cl;
            if (cl < cl_min) cl_min = cl;
            ++n_report;
        }

        if (step % 500 == 0) {
            std::cout << "  step " << std::setw(6) << step
                      << "  Cd = " << std::fixed << std::setprecision(4) << cd
                      << "  Cl = " << std::fixed << std::setprecision(4) << cl
                      << std::endl;
        }
    }

    double cd_mean = (n_report > 0) ? cd_sum / n_report : 0.0;
    double cl_amp = (cl_max - cl_min) / 2.0;

    std::cout << "==============================================" << std::endl;
    std::cout << " RESULTS" << std::endl;
    std::cout << "  Mean Cd = " << cd_mean << std::endl;
    std::cout << "  Cl amplitude = " << cl_amp << std::endl;
    std::cout << "  Steps = " << params.num_steps << std::endl;
    std::cout << "==============================================" << std::endl;

    return 0;
}
