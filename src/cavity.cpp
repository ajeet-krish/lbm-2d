#include "lbm.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include <filesystem>

// ==========================================================================
// LBM-2D Lid-Driven Cavity Entry Point
// ==========================================================================
// Usage:
//   ./LBM_Cavity                       (default: Re=100, 256x256 grid)
//   ./LBM_Cavity <Re> <nx> <steps>     (e.g. ./LBM_Cavity 400 256 50000)
//
// Validates against Ghia, Ghia & Shin 1982 centerline velocity profiles.
// Re = U_lid * NX / nu where NX is the cavity width.
// ==========================================================================

struct CavityParams {
    double tau;
    double u_lid;
    int num_steps;
    int vtk_interval;
};

CavityParams compute_params(double Re, int nx, int steps) {
    double u_lid = 0.1;
    double nu = u_lid * nx / Re;
    double tau = 0.5 + 3.0 * nu;
    int num_steps = (steps > 0) ? steps : std::max(10000, static_cast<int>(5.0 * nx / u_lid));
    int vtk_interval = num_steps / 50;

    return {tau, u_lid, num_steps, vtk_interval};
}

int main(int argc, char* argv[]) {
    std::cout << "==============================================" << std::endl;
    std::cout << " LBM-2D: Lid-Driven Cavity" << std::endl;
    std::cout << " D2Q9 | OpenMP | Cache-Optimized" << std::endl;
    std::cout << "==============================================" << std::endl;

    double Re = 100.0;
    int nx = 256;
    int ny = 256;
    int steps = -1;

    int positional_idx = 1;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--use-les") {
            g_use_les = true;
        } else if (arg == "--cs" && i + 1 < argc) {
            g_cs = std::stod(argv[++i]);
        } else if (arg.find("--") != 0) {
            if (positional_idx == 1) Re = std::stod(arg);
            else if (positional_idx == 2) nx = std::stoi(arg);
            else if (positional_idx == 3) steps = std::stoi(arg);
            ++positional_idx;
        }
    }
    ny = nx;  // square cavity

    // Set globals for cavity mode
    NX = nx;
    NY = ny;
    g_case = CaseType::CAVITY;

    auto params = compute_params(Re, nx, steps);
    LBMCapabilities system;

    // Place bounce-back walls on all 4 boundaries
    place_walls(system);
    // Note: the top wall (y = NY-1) uses moving-wall bounce-back with
    // momentum correction applied during streaming to enforce u = u_lid.

    int n_nodes = NX * NY;

    // Initialize fluid at rest
    for (int n = 0; n < n_nodes; ++n) {
        double* f_node = &system.f[n * 9];
        for (int i = 0; i < 9; ++i) {
            f_node[i] = compute_equilibrium(i, 1.0, 0.0, 0.0);
        }
    }

    std::cout << "Re = " << Re
              << "  grid = " << NX << "x" << NY
              << "  tau = " << params.tau
              << "  u_lid = " << params.u_lid
              << "  steps = " << params.num_steps
              << (g_use_les ? "  LES(Cs=" + std::to_string(g_cs) + ")" : "")
              << std::endl;

    // Create output directory
    std::string subdir = "output/cavity/re" + std::to_string(static_cast<int>(Re));
    std::filesystem::create_directories(subdir + "/frames");

    for (int step = 0; step <= params.num_steps; ++step) {
        execute_time_step(system, params.tau, params.u_lid);

        if (step % params.vtk_interval == 0) {
            save_json_frame(system, step, subdir);
            save_vtk_frame(system, step, subdir);
        }

        if (step % 1000 == 0) {
            // Sample centerline u velocity at x = NX/2, various y
            int mid_x = NX / 2;
            std::cout << "  step " << std::setw(6) << step;
            // Show u-velocity at a few points along the vertical centerline
            for (int frac : {1, 3, 5, 7, 9}) {
                int y = NY * frac / 10;
                int idx = node_index(mid_x, y);
                double rho, u, v;
                compute_macros(&system.f[idx * 9], rho, u, v);
                std::cout << "  u(y=" << std::setw(2) << frac * 10 << "%)=" << std::fixed << std::setprecision(4) << u;
            }
            std::cout << std::endl;
        }
    }

    // Extract final centerline profiles for validation
    std::string profile_file = subdir + "/centerline.csv";
    std::ofstream csv_out(profile_file);
    csv_out << "y,mid_x,u,v\n";
    int mid_x = NX / 2;
    for (int y = 0; y < NY; ++y) {
        int idx = node_index(mid_x, y);
        double rho, u, v;
        compute_macros(&system.f[idx * 9], rho, u, v);
        csv_out << y << "," << mid_x << "," << u << "," << v << "\n";
    }

    std::cout << "==============================================" << std::endl;
    std::cout << " Cavity simulation complete." << std::endl;
    std::cout << "  Re = " << Re << std::endl;
    std::cout << "  Grid = " << NX << "x" << NY << std::endl;
    std::cout << "  Steps = " << params.num_steps << std::endl;
    std::cout << "  Centerline profile saved to " << profile_file << std::endl;
    std::cout << "==============================================" << std::endl;

    return 0;
}
