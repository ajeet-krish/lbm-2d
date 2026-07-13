#include "lbm.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include <filesystem>

// ==========================================================================
// LBM-2D: Periodic Hills (Canonical LES Benchmark)
// ==========================================================================
// Usage:
//   ./build/LBM_PeriodicHills 100           (Re=100, laminar)
//   ./build/LBM_PeriodicHills 2800          (Re=2800, turbulent, uses LES)
//   ./build/LBM_PeriodicHills 100 20000     (Re=100, 20000 steps)
//
// The periodic hills geometry is the canonical test case for separated
// turbulent flows (Moser, Kim, Moin 1993; Le, Moin, Kim 1997).
// A sinusoidal hill profile on the bottom wall drives flow separation
// and recirculation. Periodic in x, no-slip on hills, moving lid on top.
//
// Geometry:
//   Bottom wall: y = H - h_max * (1 - cos(2*pi*x/L))^alpha / 2^alpha
//   Top wall: y = H (flat, moving at u_lid)
//   Periodic in x-direction
//
// Parameters:
//   H = channel half-height = NY * 2/3
//   h_max = hill height = H * hmax_frac (default 1/6; was 1/2 which nearly
//           closes the channel and hides the sinusoidal profile). Keeping
//           h_max moderate reveals 2-3 full hill-valley periods.
//   L = hill wavelength = 9 * H (standard)
//   Re = u_lid * H / nu
// ==========================================================================

struct HillParams {
    double tau;
    double u_lid;
    int num_steps;
    int save_interval;
    double length_scale;
    int H;
    int h_max;
    int L;
};

HillParams compute_params(double Re, int steps = -1, double hmax_frac = 1.0 / 6.0) {
    double u_lid = 0.1;
    int H = NY * 2 / 3;
    int h_max = std::max(1, static_cast<int>(H * hmax_frac));
    int L = 9 * H;

    double length_scale = static_cast<double>(H);
    double nu = u_lid * length_scale / Re;
    double tau = 0.5 + 3.0 * nu;

    int num_steps = (steps > 0) ? steps
        : std::max(40000, static_cast<int>(30.0 * NX / u_lid));
    int save_interval = num_steps / 50;

    return {tau, u_lid, num_steps, save_interval, length_scale, H, h_max, L};
}

void place_hills(LBMCapabilities& system, const HillParams& p) {
    int H = p.H;
    int h_max = p.h_max;
    int L = p.L;

    for (int x = 0; x < NX; ++x) {
        double x_rel = static_cast<double>(x) / static_cast<double>(L);
        double hill_y = H - h_max * 0.5
                        * (1.0 - std::cos(2.0 * M_PI * x_rel))
                        * (1.0 - std::cos(2.0 * M_PI * x_rel));
        int hill_iy = static_cast<int>(std::ceil(hill_y));

        for (int y = 0; y < NY; ++y) {
            if (y == NY - 1) system.obstacle[node_index(x, y)] = true;
            if (y < hill_iy) system.obstacle[node_index(x, y)] = true;
        }
    }
}

int main(int argc, char* argv[]) {
    std::cout << "==============================================" << std::endl;
    std::cout << " LBM-2D: Periodic Hills" << std::endl;
    std::cout << " D2Q9 | MRT | OpenMP | Cache-Optimized" << std::endl;
    std::cout << "==============================================" << std::endl;

    double Re = 100.0;
    int steps = -1;
    double hmax_frac = 1.0 / 6.0;

    int positional_idx = 1;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--use-les") { g_use_les = true;
        } else if (arg == "--cs" && i + 1 < argc) { g_cs = std::stod(argv[++i]);
        } else if (arg == "--hmax" && i + 1 < argc) { hmax_frac = std::stod(argv[++i]);
        } else if (arg.find("--") != 0) {
            if (positional_idx == 1) Re = std::stod(arg);
            else if (positional_idx == 2) steps = std::stoi(arg);
            ++positional_idx;
        }
    }

    g_case = CaseType::PERIODIC_HILLS;

    auto params = compute_params(Re, steps, hmax_frac);

    // Auto-LES: enable Smagorinsky for low-tau (high-Re) runs for stability.
    if (params.tau < 0.55) g_use_les = true;
    LBMCapabilities system;

    place_hills(system, params);

    for (int n = 0; n < NX * NY; ++n) {
        double* f_node = &system.f[n * 9];
        for (int i = 0; i < 9; ++i) {
            f_node[i] = compute_equilibrium(i, 1.0, params.u_lid, 0.0);
        }
    }

    std::string subdir = "output/periodic_hills/re" + std::to_string(static_cast<int>(Re));
    std::filesystem::create_directories(subdir + "/frames");

    save_meta_json(subdir, Re, params.tau, params.u_lid,
                   params.length_scale, "periodic-hills", NX, NY);

    std::cout << "Re = " << Re
              << "  tau = " << params.tau
              << "  steps = " << params.num_steps
              << "  H = " << params.H
              << "  h_max = " << params.h_max
              << "  h_max/H = " << (static_cast<double>(params.h_max) / params.H)
              << "  L = " << params.L
              << (g_use_les ? "  LES(Cs=" + std::to_string(g_cs) + ")" : "")
              << std::endl;

    for (int step = 0; step <= params.num_steps; ++step) {
        execute_time_step(system, params.tau, params.u_lid);

        double fx_total = 0.0, fy_total = 0.0;
        for (int n = 0; n < NX * NY; ++n) {
            fx_total += system.fx_body[n];
            fy_total += system.fy_body[n];
        }

        save_forces_jsonl(subdir, step, fx_total, fy_total);

        if (step % params.save_interval == 0) {
            save_json_frame(system, step, subdir);
        }

        if (step % 5000 == 0) {
            std::cout << "  step " << std::setw(6) << step
                      << "  Fx = " << std::fixed << std::setprecision(4) << fx_total
                      << std::endl;
        }
    }

    std::cout << "==============================================" << std::endl;
    std::cout << " Periodic hills simulation complete." << std::endl;
    std::cout << "  Re = " << Re << std::endl;
    std::cout << "  Steps = " << params.num_steps << std::endl;
    std::cout << "==============================================" << std::endl;

    return 0;
}
