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
//   ./build/LBM_PeriodicHills 100 --force 5e-6  (override body force)
//   ./build/LBM_PeriodicHills 100 --hmax 0.166   (hill height fraction)
//
// The periodic hills geometry is the canonical test case for separated
// turbulent flows (Moser, Kim, Moin 1993; Le, Moin, Kim 1997).
// A sinusoidal hill profile on the bottom wall drives flow separation
// and recirculation. Periodic in x, no-slip on hills and top wall; the flow
// is sustained by a constant streamwise body force (not a moving lid).
//
// Geometry:
//   Bottom wall: y = H - h_max * (1 - cos(2*pi*3*x/NX))^2 / 2
//   Top wall: y = H (flat, stationary no-slip)
//   Periodic in x-direction
//   Three hill-valley periods tiled across the domain (n_cycles = 3),
//   each of wavelength NX/3.
//
// Driving mechanism:
//   The flow is sustained by a constant streamwise body force
//   (system.body_force_x) so the initial field does not decay to rest.
//   This matches the canonical pressure-driven periodic-hills setup.
//   The body force is wired into the MRT collision via Fscale.
//
// Parameters:
//   H = channel half-height = NY * 2/3
//   h_max = hill height = H * hmax_frac (default 1/6; was 1/2 which nearly
//           closes the channel and hides the sinusoidal profile). Keeping
//           h_max moderate reveals 2-3 full hill-valley periods.
//   L = NX (periodic domain); 3 hill cycles, each of wavelength NX/3.
//   Re = u_bulk * H / nu  (u_bulk from body-force balance)
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
    double body_force_x;
};

// Body force for pressure-driven periodic hills. For a channel of half-height H
// driven by streamwise acceleration a, steady Poiseuille gives u_bulk = a*H^2/(8*nu)
// (nu = (tau-0.5)/3 in lattice units). Choosing u_bulk = Re*nu/H yields:
//   a = 8 * (tau-0.5) * u_bulk / (3 * H^2)
// The hill geometry constricts the flow so we add a safety factor to reach target Re.
double compute_body_force(double tau, double Re, int H) {
    double nu = (tau - 0.5) / 3.0;
    double u_bulk = Re * nu / static_cast<double>(H);
    double a = 8.0 * (tau - 0.5) * u_bulk / (3.0 * static_cast<double>(H) * H);
    return a * 28.0;  // hill constriction reduces free area; empirically tuned
}

HillParams compute_params(double Re, int steps = -1, double hmax_frac = 1.0 / 6.0) {
    double u_lid = 0.1;
    int H = NY * 2 / 3;
    int h_max = std::max(1, static_cast<int>(H * hmax_frac));
    int L = 9 * H;

    double length_scale = static_cast<double>(H);
    double nu = u_lid * length_scale / Re;
    double tau = 0.5 + 3.0 * nu;
    double body_force_x = compute_body_force(tau, Re, H);

    int num_steps = (steps > 0) ? steps
        : std::max(40000, static_cast<int>(30.0 * NX / u_lid));
    int save_interval = num_steps / 50;

    // NOTE: L (= NX) sets the periodic domain length; the profile repeats
    // n_cycles = 3 times across it so the periodic x-boundary connects identical
    // hill heights. The original L = 9*H (1800) exceeded NX (800), leaving the
    // hill incomplete at x = NX-1 and producing a height discontinuity that
    // forced NaN divergence at step 2.
    L = NX;

    return {tau, u_lid, num_steps, save_interval, length_scale, H, h_max, L, body_force_x};
}

void place_hills(LBMCapabilities& system, const HillParams& p) {
    int H = p.H;
    int h_max = p.h_max;
    int L = p.L;
    int n_cycles = 3;  // show three hill-valley periods across the domain

    for (int x = 0; x < NX; ++x) {
        double x_rel = static_cast<double>(x) / static_cast<double>(L) * n_cycles;
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
    double force_override = -1.0;

    int positional_idx = 1;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--use-les") { g_use_les = true;
        } else if (arg == "--cs" && i + 1 < argc) { g_cs = std::stod(argv[++i]);
        } else if (arg == "--hmax" && i + 1 < argc) { hmax_frac = std::stod(argv[++i]);
        } else if (arg == "--force" && i + 1 < argc) { force_override = std::stod(argv[++i]);
        } else if (arg.find("--") != 0) {
            if (positional_idx == 1) Re = std::stod(arg);
            else if (positional_idx == 2) steps = std::stoi(arg);
            ++positional_idx;
        }
    }

    g_case = CaseType::PERIODIC_HILLS;

    auto params = compute_params(Re, steps, hmax_frac);
    if (force_override > 0.0) params.body_force_x = force_override;

    // Auto-LES: enable Smagorinsky for low-tau (high-Re) runs for stability.
    if (params.tau < 0.55) g_use_les = true;
    LBMCapabilities system;

    system.body_force_x = params.body_force_x;

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
              << "  Fx = " << params.body_force_x
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
