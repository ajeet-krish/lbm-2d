#include "lbm.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include <filesystem>

// ==========================================================================
// LBM-2D: Natural Convection in a Square Cavity (Thermal LBM, Upgrade 4)
// ==========================================================================
// Usage:
//   ./build/LBM_ThermalCavity 1e4          (Ra = 1e4, steady)
//   ./build/LBM_ThermalCavity 1e6 40000    (Ra = 1e6, 40000 steps)
//
// Canonical benchmark (de Vahl Davis 1983, de Zwaan 1978):
//   Square cavity, aspect ratio 1. Hot left wall (Th), cold right wall (Tc).
//   Top and bottom walls adiabatic. No-slip everywhere. Driven purely by
//   buoyancy (Boussinesq). No forced inflow.
//
//   Ra = g * beta * (Th - Tc) * L^3 / (nu * alpha)
//   Pr = nu / alpha (0.71 for air)
//   Nu_local = (dT/dx)|_wall * L / (Th - Tc)  (at hot wall)
//
// Validation targets (de Vahl Davis 1983):
//   Ra=1e4: Nu_avg = 2.238, max|psi| = 2.07e-3 (Pr=0.71)
//   Ra=1e5: Nu_avg = 4.505, max|psi| = 9.11e-3
//   Ra=1e6: Nu_avg = 8.817, max|psi| = 1.62e-2
// ==========================================================================

int main(int argc, char* argv[]) {
    std::cout << "=============================================" << std::endl;
    std::cout << " LBM-2D: Natural Convection Cavity" << std::endl;
    std::cout << " D2Q9 | MRT | Thermal DDF | Boussinesq" << std::endl;
    std::cout << "=============================================" << std::endl;

    double Ra = 1e4;
    int steps = -1;

    int positional_idx = 1;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--use-les") {
            g_use_les = true;
        } else if (arg == "--nx" && i + 1 < argc) {
            NX = std::stoi(argv[++i]);
        } else if (arg == "--ny" && i + 1 < argc) {
            NY = std::stoi(argv[++i]);
        } else if (arg.find("--") != 0) {
            if (positional_idx == 1) Ra = std::stod(arg);
            else if (positional_idx == 2) steps = std::stoi(arg);
            ++positional_idx;
        }
    }

    // Fixed geometry: square cavity, L = NY (lattice units)
    // For the benchmark the cavity must be square, so force NX = NY.
    NX = NY;
    double L = static_cast<double>(NY);
    double Pr = 0.71;
    double Th = 1.0 + 0.5;   // hot wall (left)
    double Tc = 1.0 - 0.5;   // cold wall (right)
    double dT = Th - Tc;
    double T_ref = 1.0;

    // Derive nu and alpha from a STABLE target tau, then back out the
    // Boussinesq coefficient product (g * beta) from the Ra definition:
    //   Ra = g * beta * dT * L^3 / (nu * alpha)
    // Choose tau for stability (U ~ 0.02-0.05), then:
    //   nu * alpha = (tau - 0.5) / 3 * nu / Pr
    //   g * beta   = Ra * (nu * alpha) / (dT * L^3)
    double tau = 0.8;                       // stable, well-resolved
    double nu = (tau - 0.5) / 3.0;
    double alpha = nu / Pr;
    double nu_alpha = nu * alpha;
    double gb_product = Ra * nu_alpha / (dT * L * L * L);
    // Split: beta = 0.001 (Boussinesq coeff), g = gb_product / beta
    double beta = 0.001;
    double g_accel = gb_product / beta;
    double omega_k = 1.0 / (0.5 + 3.0 * alpha);

    int num_steps = (steps > 0) ? steps
        : (Ra <= 1e4 ? 20000 : (Ra <= 1e5 ? 40000 : 60000));
    int save_interval = num_steps / 50;

    std::cout << "  Ra = " << Ra << ", Pr = " << Pr << std::endl;
    std::cout << "  nu = " << nu << ", tau = " << tau << std::endl;
    std::cout << "  alpha = " << alpha << ", omega_k = " << omega_k << std::endl;
    std::cout << "  beta = " << beta << ", g = " << g_accel << std::endl;
    std::cout << "  Th = " << Th << ", Tc = " << Tc << std::endl;
    std::cout << "  Steps = " << num_steps << std::endl;
    std::cout << "=============================================" << std::endl;

    LBMCapabilities system;
    system.use_thermal = true;
    system.T_ref = T_ref;
    system.alpha = alpha;
    system.omega_k = omega_k;
    system.beta = beta;
    system.g_buoyancy = -g_accel;  // downward (negative y)

    // Mark the 4 domain walls as solid (obstacle) nodes, exactly as the
    // lid-driven cavity does. Thermal BCs are then applied on these solid
    // nodes (which survive streaming), so the temperature is correctly pinned.
    //   1 = hot (left,  x=0)
    //   2 = cold (right, x=NX-1)
    //   3 = adiabatic (top/bottom, y=0, y=NY-1)
    std::vector<int> thermal_bc(NX * NY, 0);
    for (int y = 0; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {
            int idx = node_index(x, y);
            if (x == 0) { system.obstacle[idx] = true; thermal_bc[idx] = 1; }
            else if (x == NX - 1) { system.obstacle[idx] = true; thermal_bc[idx] = 2; }
            else if (y == 0 || y == NY - 1) { system.obstacle[idx] = true; thermal_bc[idx] = 3; }
        }
    }

    // Initialize at rest, uniform T = T_ref
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {
            int idx = node_index(x, y);
            double rho = 1.0;
            double u = 0.0, v = 0.0;
            for (int i = 0; i < 9; ++i) {
                system.f[idx * 9 + i] = compute_equilibrium(i, rho, u, v);
                system.g_thermal[idx * 9 + i] = thermal_equilibrium(i, T_ref, u, v);
            }
        }
    }

    // Set thermal BCs on solid wall nodes before the loop
    #pragma omp parallel for
    for (int y = 0; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {
            int idx = node_index(x, y);
            if (!system.obstacle[idx]) continue;
            int bc = thermal_bc[idx];
            if (bc == 1) apply_thermal_isothermal(&system.g_thermal[idx * 9], Th);
            else if (bc == 2) apply_thermal_isothermal(&system.g_thermal[idx * 9], Tc);
            else if (bc == 3) {
                // adiabatic: bounce-back on g
                for (int i = 0; i < 9; ++i) {
                    int bb = bounce_back[i];
                    system.g_thermal[idx * 9 + i] = system.g_thermal[idx * 9 + bb];
                }
            }
        }
    }

    std::string outdir = "output/thermal_cavity_ra" + std::to_string(static_cast<int>(Ra));
    std::filesystem::create_directories(outdir + "/frames");

    std::ofstream meta(outdir + "/meta.json");
    meta << "{\"case\":\"thermal_cavity\",\"Ra\":" << Ra
         << ",\"Pr\":" << Pr << ",\"Th\":" << Th << ",\"Tc\":" << Tc
         << ",\"nu\":" << nu << ",\"alpha\":" << alpha
         << ",\"nx\":" << NX << ",\"ny\":" << NY << "}" << std::endl;
    meta.close();

    for (int step = 0; step <= num_steps; ++step) {
        execute_time_step(system, tau, 0.0);  // no inflow

        // Re-impose thermal BCs after each step. The default thermal streaming
        // treats solid walls as adiabatic, so we must PIN the temperature field
        // by setting the first FLUID node adjacent to each wall to thermal
        // equilibrium at the wall temperature (Dirichlet condition). This is the
        // standard "equilibrium BC" for the temperature distribution.
        #pragma omp parallel for
        for (int y = 1; y < NY - 1; ++y) {
            // Hot left wall: fluid node at x=1
            {
                int idx = node_index(1, y);
                double rho, u, v;
                compute_macros(&system.f[idx * 9], rho, u, v);
                for (int i = 0; i < 9; ++i)
                    system.g_thermal[idx * 9 + i] = thermal_equilibrium(i, Th, u, v);
            }
            // Cold right wall: fluid node at x=NX-2
            {
                int idx = node_index(NX - 2, y);
                double rho, u, v;
                compute_macros(&system.f[idx * 9], rho, u, v);
                for (int i = 0; i < 9; ++i)
                    system.g_thermal[idx * 9 + i] = thermal_equilibrium(i, Tc, u, v);
            }
        }
        #pragma omp parallel for
        for (int x = 1; x < NX - 1; ++x) {
            // Adiabatic top/bottom: bounce-back on g at fluid node x=1,NY-2
            {
                int idx = node_index(x, 1);
                for (int i = 0; i < 9; ++i) {
                    int bb = bounce_back[i];
                    system.g_thermal[idx * 9 + i] = system.g_thermal[idx * 9 + bb];
                }
            }
            {
                int idx = node_index(x, NY - 2);
                for (int i = 0; i < 9; ++i) {
                    int bb = bounce_back[i];
                    system.g_thermal[idx * 9 + i] = system.g_thermal[idx * 9 + bb];
                }
            }
        }

        if (step % save_interval == 0) {
            // Nusselt at hot wall: Nu = (dT/dx)|_wall * L / dT
            // Fluid nodes x=1 (Th) and x=2 give the gradient into the fluid.
            double Nu_avg = 0.0;
            int n_wall = 0;
            #pragma omp parallel for reduction(+:Nu_avg, n_wall)
            for (int y = 1; y < NY - 1; ++y) {
                int idx1 = node_index(1, y);       // pinned to Th
                int idx2 = node_index(2, y);       // interior fluid
                double T1, T2;
                compute_temperature(&system.g_thermal[idx1 * 9], T1);
                compute_temperature(&system.g_thermal[idx2 * 9], T2);
                double dTdx = (T2 - T1) / 1.0;  // outward normal (into fluid)
                double Nu = -dTdx * L / dT;     // |Nu| at hot wall (positive)
                Nu_avg += Nu;
                ++n_wall;
            }
            if (n_wall > 0) Nu_avg /= n_wall;

            std::cout << "  Step " << step << " / " << num_steps
                      << " | Nu_avg = " << Nu_avg << std::endl;
            save_json_frame_thermal(system, step, outdir, Th);
        }
    }

    std::cout << "  Done. Output: " << outdir << std::endl;
    return 0;
}
