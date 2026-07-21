#include "lbm.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include <filesystem>
#include <random>

// ==========================================================================
// LBM-2D: Heated Cylinder in Crossflow (Natural Convection + Forced Convection)
// ==========================================================================
// Usage:
//   ./build/LBM_HeatedCylinder 100              (Re=100, T_wall=1.5, T_inf=1.0)
//   ./build/LBM_HeatedCylinder 200 1.5 30000   (Re=200, T_wall=1.5, 30000 steps)
//
// This case demonstrates the Double Distribution Function (DDF) thermal LBM.
// A heated cylinder at temperature T_wall in a crossflow at T_inf. The
// thermal field (temperature distribution) is solved alongside the momentum
// field, enabling:
//   - Local Nusselt number computation (heat transfer coefficient)
//   - Mixed convection (buoyancy + forced flow)
//   - Validation against Eshghy (1970), Nakai & Okazaki (1975) Nusselt data
//
// Parameters:
//   Re = u_inflow * D / nu,  D = 2 * NY/10 = 60
//   Pr = nu / alpha (0.71 for air)
//   Ra = g * beta * dT * D^3 / (nu * alpha)  (natural convection)
//   Nu = h * D / k
// ==========================================================================

struct HeatParams {
    double tau;
    double u_inflow;
    int num_steps;
    int save_interval;
    double length_scale;
    int cx_cyl, cy_cyl, radius;
    double T_wall, T_inf;
    bool use_buoyancy;
};

HeatParams compute_params(double Re, double T_wall, int steps = -1) {
    double u_inflow = 0.1;
    int radius = NY / 10;
    int D = 2 * radius;
    double length_scale = static_cast<double>(D);
    double nu = u_inflow * length_scale / Re;
    double tau = 0.5 + 3.0 * nu;

    int cx_cyl = NX / 4;
    int cy_cyl = NY / 2;

    int num_steps = (steps > 0) ? steps
        : std::max(4000, static_cast<int>(10.0 * NX / u_inflow));
    int save_interval = num_steps / 50;

    return {tau, u_inflow, num_steps, save_interval, length_scale,
            cx_cyl, cy_cyl, radius, T_wall, 1.0, false};
}

int main(int argc, char* argv[]) {
    std::cout << "==============================================" << std::endl;
    std::cout << " LBM-2D: Heated Cylinder in Crossflow" << std::endl;
    std::cout << " D2Q9 | MRT | Thermal DDF | OpenMP" << std::endl;
    std::cout << "==============================================" << std::endl;

    double Re = 100.0;
    double T_wall = 1.5;
    int steps = -1;
    bool use_buoyancy = false;

    int positional_idx = 1;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--use-les") {
            g_use_les = true;
        } else if (arg == "--buoyancy") {
            use_buoyancy = true;
        } else if (arg == "--cs" && i + 1 < argc) {
            g_cs = std::stod(argv[++i]);
        } else if (arg.find("--") != 0) {
            if (positional_idx == 1) Re = std::stod(arg);
            else if (positional_idx == 2) T_wall = std::stod(arg);
            else if (positional_idx == 3) steps = std::stoi(arg);
            ++positional_idx;
        }
    }

    auto params = compute_params(Re, T_wall, steps);

    std::cout << "  Re = " << Re << std::endl;
    std::cout << "  T_wall = " << T_wall << ", T_inf = 1.0" << std::endl;
    std::cout << "  Buoyancy: " << (use_buoyancy ? "ON" : "OFF") << std::endl;
    std::cout << "  tau = " << params.tau << std::endl;
    std::cout << "==============================================" << std::endl;

    LBMCapabilities system;
    system.use_thermal = true;
    system.T_ref = 1.0;
    system.alpha = params.tau > 0.0 ? (params.tau - 0.5) / 3.0 / 0.71 : 0.1667;
    system.omega_k = 1.0 / (0.5 + 3.0 * system.alpha);
    system.beta = use_buoyancy ? 0.001 : 0.0;  // thermal expansion (Boussinesq)
    system.g_buoyancy = use_buoyancy ? -1e-5 : 0.0;  // gravitational accel (downward)

    int cx_cyl = params.cx_cyl;
    int cy_cyl = params.cy_cyl;
    int radius = params.radius;
    place_cylinder(system, cx_cyl, cy_cyl, radius);

    // Initialize with equilibrium (uniform velocity + temperature)
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {
            int idx = node_index(x, y);
            double rho = 1.0;
            double u = params.u_inflow;
            double v = 0.0;
            double T = 1.0;  // T_inf
            for (int i = 0; i < 9; ++i) {
                system.f[idx * 9 + i] = compute_equilibrium(i, rho, u, v);
                system.g_thermal[idx * 9 + i] = thermal_equilibrium(i, T, u, v);
            }
        }
    }

    // Set thermal boundary at cylinder wall
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {
            int idx = node_index(x, y);
            if (system.obstacle[idx]) {
                // Isothermal wall: set g_i = g_i^eq(T_wall, 0, 0)
                apply_thermal_isothermal(&system.g_thermal[idx * 9], T_wall);
            }
        }
    }

    // Run simulation
    std::string outdir = "output/heated_cylinder_re" + std::to_string(static_cast<int>(Re))
                        + "_tw" + std::to_string(static_cast<int>(T_wall * 100));
    std::filesystem::create_directories(outdir);

    // Write meta
    std::ofstream meta(outdir + "/meta.json");
    meta << "{\"case\":\"heated_cylinder\",\"Re\":" << Re
         << ",\"T_wall\":" << T_wall
         << ",\"T_inf\":1.0,\"u_inflow\":" << params.u_inflow
         << ",\"D\":" << 2 * radius << ",\"Pr\":0.71"
         << ",\"nx\":" << NX << ",\"ny\":" << NY << "}" << std::endl;
    meta.close();

    for (int step = 0; step <= params.num_steps; ++step) {
        execute_time_step(system, params.tau, params.u_inflow);

        // Re-impose thermal wall boundary each step
        #pragma omp parallel for collapse(2)
        for (int y = 0; y < NY; ++y) {
            for (int x = 0; x < NX; ++x) {
                int idx = node_index(x, y);
                if (system.obstacle[idx]) {
                    apply_thermal_isothermal(&system.g_thermal[idx * 9], T_wall);
                }
            }
        }

        if (step % params.save_interval == 0) {
            // Compute local Nusselt number along cylinder surface
            double Nu_avg = 0.0;
            int n_surface = 0;

            #pragma omp parallel for collapse(2) reduction(+:Nu_avg, n_surface)
            for (int y = 0; y < NY; ++y) {
                for (int x = 0; x < NX; ++x) {
                    int idx = node_index(x, y);
                    if (!system.obstacle[idx]) continue;
                    // Check if adjacent to fluid node
                    for (int i = 0; i < 9; ++i) {
                        int nx = x + cx[i], ny = y + cy[i];
                        if (nx < 0 || nx >= NX || ny < 0 || ny >= NY) continue;
                        int nidx = node_index(nx, ny);
                        if (!system.obstacle[nidx]) {
                            // Compute temperature at fluid node
                            double T_fluid;
                            compute_temperature(&system.g_thermal[nidx * 9], T_fluid);
                            // Negative gradient (heat flows from wall to fluid)
                            double dTdr = (T_fluid - T_wall) / 1.0;  // outward normal
                            double Nu = compute_nusselt(dTdr, 2.0 * radius, T_wall, 1.0);
                            Nu_avg += Nu;
                            ++n_surface;
                            break;
                        }
                    }
                }
            }

            if (n_surface > 0) Nu_avg /= n_surface;

            std::cout << "  Step " << step << " / " << params.num_steps
                      << " | Nu_avg = " << Nu_avg << std::endl;

            // Save frame with thermal data
            save_json_frame_thermal(system, step, outdir, T_wall);
        }
    }

    std::cout << "  Done. Output: " << outdir << std::endl;
    return 0;
}
