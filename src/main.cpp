#include "lbm.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <string>
#include <random>

// ==========================================================================
// LBM-2D Solver Entry Point
// ==========================================================================
// Usage:
//   ./LBM_Engine                      (default: Re_D=100, 10000 steps)
//   ./LBM_Engine <Re_D> <steps>       (e.g. ./LBM_Engine 200 15000)
//
// The Reynolds number Re_D = u_inflow * D / nu is based on cylinder
// diameter D = 2 * radius = NY/5 = 30. The relaxation parameter tau
// follows from:
//   nu = c_s^2 * (tau - 0.5) * dt
//   With c_s^2 = 1/3, dt = 1, dx = 1:  tau = 0.5 + 3.0 * u_inflow * D / Re_D
// ==========================================================================

static constexpr double CYL_DIAMETER = 2.0 * (NY / 10);  // D = 30

struct SimParams {
    double tau;
    double u_inflow;
    int num_steps;
    int vtk_interval;
};

SimParams compute_params(double Re, int steps = -1) {
    double u_inflow = 0.1;               // lattice Mach number (< 0.3 for incompressibility)
    double nu = u_inflow * CYL_DIAMETER / Re;  // kinematic viscosity (cylinder-diameter based Re)
    double tau = 0.5 + 3.0 * nu;         // relaxation time from viscosity relation

    // Domain transit time = NX / u = 4000 steps at u=0.1
    // Run for 2-3 transit times to reach steady-state shedding
    int num_steps = (steps > 0) ? steps : std::max(4000, static_cast<int>(10.0 * NX / u_inflow));
    int vtk_interval = num_steps / 50;   // ~50 snapshots

    return {tau, u_inflow, num_steps, vtk_interval};
}

// Force accumulation across all timesteps
struct ForceHistory {
    std::vector<double> t;
    std::vector<double> cd;
    std::vector<double> cl;
};

ForceHistory run_simulation(double Re, int steps = -1) {
    auto params = compute_params(Re, steps);
    LBMCapabilities system;

    // Cylinder at x = NX/4, y = NY/2 + 1 (off-center to break symmetry)
    int cx_cyl = NX / 4;
    int cy_cyl = NY / 2 + 1;
    int radius = NY / 10;
    place_cylinder(system, cx_cyl, cy_cyl, radius);

    // Initialize with equilibrium at rho=1, u=u_inflow, v=0
    for (int n = 0; n < NX * NY; ++n) {
        double* f_node = &system.f[n * 9];
        for (int i = 0; i < 9; ++i) {
            f_node[i] = compute_equilibrium(i, 1.0, params.u_inflow, 0.0);
        }
    }

    // Small random perturbation to break symmetry and trigger vortex shedding
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> pert_dist(-1e-4, 1e-4);
    for (int x = cx_cyl + 5; x < std::min(NX, cx_cyl + 60); ++x) {
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

    ForceHistory history;

    std::cout << "Re = " << Re
              << "  tau = " << params.tau
              << "  steps = " << params.num_steps
              << "  u_in = " << params.u_inflow
              << std::endl;

    for (int step = 0; step <= params.num_steps; ++step) {
        execute_time_step(system, params.tau, params.u_inflow);

        // Compute total forces on cylinder (summed over all boundary links)
        double fx_total = 0.0, fy_total = 0.0;
        for (int n = 0; n < NX * NY; ++n) {
            fx_total += system.fx_cyl[n];
            fy_total += system.fy_cyl[n];
        }

        // Drag coefficient: Cd = 2 * Fx / (rho * u_inflow^2 * D)
        // Lift coefficient: Cl = 2 * Fy / (rho * u_inflow^2 * D)
        double cd = 2.0 * fx_total / (CYL_DIAMETER * params.u_inflow * params.u_inflow);
        double cl = 2.0 * fy_total / (CYL_DIAMETER * params.u_inflow * params.u_inflow);

        history.t.push_back(static_cast<double>(step));
        history.cd.push_back(cd);
        history.cl.push_back(cl);

        // Save VTK frames at intervals
        if (step % params.vtk_interval == 0) {
            std::string subdir = "output/re" + std::to_string(static_cast<int>(Re));
            save_vtk_frame(system, step, subdir);
        }

        if (step % 500 == 0) {
            std::cout << "  step " << std::setw(6) << step
                      << "  Cd = " << std::fixed << std::setprecision(4) << cd
                      << "  Cl = " << std::fixed << std::setprecision(4) << cl
                      << std::endl;
        }
    }

    return history;
}

int main(int argc, char* argv[]) {
    std::cout << "==============================================" << std::endl;
    std::cout << " LBM-2D: High-Performance Lattice Boltzmann CFD" << std::endl;
    std::cout << " D2Q9 | OpenMP | Cache-Optimized" << std::endl;
    std::cout << "==============================================" << std::endl;

    double Re = 100.0;
    int steps = -1;

    if (argc >= 2) Re = std::stod(argv[1]);
    if (argc >= 3) steps = std::stoi(argv[2]);

    // Create output subdirectory
    std::string subdir = "output/re" + std::to_string(static_cast<int>(Re));
    std::string mkdir_cmd = "mkdir -p " + subdir;
    system(mkdir_cmd.c_str());

    auto history = run_simulation(Re, steps);

    // Compute mean Cd and Cl amplitude (for shedding cases)
    double cd_sum = 0.0, cl_max = 0.0, cl_min = 0.0;
    int n_late = history.cd.size() * 0.5;  // use latter half for statistics
    for (int i = n_late; i < static_cast<int>(history.cd.size()); ++i) {
        cd_sum += history.cd[i];
        if (history.cl[i] > cl_max) cl_max = history.cl[i];
        if (history.cl[i] < cl_min) cl_min = history.cl[i];
    }
    double cd_mean = cd_sum / (history.cd.size() - n_late);
    double cl_amplitude = (cl_max - cl_min) / 2.0;

    std::cout << "==============================================" << std::endl;
    std::cout << " RESULTS" << std::endl;
    std::cout << "  Mean Cd = " << cd_mean << std::endl;
    std::cout << "  Cl amplitude = " << cl_amplitude << std::endl;
    std::cout << "  Steps completed = " << history.t.size() << std::endl;
    std::cout << "==============================================" << std::endl;

    return 0;
}
