#include "lbm.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include <filesystem>
#include <random>

// ==========================================================================
// AK-Vortex: Rotating Cylinder (Magnus Effect)
// ==========================================================================
// Usage:
//   ./build/LBM_RotatingCylinder 100 1.0     (Re=100, omega=1.0)
//   ./build/LBM_RotatingCylinder 200 2.0     (Re=200, omega=2.0)
//   ./build/LBM_RotatingCylinder 100 0.5 20000 (Re=100, omega=0.5, 20000 steps)
//
// A cylinder rotating about its center in a uniform flow. The rotation
// creates an asymmetric velocity distribution (Bernoulli), generating
// lift perpendicular to the flow (Magnus effect).
//
// Boundary condition: Ladd (1994) moving boundary with tangential velocity
// on the cylinder surface to simulate rotation.
//
// Parameters:
//   Re = u_inflow * D / nu,  D = 2 * NY/10 = 60
//   omega = spin ratio (dimensionless, S = u_surface / u_inflow)
//   Lattice angular velocity: omega_lat = omega * u_inflow / R
//   Surface velocity at angle theta: u_theta = omega_lat * R
//
// The lift coefficient for a rotating cylinder in inviscid flow:
//   Cl = 2 * pi * S (Kutta-Joukowski theorem, upper bound)
// ==========================================================================

struct RotatingParams {
    double tau;
    double u_inflow;
    int num_steps;
    int save_interval;
    double length_scale;
    int cx_cyl, cy_cyl, radius;
    double omega;
    double spin_ratio;
};

RotatingParams compute_params(double Re, double omega, int steps = -1) {
    double u_inflow = 0.1;
    int radius = NY / 10;
    int D = 2 * radius;
    double length_scale = static_cast<double>(D);
    double nu = u_inflow * length_scale / Re;
    double tau = 0.5 + 3.0 * nu;

    int cx_cyl = NX / 4;
    int cy_cyl = NY / 2;
    // omega is the dimensionless spin ratio S = u_surface / u_inflow
    double spin_ratio = omega;

    int num_steps = (steps > 0) ? steps
        : std::max(4000, static_cast<int>(10.0 * NX / u_inflow));
    int save_interval = num_steps / 50;

    return {tau, u_inflow, num_steps, save_interval, length_scale,
            cx_cyl, cy_cyl, radius, omega, spin_ratio};
}

int main(int argc, char* argv[]) {
    std::cout << "==============================================" << std::endl;
    std::cout << " AK-Vortex: Rotating Cylinder (Magnus Effect)" << std::endl;
    std::cout << " D2Q9 | MRT | OpenMP | Cache-Optimized" << std::endl;
    std::cout << " Ladd (1994) Moving Boundary" << std::endl;
    std::cout << "==============================================" << std::endl;

    double Re = 100.0;
    double omega = 1.0;
    int steps = -1;

    int positional_idx = 1;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--use-les") { g_use_les = true;
        } else if (arg == "--cs" && i + 1 < argc) { g_cs = std::stod(argv[++i]);
        } else if (arg.find("--") != 0) {
            if (positional_idx == 1) Re = std::stod(arg);
            else if (positional_idx == 2) omega = std::stod(arg);
            else if (positional_idx == 3) steps = std::stoi(arg);
            ++positional_idx;
        }
    }

    g_case = CaseType::ROTATING_CYLINDER;

    auto params = compute_params(Re, omega, steps);
    LBMCapabilities system;

    place_cylinder(system, params.cx_cyl, params.cy_cyl, params.radius);

    // Configure Ladd moving boundary
    // omega is spin ratio (user input), convert to lattice angular velocity
    // omega_lat = omega * u_inflow / R  (so spin_ratio = omega * R / u_inflow = omega)
    system.bb_geom.has_moving_wall = true;
    system.bb_geom.omega = params.omega * params.u_inflow / params.radius;
    system.bb_geom.rot_cx = static_cast<double>(params.cx_cyl);
    system.bb_geom.rot_cy = static_cast<double>(params.cy_cyl);

    // Auto-LES for high Re
    if (params.tau < 0.55 && !g_use_les) {
        g_use_les = true;
        std::cout << "  Auto-LES: tau=" << params.tau << " < 0.55, enabling Smagorinsky" << std::endl;
    }

    for (int n = 0; n < NX * NY; ++n) {
        double* f_node = &system.f[n * 9];
        for (int i = 0; i < 9; ++i) {
            f_node[i] = compute_equilibrium(i, 1.0, params.u_inflow, 0.0);
        }
    }

    // Small asymmetric perturbation to break symmetry and seed vortex shedding
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

    std::string subdir = "output/rotating_cylinder/re"
                         + std::to_string(static_cast<int>(Re))
                         + "_w" + std::to_string(static_cast<int>(omega * 10));
    std::filesystem::create_directories(subdir + "/frames");

    save_meta_json(subdir, Re, params.tau, params.u_inflow,
                   params.length_scale, "rotating-cylinder", NX, NY);

    std::cout << "Re = " << Re
              << "  omega = " << omega
              << "  spin_ratio S = " << std::fixed << std::setprecision(3) << params.spin_ratio
              << "  tau = " << params.tau
              << "  steps = " << params.num_steps
              << "  D = " << 2 * params.radius
              << "  Ladd moving boundary"
              << (g_use_les ? "  LES(Cs=" + std::to_string(g_cs) + ")" : "")
              << std::endl;

    std::cout << "  Lattice omega = " << system.bb_geom.omega
              << "  Actual S = " << (system.bb_geom.omega * params.radius / params.u_inflow)
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
    std::cout << " Rotating cylinder simulation complete." << std::endl;
    std::cout << "  Re = " << Re << "  omega = " << omega << std::endl;
    std::cout << "  Spin ratio = " << params.spin_ratio << std::endl;
    std::cout << "  Cl_kutta = " << (2.0 * M_PI * params.spin_ratio) << std::endl;
    std::cout << "  Steps = " << params.num_steps << std::endl;
    std::cout << "==============================================" << std::endl;

    return 0;
}
