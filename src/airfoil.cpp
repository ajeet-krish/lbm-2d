#include "lbm.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <string>
#include <vector>

// ==========================================================================
// LBM-2D Airfoil Analysis Entry Point
// ==========================================================================
// Usage:
//   ./LBM_Airfoil <series> <Re> <AoA> <steps>
//
// Examples:
//   ./LBM_Airfoil 0012 1000 0 30000    NACA 0012 at 0 deg, Re=1000
//   ./LBM_Airfoil 2412 2000 4 30000    NACA 2412 at 4 deg, Re=2000
//
// Grid: NX x NY, chord = NY/2 + 5 = 80, LE at (NX/4, NY/2)
// ==========================================================================

static const double CHORD = static_cast<double>(NY) / 2.0 + 5.0;  // 80

struct SimParams {
    double tau;
    double u_inflow;
    int num_steps;
    int vtk_interval;
    int report_interval;
};

SimParams compute_params(double Re, int steps) {
    double u_inflow = 0.1;
    double nu = u_inflow * CHORD / Re;
    double tau = 0.5 + 3.0 * nu;
    int num_steps = (steps > 0) ? steps : std::max(5000, static_cast<int>(8.0 * NX / u_inflow));
    int vtk_interval = std::max(1, num_steps / 50);
    int report_interval = std::max(1, num_steps / 200);
    return {tau, u_inflow, num_steps, vtk_interval, report_interval};
}

int main(int argc, char* argv[]) {
    std::cout << "==============================================" << std::endl;
    std::cout << " LBM-2D: Airfoil Analysis" << std::endl;
    std::cout << " D2Q9 | OpenMP | Cache-Optimized" << std::endl;
    std::cout << "==============================================" << std::endl;

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <series> <Re> <AoA_deg> <steps>\n"
                  << "  e.g. " << argv[0] << " 0012 1000 0 30000\n"
                  << "       " << argv[0] << " 2412 2000 4 30000\n";
        return 1;
    }

    int series = std::stoi(argv[1]);
    double Re = (argc >= 3) ? std::stod(argv[2]) : 1000.0;
    double aoa = (argc >= 4) ? std::stod(argv[3]) : 0.0;
    int steps = (argc >= 5) ? std::stoi(argv[4]) : -1;

    g_case = CaseType::CYLINDER;  // same BC: periodic y, Zou/He inlet, convective outlet

    auto params = compute_params(Re, steps);

    std::cout << "NACA " << series
              << "  Re = " << Re
              << "  AoA = " << aoa
              << "  tau = " << params.tau
              << "  steps = " << params.num_steps
              << std::endl;

    // Generate NACA airfoil
    int cx = NX / 4;     // LE at x = 100
    int cy = NY / 2;     // centered vertically
    auto poly = naca_coords(series, 200);
    transform_points(poly,
        static_cast<double>(cx),
        static_cast<double>(cy),
        CHORD,
        -aoa);   // negative: positive AoA = nose-up (LE higher than TE)

    LBMCapabilities system;
    system.obstacle.assign(NX * NY, false);
    system.fx_cyl.assign(NX * NY, 0.0);
    system.fy_cyl.assign(NX * NY, 0.0);
    place_polygon(system, poly);

    // Initialize with equilibrium at rho=1, u=u_inflow, v=0
    for (int n = 0; n < NX * NY; ++n) {
        double* f_node = &system.f[n * 9];
        for (int i = 0; i < 9; ++i) {
            f_node[i] = compute_equilibrium(i, 1.0, params.u_inflow, 0.0);
        }
    }

    // Format series as 4-digit code
    char buf[16];
    snprintf(buf, sizeof(buf), "%04d", series);
    std::string series_str(buf);

    std::string subdir = "output/airfoil/naca" + series_str
                       + "_re" + std::to_string(static_cast<int>(Re))
                       + "_aoa" + std::to_string(static_cast<int>(aoa));
    {
        std::string mkdir_cmd = "mkdir -p " + subdir;
        std::system(mkdir_cmd.c_str());
    }

    // Force history storage
    std::vector<double> cd_history, cl_history, step_history;

    for (int step = 0; step <= params.num_steps; ++step) {
        execute_time_step(system, params.tau, params.u_inflow);

        if (step % params.vtk_interval == 0) {
            save_vtk_frame(system, step, subdir);
        }

        if (step % params.report_interval == 0 && step > 0) {
            double fx_total = 0.0, fy_total = 0.0;
            for (int n = 0; n < NX * NY; ++n) {
                fx_total += system.fx_cyl[n];
                fy_total += system.fy_cyl[n];
            }
            double cd = 2.0 * fx_total / (CHORD * params.u_inflow * params.u_inflow);
            double cl = 2.0 * fy_total / (CHORD * params.u_inflow * params.u_inflow);
            cd_history.push_back(cd);
            cl_history.push_back(cl);
            step_history.push_back(step);

            std::cout << "  step " << std::setw(6) << step
                      << "  Cd = " << std::fixed << std::setprecision(5) << cd
                      << "  Cl = " << std::fixed << std::setprecision(5) << cl
                      << std::endl;
        }
    }

    // Save force history to CSV
    {
        std::string csv_file = subdir + "/forces.csv";
        std::ofstream csv(csv_file);
        csv << "step,cd,cl\n";
        for (size_t i = 0; i < step_history.size(); ++i) {
            csv << step_history[i] << "," << cd_history[i] << "," << cl_history[i] << "\n";
        }
    }

    // Compute mean Cd/Cl over last half of simulation (after flow developed)
    int n_history = static_cast<int>(cd_history.size());
    int start_mean = n_history / 2;
    double cd_mean = 0.0, cl_mean = 0.0;
    for (int i = start_mean; i < n_history; ++i) {
        cd_mean += cd_history[i];
        cl_mean += cl_history[i];
    }
    int n_mean = n_history - start_mean;
    if (n_mean > 0) {
        cd_mean /= static_cast<double>(n_mean);
        cl_mean /= static_cast<double>(n_mean);
    }

    // Final instantaneous values (last step)
    double cd_final = cd_history.empty() ? 0.0 : cd_history.back();
    double cl_final = cl_history.empty() ? 0.0 : cl_history.back();

    std::cout << "==============================================" << std::endl;
    std::cout << " RESULTS" << std::endl;
    std::cout << "  Cd (final)     = " << cd_final << std::endl;
    std::cout << "  Cd (mean)      = " << cd_mean << std::endl;
    std::cout << "  Cl (final)     = " << cl_final << std::endl;
    std::cout << "  Cl (mean)      = " << cl_mean << std::endl;
    std::cout << "==============================================" << std::endl;

    return 0;
}
