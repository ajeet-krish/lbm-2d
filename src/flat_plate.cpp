#include "lbm.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include <filesystem>

// ==========================================================================
// LBM-2D: Flat Plate Boundary Layer (PRIMARY Validation Case)
// ==========================================================================
// Usage:
//   ./build/LBM_FlatPlate 1000 0              (Re=1000, AoA=0)
//   ./build/LBM_FlatPlate 1000 10             (Re=1000, AoA=10 deg)
//   ./build/LBM_FlatPlate 1000 -5             (Re=1000, AoA=-5 deg)
//   ./build/LBM_FlatPlate 2000 0 --chord 200  (Re=2000, custom chord)
//
// A 2-cell thick flat plate at specified chord Reynolds number and angle
// of attack. This is the primary validation case because:
//   - Geometrically exact at any resolution (no polygon approximation)
//   - Well-known analytical solutions (Blasius boundary layer)
//   - Supports full AoA sweep for drag polar generation
//
// Validation:
//   - Cf_x = 0.664 / sqrt(Re_x) (Blasius, laminar)
//   - Cd = 1.328 / sqrt(Re_L) (integrated laminar flat plate drag)
//   - Cl ~ 2*pi*AoA (thin airfoil theory, small AoA)
//
// Parameters:
//   Re = u_inflow * chord / nu
//   chord = NY/2 + 5 (default, or --chord override)
//   thickness = 2 cells
// ==========================================================================

struct FlatPlateParams {
    double tau;
    double u_inflow;
    double chord;
    int num_steps;
    int save_interval;
    double length_scale;
    double aoa_deg;
    int cx, cy;
};

FlatPlateParams compute_params(double Re, double chord, double aoa_deg, int steps = -1) {
    double u_inflow = 0.1;
    double nu = u_inflow * chord / Re;
    double tau = 0.5 + 3.0 * nu;

    int cx = NX / 4;
    int cy = NY / 2;

    int num_steps = (steps > 0) ? steps
        : std::max(4000, static_cast<int>(10.0 * NX / u_inflow));
    int save_interval = num_steps / 50;

    return {tau, u_inflow, chord, num_steps, save_interval, chord, aoa_deg, cx, cy};
}

int main(int argc, char* argv[]) {
    std::cout << "==============================================" << std::endl;
    std::cout << " LBM-2D: Flat Plate Boundary Layer" << std::endl;
    std::cout << " D2Q9 | MRT | OpenMP | Cache-Optimized" << std::endl;
    std::cout << "==============================================" << std::endl;

    double Re = 1000.0;
    double aoa_deg = 0.0;
    double chord = NY / 2.0 + 5.0;
    int steps = -1;

    int positional_idx = 1;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--chord" && i + 1 < argc) { chord = std::stod(argv[++i]);
        } else if (arg == "--use-les") { g_use_les = true;
        } else if (arg == "--cs" && i + 1 < argc) { g_cs = std::stod(argv[++i]);
        } else if (arg.find("--") != 0) {
            if (positional_idx == 1) Re = std::stod(arg);
            else if (positional_idx == 2) aoa_deg = std::stod(arg);
            else if (positional_idx == 3) steps = std::stoi(arg);
            ++positional_idx;
        }
    }

    g_case = CaseType::FLAT_PLATE;

    auto params = compute_params(Re, chord, aoa_deg, steps);
    LBMCapabilities system;

    double half_thickness = 1.0;
    double rad = -aoa_deg * M_PI / 180.0;
    double cos_a = std::cos(rad), sin_a = std::sin(rad);

    double hx = chord / 2.0, hy = half_thickness / 2.0;
    double corners[][2] = {
        {-hx, -hy}, {hx, -hy}, {hx, hy}, {-hx, hy}
    };
    std::vector<std::pair<double,double>> plate_poly;
    for (auto& c : corners) {
        double xr = c[0] * cos_a - c[1] * sin_a + params.cx;
        double yr = c[0] * sin_a + c[1] * cos_a + params.cy;
        plate_poly.push_back({xr, yr});
    }

    place_polygon(system, plate_poly);

    for (int n = 0; n < NX * NY; ++n) {
        double* f_node = &system.f[n * 9];
        for (int i = 0; i < 9; ++i) {
            f_node[i] = compute_equilibrium(i, 1.0, params.u_inflow, 0.0);
        }
    }

    std::string aoa_str = (aoa_deg >= 0 ? "aoa" : "aoa-")
                          + std::to_string(static_cast<int>(std::abs(aoa_deg)));
    std::string subdir = "output/flatplate/re" + std::to_string(static_cast<int>(Re))
                         + "_" + aoa_str;
    std::filesystem::create_directories(subdir + "/frames");

    save_meta_json(subdir, Re, params.tau, params.u_inflow,
                   params.length_scale, "flat-plate", NX, NY);

    std::cout << "Re = " << Re
              << "  AoA = " << aoa_deg << " deg"
              << "  chord = " << chord
              << "  tau = " << params.tau
              << "  steps = " << params.num_steps
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
    double cf_lam = 1.328 / std::sqrt(Re);

    std::cout << "==============================================" << std::endl;
    std::cout << " RESULTS" << std::endl;
    std::cout << "  Mean Cd = " << cd_mean << std::endl;
    std::cout << "  Cl amplitude = " << cl_amp << std::endl;
    std::cout << "  Blasius Cf (laminar) = " << cf_lam << std::endl;
    std::cout << "  Cd/Cf_lam ratio = " << (cf_lam > 0 ? cd_mean / cf_lam : 0) << std::endl;
    std::cout << "  Steps = " << params.num_steps << std::endl;
    std::cout << "==============================================" << std::endl;

    return 0;
}
