#include "lbm.hpp"
#include "geometry.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include <cstdlib>
#include <vector>
#include <utility>

// ==========================================================================
// LBM-2D: Ahmed Body Flow (2D slice)
// ==========================================================================
// Usage:
//   ./build/LBM_Ahmed                              (default: Re_H=1000, slant=30 deg)
//   ./build/LBM_Ahmed 1000 25                      (Re_H=1000, slant=25 deg)
//
// 2D simplified Ahmed body with slanted rear.
// Re_H = u_inflow * H_body / nu
// Validation: Cd(alpha) trend vs Ahmed 1984 (qualitative for 2D)
//
// NOTE: The real Ahmed body flow is 3D with longitudinal vortices.
// The 2D slice captures base pressure drag trend but not the
// characteristic Cd drop at 25 degrees.
// ==========================================================================

struct AhmedParams {
    double tau;
    double u_inflow;
    int num_steps;
    int save_interval;
    int h_body;
    int l_body;
    int flat_len;
    double slant_angle_deg;
    int body_x0;
};

AhmedParams compute_params(double Re_H, double slant_deg, int steps = -1) {
    int h_body = NY / 5;                         // 64 at NY=320
    int l_body = static_cast<int>(3.5 * h_body); // ~224
    int flat_len = static_cast<int>(0.6 * l_body);
    double u_inflow = 0.1;
    double nu = u_inflow * h_body / Re_H;
    double tau = 0.5 + 3.0 * nu;

    int body_x0 = NX / 4;

    int num_steps = (steps > 0) ? steps
        : std::max(30000, static_cast<int>(20.0 * NX / u_inflow));
    int save_interval = num_steps / 50;

    return {tau, u_inflow, num_steps, save_interval,
            h_body, l_body, flat_len, slant_deg, body_x0};
}

int main(int argc, char* argv[]) {
    std::cout << "==============================================" << std::endl;
    std::cout << " LBM-2D: Ahmed Body (2D Slice)" << std::endl;
    std::cout << " D2Q9 | MRT | OpenMP | Cache-Optimized" << std::endl;
    std::cout << "==============================================" << std::endl;

    // Override grid for Ahmed body
    NX = 800;
    NY = 320;

    double Re = 1000.0;
    double slant_deg = 30.0;
    int steps = -1;
    bool save_vtk = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--vtk") {
            save_vtk = true;
        } else if (i == 1) {
            Re = std::stod(arg);
        } else if (i == 2) {
            slant_deg = std::stod(arg);
        } else if (i == 3) {
            steps = std::stoi(arg);
        }
    }

    g_case = CaseType::AHMED;

    auto params = compute_params(Re, slant_deg, steps);
    LBMCapabilities system;

    // Build Ahmed body polygon
    int bx = params.body_x0;
    int by = 0;  // body sits on ground
    int bh = params.h_body;
    int bl = params.l_body;
    int bflat = params.flat_len;
    double rad = params.slant_angle_deg * M_PI / 180.0;

    // Four vertices of the body polygon
    //  (bx, by) -- front bottom
    //  (bx, by+bh) -- front top
    //  (bx+bflat, by+bh) -- start of slant
    //  (bx+bl, by) -- rear bottom
    std::vector<std::pair<double, double>> body_poly = {
        {static_cast<double>(bx), static_cast<double>(by)},
        {static_cast<double>(bx), static_cast<double>(by + bh)},
        {static_cast<double>(bx + bflat), static_cast<double>(by + bh)},
        {static_cast<double>(bx + bl), static_cast<double>(by)}
    };

    // Place Ahmed body using polygon
    place_polygon(system, body_poly);

    // Add bottom wall after polygon (polygon only covers body, not the wall)
    for (int x = 0; x < NX; ++x) {
        system.obstacle[node_index(x, 0)] = true;
    }

    for (int n = 0; n < NX * NY; ++n) {
        double* f_node = &system.f[n * 9];
        for (int i = 0; i < 9; ++i) {
            f_node[i] = compute_equilibrium(i, 1.0, params.u_inflow, 0.0);
        }
    }

    std::string subdir = "output/ahmed_re" + std::to_string(static_cast<int>(Re))
                         + "_slant" + std::to_string(static_cast<int>(slant_deg));
    std::string mkdir_cmd = "mkdir -p " + subdir + "/frames";
    ::system(mkdir_cmd.c_str());

    save_meta_json(subdir, Re, params.tau, params.u_inflow,
                   static_cast<double>(params.h_body), "ahmed-body", NX, NY);

    std::cout << "Re_H = " << Re
              << "  tau = " << params.tau
              << "  steps = " << params.num_steps
              << "  u_inflow = " << params.u_inflow
              << "  h_body = " << params.h_body
              << "  l_body = " << params.l_body
              << "  slant = " << params.slant_angle_deg << " deg"
              << "  body at x = " << params.body_x0
              << std::endl;

    double body_height = static_cast<double>(params.h_body);

    for (int step = 0; step <= params.num_steps; ++step) {
        execute_time_step(system, params.tau, params.u_inflow);

        double fx_total = 0.0, fy_total = 0.0;
        for (int n = 0; n < NX * NY; ++n) {
            fx_total += system.fx_cyl[n];
            fy_total += system.fy_cyl[n];
        }

        double cd = 2.0 * fx_total / (body_height * params.u_inflow * params.u_inflow);
        double cl = 2.0 * fy_total / (body_height * params.u_inflow * params.u_inflow);

        save_forces_jsonl(subdir, step, cd, cl);

        if (step % params.save_interval == 0) {
            save_json_frame(system, step, subdir);
            if (save_vtk) {
                save_vtk_frame(system, step, subdir);
            }
        }

        if (step % 2000 == 0) {
            std::cout << "  step " << std::setw(6) << step
                      << "  Cd = " << std::fixed << std::setprecision(4) << cd
                      << "  Cl = " << std::fixed << std::setprecision(4) << cl
                      << std::endl;
        }
    }

    std::cout << "==============================================" << std::endl;
    std::cout << " Ahmed body simulation complete." << std::endl;
    std::cout << "  Re_H = " << Re << std::endl;
    std::cout << "  Slant = " << params.slant_angle_deg << " deg" << std::endl;
    std::cout << "  Steps = " << params.num_steps << std::endl;
    std::cout << "==============================================" << std::endl;

    return 0;
}
