#include "lbm.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include <filesystem>
#include <vector>

// ==========================================================================
// AK-Vortex: Orifice Plate (single and multi-stage)
// ==========================================================================
// Usage:
//   ./build/LBM_OrificePlate 100 1p1h          (single plate, single hole)
//   ./build/LBM_OrificePlate 100 1p3h          (single plate, 3 holes)
//   ./build/LBM_OrificePlate 100 2p            (2 plates, staggered holes)
//   ./build/LBM_OrificePlate 100 3p 50000      (3 plates, staggered, 50000 steps)
//   ./build/LBM_OrificePlate 100 3p --hole-w 30
//
// Four configurations:
//   1p1h : one plate with a single centered opening (classic ISO 5167 orifice)
//   1p3h : one plate with three evenly spaced openings (perforated baffle)
//   2p   : two plates, one hole each at contrasting heights (serpentine)
//   3p   : three plates, one hole each at contrasting heights (serpentine)
//
// Industrial relevance:
//   - Orifice plates for flow metering (ISO 5167)
//   - Labyrinth seals in turbomachinery
//   - Muffler/silencer baffle plates
//   - Heat exchanger baffles for cross-flow
//
// Validation:
//   - Single orifice: Cd = Q / (A * sqrt(2*Delta_p/rho))
//   - Multi-stage: total pressure drop = sum of individual drops
//   - For laminar flow at low Re: Delta_p ~ mu * Q (linear regime)
// ==========================================================================

struct OrificeParams {
    double tau;
    double u_inflow;
    int num_steps;
    int save_interval;
    double length_scale;
    std::string config;
    int hole_width;
    int plate_thickness;
    std::vector<int> plate_cx;                // x-position of each plate
    std::vector<std::vector<int>> holes_cy;   // y-centers of holes per plate
};

OrificeParams compute_params(double Re, const std::string& config,
                             int steps = -1, int hole_w = -1) {
    // Low inflow velocity keeps the accelerated jet through a single narrow
    // opening below the LBM stability (Mach) limit. A single NY/8 hole
    // accelerates the flow ~8x; at u_inflow=0.025 the jet peaks near 0.2 lu/ts.
    double u_inflow = 0.025;
    int hole_width = (hole_w > 0) ? hole_w : NY / 8;
    double length_scale = static_cast<double>(NY);
    double nu = u_inflow * length_scale / Re;
    double tau = 0.5 + 3.0 * nu;

    int plate_thickness = 2;

    std::vector<int> plate_cx;
    std::vector<std::vector<int>> holes_cy;

    if (config == "1p1h") {
        plate_cx.push_back(NX / 2);
        holes_cy.push_back({NY / 2});
    } else if (config == "1p3h") {
        plate_cx.push_back(NX / 2);
        holes_cy.push_back({NY / 4, NY / 2, NY * 3 / 4});
    } else if (config == "2p") {
        int start_x = NX / 3;
        int spacing = NX / 3;
        for (int i = 0; i < 2; ++i) {
            plate_cx.push_back(start_x + i * spacing);
            holes_cy.push_back({(i % 2 == 0) ? NY * 3 / 4 : NY / 4});
        }
    } else {  // "3p"
        int start_x = NX / 4;
        int spacing = NX / 4;
        for (int i = 0; i < 3; ++i) {
            plate_cx.push_back(start_x + i * spacing);
            holes_cy.push_back({(i % 2 == 0) ? NY * 3 / 4 : NY / 4});
        }
    }

    int num_steps = (steps > 0) ? steps : 60000;
    int save_interval = num_steps / 50;

    return {tau, u_inflow, num_steps, save_interval, length_scale,
            config, hole_width, plate_thickness, plate_cx, holes_cy};
}

void place_orifice_plates(LBMCapabilities& system, const OrificeParams& p) {
    for (size_t pi = 0; pi < p.plate_cx.size(); ++pi) {
        int cx = p.plate_cx[pi];
        for (int dx = 0; dx < p.plate_thickness; ++dx) {
            int x = cx + dx;
            if (x < 0 || x >= NX) continue;
            for (int y = 0; y < NY; ++y) {
                bool in_hole = false;
                for (int hcy : p.holes_cy[pi]) {
                    if (y >= hcy - p.hole_width / 2 &&
                        y <= hcy + p.hole_width / 2) {
                        in_hole = true;
                        break;
                    }
                }
                if (in_hole) continue;
                system.obstacle[node_index(x, y)] = true;
            }
        }
    }
}

int main(int argc, char* argv[]) {
    std::cout << "==============================================" << std::endl;
    std::cout << " AK-Vortex: Orifice Plate" << std::endl;
    std::cout << " D2Q9 | MRT | OpenMP | Cache-Optimized" << std::endl;
    std::cout << "==============================================" << std::endl;

    double Re = 100.0;
    std::string config = "3p";
    int steps = -1;
    int hole_w = -1;

    int positional_idx = 1;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--use-les") { g_use_les = true;
        } else if (arg == "--cs" && i + 1 < argc) { g_cs = std::stod(argv[++i]);
        } else if (arg == "--hole-w" && i + 1 < argc) { hole_w = std::stoi(argv[++i]);
        } else if (arg.find("--") != 0) {
            if (positional_idx == 1) Re = std::stod(arg);
            else if (positional_idx == 2) config = arg;
            else if (positional_idx == 3) steps = std::stoi(arg);
            ++positional_idx;
        }
    }

    if (config != "1p1h" && config != "1p3h" &&
        config != "2p" && config != "3p") {
        std::cerr << "Invalid config '" << config
                  << "'. Use one of: 1p1h, 1p3h, 2p, 3p" << std::endl;
        return 1;
    }

    g_case = CaseType::ORIFICE_PLATE;

    // Enable Smagorinsky LES for stability of the accelerated jet shear layers.
    g_use_les = true;

    auto params = compute_params(Re, config, steps, hole_w);
    LBMCapabilities system;

    place_orifice_plates(system, params);

    for (int n = 0; n < NX * NY; ++n) {
        double* f_node = &system.f[n * 9];
        for (int i = 0; i < 9; ++i) {
            f_node[i] = compute_equilibrium(i, 1.0, params.u_inflow, 0.0);
        }
    }

    std::string subdir = "output/orifice_plate/re"
                         + std::to_string(static_cast<int>(Re))
                         + "_" + config;
    std::filesystem::create_directories(subdir + "/frames");

    save_meta_json(subdir, Re, params.tau, params.u_inflow,
                   params.length_scale, "orifice-plate", NX, NY);

    std::cout << "Re = " << Re
              << "  config = " << config
              << "  tau = " << params.tau
              << "  steps = " << params.num_steps
              << "  hole_width = " << params.hole_width
              << (g_use_les ? "  LES(Cs=" + std::to_string(g_cs) + ")" : "")
              << std::endl;

    for (size_t pi = 0; pi < params.plate_cx.size(); ++pi) {
        std::cout << "  Plate " << (pi + 1) << ": x=" << params.plate_cx[pi]
                  << "  holes_cy=";
        for (int hcy : params.holes_cy[pi]) std::cout << hcy << " ";
        std::cout << std::endl;
    }

    for (int step = 0; step <= params.num_steps; ++step) {
        execute_time_step(system, params.tau, params.u_inflow);

        double fx_total = 0.0, fy_total = 0.0;
        for (int n = 0; n < NX * NY; ++n) {
            fx_total += system.fx_body[n];
            fy_total += system.fy_body[n];
        }

        save_forces_jsonl(subdir, step, fx_total, fy_total);

        if (step % params.save_interval == 0) {
            save_json_frame(system, step, subdir);
        }

        if (step % 500 == 0) {
            std::cout << "  step " << std::setw(6) << step
                      << "  Fx = " << std::fixed << std::setprecision(4) << fx_total
                      << "  Fy = " << std::fixed << std::setprecision(4) << fy_total
                      << std::endl;
        }
    }

    std::cout << "==============================================" << std::endl;
    std::cout << " Orifice plate simulation complete." << std::endl;
    std::cout << "  Re = " << Re << "  config = " << config << std::endl;
    std::cout << "  Steps = " << params.num_steps << std::endl;
    std::cout << "==============================================" << std::endl;

    return 0;
}
