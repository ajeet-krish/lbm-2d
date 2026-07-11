#include "lbm.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include <cstdlib>
#include <vector>
#include <utility>

// ==========================================================================
// LBM-2D: Urban Canyon Flow
// ==========================================================================
// Usage (side-view canyon):
//   ./build/LBM_UrbanCanyon --mode side --ar 0.5
//   ./build/LBM_UrbanCanyon --mode side --ar 0.3 --Re 100
//   Three Oke 1988 regimes: ar=0.3 (isolated), 0.5 (wake interference),
//                           0.8 (skimming flow)
//
// Usage (top-down plan view):
//   ./build/LBM_UrbanCanyon                          (default: top-down)
//   ./build/LBM_UrbanCanyon 200
//
// ==========================================================================

enum class UrbanMode { SIDE, TOPDOWN };

struct UrbanParams {
    double tau;
    double u_ref;
    int num_steps;
    int save_interval;
    double length_scale;
    // Common
    UrbanMode mode;
    // Side view
    double aspect_ratio;  // H/W
    int bldg_height;
    int bldg_width;
    int canyon_width;
    int bldg1_x0, bldg1_x1;
    int bldg2_x0, bldg2_x1;
    // Top-down view
    int w_bldg;
    int l_bldg;
    int bldg1_x0_td;
    int bldg2_x0_td;
    int bldg_y0, bldg_y1;
};

UrbanParams compute_side_params(double Re, double ar) {
    double u_ref = 0.1;
    int H = NY / 5;                        // building height
    int W = static_cast<int>(static_cast<double>(H) / ar);
    if (W < 10) W = 10;
    int bldg_wid = H;                      // building width = height

    double length_scale = static_cast<double>(H);
    double nu = u_ref * length_scale / Re;
    double tau = 0.5 + 3.0 * nu;

    int b1_x0 = NX / 4 - bldg_wid;
    int b1_x1 = b1_x0 + bldg_wid;
    int b2_x0 = b1_x1 + W;
    int b2_x1 = b2_x0 + bldg_wid;

    int num_steps = std::max(20000, static_cast<int>(20.0 * NX / u_ref));
    int save_interval = num_steps / 50;

    return {tau, u_ref, num_steps, save_interval, length_scale,
            UrbanMode::SIDE, ar, H, bldg_wid, W, b1_x0, b1_x1, b2_x0, b2_x1,
            0, 0, 0, 0, 0, 0};
}

UrbanParams compute_topdown_params(double Re) {
    int w_bldg = 30;
    int l_bldg = NY / 3;
    int canyon = w_bldg;
    double u_ref = 0.1;
    double length_scale = static_cast<double>(w_bldg);
    double nu = u_ref * length_scale / Re;
    double tau = 0.5 + 3.0 * nu;

    int b1_x0 = NX * 3 / 8 - w_bldg / 2;
    int b2_x0 = b1_x0 + w_bldg + canyon;
    int bldg_y0 = NY / 2 - l_bldg / 2;
    int bldg_y1 = bldg_y0 + l_bldg;

    int num_steps = std::max(20000, static_cast<int>(15.0 * NX / u_ref));
    int save_interval = num_steps / 50;

    return {tau, u_ref, num_steps, save_interval, length_scale,
            UrbanMode::TOPDOWN, 0.0, 0, 0, 0, 0, 0, 0, 0,
            w_bldg, l_bldg, b1_x0, b2_x0, bldg_y0, bldg_y1};
}

void place_side_obstacles(LBMCapabilities& system, const UrbanParams& p) {
    for (int y = 0; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {
            // Bottom wall
            if (y == 0) {
                system.obstacle[node_index(x, y)] = true;
            }
            // Top wall (free-slip -- treat as obstacle for no-slip)
            if (y == NY - 1) {
                system.obstacle[node_index(x, y)] = true;
            }
            // Building 1 (upstream)
            if (x >= p.bldg1_x0 && x < p.bldg1_x1 && y < p.bldg_height) {
                system.obstacle[node_index(x, y)] = true;
            }
            // Building 2 (downstream)
            if (x >= p.bldg2_x0 && x < p.bldg2_x1 && y < p.bldg_height) {
                system.obstacle[node_index(x, y)] = true;
            }
        }
    }
}

void place_topdown_obstacles(LBMCapabilities& system, const UrbanParams& p) {
    for (int y = 0; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {
            if (y == 0 || y == NY - 1) {
                system.obstacle[node_index(x, y)] = true;
            }
            if (x >= p.bldg1_x0_td && x < p.bldg1_x0_td + p.w_bldg
                && y >= p.bldg_y0 && y < p.bldg_y1) {
                system.obstacle[node_index(x, y)] = true;
            }
            if (x >= p.bldg2_x0_td && x < p.bldg2_x0_td + p.w_bldg
                && y >= p.bldg_y0 && y < p.bldg_y1) {
                system.obstacle[node_index(x, y)] = true;
            }
        }
    }
}

int main(int argc, char* argv[]) {
    UrbanMode mode = UrbanMode::TOPDOWN;
    double Re = 100.0;
    double aspect_ratio = 0.5;
    int steps = -1;
    bool save_vtk = false;

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--mode") {
            if (i + 1 < argc) {
                std::string m = argv[i + 1];
                if (m == "side") mode = UrbanMode::SIDE;
                else if (m == "topdown") mode = UrbanMode::TOPDOWN;
                else {
                    std::cerr << "Unknown mode: " << m << " (use side|topdown)" << std::endl;
                    return 1;
                }
                ++i;
            }
        } else if (arg == "--ar") {
            if (i + 1 < argc) {
                aspect_ratio = std::stod(argv[i + 1]);
                ++i;
            }
        } else if (arg == "--vtk") {
            save_vtk = true;
        } else if (i == 1 && arg.find("--") != 0) {
            Re = std::stod(arg);
        } else if (i == 2 && arg.find("--") != 0) {
            steps = std::stoi(arg);
        }
    }

    g_case = CaseType::URBAN_CANYON;

    // Override grid for urban canyon
    NX = 600;
    NY = 300;

    UrbanParams params;
    if (mode == UrbanMode::SIDE) {
        params = compute_side_params(Re, aspect_ratio);
    } else {
        params = compute_topdown_params(Re);
    }
    params.mode = mode;

    LBMCapabilities system;

    // Build obstacle map
    if (mode == UrbanMode::SIDE) {
        place_side_obstacles(system, params);
    } else {
        place_topdown_obstacles(system, params);
    }

    // Initialize with equilibrium
    for (int n = 0; n < NX * NY; ++n) {
        double* f_node = &system.f[n * 9];
        for (int i = 0; i < 9; ++i) {
            f_node[i] = compute_equilibrium(i, 1.0, params.u_ref, 0.0);
        }
    }

    // Output directory
    std::string mode_str = (mode == UrbanMode::SIDE) ? "side" : "topdown";
    std::string ar_str = "";
    if (mode == UrbanMode::SIDE) {
        int ar_int = static_cast<int>(aspect_ratio * 10.0 + 0.5);
        ar_str = "_ar" + std::to_string(ar_int);
    }
    std::string subdir = "output/urban_" + mode_str + ar_str
                         + "_re" + std::to_string(static_cast<int>(Re));
    std::string mkdir_cmd = "mkdir -p " + subdir + "/frames";
    ::system(mkdir_cmd.c_str());

    save_meta_json(subdir, Re, params.tau, params.u_ref,
                   params.length_scale, "urban-" + mode_str, NX, NY);

    std::cout << "Mode: " << mode_str
              << "  Re = " << Re
              << "  tau = " << params.tau
              << "  steps = " << params.num_steps
              << "  u_ref = " << params.u_ref;
    if (mode == UrbanMode::SIDE) {
        std::cout << "  H/W = " << params.aspect_ratio
                  << "  H = " << params.bldg_height
                  << "  W = " << params.canyon_width;
    }
    std::cout << std::endl;

    for (int step = 0; step <= params.num_steps; ++step) {
        execute_time_step(system, params.tau, params.u_ref);

        double fx_total = 0.0, fy_total = 0.0;
        for (int n = 0; n < NX * NY; ++n) {
            fx_total += system.fx_cyl[n];
            fy_total += system.fy_cyl[n];
        }

        save_forces_jsonl(subdir, step, fx_total, fy_total);

        if (step % params.save_interval == 0) {
            save_json_frame(system, step, subdir);
            if (save_vtk) {
                save_vtk_frame(system, step, subdir);
            }
        }

        if (step % 2000 == 0) {
            std::cout << "  step " << std::setw(6) << step
                      << "  Fx = " << std::fixed << std::setprecision(4) << fx_total
                      << std::endl;
        }
    }

    std::cout << "==============================================" << std::endl;
    std::cout << " Urban canyon simulation complete." << std::endl;
    std::cout << "  Mode: " << mode_str << std::endl;
    std::cout << "  Re = " << Re << std::endl;
    if (mode == UrbanMode::SIDE) {
        std::cout << "  H/W = " << params.aspect_ratio << std::endl;
    }
    std::cout << "  Steps = " << params.num_steps << std::endl;
    std::cout << "==============================================" << std::endl;

    return 0;
}
