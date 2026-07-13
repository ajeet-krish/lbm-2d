#include "lbm.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include <filesystem>
#include <vector>
#include <utility>

// ==========================================================================
// LBM-2D: Urban Canyon Flow
// ==========================================================================
// Side-view canyon (Re=100):
//   ./build/LBM_UrbanCanyon --mode side --ar 0.3      (2 buildings, H/W=0.3)
//   ./build/LBM_UrbanCanyon --mode side --ar 0.5      (2 buildings, H/W=0.5)
//   ./build/LBM_UrbanCanyon --mode side --ar 0.8      (2 buildings, H/W=0.8)
//   ./build/LBM_UrbanCanyon --mode side --ar 0.6 --nb 3  (3 buildings, denser
//           downtown network; H/W=0.6)
//   Oke 1988 regimes: ar=0.3 isolated, 0.5 wake interference, 0.8 skimming.
//
// Top-down plan view (Re=100):
//   ./build/LBM_UrbanCanyon --mode topdown            (3 vertical buildings)
//   ./build/LBM_UrbanCanyon --mode topdown --orient horizontal  (3 long
//           buildings; wind funneled along the pedestrian, orifice-like)
// ==========================================================================

enum class UrbanMode { SIDE, TOPDOWN };
enum class TDOrient { VERTICAL, HORIZONTAL };

struct UrbanParams {
    double tau;
    double u_ref;
    int num_steps;
    int save_interval;
    double length_scale;
    UrbanMode mode;
    TDOrient orient;
    double aspect_ratio;
    int n_bldg;
    int bldg_height;
    int bldg_width;
    int canyon_width;
    int bldg1_x0, bldg1_x1;
    int bldg2_x0, bldg2_x1;
    int bldg3_x0, bldg3_x1;
    int w_bldg;
    int l_bldg;
    int bldg1_x0_td;
    int bldg2_x0_td;
    int bldg3_x0_td;
    int bldg_y0, bldg_y1;
};

// Place n buildings of width bw with equal canyons (n-1 gaps) centred.
// Returns bounding x of buildings 1..3 (b3 only valid when n >= 3).
void layout_side(int n, int bw, int W, int& b1_0, int& b1_1,
                 int& b2_0, int& b3_0, int& b3_1) {
    int total = n * bw + (n - 1) * W;
    int start = (NX - total) / 2;
    if (start < 1) start = 1;
    b1_0 = start;
    b1_1 = b1_0 + bw;
    b2_0 = b1_1 + W;
    if (n >= 3) {
        b3_0 = b2_0 + bw + W;
        b3_1 = b3_0 + bw;
    } else {
        b3_0 = b2_0;   // unused for n == 2
        b3_1 = b2_0;
    }
}

UrbanParams compute_side_params(double Re, double ar, int n_bldg) {
    double u_ref = 0.1;
    int H = NY / 5;                        // building height
    int W = static_cast<int>(static_cast<double>(H) / ar);
    if (W < 10) W = 10;
    int bldg_wid = static_cast<int>(1.2 * H);  // building width = 1.2 * height

    // Ensure n buildings + (n-1) canyons fit within NX with margin
    int total_needed = n_bldg * bldg_wid + (n_bldg - 1) * W;
    if (total_needed > NX - 20) {
        bldg_wid = (NX - 20 - (n_bldg - 1) * W) / n_bldg;
        if (bldg_wid < 20) bldg_wid = 20;
    }

    double length_scale = static_cast<double>(H);
    double nu = u_ref * length_scale / Re;
    double tau = 0.5 + 3.0 * nu;

    int b1_0, b1_1, b2_0, b3_0, b3_1;
    layout_side(n_bldg, bldg_wid, W, b1_0, b1_1, b2_0, b3_0, b3_1);

    int num_steps = std::max(20000, static_cast<int>(20.0 * NX / u_ref));
    int save_interval = num_steps / 50;

    return {tau, u_ref, num_steps, save_interval, length_scale,
            UrbanMode::SIDE, TDOrient::VERTICAL, ar, n_bldg, H, bldg_wid, W,
            b1_0, b1_1, b2_0, b2_0 + bldg_wid, b3_0, b3_1,
            0, 0, 0, 0, 0, 0, 0};
}

UrbanParams compute_topdown_params(double Re, TDOrient orient) {
    int w_bldg = 80;
    double u_ref = 0.1;
    double length_scale = static_cast<double>(w_bldg);
    double nu = u_ref * length_scale / Re;
    double tau = 0.5 + 3.0 * nu;

    int b1_x0, b2_x0, b3_x0, bldg_y0, bldg_y1, l_bldg;
    if (orient == TDOrient::VERTICAL) {
        // 3 tall buildings (long in y), wide street spacing
        int canyon = 2 * w_bldg;
        l_bldg = NY / 2;
        b1_x0 = NX * 3 / 8 - w_bldg / 2;
        b2_x0 = b1_x0 + w_bldg + canyon;
        b3_x0 = b2_x0 + w_bldg + canyon;
        bldg_y0 = NY / 2 - l_bldg / 2;
        bldg_y1 = bldg_y0 + l_bldg;
    } else {
        // 3 long buildings (long in x), short in y -> wind funneled through gaps
        int h_bldg = NY / 8;
        int gap = 2 * h_bldg;
        l_bldg = h_bldg;
        b1_x0 = NX * 3 / 8 - w_bldg / 2;
        b2_x0 = b1_x0 + w_bldg + gap;
        b3_x0 = b2_x0 + w_bldg + gap;
        bldg_y0 = NY / 2 - h_bldg / 2;
        bldg_y1 = bldg_y0 + h_bldg;
    }

    int num_steps = std::max(20000, static_cast<int>(15.0 * NX / u_ref));
    int save_interval = num_steps / 50;

    return {tau, u_ref, num_steps, save_interval, length_scale,
            UrbanMode::TOPDOWN, orient, 0.0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            w_bldg, l_bldg, b1_x0, b2_x0, b3_x0, bldg_y0, bldg_y1};
}

void place_side_obstacles(LBMCapabilities& system, const UrbanParams& p) {
    for (int y = 0; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {
            if (y == 0) system.obstacle[node_index(x, y)] = true;
            if (y == NY - 1) system.obstacle[node_index(x, y)] = true;
            if (x >= p.bldg1_x0 && x < p.bldg1_x1 && y < p.bldg_height)
                system.obstacle[node_index(x, y)] = true;
            if (x >= p.bldg2_x0 && x < p.bldg2_x1 && y < p.bldg_height)
                system.obstacle[node_index(x, y)] = true;
            if (p.n_bldg >= 3 &&
                x >= p.bldg3_x0 && x < p.bldg3_x1 && y < p.bldg_height)
                system.obstacle[node_index(x, y)] = true;
        }
    }
}

void place_topdown_obstacles(LBMCapabilities& system, const UrbanParams& p) {
    for (int y = 0; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {
            if (y == 0 || y == NY - 1) system.obstacle[node_index(x, y)] = true;
            if (x >= p.bldg1_x0_td && x < p.bldg1_x0_td + p.w_bldg
                && y >= p.bldg_y0 && y < p.bldg_y1)
                system.obstacle[node_index(x, y)] = true;
            if (x >= p.bldg2_x0_td && x < p.bldg2_x0_td + p.w_bldg
                && y >= p.bldg_y0 && y < p.bldg_y1)
                system.obstacle[node_index(x, y)] = true;
            if (x >= p.bldg3_x0_td && x < p.bldg3_x0_td + p.w_bldg
                && y >= p.bldg_y0 && y < p.bldg_y1)
                system.obstacle[node_index(x, y)] = true;
        }
    }
}

int main(int argc, char* argv[]) {
    UrbanMode mode = UrbanMode::TOPDOWN;
    TDOrient orient = TDOrient::VERTICAL;
    double Re = 100.0;
    double aspect_ratio = 0.5;
    int n_bldg = 2;
    int steps = -1;
    bool save_vtk = false;

    int positional_idx = 1;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--mode") {
            if (i + 1 < argc) {
                std::string m = argv[i + 1];
                if (m == "side") mode = UrbanMode::SIDE;
                else if (m == "topdown") mode = UrbanMode::TOPDOWN;
                else { std::cerr << "Unknown mode: " << m << std::endl; return 1; }
                ++i;
            }
        } else if (arg == "--ar") {
            if (i + 1 < argc) { aspect_ratio = std::stod(argv[i + 1]); ++i; }
        } else if (arg == "--nb") {
            if (i + 1 < argc) { n_bldg = std::stoi(argv[++i]); }
        } else if (arg == "--orient") {
            if (i + 1 < argc) {
                std::string o = argv[i + 1];
                if (o == "horizontal") orient = TDOrient::HORIZONTAL;
                else if (o == "vertical") orient = TDOrient::VERTICAL;
                else { std::cerr << "Unknown orient: " << o << std::endl; return 1; }
                ++i;
            }
        } else if (arg == "--vtk") { save_vtk = true;
        } else if (arg == "--use-les") { g_use_les = true;
        } else if (arg == "--cs" && i + 1 < argc) { g_cs = std::stod(argv[++i]);
        } else if (arg.find("--") != 0) {
            if (positional_idx == 1) Re = std::stod(arg);
            else if (positional_idx == 2) steps = std::stoi(arg);
            ++positional_idx;
        }
    }

    g_case = CaseType::URBAN_CANYON;

    NX = 900;
    NY = 400;

    UrbanParams params;
    if (mode == UrbanMode::SIDE) {
        params = compute_side_params(Re, aspect_ratio, n_bldg);
    } else {
        params = compute_topdown_params(Re, orient);
    }
    params.mode = mode;

    LBMCapabilities system;

    if (mode == UrbanMode::SIDE) {
        place_side_obstacles(system, params);
    } else {
        place_topdown_obstacles(system, params);
    }

    for (int n = 0; n < NX * NY; ++n) {
        double* f_node = &system.f[n * 9];
        for (int i = 0; i < 9; ++i) {
            f_node[i] = compute_equilibrium(i, 1.0, params.u_ref, 0.0);
        }
    }

    std::string mode_str = (mode == UrbanMode::SIDE) ? "side" : "topdown";
    std::string ar_str = "";
    std::string orient_str = "";
    if (mode == UrbanMode::SIDE) {
        int ar_int = static_cast<int>(aspect_ratio * 10.0 + 0.5);
        ar_str = "_ar" + std::to_string(ar_int);
        if (n_bldg >= 3) ar_str += "_3b";
    } else if (orient == TDOrient::HORIZONTAL) {
        orient_str = "_h";
    }
    std::string subdir = "output/urban_" + mode_str + ar_str + orient_str
                         + "_re" + std::to_string(static_cast<int>(Re));
    std::filesystem::create_directories(subdir + "/frames");

    save_meta_json(subdir, Re, params.tau, params.u_ref,
                   params.length_scale, "urban-" + mode_str, NX, NY);

    std::cout << "Mode: " << mode_str << orient_str
              << "  Re = " << Re
              << "  tau = " << params.tau
              << "  steps = " << params.num_steps
              << "  u_ref = " << params.u_ref;
    if (mode == UrbanMode::SIDE) {
        std::cout << "  H/W = " << params.aspect_ratio
                  << "  n_bldg = " << params.n_bldg
                  << "  H = " << params.bldg_height
                  << "  W = " << params.canyon_width
                  << "  W_bldg = " << params.bldg_width;
    }
    std::cout << (g_use_les ? "  LES(Cs=" + std::to_string(g_cs) + ")" : "")
              << std::endl;

    for (int step = 0; step <= params.num_steps; ++step) {
        execute_time_step(system, params.tau, params.u_ref);

        double fx_total = 0.0, fy_total = 0.0;
        for (int n = 0; n < NX * NY; ++n) {
            fx_total += system.fx_body[n];
            fy_total += system.fy_body[n];
        }

        save_forces_jsonl(subdir, step, fx_total, fy_total);

        if (step % params.save_interval == 0) {
            save_json_frame(system, step, subdir);
            if (save_vtk) save_vtk_frame(system, step, subdir);
        }

        if (step % 2000 == 0) {
            std::cout << "  step " << std::setw(6) << step
                      << "  Fx = " << std::fixed << std::setprecision(4) << fx_total
                      << std::endl;
        }
    }

    std::cout << "==============================================" << std::endl;
    std::cout << " Urban canyon simulation complete." << std::endl;
    std::cout << "  Mode: " << mode_str << orient_str << std::endl;
    std::cout << "  Re = " << Re << std::endl;
    if (mode == UrbanMode::SIDE) {
        std::cout << "  H/W = " << params.aspect_ratio
                  << "  Buildings: " << params.n_bldg << std::endl;
    }
    std::cout << "  Steps = " << params.num_steps << std::endl;
    std::cout << "==============================================" << std::endl;

    return 0;
}
