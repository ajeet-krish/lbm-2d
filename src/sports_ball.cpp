#include "lbm.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include <filesystem>
#include <vector>

// ==========================================================================
// LBM-2D: Sports-Ball Surface Roughness (Dimpled Cylinder)
// ==========================================================================
// Usage:
//   ./build/LBM_SportsBall 100            (smooth cylinder baseline, Re=100)
//   ./build/LBM_SportsBall 100 1          (dimpled cylinder, Re=100)
//   ./build/LBM_SportsBall 100 1 20000    (dimpled, 20000 steps)
//
// Studies how a low-resolution surface-roughness pattern (golf-ball style
// dimples) modifies the drag of a sphere/cylinder at modest Reynolds number.
// The cylinder body is the same diameter as the other cylinder cases; the
// "dimpled" variant adds a ring of small rectangular bumps protruding from
// the surface to emulate dimples/seams at coarse lattice resolution.
//
// Applications:
//   - Golf-ball / sports-ball drag reduction via surface roughness
//   - Low-Re separation delay (roughness can trip the boundary layer)
//   - Demonstrates that surface texture, not just shape, controls Cd
//
// Parameters:
//   Re = u_inflow * D / nu,  D = 2 * NY/10 = 60
//   bumps: nb bumps around the perimeter, each nb_w x nb_d cells
// ==========================================================================

struct SportsBallParams {
    double tau;
    double u_inflow;
    int num_steps;
    int save_interval;
    double length_scale;
    int cx, cy, radius;
    bool dimpled;
    int nb_bumps;
    int bump_w;
    int bump_d;
};

SportsBallParams compute_params(double Re, bool dimpled, int steps = -1) {
    double u_inflow = 0.1;
    int radius = NY / 10;
    int D = 2 * radius;
    double length_scale = static_cast<double>(D);
    double nu = u_inflow * length_scale / Re;
    double tau = 0.5 + 3.0 * nu;

    int num_steps = (steps > 0) ? steps
        : std::max(4000, static_cast<int>(10.0 * NX / u_inflow));
    int save_interval = num_steps / 50;

    return {tau, u_inflow, num_steps, save_interval, length_scale,
            NX / 4, NY / 2, radius, dimpled, 16, 3, 2};
}

void place_dimpled_cylinder(LBMCapabilities& system, const SportsBallParams& p) {
    // Base cylinder interior + boundary (reuse standard cylinder mask)
    for (int y = 0; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {
            int dx = x - p.cx;
            int dy = y - p.cy;
            if (dx * dx + dy * dy <= p.radius * p.radius) {
                system.obstacle[node_index(x, y)] = true;
            }
        }
    }
    if (!p.dimpled) return;
    // Protruding bumps around the perimeter
    for (int k = 0; k < p.nb_bumps; ++k) {
        double theta = 2.0 * M_PI * k / p.nb_bumps;
        int nx = static_cast<int>(std::round(std::cos(theta) * p.radius));
        int ny = static_cast<int>(std::round(std::sin(theta) * p.radius));
        int bx = p.cx + nx;
        int by = p.cy + ny;
        for (int dy = -p.bump_d; dy <= p.bump_d; ++dy) {
            for (int dx = -p.bump_w / 2; dx <= p.bump_w / 2; ++dx) {
                int x = bx + dx;
                int y = by + dy;
                if (x < 0 || x >= NX || y < 0 || y >= NY) continue;
                // Only add bumps just outside the smooth surface
                int r2 = (x - p.cx) * (x - p.cx) + (y - p.cy) * (y - p.cy);
                if (r2 > p.radius * p.radius) {
                    system.obstacle[node_index(x, y)] = true;
                }
            }
        }
    }
}

int main(int argc, char* argv[]) {
    std::cout << "==============================================" << std::endl;
    std::cout << " LBM-2D: Sports-Ball Surface Roughness" << std::endl;
    std::cout << " D2Q9 | MRT | OpenMP | Cache-Optimized" << std::endl;
    std::cout << "==============================================" << std::endl;

    double Re = 100.0;
    bool dimpled = false;
    int steps = -1;

    int positional_idx = 1;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--use-les") { g_use_les = true;
        } else if (arg == "--cs" && i + 1 < argc) { g_cs = std::stod(argv[++i]);
        } else if (arg.find("--") != 0) {
            if (positional_idx == 1) Re = std::stod(arg);
            else if (positional_idx == 2) dimpled = (std::stoi(arg) != 0);
            else if (positional_idx == 3) steps = std::stoi(arg);
            ++positional_idx;
        }
    }

    g_case = CaseType::SPORTS_BALL;

    auto params = compute_params(Re, dimpled, steps);
    LBMCapabilities system;

    place_dimpled_cylinder(system, params);

    for (int n = 0; n < NX * NY; ++n) {
        double* f_node = &system.f[n * 9];
        for (int i = 0; i < 9; ++i) {
            f_node[i] = compute_equilibrium(i, 1.0, params.u_inflow, 0.0);
        }
    }

    std::string subdir = "output/sports_ball/re"
                         + std::to_string(static_cast<int>(Re))
                         + (params.dimpled ? "_dimpled" : "_smooth");
    std::filesystem::create_directories(subdir + "/frames");

    save_meta_json(subdir, Re, params.tau, params.u_inflow,
                   params.length_scale, "sports-ball", NX, NY);

    std::cout << "Re = " << Re
              << "  dimpled = " << (params.dimpled ? "yes" : "no")
              << "  tau = " << params.tau
              << "  steps = " << params.num_steps
              << "  D = " << 2 * params.radius
              << "  bumps = " << params.nb_bumps
              << (g_use_les ? "  LES(Cs=" + std::to_string(g_cs) + ")" : "")
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
    std::cout << " Sports-ball simulation complete." << std::endl;
    std::cout << "  Re = " << Re << "  dimpled = " << (params.dimpled ? "yes" : "no") << std::endl;
    std::cout << "  Steps = " << params.num_steps << std::endl;
    std::cout << "==============================================" << std::endl;

    return 0;
}
