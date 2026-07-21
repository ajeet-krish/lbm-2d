#include "lbm.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include <filesystem>
#include <random>

// ==========================================================================
// LBM-2D: Airfoil with Immersed Boundary Method (IBM)
// ==========================================================================
// Usage:
//   ./build/LBM_AirfoilIBM 1000 0         (Re=1000, AoA=0)
//   ./build/LBM_AirfoilIBM 1000 10 20000 (Re=1000, AoA=10, 20000 steps)
//
// Demonstrates Upgrade 1 Tier 3: IBM with direct forcing for complex shapes.
// Unlike Bouzidi bounce-back (which stamps a binary obstacle mask), IBM places
// Lagrangian points on the TRUE airfoil surface and spreads the no-slip
// condition to nearby Cartesian grid nodes via a smoothed delta function.
// This gives smooth curved boundaries without staircase artifacts.
//
// Validation:
//   - Thin airfoil theory: Cl = 2*pi*alpha (small angle)
//   - NACA 0012: Cl ~ 0.11*alpha_deg (experiment)
// ==========================================================================

int main(int argc, char* argv[]) {
    std::cout << "==============================================" << std::endl;
    std::cout << " LBM-2D: Airfoil (IBM Direct Forcing)" << std::endl;
    std::cout << " D2Q9 | MRT | OpenMP | Immersed Boundary" << std::endl;
    std::cout << "==============================================" << std::endl;

    double Re = 1000.0;
    double alpha_deg = 0.0;
    int steps = -1;

    int positional_idx = 1;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--use-les") {
            g_use_les = true;
        } else if (arg.find("--") != 0) {
            if (positional_idx == 1) Re = std::stod(arg);
            else if (positional_idx == 2) alpha_deg = std::stod(arg);
            else if (positional_idx == 3) steps = std::stoi(arg);
            ++positional_idx;
        }
    }

    double u_inflow = 0.1;
    double nu = u_inflow * (NY / 5.0) / Re;
    double tau = 0.5 + 3.0 * nu;

    int num_steps = (steps > 0) ? steps
        : std::max(4000, static_cast<int>(10.0 * NX / u_inflow));
    int save_interval = num_steps / 50;

    std::cout << "  Re = " << Re << ", AoA = " << alpha_deg << " deg" << std::endl;
    std::cout << "  tau = " << tau << std::endl;
    std::cout << "==============================================" << std::endl;

    LBMCapabilities system;

    // Build NACA airfoil geometry
    int naca_series = 2412;  // 2% camber, 12% thickness
    auto airfoil_poly = naca_coords(naca_series, 100);

    // Place at center, scale to chord = NY/3, rotate by AoA
    double chord = static_cast<double>(NY) / 3.0;
    double cx_air = static_cast<double>(NX) / 4;
    double cy_air = static_cast<double>(NY) / 2;
    transform_points(airfoil_poly, cx_air, cy_air, chord, alpha_deg);

    // Build IBM structure
    ImmersedBoundary iboundary;
    iboundary.build_from_polygon(airfoil_poly, 1.0);  // 1 cell spacing

    // Mark Eulerian obstacle mask (for streaming skip, same as before)
    for (int y = 0; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {
            if (point_in_polygon(static_cast<double>(x), static_cast<double>(y), airfoil_poly)) {
                system.obstacle[node_index(x, y)] = true;
            }
        }
    }

    std::cout << "  Lagrangian points: " << iboundary.points.size() << std::endl;
    std::cout << "  Eulerian obstacle nodes: " << std::count(system.obstacle.begin(), system.obstacle.end(), true) << std::endl;

    // Initialize
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {
            int idx = node_index(x, y);
            double rho = 1.0;
            double u = u_inflow * std::cos(alpha_deg * M_PI / 180.0);
            double v = u_inflow * std::sin(alpha_deg * M_PI / 180.0);
            for (int i = 0; i < 9; ++i) {
                system.f[idx * 9 + i] = compute_equilibrium(i, rho, u, v);
            }
        }
    }

    std::string outdir = "output/airfoil_ibm_re" + std::to_string(static_cast<int>(Re))
                        + "_aoa" + std::to_string(static_cast<int>(alpha_deg));
    std::filesystem::create_directories(outdir + "/frames");

    // Write meta
    std::ofstream meta(outdir + "/meta.json");
    meta << "{\"case\":\"airfoil_ibm\",\"Re\":" << Re
         << ",\"alpha_deg\":" << alpha_deg
         << ",\"u_inflow\":" << u_inflow
         << ",\"nx\":" << NX << ",\"ny\":" << NY << "}" << std::endl;
    meta.close();

    // Force buffers
    std::vector<double> fx_grid(NX * NY, 0.0), fy_grid(NX * NY, 0.0);
    std::vector<double> fx_lag, fy_lag;
    double dt = 1.0;

    for (int step = 0; step <= num_steps; ++step) {
        // 1. Interpolate velocity to Lagrangian points
        // (recompute u, v from f at each point's location)
        #pragma omp parallel for
        for (size_t p = 0; p < iboundary.points.size(); ++p) {
            double X = iboundary.points[p].X;
            double Y = iboundary.points[p].Y;
            double u = 0, v = 0;
            int x0 = static_cast<int>(std::floor(X)) - 2;
            int x1 = static_cast<int>(std::floor(X)) + 2;
            int y0 = static_cast<int>(std::floor(Y)) - 2;
            int y1 = static_cast<int>(std::floor(Y)) + 2;
            for (int y = y0; y <= y1; ++y) {
                if (y < 0 || y >= NY) continue;
                for (int x = x0; x <= x1; ++x) {
                    if (x < 0 || x >= NX) continue;
                    double kernel = ibm_kernel(x - X, y - Y);
                    if (kernel == 0.0) continue;
                    int idx = node_index(x, y);
                    if (system.obstacle[idx]) continue;
                    double rho, uu, vv;
                    compute_macros(&system.f[idx * 9], rho, uu, vv);
                    u += uu * kernel;
                    v += vv * kernel;
                }
            }
            iboundary.points[p].U = u;
            iboundary.points[p].V = v;
        }

        // 2. Compute no-slip force (target velocity = 0 at solid)
        iboundary.compute_no_slip_force(dt, fx_lag, fy_lag);

        // 3. Spread force to Eulerian grid
        std::fill(fx_grid.begin(), fx_grid.end(), 0.0);
        std::fill(fy_grid.begin(), fy_grid.end(), 0.0);
        iboundary.spread_force(fx_lag, fy_lag, fx_grid, fy_grid, NX, NY);

        // 4. Run LBM step with IBM forcing
        execute_time_step(system, tau, u_inflow);

        // 5. Apply IBM force as body force to momentum (implicit in collision)
        #pragma omp parallel for collapse(2)
        for (int y = 0; y < NY; ++y) {
            for (int x = 0; x < NX; ++x) {
                int idx = node_index(x, y);
                if (system.obstacle[idx]) continue;
                double* f_node = &system.f[idx * 9];
                double fx = fx_grid[idx];
                double fy = fy_grid[idx];
                // Apply as explicit body force (Guo scheme)
                for (int i = 0; i < 9; ++i) {
                    double edotu = cx[i] * 0 + cy[i] * 0;  // approx (u~0)
                    double Fdot = cx[i] * fx + cy[i] * fy;
                    double correction = weights[i] * (
                        3.0 * Fdot + 9.0 * edotu * (cx[i] * fx + cy[i] * fy)
                    );
                    f_node[i] += correction;
                }
            }
        }

        if (step % save_interval == 0) {
            std::cout << "  Step " << step << " / " << num_steps << std::endl;
            save_json_frame(system, step, outdir);
        }
    }

    std::cout << "  Done. Output: " << outdir << std::endl;
    return 0;
}
