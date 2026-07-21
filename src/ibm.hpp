#pragma once
#include <vector>
#include <utility>
#include <cmath>
#include <algorithm>

// ==========================================================================
// Immersed Boundary Method (IBM) with direct forcing
// For airfoils and complex shapes where Cartesian staircase is inaccurate.
// Lagrangian points on the true surface, force spreading to nearby Eulerian
// (grid) nodes via a smoothed delta function, velocity interpolation back.
// ==========================================================================

// 4-point smoothed delta function (Peskin 2002)
// phi(r) = (1/4) * (1 + cos(pi*r/2)) for |r| <= 2, 0 otherwise
inline double ibm_delta(double r) {
    if (std::abs(r) >= 2.0) return 0.0;
    return 0.25 * (1.0 + std::cos(0.5 * M_PI * r));
}

// 2D smoothed delta: phi(x - X) * phi(y - Y)
inline double ibm_kernel(double dx, double dy) {
    return ibm_delta(dx) * ibm_delta(dy);
}

// Lagrangian point on the immersed boundary
struct LagrangianPoint {
    double X;       // physical x position
    double Y;       // physical y position
    double FX = 0;  // force in x
    double FY = 0;  // force in y
    double U = 0;   // interpolated fluid velocity x
    double V = 0;   // interpolated fluid velocity y
};

// Immersed boundary structure
struct ImmersedBoundary {
    std::vector<LagrangianPoint> points;
    std::vector<std::pair<double,double>> surface;  // closed polygon for reference

    // Build Lagrangian points from a closed polygon (e.g., NACA airfoil)
    // spacing = approximate distance between Lagrangian points (in lattice units)
    void build_from_polygon(const std::vector<std::pair<double,double>>& poly,
                            double spacing = 1.0) {
        surface = poly;
        points.clear();
        int n = static_cast<int>(poly.size());
        if (n < 3) return;

        double total_len = 0.0;
        for (int i = 0; i < n; ++i) {
            int j = (i + 1) % n;
            double dx = poly[j].first - poly[i].first;
            double dy = poly[j].second - poly[i].second;
            total_len += std::sqrt(dx*dx + dy*dy);
        }

        int n_points = std::max(8, static_cast<int>(total_len / spacing));
        for (int k = 0; k < n_points; ++k) {
            double t = static_cast<double>(k) / n_points;
            double seg = t * n;  // which segment
            int i = static_cast<int>(seg) % n;
            int j = (i + 1) % n;
            double f = seg - std::floor(seg);
            double x = poly[i].first + f * (poly[j].first - poly[i].first);
            double y = poly[i].second + f * (poly[j].second - poly[i].second);
            points.push_back({x, y, 0, 0, 0, 0});
        }
    }

    // Spread Lagrangian force to Eulerian grid using smoothed delta
    // Adds force density to force_field[node * 2 + {0,1}] for each nearby node
    void spread_force(const std::vector<double>& fx_lag,
                      const std::vector<double>& fy_lag,
                      std::vector<double>& fx_grid,
                      std::vector<double>& fy_grid,
                      int NX, int NY) const {
        int npts = static_cast<int>(points.size());
        for (int p = 0; p < npts; ++p) {
            double X = points[p].X;
            double Y = points[p].Y;
            double fx = fx_lag[p];
            double fy = fy_lag[p];

            // Spread to nodes within 2 cells of (X, Y)
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
                    int idx = y * NX + x;
                    fx_grid[idx] += fx * kernel;
                    fy_grid[idx] += fy * kernel;
                }
            }
        }
    }

    // Interpolate Eulerian velocity to Lagrangian points
    void interpolate_velocity(const std::vector<double>& u_grid,
                             const std::vector<double>& v_grid,
                             int NX, int NY) {
        int npts = static_cast<int>(points.size());
        for (int p = 0; p < npts; ++p) {
            double X = points[p].X;
            double Y = points[p].Y;
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
                    int idx = y * NX + x;
                    u += u_grid[idx] * kernel;
                    v += v_grid[idx] * kernel;
                }
            }
            points[p].U = u;
            points[p].V = v;
        }
    }

    // Compute boundary force from velocity defect (direct forcing)
    // F = (U_target - U_interp) / dt  (penalty-like, implicit in explicit form)
    // For no-slip: U_target = 0, so F = -U_interp / dt
    void compute_no_slip_force(double dt, std::vector<double>& fx_lag,
                               std::vector<double>& fy_lag,
                               double u_wall_x = 0.0, double u_wall_y = 0.0) const {
        int npts = static_cast<int>(points.size());
        fx_lag.resize(npts);
        fy_lag.resize(npts);
        for (int p = 0; p < npts; ++p) {
            fx_lag[p] = (u_wall_x - points[p].U) / dt;
            fy_lag[p] = (u_wall_y - points[p].V) / dt;
        }
    }
};
