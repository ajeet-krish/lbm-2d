#pragma once
#include <vector>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>

// ==========================================================================
// D2Q9 LATTICE BOLTZMANN -- Type definitions and lattice constants
// ==========================================================================

// Grid dimensions (runtime variables, set by entry point)
inline int NX = 800;
inline int NY = 300;
constexpr int NUM_DIRECTIONS = 9;

// Simulation case type
enum class CaseType { CYLINDER, CAVITY, STEP, RIBS, URBAN_CANYON, DOWNWASH,
                      SQUARE_CYLINDER, FLAT_PLATE, SPORTS_BALL,
                      PERIODIC_HILLS, CYLINDER_NEAR_WALL, SIDE_BY_SIDE,
                      ROTATING_CYLINDER, ORIFICE_PLATE };

// Collision operator type
enum class CollisionType { MRT, BGK };
inline CollisionType g_collision = CollisionType::MRT;   // MRT default, BGK fallback
inline CaseType g_case = CaseType::CYLINDER;

// Smagorinsky LES subgrid-scale model (Phase 1)
inline bool g_use_les = false;          // toggle LES on/off (default off)
inline double g_cs = 0.12;              // Smagorinsky constant (typical 0.1-0.2)

// ------------------------------------------------------------------
// D2Q9 velocity vectors (cx[i], cy[i]) for i = 0..8
// Index convention:
//   6  2  5
//   3  0  1
//   7  4  8
// ------------------------------------------------------------------
constexpr std::array<int, 9> cx = { 0,  1,  0, -1,  0,  1, -1, -1,  1 };
constexpr std::array<int, 9> cy = { 0,  0,  1,  0, -1,  1,  1, -1, -1 };

// D2Q9 quadrature weights
constexpr std::array<double, 9> weights = {
    4.0 / 9.0,                       // i=0  (rest)
    1.0 / 9.0, 1.0 / 9.0,           // i=1,3 (axial)
    1.0 / 9.0, 1.0 / 9.0,           // i=2,4 (axial)
    1.0 / 36.0, 1.0 / 36.0,         // i=5,7 (diagonal)
    1.0 / 36.0, 1.0 / 36.0          // i=6,8 (diagonal)
};

// Inverse direction mapping for bounce-back
// bounce_back[i] gives the direction index opposite to i
constexpr std::array<int, 9> bounce_back = { 0, 3, 4, 1, 2, 7, 8, 5, 6 };

// ------------------------------------------------------------------
// MRT relaxation parameters (d'Humieres 2002)
// ------------------------------------------------------------------
struct MRTParams {
    // s_shear = 1/tau controls physical viscosity (same as BGK)
    // s_bulk controls bulk viscosity (tune for stability, typical 1.0-1.5)
    // s_normal controls ghost modes (typically 1.0 = fully relax)
    double s_shear;    // s_7 = s_8 = pxx, pxy
    double s_bulk;     // s_1 = e,  s_2 = epsilon
    double s_normal;   // s_4 = qx, s_6 = qy

    // Construct from tau (d'Humieres 2002):
    //   s_shear = 1/tau    -- controls shear viscosity (same as BGK)
    //   s_bulk  = 1.2      -- fixed bulk viscosity for stability
    //   s_normal = 1.0     -- fully relax ghost modes
    static MRTParams from_tau(double tau) {
        double s = 1.0 / tau;
        auto clamp = [](double x) { return (x < 0.5) ? 0.5 : ((x > 1.99) ? 1.99 : x); };
        return { clamp(s), 1.2, 1.0 };
    }
};

// ------------------------------------------------------------------
// 1D index helpers
// ------------------------------------------------------------------
inline int node_index(int x, int y) {
    return y * NX + x;
}

// ------------------------------------------------------------------
// Interpolated bounce-back geometry descriptor
// Describes the obstacle surface for computing q = boundary distance
// ------------------------------------------------------------------
struct BounceBackGeometry {
    // Cylinder
    double cx = 0.0, cy = 0.0, radius = 0.0;
    // Polygon vertices (closed polygon, first != last)
    std::vector<std::pair<double,double>> poly_vertices;
    bool is_polygon = false;

    // Moving boundary (Ladd 1994)
    bool has_moving_wall = false;
    double omega = 0.0;           // angular velocity (rad/lattice-time)
    double rot_cx = 0.0, rot_cy = 0.0;  // rotation center

    bool is_valid() const { return radius > 0.0 || is_polygon; }

    // Compute wall velocity at a point for rotating cylinder
    // u_wall = omega * r_hat (tangential velocity)
    void compute_wall_velocity(double px, double py,
                               double& u_wall_x, double& u_wall_y) const {
        if (!has_moving_wall) { u_wall_x = 0.0; u_wall_y = 0.0; return; }
        double dx = px - rot_cx;
        double dy = py - rot_cy;
        // Tangential velocity: u = omega * (-dy, dx)
        u_wall_x = -omega * dy;
        u_wall_y =  omega * dx;
    }

    // Compute normalized boundary distance q along direction i
    // from fluid node (xf, yf) toward the obstacle
    double q_cylinder(double xf, double yf, int i) const {
        double dx = xf - cx;
        double dy = yf - cy;
        double ex = static_cast<double>(::cx[i]);
        double ey = static_cast<double>(::cy[i]);
        double len2 = ex*ex + ey*ey;

        double b = 2.0 * (dx*ex + dy*ey);
        double c = dx*dx + dy*dy - radius*radius;
        double disc = b*b - 4.0*len2*c;

        if (disc < 0.0) return 1.0;

        double sqrt_disc = std::sqrt(disc);
        double t1 = (-b - sqrt_disc) / (2.0 * len2);
        double t2 = (-b + sqrt_disc) / (2.0 * len2);

        if (t1 > 0.0 && t1 <= 1.0) return t1;
        if (t2 > 0.0 && t2 <= 1.0) return t2;
        return 1.0;
    }

    // Compute q for a polygon boundary using line-segment intersection
    // Ray from (xf, yf) in direction i intersects polygon edges
    // Returns smallest positive t <= 1.0
    double q_polygon(double xf, double yf, int i) const {
        double ex = static_cast<double>(::cx[i]);
        double ey = static_cast<double>(::cy[i]);
        int n = static_cast<int>(poly_vertices.size());
        if (n < 3) return 1.0;

        double best_t = 1.0;
        for (int v = 0; v < n; ++v) {
            int v2 = (v + 1) % n;
            double x1 = poly_vertices[v].first;
            double y1 = poly_vertices[v].second;
            double x2 = poly_vertices[v2].first;
            double y2 = poly_vertices[v2].second;

            double dx_e = x2 - x1;
            double dy_e = y2 - y1;
            double denom = ex * dy_e - ey * dx_e;
            if (std::abs(denom) < 1e-15) continue;  // parallel or near-parallel

            double t = ((x1 - xf) * dy_e - (y1 - yf) * dx_e) / denom;
            double s = ((x1 - xf) * ey - (y1 - yf) * ex) / denom;

            if (t > 0.0 && t <= best_t && s >= 0.0 && s <= 1.0) {
                best_t = t;
            }
        }
        return best_t;
    }

    // Unified q computation
    double compute_q(double xf, double yf, int i) const {
        if (is_polygon) return q_polygon(xf, yf, i);
        return q_cylinder(xf, yf, i);
    }
};

// ------------------------------------------------------------------
// System state: flat 1D arrays for cache efficiency
// ------------------------------------------------------------------
struct LBMCapabilities {
    std::vector<double> f;          // current distribution f[node * 9 + i]
    std::vector<double> f_next;     // buffer for next timestep
    std::vector<uint8_t> obstacle;  // true if node is inside obstacle

    std::vector<double> fx_body;    // cumulative drag force on obstacle
    std::vector<double> fy_body;    // cumulative lift force on obstacle

    BounceBackGeometry bb_geom;     // for interpolated bounce-back
    double body_force_x = 0.0;      // body force for periodic channel flows

    LBMCapabilities() {
        int n_nodes = NX * NY;
        f.resize(n_nodes * NUM_DIRECTIONS, 0.0);
        f_next.resize(n_nodes * NUM_DIRECTIONS, 0.0);
        obstacle.resize(n_nodes, false);
        reset_forces();
    }

    void reset_forces() {
        fx_body.assign(NX * NY, 0.0);
        fy_body.assign(NX * NY, 0.0);
    }
};

// ------------------------------------------------------------------
// Equilibrium distribution: f_i^eq = w_i * rho * (1 + 3 e.u + 4.5 (e.u)^2 - 1.5 u.u)
// ------------------------------------------------------------------
inline double compute_equilibrium(int i, double rho, double u, double v) {
    double edotc = cx[i] * u + cy[i] * v;
    double u2 = u * u + v * v;
    return weights[i] * rho * (1.0 + 3.0 * edotc + 4.5 * edotc * edotc - 1.5 * u2);
}

// ------------------------------------------------------------------
// Macroscopic properties from distribution moments
// ------------------------------------------------------------------
inline void compute_macros(const double* f_node, double& rho, double& u, double& v) {
    rho = 0.0;
    double mx = 0.0, my = 0.0;
    for (int i = 0; i < 9; ++i) {
        rho += f_node[i];
        mx   += f_node[i] * cx[i];
        my   += f_node[i] * cy[i];
    }
    u = mx / rho;
    v = my / rho;
}
