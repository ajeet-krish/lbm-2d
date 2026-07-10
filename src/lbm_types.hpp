#pragma once
#include <vector>
#include <array>
#include <cmath>

// ==========================================================================
// D2Q9 LATTICE BOLTZMANN -- Type definitions and lattice constants
// ==========================================================================

// Grid dimensions (runtime variables, set by entry point)
inline int NX = 400;
inline int NY = 150;
constexpr int NUM_DIRECTIONS = 9;

// Simulation case type
enum class CaseType { CYLINDER, CAVITY };
inline CaseType g_case = CaseType::CYLINDER;

// D2Q9 velocity vectors (cx[i], cy[i]) for i = 0..8
// Index convention:
//   6  2  5
//   3  0  1
//   7  4  8
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
// 1D index helpers
// ------------------------------------------------------------------
inline int node_index(int x, int y) {
    return y * NX + x;
}

inline int dist_index(int x, int y, int i) {
    return (y * NX + x) * 9 + i;
}

// ------------------------------------------------------------------
// System state: flat 1D arrays for cache efficiency
// ------------------------------------------------------------------
struct LBMCapabilities {
    std::vector<double> f;          // current distribution f[node * 9 + i]
    std::vector<double> f_next;     // buffer for next timestep
    std::vector<bool> obstacle;     // true if node is inside cylinder

    std::vector<double> fx_cyl;     // cumulative drag force on cylinder
    std::vector<double> fy_cyl;     // cumulative lift force on cylinder
    int n_cyl_nodes;                // number of cylinder boundary nodes

    LBMCapabilities() {
        int n_nodes = NX * NY;
        f.resize(n_nodes * NUM_DIRECTIONS, 0.0);
        f_next.resize(n_nodes * NUM_DIRECTIONS, 0.0);
        obstacle.resize(n_nodes, false);
        reset_forces();
    }

    void reset_forces() {
        fx_cyl.assign(NX * NY, 0.0);
        fy_cyl.assign(NX * NY, 0.0);
        n_cyl_nodes = 0;
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
