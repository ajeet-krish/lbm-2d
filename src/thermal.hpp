#pragma once
#include <vector>
#include <cmath>
#include <array>

// ==========================================================================
// Thermal LBM: Double Distribution Function (DDF) approach
// Keep D2Q9 f_i for momentum, add D2Q9 (or D2Q5) g_i for temperature.
// Solves: dT/dt + u.grad(T) = alpha * laplacian(T)
// Boussinesq coupling: F_buoyancy = -rho_0 * beta * (T - T_ref) * g
// Reference: Guo et al. (2002) "Thermal lattice Boltzmann equation",
//            He et al. (1998) "A priori derivation of the lattice Boltzmann
//            equation for thermal flows".
// ==========================================================================

// D2Q9 thermal lattice weights (same as momentum, different relaxation)
constexpr std::array<double, 9> thermal_weights = {
    4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0
};

// Thermal equilibrium distribution for temperature field
// g_i^eq = w_i * (T / T_0) * (1 + 3 e_i.u + 4.5 (e_i.u)^2 - 1.5 u^2)
// where T_0 is reference temperature (usually 1.0 in lattice units)
inline double thermal_equilibrium(int i, double T, double u, double v) {
    double edotc = cx[i] * u + cy[i] * v;
    double u2 = u * u + v * v;
    // T_0 = 1.0, so T / T_0 = T
    return thermal_weights[i] * T * (1.0 + 3.0 * edotc + 4.5 * edotc * edotc - 1.5 * u2);
}

// Single relaxation time (SRT) thermal collision
// g_i^{new} = g_i - omega_k * (g_i - g_i^eq)
// where omega_k = 1 / tau_k, tau_k = 0.5 + 3 * alpha
inline void thermal_collide(double* g_node, double T, double u, double v,
                            double omega_k) {
    for (int i = 0; i < 9; ++i) {
        double g_eq = thermal_equilibrium(i, T, u, v);
        g_node[i] -= omega_k * (g_node[i] - g_eq);
    }
}

// Boussinesq force term for momentum equation (added during momentum collision)
// F_buoyancy = -rho_0 * beta * (T - T_ref) * g
// Returns force components (fx, fy)
inline void boussinesq_force(double T, double T_ref, double rho_0,
                             double beta, double g_x, double g_y,
                             double& fx, double& fy) {
    double dT = T - T_ref;
    fx = -rho_0 * beta * dT * g_x;
    fy = -rho_0 * beta * dT * g_y;
}

// Macroscopic temperature from distribution moments
inline void compute_temperature(const double* g_node, double& T) {
    T = 0.0;
    for (int i = 0; i < 9; ++i) {
        T += g_node[i];
    }
}

// Thermal boundary conditions
enum class ThermalBC {
    ISOTHERMAL,    // Dirichlet: fixed temperature at wall
    ADIABATIC,     // Neumann: zero heat flux (bounce-back)
    HEAT_FLUX      // Prescribed heat flux (Neumann with non-zero value)
};

// Apply isothermal boundary: set g_i = g_i^eq(T_wall, u_wall)
inline void apply_thermal_isothermal(double* g_node, double T_wall,
                                     double u_wall = 0.0, double v_wall = 0.0) {
    for (int i = 0; i < 9; ++i) {
        g_node[i] = thermal_equilibrium(i, T_wall, u_wall, v_wall);
    }
}

// Apply adiabatic boundary: bounce-back on g_i (zero heat flux)
inline void apply_thermal_adiabatic(double* g_node, const double* g_opp) {
    for (int i = 0; i < 9; ++i) {
        int bb = bounce_back[i];
        g_node[i] = g_opp[bb];
    }
}

// Compute Nusselt number: Nu = h * L / k = (q_wall * L) / (k * dT)
// q_wall = -k * dT/dn (heat flux at wall)
// For a cylinder: Nu = -2 * (dT/dr)|_wall * D / (T_wall - T_inf)
inline double compute_nusselt(double dTdr_wall, double D, double T_wall, double T_inf) {
    if (std::abs(T_wall - T_inf) < 1e-12) return 0.0;
    return -2.0 * dTdr_wall * D / (T_wall - T_inf);
}
