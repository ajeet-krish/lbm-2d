#pragma once
#include <vector>
#include <cmath>
#include <algorithm>

// ==========================================================================
// Wall Function Bounce-back (WFB) for LBM
// Slip-velocity approach: compute wall shear stress from resolved gradient,
// apply log-law to determine slip velocity, impose via modified bounce-back.
// Enables accurate high-Re wall-bounded flows without y+ ~ 1 grid resolution.
// Reference: Ponsin & Lozano (2025), "Wall boundary conditions for lattice
// Boltzmann simulations of turbulent flows modeled using RANS equations
// with wall functions."
// ==========================================================================

struct WallFunctionParams {
    double kappa = 0.41;       // von Karman constant
    double B = 5.0;            // log-law additive constant (smooth wall)
    double A_plus = 26.0;      // Van Driest constant
    double y_plus_min = 30.0;  // minimum y+ for wall function applicability
    bool enabled = false;
};

// Compute wall shear stress (tau_wall) from the velocity gradient at the wall
// Using finite difference of velocity at the first two fluid nodes near the wall.
// u_tau = sqrt(tau_wall / rho)
inline double compute_u_tau(double u_first, double u_second,
                            double y_first, double y_second,
                            double rho, double nu) {
    if (y_second <= y_first) return 0.0;
    double du_dy = (u_second - u_first) / (y_second - y_first);
    // tau_wall = nu * rho * du/dy (Newtonian fluid)
    double tau_wall = nu * rho * du_dy;
    if (tau_wall <= 0.0) return 0.0;
    return std::sqrt(tau_wall);
}

// Compute slip velocity from log-law: u+ = (1/kappa) * ln(y+) + B
// Returns u_slip = u_tau * u+
inline double compute_slip_velocity(double u_tau, double y_plus,
                                     const WallFunctionParams& wf) {
    if (u_tau <= 0.0) return 0.0;
    if (y_plus < 1e-6) return 0.0;
    double u_plus = (1.0 / wf.kappa) * std::log(y_plus) + wf.B;
    return u_tau * u_plus;
}

// Check if wall function should be applied based on y+
// Only apply if first cell y+ > threshold (buffer/log layer, not viscous sublayer)
inline bool should_apply_wall_function(double y_plus,
                                       const WallFunctionParams& wf) {
    return wf.enabled && y_plus > wf.y_plus_min;
}

// Apply wall function correction to a bounce-back distribution
// f_bb = f_opp - 2 * w_i * rho * (e_i . u_slip) / c_s^2
// where u_slip is the slip velocity (tangential) at the wall
inline void apply_wall_function_bb(double* f_next, int node_idx, int i,
                                    double f_opp, double rho,
                                    double u_slip_x, double u_slip_y) {
    // cx, cy, weights are defined in lbm_types.hpp (global lattice constants)
    // c_s^2 = 1/3 in lattice units, so / c_s^2 = * 3
    double edot_u = cx[i] * u_slip_x + cy[i] * u_slip_y;
    double correction = 2.0 * weights[i] * rho * edot_u * 3.0;
    f_next[node_idx * 9 + i] = f_opp - correction;
}
