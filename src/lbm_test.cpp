#include <gtest/gtest.h>
#include "lbm.hpp"
#include <cmath>

// ==========================================================================
// Unit tests for D2Q9 LBM components
// ==========================================================================

// ------------------------------------------------------------------
// Equilibrium distribution tests
// ------------------------------------------------------------------
TEST(EquilibriumTest, SumEqualsRho) {
    double rho = 1.2;
    double u = 0.1, v = 0.05;
    double sum = 0.0;
    for (int i = 0; i < 9; ++i) {
        sum += compute_equilibrium(i, rho, u, v);
    }
    EXPECT_NEAR(sum, rho, 1e-12);
}

TEST(EquilibriumTest, MomentumMatches) {
    double rho = 1.0;
    double u = 0.15, v = 0.0;
    double mx = 0.0, my = 0.0;
    for (int i = 0; i < 9; ++i) {
        double feq = compute_equilibrium(i, rho, u, v);
        mx += feq * cx[i];
        my += feq * cy[i];
    }
    EXPECT_NEAR(mx, rho * u, 1e-12);
    EXPECT_NEAR(my, 0.0, 1e-12);
}

TEST(EquilibriumTest, RestState) {
    double rho = 1.0;
    double u = 0.0, v = 0.0;
    for (int i = 0; i < 9; ++i) {
        double feq = compute_equilibrium(i, rho, u, v);
        EXPECT_NEAR(feq, weights[i], 1e-15);
    }
}

TEST(EquilibriumTest, AllDirectionsCovered) {
    // Verify all 9 directions produce non-negative values for reasonable inputs
    double rho = 1.0;
    double u = 0.1, v = 0.05;
    for (int i = 0; i < 9; ++i) {
        double feq = compute_equilibrium(i, rho, u, v);
        EXPECT_GT(feq, 0.0);
    }
}

// ------------------------------------------------------------------
// Macroscopic property extraction tests
// ------------------------------------------------------------------
TEST(MacrosTest, RecoverInput) {
    double f_node[9];
    double rho_in = 1.5;
    double u_in = 0.12, v_in = -0.08;

    for (int i = 0; i < 9; ++i) {
        f_node[i] = compute_equilibrium(i, rho_in, u_in, v_in);
    }

    double rho_out, u_out, v_out;
    compute_macros(f_node, rho_out, u_out, v_out);

    EXPECT_NEAR(rho_out, rho_in, 1e-12);
    EXPECT_NEAR(u_out, u_in, 1e-12);
    EXPECT_NEAR(v_out, v_in, 1e-12);
}

TEST(MacrosTest, ZeroVelocity) {
    double f_node[9];
    for (int i = 0; i < 9; ++i) {
        f_node[i] = weights[i];
    }

    double rho, u, v;
    compute_macros(f_node, rho, u, v);

    EXPECT_NEAR(rho, 1.0, 1e-15);
    EXPECT_NEAR(u, 0.0, 1e-15);
    EXPECT_NEAR(v, 0.0, 1e-15);
}

// ------------------------------------------------------------------
// Index helper tests
// ------------------------------------------------------------------
TEST(IndexTest, NodeIndex) {
    EXPECT_EQ(node_index(0, 0), 0);
    EXPECT_EQ(node_index(1, 0), 1);
    EXPECT_EQ(node_index(0, 1), NX);
    EXPECT_EQ(node_index(NX - 1, NY - 1), NX * NY - 1);
}

TEST(IndexTest, DistributionIndex) {
    EXPECT_EQ(dist_index(0, 0, 0), 0);
    EXPECT_EQ(dist_index(0, 0, 1), 1);
    EXPECT_EQ(dist_index(1, 0, 0), 9);
    EXPECT_EQ(dist_index(0, 1, 0), NX * 9);
}

// ------------------------------------------------------------------
// Bounce-back direction test
// ------------------------------------------------------------------
TEST(BounceBackTest, OppositeDirections) {
    // Bounce-back reverses the velocity: i and bounce_back[i] should be opposites
    for (int i = 0; i < 9; ++i) {
        EXPECT_EQ(cx[bounce_back[i]], -cx[i]);
        EXPECT_EQ(cy[bounce_back[i]], -cy[i]);
    }
    // Self-inverse property
    for (int i = 0; i < 9; ++i) {
        EXPECT_EQ(bounce_back[bounce_back[i]], i);
    }
}

// ------------------------------------------------------------------
// Obstacle placement test
// ------------------------------------------------------------------
TEST(ObstacleTest, CylinderPlacement) {
    LBMCapabilities sys;
    int cx_cyl = NX / 4;
    int cy_cyl = NY / 2;
    int radius = NY / 10;

    place_cylinder(sys, cx_cyl, cy_cyl, radius);

    // Center of cylinder should be obstacle
    EXPECT_TRUE(sys.obstacle[node_index(cx_cyl, cy_cyl)]);

    // Far corner should NOT be obstacle
    EXPECT_FALSE(sys.obstacle[node_index(0, 0)]);
    EXPECT_FALSE(sys.obstacle[node_index(NX - 1, NY - 1)]);

    // Count obstacle nodes (should be approximately pi * r^2)
    int count = 0;
    for (int n = 0; n < NX * NY; ++n) {
        if (sys.obstacle[n]) ++count;
    }
    double expected = M_PI * radius * radius;
    double ratio = static_cast<double>(count) / expected;
    // Obstacle count for a circle on a grid should be within ~20% of area
    EXPECT_GT(ratio, 0.6);
    EXPECT_LT(ratio, 1.5);
}

// ------------------------------------------------------------------
// Force extraction initialization test
// ------------------------------------------------------------------
TEST(ForceTest, ResetForces) {
    LBMCapabilities sys;
    sys.fx_cyl[0] = 1.0;
    sys.fy_cyl[0] = 2.0;
    sys.n_cyl_nodes = 5;
    sys.reset_forces();
    EXPECT_DOUBLE_EQ(sys.fx_cyl[0], 0.0);
    EXPECT_DOUBLE_EQ(sys.fy_cyl[0], 0.0);
    EXPECT_EQ(sys.n_cyl_nodes, 0);
}

// ------------------------------------------------------------------
// Time step conservation test
// ------------------------------------------------------------------
TEST(TimeStepTest, MassConservation) {
    LBMCapabilities sys;
    int cx_cyl = NX / 4;
    int cy_cyl = NY / 2;
    int radius = NY / 10;
    place_cylinder(sys, cx_cyl, cy_cyl, radius);

    // Initialize with uniform equilibrium
    for (int n = 0; n < NX * NY; ++n) {
        double* f_node = &sys.f[n * 9];
        for (int i = 0; i < 9; ++i) {
            f_node[i] = compute_equilibrium(i, 1.0, 0.1, 0.0);
        }
    }

    // Compute initial total mass
    double mass0 = 0.0;
    for (int n = 0; n < NX * NY; ++n) {
        if (sys.obstacle[n]) continue;
        for (int i = 0; i < 9; ++i) {
            mass0 += sys.f[n * 9 + i];
        }
    }

    // Run 10 steps
    double tau = 0.56;
    for (int step = 0; step < 10; ++step) {
        execute_time_step(sys, tau, 0.1);
    }

    // Compute final total mass (should be conserved within a small tolerance)
    double mass1 = 0.0;
    for (int n = 0; n < NX * NY; ++n) {
        if (sys.obstacle[n]) continue;
        for (int i = 0; i < 9; ++i) {
            mass1 += sys.f[n * 9 + i];
        }
    }

    // Allow 1% tolerance for boundary losses
    EXPECT_NEAR(mass1, mass0, 0.01 * mass0);
}

// ------------------------------------------------------------------
// Parameter computation tests
// ------------------------------------------------------------------
TEST(ParamsTest, TauFromRe) {
    // For Re = 100, u_inflow = 0.1, NX = 800:
    // nu = 0.1 * 800 / 100 = 0.8
    // tau = 0.5 + 3 * 0.8 = 2.9
    double u_inflow = 0.1;
    double Re = 100.0;
    double nu = u_inflow * NX / Re;
    double tau = 0.5 + 3.0 * nu;
    EXPECT_NEAR(tau, 2.9, 1e-12);

    // Re = 200: nu = 0.1 * 800 / 200 = 0.4, tau = 0.5 + 3 * 0.4 = 1.7
    Re = 200.0;
    nu = u_inflow * NX / Re;
    tau = 0.5 + 3.0 * nu;
    EXPECT_NEAR(tau, 1.7, 1e-12);
}
