#include "amr.hpp"
#include <iostream>
#include <iomanip>
#include <random>

int main(int argc, char* argv[]) {
    std::cout << "==============================================" << std::endl;
    std::cout << " LBM-2D AMR Test: Cylinder Flow" << std::endl;
    std::cout << "==============================================" << std::endl;

    double Re = 100.0;
    int steps = 1000;
    int positional_idx = 1;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--use-les") { g_use_les = true;
        } else if (arg == "--cs" && i + 1 < argc) { g_cs = std::stod(argv[++i]);
        } else if (arg.find("--") != 0) {
            if (positional_idx == 1) Re = std::stod(arg);
            else if (positional_idx == 2) steps = std::stoi(arg);
            ++positional_idx;
        }
    }

    double u_inflow = 0.1;
    double length_scale = static_cast<double>(NY) / 5.0;
    double tau = 0.5 + 3.0 * u_inflow * length_scale / Re;

    int cx_cyl = NX / 4, cy_cyl = NY / 2 + 1, radius = NY / 10;

    AMRGrid grid(NX, NY);
    AMRBlock& base = grid.levels[0][0];
    BounceBackGeometry geom;
    geom.cx = cx_cyl; geom.cy = cy_cyl; geom.radius = radius;
    base.bb_geom = geom;

    for (int y = 0; y < base.ny; ++y)
        for (int x = 0; x < base.nx; ++x)
            if (std::sqrt(double((x-cx_cyl)*(x-cx_cyl)+(y-cy_cyl)*(y-cy_cyl))) < radius)
                base.obstacle[base.idx(x, y)] = true;

    base.init_equilibrium(1.0, u_inflow, 0.0);

    // Perturbation
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> pert(-1e-4, 1e-4);
    for (int x = cx_cyl + 5; x < std::min(NX, cx_cyl + 60); ++x)
        for (int y = 0; y < NY; ++y) {
            int ci = base.idx(x, y);
            if (base.obstacle[ci]) continue;
            double* f_node = &base.f[ci * 9];
            double vp = pert(rng);
            double rho, u, v;
            compute_macros(f_node, rho, u, v);
            for (int d = 0; d < 9; ++d)
                f_node[d] = compute_equilibrium(d, rho, u, v + vp);
        }

    std::cout << "Re=" << Re << " tau=" << tau << " steps=" << steps
              << (g_use_les ? " LES(Cs=" + std::to_string(g_cs) + ")" : "")
              << std::endl;

    for (int step = 0; step <= steps; ++step) {
        grid.execute_amr_step(tau, u_inflow);

        double fx = 0, fy = 0;
        for (int n = 0; n < base.total_nodes(); ++n) {
            fx += base.fx_body[n]; fy += base.fy_body[n];
        }
        double cd = 2.0 * fx / (length_scale * u_inflow * u_inflow);
        double cl = 2.0 * fy / (length_scale * u_inflow * u_inflow);

        if (step % 500 == 0)
            std::cout << "  step " << std::setw(6) << step
                      << "  Cd=" << std::fixed << std::setprecision(4) << cd
                      << "  Cl=" << std::fixed << std::setprecision(4) << cl << std::endl;
    }

    return 0;
}
