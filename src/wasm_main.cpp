// ==========================================================================
// LBM-2D: WebAssembly real-time solver
// D2Q9 BGK on a fixed 100x60 grid with polygon obstacle support.
// No OpenMP, no file I/O -- compiled with Emscripten for browser use.
// ==========================================================================
#include <cstdint>
#include <cmath>
#include <cstring>
#include <algorithm>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

// --------------------------------------------------------------------------
// Grid dimensions -- fixed for WASM
// --------------------------------------------------------------------------
static constexpr int NX = 100;
static constexpr int NY = 60;
static constexpr int NN = NX * NY;
static constexpr int ND = NN * 9;  // 9 distributions per node

// --------------------------------------------------------------------------
// D2Q9 lattice constants
// --------------------------------------------------------------------------
static constexpr int cx[9] = { 0, 1, 0, -1, 0, 1, -1, -1, 1 };
static constexpr int cy[9] = { 0, 0, 1, 0, -1, 1, 1, -1, -1 };
static constexpr double w[9] = {
    4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0
};
static constexpr int bounce_back[9] = { 0, 3, 4, 1, 2, 7, 8, 5, 6 };

// --------------------------------------------------------------------------
// Solver state
// --------------------------------------------------------------------------
static double f[ND];           // distributions (current time step)
static double f_next[ND];      // distributions (next time step)
static int obstacle[NN];       // 1 = solid, 0 = fluid
static double fx_cyl[NN];      // x-force per node (accumulated each step)
static double fy_cyl[NN];      // y-force per node
static double u_field[NN];     // velocity x-component (computed each step)
static double v_field[NN];     // velocity y-component (computed each step)
static double rho_field[NN];   // density (computed each step)
static double u_inflow = 0.1;  // inflow velocity
static double tau = 0.74;      // relaxation time
static double cd_last = 0.0;   // last computed drag coefficient
static double cl_last = 0.0;   // last computed lift coefficient
static int step_count = 0;     // number of steps executed

// Shape geometry parameters
static int shape_type = 0;     // 0=cylinder, 1=square, 2=diamond, 3=star, 4=airfoil
static double shape_cx = 30;   // shape center x (grid units)
static double shape_cy = 30;   // shape center y (grid units)
static double shape_size = 12; // characteristic size (radius, half-width, etc.)
static int airfoil_series = 0; // NACA series (e.g. 12 for 0012)
static double airfoil_aoa = 0; // angle of attack (degrees)

// --------------------------------------------------------------------------
// Helpers
// --------------------------------------------------------------------------
static inline int node_index(int x, int y) { return y * NX + x; }

static inline double compute_equilibrium(int i, double rho, double u, double v) {
    double cu = cx[i] * u + cy[i] * v;
    double usq = u * u + v * v;
    return w[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq);
}

static inline void compute_macros(const double* f_node, double& rho, double& u, double& v) {
    rho = 0.0; u = 0.0; v = 0.0;
    for (int i = 0; i < 9; ++i) {
        rho += f_node[i];
        u += f_node[i] * cx[i];
        v += f_node[i] * cy[i];
    }
    u /= rho;
    v /= rho;
}

// --------------------------------------------------------------------------
// Point-in-polygon (ray casting)
// --------------------------------------------------------------------------
static bool point_in_polygon(double px, double py,
    const double* poly_x, const double* poly_y, int n)
{
    bool inside = false;
    for (int i = 0, j = n - 1; i < n; j = i++) {
        double xi = poly_x[i], yi = poly_y[i];
        double xj = poly_x[j], yj = poly_y[j];
        bool intersect = (yi > py) != (yj > py)
            && (px < (xj - xi) * (py - yi) / (yj - yi) + xi);
        if (intersect) inside = !inside;
    }
    return inside;
}

// --------------------------------------------------------------------------
// NACA 4-digit airfoil coordinates
// Returns polygon vertices in poly_x, poly_y (2*n points for closed polygon).
// --------------------------------------------------------------------------
static int naca_coords(int series, double* poly_x, double* poly_y, int max_pts) {
    int m_code = series / 1000;
    int p_code = (series / 100) % 10;
    int t_code = series % 100;

    double m = static_cast<double>(m_code) / 100.0;
    double p = static_cast<double>(p_code) / 10.0;
    double t = static_cast<double>(t_code) / 100.0;

    int n = max_pts / 2;  // half the points for upper+lower
    if (n < 10) n = 10;

    // Cosine spacing
    double x[200];
    for (int i = 0; i < n; ++i) {
        double beta = static_cast<double>(i) / (n - 1) * 3.141592653589793;
        x[i] = (1.0 - std::cos(beta)) / 2.0;
    }

    // Thickness
    double yt[200];
    for (int i = 0; i < n; ++i) {
        yt[i] = 5.0 * t * (
            0.2969 * std::sqrt(x[i])
            - 0.1260 * x[i]
            - 0.3516 * x[i] * x[i]
            + 0.2843 * x[i] * x[i] * x[i]
            - 0.1015 * x[i] * x[i] * x[i] * x[i]
        );
    }

    // Camber line
    double yc[200] = {0}, dyc_dx[200] = {0};
    if (m > 1e-10 && p > 1e-10) {
        for (int i = 0; i < n; ++i) {
            if (x[i] < p) {
                yc[i] = (m / (p * p)) * (2.0 * p * x[i] - x[i] * x[i]);
                dyc_dx[i] = (2.0 * m / (p * p)) * (p - x[i]);
            } else {
                yc[i] = (m / ((1.0 - p) * (1.0 - p))) * (
                    (1.0 - 2.0 * p) + 2.0 * p * x[i] - x[i] * x[i]
                );
                dyc_dx[i] = (2.0 * m / ((1.0 - p) * (1.0 - p))) * (p - x[i]);
            }
        }
    }

    // Upper and lower surfaces
    double upper_x[200], upper_y[200];
    double lower_x[200], lower_y[200];
    for (int i = 0; i < n; ++i) {
        double theta = std::atan(dyc_dx[i]);
        upper_x[i] = x[i] - yt[i] * std::sin(theta);
        upper_y[i] = yc[i] + yt[i] * std::cos(theta);
        lower_x[i] = x[i] + yt[i] * std::sin(theta);
        lower_y[i] = yc[i] - yt[i] * std::cos(theta);
    }

    // Build closed polygon: upper surface (TE->LE) then lower (LE->TE)
    int idx = 0;
    for (int i = n - 1; i >= 0 && idx < max_pts; --i) {
        poly_x[idx] = upper_x[i];
        poly_y[idx] = upper_y[i];
        idx++;
    }
    for (int i = 1; i < n && idx < max_pts; ++i) {
        poly_x[idx] = lower_x[i];
        poly_y[idx] = lower_y[i];
        idx++;
    }
    return idx; // number of vertices
}

// --------------------------------------------------------------------------
// Place polygon obstacle
// --------------------------------------------------------------------------
static void place_polygon(const double* poly_x, const double* poly_y, int n_verts) {
    for (int y = 0; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {
            if (point_in_polygon(static_cast<double>(x),
                                 static_cast<double>(y),
                                 poly_x, poly_y, n_verts)) {
                obstacle[node_index(x, y)] = 1;
            }
        }
    }
}

// --------------------------------------------------------------------------
// Transform points (rotate, scale, translate)
// --------------------------------------------------------------------------
static void transform_points(double* pts_x, double* pts_y, int n,
    double cx_t, double cy_t, double scale, double angle_deg)
{
    double angle_rad = angle_deg * 3.141592653589793 / 180.0;
    double cos_a = std::cos(angle_rad);
    double sin_a = std::sin(angle_rad);
    for (int i = 0; i < n; ++i) {
        double xs = pts_x[i] * scale;
        double ys = pts_y[i] * scale;
        double xr = xs * cos_a - ys * sin_a;
        double yr = xs * sin_a + ys * cos_a;
        pts_x[i] = xr + cx_t;
        pts_y[i] = yr + cy_t;
    }
}

// --------------------------------------------------------------------------
// Initialize obstacle based on shape_type
// --------------------------------------------------------------------------
static void place_shape() {
    // Clear obstacles
    std::fill(obstacle, obstacle + NN, 0);

    double poly_x[400], poly_y[400];
    int n_verts = 0;

    switch (shape_type) {
        case 0: { // Cylinder
            int icx = static_cast<int>(shape_cx);
            int icy = static_cast<int>(shape_cy);
            int rad = static_cast<int>(shape_size);
            for (int y = 0; y < NY; ++y) {
                for (int x = 0; x < NX; ++x) {
                    double dx = static_cast<double>(x - icx);
                    double dy = static_cast<double>(y - icy);
                    if (std::sqrt(dx * dx + dy * dy) < rad) {
                        obstacle[node_index(x, y)] = 1;
                    }
                }
            }
            return;
        }
        case 1: { // Square (axis-aligned)
            int icx = static_cast<int>(shape_cx);
            int icy = static_cast<int>(shape_cy);
            int half = static_cast<int>(shape_size);
            int x0 = std::max(0, icx - half);
            int x1 = std::min(NX - 1, icx + half);
            int y0 = std::max(0, icy - half);
            int y1 = std::min(NY - 1, icy + half);
            for (int y = y0; y <= y1; ++y) {
                for (int x = x0; x <= x1; ++x) {
                    obstacle[node_index(x, y)] = 1;
                }
            }
            return;
        }
        case 2: { // Diamond (45-degree rotated square)
            double half = shape_size;
            n_verts = 4;
            poly_x[0] = 0; poly_y[0] = half;
            poly_x[1] = half; poly_y[1] = 0;
            poly_x[2] = 0; poly_y[2] = -half;
            poly_x[3] = -half; poly_y[3] = 0;
            transform_points(poly_x, poly_y, n_verts, shape_cx, shape_cy, 1.0, 0.0);
            break;
        }
        case 3: { // Star (5-pointed)
            n_verts = 10;
            double outer_r = shape_size;
            double inner_r = shape_size * 0.4;
            for (int i = 0; i < 10; ++i) {
                double angle = -3.141592653589793 / 2.0 + static_cast<double>(i) * 3.141592653589793 / 5.0;
                double r = (i % 2 == 0) ? outer_r : inner_r;
                poly_x[i] = r * std::cos(angle);
                poly_y[i] = r * std::sin(angle);
            }
            transform_points(poly_x, poly_y, n_verts, shape_cx, shape_cy, 1.0, 0.0);
            break;
        }
        case 4: { // NACA airfoil
            int n_coords = naca_coords(airfoil_series, poly_x, poly_y, 400);
            n_verts = n_coords;
            transform_points(poly_x, poly_y, n_verts,
                shape_cx, shape_cy, shape_size, -airfoil_aoa);
            break;
        }
    }

    if (n_verts > 0) {
        place_polygon(poly_x, poly_y, n_verts);
    }
}

// --------------------------------------------------------------------------
// Boundary conditions: Zou/He inlet (x=0)
// --------------------------------------------------------------------------
static void enforce_inflow() {
    for (int y = 0; y < NY; ++y) {
        int idx = node_index(0, y);
        if (obstacle[idx]) continue;
        double rho, u_cur, v_cur;
        compute_macros(&f[idx * 9], rho, u_cur, v_cur);
        // Zou/He: compute rho from known distributions
        double rho_in = (f[idx * 9 + 0] + f[idx * 9 + 2] + f[idx * 9 + 4]
            + 2.0 * (f[idx * 9 + 3] + f[idx * 9 + 6] + f[idx * 9 + 7]))
            / (1.0 - 0.0);  // u_inflow = 0 => simplified: rho known from equilibrium
        // Simplified: set unknown distributions (i=1,5,8) to equilibrium
        f[idx * 9 + 1] = compute_equilibrium(1, rho_in, u_inflow, 0.0);
        f[idx * 9 + 5] = compute_equilibrium(5, rho_in, u_inflow, 0.0);
        f[idx * 9 + 8] = compute_equilibrium(8, rho_in, u_inflow, 0.0);
    }
}

// --------------------------------------------------------------------------
// Boundary conditions: convective outlet (x=NX-1)
// --------------------------------------------------------------------------
static void enforce_outflow() {
    for (int y = 0; y < NY; ++y) {
        int idx = node_index(NX - 1, y);
        if (obstacle[idx]) continue;
        // Zero-gradient extrapolation for all distributions
        for (int i = 0; i < 9; ++i) {
            f[idx * 9 + i] = f[(idx - 1) * 9 + i];
        }
    }
}

// --------------------------------------------------------------------------
// Execute one time step (collide + stream + BC + force extraction)
// --------------------------------------------------------------------------
static void step() {
    // --- Collide ---
    for (int y = 0; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {
            int idx = node_index(x, y);
            if (obstacle[idx]) continue;

            double rho, u, v;
            compute_macros(&f[idx * 9], rho, u, v);

            for (int i = 0; i < 9; ++i) {
                double feq = compute_equilibrium(i, rho, u, v);
                f_next[idx * 9 + i] = f[idx * 9 + i] + (feq - f[idx * 9 + i]) / tau;
            }
        }
    }

    // --- Stream with bounce-back ---
    std::copy(f_next, f_next + ND, f);

    for (int y = 0; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {
            int idx = node_index(x, y);
            if (obstacle[idx]) continue;

            for (int i = 0; i < 9; ++i) {
                int nx = x + cx[i];
                int ny = y + cy[i];
                // Periodic y boundaries
                if (ny < 0) ny += NY;
                if (ny >= NY) ny -= NY;
                // Outlet: outflow is OK
                if (nx < 0 || nx >= NX) continue;

                int nbr = node_index(nx, ny);
                if (obstacle[nbr]) {
                    // Bounce-back: reflect direction
                    f_next[idx * 9 + bounce_back[i]] = f[idx * 9 + i];
                }
            }
        }
    }

    // Swap buffers
    std::copy(f_next, f_next + ND, f);

    // --- Boundary conditions ---
    enforce_inflow();
    enforce_outflow();

    // --- Force extraction (momentum exchange) ---
    std::fill(fx_cyl, fx_cyl + NN, 0.0);
    std::fill(fy_cyl, fy_cyl + NN, 0.0);

    for (int y = 0; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {
            int idx = node_index(x, y);
            if (obstacle[idx]) continue;

            for (int i = 0; i < 9; ++i) {
                int nx = x + cx[i];
                int ny = y + cy[i];
                if (ny < 0) ny += NY;
                if (ny >= NY) ny -= NY;
                if (nx < 0 || nx >= NX) continue;

                int nbr = node_index(nx, ny);
                if (obstacle[nbr]) {
                    double f_boundary = f[idx * 9 + bounce_back[i]];
                    fx_cyl[idx] += cx[i] * 2.0 * f_boundary;
                    fy_cyl[idx] += cy[i] * 2.0 * f_boundary;
                }
            }
        }
    }

    // --- Compute macroscopic fields for JS rendering ---
    for (int y = 0; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {
            int idx = node_index(x, y);
            if (obstacle[idx]) {
                u_field[idx] = 0.0;
                v_field[idx] = 0.0;
                rho_field[idx] = 1.0;
            } else {
                compute_macros(&f[idx * 9], rho_field[idx], u_field[idx], v_field[idx]);
            }
        }
    }

    // Sum forces and compute Cd/Cl
    double fx_total = 0.0, fy_total = 0.0;
    for (int n = 0; n < NN; ++n) {
        fx_total += fx_cyl[n];
        fy_total += fy_cyl[n];
    }
    double ref_len = shape_size * 2.0; // characteristic length
    if (shape_type == 1 || shape_type == 2) ref_len = shape_size * 2.0; // square/diamond: width
    if (shape_type == 4) ref_len = shape_size; // airfoil: chord
    cd_last = 2.0 * fx_total / (ref_len * u_inflow * u_inflow);
    cl_last = 2.0 * fy_total / (ref_len * u_inflow * u_inflow);

    step_count++;
}

// --------------------------------------------------------------------------
// Exported C API
// --------------------------------------------------------------------------
#ifdef __EMSCRIPTEN__
extern "C" {

EMSCRIPTEN_KEEPALIVE
void wasm_init(int nx, int ny, double u_in, double tau_in) {
    (void)nx; (void)ny; // fixed grid
    u_inflow = u_in;
    tau = tau_in;
    step_count = 0;

    // Initialize with equilibrium at uniform flow
    for (int n = 0; n < NN; ++n) {
        for (int i = 0; i < 9; ++i) {
            f[n * 9 + i] = compute_equilibrium(i, 1.0, u_inflow, 0.0);
            f_next[n * 9 + i] = f[n * 9 + i];
        }
    }

    std::fill(obstacle, obstacle + NN, 0);
    std::fill(fx_cyl, fx_cyl + NN, 0.0);
    std::fill(fy_cyl, fy_cyl + NN, 0.0);
    cd_last = 0.0;
    cl_last = 0.0;
}

EMSCRIPTEN_KEEPALIVE
void wasm_set_shape(int type, double param1, double param2, double param3, double param4) {
    // param1 = cx, param2 = cy, param3 = size, param4 = aoa (airfoil) / airfoil_series (airfoil)
    shape_type = type;
    shape_cx = param1;
    shape_cy = param2;
    shape_size = param3;
    if (type == 4) {
        airfoil_series = static_cast<int>(param4);
        // param4 is series for airfoil (e.g. 12 for 0012)
    }
    // Re-initialize flow and place shape
    wasm_init(NX, NY, u_inflow, tau);
    place_shape();
}

EMSCRIPTEN_KEEPALIVE
void wasm_set_tau(double tau_in) {
    tau = tau_in;
}

EMSCRIPTEN_KEEPALIVE
void wasm_step() {
    step();
}

EMSCRIPTEN_KEEPALIVE
double wasm_get_cd() {
    return cd_last;
}

EMSCRIPTEN_KEEPALIVE
double wasm_get_cl() {
    return cl_last;
}

EMSCRIPTEN_KEEPALIVE
int wasm_get_step() {
    return step_count;
}

// Pointers to shared memory for JS access
EMSCRIPTEN_KEEPALIVE
double* wasm_get_f_ptr() {
    return f;
}

EMSCRIPTEN_KEEPALIVE
int* wasm_get_obstacle_ptr() {
    return obstacle;
}

EMSCRIPTEN_KEEPALIVE
double* wasm_get_u_ptr() {
    return u_field;
}

EMSCRIPTEN_KEEPALIVE
double* wasm_get_v_ptr() {
    return v_field;
}

EMSCRIPTEN_KEEPALIVE
double* wasm_get_rho_ptr() {
    return rho_field;
}

EMSCRIPTEN_KEEPALIVE
int wasm_get_nx() { return NX; }

EMSCRIPTEN_KEEPALIVE
int wasm_get_ny() { return NY; }

} // extern "C"
#endif // __EMSCRIPTEN__
