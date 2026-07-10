#pragma once
#include <vector>
#include <utility>
#include <cmath>
#include <algorithm>

// ==========================================================================
// 2D GEOMETRY UTILITIES for obstacle placement
// ==========================================================================

// ------------------------------------------------------------------
// NACA 4-digit airfoil coordinate generation
// Returns a closed polygon (upper + lower surface, TE to LE to TE).
//   series: 4-digit code, e.g. 0012 (symmetric, 12% thick), 2412 (2% camber)
//   n: number of points (recommended 200+)
// ------------------------------------------------------------------
inline std::vector<std::pair<double,double>> naca_coords(int series, int n = 200) {
    int m_code = series / 1000;            // first digit: max camber in % of chord
    int p_code = (series / 100) % 10;       // second digit: position of max camber in tenths
    int t_code = series % 100;              // last two digits: max thickness in % of chord

    double m = static_cast<double>(m_code) / 100.0;
    double p = static_cast<double>(p_code) / 10.0;
    double t = static_cast<double>(t_code) / 100.0;

    // Cosine spacing for LE/TE clustering
    std::vector<double> x(n);
    for (int i = 0; i < n; ++i) {
        double beta = static_cast<double>(i) / (n - 1) * M_PI;
        x[i] = (1.0 - std::cos(beta)) / 2.0;
    }

    // Thickness distribution
    std::vector<double> yt(n);
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
    std::vector<double> yc(n, 0.0);
    std::vector<double> dyc_dx(n, 0.0);
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

    // Combine perpendicular to camber line
    std::vector<std::pair<double,double>> upper(n), lower(n);
    for (int i = 0; i < n; ++i) {
        double theta = std::atan(dyc_dx[i]);
        double xt = x[i];
        double xu = xt - yt[i] * std::sin(theta);
        double yu = yc[i] + yt[i] * std::cos(theta);
        double xl = xt + yt[i] * std::sin(theta);
        double yl = yc[i] - yt[i] * std::cos(theta);
        upper[i] = {xu, yu};
        lower[i] = {xl, yl};
    }

    // Build closed polygon: trailing edge -> upper surface -> leading edge -> lower surface -> trailing edge
    std::vector<std::pair<double,double>> result;
    result.reserve(n * 2);

    // Upper surface: from LE (index 0) to TE (index n-1), reversed so we go TE->LE
    // Actually we want a clockwise polygon: start at TE, go along upper to LE, then back along lower to TE
    for (int i = n - 1; i >= 0; --i) {
        result.push_back(upper[i]);
    }
    // Lower surface: skip first point (LE, already included) and last point (TE, already included)
    for (int i = 1; i < n; ++i) {
        result.push_back(lower[i]);
    }

    return result;
}

// ------------------------------------------------------------------
// 2D transformation: rotate, scale, translate
// ------------------------------------------------------------------
inline void transform_points(
    std::vector<std::pair<double,double>>& pts,
    double cx, double cy, double scale, double angle_deg)
{
    double angle_rad = angle_deg * M_PI / 180.0;
    double cos_a = std::cos(angle_rad);
    double sin_a = std::sin(angle_rad);

    for (auto& pt : pts) {
        // Scale around origin
        double xs = pt.first * scale;
        double ys = pt.second * scale;
        // Rotate
        double xr = xs * cos_a - ys * sin_a;
        double yr = xs * sin_a + ys * cos_a;
        // Translate
        pt.first = xr + cx;
        pt.second = yr + cy;
    }
}

// ------------------------------------------------------------------
// Ray-casting point-in-polygon test
// Returns true if point (px, py) is inside the closed polygon.
// ------------------------------------------------------------------
inline bool point_in_polygon(double px, double py,
    const std::vector<std::pair<double,double>>& poly)
{
    bool inside = false;
    size_t n = poly.size();
    for (size_t i = 0, j = n - 1; i < n; j = i++) {
        double xi = poly[i].first, yi = poly[i].second;
        double xj = poly[j].first, yj = poly[j].second;

        bool intersect = ((yi > py) != (yj > py))
            && (px < (xj - xi) * (py - yi) / (yj - yi) + xi);
        if (intersect) inside = !inside;
    }
    return inside;
}
