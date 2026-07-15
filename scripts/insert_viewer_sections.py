#!/usr/bin/env python3
"""Insert the LBM Evolution + PINN Prediction viewer sections into all
non-cavity case HTML pages. Cavity already has the section (hand-built)."""

import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DOCS = os.path.join(ROOT, "docs")

# Per-case configuration for the viewer.
# (page, data_dir, param_label, grid_w, grid_h, [(label, file), ...])
CASES = [
    ("cylinder", "cylinder", "Re", 100, 38, [
        ("100", "100"), ("200", "200"), ("1000", "1000")]),
    ("step", "step", "Re", 100, 38, [
        ("100", "100"), ("200", "200"), ("400", "400")]),
    ("flat_plate", "flatplate", "Config", 100, 38, [
        ("AoA 0", "aoa0"), ("AoA 5", "aoa5"), ("AoA 10", "aoa10"),
        ("Re 500", "re500"), ("Re 2000", "re2000")]),
    ("orifice_plate", "orifice_plate", "Config", 100, 38, [
        ("1p1h", "1p1h"), ("1p3h", "1p3h"), ("2p", "2p"), ("3p", "3p")]),
    ("periodic_hills", "periodic_hills", "Re", 100, 38, [
        ("100", "100"), ("1000", "1000"), ("2800", "2800")]),
    ("square_cylinder", "square_cylinder", "Re", 100, 38, [
        ("200", "200")]),
    ("side_by_side", "side_by_side", "S/D", 100, 38, [
        ("2", "sd20"), ("3", "sd30"), ("5", "sd50")]),
    ("cylinder_near_wall", "cylinder_near_wall", "Gap", 100, 38, [
        ("10", "gap10"), ("20", "gap20"), ("40", "gap40")]),
    ("rotating_cylinder", "rotating_cylinder", "omega", 100, 38, [
        ("0.5", "w5"), ("1.0", "w10"), ("2.0", "w20")]),
    ("urban", "urban", "Config", 100, 50, [
        ("Side AR=0.3", "side_a03"), ("Side AR=0.5", "side_a05"),
        ("Side AR=0.8", "side_a08"), ("Top-down", "topdown"),
        ("Downwash", "downwash")]),
]


def section_html(case):
    page, data_dir, param_label, w, h, configs = case
    cfg_li = ",\n".join(
        "          { label: '%s', file: '%s' }" % (label, file) for label, file in configs
    )
    return f"""    <h2>Interactive Flow Evolution</h2>

    <p class="section-intro">
      Interactive side-by-side of the C++ LBM solver and the PINN surrogate. The
      LBM viewer animates the solver developing the flow from rest to steady state;
      the PINN viewer shows the parametric surrogate prediction. Velocity magnitude
      is rendered as a contour with streamlines overlaid.
    </p>

    <h3>LBM Evolution</h3>

    <p class="section-intro">
      The C++ MRT-LBM solver developing the flow from rest. Press Play to animate;
      use the scrubber to inspect any frame.
    </p>

    <div class="fv-stage">
      <canvas id="lbmCanvas" class="fv-canvas" width="{w}" height="{h}"></canvas>
    </div>

    <div class="fv-controls">
      <button class="fv-btn" id="lbmPlay">&#9654; Play</button>
      <input type="range" id="lbmScrubber" min="0" max="50" value="0" class="fv-slider">
      <span class="fv-frame" id="lbmFrameLabel">Frame 0 / 50</span>
    </div>

    <div class="fv-controls" id="lbmReGroup">
      <div class="fv-group"></div>
    </div>

    <h3>PINN Prediction</h3>

    <p class="section-intro">
      The parametric PINN surrogate predicts the steady-state flow directly from
      position and the case parameter. Training for this case is in progress; the
      placeholder below will be replaced by the live surrogate once trained
      (Phase 6.8, time-parametric PINN).
    </p>

    <div class="fv-stage">
      <canvas id="pinnCanvas" class="fv-canvas" width="{w}" height="{h}"></canvas>
    </div>

    <div class="fv-controls" id="pinnControls"></div>

    <p class="fv-note" id="fvStatus"></p>

"""


def script_block(case):
    page, data_dir, param_label, w, h, configs = case
    cfg_li = ",\n".join(
        "          { label: '%s', file: '%s' }" % (label, file) for label, file in configs
    )
    return f"""    <script src="assets/js/colormaps.js"></script>
    <script src="assets/js/flow-viewer.js"></script>
    <script src="assets/js/pinn-inference.js"></script>
    <script src="assets/js/viewer-common.js"></script>
    <script>
      initFlowViewerSections({{
        dataDir: 'assets/data/{data_dir}',
        paramLabel: '{param_label}',
        lbmConfigs: [
{cfg_li}
        ],
        pinn: null
      }});
    </script>
"""


def main():
    for case in CASES:
        page = case[0]
        path = os.path.join(DOCS, f"{page}.html")
        if not os.path.exists(path):
            print(f"SKIP {page}.html: not found")
            continue
        with open(path, "r") as f:
            html = f.read()

        if "initFlowViewerSections" in html or "lbmCanvas" in html:
            print(f"SKIP {page}.html: viewer already present")
            continue

        # Insert section before footer.
        footer_marker = '<footer class="site-footer">'
        if footer_marker not in html:
            print(f"SKIP {page}.html: no footer marker")
            continue
        html = html.replace(footer_marker, section_html(case) + "    " + footer_marker, 1)

        # Insert scripts before </body>.
        body_end = "</body>"
        if body_end not in html:
            print(f"SKIP {page}.html: no </body>")
            continue
        html = html.replace(body_end, script_block(case) + "\n" + body_end, 1)

        with open(path, "w") as f:
            f.write(html)
        print(f"UPDATED {page}.html")


if __name__ == "__main__":
    main()
