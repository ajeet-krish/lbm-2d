#!/bin/bash
# Batch postprocess all simulation configs and copy images to docs
# Usage: bash scripts/batch_postprocess.sh

set -e

PROJROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$PROJROOT/pinn/.venv/bin/python3"
POST="$PROJROOT/scripts/postprocess.py"
DOCSIMAGES="$PROJROOT/docs/assets/images"

# Map: output_dir => docs_image_subdir/image_prefix
# Format: "output_relative_path|docs_subdir|image_prefix"
CONFIGS=(
  # Cylinder
  "output/cylinder/re100|cylinder|re100"
  "output/cylinder/re200|cylinder|re200"
  # Cavity
  "output/cavity/re100|cavity|re100"
  "output/cavity/re400|cavity|re400"
  "output/cavity/re1000|cavity|re1000"
  # Flat plate
  "output/flatplate/re500_aoa0|flatplate|re500_aoa0"
  "output/flatplate/re1000_aoa-10|flatplate|re1000_aoa-10"
  "output/flatplate/re1000_aoa-5|flatplate|re1000_aoa-5"
  "output/flatplate/re1000_aoa0|flatplate|re1000_aoa0"
  "output/flatplate/re1000_aoa5|flatplate|re1000_aoa5"
  "output/flatplate/re1000_aoa10|flatplate|re1000_aoa10"
  "output/flatplate/re1000_aoa15|flatplate|re1000_aoa15"
  "output/flatplate/re2000_aoa0|flatplate|re2000_aoa0"
  # Step
  "output/step/re100|step|re100"
  "output/step/re200|step|re200"
  "output/step/re400|step|re400"
  # Square cylinder
  "output/square_cylinder/re200|square_cylinder|re200"
  # Orifice plate
  "output/orifice_plate/re100_1p1h|orifice_plate|1p1h"
  "output/orifice_plate/re100_1p3h|orifice_plate|1p3h"
  "output/orifice_plate/re100_2p|orifice_plate|2p"
  "output/orifice_plate/re100_3p|orifice_plate|3p"
  # Periodic hills
  "output/periodic_hills/re100|periodic_hills|re100"
  "output/periodic_hills/re1000|periodic_hills|re1000"
  "output/periodic_hills/re2800|periodic_hills|re2800"
  # Cylinder near wall
  "output/cylinder_near_wall/re100_gap10|cylinder_near_wall|gap10"
  "output/cylinder_near_wall/re100_gap20|cylinder_near_wall|gap20"
  "output/cylinder_near_wall/re100_gap40|cylinder_near_wall|gap40"
  # Side-by-side
  "output/side_by_side/re100_sd20|side_by_side|sd20"
  "output/side_by_side/re100_sd30|side_by_side|sd30"
  "output/side_by_side/re100_sd50|side_by_side|sd50"
  # Rotating cylinder
  "output/rotating_cylinder/re100_w5|rotating_cylinder|w5"
  "output/rotating_cylinder/re100_w10|rotating_cylinder|w10"
  "output/rotating_cylinder/re100_w20|rotating_cylinder|w20"
  # Urban side (flat naming)
  "output/urban_side_ar3_re100|urban/side|re0.3"
  "output/urban_side_ar5_re100|urban/side|re0.5"
  "output/urban_side_ar6_3b_re100|urban/side|re0.6_3b"
  "output/urban_side_ar8_re100|urban/side|re0.8"
  # Urban topdown
  "output/urban_topdown_re100|urban/topdown|re100"
  # Urban topdown horizontal
  "output/urban_topdown_h_re100|urban/topdown_h|re100"
  # Urban downwash
  "output/urban/downwash_re100|urban/downwash|re100"
  # Ribs
  "output/ribs/re50|ribs|re50"
  "output/ribs/re100|ribs|re100"
  "output/ribs/re200|ribs|re200"
)

SUCCESS=0
FAIL=0
SKIP=0

for entry in "${CONFIGS[@]}"; do
  IFS='|' read -r outdir docsub prefix <<< "$entry"
  fullout="$PROJROOT/$outdir"

  # Check if frames exist
  if [ ! -d "$fullout/frames" ]; then
    echo "SKIP (no frames/): $outdir"
    SKIP=$((SKIP + 1))
    continue
  fi

  nframes=$(ls "$fullout"/frames/frame_*.json 2>/dev/null | wc -l | tr -d ' ')
  if [ "$nframes" -eq 0 ]; then
    echo "SKIP (0 frames): $outdir"
    SKIP=$((SKIP + 1))
    continue
  fi

  echo "Processing: $outdir ($nframes frames) -> $docsub/$prefix"

  # Run postprocess
  if ! "$PY" "$POST" "$fullout" --split --last-only 2>&1; then
    echo "  FAIL: postprocess failed for $outdir"
    FAIL=$((FAIL + 1))
    continue
  fi

  # Find the generated contour and streamlines PNGs (last frame number varies)
  contour_png=$(ls -t "$fullout"/contour_*.png 2>/dev/null | head -1)
  stream_png=$(ls -t "$fullout"/streamlines_*.png 2>/dev/null | head -1)

  if [ -z "$contour_png" ] || [ -z "$stream_png" ]; then
    echo "  FAIL: no PNGs generated for $outdir"
    FAIL=$((FAIL + 1))
    continue
  fi

  # Copy to docs
  destdir="$DOCSIMAGES/$docsub"
  mkdir -p "$destdir"
  cp "$contour_png" "$destdir/${prefix}_contour.png"
  cp "$stream_png" "$destdir/${prefix}_streamlines.png"
  echo "  OK: $destdir/${prefix}_{contour,streamlines}.png"
  SUCCESS=$((SUCCESS + 1))
done

echo ""
echo "=== SUMMARY ==="
echo "Success: $SUCCESS"
echo "Failed:  $FAIL"
echo "Skipped: $SKIP"
echo "Total:   ${#CONFIGS[@]}"
