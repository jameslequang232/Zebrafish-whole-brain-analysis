"""
register_mapzebrain_to_template.py
===================================
Registers the MapZebrain GCaMP reference brain to the 300 µm data mean brain
(template_mean_brain.nii.gz), then warps every region mask (.tif) from the
MapZebrain regions folder into that same template space.

Pipeline mirrors run_registration_syn.py exactly:
    reorient_image2 (LPS)
    → resample_image (isotropic TARGET_SPACING µm, default 3.0)
    → iMath_normalize
    → get_mask(cleanup=True) + iMath("MD", 1)
    → affine_initializer(search_factor=20, radian_fraction=0.1)
    → syn_registration() from registration.py
        (SyNRA, aff_metric=MI, syn_metric=CC, shrink=(8,4,2,1),
         iters=(120,80,40,0), linear interpolator)
    → apply_transforms at FULL RESOLUTION on the original fixed grid
        • welchWindowedSinc  for the reference brain (continuous intensity)
        • genericLabel       for each binary region mask (preserves labels)
    → QC overlay PDF saved (gray = fixed, viridis = warped MapZebrain)

Inputs
------
Moving (MapZebrain GCaMP brain):
    /home/ychiu/yun/lightsheet/mapzebrain/ants_template/
    T_AVG_HuCH2BGCaMP2-tg_ch0.tif
    Stored (Z, Y, X) in the .tif; transposed to (X, Y, Z) on load.
    Spacing: (0.9940709, 0.9939616, 1.0) µm — from ZBrain paper.

Fixed (data mean brain):
    /home/ychiu/yun/lightsheet/analysis_output/registration/
    template_mean_brain.nii.gz

Region masks:
    /home/ychiu/yun/lightsheet/mapzebrain/regions/*.tif
    Each mask is (Z, Y, X) binary .tif; transposed to (X, Y, Z) on load.
    Physical spacing is assumed identical to the MapZebrain reference brain.

Outputs  (all under OUTPUT_DIR)
-------
    mapzebrain_to_mean_warp.nii.gz              — forward warp field
    mapzebrain_to_mean_affine.mat               — forward affine matrix
    mapzebrain_to_mean_registered.nii.gz        — warped MapZebrain ref (downsampled)
    mapzebrain_to_mean_registered_final.nii.gz  — warped MapZebrain ref (full-res)
    qc_overlay.pdf                              — registration QC
    <region_name>_in_template.nii.gz            — one per region mask

Usage
-----
    python registration/register_mapzebrain_to_template.py [--dry-run] [--overwrite]

    --dry-run   : validate paths, then exit
    --overwrite : redo registration even if transform files already exist

Location:
    ~/Zebrafish-whole-brain-analysis/registration/register_mapzebrain_to_template.py
"""

import argparse
import multiprocessing
import os
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless — safe for sbatch
import matplotlib.pyplot as plt
import numpy as np
from skimage import io as skio

# ── ITK threads + TMPDIR (same as run_registration_syn.py) ──────────────────
try:
    n_threads = str(len(os.sched_getaffinity(0)) - 2)
except AttributeError:
    n_threads = str(multiprocessing.cpu_count() - 2)

HPC_USERNAME = "ychiu"
os.environ["TMPDIR"] = f"/resnick/scratch/{HPC_USERNAME}/tmp_ants"
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = n_threads
os.makedirs(os.environ["TMPDIR"], exist_ok=True)

import ants

# ── Import syn_registration from the existing registration.py module ─────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from registration import load_mapzebrain_reference_brain, syn_registration

print(f"✅ TMPDIR={os.environ['TMPDIR']} | ITK threads={n_threads}")


# ─────────────────────────────────────────────────────────────────────────────
# PATHS  — edit here if layout changes
# ─────────────────────────────────────────────────────────────────────────────

# MapZebrain GCaMP reference brain (moving)
MAPZEBRAIN_BRAIN = (
    "/home/ychiu/yun/lightsheet/"
    "mapzebrain/ants_template/T_AVG_HuCH2BGCaMP2-tg_ch0.tif"
)

# Voxel spacing (µm) for the MapZebrain reference — from ZBrain paper
# (X, Y, Z) after transposing from raw (Z, Y, X) tif
MAPZEBRAIN_SPACING = (0.9940709, 0.9939616, 1.0)

# Data mean brain (fixed) — the 300 µm Zstack template
FIXED_BRAIN = (
    "/home/ychiu/yun/lightsheet/analysis_output/registration/"
    "template_mean_brain.nii.gz"
)

# MapZebrain region masks directory (binary .tif files, one per region)
MASKS_DIR = "/home/ychiu/yun/lightsheet/mapzebrain/regions/"

# Where ALL outputs of this script are written
OUTPUT_DIR = (
    "/home/ychiu/yun/lightsheet/analysis_output/registration/"
    "300um_template/mapzebrain_in_template/"
)

# Isotropic spacing used during registration (µm) — matches run_registration_syn.py
TARGET_SPACING = (3.0, 3.0, 3.0)


# ─────────────────────────────────────────────────────────────────────────────
# QC HELPER  (matches save_qc_overlay in run_registration_syn.py)
# ─────────────────────────────────────────────────────────────────────────────

def _norm_slice(arr):
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-9)


def save_qc_overlay(fixed_atlas, warped_path, out_dir, n_planes=10):
    """
    Save a QC overlay PDF:
        gray   = fixed mean brain
        viridis = warped MapZebrain reference brain
    """
    if not os.path.exists(warped_path):
        print(f"  ⚠️  Warped image not found for QC: {warped_path}")
        return

    warped   = ants.image_read(warped_path, pixeltype="float")
    atlas_np = fixed_atlas.numpy()
    reg_np   = warped.numpy()

    # Resample to atlas grid if grids don't match (shouldn't happen, but safe)
    if (atlas_np.shape != reg_np.shape or
        not np.allclose(warped.spacing, fixed_atlas.spacing)):
        print("  ⚠️  Grid mismatch — resampling warped to atlas grid for QC.")
        warped = ants.resample_image_to_target(warped, fixed_atlas, interp_type="linear")
        reg_np = warped.numpy()

    n_z      = atlas_np.shape[2]
    z_slices = np.linspace(0, n_z - 1, min(n_planes, n_z), dtype=int)
    ncols    = 5
    nrows    = int(np.ceil(len(z_slices) / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4, nrows * 8),
                             squeeze=False)
    axes = axes.ravel()
    for i, z in enumerate(z_slices):
        ax = axes[i]
        ax.imshow(_norm_slice(atlas_np[:, :, z].T), cmap="gray",
                  origin="lower", aspect="auto")
        ax.imshow(_norm_slice(reg_np[:, :, z].T),   cmap="viridis",
                  origin="lower", aspect="auto", alpha=0.7)
        ax.set_title(f"Z={z}", fontsize=9)
        ax.invert_yaxis()
        ax.axis("off")
    for j in range(i + 1, nrows * ncols):
        fig.delaxes(axes[j])

    fig.suptitle(
        "Registration QC: MapZebrain → data mean brain\n"
        "Gray = fixed (mean brain)  |  Green = warped MapZebrain",
        fontsize=12,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    pdf_path = os.path.join(out_dir, "qc_overlay.pdf")
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"  ✅ QC overlay saved: {pdf_path}")



# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — REGISTER MapZebrain → data mean brain
# ─────────────────────────────────────────────────────────────────────────────

def register_mapzebrain(mapzebrain_img, fixed_img, output_dir, overwrite=False):
    """
    Run the full SyNRA pipeline and save transforms + warped brain.

    Returns dict: {warp, affine, warped}  — all absolute file paths.
    """
    warp_path   = os.path.join(output_dir, "mapzebrain_to_mean_warp.nii.gz")
    affine_path = os.path.join(output_dir, "mapzebrain_to_mean_affine.mat")
    warped_path = os.path.join(output_dir, "mapzebrain_to_mean_registered_final.nii.gz")

    if not overwrite and all(os.path.exists(p) for p in [warp_path, affine_path, warped_path]):
        print("  ✅ Transforms already exist — skipping registration "
              "(use --overwrite to redo).")
        print(f"     warp   : {warp_path}")
        print(f"     affine : {affine_path}")
        print(f"     warped : {warped_path}")
        return {"warp": warp_path, "affine": affine_path, "warped": warped_path}

    # ── 1a: Reorient to LPS ─────────────────────────────────────────────────
    print("\n[1a] Reorienting to LPS...")
    moving_lps = ants.reorient_image2(mapzebrain_img, orientation="LPS")
    fixed_lps  = ants.reorient_image2(fixed_img,      orientation="LPS")
    print(f"     moving shape={moving_lps.shape}, spacing={moving_lps.spacing}")
    print(f"     fixed  shape={fixed_lps.shape},  spacing={fixed_lps.spacing}")

    # ── 1b: Resample to isotropic TARGET_SPACING ────────────────────────────
    print(f"\n[1b] Resampling to {TARGET_SPACING} µm isotropic...")
    moving_ds = ants.resample_image(moving_lps, TARGET_SPACING,
                                    use_voxels=False, interp_type=1)
    fixed_ds  = ants.resample_image(fixed_lps,  TARGET_SPACING,
                                    use_voxels=False, interp_type=1)
    print(f"     moving resampled: {moving_ds.shape}")
    print(f"     fixed  resampled: {fixed_ds.shape}")

    # ── 1c: Normalize + brain masks ──────────────────────────────────────────
    print("\n[1c] Normalizing + making brain masks...")
    moving_norm = ants.iMath_normalize(moving_ds)
    fixed_norm  = ants.iMath_normalize(fixed_ds)

    fixed_mask  = ants.iMath(ants.get_mask(fixed_norm,  cleanup=True), "MD", 1)
    moving_mask = ants.iMath(ants.get_mask(moving_norm, cleanup=True), "MD", 1)

    # ── 1d: Affine initializer ───────────────────────────────────────────────
    print("\n[1d] Affine initializer (search_factor=20, radian_fraction=0.1)...")
    init_tx = ants.affine_initializer(
        fixed_norm, moving_norm,
        search_factor=20,
        radian_fraction=0.1,
    )
    print(f"     init_tx: {init_tx}")

    # ── 1e: SyNRA via syn_registration() from registration.py ────────────────
    print("\n[1e] SyNRA registration (MI affine + CC SyN)...")
    warp_path, affine_path = syn_registration(
        moving_norm=moving_norm,
        fixed_norm=fixed_norm,
        dir_ants_output=output_dir,
        file_name_warp="mapzebrain_to_mean_warp.nii.gz",
        file_name_affine="mapzebrain_to_mean_affine.mat",
        file_name_syn_registration="mapzebrain_to_mean_registered.nii.gz",
        init_tx=init_tx,
        fixed_mask=fixed_mask,
        moving_mask=moving_mask,
    )

    # ── 1f: Apply transforms at FULL RESOLUTION on original fixed grid ───────
    # Fixed reference = fixed_lps (full resolution, LPS-oriented)
    # Moving          = moving_lps (full resolution, LPS-oriented)
    # Interpolator    = welchWindowedSinc (continuous intensity image)
    print("\n[1f] Applying composite transform at full resolution "
          "(welchWindowedSinc)...")
    warped_fullres = ants.apply_transforms(
        fixed=fixed_lps,
        moving=moving_lps,
        transformlist=[warp_path, affine_path],
        interpolator="welchWindowedSinc",
        verbose=True,
    )
    ants.image_write(warped_fullres, warped_path)
    print(f"  ✅ Full-res warped brain saved: {warped_path}")

    return {"warp": warp_path, "affine": affine_path, "warped": warped_path}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — TRANSFORM REGION MASKS
# ─────────────────────────────────────────────────────────────────────────────

def transform_region_masks(masks_dir, fixed_img, transforms, output_dir,
                            overwrite=False):
    """
    Apply the MapZebrain→template transforms to every .tif region mask.

    Each mask tif is (Z, Y, X) binary → transposed to (X, Y, Z) with the
    same spacing as the MapZebrain reference brain.

    Uses 'genericLabel' interpolation (nearest-neighbour) to preserve
    integer / binary mask values.

    Returns list of output .nii.gz paths.
    """
    masks_path = Path(masks_dir)
    tif_files  = sorted(list(masks_path.glob("*.tif")) +
                        list(masks_path.glob("*.tiff")))

    if not tif_files:
        print(f"  ⚠️  No .tif files found in {masks_dir}")
        return []

    print(f"\n  {len(tif_files)} region mask(s) found in {masks_dir}")

    # Full-resolution fixed image in LPS orientation
    # — same grid that was used as fixed in apply_transforms above
    fixed_lps = ants.reorient_image2(fixed_img, orientation="LPS")

    transform_list = [transforms["warp"], transforms["affine"]]

    out_paths = []
    for i, tif_path in enumerate(tif_files):
        stem     = tif_path.stem
        out_path = os.path.join(output_dir, f"{stem}_in_template.nii.gz")

        if not overwrite and os.path.exists(out_path):
            print(f"  [{i+1:3d}/{len(tif_files)}] ⏩ {stem}: exists, skipping.")
            out_paths.append(out_path)
            continue

        print(f"  [{i+1:3d}/{len(tif_files)}] {stem} …", end=" ", flush=True)

        # Load mask and set physical space identical to MapZebrain reference
        raw       = skio.imread(str(tif_path))
        transposed = np.transpose(raw.astype(np.float32), axes=(2, 1, 0))
        mask_img  = ants.from_numpy(transposed)
        mask_img.set_spacing(MAPZEBRAIN_SPACING)

        # Reorient to LPS — must match the orientation used during registration
        mask_lps = ants.reorient_image2(mask_img, orientation="LPS")

        # Apply the same forward transforms with nearest-neighbour for labels
        mask_warped = ants.apply_transforms(
            fixed=fixed_lps,
            moving=mask_lps,
            transformlist=transform_list,
            interpolator="genericLabel",
            verbose=False,
        )

        ants.image_write(mask_warped, out_path)
        print("saved.")
        out_paths.append(out_path)

    return out_paths


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Register MapZebrain GCaMP brain → data mean brain "
            "and warp all region masks into template space."
        )
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate paths and exit without running ANTs.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-run registration even if transform files already exist.",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("register_mapzebrain_to_template.py")
    print("=" * 70)
    print(f"  Moving (MapZebrain) : {MAPZEBRAIN_BRAIN}")
    print(f"  Fixed (data mean)   : {FIXED_BRAIN}")
    print(f"  Region masks dir    : {MASKS_DIR}")
    print(f"  Output dir          : {OUTPUT_DIR}")
    print(f"  Target spacing      : {TARGET_SPACING} µm")
    print(f"  Overwrite           : {args.overwrite}")

    # ── Validate paths ───────────────────────────────────────────────────────
    missing = [p for p in [MAPZEBRAIN_BRAIN, FIXED_BRAIN, MASKS_DIR]
               if not os.path.exists(p)]
    if missing:
        print("\n❌ Missing paths:")
        for p in missing:
            print(f"   {p}")
        sys.exit(1)

    print("\n✅ All input paths found.")

    if args.dry_run:
        print("--dry-run: exiting without running registration.")
        sys.exit(0)

    # ── Create output directory ───────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load images ───────────────────────────────────────────────────────────
    print("\n── Loading images ──────────────────────────────────────────────")
    print("Loading MapZebrain reference brain (moving)...")
    # load_mapzebrain_reference_brain() constructs the path internally as:
    #   dir_voluseg/mapzebrain/ants_template/T_AVG_HuCH2BGCaMP2-tg_ch0.tif
    MAPZEBRAIN_DIR = "/home/ychiu/yun/lightsheet/"
    mapzebrain_img = load_mapzebrain_reference_brain(MAPZEBRAIN_DIR)

    print("\nLoading data mean brain (fixed)...")
    fixed_img = ants.image_read(FIXED_BRAIN)
    print(f"  Fixed shape={fixed_img.shape}, spacing={fixed_img.spacing}")

    # ── Step 1: Register ──────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("STEP 1 — Register MapZebrain → data mean brain")
    print("─" * 70)
    transforms = register_mapzebrain(
        mapzebrain_img=mapzebrain_img,
        fixed_img=fixed_img,
        output_dir=OUTPUT_DIR,
        overwrite=args.overwrite,
    )

    # ── QC overlay ────────────────────────────────────────────────────────────
    print("\n── Saving QC overlay...")
    save_qc_overlay(
        fixed_atlas=fixed_img,
        warped_path=transforms["warped"],
        out_dir=OUTPUT_DIR,
        n_planes=10,
    )

    # ── Step 2: Transform region masks ────────────────────────────────────────
    print("\n" + "─" * 70)
    print("STEP 2 — Transform region masks into template space")
    print("─" * 70)
    out_paths = transform_region_masks(
        masks_dir=MASKS_DIR,
        fixed_img=fixed_img,
        transforms=transforms,
        output_dir=OUTPUT_DIR,
        overwrite=args.overwrite,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("✅ Done!")
    print(f"   Warp            : {transforms['warp']}")
    print(f"   Affine          : {transforms['affine']}")
    print(f"   Warped brain    : {transforms['warped']}")
    print(f"   Region masks    : {len(out_paths)} saved to {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
