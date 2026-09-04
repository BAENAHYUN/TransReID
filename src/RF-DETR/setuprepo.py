"""
Repository structure setup script
======================

Move scattered files into the standard structure and prepare the repository for Git.

    python setup_repo.py              # Preview planned changes only (default)
    python setup_repo.py --apply      # Apply changes

What this script does:
  1) Create the folder structure and __init__.py files
  2) Find files by name and move them to their target locations
  3) Normalize the import path in qdrant_store.py
     (from src.RFDETR.pipeline.config -> from config)
  4) Create .gitignore and requirements.txt
  5) Report missing files

Safety checks:
  * Dry-run is the default. Files are changed only when --apply is provided.
  * Files already in the correct location are left untouched.
  * If a different file already exists at the destination, do not overwrite it; only warn.
  * If multiple files with the same name are found, stop and report the ambiguity.
"""

from __future__ import annotations

import argparse
import filecmp
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Target structure: filename -> destination (relative to project root)
# --------------------------------------------------------------------------- #
LAYOUT: Dict[str, str] = {
    # Root-level files
    "pipeline.yaml":        ".",
    "config.py":            ".",
    "registry.py":          ".",
    "router.py":            ".",
    "qdrant_store.py":      ".",
    "rfdetr_adapter.py":    ".",
    "build_db.py":          ".",

    # Embedders
    "base.py":              "embedders",
    "siglip2_embedder.py":  "embedders",
    "irra_embedder.py":     "embedders/human",
    "solider_embedder.py":  "embedders/human",
    "dinov2_embedder.py":   "embedders/object",

    # Detection
    "detect_rf.py":         "detect",

    # Evaluation files (not required for setup, but managed together)
    "eval_metrics.py":      "eval",
    "eval_i2i.py":          "eval",
    "eval_t2i.py":          "eval",
}

# Directories that must be Python packages (__init__.py required)
PACKAGE_DIRS = [
    "embedders",
    "embedders/human",
    "embedders/object",
]

# Directories that only need to exist
PLAIN_DIRS = [
    "detect",
    "eval",
    "weights/solider",
    "data/crops",
    "third_party",
]

# Directories excluded from file search (external repos / virtual envs / caches)
SKIP_DIRS = {
    ".git", "venv", ".venv", "env", "__pycache__", ".idea", ".vscode",
    "IRRA", "third_party", "node_modules", "eval_cache", ".mypy_cache",
    "site-packages",
}

# Optional files: useful to have, but not required for setup
OPTIONAL = {"eval_metrics.py", "eval_i2i.py", "eval_t2i.py", "build_db.py"}


GITIGNORE = """\
# --- Virtual environments / caches ---------------------------------------
venv/
.venv/
env/
__pycache__/
*.py[cod]
.mypy_cache/
.pytest_cache/
.ipynb_checkpoints/

# --- Model weights (hundreds of MB; GitHub 100 MB limit) ------------------
weights/
*.pth
*.pt
*.ckpt
*.safetensors
*.bin

# --- External repositories (clone separately) -----------------------------
IRRA/
third_party/

# --- Data -----------------------------------------------------------------
# Note: filter_stats.json contains absolute paths to the original images.
#       Keep it excluded so sensitive source paths are not exposed in a public repo.
data/
!data/.gitkeep

# --- Evaluation caches / outputs -----------------------------------------
eval_cache/
*.npy
*.npz
*.log

# --- OS -------------------------------------------------------------------
.DS_Store
Thumbs.db
desktop.ini
"""

REQUIREMENTS = """\
# Install torch / torchvision separately for the CUDA version in use:
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
torch
torchvision

transformers>=4.45
qdrant-client>=1.11
numpy
pillow
pyyaml
opencv-python
"""


# --------------------------------------------------------------------------- #
def find_files(root: Path, name: str) -> List[Path]:
    """Find all files in the project whose filename matches the given name."""
    hits: List[Path] = []
    for p in root.rglob(name):
        if any(part in SKIP_DIRS for part in p.relative_to(root).parts[:-1]):
            continue
        if p.is_file():
            hits.append(p)
    return hits


def plan_move(root: Path, name: str, dest_dir: str) -> Tuple[str, Optional[Path], Optional[Path], str]:
    """
    Returns: (status, source, destination, message)
    Status: ok_inplace | move | missing | conflict | ambiguous | exists_differs
    """
    dest = (root / dest_dir / name).resolve() if dest_dir != "." else (root / name).resolve()
    hits = find_files(root, name)

    if not hits:
        return "missing", None, dest, "Not found"

    resolved = [h.resolve() for h in hits]

    if dest in resolved:
        others = [h for h in resolved if h != dest]
        if others:
            rel = ", ".join(str(o.relative_to(root)) for o in others)
            return "conflict", dest, dest, f"Already in target location, but extra copies exist: {rel}"
        return "ok_inplace", dest, dest, "Already in the correct location"

    if len(resolved) > 1:
        rel = ", ".join(str(h.relative_to(root)) for h in resolved)
        return "ambiguous", None, dest, f"Found in multiple locations: {rel}"

    src = resolved[0]

    if dest.exists():
        same = filecmp.cmp(src, dest, shallow=False)
        if same:
            return "ok_inplace", src, dest, "An identical file already exists at the destination"
        return "exists_differs", src, dest, "A different file already exists at the destination"

    return "move", src, dest, ""


def fix_qdrant_import(root: Path, apply: bool) -> Optional[str]:
    """Normalize the config import path used by qdrant_store.py.

    registry.py / router.py use `from config import PipelineConfig`, but
    qdrant_store.py used `from src.RFDETR.pipeline.config import ...`.
    This can load the same file as two different modules, and in a fresh
    environment without src/RFDETR/pipeline on the import path, it can raise ImportError.

    During dry-run, files have not moved yet, so search from their original locations.
    """
    candidates = find_files(root, "qdrant_store.py")
    if not candidates:
        return None

    old = "from src.RFDETR.pipeline.config import PipelineConfig"
    new = "from config import PipelineConfig"

    for path in candidates:
        text = path.read_text(encoding="utf-8")
        if old not in text:
            continue
        if apply:
            path.write_text(text.replace(old, new), encoding="utf-8")
        return f"{old}\n                   -> {new}"
    return None


def write_if_absent(path: Path, content: str, apply: bool) -> str:
    if path.exists():
        return "Already exists (left unchanged)"
    if apply:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return "Create"


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="Repository structure setup")
    ap.add_argument("--root", default=".", help="Project root (default: current directory)")
    ap.add_argument("--apply", action="store_true",
                    help="Apply file changes. Without this option, run in dry-run mode.")
    ap.add_argument("--copy", action="store_true",
                    help="Copy files instead of moving them (preserve originals).")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        print(f"Project root not found: {root}")
        return 1

    mode = "APPLY" if args.apply else "PREVIEW (use --apply to make actual changes)"
    verb = "COPY" if args.copy else "MOVE"
    print(f"Root : {root}")
    print(f"Mode : {mode} / {verb}")
    print("=" * 72)

    # ---- 1. Directories ----
    print("\n[1] Directory structure")
    for d in PACKAGE_DIRS + PLAIN_DIRS:
        target = root / d
        exists = target.is_dir()
        print(f"  {'EXISTS ' if exists else 'CREATE '} {d}/")
        if args.apply and not exists:
            target.mkdir(parents=True, exist_ok=True)

    # ---- 2. __init__.py files ----
    print("\n[2] __init__.py")
    for d in PACKAGE_DIRS:
        init = root / d / "__init__.py"
        if init.exists():
            print(f"  EXISTS {d}/__init__.py")
        else:
            print(f"  CREATE {d}/__init__.py")
            if args.apply:
                init.parent.mkdir(parents=True, exist_ok=True)
                init.write_text("", encoding="utf-8")

    # ---- 3. File placement ----
    print("\n[3] File placement")
    missing_required: List[str] = []
    missing_optional: List[str] = []
    problems: List[str] = []

    for name, dest_dir in LAYOUT.items():
        status, src, dest, msg = plan_move(root, name, dest_dir)
        rel_dest = dest.relative_to(root) if dest else Path(name)

        if status == "ok_inplace":
            print(f"  OK     {rel_dest}")

        elif status == "move":
            rel_src = src.relative_to(root)
            print(f"  {verb}   {rel_src}  ->  {rel_dest}")
            if args.apply:
                dest.parent.mkdir(parents=True, exist_ok=True)
                if args.copy:
                    shutil.copy2(src, dest)
                else:
                    shutil.move(str(src), str(dest))

        elif status == "missing":
            bucket = missing_optional if name in OPTIONAL else missing_required
            bucket.append(f"{rel_dest}")
            print(f"  MISSING {rel_dest}")

        else:  # conflict / ambiguous / exists_differs
            print(f"  WARNING {rel_dest} — {msg}")
            problems.append(f"{name}: {msg}")

    # ---- 4. Normalize import paths (run after file placement) ----
    print("\n[4] Normalize import paths")
    fixed = fix_qdrant_import(root, args.apply)
    if fixed:
        print(f"  qdrant_store.py: {fixed}")
    else:
        print("  qdrant_store.py: No changes needed")

    # ---- 5. Supporting files ----
    print("\n[5] Supporting files")
    print(f"  .gitignore        {write_if_absent(root / '.gitignore', GITIGNORE, args.apply)}")
    print(f"  requirements.txt  {write_if_absent(root / 'requirements.txt', REQUIREMENTS, args.apply)}")
    gitkeep = root / "data" / ".gitkeep"
    print(f"  data/.gitkeep     {write_if_absent(gitkeep, '', args.apply)}")

    # ---- Summary ----
    print("\n" + "=" * 72)
    if missing_required:
        print("\n[Missing required files] Setup cannot complete without these:")
        for m in missing_required:
            print(f"  - {m}")
    if missing_optional:
        print("\n[Missing optional files] Setup can continue without these:")
        for m in missing_optional:
            print(f"  - {m}")
    if problems:
        print("\n[Manual review required]")
        for p in problems:
            print(f"  - {p}")

    if not args.apply:
        print("\nTo apply changes:      python setup_repo.py --apply")
        print("To preserve originals: python setup_repo.py --apply --copy")
    else:
        print("\nSetup applied. Verify with the following commands:")
        print("  python config.py pipeline.yaml")
        print("  python -c \"from config import PipelineConfig; "
              "from qdrant_store import QdrantStore; "
              "QdrantStore(PipelineConfig.load('pipeline.yaml')).ensure_collection()\"")
        print("\nBefore pushing to Git, make sure to review .gitignore.")
        print("  data/filter_stats.json contains absolute paths to the original images.")

    return 1 if (missing_required or problems) else 0


if __name__ == "__main__":
    sys.exit(main())