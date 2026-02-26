"""Render mutated .flame files to PNG using JWildfire CLI."""

import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
JWF_JAR = SCRIPT_DIR / "lib" / "lib" / "j-wildfire.jar"

# Find Java
JAVA_CANDIDATES = [
    "/opt/homebrew/opt/openjdk/bin/java",
    "/usr/bin/java",
    "java",
]


def find_java() -> str:
    for candidate in JAVA_CANDIDATES:
        try:
            subprocess.run(
                [candidate, "-version"],
                capture_output=True,
                timeout=5,
            )
            return candidate
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    raise RuntimeError("Java not found. Run: brew install openjdk")


def render_one(flame_path: Path, output_dir: Path, java_cmd: str) -> tuple[str, bool]:
    """Render a single .flame file. Returns (name, success)."""
    name = flame_path.stem
    outfile = output_dir / f"{name}.png"

    if outfile.exists():
        return name, True

    # JWildfire saves PNG next to the .flame file
    rendered_png = flame_path.with_suffix(".png")

    try:
        subprocess.run(
            [
                java_cmd, "-cp", str(JWF_JAR),
                "org.jwildfire.cli.FlameRenderer",
                "-f", str(flame_path),
                "-w", "512", "-h", "512",
                "-q", "200",
            ],
            capture_output=True,
            timeout=60,
        )

        if rendered_png.exists():
            rendered_png.rename(outfile)
            return name, True
        else:
            return name, False
    except (subprocess.TimeoutExpired, Exception) as e:
        return name, False


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Render .flame files to PNG")
    parser.add_argument("flame_dir", help="Directory containing .flame files")
    parser.add_argument("output_dir", help="Output directory for PNGs")
    parser.add_argument("-w", "--workers", type=int, default=8)
    args = parser.parse_args()

    flame_dir = Path(args.flame_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    java_cmd = find_java()
    flames = sorted(flame_dir.glob("*.flame"))

    # Count already rendered
    existing = sum(1 for f in flames if (output_dir / f"{f.stem}.png").exists())

    print(f"Found {len(flames)} flame files, {existing} already rendered")
    print(f"Rendering {len(flames) - existing} remaining with {args.workers} workers...")
    print(f"  Java: {java_cmd}")
    print(f"  JAR:  {JWF_JAR}")

    success = existing
    failed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(render_one, f, output_dir, java_cmd): f
            for f in flames
            if not (output_dir / f"{f.stem}.png").exists()
        }

        for i, future in enumerate(as_completed(futures), 1):
            name, ok = future.result()
            if ok:
                success += 1
            else:
                failed += 1

            if i % 100 == 0 or i == len(futures):
                print(f"  [{i}/{len(futures)}] success={success} failed={failed}")

    print(f"\nDone. {success} images in {output_dir}/ ({failed} failures)")


if __name__ == "__main__":
    main()
