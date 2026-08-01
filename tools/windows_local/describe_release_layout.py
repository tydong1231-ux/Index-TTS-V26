import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Describe a packaged Windows release layout.")
    parser.add_argument("--release-root", required=True, help="Path to the frozen release root.")
    parser.add_argument(
        "--report-json",
        default="release_layout_report.json",
        help="Output JSON filename, relative to release root unless absolute.",
    )
    return parser.parse_args()


def resolve_output(release_root: Path, report_json: str) -> Path:
    path = Path(report_json)
    if path.is_absolute():
        return path
    return release_root / path


def collect_entry(path: Path) -> dict:
    if not path.exists():
        return {"path": str(path), "exists": False}
    if path.is_file():
        return {
            "path": str(path),
            "exists": True,
            "type": "file",
            "size_bytes": path.stat().st_size,
        }
    total_size = 0
    file_count = 0
    for file_path in path.rglob("*"):
        if file_path.is_file():
            file_count += 1
            total_size += file_path.stat().st_size
    return {
        "path": str(path),
        "exists": True,
        "type": "directory",
        "file_count": file_count,
        "size_bytes": total_size,
    }


def main() -> int:
    args = parse_args()
    release_root = Path(args.release_root).resolve()
    report_path = resolve_output(release_root, args.report_json).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)

    entries = {
        "app": collect_entry(release_root / "app"),
        "runtime": collect_entry(release_root / "runtime"),
        "wheelhouse": collect_entry(release_root / "wheelhouse"),
        "checkpoints": collect_entry(release_root / "data" / "checkpoints"),
        "hf_cache": collect_entry(release_root / "data" / "hf_cache"),
        "outputs": collect_entry(release_root / "data" / "outputs"),
        "start_webui_auto.bat": collect_entry(release_root / "start_webui_auto.bat"),
        "check_support.bat": collect_entry(release_root / "check_support.bat"),
    }

    report = {
        "release_root": str(release_root),
        "entries": entries,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved release layout report to: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
