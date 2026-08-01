import argparse
import json
import os
import sys
import time
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark official latest IndexTTS normal vs DeepSpeed inference."
    )
    parser.add_argument(
        "--upstream-repo",
        required=True,
        help="Path to the clean official latest repository checkout.",
    )
    parser.add_argument(
        "--runtime-root",
        required=True,
        help="Working directory containing checkpoints/examples/outputs.",
    )
    parser.add_argument(
        "--text",
        required=True,
        help="Benchmark text to synthesize.",
    )
    parser.add_argument(
        "--speaker-wav",
        default="examples/voice_01.wav",
        help="Speaker prompt path relative to runtime root, or absolute path.",
    )
    parser.add_argument(
        "--checkpoints",
        default="checkpoints",
        help="Checkpoint directory relative to runtime root, or absolute path.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/benchmarks",
        help="Output directory relative to runtime root, or absolute path.",
    )
    parser.add_argument(
        "--results-json",
        default="outputs/benchmarks/official_latest_benchmark.json",
        help="JSON report path relative to runtime root, or absolute path.",
    )
    parser.add_argument(
        "--max-text-tokens-per-segment",
        type=int,
        default=120,
        help="Generation segment token limit.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Enable fp16 for both modes.",
    )
    parser.add_argument(
        "--skip-normal",
        action="store_true",
        default=False,
        help="Skip the non-DeepSpeed benchmark pass.",
    )
    parser.add_argument(
        "--skip-deepspeed",
        action="store_true",
        default=False,
        help="Skip the DeepSpeed benchmark pass.",
    )
    return parser


def resolve_path(root: Path, maybe_relative: str) -> Path:
    candidate = Path(maybe_relative)
    if candidate.is_absolute():
        return candidate
    return root / candidate


def prepare_imports(upstream_repo: Path) -> None:
    repo_str = str(upstream_repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    indextts_path = str(upstream_repo / "indextts")
    if indextts_path not in sys.path:
        sys.path.insert(0, indextts_path)


def run_case(
    *,
    mode_name: str,
    use_deepspeed: bool,
    use_fp16: bool,
    checkpoints_dir: Path,
    speaker_wav: Path,
    output_dir: Path,
    text: str,
    max_text_tokens_per_segment: int,
) -> dict:
    from indextts.infer_v2 import IndexTTS2

    output_path = output_dir / f"official_latest_{mode_name}.wav"
    if output_path.exists():
        output_path.unlink()

    init_start = time.perf_counter()
    tts = IndexTTS2(
        model_dir=str(checkpoints_dir),
        cfg_path=str(checkpoints_dir / "config.yaml"),
        use_fp16=use_fp16,
        use_deepspeed=use_deepspeed,
    )
    init_seconds = time.perf_counter() - init_start

    infer_start = time.perf_counter()
    returned_path = tts.infer(
        spk_audio_prompt=str(speaker_wav),
        text=text,
        output_path=str(output_path),
        verbose=True,
        max_text_tokens_per_segment=max_text_tokens_per_segment,
    )
    infer_seconds = time.perf_counter() - infer_start

    file_exists = output_path.exists()
    file_size = output_path.stat().st_size if file_exists else -1

    return {
        "mode": mode_name,
        "use_deepspeed": use_deepspeed,
        "use_fp16": use_fp16,
        "init_seconds": round(init_seconds, 4),
        "infer_seconds": round(infer_seconds, 4),
        "output_path": str(output_path),
        "returned_path": str(returned_path),
        "output_exists": file_exists,
        "output_size_bytes": file_size,
    }


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    upstream_repo = Path(args.upstream_repo).resolve()
    runtime_root = Path(args.runtime_root).resolve()
    checkpoints_dir = resolve_path(runtime_root, args.checkpoints).resolve()
    speaker_wav = resolve_path(runtime_root, args.speaker_wav).resolve()
    output_dir = resolve_path(runtime_root, args.output_dir).resolve()
    results_json = resolve_path(runtime_root, args.results_json).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    results_json.parent.mkdir(parents=True, exist_ok=True)

    os.chdir(runtime_root)
    prepare_imports(upstream_repo)

    report = {
        "upstream_repo": str(upstream_repo),
        "runtime_root": str(runtime_root),
        "checkpoints_dir": str(checkpoints_dir),
        "speaker_wav": str(speaker_wav),
        "text": args.text,
        "max_text_tokens_per_segment": args.max_text_tokens_per_segment,
        "fp16": args.fp16,
        "results": [],
    }

    if not args.skip_normal:
        report["results"].append(
            run_case(
                mode_name="normal",
                use_deepspeed=False,
                use_fp16=args.fp16,
                checkpoints_dir=checkpoints_dir,
                speaker_wav=speaker_wav,
                output_dir=output_dir,
                text=args.text,
                max_text_tokens_per_segment=args.max_text_tokens_per_segment,
            )
        )

    if not args.skip_deepspeed:
        report["results"].append(
            run_case(
                mode_name="deepspeed",
                use_deepspeed=True,
                use_fp16=args.fp16,
                checkpoints_dir=checkpoints_dir,
                speaker_wav=speaker_wav,
                output_dir=output_dir,
                text=args.text,
                max_text_tokens_per_segment=args.max_text_tokens_per_segment,
            )
        )

    if len(report["results"]) == 2:
        normal = next(item for item in report["results"] if item["mode"] == "normal")
        deepspeed = next(item for item in report["results"] if item["mode"] == "deepspeed")
        if normal["infer_seconds"] > 0:
            report["summary"] = {
                "speedup_vs_normal": round(normal["infer_seconds"] / deepspeed["infer_seconds"], 4),
                "normal_infer_seconds": normal["infer_seconds"],
                "deepspeed_infer_seconds": deepspeed["infer_seconds"],
            }

    results_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved benchmark report to: {results_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
