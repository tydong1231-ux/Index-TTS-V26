import argparse
import contextlib
import importlib.metadata
import importlib.util
import io
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether the local machine matches the packaged Windows release baseline."
    )
    parser.add_argument(
        "--report-json",
        default="outputs/release_support_report.json",
        help="Where to save the compatibility report.",
    )
    parser.add_argument(
        "--validated-driver",
        default="591.86",
        help="Driver version validated on the release build machine.",
    )
    parser.add_argument(
        "--min-driver",
        default="",
        help="Optional hard minimum driver version. Leave empty to only warn below validated baseline.",
    )
    parser.add_argument(
        "--min-compute-capability",
        default="6.1",
        help="Minimum supported CUDA compute capability for the packaged build.",
    )
    parser.add_argument(
        "--require-deepspeed",
        action="store_true",
        default=False,
        help="Fail the check if DeepSpeed inference support is not available.",
    )
    parser.add_argument(
        "--cache-root",
        default="hf_cache",
        help="Project-local Hugging Face cache root to validate.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        default=False,
        help="Print a concise human-readable result instead of the full JSON payload.",
    )
    return parser.parse_args()


def parse_version(version: str) -> tuple[int, ...]:
    parts: list[int] = []
    for item in version.replace("-", ".").split("."):
        if item.isdigit():
            parts.append(int(item))
        else:
            digits = "".join(ch for ch in item if ch.isdigit())
            if digits:
                parts.append(int(digits))
    return tuple(parts)


def version_lt(left: str, right: str) -> bool:
    return parse_version(left) < parse_version(right)


def resolve_report_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def path_for_report(path: str | Path, base: Path) -> str:
    resolved_path = Path(path).resolve()
    resolved_base = base.resolve()
    try:
        return str(resolved_path.relative_to(resolved_base))
    except ValueError:
        return str(resolved_path)


def run_text(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    return completed.stdout.strip()


def get_nvidia_smi_info() -> dict:
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,compute_cap",
        "--format=csv,noheader",
    ]
    try:
        output = run_text(command)
    except Exception as exc:
        return {"available": False, "error": str(exc)}

    rows = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 3:
            rows.append(
                {
                    "name": parts[0],
                    "driver_version": parts[1],
                    "compute_capability": parts[2],
                }
            )
    return {"available": True, "gpus": rows}


def get_torch_info() -> dict:
    try:
        with silence_stdio():
            import torch

            version = torch.__version__
            torch_cuda = torch.version.cuda
            cuda_available = torch.cuda.is_available()
            device_count = torch.cuda.device_count() if cuda_available else 0
            gpus = []
            if cuda_available:
                for idx in range(device_count):
                    capability = torch.cuda.get_device_capability(idx)
                    gpus.append(
                        {
                            "index": idx,
                            "name": torch.cuda.get_device_name(idx),
                            "compute_capability": f"{capability[0]}.{capability[1]}",
                        }
                    )
    except Exception as exc:
        return {"available": False, "error": str(exc)}

    info = {
        "available": True,
        "version": version,
        "torch_cuda": torch_cuda,
        "cuda_available": cuda_available,
        "device_count": device_count,
        "gpus": gpus,
    }
    return info


@contextlib.contextmanager
def silence_stdio():
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)
    saved_stdout = sys.stdout
    saved_stderr = sys.stderr
    try:
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        sys.stdout = io.TextIOWrapper(os.fdopen(os.dup(1), "wb"), encoding="utf-8", write_through=True)
        sys.stderr = io.TextIOWrapper(os.fdopen(os.dup(2), "wb"), encoding="utf-8", write_through=True)
        yield
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        sys.stdout = saved_stdout
        sys.stderr = saved_stderr
        os.dup2(saved_stdout_fd, 1)
        os.dup2(saved_stderr_fd, 2)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)
        os.close(devnull_fd)


def get_deepspeed_info() -> dict:
    try:
        version = importlib.metadata.version("deepspeed")
    except Exception as exc:
        return {"available": False, "error": str(exc)}

    module_name = "deepspeed.ops.transformer.inference.transformer_inference_op"
    transformer_inference_installed = importlib.util.find_spec(module_name) is not None
    transformer_inference_compatible = False
    compatibility_error = None

    try:
        with silence_stdio():
            from deepspeed.ops.op_builder.transformer_inference import InferenceBuilder

            transformer_inference_compatible = bool(InferenceBuilder().is_compatible(verbose=False))
    except Exception as exc:
        compatibility_error = str(exc)

    return {
        "available": True,
        "version": version,
        "transformer_inference_module": module_name,
        "transformer_inference_installed": transformer_inference_installed,
        "transformer_inference_compatible": transformer_inference_compatible,
        "compatibility_error": compatibility_error,
    }


def get_cache_info(cache_root: Path, report_base: Path) -> dict:
    required_relative_paths = [
        Path("models--facebook--w2v-bert-2.0") / "snapshots",
        Path("models--amphion--MaskGCT") / "snapshots",
        Path("models--funasr--campplus") / "snapshots",
        Path("models--nvidia--bigvgan_v2_22khz_80band_256x") / "snapshots",
    ]
    checks = []
    for relative in required_relative_paths:
        path = cache_root / relative
        has_any = path.exists() and any(path.iterdir())
        checks.append(
            {
                "path": path_for_report(path, report_base),
                "exists": path.exists(),
                "non_empty": has_any,
            }
        )
    return {
        "root": path_for_report(cache_root, report_base),
        "checks": checks,
    }


def evaluate(
    *,
    smi_info: dict,
    torch_info: dict,
    deepspeed_info: dict,
    cache_info: dict,
    validated_driver: str,
    min_driver: str,
    min_compute_capability: str,
    require_deepspeed: bool,
) -> tuple[str, list[str], list[str]]:
    status = "pass"
    reasons: list[str] = []
    recommendations: list[str] = []

    if not torch_info.get("available"):
        return "fail", ["PyTorch import failed."], ["Repair the packaged runtime before release."]

    if not torch_info.get("cuda_available"):
        return "fail", ["CUDA is not available in PyTorch."], ["Use an NVIDIA GPU and the packaged CUDA-enabled runtime."]

    min_cc = parse_version(min_compute_capability)
    for gpu in torch_info.get("gpus", []):
        gpu_cc = gpu.get("compute_capability", "")
        if version_lt(gpu_cc, min_compute_capability):
            status = "fail"
            reasons.append(
                f"GPU {gpu['name']} compute capability {gpu_cc} is below the packaged minimum {min_compute_capability}."
            )

    if smi_info.get("available") and smi_info.get("gpus"):
        driver_version = smi_info["gpus"][0].get("driver_version", "")
        if min_driver and version_lt(driver_version, min_driver):
            status = "fail"
            reasons.append(f"NVIDIA driver {driver_version} is below the hard minimum {min_driver}.")
        elif validated_driver and version_lt(driver_version, validated_driver):
            if status != "fail":
                status = "warn"
            reasons.append(
                f"NVIDIA driver {driver_version} is below the validated release baseline {validated_driver}."
            )
            recommendations.append("Upgrade the GPU driver to the validated baseline or newer before public release.")
    else:
        if status != "fail":
            status = "warn"
        reasons.append("nvidia-smi is unavailable, so the driver baseline could not be verified.")
        recommendations.append("Bundle a startup precheck and document the validated driver baseline in the release notes.")

    if require_deepspeed:
        if not deepspeed_info.get("available"):
            status = "fail"
            reasons.append("DeepSpeed is not importable in the packaged runtime.")
        else:
            if not deepspeed_info.get("transformer_inference_installed"):
                status = "fail"
                reasons.append("DeepSpeed transformer_inference op is not installed.")
            if not deepspeed_info.get("transformer_inference_compatible"):
                status = "fail"
                reasons.append("DeepSpeed transformer_inference op is not compatible on this machine.")

    missing_cache = [item["path"] for item in cache_info.get("checks", []) if not item["non_empty"]]
    if missing_cache:
        status = "fail"
        reasons.append("Required project-local model cache is incomplete.")
        recommendations.append("Bundle the Hugging Face cache snapshots inside the package and point runtime env vars to that cache.")

    if not reasons:
        reasons.append("This machine matches the packaged release baseline checks.")
    if not recommendations:
        recommendations.append("Use this report as the release precheck result for Windows GPU builds.")
    return status, reasons, recommendations


def main() -> int:
    args = parse_args()
    report_path = resolve_report_path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    cache_root = resolve_report_path(args.cache_root)
    report_base = Path.cwd().parent if Path.cwd().name.lower() == "data" else Path.cwd()

    smi_info = get_nvidia_smi_info()
    torch_info = get_torch_info()
    deepspeed_info = get_deepspeed_info()
    cache_info = get_cache_info(cache_root, report_base)
    status, reasons, recommendations = evaluate(
        smi_info=smi_info,
        torch_info=torch_info,
        deepspeed_info=deepspeed_info,
        cache_info=cache_info,
        validated_driver=args.validated_driver.strip(),
        min_driver=args.min_driver.strip(),
        min_compute_capability=args.min_compute_capability.strip(),
        require_deepspeed=args.require_deepspeed,
    )

    report = {
        "status": status,
        "validated_driver": args.validated_driver,
        "min_driver": args.min_driver,
        "min_compute_capability": args.min_compute_capability,
        "require_deepspeed": args.require_deepspeed,
        "python": sys.version.split()[0],
        "python_executable": path_for_report(sys.executable, report_base),
        "platform": platform.platform(),
        "cwd": path_for_report(Path.cwd(), report_base),
        "nvidia_smi": smi_info,
        "torch": torch_info,
        "deepspeed": deepspeed_info,
        "cache": cache_info,
        "reasons": reasons,
        "recommendations": recommendations,
    }

    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.summary_only:
        reason_map = {
            "This machine matches the packaged release baseline checks.": "当前机器满足本发布版的基线要求。",
            "PyTorch import failed.": "PyTorch 导入失败。",
            "CUDA is not available in PyTorch.": "当前 PyTorch 没有可用的 CUDA。",
            "DeepSpeed is not importable in the packaged runtime.": "当前发布版运行时里无法导入 DeepSpeed。",
            "DeepSpeed transformer_inference op is not installed.": "DeepSpeed transformer_inference 组件未安装。",
            "DeepSpeed transformer_inference op is not compatible on this machine.": "当前机器不兼容 DeepSpeed transformer_inference。",
            "Required project-local model cache is incomplete.": "项目内模型缓存不完整。",
        }
        recommendation_map = {
            "Use this report as the release precheck result for Windows GPU builds.": "当前环境可作为 Windows GPU 发布版的预检通过结果。",
            "Upgrade the GPU driver to the validated baseline or newer before public release.": "建议先升级显卡驱动到验证基线版本或更高版本。",
            "Bundle a startup precheck and document the validated driver baseline in the release notes.": "建议保留启动前环境检测，并在发布说明里写明驱动基线。",
            "Bundle the Hugging Face cache snapshots inside the package and point runtime env vars to that cache.": "建议把 Hugging Face 缓存快照完整打包，并让运行时固定指向项目内缓存。",
            "Repair the packaged runtime before release.": "发布前需要先修复当前运行时。",
            "Use an NVIDIA GPU and the packaged CUDA-enabled runtime.": "请使用 NVIDIA 显卡，并使用当前打包的 CUDA 运行时。",
        }
        summary_reasons = [reason_map.get(item, item) for item in reasons]
        summary_recommendations = [recommendation_map.get(item, item) for item in recommendations]

        print("===============================================")
        print("IndexTTS V26 环境检测结果")
        print("===============================================")

        if status == "pass":
            if args.require_deepspeed and deepspeed_info.get("transformer_inference_compatible"):
                print("结论: 当前环境可以直接运行发布版，并可使用 DeepSpeed 模式。")
                print("建议启动: 启动新版.bat")
            else:
                print("结论: 当前环境可以运行发布版。")
                print("建议启动: 小显卡启动版.bat")
        elif status == "warn":
            print("结论: 当前环境基本可用，但有风险项，建议先看下面提示。")
            print("建议启动: 小显卡启动版.bat")
        else:
            print("结论: 当前环境不满足当前发布版要求，暂时不要直接启动。")

        print("")
        print("检测说明:")
        for reason in summary_reasons:
            print(f"- {reason}")

        if summary_recommendations:
            print("")
            print("建议:")
            for item in summary_recommendations:
                print(f"- {item}")

        print("")
        print(f"详细报告: {report_path}")
        print("===============================================")
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"Saved release support report to: {report_path}")
    return 0 if status in {"pass", "warn"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
