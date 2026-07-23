from __future__ import annotations

"""WiSED 简化运行入口。

直接在本文件顶部修改实验项目、噪声水平和设备，然后在 VSCode 中点击运行，
或在终端执行：

    python run.py

也可以临时用命令行覆盖顶部配置：

    python run.py --equations burgers_2d,fhn_2d --noise 0,0.01 --device cpu
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

# ============================================================
# 运行配置区：日常实验只改这里
# ============================================================
EXPERIMENT_PROJECTS: List[str] = [
    #"burgers_1d",
    #"kdv_1d",
    #"burgers_2d",
    "fhn_2d",
]
NOISE_LEVELS: List[float] = [ 0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.10]
SEEDS: List[int] = [42]
DEVICE: str = "cuda"
TARGETS: Optional[List[str]] = None          # 例如 ["u"] 或 ["u", "v"]；None 表示使用方程默认目标
SKIP_EXISTING: bool = False
CONTINUE_ON_ERROR: bool = True

# 单次数据集生成入口。需要生成数据时，把 RUN_MODE 改成 "data"。
RUN_MODE: str = "experiment"                 # "experiment" 或 "data"
DATA_BATCH_SIZE: Optional[int] = None
DATA_SAVE_DIR: str = "data/dataset"

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _parse_csv_floats(text: Optional[str], default: Sequence[float]) -> List[float]:
    if text is None or str(text).strip() == "":
        return [float(x) for x in default]
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def _parse_csv_ints(text: Optional[str], default: Sequence[int]) -> List[int]:
    if text is None or str(text).strip() == "":
        return [int(x) for x in default]
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def _parse_csv_strings(text: Optional[str], default: Optional[Sequence[str]]) -> Optional[List[str]]:
    if text is None:
        return None if default is None else [str(x) for x in default]
    items = [x.strip() for x in str(text).split(",") if x.strip()]
    return items or None


def _parse_overrides(items: Optional[Sequence[str]]) -> Dict[str, object]:
    from src.wised.experiment_config import parse_overrides
    return parse_overrides(items or [])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="WiSED simplified runner")
    parser.add_argument("--mode", choices=["experiment", "data"], default=RUN_MODE)
    parser.add_argument("--equations", default=",".join(EXPERIMENT_PROJECTS))
    parser.add_argument("--noise", default=",".join(str(x) for x in NOISE_LEVELS))
    parser.add_argument("--seeds", default=",".join(str(x) for x in SEEDS))
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--targets", default=None if TARGETS is None else ",".join(TARGETS))
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--skip-existing", action="store_true", default=SKIP_EXISTING)
    parser.add_argument("--stop-on-error", action="store_true", help="遇到单个实验失败时立即停止。")
    parser.add_argument("--data-save-dir", default=DATA_SAVE_DIR)
    parser.add_argument("--data-batch-size", type=int, default=DATA_BATCH_SIZE)
    return parser


def run_experiments(
    equations: Iterable[str],
    noises: Iterable[float],
    seeds: Iterable[int],
    *,
    device: str,
    targets: Optional[List[str]],
    overrides: Optional[Dict[str, object]],
    skip_existing: bool,
    continue_on_error: bool,
) -> int:
    from src.wised.experiment import run_discovery_experiment
    from src.wised.experiment_config import case_dir_name

    exit_code = 0
    for equation in equations:
        for noise in noises:
            for seed in seeds:
                case = case_dir_name(noise_level=float(noise), seed=int(seed))
                summary = PROJECT_ROOT / "results" / equation / case / f"{equation}_{case}_summary.json"
                if skip_existing and summary.exists():
                    print(f"[SKIP] {equation} noise={noise:g} seed={seed} -> {summary}")
                    continue
                print(f"\n[RUN] equation={equation} | noise={noise:g} | seed={seed} | device={device}")
                try:
                    run_discovery_experiment(
                        equation,
                        device=device,
                        seed=int(seed),
                        noise_level=float(noise),
                        targets=targets,
                        config_overrides=overrides,
                    )
                except Exception as exc:
                    exit_code = 1
                    print(f"[ERROR] {equation} noise={noise:g} seed={seed}: {exc}")
                    if not continue_on_error:
                        raise
    return exit_code


def run_data_generation(
    equations: Iterable[str],
    noises: Iterable[float],
    seeds: Iterable[int],
    *,
    save_dir: str,
    batch_size: Optional[int],
) -> int:
    from src.wised.data_io import load_or_generate_dataset
    from src.wised.experiment_config import build_run_config

    class _Logger:
        def info(self, msg): print(msg)
        def warning(self, msg): print(f"[WARN] {msg}")
        def log_kv(self, title, data): print(title, data)

    logger = _Logger()
    for equation in equations:
        for noise in noises:
            for seed in seeds:
                cfg = build_run_config(equation, noise_level=float(noise), seed=int(seed), device=DEVICE)
                cfg["dataset_dir"] = save_dir
                if batch_size is not None:
                    cfg["batch_size"] = int(batch_size)
                print(f"\n[DATA] equation={equation} | noise={noise:g} | seed={seed}")
                load_or_generate_dataset(equation, cfg, logger, dataset_dir=save_dir)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    equations = _parse_csv_strings(args.equations, EXPERIMENT_PROJECTS) or []
    noises = _parse_csv_floats(args.noise, NOISE_LEVELS)
    seeds = _parse_csv_ints(args.seeds, SEEDS)
    targets = _parse_csv_strings(args.targets, TARGETS)
    overrides = _parse_overrides(args.set)

    if args.mode == "data":
        return run_data_generation(
            equations,
            noises,
            seeds,
            save_dir=args.data_save_dir,
            batch_size=args.data_batch_size,
        )

    return run_experiments(
        equations,
        noises,
        seeds,
        device=str(args.device),
        targets=targets,
        overrides=overrides,
        skip_existing=bool(args.skip_existing),
        continue_on_error=not bool(args.stop_on_error) and bool(CONTINUE_ON_ERROR),
    )


if __name__ == "__main__":
    raise SystemExit(main())
