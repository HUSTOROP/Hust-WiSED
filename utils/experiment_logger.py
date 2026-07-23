from __future__ import annotations

import csv
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from typing import Any, Dict, Iterable, Optional


class _FilePathRedactionFilter(logging.Filter):
    """Remove filesystem paths from records written to persistent log files.

    Console output remains unchanged; only the file handler uses this filter.
    """

    _WINDOWS_PATH = re.compile(r"(?<![A-Za-z0-9_])(?:[A-Za-z]:[\\/])[^\s,;\]\)}\"']+")
    _POSIX_PATH = re.compile(r"(?<![A-Za-z0-9_])/(?:[^\s,;\]\)}\"']+/)*[^\s,;\]\)}\"']+")

    def _redact(self, text: str) -> str:
        text = self._WINDOWS_PATH.sub("[path omitted]", text)
        text = self._POSIX_PATH.sub("[path omitted]", text)
        return text

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = self._redact(str(record.getMessage()))
        record.args = ()
        return True


class WiSEDLogger:
    """Unified logger for WiSED experiments.

    Text logs are human-readable, CSV logs contain one row per epoch, and the
    equation log records improved symbolic candidates.
    """

    DEFAULT_CSV_FIELDS = [
        "epoch",
        "kl_loss",
        "reinforce_loss",
        "struct_loss",
        "total_loss",
        "best_fitness",
        "best_mse",
        "batch_min_fitness",
        "batch_mean_fitness",
        "batch_std_fitness",
        "batch_min_mse",
        "batch_mean_mse",
        "batch_std_mse",
        "n_valid_eqs",
        "n_seq_valid",
        "n_sampled",
        "diversity_score",
        "population_size",
        "subtree_cache_queries",
        "subtree_cache_hits",
        "subtree_cache_misses",
        "subtree_cache_hit_rate",
        "subtree_cache_saved_evals",
        "subtree_cache_queries_epoch",
        "subtree_cache_hits_epoch",
        "subtree_cache_misses_epoch",
        "subtree_cache_hit_rate_epoch",
        "subtree_cache_saved_evals_epoch",
        "template_cache_queries",
        "template_cache_hits",
        "template_cache_misses",
        "template_cache_hit_rate",
        "template_cache_saved_evals",
        "template_cache_queries_epoch",
        "template_cache_hits_epoch",
        "template_cache_misses_epoch",
        "template_cache_hit_rate_epoch",
        "template_cache_saved_evals_epoch",
        "shared_subtrees_warmed",
        "dr_mask_pruned",
        "coarse_candidates",
        "refine_candidates",
        "gamma",
        "lr",
        "epoch_time_sec",
        "elapsed_sec",
    ]

    def __init__(self, log_dir: str, experiment_name: str):
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"{experiment_name}_{timestamp}.log")
        self.loss_csv = os.path.join(log_dir, f"{experiment_name}_losses.csv")
        self.eq_log = os.path.join(log_dir, f"{experiment_name}_equations.txt")
        self._csv_fields = list(self.DEFAULT_CSV_FIELDS)
        self.start_time = time.time()
        self.training_start_time: Optional[float] = None
        self.last_training_elapsed_sec: Optional[float] = None
        self._seen_once_keys = set()

        self.logger = logging.getLogger(f"WiSED::{experiment_name}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()
        self.logger.propagate = False

        fmt = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", datefmt="%H:%M:%S")

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(fmt)
        self.logger.addHandler(console_handler)

        file_handler = logging.FileHandler(self.log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(fmt)
        file_handler.addFilter(_FilePathRedactionFilter())
        self.logger.addHandler(file_handler)

        self._init_csv()
        self._init_equation_log()

        print(f"Logger initialised -> {self.log_file}")
        print(f"Loss CSV          -> {self.loss_csv}")
        print(f"Equation log      -> {self.eq_log}")
        self.info("Logger initialised.")
        self.info("Loss CSV initialised.")
        self.info("Equation log initialised.")

    def _init_csv(self) -> None:
        with open(self.loss_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._csv_fields)
            writer.writeheader()

    def _init_equation_log(self) -> None:
        with open(self.eq_log, "w", encoding="utf-8") as f:
            f.write(f"WiSED Equation Discovery Log - {self.experiment_name}\n")
            f.write("=" * 80 + "\n\n")

    def _coerce_text(self, msg: Any) -> str:
        if isinstance(msg, str):
            return msg
        try:
            return json.dumps(msg, ensure_ascii=False)
        except Exception:
            return str(msg)

    def info(self, msg: Any) -> None:
        self.logger.info(self._coerce_text(msg))

    def debug(self, msg: Any) -> None:
        self.logger.debug(self._coerce_text(msg))

    def warning(self, msg: Any) -> None:
        self.logger.warning(self._coerce_text(msg))

    def error(self, msg: Any) -> None:
        self.logger.error(self._coerce_text(msg))

    def info_once(self, key: str, msg: Any) -> None:
        key = str(key)
        if key in self._seen_once_keys:
            return
        self._seen_once_keys.add(key)
        self.info(msg)

    def section(self, title: str) -> None:
        line = "=" * 68
        self.info(line)
        self.info(title)
        self.info(line)

    def log_kv(self, title: str, payload: Dict[str, Any], *, sort_keys: bool = False) -> None:
        self.section(title)
        keys: Iterable[str] = sorted(payload) if sort_keys else payload.keys()
        for key in keys:
            self.info(f"  {str(key):24s}: {payload[key]}")

    def _without_persisted_paths(self, obj: Any, key_hint: str = "") -> Any:
        """Return a copy suitable for saved logs by replacing path-like metadata."""
        hint = str(key_hint).lower()
        path_key = any(token in hint for token in ("path", "dir", "root"))
        if isinstance(obj, dict):
            return {str(k): self._without_persisted_paths(v, str(k)) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            if path_key:
                return "[not saved]"
            return [self._without_persisted_paths(v, key_hint) for v in obj]
        if isinstance(obj, str):
            stripped = obj.strip()
            looks_absolute = bool(_FilePathRedactionFilter._WINDOWS_PATH.search(stripped) or _FilePathRedactionFilter._POSIX_PATH.search(stripped))
            looks_project_relative = stripped.startswith((
                "data/", "data\\", "logs/", "logs\\", "results/", "results\\",
                "engineering_validation/", "engineering_validation\\", "src/", "src\\",
            ))
            if path_key or looks_absolute or looks_project_relative:
                return "[not saved]"
        return obj

    def log_config(self, cfg: Dict[str, Any], *, title: str = "RUN CONFIG") -> None:
        self.section(title)
        safe_cfg = self._without_persisted_paths(cfg)
        self.info(json.dumps(safe_cfg, ensure_ascii=False, indent=2, sort_keys=True))

    def mark_training_start(self) -> None:
        self.training_start_time = time.time()
        self.last_training_elapsed_sec = None

    def mark_training_end(self, elapsed_sec: Optional[float] = None) -> float:
        if elapsed_sec is None:
            start = self.training_start_time if self.training_start_time is not None else self.start_time
            elapsed_sec = time.time() - start
        self.last_training_elapsed_sec = float(elapsed_sec)
        return self.last_training_elapsed_sec

    def get_last_training_elapsed(self, default: Optional[float] = None) -> Optional[float]:
        if self.last_training_elapsed_sec is not None:
            return float(self.last_training_elapsed_sec)
        if default is None:
            return None
        return float(default)

    def log_epoch(self, epoch: int, metrics: Dict[str, Any]) -> None:
        """Write exactly one CSV row for the epoch."""
        row = {field: metrics.get(field, "") for field in self._csv_fields}
        row["epoch"] = int(epoch)
        start = self.training_start_time if self.training_start_time is not None else self.start_time
        row["elapsed_sec"] = round(time.time() - start, 2)
        with open(self.loss_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._csv_fields)
            writer.writerow(row)

    def log_equation(
        self,
        epoch: int,
        eq_str: str,
        consts: Any,
        mse: float,
        fitness: float,
        tag: str = "",
    ) -> None:
        with open(self.eq_log, "a", encoding="utf-8") as f:
            label = f" [{tag}]" if tag else ""
            f.write(f"[Ep {int(epoch):4d}]{label}\n")
            f.write(f"  Equation : {eq_str}\n")
            f.write(f"  Consts   : {consts}\n")
            f.write(f"  MSE      : {float(mse):.8e}\n")
            f.write(f"  Fitness  : {float(fitness):.8e}\n\n")

    def final_summary(
        self,
        results: Dict[str, Any],
        *,
        title: str = "FINAL SUMMARY",
        elapsed_sec: Optional[float] = None,
        elapsed_label: str = "Total elapsed",
    ) -> None:
        self.log_kv(title, results)
        elapsed = float(time.time() - self.start_time) if elapsed_sec is None else float(elapsed_sec)
        self.info(f"  {elapsed_label:24s}: {elapsed:.1f}s")


