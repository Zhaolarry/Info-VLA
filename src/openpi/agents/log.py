from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, Path]


def create_log(log_dir: PathLike, name: str = "log", suffix: str = ".log", rank: int = 0) -> Path:
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = log_dir_path / f"{name}-{timestamp}{suffix}{rank}"
    log_path.touch(exist_ok=True)
    return log_path


def write_log(log_file: PathLike, message: str, with_time: bool = True, end: str = "\n") -> None:
    log_path = Path(log_file)
    prefix = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] " if with_time else ""
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{prefix}{message}{end}")


class Logger:
    def __init__(
        self,
        log_dir: PathLike,
        name: str = "log",
        suffix: str = ".log",
        rank: int = 0,
        with_time: bool = True,
    ) -> None:
        self.log_path = create_log(log_dir=log_dir, name=name, suffix=suffix, rank=rank)
        self.with_time = with_time

    @classmethod
    def from_file(cls, log_file: PathLike, with_time: bool = True) -> "Logger":
        obj = cls.__new__(cls)
        obj.log_path = Path(log_file)
        obj.with_time = with_time
        return obj

    def write(self, message: str, *, with_time: Optional[bool] = None, end: str = "\n") -> None:
        write_log(self.log_path, message, with_time=self.with_time if with_time is None else with_time, end=end)

    def path(self) -> Path:
        return self.log_path


__all__ = ["create_log", "write_log", "Logger"]
