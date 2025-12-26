"""Lightweight signal logging for debugging external interrupts."""
from __future__ import annotations

import os
import signal
import time
from pathlib import Path

_ENABLED = False
_LOG_PATH: Path | None = None
_HANDLER = None


def _truthy_env(name: str) -> bool:
    value = os.getenv(name, "").strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


def _read_cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            raw = fh.read().strip(b"\x00")
        if not raw:
            return ""
        return raw.replace(b"\x00", b" ").decode("utf-8", "replace")
    except Exception:
        return ""


def _read_comm(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/comm", "r", encoding="utf-8") as fh:
            return fh.read().strip()
    except Exception:
        return ""


def _read_ppid(pid: int) -> int | None:
    try:
        with open(f"/proc/{pid}/status", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("PPid:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1])
    except Exception:
        return None
    return None


def _proc_chain(pid: int, max_depth: int = 12) -> list[tuple[int, str]]:
    chain: list[tuple[int, str]] = []
    cur = pid
    for _ in range(max_depth):
        cmd = _read_cmdline(cur)
        if not cmd:
            cmd = _read_comm(cur)
        chain.append((cur, cmd))
        ppid = _read_ppid(cur)
        if ppid is None or ppid <= 1 or ppid == cur:
            break
        cur = ppid
    return chain


def _write_log(signum: int) -> None:
    log_path = _LOG_PATH
    if log_path is None:
        return
    ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    pid = os.getpid()
    ppid = os.getppid()
    sig_name = signal.Signals(signum).name if signum in signal.Signals else str(signum)
    lines = [
        f"{ts} signal={sig_name} pid={pid} ppid={ppid}",
        f"cwd={os.getcwd()}",
        f"argv={' '.join(os.sys.argv)}",
    ]
    chain = _proc_chain(pid)
    if chain:
        chain_txt = " -> ".join(f"{p}:{cmd or '?'}" for p, cmd in chain)
        lines.append(f"process_chain={chain_txt}")
    try:
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
    except Exception:
        pass


def logging_requested() -> bool:
    return _truthy_env("LOG_SIGNALS") or bool(os.getenv("SIGNAL_LOG_PATH"))


def get_signal_handler():
    if not _ENABLED:
        enable_signal_logging()
    return _HANDLER


def enable_signal_logging() -> None:
    """Install SIGINT/SIGTERM handlers that append diagnostics to a log file.

    Enable via LOG_SIGNALS=1 or SIGNAL_LOG_PATH=/path/to/log.
    """
    global _ENABLED
    if _ENABLED:
        return

    if not logging_requested():
        return

    log_path = Path(os.getenv("SIGNAL_LOG_PATH", "outputs/signal.log"))
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        # If we cannot create directories, fall back to current directory.
        log_path = Path("signal.log")

    global _LOG_PATH, _HANDLER
    _LOG_PATH = log_path

    def _handler(signum, _frame):
        _write_log(signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handler)
        except Exception:
            pass
    _HANDLER = _handler
    _ENABLED = True
