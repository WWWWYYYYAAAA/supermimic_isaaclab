"""Create fresh training directories without overwriting an earlier experiment."""

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def create_run_directory(log_dir, timestamp=True):
    """Return a created run path, adding a unique Beijing-time suffix by default."""
    base = Path(log_dir).expanduser().resolve()
    if not timestamp:
        base.mkdir(parents=True, exist_ok=True)
        if any(base.iterdir()):
            raise ValueError(
                f"Training output is not empty: {base}; use --timestamp or a new --log_dir"
            )
        return base

    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f")
    base = base.with_name(f"{base.name}_{stamp}")
    candidate = base
    index = 0
    while True:
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            index += 1
            candidate = base.with_name(f"{base.name}_{index:02d}")
