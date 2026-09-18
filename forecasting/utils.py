"""Small standalone helpers that don't belong to any one class."""
import platform

import psutil


def hardware_info() -> dict:
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "total_ram_gb": round(psutil.virtual_memory().total / (1024**3), 1),
    }
