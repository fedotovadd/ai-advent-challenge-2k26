"""Local Ollama isolation and resource measurements for Day 29."""

import json
import subprocess
import threading
import time
from urllib.request import urlopen

from deepseek_app.providers.providers import LOCAL_MODELS, ask_model


BASE = "http://127.0.0.1:11434"


def _same_model(reported, requested):
    return reported == requested or reported == requested + ":latest"


def _is_local_model(name):
    return any(_same_model(name, allowed) for allowed in LOCAL_MODELS)


def _get(path):
    with urlopen(BASE + path, timeout=10) as response:
        return json.load(response)


def _loaded():
    return _get("/api/ps").get("models", [])


def installed_digest(model):
    tags = _get("/api/tags").get("models", [])
    info = next((row for row in tags if _same_model(row.get("name"), model)), None)
    if info is None:
        raise RuntimeError(f"В Ollama отсутствует модель {model}")
    return info["digest"]


def _rss_bytes():
    output = subprocess.check_output(["ps", "-axo", "rss=,command="], text=True)
    return _ollama_rss_from_ps(output)


def _ollama_rss_from_ps(output):
    total = 0
    for line in output.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            continue
        argv = parts[1].split()
        executable = argv[0]
        command = argv[1] if len(argv) > 1 else None
        if ((executable == "ollama" or executable.endswith("/ollama"))
                and command in ("serve", "runner")) or executable.endswith("/llama-server"):
            total += int(parts[0]) * 1024
    return total


class OllamaRuntime:
    """Unload, warm and sample a single model block at 100 ms intervals."""

    def __init__(self):
        self._thread = None
        self._stop_event = None
        self._peak = None

    def _unload(self):
        for model in _loaded():
            if _is_local_model(model.get("name")):
                subprocess.run(["ollama", "stop", model["name"]], check=True,
                               capture_output=True, text=True)
        remaining = [row["name"] for row in _loaded() if _is_local_model(row.get("name"))]
        if remaining:
            raise RuntimeError(f"Не удалось выгрузить варианты Ollama: {remaining}")

    def _sample(self):
        while not self._stop_event.wait(.1):
            self._peak = max(self._peak, _rss_bytes())

    def begin(self, profile):
        self._unload()
        background = _rss_bytes()
        model = profile["model"]
        tags = _get("/api/tags")["models"]
        info = next((row for row in tags if _same_model(row["name"], model)), None)
        if info is None:
            raise RuntimeError(f"В Ollama отсутствует модель {model}")
        if profile.get("model_digest") and info["digest"] != profile["model_digest"]:
            raise ValueError("Изменился digest локальной модели")
        started = time.perf_counter()
        ask_model({"model": model, "temperature": 0,
                   "messages": [{"role": "user", "content": "Скажи: готово."}]},
                  max_tokens=8, timeout=120)
        cold_seconds = time.perf_counter() - started
        loaded = next((row for row in _loaded() if _same_model(row.get("name"), model)), None)
        if loaded is None:
            raise RuntimeError(f"Нельзя подтвердить размещение модели {model}")
        actual_ctx = loaded.get("context_length")
        if actual_ctx != profile["num_ctx"]:
            raise RuntimeError(f"Контекст {model}: ожидали {profile['num_ctx']}, выделено {actual_ctx}")
        self._peak = _rss_bytes()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return {"model_digest": info["digest"],
                "quantization": info["details"].get("quantization_level"),
                "format": info["details"].get("format"),
                "disk_bytes": info.get("size"), "actual_num_ctx": actual_ctx,
                "cold_seconds": cold_seconds, "background_rss_bytes": background,
                "after_warm_rss_bytes": self._peak,
                "placement": loaded.get("size_vram"),
                "loaded_model": loaded, "sample_interval_seconds": .1}

    def end(self, profile):
        self._stop_event.set()
        self._thread.join(timeout=5)
        peak = max(self._peak, _rss_bytes())
        self._unload()
        background = _rss_bytes()
        self._thread = None
        return {"peak_rss_bytes": peak, "after_unload_rss_bytes": background,
                "unloaded_confirmed": True}

    def abort(self):
        if self._stop_event is not None:
            self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._unload()
