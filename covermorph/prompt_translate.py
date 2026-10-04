"""Korean/Japanese prompt -> English with the Qwen3-4B GGUF already used as the V2 text encoder.

Why: Z-Image-Turbo paints CJK prompt words into the picture as lettering (2/3 seeds with the raw Korean
prompt, 0/3 with English, 3/3 when told "do not write these words"; measured 2026-10-04), and the small
SDXL vocabulary drops words it does not know. llama.cpp's CPU ``llama-completion.exe`` translates a
prompt in ~6 s without touching the GPU and exits afterwards. If it is missing or fails, callers fall
back to the vocabulary translation and warn.
"""
from __future__ import annotations

import re
import subprocess
import tempfile
import time
from functools import lru_cache
from pathlib import Path

from .quality_engines import QWEN3_4B_FILE, ascii_file, ascii_work_root

LLAMACPP_RELEASE = "b11381"
_CJK = re.compile(r"[ᄀ-ᇿ぀-ヿ㄰-㆏㐀-鿿가-힯ｦ-ﾟ]")
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
SYSTEM = ("You translate image-generation prompts into English. Translate literally and completely: keep every "
          "person, nationality, age, pose, view (for example side profile), clothing, place, time, weather and mood "
          "word. "
          "Write ages in words, for example 'in their fifties' (never '50s', which reads as a decade). "
          "Do not add, summarize or explain anything. English text stays as it is. Output only the English prompt.")


# Anchors for the 4B model: without them it turned "눈 오는 날" (snowy day) into "rainy day" and wrote "60s woman".
FEW_SHOT = (
    ("눈 오는 밤 버스 정류장에 서 있는 40대 남성", "A man in his forties standing at a bus stop on a snowy night"),
    ("雨の日のカフェの窓辺に座る60代の女性", "A woman in her sixties sitting by a café window on a rainy day"),
    ("공원 벤치에 앉은 30대 일본인 여성", "A Japanese woman in her thirties sitting on a park bench"),
    # edit instructions stay instructions (without this one, a mixed-language edit was replaced by example 2)
    ("keep the people, 배경을 해 질 녘 바닷가로 바꿔줘",
     "keep the people, change the background to a beach at sunset"),
)


# Safety net for words that change who is in the picture: the 4B model once dropped "일본인" (Japanese).
# If the source has one of these and the translation lacks the English word, it is appended.
KEEP_TERMS = {
    "일본인": "Japanese", "日本人": "Japanese", "한국인": "Korean", "韓国人": "Korean", "중국인": "Chinese",
    "中国人": "Chinese", "서양인": "Western", "西洋人": "Western", "백인": "white", "흑인": "Black",
    "10대": "teenage", "20대": "in their twenties", "30대": "in their thirties", "40대": "in their forties",
    "50대": "in their fifties", "60대": "in their sixties", "70대": "in their seventies",
    "20代": "in their twenties", "30代": "in their thirties", "40代": "in their forties", "50代": "in their fifties",
    "60代": "in their sixties", "70代": "in their seventies",
}
_AGE_WORDS = {"twenties": "20", "thirties": "30", "forties": "40", "fifties": "50", "sixties": "60", "seventies": "70"}


def restore_kept_terms(source: str, english: str) -> tuple[str, list[str]]:
    added = []
    for term, word in KEEP_TERMS.items():
        if term not in source:
            continue
        key = word.split()[-1].casefold()
        variants = {word.casefold(), key, *(f"{n}s" for k, n in _AGE_WORDS.items() if k == key),
                    *(f"{n}-" for k, n in _AGE_WORDS.items() if k == key)}
        if not any(v in english.casefold() for v in variants) and word not in added:
            added.append(word)
    if added:
        english = english.rstrip(". ") + ", " + ", ".join(
            w if w.startswith("in their") else f"{w} people" for w in added)
    return english, added


def needs_translation(text: str) -> bool:
    return bool(_CJK.search(text or ""))


def llama_completion(models_dir: Path) -> Path:
    return Path(models_dir) / "quality_v2" / "llamacpp" / "llama-completion.exe"


def translator_ready(models_dir: Path) -> bool:
    return llama_completion(models_dir).exists() and (Path(models_dir) / "quality_v2" / QWEN3_4B_FILE).exists()


@lru_cache(maxsize=256)
def _translate_cached(text: str, models_dir: str, threads: int, timeout: float) -> str:
    exe = llama_completion(Path(models_dir))
    model = ascii_file(Path(models_dir) / "quality_v2" / QWEN3_4B_FILE)
    turn = "<|im_start|>user\n{} /no_think<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    shots = "".join(turn.format(source) + f"{target}<|im_end|>\n" for source, target in FEW_SHOT)
    prompt = f"<|im_start|>system\n{SYSTEM}<|im_end|>\n{shots}" + turn.format(text)
    parent = None if tempfile.gettempdir().isascii() else ascii_work_root()
    with tempfile.TemporaryDirectory(prefix="covermorph_tr_", dir=parent) as tmp:
        prompt_file = Path(tmp) / "prompt.txt"
        prompt_file.write_text(prompt, encoding="utf-8")
        done = subprocess.run(
            [str(exe), "-m", str(model), "-f", str(prompt_file), "-n", "160", "--temp", "0", "--no-display-prompt",
             "-no-cnv", "-t", str(threads)],  # --log-disable would also drop the generated text when piped
            capture_output=True, stdin=subprocess.DEVNULL, timeout=timeout, cwd=str(exe.parent),
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    output = _ANSI.sub("", done.stdout.decode("utf-8", errors="replace"))
    output = re.sub(r"</?think>", "", output.replace("[end of text]", "")).strip().strip('"').strip()
    output = " ".join(output.split())
    if done.returncode != 0 or not output or needs_translation(output) or len(output) > 600:
        raise RuntimeError(f"translation failed (exit {done.returncode}): {output[:120]!r}")
    return output


def translate_prompt(text: str, models_dir: Path, *, threads: int = 8, timeout: float = 90.0) -> dict:
    """Return {"english", "method", "seconds", "error"}; ``english`` never contains CJK script."""
    from .thumbnail_bridge_ai import translate_scene_terms

    text = (text or "").strip()
    if not needs_translation(text):
        return {"english": text, "method": "none", "seconds": 0.0, "error": ""}
    started = time.perf_counter()
    if translator_ready(models_dir):
        try:
            english = _translate_cached(text, str(Path(models_dir)), threads, timeout)
            english, restored = restore_kept_terms(text, english)
            return {"english": english, "method": "qwen3-4b llama.cpp", "seconds": round(time.perf_counter() - started, 2),
                    "error": "", "restored_terms": restored}
        except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
            error = str(exc)
    else:
        error = "translator not installed (scripts/prepare_quality_v2_models.py --only llamacpp)"
    english, _, untranslated = translate_scene_terms(text)
    english = _CJK.sub(" ", english)
    english = " ".join(english.split())
    note = "; some words were not understood" if untranslated else ""
    return {"english": english, "method": "vocabulary", "seconds": round(time.perf_counter() - started, 2),
            "error": error + note}
