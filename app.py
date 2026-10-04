from __future__ import annotations

import sys


def _run_gui() -> int:
    from tkinter import messagebox

    from covermorph.gui import CoverMorphApp, app_root
    from covermorph.logger import write_exception

    try:
        app = CoverMorphApp()
        app.mainloop()
        return 0
    except Exception as exc:
        write_exception(app_root(), "TOPLEVEL", exc)
        try:
            messagebox.showerror("CoverMorph Studio", f"프로그램 오류가 발생했습니다.\n{exc}")
        except Exception:
            pass
        raise


def _record_exe_path() -> None:
    """Let youtubesum find this EXE without environment variables (it reads state.exe_path)."""
    if not getattr(sys, "frozen", False):
        return
    try:
        from covermorph import __version__, app_paths
        app_paths.update_section("state", {"exe_path": sys.executable, "exe_version": __version__})
    except Exception:
        pass


def main() -> int:
    # Headless bridge: --thumbnail-bridge-json (contract), or youtubesum's --image-bridge / --action forms.
    argv = sys.argv[1:]
    if not argv or argv[0] in ("--studio-selftest", "--studio-endurance"):
        _record_exe_path()
    if argv[:1] == ["--studio-selftest"]:  # packaged-app acceptance run of the AI 이미지 스튜디오
        from covermorph.creator_gui import app_root_dir, run_selftest
        return run_selftest(app_root_dir(), argv[1] if len(argv) > 1 else "studio_selftest")
    if argv[:1] == ["--studio-endurance"]:  # packaged-app endurance phases (scripts/validate_endurance.py)
        from covermorph.creator_gui import app_root_dir, run_endurance_phase
        return run_endurance_phase(app_root_dir(), argv[1], argv[2])
    if argv[:1] == ["--ocr-worker"]:  # short-lived OCR process for the product logo check
        from covermorph.ocr_worker import main as ocr_main
        return ocr_main(argv[1:])
    if "--thumbnail-bridge-json" in argv or "--image-bridge" in argv or "--action" in argv:
        from covermorph.thumbnail_bridge_runtime import run_bridge_cli
        return run_bridge_cli(argv=argv)
    return _run_gui()


if __name__ == "__main__":
    raise SystemExit(main())
