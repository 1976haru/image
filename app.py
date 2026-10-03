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


def main() -> int:
    # Headless bridge: --thumbnail-bridge-json (contract), or youtubesum's --image-bridge / --action forms.
    argv = sys.argv[1:]
    if argv[:1] == ["--ocr-worker"]:  # short-lived OCR process for the product logo check
        from covermorph.ocr_worker import main as ocr_main
        return ocr_main(argv[1:])
    if "--thumbnail-bridge-json" in argv or "--image-bridge" in argv or "--action" in argv:
        from covermorph.thumbnail_bridge_runtime import run_bridge_cli
        return run_bridge_cli(argv=argv)
    return _run_gui()


if __name__ == "__main__":
    raise SystemExit(main())
