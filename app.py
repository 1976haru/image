from pathlib import Path
from tkinter import messagebox

from covermorph.logger import write_exception
from covermorph.quick_gui import QuickCoverApp


def main() -> None:
    app = QuickCoverApp(Path(__file__).resolve().parent)
    app.mainloop()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        write_exception(Path(__file__).resolve().parent, "TOPLEVEL", exc)
        try:
            messagebox.showerror("CoverMorph Studio", f"프로그램 오류가 발생했습니다.\n{exc}")
        except Exception:
            pass
        raise
