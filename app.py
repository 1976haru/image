from tkinter import messagebox

from covermorph.gui import CoverMorphApp, app_root
from covermorph.logger import write_exception


def main() -> None:
    app = CoverMorphApp()
    app.mainloop()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        write_exception(app_root(), "TOPLEVEL", exc)
        try:
            messagebox.showerror("CoverMorph Studio", f"프로그램 오류가 발생했습니다.\n{exc}")
        except Exception:
            pass
        raise
