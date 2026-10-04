"""Studio dialogs: product mask editor, product position/size, first-run setup wizard, friendly error dialog."""
from __future__ import annotations

import shutil
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog
from typing import Any, Callable

import customtkinter as ctk
from PIL import Image, ImageTk

from .creator_presets import MEMORY_LABELS, QUALITY_LABELS
from .errors import UserError


# ====================================================================== error dialog
class ErrorDialog(ctk.CTkToplevel):
    """Title, plain message and what to do; technical details only behind '자세히 보기'."""

    def __init__(self, master: Any, error: UserError, details_path: Path | None = None):
        super().__init__(master)
        self.title(error.title)
        self.geometry("620x300")
        self.transient(master)
        ctk.CTkLabel(self, text=error.title, font=ctk.CTkFont(size=16, weight="bold"), anchor="w").pack(fill="x", padx=16, pady=(16, 4))
        ctk.CTkLabel(self, text=error.message, anchor="w", justify="left", wraplength=580).pack(fill="x", padx=16)
        if error.action:
            ctk.CTkLabel(self, text="해결 방법: " + error.action, anchor="w", justify="left", wraplength=580,
                         text_color="#93c5fd").pack(fill="x", padx=16, pady=(8, 0))
        self.details = ctk.CTkTextbox(self, height=1)
        self._details_text = f"오류 코드: {error.code}\n" + (f"로그: {details_path}\n\n" if details_path else "\n") + (error.details or "")
        buttons = ctk.CTkFrame(self, fg_color="transparent")
        buttons.pack(fill="x", padx=16, pady=12, side="bottom")
        ctk.CTkButton(buttons, text="확인", width=90, command=self.destroy).pack(side="right")
        ctk.CTkButton(buttons, text="자세히 보기", width=110, fg_color="#475569", command=self._toggle).pack(side="right", padx=8)
        self._shown = False

    def _toggle(self) -> None:
        if self._shown:
            self.details.pack_forget()
            self.geometry("620x300")
        else:
            self.details.configure(height=220)
            self.details.delete("1.0", "end")
            self.details.insert("1.0", self._details_text)
            self.details.pack(fill="both", expand=True, padx=16)
            self.geometry("760x560")
        self._shown = not self._shown


def show_error(master: Any, exc: BaseException, label: str = "ui") -> None:
    from .errors import classify, remember_last_error, write_details
    error = classify(exc)
    path = write_details(error, label)
    remember_last_error(error, path)
    ErrorDialog(master, error, path)


# ====================================================================== mask editor
class MaskEditor(ctk.CTkToplevel):
    """Automatic product mask + simple corrections. Red tint = left out of the product."""

    VIEW = 620

    def __init__(self, master: Any, image_path: Path, mask_path: str | None, on_save: Callable[[Path], None]):
        super().__init__(master)
        from .product_checks import auto_mask
        from .product_masks import load_mask

        self.title("상품 마스크 확인·수정")
        self.geometry("1040x760")
        self.transient(master)
        self.image_path = Path(image_path)
        with Image.open(image_path) as opened:
            self.image = opened.convert("RGBA") if opened.mode in ("RGBA", "LA") else opened.convert("RGB")
        self.on_save = on_save
        self.auto = auto_mask(self.image)
        self.base = load_mask(mask_path, self.image.size) or self.auto
        self.strokes: list[tuple[float, float, float, bool]] = []
        self.scale = min(self.VIEW / self.image.width, self.VIEW / self.image.height)
        self._photo = None
        self._pending = False

        left = ctk.CTkFrame(self)
        left.pack(side="left", padx=10, pady=10)
        self.canvas = tk.Canvas(left, width=int(self.image.width * self.scale), height=int(self.image.height * self.scale),
                                highlightthickness=0, bg="#111827", cursor="circle")
        self.canvas.pack()
        self.canvas.bind("<ButtonPress-1>", self._stroke)
        self.canvas.bind("<B1-Motion>", self._stroke)
        side = ctk.CTkFrame(self)
        side.pack(side="left", fill="both", expand=True, padx=(0, 10), pady=10)
        self.report_label = ctk.CTkLabel(side, text="", wraplength=330, justify="left", anchor="w")
        self.report_label.pack(fill="x", padx=10, pady=(10, 6))
        ctk.CTkLabel(side, text="빨간 부분은 상품에서 제외됩니다. 상품이 빨갛게 칠해졌다면 '추가' 브러시로 칠하세요.",
                     wraplength=330, justify="left", anchor="w", text_color="#94a3b8").pack(fill="x", padx=10)
        self.invert_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(side, text="반전", variable=self.invert_var, command=self._redraw).pack(anchor="w", padx=10, pady=6)
        self.grow = self._slider(side, "확장(+) / 축소(-)", -15, 15, 0)
        self.feather = self._slider(side, "가장자리 부드럽게", 0, 8, 1)
        ctk.CTkLabel(side, text="브러시").pack(anchor="w", padx=10, pady=(10, 0))
        self.brush_mode = ctk.StringVar(value="추가")
        ctk.CTkSegmentedButton(side, values=["추가", "삭제"], variable=self.brush_mode).pack(fill="x", padx=10)
        self.brush = self._slider(side, "브러시 크기", 4, 80, 24, redraw=False)
        row = ctk.CTkFrame(side, fg_color="transparent")
        row.pack(fill="x", padx=10, pady=8)
        ctk.CTkButton(row, text="되돌리기", width=90, command=self._undo).pack(side="left")
        ctk.CTkButton(row, text="자동 마스크로", width=110, command=self._reset).pack(side="left", padx=6)
        bottom = ctk.CTkFrame(side, fg_color="transparent")
        bottom.pack(fill="x", padx=10, pady=14, side="bottom")
        ctk.CTkButton(bottom, text="저장", fg_color="#16a34a", command=self._save).pack(side="right")
        ctk.CTkButton(bottom, text="취소", fg_color="#475569", command=self.destroy).pack(side="right", padx=8)
        self._redraw()

    def _slider(self, parent, label, low, high, value, redraw=True):
        ctk.CTkLabel(parent, text=label).pack(anchor="w", padx=10, pady=(8, 0))
        slider = ctk.CTkSlider(parent, from_=low, to=high, number_of_steps=int(high - low),
                               command=(lambda _v: self._redraw()) if redraw else None)
        slider.set(value)
        slider.pack(fill="x", padx=10)
        return slider

    def current(self) -> Image.Image:
        from .product_masks import paint, refine
        mask = refine(self.base, invert=bool(self.invert_var.get()), grow=int(round(self.grow.get())))
        mask = paint(mask, self.strokes)
        return refine(mask, feather=float(self.feather.get()))

    def _stroke(self, event) -> None:
        x, y = event.x / self.scale, event.y / self.scale
        self.strokes.append((x, y, float(self.brush.get()) / self.scale / 2, self.brush_mode.get() == "추가"))
        if not self._pending:
            self._pending = True
            self.after(40, self._redraw)

    def _undo(self) -> None:
        # one drag adds many points: undo the last ~ stroke worth of points
        del self.strokes[-12:]
        self._redraw()

    def _reset(self) -> None:
        self.base, self.strokes = self.auto, []
        self.invert_var.set(False)
        self.grow.set(0)
        self._redraw()

    def _redraw(self) -> None:
        from .product_masks import assess, preview
        self._pending = False
        mask = self.current()
        shown = preview(self.image, mask, (int(self.image.width * self.scale), int(self.image.height * self.scale)))
        self._photo = ImageTk.PhotoImage(shown)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo)
        report = assess(self.image, mask)
        self.report_label.configure(text=f"상품 영역 {report.coverage:.0%} · 뚫린 부분 {report.holes}개\n{report.message}",
                                    text_color="#fbbf24" if report.uncertain else "#86efac")

    def _save(self) -> None:
        from .product_masks import save_mask
        path = save_mask(self.image_path, self.current())
        self.on_save(path)
        self.destroy()


# ====================================================================== placement
class PlacementDialog(ctk.CTkToplevel):
    """Move/resize the original product on the same background (no AI; product pixels unchanged)."""

    def __init__(self, master: Any, job_dir: Path, record: dict[str, Any], purpose: Any, on_done: Callable[[dict], None]):
        super().__init__(master)
        from .product_checks import Placement
        self.title("상품 위치·크기 조정")
        self.geometry("980x700")
        self.transient(master)
        self.job_dir, self.record, self.purpose, self.on_done = Path(job_dir), record, purpose, on_done
        with Image.open(record["background"]) as opened:
            self.background = opened.convert("RGB")
        self.cutout = Image.open(self.job_dir / "refs" / "product_cutout.png").convert("RGBA")
        start = record.get("placement") or Placement(0.8, 0.5, 0.35).to_dict()
        self.label = ctk.CTkLabel(self, text="")
        self.label.pack(padx=10, pady=10)
        self.baseline = self._slider("바닥 높이 (위↔아래)", 0.3, 0.99, start["baseline"])
        self.center = self._slider("좌우 위치", 0.05, 0.95, start["center_x"])
        self.size = self._slider("크기", 0.08, 0.9, start["scale"])
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", padx=10, pady=10)
        ctk.CTkButton(row, text="새 후보로 저장", fg_color="#16a34a", command=self._apply).pack(side="right")
        ctk.CTkButton(row, text="취소", fg_color="#475569", command=self.destroy).pack(side="right", padx=8)
        self._small_bg = self.background.copy()
        self._small_bg.thumbnail((900, 520))
        factor = self._small_bg.width / self.background.width
        self._small_cut = self.cutout.resize((max(1, int(self.cutout.width * factor)), max(1, int(self.cutout.height * factor))))
        self._redraw()

    def _slider(self, label: str, low: float, high: float, value: float):
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", padx=10)
        ctk.CTkLabel(row, text=label, width=150, anchor="w").pack(side="left")
        slider = ctk.CTkSlider(row, from_=low, to=high, command=lambda _v: self._redraw())
        slider.set(value)
        slider.pack(side="left", fill="x", expand=True)
        return slider

    def _placement(self):
        from .product_checks import Placement
        return Placement(float(self.baseline.get()), float(self.center.get()), float(self.size.get()))

    def _redraw(self) -> None:
        from .product_checks import composite_product
        image, _ = composite_product(self._small_bg, self._small_cut, self.purpose.subject_region, placement=self._placement())
        self._photo = ctk.CTkImage(light_image=image, dark_image=image, size=image.size)
        self.label.configure(image=self._photo)

    def _apply(self) -> None:
        from .creator_jobs import recomposite
        placement = self._placement()
        record = recomposite(self.job_dir, self.record, self.purpose, baseline=placement.baseline,
                             center_x=placement.center_x, scale=placement.scale)
        self.on_done(record)
        self.destroy()


# ====================================================================== first-run wizard
class SetupWizard(ctk.CTkToplevel):
    """Four short steps: model folder (checked) → usage → quality → PC usage. No backend jargon."""

    def __init__(self, master: Any, app_root: Path, settings: dict[str, Any], on_finish: Callable[[dict[str, Any]], None]):
        super().__init__(master)
        self.title("처음 설정 — CoverMorph Studio")
        self.geometry("760x560")
        self.transient(master)
        self.grab_set()
        self.app_root, self.settings, self.on_finish = Path(app_root), dict(settings), on_finish
        self.step = 0
        self.body = ctk.CTkFrame(self)
        self.body.pack(fill="both", expand=True, padx=14, pady=14)
        nav = ctk.CTkFrame(self, fg_color="transparent")
        nav.pack(fill="x", padx=14, pady=(0, 14))
        self.step_label = ctk.CTkLabel(nav, text="")
        self.step_label.pack(side="left")
        self.next_button = ctk.CTkButton(nav, text="다음", width=110, command=self._next)
        self.next_button.pack(side="right")
        self.back_button = ctk.CTkButton(nav, text="이전", width=90, fg_color="#475569", command=self._back)
        self.back_button.pack(side="right", padx=8)
        from .creator_settings import resolve_models_dir
        self.models_var = ctk.StringVar(value=str(resolve_models_dir(self.app_root, self.settings)))
        self.usage_var = ctk.StringVar(value={"youtube": "YouTube", "shopify": "Shopify"}.get(self.settings.get("usage"), "둘 다"))
        self.quality_var = ctk.StringVar(value=self.settings.get("quality_label") or "일반")
        self.memory_var = ctk.StringVar(value=self.settings.get("memory_label") or "작업 중 PC 우선")
        self.ready = False
        self._render()

    def _clear(self) -> None:
        for child in self.body.winfo_children():
            child.destroy()

    def _title(self, text: str, hint: str = "") -> None:
        ctk.CTkLabel(self.body, text=text, font=ctk.CTkFont(size=17, weight="bold"), anchor="w").pack(fill="x", padx=12, pady=(12, 4))
        if hint:
            ctk.CTkLabel(self.body, text=hint, anchor="w", justify="left", wraplength=680, text_color="#94a3b8").pack(fill="x", padx=12)

    def _render(self) -> None:
        self._clear()
        self.step_label.configure(text=f"{self.step + 1} / 4 단계")
        self.back_button.configure(state="normal" if self.step else "disabled")
        self.next_button.configure(text="완료" if self.step == 3 else "다음")
        if self.step == 0:
            self._title("1. 모델 폴더", "이미지 엔진 파일(quality_v2 폴더 등)이 들어 있는 폴더를 한 번만 지정하면 됩니다. "
                        "설정은 사용자 폴더에 저장되어 프로그램을 업데이트해도 유지됩니다.")
            row = ctk.CTkFrame(self.body, fg_color="transparent")
            row.pack(fill="x", padx=12, pady=10)
            ctk.CTkEntry(row, textvariable=self.models_var, width=520).pack(side="left")
            ctk.CTkButton(row, text="찾아보기", width=90, command=self._browse).pack(side="left", padx=6)
            ctk.CTkButton(row, text="확인", width=70, command=self._check).pack(side="left")
            self.status = ctk.CTkTextbox(self.body, height=230)
            self.status.pack(fill="both", expand=True, padx=12, pady=6)
            self._check()
        elif self.step == 1:
            self._title("2. 주로 무엇을 만드나요?", "처음 화면에 먼저 보일 작업입니다. 언제든 바꿀 수 있습니다.")
            ctk.CTkSegmentedButton(self.body, values=["YouTube", "Shopify", "둘 다"], variable=self.usage_var,
                                   height=40).pack(fill="x", padx=12, pady=20)
        elif self.step == 2:
            self._title("3. 기본 품질", "빠른 미리보기: 약 20초 1장 · 일반: 장당 약 35초, 후보 2장 · 최고 품질: 여러 엔진을 차례로 비교(느림)")
            ctk.CTkSegmentedButton(self.body, values=list(QUALITY_LABELS), variable=self.quality_var,
                                   height=40).pack(fill="x", padx=12, pady=20)
        else:
            self._title("4. PC 사용 방식", "작업 중 PC 우선: 한 장씩 만들고 바로 메모리를 돌려줍니다(권장) · 균형 · "
                        "자리 비움: 자리를 비운 동안 최고 해상도로 진행")
            ctk.CTkSegmentedButton(self.body, values=list(MEMORY_LABELS), variable=self.memory_var,
                                   height=40).pack(fill="x", padx=12, pady=20)

    def _browse(self) -> None:
        chosen = filedialog.askdirectory(parent=self, title="모델 폴더 선택")
        if chosen:
            self.models_var.set(chosen)
            self._check()

    def _check(self) -> None:
        self.status.delete("1.0", "end")
        self.status.insert("1.0", "확인 중…")
        threading.Thread(target=self._check_worker, args=(Path(self.models_var.get()),), daemon=True).start()

    def _check_worker(self, models: Path) -> None:
        from .creator_settings import engine_readiness
        from .creator_settings import resolve_output_dir
        from .diagnostics import _gpu
        lines = []
        if not models.is_dir():
            lines.append("이 폴더가 없습니다. 다시 선택하세요.")
            ready = None
        else:
            ready = engine_readiness(models)
            names = {"zimage_turbo": "Z-Image (기본 이미지 생성)", "flux2_klein_4b": "FLUX.2 (레퍼런스·편집)",
                     "realvisxl_v5": "RealVis (대체 엔진)"}
            for key, label in names.items():
                lines.append(f"{'준비됨' if ready['engines'][key]['ready'] else '없음  '} · {label}")
            lines.append(f"{'준비됨' if ready['translator'] else '없음  '} · 한국어/일본어 프롬프트 번역")
        lines.append("")
        lines.append(f"GPU: {_gpu()}")
        try:
            output = resolve_output_dir(self.app_root, self.settings)
            probe = output if output.exists() else next((p for p in output.parents if p.exists()), output)
            lines.append(f"저장 공간(출력 폴더 드라이브): {shutil.disk_usage(probe).free / 1024 ** 3:.0f} GB 여유")
        except OSError:
            pass
        ok = bool(ready and ready["engines"]["zimage_turbo"]["ready"] and ready["engines"]["flux2_klein_4b"]["ready"])
        if not ok:
            lines += ["", "기본 엔진(Z-Image, FLUX.2)이 없으면 이미지를 만들 수 없습니다. 모델 폴더를 확인하세요."]
        self.after(0, lambda: self._show_check(lines, ok))

    def _show_check(self, lines: list[str], ok: bool) -> None:
        self.ready = ok
        try:
            self.status.delete("1.0", "end")
            self.status.insert("1.0", "\n".join(lines))
        except tk.TclError:
            pass

    def _next(self) -> None:
        if self.step == 0 and not self.ready:
            from tkinter import messagebox
            if not messagebox.askyesno("모델 폴더", "기본 엔진을 찾지 못했습니다. 그래도 계속할까요?\n(나중에 설정 탭에서 바꿀 수 있습니다)",
                                       parent=self):
                return
        if self.step < 3:
            self.step += 1
            self._render()
            return
        self.settings.update(models_dir=self.models_var.get().strip(),
                             usage={"YouTube": "youtube", "Shopify": "shopify"}.get(self.usage_var.get(), "both"),
                             quality_label=self.quality_var.get(), memory_label=self.memory_var.get(),
                             setup_completed=True)
        self.grab_release()
        self.on_finish(self.settings)
        self.destroy()

    def _back(self) -> None:
        if self.step:
            self.step -= 1
            self._render()
