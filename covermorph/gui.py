from __future__ import annotations

import queue
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Any

import customtkinter as ctk
from PIL import Image, ImageTk, UnidentifiedImageError

from . import __version__
from .ai_plugins import AIBackends
from .logger import write_exception, write_log
from .pipeline import DEFAULT_OUTPAINT_PROMPT, PipelineOptions, process_files
from .presets import PRESETS
from .processor import Rect, inpaint_text_opencv, mask_pil_from_boxes

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


def app_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


class CoverMorphApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.root_dir = app_root()
        self.ai = AIBackends(self.root_dir)

        self.title(f"CoverMorph Studio v{__version__}")
        self.geometry("1460x900")
        self.minsize(1220, 760)

        self.files: list[Path] = []
        self.original: Image.Image | None = None
        self.current: Image.Image | None = None
        self.before_image: Image.Image | None = None
        self.after_image: Image.Image | None = None
        self.showing_after = True

        self.preview_tk: ImageTk.PhotoImage | None = None
        self.text_boxes: list[Rect] = []
        self.preview_scale = 1.0
        self.preview_offset = (0, 0)
        self.drag_start: tuple[int, int] | None = None
        self.output_dir: Path | None = None

        self.worker_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self.worker_thread: threading.Thread | None = None
        self.status_thread: threading.Thread | None = None
        self.cancel_event = threading.Event()
        self.busy_task: str | None = None
        self.busy_widgets: list[Any] = []

        self.build_ui()
        self.refresh_ai_status()
        write_log(self.root_dir, f"CoverMorph Studio v{__version__} started")

    def build_ui(self) -> None:
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        left = ctk.CTkScrollableFrame(self, width=370)
        left.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        right = ctk.CTkFrame(self)
        right.grid(row=0, column=1, sticky="nsew", padx=(0, 10), pady=10)
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(0, weight=1)

        ctk.CTkLabel(
            left,
            text="CoverMorph Studio",
            font=ctk.CTkFont(size=25, weight="bold"),
        ).pack(pady=(12, 2))
        ctk.CTkLabel(
            left,
            text=f"v{__version__} Stability Edition",
            text_color="#9ca3af",
        ).pack(pady=(0, 12))

        self.load_button = ctk.CTkButton(
            left,
            text="1. 커버 이미지 불러오기",
            command=self.open_files,
            height=40,
        )
        self.load_button.pack(fill="x", padx=10, pady=5)
        self.output_button = ctk.CTkButton(
            left,
            text="2. 출력 폴더 지정",
            command=self.choose_output,
            height=38,
        )
        self.output_button.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(
            left,
            text="채널 프리셋",
            anchor="w",
            font=ctk.CTkFont(weight="bold"),
        ).pack(fill="x", padx=12, pady=(16, 5))
        self.preset_name = ctk.StringVar(value="OldPopLounge")
        self.preset_menu = ctk.CTkOptionMenu(
            left,
            variable=self.preset_name,
            values=list(PRESETS.keys()),
            command=self.update_preset_desc,
        )
        self.preset_menu.pack(fill="x", padx=10)
        self.preset_desc = ctk.CTkLabel(
            left,
            text=PRESETS["OldPopLounge"].description,
            wraplength=315,
            justify="left",
            text_color="#cbd5e1",
        )
        self.preset_desc.pack(fill="x", padx=12, pady=6)

        ctk.CTkLabel(
            left,
            text="OCR 언어",
            anchor="w",
            font=ctk.CTkFont(weight="bold"),
        ).pack(fill="x", padx=12, pady=(12, 5))
        self.ocr_lang = ctk.StringVar(value="영어")
        self.ocr_menu = ctk.CTkOptionMenu(
            left,
            variable=self.ocr_lang,
            values=["영어", "한국어+영어", "일본어+영어"],
        )
        self.ocr_menu.pack(fill="x", padx=10)

        ctk.CTkLabel(
            left,
            text="AI 엔진",
            anchor="w",
            font=ctk.CTkFont(weight="bold"),
        ).pack(fill="x", padx=12, pady=(15, 5))
        self.ai_status = ctk.CTkLabel(left, text="", wraplength=315, justify="left")
        self.ai_status.pack(fill="x", padx=12, pady=4)
        self.ai_status_button = ctk.CTkButton(
            left,
            text="AI 상태 새로고침",
            command=self.refresh_ai_status,
            height=30,
            fg_color="#4b5563",
        )
        self.ai_status_button.pack(fill="x", padx=10, pady=4)

        self.prefer_lama = ctk.BooleanVar(value=True)
        self.prefer_esrgan = ctk.BooleanVar(value=True)
        self.auto_remove = ctk.BooleanVar(value=True)
        self.enhance = ctk.BooleanVar(value=True)
        self.out_square = ctk.BooleanVar(value=True)
        self.out_thumb = ctk.BooleanVar(value=True)
        self.out_shorts = ctk.BooleanVar(value=True)
        self.use_outpaint = ctk.BooleanVar(value=False)
        self.protect_person = ctk.BooleanVar(value=True)

        self.option_widgets = [
            ctk.CTkCheckBox(left, text="LaMa 우선 글자 제거", variable=self.prefer_lama),
            ctk.CTkCheckBox(left, text="Real-ESRGAN 우선 업스케일", variable=self.prefer_esrgan),
            ctk.CTkCheckBox(left, text="SDXL AI 배경 확장", variable=self.use_outpaint),
            ctk.CTkCheckBox(left, text="사람 원본 픽셀 보호", variable=self.protect_person),
            ctk.CTkCheckBox(left, text="글자 자동 삭제", variable=self.auto_remove),
            ctk.CTkCheckBox(left, text="기본 화질 보정", variable=self.enhance),
            ctk.CTkCheckBox(left, text="1:1 클린 커버", variable=self.out_square),
            ctk.CTkCheckBox(left, text="16:9 썸네일", variable=self.out_thumb),
            ctk.CTkCheckBox(left, text="9:16 숏츠", variable=self.out_shorts),
        ]
        for index, widget in enumerate(self.option_widgets):
            pady = (10, 3) if index == 0 else 3
            widget.pack(anchor="w", padx=13, pady=pady)

        ctk.CTkLabel(
            left,
            text="AI 배경 프롬프트",
            anchor="w",
            font=ctk.CTkFont(weight="bold"),
        ).pack(fill="x", padx=12, pady=(12, 4))
        self.outpaint_prompt = ctk.CTkTextbox(left, height=72)
        self.outpaint_prompt.pack(fill="x", padx=10)
        self.outpaint_prompt.insert("1.0", DEFAULT_OUTPAINT_PROMPT)

        self.detect_button = ctk.CTkButton(
            left,
            text="글자 자동 탐지",
            command=self.detect_text,
            height=34,
            fg_color="#4b5563",
        )
        self.detect_button.pack(fill="x", padx=10, pady=(13, 4))
        self.preview_button = ctk.CTkButton(
            left,
            text="미리보기 글자 제거",
            command=self.preview_remove,
            height=34,
            fg_color="#4b5563",
        )
        self.preview_button.pack(fill="x", padx=10, pady=4)
        self.toggle_button = ctk.CTkButton(
            left,
            text="Before / After 전환",
            command=self.toggle_before_after,
            height=34,
            fg_color="#374151",
        )
        self.toggle_button.pack(fill="x", padx=10, pady=4)
        self.clear_button = ctk.CTkButton(
            left,
            text="마스크 초기화",
            command=self.clear_boxes,
            height=32,
            fg_color="#374151",
        )
        self.clear_button.pack(fill="x", padx=10, pady=4)

        self.status = ctk.CTkLabel(
            left,
            text="커버 이미지를 불러오세요.",
            wraplength=315,
            justify="left",
        )
        self.status.pack(fill="x", padx=12, pady=(15, 8))
        self.progress_label = ctk.CTkLabel(
            left,
            text="0/0 | 성공 0 | 실패 0",
            text_color="#cbd5e1",
        )
        self.progress_label.pack(fill="x", padx=12, pady=(0, 6))
        self.progress_bar = ctk.CTkProgressBar(left)
        self.progress_bar.pack(fill="x", padx=10, pady=(0, 8))
        self.progress_bar.set(0)

        self.run_button = ctk.CTkButton(
            left,
            text="원클릭 AI 자동 변환",
            command=self.run_pipeline,
            height=52,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color="#198754",
            hover_color="#146c43",
        )
        self.run_button.pack(fill="x", padx=10, pady=(4, 6))
        self.cancel_button = ctk.CTkButton(
            left,
            text="작업 취소",
            command=self.cancel_work,
            height=38,
            fg_color="#7f1d1d",
            hover_color="#991b1b",
            state="disabled",
        )
        self.cancel_button.pack(fill="x", padx=10, pady=(0, 16))

        self.busy_widgets = [
            self.load_button,
            self.output_button,
            self.preset_menu,
            self.ocr_menu,
            self.ai_status_button,
            *self.option_widgets,
            self.detect_button,
            self.preview_button,
            self.toggle_button,
            self.clear_button,
            self.run_button,
        ]

        self.canvas = tk.Canvas(right, bg="#0f172a", highlightthickness=0, cursor="cross")
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Configure>", lambda _event: self.draw_preview())

        footer = ctk.CTkFrame(right)
        footer.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))
        ctk.CTkLabel(
            footer,
            text="자동 탐지에서 빠진 글자는 미리보기에서 직접 드래그해 마스크를 추가하세요.",
        ).pack(side="left", padx=12, pady=8)
        self.view_label = ctk.CTkLabel(footer, text="VIEW: AFTER")
        self.view_label.pack(side="right", padx=12)

    def set_busy(self, busy: bool, task: str | None = None) -> None:
        self.busy_task = task if busy else None
        state = "disabled" if busy else "normal"
        for widget in self.busy_widgets:
            widget.configure(state=state)
        self.cancel_button.configure(state="normal" if busy else "disabled")

    def start_worker(self, task: str, worker: Any) -> None:
        if self.worker_thread is not None and self.worker_thread.is_alive():
            messagebox.showwarning("작업 중", "이미 실행 중인 작업이 있습니다.")
            return
        self.cancel_event.clear()
        self.set_busy(True, task)

        def runner() -> None:
            try:
                worker()
            except Exception as exc:
                write_exception(self.root_dir, task, exc)
                self.worker_queue.put({"type": "worker_error", "task": task, "error": str(exc)})
            finally:
                self.worker_queue.put({"type": "worker_finished", "task": task})

        self.worker_thread = threading.Thread(target=runner, name=f"CoverMorph-{task}", daemon=True)
        self.worker_thread.start()
        self.after(100, self.poll_worker_queue)

    def poll_worker_queue(self) -> None:
        while True:
            try:
                event = self.worker_queue.get_nowait()
            except queue.Empty:
                break
            self.handle_worker_event(event)

        worker_alive = self.worker_thread is not None and self.worker_thread.is_alive()
        status_alive = self.status_thread is not None and self.status_thread.is_alive()
        if worker_alive or status_alive or not self.worker_queue.empty():
            self.after(100, self.poll_worker_queue)

    def handle_worker_event(self, event: dict[str, Any]) -> None:
        event_type = event.get("type")
        if event_type == "status":
            self.status.configure(text=event.get("message", "작업 중..."))
        elif event_type == "ai_status_done":
            status = event["status"]
            self.ai_status.configure(
                text=(
                    f"LaMa: {'사용 가능' if status['lama'] else '미설치'}\n"
                    f"Real-ESRGAN: {'사용 가능' if status['realesrgan'] else '실행파일 없음'}\n"
                    f"CUDA: {'감지됨' if status['cuda'] else '없음/미사용'}\n"
                    f"사람 분리: {'사용 가능' if status['rembg'] else '미설치'}\n"
                    f"SDXL: {'사용 가능' if status['sdxl'] else '미설치'}"
                )
            )
        elif event_type == "ocr_done":
            self.text_boxes = event["boxes"]
            self.status.configure(
                text=f"글자 영역 {len(self.text_boxes)}개 탐지\n빠진 글자는 직접 드래그하세요."
            )
            self.draw_preview()
        elif event_type == "preview_done":
            self.current = event["image"]
            self.after_image = event["image"].copy()
            self.showing_after = True
            self.text_boxes = []
            self.status.configure(text=f"미리보기 제거 완료: {event['engine']}")
            self.draw_preview()
        elif event_type == "pipeline_progress":
            self.handle_pipeline_progress(event["payload"])
        elif event_type == "pipeline_done":
            self.handle_pipeline_done(event["results"], event["output_dir"])
        elif event_type == "worker_error":
            self.status.configure(text=f"오류: {event['error']}\n수동 마스크나 로그를 확인하세요.")
            messagebox.showerror("CoverMorph 오류", event["error"])
        elif event_type == "worker_finished":
            self.worker_thread = None
            self.set_busy(False)

    def handle_pipeline_progress(self, payload: dict[str, Any]) -> None:
        event_type = payload.get("type")
        if event_type == "status":
            self.status.configure(text=payload.get("message", "작업 중..."))
            return
        if event_type == "file_start":
            index = payload["index"]
            total = payload["total"]
            self.progress_bar.set((index - 1) / max(1, total))
            self.status.configure(text=f"처리 중 {index}/{total}\n{payload['filename']}")
            self.progress_label.configure(
                text=f"{index - 1}/{total} | 성공 {payload['success_files']} | 실패 {payload['failed_files']}"
            )
        elif event_type == "file_done":
            index = payload["index"]
            total = payload["total"]
            self.progress_bar.set(index / max(1, total))
            self.status.configure(
                text=(
                    f"처리 완료 {index}/{total}\n"
                    f"{payload['filename']}\n"
                    f"상태: {payload['status']}"
                )
            )
            self.progress_label.configure(
                text=(
                    f"{index}/{total} | 성공 {payload['success_files']} | "
                    f"실패 {payload['failed_files']} | 결과 {payload['success_outputs']}"
                )
            )
        elif event_type == "cancelled":
            self.status.configure(text=payload.get("message", "작업이 취소되었습니다."))
        elif event_type == "batch_done":
            processed = payload["processed"]
            total = payload["total"]
            self.progress_bar.set(processed / max(1, total))

    def handle_pipeline_done(self, results: list[Any], output_dir: Path) -> None:
        success_files = sum(1 for result in results if result.status == "success")
        failed_results = [result for result in results if result.status != "success"]
        failed_names = [result.source.name for result in failed_results]
        success_outputs = sum(result.success_outputs for result in results)
        failed_outputs = sum(result.failed_outputs for result in results)

        details = "\n".join(failed_names[:12])
        if len(failed_names) > 12:
            details += f"\n...외 {len(failed_names) - 12}개"

        message = (
            f"성공 파일: {success_files}개\n"
            f"실패/부분 실패 파일: {len(failed_results)}개\n"
            f"생성 결과물: {success_outputs}개\n"
            f"저장 실패 결과물: {failed_outputs}개"
        )
        if failed_names:
            message += f"\n\n실패 파일:\n{details}"
        self.status.configure(text=f"완료\n{output_dir}\n{message}")
        messagebox.showinfo("CoverMorph 완료", message)

    def refresh_ai_status(self) -> None:
        if self.status_thread is not None and self.status_thread.is_alive():
            return
        self.ai_status.configure(text="AI 상태 확인 중...")

        def runner() -> None:
            try:
                self.worker_queue.put({"type": "ai_status_done", "status": self.ai.status()})
            except Exception as exc:
                write_exception(self.root_dir, "AI status", exc)
                self.worker_queue.put({"type": "status", "message": f"AI 상태 확인 실패: {exc}"})

        self.status_thread = threading.Thread(target=runner, name="CoverMorph-AIStatus", daemon=True)
        self.status_thread.start()
        self.after(100, self.poll_worker_queue)

    def lang_codes(self) -> tuple[str, ...]:
        value = self.ocr_lang.get()
        if "한국어" in value:
            return ("ko", "en")
        if "일본어" in value:
            return ("ja", "en")
        return ("en",)

    def update_preset_desc(self, *_args: Any) -> None:
        self.preset_desc.configure(text=PRESETS[self.preset_name.get()].description)

    def open_files(self) -> None:
        paths = filedialog.askopenfilenames(filetypes=[("Images", "*.jpg *.jpeg *.png *.webp *.bmp")])
        if not paths:
            return
        self.files = [Path(path) for path in paths]
        try:
            with Image.open(self.files[0]) as opened:
                self.original = opened.convert("RGB")
        except (OSError, UnidentifiedImageError) as exc:
            write_exception(self.root_dir, "Open image", exc)
            messagebox.showerror("이미지 오류", f"이미지를 열 수 없습니다.\n{self.files[0].name}\n{exc}")
            self.files = []
            return

        self.current = self.original.copy()
        self.before_image = self.original.copy()
        self.after_image = self.current.copy()
        self.text_boxes = []
        self.status.configure(text=f"{len(self.files)}장 선택\n첫 이미지: {self.files[0].name}")
        self.progress_label.configure(text=f"0/{len(self.files)} | 성공 0 | 실패 0")
        self.progress_bar.set(0)
        self.draw_preview()

    def choose_output(self) -> None:
        directory = filedialog.askdirectory(title="자동 저장 폴더")
        if directory:
            self.output_dir = Path(directory)
            self.status.configure(text=f"출력 폴더:\n{self.output_dir}")

    def detect_text(self) -> None:
        if self.current is None:
            return
        img = self.current.copy()
        langs = self.lang_codes()
        self.status.configure(text="글자 탐지 준비 중...")

        def worker() -> None:
            from .processor import detect_text_boxes_easyocr

            boxes = detect_text_boxes_easyocr(
                img,
                langs,
                status_callback=lambda message: self.worker_queue.put(
                    {"type": "status", "message": message}
                ),
            )
            self.worker_queue.put({"type": "ocr_done", "boxes": boxes})

        self.start_worker("OCR", worker)

    def remove_with_backend(self, img: Image.Image, boxes: list[Rect], prefer_lama: bool) -> tuple[Image.Image, str]:
        if not boxes:
            return img.copy(), "No text mask"
        if prefer_lama and self.ai.lama_available():
            try:
                return self.ai.inpaint(img, mask_pil_from_boxes(img.size, boxes))
            except Exception as exc:
                write_exception(self.root_dir, "LaMa preview fallback", exc)
        return inpaint_text_opencv(img, boxes), "OpenCV Telea"

    def preview_remove(self) -> None:
        if self.current is None or not self.text_boxes:
            return
        img = self.current.copy()
        boxes = list(self.text_boxes)
        prefer_lama = self.prefer_lama.get()
        self.before_image = self.current.copy()
        self.status.configure(text="미리보기 글자 제거 중...")

        def worker() -> None:
            out, engine = self.remove_with_backend(img, boxes, prefer_lama)
            self.worker_queue.put({"type": "preview_done", "image": out, "engine": engine})

        self.start_worker("PreviewRemove", worker)

    def toggle_before_after(self) -> None:
        if self.before_image is None or self.after_image is None:
            return
        self.showing_after = not self.showing_after
        self.current = (self.after_image if self.showing_after else self.before_image).copy()
        self.view_label.configure(text="VIEW: AFTER" if self.showing_after else "VIEW: BEFORE")
        self.draw_preview()

    def clear_boxes(self) -> None:
        self.text_boxes = []
        self.draw_preview()

    def on_press(self, event: tk.Event) -> None:
        if self.current is not None:
            self.drag_start = (event.x, event.y)

    def on_release(self, event: tk.Event) -> None:
        if self.current is None or self.drag_start is None:
            return
        x1, y1 = self.drag_start
        x2, y2 = event.x, event.y
        offset_x, offset_y = self.preview_offset
        scale = self.preview_scale
        img_x1 = int((min(x1, x2) - offset_x) / scale)
        img_y1 = int((min(y1, y2) - offset_y) / scale)
        img_x2 = int((max(x1, x2) - offset_x) / scale)
        img_y2 = int((max(y1, y2) - offset_y) / scale)
        img_x1 = max(0, min(img_x1, self.current.width))
        img_x2 = max(0, min(img_x2, self.current.width))
        img_y1 = max(0, min(img_y1, self.current.height))
        img_y2 = max(0, min(img_y2, self.current.height))
        if img_x2 - img_x1 > 5 and img_y2 - img_y1 > 5:
            self.text_boxes.append((img_x1, img_y1, img_x2, img_y2))
        self.drag_start = None
        self.draw_preview()

    def draw_preview(self) -> None:
        self.canvas.delete("all")
        if self.current is None:
            return
        canvas_width = max(100, self.canvas.winfo_width())
        canvas_height = max(100, self.canvas.winfo_height())
        scale = min((canvas_width - 30) / self.current.width, (canvas_height - 30) / self.current.height)
        preview_width = int(self.current.width * scale)
        preview_height = int(self.current.height * scale)
        preview = self.current.resize((preview_width, preview_height), Image.Resampling.LANCZOS)
        self.preview_tk = ImageTk.PhotoImage(preview)
        offset_x = (canvas_width - preview_width) // 2
        offset_y = (canvas_height - preview_height) // 2
        self.preview_scale = scale
        self.preview_offset = (offset_x, offset_y)
        self.canvas.create_image(offset_x, offset_y, image=self.preview_tk, anchor="nw")
        for left, top, right, bottom in self.text_boxes:
            self.canvas.create_rectangle(
                offset_x + left * scale,
                offset_y + top * scale,
                offset_x + right * scale,
                offset_y + bottom * scale,
                outline="#ff416c",
                width=2,
            )

    def pipeline_options(self) -> PipelineOptions:
        output_dir = self.output_dir
        if output_dir is None:
            raise RuntimeError("출력 폴더가 지정되지 않았습니다.")
        return PipelineOptions(
            output_dir=output_dir,
            preset_name=self.preset_name.get(),
            ocr_languages=self.lang_codes(),
            manual_boxes=tuple(self.text_boxes),
            auto_remove_text=self.auto_remove.get(),
            prefer_lama=self.prefer_lama.get(),
            prefer_esrgan=self.prefer_esrgan.get(),
            enhance=self.enhance.get(),
            out_square=self.out_square.get(),
            out_thumb=self.out_thumb.get(),
            out_shorts=self.out_shorts.get(),
            use_sdxl=self.use_outpaint.get(),
            protect_person=self.protect_person.get(),
            outpaint_prompt=self.outpaint_prompt.get("1.0", "end").strip(),
        )

    def run_pipeline(self) -> None:
        if not self.files:
            messagebox.showinfo("안내", "먼저 커버 이미지를 불러오세요.")
            return
        if self.output_dir is None:
            self.choose_output()
            if self.output_dir is None:
                return
        options = self.pipeline_options()
        files = list(self.files)
        self.progress_bar.set(0)
        self.progress_label.configure(text=f"0/{len(files)} | 성공 0 | 실패 0")

        def worker() -> None:
            results = process_files(
                files,
                options,
                self.root_dir,
                ai=self.ai,
                progress_callback=lambda payload: self.worker_queue.put(
                    {"type": "pipeline_progress", "payload": payload}
                ),
                cancel_event=self.cancel_event,
            )
            self.worker_queue.put({"type": "pipeline_done", "results": results, "output_dir": options.output_dir})

        self.start_worker("Pipeline", worker)

    def cancel_work(self) -> None:
        self.cancel_event.set()
        self.status.configure(text="현재 단계가 끝나면 작업을 취소합니다.")
