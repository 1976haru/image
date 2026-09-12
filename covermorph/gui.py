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
from .settings import (
    DEFAULT_SETTINGS,
    DUPLICATE_POLICIES,
    THUMBNAIL_RESOLUTIONS,
    default_output_dir_for_source,
    ensure_output_directory,
    expected_output_count,
    has_selected_outputs,
    load_settings,
    open_folder,
    save_settings,
    selected_output_labels,
)

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
        self.settings = load_settings(self.root_dir)
        self.ai = AIBackends(self.root_dir)

        self.title(f"CoverMorph Studio v{__version__}")
        self.geometry("1480x920")
        self.minsize(1240, 780)

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
        self.apply_saved_output_directory_notice()
        self.update_summary()
        self.refresh_ai_status()
        write_log(self.root_dir, f"CoverMorph Studio v{__version__} started")

    def group(self, parent: Any, title: str, *, highlight: bool = False) -> ctk.CTkFrame:
        frame = ctk.CTkFrame(parent, fg_color="#111827" if highlight else "#1f2937", border_width=1)
        frame.pack(fill="x", padx=10, pady=7)
        ctk.CTkLabel(frame, text=title, anchor="w", font=ctk.CTkFont(weight="bold")).pack(
            fill="x",
            padx=12,
            pady=(10, 6),
        )
        return frame

    def build_ui(self) -> None:
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        left = ctk.CTkScrollableFrame(self, width=410)
        left.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        right = ctk.CTkFrame(self)
        right.grid(row=0, column=1, sticky="nsew", padx=(0, 10), pady=10)
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(0, weight=1)

        ctk.CTkLabel(left, text="CoverMorph Studio", font=ctk.CTkFont(size=25, weight="bold")).pack(
            pady=(12, 2)
        )
        ctk.CTkLabel(left, text=f"v{__version__} Output Settings Edition", text_color="#9ca3af").pack(
            pady=(0, 10)
        )

        self.prefer_lama = ctk.BooleanVar(value=True)
        self.prefer_esrgan = ctk.BooleanVar(value=True)
        self.auto_remove = ctk.BooleanVar(value=True)
        self.enhance = ctk.BooleanVar(value=True)
        self.out_square = ctk.BooleanVar(value=self.settings["output_square"])
        self.out_thumb = ctk.BooleanVar(value=self.settings["output_thumbnail"])
        self.out_shorts = ctk.BooleanVar(value=self.settings["output_shorts"])
        self.use_outpaint = ctk.BooleanVar(value=False)
        self.protect_person = ctk.BooleanVar(value=True)
        self.preset_name = ctk.StringVar(value=self.settings["last_preset"])
        self.ocr_lang = ctk.StringVar(value=self.settings["last_ocr_language"])
        self.output_path_var = ctk.StringVar(value=self.settings["output_directory"])
        self.thumbnail_resolution = ctk.StringVar(value=self.settings["thumbnail_resolution"])
        self.duplicate_policy_label = ctk.StringVar(
            value=DUPLICATE_POLICIES[self.settings["duplicate_policy"]]
        )

        self.build_input_group(left)
        self.build_output_folder_group(left)
        self.build_preset_group(left)
        self.build_ocr_group(left)
        self.build_output_selection_group(left)
        self.build_extension_group(left)
        self.build_ai_group(left)
        self.build_progress_group(left)
        self.build_run_group(left)
        self.bind_setting_traces()

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

    def build_input_group(self, parent: Any) -> None:
        frame = self.group(parent, "1. 입력 이미지")
        self.load_button = ctk.CTkButton(frame, text="커버 이미지 불러오기", command=self.open_files, height=38)
        self.load_button.pack(fill="x", padx=12, pady=(0, 12))

    def build_output_folder_group(self, parent: Any) -> None:
        frame = self.group(parent, "2. 출력 폴더")
        ctk.CTkLabel(frame, text="출력 폴더", anchor="w", text_color="#cbd5e1").pack(fill="x", padx=12)
        self.output_entry = ctk.CTkEntry(frame, textvariable=self.output_path_var)
        self.output_entry.pack(fill="x", padx=12, pady=(3, 8))

        row = ctk.CTkFrame(frame, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=(0, 12))
        self.output_browse_button = ctk.CTkButton(row, text="찾아보기", command=self.choose_output, width=96)
        self.output_browse_button.pack(side="left", padx=(0, 6))
        self.output_open_button = ctk.CTkButton(row, text="폴더 열기", command=self.open_output_folder, width=96)
        self.output_open_button.pack(side="left", padx=6)
        self.output_reset_button = ctk.CTkButton(row, text="기본값 복원", command=self.restore_default_output, width=104)
        self.output_reset_button.pack(side="left", padx=6)

    def build_preset_group(self, parent: Any) -> None:
        frame = self.group(parent, "3. 채널 프리셋")
        preset_value = self.preset_name.get()
        if preset_value not in PRESETS:
            preset_value = "OldPopLounge"
            self.preset_name.set(preset_value)
        self.preset_menu = ctk.CTkOptionMenu(
            frame,
            variable=self.preset_name,
            values=list(PRESETS.keys()),
            command=self.update_preset_desc,
        )
        self.preset_menu.pack(fill="x", padx=12)
        self.preset_desc = ctk.CTkLabel(
            frame,
            text=PRESETS[preset_value].description,
            wraplength=350,
            justify="left",
            text_color="#cbd5e1",
        )
        self.preset_desc.pack(fill="x", padx=12, pady=(6, 12))

    def build_ocr_group(self, parent: Any) -> None:
        frame = self.group(parent, "4. OCR 및 글자 제거")
        self.ocr_menu = ctk.CTkOptionMenu(
            frame,
            variable=self.ocr_lang,
            values=["영어", "한국어+영어", "일본어+영어"],
        )
        self.ocr_menu.pack(fill="x", padx=12, pady=(0, 8))
        self.auto_remove_check = ctk.CTkCheckBox(frame, text="글자 자동 삭제", variable=self.auto_remove)
        self.auto_remove_check.pack(anchor="w", padx=12, pady=3)
        self.prefer_lama_check = ctk.CTkCheckBox(frame, text="LaMa 우선 글자 제거", variable=self.prefer_lama)
        self.prefer_lama_check.pack(anchor="w", padx=12, pady=3)
        self.detect_button = ctk.CTkButton(
            frame,
            text="글자 자동 탐지",
            command=self.detect_text,
            height=32,
            fg_color="#4b5563",
        )
        self.detect_button.pack(fill="x", padx=12, pady=(9, 4))
        self.preview_button = ctk.CTkButton(
            frame,
            text="미리보기 글자 제거",
            command=self.preview_remove,
            height=32,
            fg_color="#4b5563",
        )
        self.preview_button.pack(fill="x", padx=12, pady=4)
        self.toggle_button = ctk.CTkButton(
            frame,
            text="Before / After 전환",
            command=self.toggle_before_after,
            height=32,
            fg_color="#374151",
        )
        self.toggle_button.pack(fill="x", padx=12, pady=4)
        self.clear_button = ctk.CTkButton(
            frame,
            text="마스크 초기화",
            command=self.clear_boxes,
            height=30,
            fg_color="#374151",
        )
        self.clear_button.pack(fill="x", padx=12, pady=(4, 12))

    def build_output_selection_group(self, parent: Any) -> None:
        frame = self.group(parent, "5. 출력 이미지 선택", highlight=True)
        self.square_check = ctk.CTkCheckBox(
            frame,
            text="1:1 클린 커버\n  - 1400x1400 JPG\n  - 글자를 제거한 정사각형 이미지",
            variable=self.out_square,
        )
        self.square_check.pack(anchor="w", padx=12, pady=(0, 9))
        self.thumb_check = ctk.CTkCheckBox(
            frame,
            text="16:9 유튜브 썸네일\n  - 기본 1920x1080 JPG\n  - 1280x720 선택 가능",
            variable=self.out_thumb,
        )
        self.thumb_check.pack(anchor="w", padx=12, pady=9)
        thumb_row = ctk.CTkFrame(frame, fg_color="transparent")
        thumb_row.pack(fill="x", padx=36, pady=(0, 9))
        ctk.CTkLabel(thumb_row, text="썸네일 해상도", text_color="#cbd5e1").pack(side="left", padx=(0, 8))
        self.thumbnail_resolution_menu = ctk.CTkOptionMenu(
            thumb_row,
            variable=self.thumbnail_resolution,
            values=list(THUMBNAIL_RESOLUTIONS.keys()),
            width=132,
        )
        self.thumbnail_resolution_menu.pack(side="left")
        self.shorts_check = ctk.CTkCheckBox(
            frame,
            text="9:16 숏츠 이미지\n  - 1080x1920 JPG\n  - YouTube Shorts, Instagram Reels, TikTok용",
            variable=self.out_shorts,
        )
        self.shorts_check.pack(anchor="w", padx=12, pady=9)

        button_row = ctk.CTkFrame(frame, fg_color="transparent")
        button_row.pack(fill="x", padx=12, pady=(4, 12))
        self.select_all_button = ctk.CTkButton(button_row, text="전체 선택", command=self.select_all_outputs)
        self.select_all_button.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.clear_all_button = ctk.CTkButton(button_row, text="전체 해제", command=self.clear_all_outputs)
        self.clear_all_button.pack(side="left", fill="x", expand=True, padx=(5, 0))

    def build_extension_group(self, parent: Any) -> None:
        frame = self.group(parent, "6. 이미지 확장 방식")
        self.enhance_check = ctk.CTkCheckBox(frame, text="기본 화질 보정", variable=self.enhance)
        self.enhance_check.pack(anchor="w", padx=12, pady=3)
        self.use_outpaint_check = ctk.CTkCheckBox(frame, text="SDXL AI 배경 확장", variable=self.use_outpaint)
        self.use_outpaint_check.pack(anchor="w", padx=12, pady=3)
        self.protect_person_check = ctk.CTkCheckBox(frame, text="사람 원본 픽셀 보호", variable=self.protect_person)
        self.protect_person_check.pack(anchor="w", padx=12, pady=3)
        ctk.CTkLabel(frame, text="AI 배경 프롬프트", anchor="w", text_color="#cbd5e1").pack(
            fill="x",
            padx=12,
            pady=(10, 3),
        )
        self.outpaint_prompt = ctk.CTkTextbox(frame, height=70)
        self.outpaint_prompt.pack(fill="x", padx=12, pady=(0, 12))
        self.outpaint_prompt.insert("1.0", DEFAULT_OUTPAINT_PROMPT)

    def build_ai_group(self, parent: Any) -> None:
        frame = self.group(parent, "7. AI 엔진")
        self.prefer_esrgan_check = ctk.CTkCheckBox(
            frame,
            text="Real-ESRGAN 우선 업스케일",
            variable=self.prefer_esrgan,
        )
        self.prefer_esrgan_check.pack(anchor="w", padx=12, pady=(0, 8))
        self.ai_status = ctk.CTkLabel(frame, text="", wraplength=350, justify="left")
        self.ai_status.pack(fill="x", padx=12, pady=4)
        self.ai_status_button = ctk.CTkButton(
            frame,
            text="AI 상태 새로고침",
            command=self.refresh_ai_status,
            height=30,
            fg_color="#4b5563",
        )
        self.ai_status_button.pack(fill="x", padx=12, pady=(4, 12))

    def build_progress_group(self, parent: Any) -> None:
        frame = self.group(parent, "8. 진행 상태")
        ctk.CTkLabel(frame, text="중복 파일 처리", anchor="w", text_color="#cbd5e1").pack(fill="x", padx=12)
        self.duplicate_policy_menu = ctk.CTkOptionMenu(
            frame,
            variable=self.duplicate_policy_label,
            values=list(DUPLICATE_POLICIES.values()),
        )
        self.duplicate_policy_menu.pack(fill="x", padx=12, pady=(3, 8))
        self.status = ctk.CTkLabel(frame, text="커버 이미지를 불러오세요.", wraplength=350, justify="left")
        self.status.pack(fill="x", padx=12, pady=(6, 8))
        self.progress_label = ctk.CTkLabel(frame, text="0/0 | 성공 0 | 실패 0", text_color="#cbd5e1")
        self.progress_label.pack(fill="x", padx=12, pady=(0, 6))
        self.progress_bar = ctk.CTkProgressBar(frame)
        self.progress_bar.pack(fill="x", padx=12, pady=(0, 12))
        self.progress_bar.set(0)

    def build_run_group(self, parent: Any) -> None:
        frame = self.group(parent, "9. 변환 시작 및 취소")
        self.summary_label = ctk.CTkLabel(frame, text="", wraplength=350, justify="left", anchor="w")
        self.summary_label.pack(fill="x", padx=12, pady=(0, 10))
        self.run_button = ctk.CTkButton(
            frame,
            text="원클릭 AI 자동 변환",
            command=self.run_pipeline,
            height=52,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color="#198754",
            hover_color="#146c43",
        )
        self.run_button.pack(fill="x", padx=12, pady=(0, 6))
        self.cancel_button = ctk.CTkButton(
            frame,
            text="작업 취소",
            command=self.cancel_work,
            height=38,
            fg_color="#7f1d1d",
            hover_color="#991b1b",
            state="disabled",
        )
        self.cancel_button.pack(fill="x", padx=12, pady=(0, 12))

        self.busy_widgets = [
            self.load_button,
            self.output_entry,
            self.output_browse_button,
            self.output_open_button,
            self.output_reset_button,
            self.preset_menu,
            self.ocr_menu,
            self.auto_remove_check,
            self.prefer_lama_check,
            self.detect_button,
            self.preview_button,
            self.toggle_button,
            self.clear_button,
            self.square_check,
            self.thumb_check,
            self.thumbnail_resolution_menu,
            self.shorts_check,
            self.select_all_button,
            self.clear_all_button,
            self.enhance_check,
            self.use_outpaint_check,
            self.protect_person_check,
            self.prefer_esrgan_check,
            self.ai_status_button,
            self.duplicate_policy_menu,
            self.run_button,
        ]

    def bind_setting_traces(self) -> None:
        for var in (
            self.out_square,
            self.out_thumb,
            self.out_shorts,
            self.thumbnail_resolution,
            self.preset_name,
            self.ocr_lang,
            self.duplicate_policy_label,
        ):
            var.trace_add("write", self.on_persistent_setting_changed)
        self.output_path_var.trace_add("write", lambda *_args: self.update_summary())

    def settings_payload(self, *, include_output: bool = True) -> dict[str, Any]:
        reverse_policy = {label: key for key, label in DUPLICATE_POLICIES.items()}
        output_directory = self.output_path_var.get().strip() if include_output else self.settings["output_directory"]
        return {
            "output_directory": output_directory,
            "output_square": self.out_square.get(),
            "output_thumbnail": self.out_thumb.get(),
            "output_shorts": self.out_shorts.get(),
            "thumbnail_resolution": self.thumbnail_resolution.get(),
            "duplicate_policy": reverse_policy.get(self.duplicate_policy_label.get(), "new_number"),
            "last_preset": self.preset_name.get(),
            "last_ocr_language": self.ocr_lang.get(),
        }

    def save_current_settings(self, *, include_output: bool = True) -> None:
        self.settings = self.settings_payload(include_output=include_output)
        try:
            save_settings(self.root_dir, self.settings)
        except OSError as exc:
            write_exception(self.root_dir, "Save settings", exc)

    def on_persistent_setting_changed(self, *_args: Any) -> None:
        self.save_current_settings(include_output=False)
        self.update_summary()

    def apply_saved_output_directory_notice(self) -> None:
        path_text = self.output_path_var.get().strip()
        if not path_text:
            return
        path = Path(path_text)
        self.output_dir = path
        if not path.exists() or not path.is_dir():
            self.status.configure(text="저장된 출력 폴더를 사용할 수 없습니다. 새 출력 폴더를 선택해주세요.")

    def select_all_outputs(self) -> None:
        self.out_square.set(True)
        self.out_thumb.set(True)
        self.out_shorts.set(True)
        self.save_current_settings()
        self.update_summary()

    def clear_all_outputs(self) -> None:
        self.out_square.set(False)
        self.out_thumb.set(False)
        self.out_shorts.set(False)
        self.save_current_settings()
        self.update_summary()

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
            self.handle_pipeline_done(event["results"], event["output_dir"], event["selected_labels"])
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
                text=f"처리 완료 {index}/{total}\n{payload['filename']}\n상태: {payload['status']}"
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

    def handle_pipeline_done(
        self,
        results: list[Any],
        output_dir: Path,
        selected_labels: list[str],
    ) -> None:
        success_files = sum(1 for result in results if result.status == "success")
        failed_results = [result for result in results if result.status not in {"success", "skipped"}]
        success_outputs = sum(result.success_outputs for result in results)
        failed_outputs = sum(result.failed_outputs for result in results)
        skipped_outputs = sum(result.skipped_outputs for result in results)
        failed_names = [result.source.name for result in failed_results]

        message = (
            f"저장된 출력 폴더:\n{output_dir}\n\n"
            f"생성된 이미지 개수: {success_outputs}개\n"
            f"성공한 원본 개수: {success_files}개\n"
            f"실패한 원본 개수: {len(failed_results)}개\n"
            f"건너뛴 결과물: {skipped_outputs}개\n"
            f"저장 실패 결과물: {failed_outputs}개\n"
            f"생성된 규격: {', '.join(selected_labels)}"
        )
        if failed_names:
            message += "\n\n실패 파일:\n" + "\n".join(failed_names[:12])
            if len(failed_names) > 12:
                message += f"\n...외 {len(failed_names) - 12}개"
        self.status.configure(text=f"완료\n{output_dir}\n생성 {success_outputs}개")
        self.show_completion_dialog(message, output_dir)
        self.update_summary()

    def show_completion_dialog(self, message: str, output_dir: Path) -> None:
        dialog = ctk.CTkToplevel(self)
        dialog.title("CoverMorph 완료")
        dialog.geometry("520x380")
        dialog.transient(self)
        dialog.grab_set()
        ctk.CTkLabel(dialog, text="변환 완료", font=ctk.CTkFont(size=18, weight="bold")).pack(
            anchor="w",
            padx=18,
            pady=(18, 8),
        )
        ctk.CTkLabel(dialog, text=message, justify="left", wraplength=470).pack(
            fill="both",
            expand=True,
            padx=18,
            pady=(0, 12),
        )
        row = ctk.CTkFrame(dialog, fg_color="transparent")
        row.pack(fill="x", padx=18, pady=(0, 18))
        ctk.CTkButton(row, text="출력 폴더 열기", command=lambda: self.safe_open_folder(output_dir)).pack(
            side="left",
            fill="x",
            expand=True,
            padx=(0, 6),
        )
        ctk.CTkButton(row, text="확인", command=dialog.destroy).pack(
            side="left",
            fill="x",
            expand=True,
            padx=(6, 0),
        )

    def safe_open_folder(self, path: Path) -> None:
        if not open_folder(path):
            messagebox.showwarning("폴더 열기", "출력 폴더를 열 수 없습니다.")

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
        preset_value = self.preset_name.get()
        if preset_value in PRESETS:
            self.preset_desc.configure(text=PRESETS[preset_value].description)

    def default_output_for_current_files(self) -> Path | None:
        if not self.files:
            return None
        return default_output_dir_for_source(self.files[0])

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

        if not self.output_path_var.get().strip():
            default_dir = default_output_dir_for_source(self.files[0])
            self.output_path_var.set(str(default_dir))
            self.output_dir = default_dir

        self.current = self.original.copy()
        self.before_image = self.original.copy()
        self.after_image = self.current.copy()
        self.text_boxes = []
        self.status.configure(text=f"{len(self.files)}장 선택\n첫 이미지: {self.files[0].name}")
        self.progress_label.configure(text=f"0/{len(self.files)} | 성공 0 | 실패 0")
        self.progress_bar.set(0)
        self.draw_preview()
        self.update_summary()

    def choose_output(self) -> None:
        directory = filedialog.askdirectory(title="자동 저장 폴더")
        if directory:
            self.output_path_var.set(directory)
            self.output_dir = Path(directory)
            self.save_current_settings(include_output=True)
            self.status.configure(text=f"출력 폴더:\n{self.output_dir}")
            self.update_summary()

    def restore_default_output(self) -> None:
        default_dir = self.default_output_for_current_files()
        if default_dir is None:
            self.output_path_var.set(DEFAULT_SETTINGS["output_directory"])
            self.output_dir = None
            self.status.configure(text="이미지를 불러오면 원본 폴더 아래 CoverMorph_Output을 기본값으로 사용합니다.")
        else:
            self.output_path_var.set(str(default_dir))
            self.output_dir = default_dir
            self.status.configure(text=f"기본 출력 폴더:\n{default_dir}")
        self.save_current_settings(include_output=True)
        self.update_summary()

    def open_output_folder(self) -> None:
        path_text = self.output_path_var.get().strip()
        if not path_text:
            messagebox.showinfo("출력 폴더", "출력 폴더를 먼저 지정해주세요.")
            return
        try:
            path = ensure_output_directory(path_text, create=False)
        except FileNotFoundError:
            if not messagebox.askyesno("출력 폴더", "출력 폴더가 없습니다. 지금 생성할까요?"):
                return
            try:
                path = ensure_output_directory(path_text, create=True)
            except (OSError, ValueError, PermissionError) as exc:
                messagebox.showerror("출력 폴더 오류", str(exc))
                return
        except (OSError, ValueError, PermissionError) as exc:
            messagebox.showerror("출력 폴더 오류", str(exc))
            return
        self.output_dir = path
        self.save_current_settings(include_output=True)
        self.safe_open_folder(path)

    def validate_output_directory_for_run(self) -> Path | None:
        path_text = self.output_path_var.get().strip()
        if not path_text and self.files:
            path_text = str(default_output_dir_for_source(self.files[0]))
            self.output_path_var.set(path_text)
        if not path_text:
            messagebox.showinfo("출력 폴더", "출력 폴더를 선택해주세요.")
            return None

        try:
            path = ensure_output_directory(path_text, create=False)
        except FileNotFoundError:
            if not messagebox.askyesno("출력 폴더 생성", f"출력 폴더가 없습니다.\n{path_text}\n\n생성할까요?"):
                return None
            try:
                path = ensure_output_directory(path_text, create=True)
            except (OSError, ValueError, PermissionError) as exc:
                messagebox.showerror("출력 폴더 오류", str(exc))
                return None
        except (OSError, ValueError, PermissionError) as exc:
            messagebox.showerror("출력 폴더 오류", str(exc))
            return None

        self.output_dir = path
        self.output_path_var.set(str(path))
        self.save_current_settings(include_output=True)
        return path

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
                status_callback=lambda message: self.worker_queue.put({"type": "status", "message": message}),
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

    def selected_output_labels(self) -> list[str]:
        return selected_output_labels(self.out_square.get(), self.out_thumb.get(), self.out_shorts.get())

    def update_summary(self) -> None:
        if not hasattr(self, "summary_label"):
            return
        labels = self.selected_output_labels()
        output_dir = self.output_path_var.get().strip() or "이미지 선택 후 자동 설정"
        planned = "\n ".join(labels) if labels else "선택된 출력 없음"
        file_count = len(self.files)
        estimated = expected_output_count(file_count, self.out_square.get(), self.out_thumb.get(), self.out_shorts.get())
        self.summary_label.configure(
            text=(
                f"출력 폴더:\n{output_dir}\n\n"
                f"생성 예정:\n {planned}\n\n"
                f"선택 이미지: {file_count}장\n"
                f"예상 결과물: {estimated}개"
            )
        )

    def pipeline_options(self, output_dir: Path) -> PipelineOptions:
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
            thumbnail_resolution=self.thumbnail_resolution.get(),
            duplicate_policy=self.settings_payload()["duplicate_policy"],
            use_sdxl=self.use_outpaint.get(),
            protect_person=self.protect_person.get(),
            outpaint_prompt=self.outpaint_prompt.get("1.0", "end").strip(),
        )

    def run_pipeline(self) -> None:
        if not self.files:
            messagebox.showinfo("안내", "먼저 커버 이미지를 불러오세요.")
            return
        if not has_selected_outputs(self.out_square.get(), self.out_thumb.get(), self.out_shorts.get()):
            messagebox.showinfo("안내", "생성할 출력 이미지 규격을 하나 이상 선택해주세요.")
            return

        output_dir = self.validate_output_directory_for_run()
        if output_dir is None:
            return

        options = self.pipeline_options(output_dir)
        files = list(self.files)
        selected_labels = self.selected_output_labels()
        self.progress_bar.set(0)
        self.progress_label.configure(text=f"0/{len(files)} | 성공 0 | 실패 0")
        self.update_summary()

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
            self.worker_queue.put(
                {
                    "type": "pipeline_done",
                    "results": results,
                    "output_dir": options.output_dir,
                    "selected_labels": selected_labels,
                }
            )

        self.start_worker("Pipeline", worker)

    def cancel_work(self) -> None:
        self.cancel_event.set()
        self.status.configure(text="현재 단계가 끝나면 작업을 취소합니다.")
