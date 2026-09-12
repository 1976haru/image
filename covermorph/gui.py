from __future__ import annotations

import queue
import shutil
import sys
import threading
import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Any

import customtkinter as ctk
from PIL import Image, ImageColor, ImageOps, ImageTk, UnidentifiedImageError

from . import __version__
from .ai_plugins import AIBackends
from .logger import write_exception, write_log
from .pipeline import (
    DEFAULT_OUTPAINT_PROMPT,
    ImageJob,
    PipelineOptions,
    count_selected_outputs_for_jobs,
    output_count_by_kind,
    process_image_jobs,
)
from .presets import PRESETS
from .processor import (
    Rect,
    build_full_frame_outpaint_canvas,
    detect_text_boxes_multilang,
    inpaint_text_opencv,
    make_square,
    mask_pil_from_boxes,
    render_full_frame_format,
)
from .settings import (
    DEFAULT_SETTINGS,
    DUPLICATE_POLICIES,
    EXTENSION_MODES,
    THUMBNAIL_RESOLUTIONS,
    default_output_dir_for_source,
    ensure_output_directory,
    load_settings,
    open_folder,
    save_settings,
    thumbnail_size,
)

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD

    class _CoverMorphWindow(ctk.CTk, TkinterDnD.DnDWrapper):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            try:
                self.TkdndVersion = TkinterDnD._require(self)
                self._covermorph_dnd_ready = True
            except tk.TclError:
                self._covermorph_dnd_ready = False

except ImportError:
    DND_FILES = None

    class _CoverMorphWindow(ctk.CTk):
        _covermorph_dnd_ready = False


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

MAX_IMAGES = 5
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
OCR_LANGUAGE_LABELS = {
    "영어": ("en",),
    "한국어+영어": ("ko", "en"),
    "일본어+영어": ("ja", "en"),
    "한국어+영어+일본어": ("ko", "en", "ja"),
    "자동 전체 언어 탐지": ("en", "ja", "ko", "fr"),
}
PREVIEW_TABS = ["원본", "OCR 마스크", "수동 마스크", "글자 제거 결과", "최종 결과", "1:1 미리보기", "16:9 미리보기", "9:16 미리보기"]
OUTPUT_KEYS = {
    "square_1x1": "1:1 클린 커버",
    "thumbnail_16x9": "16:9 썸네일",
    "shorts_9x16": "9:16 숏츠",
}


@dataclass(slots=True)
class ImageRowState:
    job: ImageJob
    original: Image.Image
    thumbnail_photo: ImageTk.PhotoImage
    clean_preview: Image.Image | None = None
    row_frame: ctk.CTkFrame | None = None
    square_var: ctk.BooleanVar | None = None
    thumb_var: ctk.BooleanVar | None = None
    shorts_var: ctk.BooleanVar | None = None
    extension_var: ctk.StringVar | None = None
    status_label: ctk.CTkLabel | None = None
    row_widgets: list[Any] = field(default_factory=list)
    result: Any | None = None


def app_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


class CoverMorphApp(_CoverMorphWindow):
    def __init__(self) -> None:
        super().__init__()
        self.root_dir = app_root()
        self.settings = load_settings(self.root_dir)
        self.ai = AIBackends(self.root_dir)

        self.title(f"CoverMorph Studio v{__version__}")
        self.geometry("1580x940")
        self.minsize(1280, 780)

        self.image_states: list[ImageRowState] = []
        self.selected_index: int | None = None
        self.output_dir: Path | None = None

        self.preview_tk: ImageTk.PhotoImage | None = None
        self.preview_scale = 1.0
        self.preview_offset = (0, 0)
        self.drag_start: tuple[int, int] | None = None

        self.worker_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self.worker_thread: threading.Thread | None = None
        self.status_thread: threading.Thread | None = None
        self.cancel_event = threading.Event()
        self.busy_task: str | None = None
        self.fixed_busy_widgets: list[Any] = []
        self._loading_selection = False

        self.build_ui()
        self.setup_drag_drop()
        self.apply_saved_output_directory_notice()
        self.update_summary()
        self.refresh_ai_status()
        write_log(self.root_dir, f"CoverMorph Studio v{__version__} started")

    def group(self, parent: Any, title: str, *, highlight: bool = False) -> ctk.CTkFrame:
        frame = ctk.CTkFrame(
            parent,
            fg_color="#111827" if highlight else "#1f2937",
            border_color="#38bdf8" if highlight else "#334155",
            border_width=1,
            corner_radius=8,
        )
        frame.pack(fill="x", padx=10, pady=7)
        ctk.CTkLabel(frame, text=title, anchor="w", font=ctk.CTkFont(weight="bold")).pack(
            fill="x",
            padx=12,
            pady=(10, 6),
        )
        return frame

    def build_ui(self) -> None:
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self.build_top_control_bar()

        left = ctk.CTkScrollableFrame(self, width=420)
        left.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))

        right = ctk.CTkFrame(self)
        right.grid(row=1, column=1, sticky="nsew", padx=(0, 10), pady=(0, 10))
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(left, text="CoverMorph Studio", font=ctk.CTkFont(size=25, weight="bold")).pack(
            pady=(12, 2)
        )
        ctk.CTkLabel(left, text=f"v{__version__} Multi-image Edition", text_color="#9ca3af").pack(
            pady=(0, 10)
        )

        self.auto_remove = ctk.BooleanVar(value=True)
        self.prefer_lama = ctk.BooleanVar(value=True)
        self.prefer_esrgan = ctk.BooleanVar(value=True)
        self.enhance = ctk.BooleanVar(value=True)
        self.protect_person = ctk.BooleanVar(value=True)
        self.protect_core = ctk.BooleanVar(value=self.settings["protect_core"])
        self.output_square_var = ctk.BooleanVar(value=self.settings["output_square"])
        self.output_thumb_var = ctk.BooleanVar(value=self.settings["output_thumbnail"])
        self.output_shorts_var = ctk.BooleanVar(value=self.settings["output_shorts"])
        self.preset_name = ctk.StringVar(value=self.settings["last_preset"])
        self.ocr_lang = ctk.StringVar(value=self.settings["last_ocr_language"])
        self.output_path_var = ctk.StringVar(value=self.settings["output_directory"])
        self.thumbnail_resolution = ctk.StringVar(value=self.settings["thumbnail_resolution"])
        self.duplicate_policy_label = ctk.StringVar(
            value=DUPLICATE_POLICIES[self.settings["duplicate_policy"]]
        )
        self.extension_mode_label = ctk.StringVar(
            value=EXTENSION_MODES[self.settings["extension_mode"]]
        )
        self.apply_selected_only = ctk.BooleanVar(value=False)
        self.preview_mode = ctk.StringVar(value=PREVIEW_TABS[0])
        self.show_protect_mask = ctk.BooleanVar(value=False)
        self.show_ai_mask = ctk.BooleanVar(value=False)
        self.subject_x = ctk.DoubleVar(value=0.0)
        self.subject_y = ctk.DoubleVar(value=0.0)
        self.subject_zoom = ctk.DoubleVar(value=1.0)

        self.build_input_group(left)
        self.build_output_folder_group(left)
        self.build_preset_group(left)
        self.build_ocr_group(left)
        self.build_output_selection_group(left)
        self.build_extension_group(left)
        self.build_ai_group(left)
        self.build_progress_group(left)
        self.build_run_group(left)
        self.build_image_list_panel(right)
        self.build_preview_panel(right)
        self.bind_setting_traces()

    def build_top_control_bar(self) -> None:
        bar = ctk.CTkFrame(self, border_width=1, border_color="#2563eb")
        bar.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=10)
        bar.grid_columnconfigure(3, weight=1)
        self.top_selection_label = ctk.CTkLabel(bar, text="선택 이미지: 0장", anchor="w")
        self.top_selection_label.grid(row=0, column=0, padx=(12, 8), pady=(8, 0), sticky="w")
        self.top_plan_label = ctk.CTkLabel(bar, text="예상 결과: 0개", anchor="w")
        self.top_plan_label.grid(row=0, column=1, padx=8, pady=(8, 0), sticky="w")
        self.top_current_label = ctk.CTkLabel(bar, text="현재 처리: 대기", anchor="w")
        self.top_current_label.grid(row=0, column=2, padx=8, pady=(8, 0), sticky="w")
        self.top_progress_label = ctk.CTkLabel(bar, text="0/0 결과 생성 완료", anchor="w")
        self.top_progress_label.grid(row=0, column=3, padx=8, pady=(8, 0), sticky="w")
        self.top_progress_bar = ctk.CTkProgressBar(bar)
        self.top_progress_bar.grid(row=1, column=0, columnspan=4, padx=12, pady=(4, 8), sticky="ew")
        self.top_progress_bar.set(0)
        buttons = ctk.CTkFrame(bar, fg_color="transparent")
        buttons.grid(row=0, column=4, rowspan=2, padx=(8, 12), pady=8)
        self.top_run_button = ctk.CTkButton(buttons, text="원클릭 변환", command=self.run_pipeline, width=116)
        self.top_run_button.pack(side="left", padx=3)
        self.top_selected_button = ctk.CTkButton(
            buttons, text="선택 항목 변환", command=self.run_selected_pipeline, width=116
        )
        self.top_selected_button.pack(side="left", padx=3)
        self.top_cancel_button = ctk.CTkButton(
            buttons, text="전체 취소", command=self.cancel_work, width=96, state="disabled", fg_color="#7f1d1d"
        )
        self.top_cancel_button.pack(side="left", padx=3)

    def build_input_group(self, parent: Any) -> None:
        frame = self.group(parent, "1. 입력 이미지")
        row = ctk.CTkFrame(frame, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=(0, 8))
        self.add_button = ctk.CTkButton(row, text="이미지 추가", command=self.open_files, height=36)
        self.add_button.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.remove_button = ctk.CTkButton(
            row,
            text="선택 삭제",
            command=self.remove_selected_image,
            height=36,
            fg_color="#4b5563",
        )
        self.remove_button.pack(side="left", fill="x", expand=True, padx=5)
        self.clear_images_button = ctk.CTkButton(
            row,
            text="전체 삭제",
            command=self.clear_images,
            height=36,
            fg_color="#4b5563",
        )
        self.clear_images_button.pack(side="left", fill="x", expand=True, padx=(5, 0))
        self.input_status = ctk.CTkLabel(
            frame,
            text="1~5장의 이미지를 선택하거나 창 위로 드래그앤드롭하세요.",
            wraplength=360,
            justify="left",
            text_color="#cbd5e1",
        )
        self.input_status.pack(fill="x", padx=12, pady=(0, 12))

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
            command=self.on_preset_changed,
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
            values=list(OCR_LANGUAGE_LABELS),
            command=self.on_image_setting_changed,
        )
        self.ocr_menu.pack(fill="x", padx=12, pady=(0, 8))
        self.auto_remove_check = ctk.CTkCheckBox(frame, text="글자 자동 제거", variable=self.auto_remove)
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
        self.clear_button = ctk.CTkButton(
            frame,
            text="선택 이미지 마스크 초기화",
            command=self.clear_boxes,
            height=30,
            fg_color="#374151",
        )
        self.clear_button.pack(fill="x", padx=12, pady=(4, 12))

    def build_output_selection_group(self, parent: Any) -> None:
        frame = self.group(parent, "5. 출력 이미지 선택", highlight=True)
        ctk.CTkLabel(
            frame,
            text="아래 선택은 새 이미지의 기본값이며, 이미지 행을 클릭한 뒤 바꾸면 해당 행에 적용됩니다.",
            wraplength=360,
            justify="left",
            text_color="#bae6fd",
        ).pack(fill="x", padx=12, pady=(0, 8))
        self.square_check = ctk.CTkCheckBox(
            frame,
            text="1:1 클린 커버\n  - 1400x1400 JPG\n  - 글자를 제거한 정사각형 이미지",
            variable=self.output_square_var,
            command=self.on_output_defaults_changed,
        )
        self.square_check.pack(anchor="w", padx=12, pady=(0, 9))
        self.thumb_check = ctk.CTkCheckBox(
            frame,
            text="16:9 유튜브 썸네일\n  - 기본 1920x1080 JPG\n  - 설정에서 1280x720 선택 가능",
            variable=self.output_thumb_var,
            command=self.on_output_defaults_changed,
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
            command=lambda _value: self.on_persistent_setting_changed(),
        )
        self.thumbnail_resolution_menu.pack(side="left")
        self.shorts_check = ctk.CTkCheckBox(
            frame,
            text="9:16 숏츠 이미지\n  - 1080x1920 JPG\n  - YouTube Shorts, Instagram Reels, TikTok용",
            variable=self.output_shorts_var,
            command=self.on_output_defaults_changed,
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
        self.extension_menu = ctk.CTkOptionMenu(
            frame,
            variable=self.extension_mode_label,
            values=list(EXTENSION_MODES.values()),
            command=self.on_extension_changed,
        )
        self.extension_menu.pack(fill="x", padx=12, pady=(0, 8))
        self.extension_desc = ctk.CTkLabel(
            frame,
            text="AI 자연 배경 확장이 기본값입니다. 실패하면 자연 배경 확장, 블러 배경 순서로 전환합니다.",
            wraplength=350,
            justify="left",
            text_color="#cbd5e1",
        )
        self.extension_desc.pack(fill="x", padx=12, pady=(0, 8))
        self.protect_core_check = ctk.CTkCheckBox(
            frame,
            text="원본 핵심 영역 보호",
            variable=self.protect_core,
            command=self.on_persistent_setting_changed,
        )
        self.protect_core_check.pack(anchor="w", padx=12, pady=3)
        self.enhance_check = ctk.CTkCheckBox(frame, text="기본 선명도 보정", variable=self.enhance)
        self.enhance_check.pack(anchor="w", padx=12, pady=3)
        ctk.CTkLabel(frame, text="AI 배경 프롬프트", anchor="w", text_color="#cbd5e1").pack(
            fill="x",
            padx=12,
            pady=(10, 3),
        )
        self.outpaint_prompt = ctk.CTkTextbox(frame, height=68)
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
        self.protect_person_check = ctk.CTkCheckBox(
            frame,
            text="사람 분할 마스크로 얼굴/인물 보호",
            variable=self.protect_person,
        )
        self.protect_person_check.pack(anchor="w", padx=12, pady=(0, 8))
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
            command=lambda _value: self.on_persistent_setting_changed(),
        )
        self.duplicate_policy_menu.pack(fill="x", padx=12, pady=(3, 8))
        self.status = ctk.CTkLabel(frame, text="이미지를 추가해주세요.", wraplength=350, justify="left")
        self.status.pack(fill="x", padx=12, pady=(6, 8))
        self.progress_label = ctk.CTkLabel(frame, text="0/0 결과물 완료 | 성공 0 | 실패 0", text_color="#cbd5e1")
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
            text="원클릭 자동 변환",
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

        # The root-level control bar is the primary action area; keep the
        # summary here but avoid duplicating the action buttons in the scroller.
        self.run_button.pack_forget()
        self.cancel_button.pack_forget()
        self.fixed_busy_widgets = [
            self.add_button,
            self.remove_button,
            self.clear_images_button,
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
            self.clear_button,
            self.square_check,
            self.thumb_check,
            self.thumbnail_resolution_menu,
            self.shorts_check,
            self.select_all_button,
            self.clear_all_button,
            self.extension_menu,
            self.protect_core_check,
            self.enhance_check,
            self.prefer_esrgan_check,
            self.protect_person_check,
            self.ai_status_button,
            self.duplicate_policy_menu,
            self.run_button,
            self.top_run_button,
            self.top_selected_button,
        ]

    def build_image_list_panel(self, parent: Any) -> None:
        frame = ctk.CTkFrame(parent)
        frame.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(frame, text="선택 이미지 목록", font=ctk.CTkFont(size=17, weight="bold")).grid(
            row=0,
            column=0,
            sticky="w",
            padx=12,
            pady=(10, 6),
        )

        button_row = ctk.CTkFrame(frame, fg_color="transparent")
        button_row.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 8))
        buttons = [
            ("모든 이미지 1:1 선택", lambda: self.apply_outputs_to_rows(square=True, thumb=False, shorts=False)),
            ("모든 이미지 16:9 선택", lambda: self.apply_outputs_to_rows(square=False, thumb=True, shorts=False)),
            ("모든 이미지 9:16 선택", lambda: self.apply_outputs_to_rows(square=False, thumb=False, shorts=True)),
            ("모든 이미지 전체 규격 선택", self.select_all_rows_outputs),
            ("모든 이미지 선택 해제", self.clear_all_rows_outputs),
        ]
        for text, command in buttons:
            ctk.CTkButton(button_row, text=text, command=command, height=30).pack(
                side="left",
                padx=(0, 6),
            )
        self.apply_selected_check = ctk.CTkCheckBox(
            button_row,
            text="선택한 행에만 적용",
            variable=self.apply_selected_only,
        )
        self.apply_selected_check.pack(side="left", padx=(6, 0))

        self.table_scroll = ctk.CTkScrollableFrame(frame, height=212)
        self.table_scroll.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 12))
        self.refresh_image_table()

    def build_preview_panel(self, parent: Any) -> None:
        frame = ctk.CTkFrame(parent)
        frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(1, weight=1)

        top = ctk.CTkFrame(frame, fg_color="transparent")
        top.grid(row=0, column=0, sticky="ew", padx=12, pady=10)
        self.preview_tabs = ctk.CTkSegmentedButton(
            top,
            values=PREVIEW_TABS,
            variable=self.preview_mode,
            command=lambda _value: self.draw_preview(),
        )
        self.preview_tabs.pack(side="left", fill="x", expand=True)
        self.regenerate_button = ctk.CTkButton(
            top,
            text="결과 재생성",
            command=self.draw_preview,
            width=110,
            fg_color="#4b5563",
        )
        self.regenerate_button.pack(side="left", padx=(8, 0))

        self.canvas = tk.Canvas(frame, bg="#0f172a", highlightthickness=0, cursor="cross")
        self.canvas.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 8))
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Configure>", lambda _event: self.draw_preview())

        controls = ctk.CTkFrame(frame)
        controls.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 12))
        controls.grid_columnconfigure(1, weight=1)
        controls.grid_columnconfigure(3, weight=1)
        controls.grid_columnconfigure(5, weight=1)

        ctk.CTkLabel(controls, text="좌우 이동").grid(row=0, column=0, padx=(10, 6), pady=8)
        self.x_slider = ctk.CTkSlider(
            controls,
            from_=-0.35,
            to=0.35,
            variable=self.subject_x,
            command=lambda _value: self.on_subject_control_changed(),
        )
        self.x_slider.grid(row=0, column=1, sticky="ew", padx=(0, 10), pady=8)
        ctk.CTkLabel(controls, text="상하 이동").grid(row=0, column=2, padx=(10, 6), pady=8)
        self.y_slider = ctk.CTkSlider(
            controls,
            from_=-0.35,
            to=0.35,
            variable=self.subject_y,
            command=lambda _value: self.on_subject_control_changed(),
        )
        self.y_slider.grid(row=0, column=3, sticky="ew", padx=(0, 10), pady=8)
        ctk.CTkLabel(controls, text="확대축소").grid(row=0, column=4, padx=(10, 6), pady=8)
        self.zoom_slider = ctk.CTkSlider(
            controls,
            from_=0.75,
            to=1.45,
            variable=self.subject_zoom,
            command=lambda _value: self.on_subject_control_changed(),
        )
        self.zoom_slider.grid(row=0, column=5, sticky="ew", padx=(0, 10), pady=8)

        self.protect_mask_check = ctk.CTkCheckBox(
            controls,
            text="원본 보호 영역 확인",
            variable=self.show_protect_mask,
            command=self.draw_preview,
        )
        self.protect_mask_check.grid(row=1, column=0, columnspan=2, sticky="w", padx=10, pady=(0, 10))
        self.ai_mask_check = ctk.CTkCheckBox(
            controls,
            text="AI 확장 마스크 확인",
            variable=self.show_ai_mask,
            command=self.draw_preview,
        )
        self.ai_mask_check.grid(row=1, column=2, columnspan=2, sticky="w", padx=10, pady=(0, 10))
        self.preview_hint = ctk.CTkLabel(
            controls,
            text="수동 글자 제거 마스크는 원본 또는 글자 제거 결과 탭에서 드래그하세요.",
            text_color="#cbd5e1",
        )
        self.preview_hint.grid(row=1, column=4, columnspan=2, sticky="e", padx=10, pady=(0, 10))

    def bind_setting_traces(self) -> None:
        for var in (
            self.thumbnail_resolution,
            self.duplicate_policy_label,
            self.protect_core,
        ):
            var.trace_add("write", lambda *_args: self.on_persistent_setting_changed())
        self.output_path_var.trace_add("write", lambda *_args: self.update_summary())

    def setup_drag_drop(self) -> None:
        if DND_FILES is None or not getattr(self, "_covermorph_dnd_ready", False):
            self.input_status.configure(text="1~5장의 이미지를 선택하세요. 드래그앤드롭은 tkinterdnd2 설치 시 활성화됩니다.")
            return
        try:
            self.drop_target_register(DND_FILES)
            self.dnd_bind("<<Drop>>", self.on_drop_files)
            self.input_status.configure(text="1~5장의 이미지를 선택하거나 창 위로 드래그앤드롭하세요.")
        except tk.TclError as exc:
            write_exception(self.root_dir, "Drag and drop setup", exc)
            self.input_status.configure(text="1~5장의 이미지를 선택하세요. 드래그앤드롭 초기화에 실패했습니다.")

    def on_drop_files(self, event: Any) -> None:
        paths = [Path(value) for value in self.tk.splitlist(event.data)]
        self.add_image_paths(paths)

    def extension_key(self) -> str:
        reverse = {label: key for key, label in EXTENSION_MODES.items()}
        return reverse.get(self.extension_mode_label.get(), "ai_natural")

    def duplicate_policy_key(self) -> str:
        reverse = {label: key for key, label in DUPLICATE_POLICIES.items()}
        return reverse.get(self.duplicate_policy_label.get(), "new_number")

    def settings_payload(self, *, include_output: bool = True) -> dict[str, Any]:
        output_directory = self.output_path_var.get().strip() if include_output else self.settings["output_directory"]
        return {
            "output_directory": output_directory,
            "output_square": self.output_square_var.get(),
            "output_thumbnail": self.output_thumb_var.get(),
            "output_shorts": self.output_shorts_var.get(),
            "thumbnail_resolution": self.thumbnail_resolution.get(),
            "extension_mode": self.extension_key(),
            "protect_core": self.protect_core.get(),
            "duplicate_policy": self.duplicate_policy_key(),
            "last_preset": self.preset_name.get(),
            "last_ocr_language": self.ocr_lang.get(),
        }

    def save_current_settings(self, *, include_output: bool = True) -> None:
        self.settings = self.settings_payload(include_output=include_output)
        try:
            save_settings(self.root_dir, self.settings)
        except OSError as exc:
            write_exception(self.root_dir, "Save settings", exc)

    def on_persistent_setting_changed(self) -> None:
        self.save_current_settings(include_output=False)
        self.update_summary()
        self.draw_preview()

    def on_preset_changed(self, *_args: Any) -> None:
        preset_value = self.preset_name.get()
        if preset_value in PRESETS:
            self.preset_desc.configure(text=PRESETS[preset_value].description)
        self.on_image_setting_changed()

    def on_extension_changed(self, *_args: Any) -> None:
        key = self.extension_key()
        if key == "ai_natural":
            desc = "AI 자연 배경 확장: SDXL로 부족한 배경만 생성하고 원본 핵심 영역을 복원합니다."
        elif key == "smart_crop":
            desc = "스마트 크롭: 원본을 확대해 화면을 채웁니다. 가장자리는 일부 잘릴 수 있습니다."
        elif key == "natural":
            desc = "자연 배경 확장: 로컬 가장자리 미러링과 블렌딩으로 빈틈 없이 확장합니다."
        elif key == "blur":
            desc = "블러 배경: 사용자가 직접 선택한 경우에만 기존 방식으로 생성합니다."
        else:
            desc = "원본 전체 맞춤: 원본을 모두 표시하므로 여백이 생길 수 있습니다."
        self.extension_desc.configure(text=desc)
        self.on_image_setting_changed()

    def on_image_setting_changed(self, *_args: Any) -> None:
        if self._loading_selection:
            return
        state = self.current_state()
        if state is not None:
            state.job.preset_name = self.preset_name.get()
            state.job.ocr_languages = self.lang_codes()
            state.job.extension_mode = self.extension_key()
            state.job.outpaint_prompt = self.outpaint_prompt.get("1.0", "end").strip()
            self.refresh_image_table()
        self.save_current_settings(include_output=False)
        self.update_summary()
        self.draw_preview()

    def on_output_defaults_changed(self) -> None:
        if self._loading_selection:
            return
        state = self.current_state()
        if state is not None:
            state.job.out_square = self.output_square_var.get()
            state.job.out_thumb = self.output_thumb_var.get()
            state.job.out_shorts = self.output_shorts_var.get()
            self.refresh_image_table()
        self.save_current_settings(include_output=False)
        self.update_summary()

    def on_subject_control_changed(self) -> None:
        if self._loading_selection:
            return
        state = self.current_state()
        if state is None:
            return
        state.job.subject_offset_x = float(self.subject_x.get())
        state.job.subject_offset_y = float(self.subject_y.get())
        state.job.subject_scale = float(self.subject_zoom.get())
        self.draw_preview()

    def apply_saved_output_directory_notice(self) -> None:
        path_text = self.output_path_var.get().strip()
        if not path_text:
            return
        path = Path(path_text)
        self.output_dir = path
        if not path.exists() or not path.is_dir():
            self.status.configure(text="저장된 출력 폴더를 사용할 수 없습니다. 새 출력 폴더를 선택해주세요.")

    def lang_codes(self) -> tuple[str, ...]:
        return OCR_LANGUAGE_LABELS.get(self.ocr_lang.get(), ("en",))

    def current_state(self) -> ImageRowState | None:
        if self.selected_index is None:
            return None
        if self.selected_index < 0 or self.selected_index >= len(self.image_states):
            return None
        return self.image_states[self.selected_index]

    def jobs(self) -> list[ImageJob]:
        return [state.job for state in self.image_states]

    def open_files(self) -> None:
        paths = filedialog.askopenfilenames(filetypes=[("Images", "*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff")])
        if paths:
            self.add_image_paths([Path(path) for path in paths])

    def add_image_paths(self, paths: list[Path]) -> None:
        candidates = [path for path in paths if path.suffix.lower() in IMAGE_EXTENSIONS]
        if not candidates:
            messagebox.showinfo("안내", "지원하는 이미지 파일을 선택해주세요.")
            return

        existing = {state.job.source.resolve() for state in self.image_states if state.job.source.exists()}
        added = 0
        errors: list[str] = []
        for path in candidates:
            if len(self.image_states) >= MAX_IMAGES:
                break
            try:
                resolved = path.resolve()
            except OSError:
                resolved = path
            if resolved in existing:
                continue
            try:
                with Image.open(path) as opened:
                    original = opened.convert("RGB")
            except (OSError, UnidentifiedImageError) as exc:
                write_exception(self.root_dir, f"Open image {path.name}", exc)
                errors.append(f"{path.name}: {exc}")
                continue

            thumb = ImageOps.contain(original.copy(), (58, 58), Image.Resampling.LANCZOS)
            thumb_bg = Image.new("RGB", (58, 58), (15, 23, 42))
            thumb_bg.paste(thumb, ((58 - thumb.width) // 2, (58 - thumb.height) // 2))
            photo = ImageTk.PhotoImage(thumb_bg)
            job = ImageJob(
                source=path,
                out_square=self.output_square_var.get(),
                out_thumb=self.output_thumb_var.get(),
                out_shorts=self.output_shorts_var.get(),
                preset_name=self.preset_name.get(),
                ocr_languages=self.lang_codes(),
                extension_mode=self.extension_key(),
                outpaint_prompt=self.outpaint_prompt.get("1.0", "end").strip(),
            )
            self.image_states.append(ImageRowState(job=job, original=original, thumbnail_photo=photo))
            existing.add(resolved)
            added += 1

        if len(self.image_states) >= MAX_IMAGES:
            self.input_status.configure(text="이미지는 최대 5장까지 선택할 수 있습니다.")
        else:
            self.input_status.configure(text=f"{len(self.image_states)}장 선택됨")

        if errors:
            messagebox.showwarning("이미지 오류", "일부 이미지를 열 수 없습니다.\n" + "\n".join(errors[:5]))

        if added and self.selected_index is None:
            self.selected_index = 0
        if added and not self.output_path_var.get().strip():
            default_dir = default_output_dir_for_source(self.image_states[0].job.source)
            self.output_path_var.set(str(default_dir))
            self.output_dir = default_dir

        self.sync_controls_from_selected()
        self.refresh_image_table()
        self.update_summary()
        self.save_current_settings(include_output=False)
        self.draw_preview()

    def remove_selected_image(self) -> None:
        if self.selected_index is None:
            return
        del self.image_states[self.selected_index]
        if not self.image_states:
            self.selected_index = None
        else:
            self.selected_index = min(self.selected_index, len(self.image_states) - 1)
        self.input_status.configure(text=f"{len(self.image_states)}장 선택됨")
        self.sync_controls_from_selected()
        self.refresh_image_table()
        self.update_summary()
        self.draw_preview()

    def clear_images(self) -> None:
        self.image_states.clear()
        self.selected_index = None
        self.input_status.configure(text="1~5장의 이미지를 선택하거나 창 위로 드래그앤드롭하세요.")
        self.refresh_image_table()
        self.update_summary()
        self.draw_preview()

    def select_image(self, index: int) -> None:
        if index < 0 or index >= len(self.image_states):
            return
        self.selected_index = index
        self.sync_controls_from_selected()
        self.refresh_image_table()
        self.draw_preview()

    def sync_controls_from_selected(self) -> None:
        state = self.current_state()
        self._loading_selection = True
        try:
            if state is None:
                return
            job = state.job
            self.output_square_var.set(job.out_square)
            self.output_thumb_var.set(job.out_thumb)
            self.output_shorts_var.set(job.out_shorts)
            self.preset_name.set(job.preset_name if job.preset_name in PRESETS else "OldPopLounge")
            self.ocr_lang.set(self.label_for_langs(job.ocr_languages))
            self.extension_mode_label.set(EXTENSION_MODES.get(job.extension_mode, EXTENSION_MODES["ai_natural"]))
            self.subject_x.set(job.subject_offset_x)
            self.subject_y.set(job.subject_offset_y)
            self.subject_zoom.set(job.subject_scale)
            self.outpaint_prompt.delete("1.0", "end")
            self.outpaint_prompt.insert("1.0", job.outpaint_prompt or DEFAULT_OUTPAINT_PROMPT)
            if self.preset_name.get() in PRESETS:
                self.preset_desc.configure(text=PRESETS[self.preset_name.get()].description)
            self.on_extension_changed()
        finally:
            self._loading_selection = False

    def label_for_langs(self, langs: tuple[str, ...]) -> str:
        for label, codes in OCR_LANGUAGE_LABELS.items():
            if tuple(codes) == tuple(langs):
                return label
        return "영어"

    def refresh_image_table(self) -> None:
        for child in self.table_scroll.winfo_children():
            child.destroy()

        headers = ["번호", "미리보기", "원본 파일명", "1:1", "16:9", "9:16", "확장 방식", "상태"]
        widths = [46, 76, 230, 54, 54, 54, 170, 86, 92, 92, 86]
        headers.extend(["개별 변환", "결과 저장", "폴더 열기"])
        for col, (header, width) in enumerate(zip(headers, widths, strict=True)):
            label = ctk.CTkLabel(self.table_scroll, text=header, width=width, anchor="w", text_color="#cbd5e1")
            label.grid(row=0, column=col, sticky="ew", padx=3, pady=(2, 6))

        for row_index, state in enumerate(self.image_states, start=1):
            color = "#172554" if row_index - 1 == self.selected_index else "#111827"
            state.row_frame = ctk.CTkFrame(self.table_scroll, fg_color=color, corner_radius=6)
            state.row_frame.grid(row=row_index, column=0, columnspan=len(headers), sticky="ew", padx=2, pady=3)
            for col, width in enumerate(widths):
                state.row_frame.grid_columnconfigure(col, minsize=width)

            state.square_var = ctk.BooleanVar(value=state.job.out_square)
            state.thumb_var = ctk.BooleanVar(value=state.job.out_thumb)
            state.shorts_var = ctk.BooleanVar(value=state.job.out_shorts)
            state.extension_var = ctk.StringVar(
                value=EXTENSION_MODES.get(state.job.extension_mode, EXTENSION_MODES["ai_natural"])
            )
            state.row_widgets = []

            widgets: list[Any] = [
                ctk.CTkLabel(state.row_frame, text=str(row_index), width=widths[0]),
                ctk.CTkLabel(state.row_frame, image=state.thumbnail_photo, text="", width=widths[1]),
                ctk.CTkLabel(state.row_frame, text=state.job.source.name, width=widths[2], anchor="w"),
                ctk.CTkCheckBox(
                    state.row_frame,
                    text="",
                    width=widths[3],
                    variable=state.square_var,
                    command=lambda item=state: self.on_row_output_changed(item),
                ),
                ctk.CTkCheckBox(
                    state.row_frame,
                    text="",
                    width=widths[4],
                    variable=state.thumb_var,
                    command=lambda item=state: self.on_row_output_changed(item),
                ),
                ctk.CTkCheckBox(
                    state.row_frame,
                    text="",
                    width=widths[5],
                    variable=state.shorts_var,
                    command=lambda item=state: self.on_row_output_changed(item),
                ),
                ctk.CTkOptionMenu(
                    state.row_frame,
                    variable=state.extension_var,
                    values=list(EXTENSION_MODES.values()),
                    width=widths[6],
                    command=lambda _value, item=state: self.on_row_extension_changed(item),
                ),
            ]
            widgets.extend(
                [
                    ctk.CTkButton(
                        state.row_frame,
                        text="개별 변환",
                        width=widths[8],
                        height=28,
                        command=lambda item=state: self.run_single_image(item),
                    ),
                    ctk.CTkButton(
                        state.row_frame,
                        text="결과 저장",
                        width=widths[9],
                        height=28,
                        command=lambda item=state: self.export_row_results(item),
                    ),
                    ctk.CTkButton(
                        state.row_frame,
                        text="폴더 열기",
                        width=widths[10],
                        height=28,
                        command=lambda item=state: self.open_row_folder(item),
                    ),
                ]
            )
            state.status_label = ctk.CTkLabel(state.row_frame, text=state.job.status, width=widths[7], anchor="w")
            widgets.insert(7, state.status_label)

            selectable_cols = {0, 1, 2, 7}
            for col, widget in enumerate(widgets):
                widget.grid(row=0, column=col, sticky="ew", padx=3, pady=6)
                if col in selectable_cols:
                    widget.bind("<Button-1>", lambda _event, i=row_index - 1: self.select_image(i))
            state.row_frame.bind("<Button-1>", lambda _event, i=row_index - 1: self.select_image(i))
            state.row_widgets = widgets
            if self.busy_task is not None:
                for widget in state.row_widgets:
                    try:
                        widget.configure(state="disabled")
                    except (tk.TclError, AttributeError):
                        pass

    def on_row_output_changed(self, state: ImageRowState) -> None:
        state.job.out_square = bool(state.square_var and state.square_var.get())
        state.job.out_thumb = bool(state.thumb_var and state.thumb_var.get())
        state.job.out_shorts = bool(state.shorts_var and state.shorts_var.get())
        if self.current_state() is state:
            self._loading_selection = True
            try:
                self.output_square_var.set(state.job.out_square)
                self.output_thumb_var.set(state.job.out_thumb)
                self.output_shorts_var.set(state.job.out_shorts)
            finally:
                self._loading_selection = False
        self.save_current_settings(include_output=False)
        self.update_summary()

    def on_row_extension_changed(self, state: ImageRowState) -> None:
        if state.extension_var is None:
            return
        reverse = {label: key for key, label in EXTENSION_MODES.items()}
        state.job.extension_mode = reverse.get(state.extension_var.get(), "ai_natural")
        if self.current_state() is state:
            self._loading_selection = True
            try:
                self.extension_mode_label.set(EXTENSION_MODES[state.job.extension_mode])
            finally:
                self._loading_selection = False
            self.on_extension_changed()
        self.save_current_settings(include_output=False)
        self.update_summary()
        self.draw_preview()

    def target_rows(self) -> list[ImageRowState]:
        state = self.current_state()
        if self.apply_selected_only.get() and state is not None:
            return [state]
        return list(self.image_states)

    def apply_outputs_to_rows(
        self,
        *,
        square: bool | None = None,
        thumb: bool | None = None,
        shorts: bool | None = None,
    ) -> None:
        for state in self.target_rows():
            if square is not None:
                state.job.out_square = square
            if thumb is not None:
                state.job.out_thumb = thumb
            if shorts is not None:
                state.job.out_shorts = shorts
        self.sync_controls_from_selected()
        self.refresh_image_table()
        self.update_summary()
        self.save_current_settings(include_output=False)

    def select_all_rows_outputs(self) -> None:
        self.apply_outputs_to_rows(square=True, thumb=True, shorts=True)

    def clear_all_rows_outputs(self) -> None:
        self.apply_outputs_to_rows(square=False, thumb=False, shorts=False)

    def select_all_outputs(self) -> None:
        self.output_square_var.set(True)
        self.output_thumb_var.set(True)
        self.output_shorts_var.set(True)
        self.select_all_rows_outputs()
        self.save_current_settings()

    def clear_all_outputs(self) -> None:
        self.output_square_var.set(False)
        self.output_thumb_var.set(False)
        self.output_shorts_var.set(False)
        self.clear_all_rows_outputs()
        self.save_current_settings()

    def default_output_for_current_files(self) -> Path | None:
        if not self.image_states:
            return None
        return default_output_dir_for_source(self.image_states[0].job.source)

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
            self.status.configure(text="이미지를 추가하면 원본 폴더 아래 CoverMorph_Output을 기본값으로 사용합니다.")
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
        if not path_text and self.image_states:
            path_text = str(default_output_dir_for_source(self.image_states[0].job.source))
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
        state = self.current_state()
        if state is None:
            return
        img = state.original.copy()
        langs = self.lang_codes()
        job_index = self.selected_index
        state.job.status = "OCR 처리 중"
        self.refresh_image_table()
        self.status.configure(text="EasyOCR 준비 중입니다. 첫 실행이면 모델 다운로드가 진행될 수 있습니다.")

        def worker() -> None:
            boxes = detect_text_boxes_multilang(
                img,
                langs,
                status_callback=lambda message: self.worker_queue.put({"type": "status", "message": message}),
            )
            self.worker_queue.put({"type": "ocr_done", "index": job_index, "boxes": boxes})

        self.start_worker("OCR", worker)

    def remove_with_backend(self, img: Image.Image, boxes: list[Rect], prefer_lama: bool) -> tuple[Image.Image, str]:
        if not boxes:
            return img.copy(), "No text mask"
        if prefer_lama and self.ai.lama_available():
            try:
                return self.ai.inpaint(img, mask_pil_from_boxes(img.size, boxes))
            except Exception as exc:
                write_exception(self.root_dir, "LaMa preview fallback", exc)
        return inpaint_text_opencv(img, boxes), "OpenCV NS/Telea (quality selected)"

    def preview_remove(self) -> None:
        state = self.current_state()
        if state is None:
            return
        boxes = list(state.job.ocr_boxes) + list(state.job.manual_boxes)
        if not boxes:
            self.status.configure(text="OCR 탐지 결과나 수동 마스크가 없습니다.")
            return
        img = state.original.copy()
        prefer_lama = self.prefer_lama.get()
        job_index = self.selected_index
        state.job.status = "글자 제거 중"
        self.refresh_image_table()
        self.status.configure(text="미리보기 글자 제거 중...")

        def worker() -> None:
            out, engine = self.remove_with_backend(img, boxes, prefer_lama)
            self.worker_queue.put({"type": "preview_done", "index": job_index, "image": out, "engine": engine})

        self.start_worker("PreviewRemove", worker)

    def clear_boxes(self) -> None:
        state = self.current_state()
        if state is None:
            return
        state.job.manual_boxes = ()
        state.job.ocr_boxes = ()
        state.clean_preview = None
        state.job.status = "대기"
        self.refresh_image_table()
        self.draw_preview()

    def on_press(self, event: tk.Event) -> None:
        if self.current_state() is not None:
            self.drag_start = (event.x, event.y)

    def on_release(self, event: tk.Event) -> None:
        state = self.current_state()
        if state is None or self.drag_start is None:
            return
        if self.preview_mode.get() not in {"원본", "글자 제거 결과"}:
            self.drag_start = None
            self.status.configure(text="수동 마스크는 원본 또는 글자 제거 결과 탭에서 지정해주세요.")
            return
        x1, y1 = self.drag_start
        x2, y2 = event.x, event.y
        offset_x, offset_y = self.preview_offset
        scale = self.preview_scale
        img_x1 = int((min(x1, x2) - offset_x) / scale)
        img_y1 = int((min(y1, y2) - offset_y) / scale)
        img_x2 = int((max(x1, x2) - offset_x) / scale)
        img_y2 = int((max(y1, y2) - offset_y) / scale)
        img_x1 = max(0, min(img_x1, state.original.width))
        img_x2 = max(0, min(img_x2, state.original.width))
        img_y1 = max(0, min(img_y1, state.original.height))
        img_y2 = max(0, min(img_y2, state.original.height))
        if img_x2 - img_x1 > 5 and img_y2 - img_y1 > 5:
            state.job.manual_boxes = (*state.job.manual_boxes, (img_x1, img_y1, img_x2, img_y2))
            state.clean_preview = None
        self.drag_start = None
        self.draw_preview()

    def draw_preview(self) -> None:
        self.canvas.delete("all")
        state = self.current_state()
        if state is None:
            self.canvas.create_text(
                max(120, self.canvas.winfo_width() // 2),
                max(100, self.canvas.winfo_height() // 2),
                text="이미지를 추가하면 미리보기가 표시됩니다.",
                fill="#cbd5e1",
                font=("Arial", 16),
            )
            return
        try:
            preview = self.preview_image_for_state(state)
        except (OSError, ValueError) as exc:
            write_exception(self.root_dir, "Preview render", exc)
            self.status.configure(text=f"미리보기 생성 실패: {exc}")
            return

        canvas_width = max(100, self.canvas.winfo_width())
        canvas_height = max(100, self.canvas.winfo_height())
        scale = min((canvas_width - 32) / preview.width, (canvas_height - 32) / preview.height)
        preview_width = max(1, int(preview.width * scale))
        preview_height = max(1, int(preview.height * scale))
        preview_resized = preview.resize((preview_width, preview_height), Image.Resampling.LANCZOS)
        self.preview_tk = ImageTk.PhotoImage(preview_resized)
        offset_x = (canvas_width - preview_width) // 2
        offset_y = (canvas_height - preview_height) // 2
        self.preview_scale = scale
        self.preview_offset = (offset_x, offset_y)
        self.canvas.create_image(offset_x, offset_y, image=self.preview_tk, anchor="nw")

        if self.preview_mode.get() in {"원본", "글자 제거 결과"}:
            self.draw_text_boxes(state, offset_x, offset_y, scale)

    def draw_text_boxes(self, state: ImageRowState, offset_x: int, offset_y: int, scale: float) -> None:
        for left, top, right, bottom in state.job.ocr_boxes:
            self.canvas.create_rectangle(
                offset_x + left * scale,
                offset_y + top * scale,
                offset_x + right * scale,
                offset_y + bottom * scale,
                outline="#facc15",
                width=2,
            )
        for left, top, right, bottom in state.job.manual_boxes:
            self.canvas.create_rectangle(
                offset_x + left * scale,
                offset_y + top * scale,
                offset_x + right * scale,
                offset_y + bottom * scale,
                outline="#fb7185",
                width=2,
            )

    def preview_image_for_state(self, state: ImageRowState) -> Image.Image:
        mode = self.preview_mode.get()
        base = state.clean_preview.copy() if state.clean_preview is not None else state.original.copy()
        if mode == "원본":
            return state.original.copy()
        if mode == "OCR 마스크":
            return self.overlay_mask(
                state.original,
                mask_pil_from_boxes(state.original.size, state.job.ocr_boxes),
                "#facc15",
            )
        if mode == "수동 마스크":
            return self.overlay_mask(
                state.original,
                mask_pil_from_boxes(state.original.size, state.job.manual_boxes),
                "#fb7185",
            )
        if mode == "글자 제거 결과":
            return base
        if mode == "최종 결과":
            if state.job.out_thumb:
                return self.format_preview(state, base, "thumbnail", thumbnail_size(self.thumbnail_resolution.get()))
            if state.job.out_shorts:
                return self.format_preview(state, base, "shorts", (1080, 1920))
            return make_square(base)
        if mode == "1:1 미리보기":
            return make_square(base)
        if mode == "16:9 미리보기":
            return self.format_preview(state, base, "thumbnail", thumbnail_size(self.thumbnail_resolution.get()))
        if mode == "9:16 미리보기":
            return self.format_preview(state, base, "shorts", (1080, 1920))
        return base

    def format_preview(
        self,
        state: ImageRowState,
        img: Image.Image,
        kind: str,
        size: tuple[int, int],
    ) -> Image.Image:
        preset = PRESETS.get(state.job.preset_name, PRESETS["OldPopLounge"])
        anchor = preset.person_anchor_16x9 if kind == "thumbnail" else preset.person_anchor_9x16
        mode = state.job.extension_mode
        out, _engine = render_full_frame_format(
            img,
            size,
            preset,
            kind,
            mode=mode,
            anchor=anchor,
            offset_x=state.job.subject_offset_x,
            offset_y=state.job.subject_offset_y,
            subject_scale=state.job.subject_scale,
        )
        # Keep the mask overlays below while sharing the exact geometry with saving.
        if mode not in {"ai_natural", "natural", "smart_crop", "fit", "blur"}:
            out = img.copy()
        if self.show_protect_mask.get() or self.show_ai_mask.get():
            _canvas, ai_mask, protect = build_full_frame_outpaint_canvas(
                img,
                size,
                preset,
                kind,
                person_mask=None,
                anchor=anchor,
                offset_x=state.job.subject_offset_x,
                offset_y=state.job.subject_offset_y,
                subject_scale=state.job.subject_scale,
                protect_core=self.protect_core.get(),
            )
            if self.show_ai_mask.get():
                out = self.overlay_mask(out, ai_mask, "#38bdf8")
            if self.show_protect_mask.get():
                out = self.overlay_mask(out, protect.getchannel("A"), "#22c55e")
        return out

    def overlay_mask(self, img: Image.Image, mask: Image.Image, color: str) -> Image.Image:
        alpha = mask.convert("L").point(lambda value: int(value * 0.38))
        overlay = Image.new("RGBA", img.size, (*ImageColor.getrgb(color), 0))
        overlay.putalpha(alpha)
        out = img.convert("RGBA")
        out.alpha_composite(overlay)
        return out.convert("RGB")

    def set_busy(self, busy: bool, task: str | None = None) -> None:
        self.busy_task = task if busy else None
        state = "disabled" if busy else "normal"
        for widget in self.fixed_busy_widgets:
            try:
                widget.configure(state=state)
            except (tk.TclError, AttributeError):
                pass
        for row_state in self.image_states:
            for widget in row_state.row_widgets:
                try:
                    widget.configure(state=state)
                except (tk.TclError, AttributeError):
                    pass
        self.cancel_button.configure(state="normal" if busy else "disabled")
        self.top_cancel_button.configure(state="normal" if busy else "disabled")

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
                self.worker_queue.put({"type": "worker_error", "error": str(exc)})
            finally:
                self.worker_queue.put({"type": "worker_finished"})

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
        if worker_alive or status_alive:
            self.after(100, self.poll_worker_queue)

    def handle_worker_event(self, event: dict[str, Any]) -> None:
        event_type = event.get("type")
        if event_type == "status":
            self.status.configure(text=event.get("message", "작업 중..."))
        elif event_type == "ai_status_done":
            self.apply_ai_status(event["status"])
        elif event_type == "ocr_done":
            index = event.get("index")
            if isinstance(index, int) and 0 <= index < len(self.image_states):
                state = self.image_states[index]
                state.job.ocr_boxes = tuple(event["boxes"])
                state.job.status = "대기"
                self.status.configure(text=f"{state.job.source.name}\n글자 영역 {len(state.job.ocr_boxes)}개 탐지")
                self.refresh_image_table()
                self.draw_preview()
        elif event_type == "preview_done":
            index = event.get("index")
            if isinstance(index, int) and 0 <= index < len(self.image_states):
                state = self.image_states[index]
                state.clean_preview = event["image"]
                state.job.status = "대기"
                self.preview_mode.set("글자 제거 결과")
                self.status.configure(text=f"미리보기 제거 완료: {event['engine']}")
                self.refresh_image_table()
                self.draw_preview()
        elif event_type == "pipeline_progress":
            self.handle_pipeline_progress(event["payload"])
        elif event_type == "pipeline_done":
            self.handle_pipeline_done(event["results"], event["output_dir"], event.get("row_indices"))
        elif event_type == "worker_error":
            self.status.configure(text=f"오류: {event['error']}\n수동 마스크나 로그를 확인해주세요.")
            messagebox.showerror("CoverMorph 오류", event["error"])
        elif event_type == "worker_finished":
            self.worker_thread = None
            self.set_busy(False)

    def handle_pipeline_progress(self, payload: dict[str, Any]) -> None:
        event_type = payload.get("type")
        if payload.get("filename"):
            self.top_current_label.configure(text=f"현재 처리: {payload['filename']}")
        if event_type == "status":
            self.status.configure(text=payload.get("message", "작업 중..."))
            return
        index = payload.get("index")
        state = self.image_states[index - 1] if isinstance(index, int) and 1 <= index <= len(self.image_states) else None
        if event_type == "file_start":
            if state is not None:
                state.job.status = "대기"
                self.refresh_image_table()
            self.status.configure(text=f"처리 중 {payload['index']}/{payload['total']}\n{payload['filename']}")
        elif event_type == "file_stage":
            if state is not None:
                state.job.status = payload.get("status", "처리 중")
                self.refresh_image_table()
            self.status.configure(text=f"{payload.get('filename', '')}\n{payload.get('status', '처리 중')}")
        elif event_type == "output_progress":
            total_outputs = int(payload.get("total_outputs") or 0)
            completed_outputs = int(payload.get("completed_outputs") or 0)
            self.progress_bar.set(payload.get("ratio", 0.0))
            self.top_progress_bar.set(payload.get("ratio", 0.0))
            self.top_progress_label.configure(text=f"{completed_outputs}/{total_outputs} 결과 생성 완료")
            self.progress_label.configure(
                text=f"{completed_outputs}/{total_outputs} 결과물 완료 | 성공 집계 중 | 실패 집계 중"
            )
        elif event_type == "file_done":
            if state is not None:
                state.job.status = self.status_label_for_result(payload.get("status", "failed"))
                self.refresh_image_table()
            total_outputs = int(payload.get("total_outputs") or 0)
            completed_outputs = int(payload.get("completed_outputs") or 0)
            self.progress_bar.set(completed_outputs / max(1, total_outputs))
            self.top_progress_bar.set(completed_outputs / max(1, total_outputs))
            self.top_progress_label.configure(text=f"{completed_outputs}/{total_outputs} 결과 생성 완료")
            self.progress_label.configure(
                text=(
                    f"{completed_outputs}/{total_outputs} 결과물 완료 | "
                    f"성공 {payload.get('success_outputs', 0)} | 실패 {payload.get('failed_outputs', 0)}"
                )
            )
        elif event_type == "cancelled":
            for row_state in self.image_states:
                if row_state.job.status not in {"저장 완료", "일부 완료", "실패", "건너뜀"}:
                    row_state.job.status = "취소됨"
            self.refresh_image_table()
            self.status.configure(text=payload.get("message", "작업이 취소되었습니다."))
        elif event_type == "batch_done":
            total_outputs = int(payload.get("total_outputs") or 0)
            completed_outputs = int(payload.get("completed_outputs") or 0)
            self.progress_bar.set(completed_outputs / max(1, total_outputs))

    def status_label_for_result(self, status: str) -> str:
        if status == "success":
            return "저장 완료"
        if status == "partial":
            return "일부 완료"
        if status == "skipped":
            return "건너뜀"
        return "실패"

    def handle_pipeline_done(
        self, results: list[Any], output_dir: Path, row_indices: list[int] | None = None
    ) -> None:
        if row_indices is None:
            row_indices = list(range(len(results)))
        for row_index, result in zip(row_indices, results, strict=False):
            if 0 <= row_index < len(self.image_states):
                self.image_states[row_index].result = result
        success_sources = sum(1 for result in results if result.success_outputs > 0)
        failed_sources = sum(1 for result in results if result.success_outputs == 0 and result.status != "skipped")
        generated_count = sum(result.success_outputs for result in results)
        failed_outputs = sum(result.failed_outputs for result in results)
        skipped_outputs = sum(result.skipped_outputs for result in results)
        generated_keys: set[str] = set()
        failed_names: list[str] = []
        fallback_lines: list[str] = []
        for result in results:
            generated_keys.update(result.metadata.get("output_files", {}))
            failed_names.extend(result.failed_file_names)
            for fallback in result.metadata.get("extension_fallbacks", []):
                fallback_lines.append(f"{result.source.name}: {fallback.get('message', fallback.get('category', 'fallback'))}")

        formats = [OUTPUT_KEYS[key] for key in ("square_1x1", "thumbnail_16x9", "shorts_9x16") if key in generated_keys]
        message = (
            f"저장된 출력 폴더:\n{output_dir}\n\n"
            f"생성된 이미지 개수: {generated_count}개\n"
            f"성공한 원본 개수: {success_sources}개\n"
            f"실패한 원본 개수: {failed_sources}개\n"
            f"건너뛴 결과물: {skipped_outputs}개\n"
            f"저장 실패 결과물: {failed_outputs}개\n"
            f"생성된 규격: {', '.join(formats) if formats else '없음'}"
        )
        if failed_names:
            message += "\n\n실패 파일:\n" + "\n".join(failed_names[:12])
            if len(failed_names) > 12:
                message += f"\n...외 {len(failed_names) - 12}개"
        if fallback_lines:
            message += "\n\n자동 전환:\n" + "\n".join(fallback_lines[:8])
            if len(fallback_lines) > 8:
                message += f"\n...외 {len(fallback_lines) - 8}건"
        self.status.configure(text=f"완료\n{output_dir}\n생성 {generated_count}개")
        self.show_completion_dialog(message, output_dir)
        self.update_summary()

    def show_completion_dialog(self, message: str, output_dir: Path) -> None:
        dialog = ctk.CTkToplevel(self)
        dialog.title("CoverMorph 완료")
        dialog.geometry("540x400")
        dialog.transient(self)
        dialog.grab_set()
        ctk.CTkLabel(dialog, text="변환 완료", font=ctk.CTkFont(size=18, weight="bold")).pack(
            anchor="w",
            padx=18,
            pady=(18, 8),
        )
        ctk.CTkLabel(dialog, text=message, justify="left", wraplength=490).pack(
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

    def apply_ai_status(self, status: dict[str, bool]) -> None:
        lines = [
            f"LaMa: {'사용 가능' if status.get('lama') else '미설치'}",
            f"Real-ESRGAN: {'사용 가능' if status.get('realesrgan') else '미설치'}",
            f"SDXL: {'사용 가능' if status.get('sdxl') else '미설치'}",
            f"CUDA: {'사용 가능' if status.get('cuda') else '없음'}",
            f"인물 분리: {'사용 가능' if status.get('rembg') else '미설치'}",
        ]
        self.ai_status.configure(text="\n".join(lines))

    def update_summary(self) -> None:
        if not hasattr(self, "summary_label"):
            return
        jobs = self.jobs()
        counts = output_count_by_kind(jobs)
        total_outputs = count_selected_outputs_for_jobs(jobs)
        labels = []
        if counts["thumbnail_16x9"]:
            labels.append(f"16:9 썸네일 {counts['thumbnail_16x9']}개")
        if counts["shorts_9x16"]:
            labels.append(f"9:16 숏츠 {counts['shorts_9x16']}개")
        if counts["square_1x1"]:
            labels.append(f"1:1 클린 커버 {counts['square_1x1']}개")
        output_dir = self.output_path_var.get().strip() or "이미지 선택 후 자동 설정"
        planned = "\n ".join(labels) if labels else "선택한 출력 없음"
        self.top_selection_label.configure(text=f"선택 이미지: {len(jobs)}장")
        self.top_plan_label.configure(text=f"예상 결과: {total_outputs}개")
        self.summary_label.configure(
            text=(
                f"출력 폴더:\n{output_dir}\n\n"
                f"생성 예정:\n {planned}\n\n"
                f"선택 이미지: {len(jobs)}장\n"
                f"총 생성 예정: {total_outputs}개"
            )
        )

    def pipeline_options(self, output_dir: Path) -> PipelineOptions:
        return PipelineOptions(
            output_dir=output_dir,
            preset_name=self.preset_name.get(),
            ocr_languages=self.lang_codes(),
            manual_boxes=(),
            auto_remove_text=self.auto_remove.get(),
            prefer_lama=self.prefer_lama.get(),
            prefer_esrgan=self.prefer_esrgan.get(),
            enhance=self.enhance.get(),
            out_square=self.output_square_var.get(),
            out_thumb=self.output_thumb_var.get(),
            out_shorts=self.output_shorts_var.get(),
            thumbnail_resolution=self.thumbnail_resolution.get(),
            duplicate_policy=self.duplicate_policy_key(),
            extension_mode=self.extension_key(),
            protect_core=self.protect_core.get(),
            use_sdxl=True,
            protect_person=self.protect_person.get(),
            outpaint_prompt=self.outpaint_prompt.get("1.0", "end").strip(),
        )

    def jobs_for_run(self) -> list[ImageJob]:
        jobs: list[ImageJob] = []
        for state in self.image_states:
            job = state.job
            jobs.append(
                ImageJob(
                    source=job.source,
                    out_square=job.out_square,
                    out_thumb=job.out_thumb,
                    out_shorts=job.out_shorts,
                    manual_boxes=tuple(job.manual_boxes),
                    ocr_boxes=tuple(job.ocr_boxes),
                    preset_name=job.preset_name,
                    ocr_languages=tuple(job.ocr_languages),
                    extension_mode=job.extension_mode,
                    subject_offset_x=job.subject_offset_x,
                    subject_offset_y=job.subject_offset_y,
                    subject_scale=job.subject_scale,
                    outpaint_prompt=job.outpaint_prompt,
                )
            )
        return jobs

    def run_single_image(self, state: ImageRowState) -> None:
        if self.worker_thread is not None and self.worker_thread.is_alive():
            messagebox.showwarning("작업 중", "현재 작업이 끝난 뒤 개별 변환을 실행해주세요.")
            return
        if count_selected_outputs_for_jobs([state.job]) == 0:
            messagebox.showinfo("안내", "이 행에서 생성할 출력 규격을 하나 이상 선택해주세요.")
            return
        output_dir = self.validate_output_directory_for_run()
        if output_dir is None:
            return
        state.job.status = "처리 중"
        state.job.error = ""
        self.refresh_image_table()
        options = self.pipeline_options(output_dir)
        job = self.jobs_for_run()[self.image_states.index(state)]
        total_outputs = count_selected_outputs_for_jobs([job])
        self.progress_bar.set(0)
        self.top_progress_bar.set(0)
        self.top_progress_label.configure(text=f"0/{total_outputs} 결과 생성 완료")

        def worker() -> None:
            results = process_image_jobs(
                [job],
                options,
                self.root_dir,
                ai=self.ai,
                progress_callback=lambda payload: self.worker_queue.put(
                    {"type": "pipeline_progress", "payload": payload, "row_indices": [self.image_states.index(state)]}
                ),
                cancel_event=self.cancel_event,
            )
            self.worker_queue.put(
                {"type": "pipeline_done", "results": results, "output_dir": options.output_dir,
                 "row_indices": [self.image_states.index(state)]}
            )

        self.start_worker("Single image", worker)

    def open_row_folder(self, state: ImageRowState) -> None:
        path = state.result.item_dir if state.result is not None else self.output_dir_for_state(state)
        self.safe_open_folder(path)

    def output_dir_for_state(self, state: ImageRowState) -> Path:
        root = self.output_path_var.get().strip()
        return Path(root) / state.job.source.stem if root else default_output_dir_for_source(state.job.source)

    def export_row_results(self, state: ImageRowState) -> None:
        if state.result is None or not state.result.metadata.get("output_files"):
            messagebox.showinfo("안내", "먼저 해당 이미지의 변환을 완료해주세요.")
            return
        directory = filedialog.askdirectory(title="결과를 내보낼 폴더")
        if not directory:
            return
        try:
            target = Path(directory)
            target.mkdir(parents=True, exist_ok=True)
            for source_name in state.result.metadata["output_files"].values():
                source_path = Path(source_name)
                shutil.copy2(source_path, target / source_path.name)
            self.status.configure(text=f"결과 저장 완료:\n{target}")
        except (OSError, shutil.Error) as exc:
            write_exception(self.root_dir, "Row result export", exc)
            messagebox.showerror("결과 저장 실패", str(exc))

    def run_selected_pipeline(self) -> None:
        if self.selected_index < 0 or self.selected_index >= len(self.image_states):
            messagebox.showinfo("안내", "먼저 목록에서 이미지를 선택해주세요.")
            return
        self.run_single_image(self.image_states[self.selected_index])

    def run_pipeline(self) -> None:
        if not self.image_states:
            messagebox.showinfo("안내", "먼저 커버 이미지를 추가해주세요.")
            return
        jobs = self.jobs_for_run()
        if count_selected_outputs_for_jobs(jobs) == 0:
            messagebox.showinfo("안내", "생성할 출력 이미지 규격을 하나 이상 선택해주세요.")
            return

        output_dir = self.validate_output_directory_for_run()
        if output_dir is None:
            return

        for state in self.image_states:
            state.job.status = "대기"
            state.job.error = ""
        self.refresh_image_table()
        options = self.pipeline_options(output_dir)
        total_outputs = count_selected_outputs_for_jobs(jobs)
        self.progress_bar.set(0)
        self.progress_label.configure(text=f"0/{total_outputs} 결과물 완료 | 성공 0 | 실패 0")
        self.update_summary()

        def worker() -> None:
            results = process_image_jobs(
                jobs,
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
                }
            )

        self.start_worker("Pipeline", worker)

    def cancel_work(self) -> None:
        self.cancel_event.set()
        self.status.configure(text="현재 단계가 끝나면 작업을 취소합니다.")
