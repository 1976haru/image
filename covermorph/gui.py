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
from .generation import (
    DEFAULT_IP_ADAPTER,
    DEFAULT_SDXL_MODEL,
    IP_ADAPTER_ENCODER,
    GenerationConfig,
    GenerationError,
    SDXLTextToImageEngine,
    detect_generation_environment,
    generate_scene_candidates,
    retry_failed_candidates,
)
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
from .project import (
    INPUT_TYPE_EXISTING_COVER,
    INPUT_TYPE_TEXTLESS,
    CandidateRecord,
    CandidateSettings,
    ChannelGenerationPreset,
    CoverMorphProject,
    ProjectAssetError,
    ProjectLoadError,
    SceneCard,
    add_candidate_from_file,
    add_input_records,
    add_person,
    add_person_reference,
    adopt_generated_candidate,
    adopt_removal_preview,
    configure_scene_prompt,
    create_project,
    load_generation_presets,
    load_project,
    parse_input_file,
    path_to_project_string,
    resolve_project_path,
    save_generation_presets,
    save_project_atomic,
    save_removal_preview,
    validate_project_assets,
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

REFERENCE_MODE_LABELS = {"off": "끔", "person": "인물", "style": "스타일"}
REFERENCE_MODE_KEYS = {label: key for key, label in REFERENCE_MODE_LABELS.items()}
GENERATION_RATIO_LABELS = {"1:1": "1:1 1024x1024", "16:9": "가로 생성 1344x768", "9:16": "세로 생성 768x1344"}
GENERATION_RATIO_KEYS = {label: key for key, label in GENERATION_RATIO_LABELS.items()}

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

MAX_IMAGES = 50
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
PROJECT_FILETYPES = [("CoverMorph Project", "covermorph_project.json"), ("JSON", "*.json")]
INPUT_TYPE_LABELS = {
    "글자 없는 이미지": INPUT_TYPE_TEXTLESS,
    "글자가 포함된 기존 커버": INPUT_TYPE_EXISTING_COVER,
}
INPUT_TYPE_STATUS = {
    INPUT_TYPE_TEXTLESS: "작업 원본 채택됨",
    INPUT_TYPE_EXISTING_COVER: "제거 결과 필요",
}
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
    candidate: CandidateRecord | None = None
    selected_var: ctk.BooleanVar | None = None
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
        self.project: CoverMorphProject | None = None
        self.project_dirty = False
        self._loading_project = False

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
        self.protocol("WM_DELETE_WINDOW", self.on_close)
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
        self.project_name_var = ctk.StringVar(value="프로젝트 없음")
        self.channel_name_var = ctk.StringVar(value="")
        self.series_name_var = ctk.StringVar(value="")
        self.lyric_mood_var = ctk.StringVar(value="")
        self.input_type_var = ctk.StringVar(value="글자 없는 이미지")

        self.generation_presets = load_generation_presets(self.root_dir / "config" / "channel_generation_presets.json")
        self.generation_preset_var = ctk.StringVar(value=self.generation_presets[0].name if self.generation_presets else "")
        self.person_name_var = ctk.StringVar(value="")
        self.scene_description_var = ctk.StringVar(value="")
        self.reference_role_var = ctk.StringVar(value="person")
        self.preset_master_prompt_var = ctk.StringVar(value="")
        self.preset_negative_prompt_var = ctk.StringVar(value="")
        self.current_scene_id: str | None = None
        self.prompt_preview_text: ctk.CTkTextbox | None = None
        self.generation_count_var = ctk.IntVar(value=1)
        self.generation_seed_var = ctk.IntVar(value=1000)
        self.generation_steps_var = ctk.IntVar(value=28)
        self.generation_guidance_var = ctk.DoubleVar(value=7.0)
        self.generation_model_var = ctk.StringVar(value=DEFAULT_SDXL_MODEL)
        self.generation_ratio_var = ctk.StringVar(value=GENERATION_RATIO_LABELS["1:1"])
        self.reference_mode_var = ctk.StringVar(value="off")
        self.reference_image_var = ctk.StringVar(value="선택 안 함")
        self.reference_strength_var = ctk.DoubleVar(value=0.5)
        self.reference_crop_var = ctk.StringVar(value="")
        self.reference_image_options: dict[str, str] = {}
        self.reference_preview_photo: ImageTk.PhotoImage | None = None
        self.generation_status_label: ctk.CTkLabel | None = None
        self.last_generation_result: Any | None = None

        self.build_project_group(left)
        self.build_generation_group(left)
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
        ctk.CTkLabel(frame, text="입력 유형", anchor="w", text_color="#cbd5e1").pack(fill="x", padx=12)
        self.input_type_menu = ctk.CTkOptionMenu(
            frame,
            variable=self.input_type_var,
            values=list(INPUT_TYPE_LABELS),
        )
        self.input_type_menu.pack(fill="x", padx=12, pady=(3, 8))
        row = ctk.CTkFrame(frame, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=(0, 8))
        self.add_button = ctk.CTkButton(row, text="후보 추가", command=self.open_files, height=36)
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
            text="프로젝트를 만든 뒤 후보 이미지를 추가하세요.",
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

    def build_project_group(self, parent: Any) -> None:
        frame = self.group(parent, "0. 프로젝트", highlight=True)
        self.project_status = ctk.CTkLabel(
            frame,
            text="새 프로젝트를 만들거나 기존 프로젝트를 열어주세요.",
            wraplength=360,
            justify="left",
            text_color="#bae6fd",
        )
        self.project_status.pack(fill="x", padx=12, pady=(0, 8))

        button_row = ctk.CTkFrame(frame, fg_color="transparent")
        button_row.pack(fill="x", padx=12, pady=(0, 8))
        self.new_project_button = ctk.CTkButton(button_row, text="새 프로젝트", command=self.new_project)
        self.new_project_button.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.open_project_button = ctk.CTkButton(button_row, text="프로젝트 열기", command=self.open_project)
        self.open_project_button.pack(side="left", fill="x", expand=True, padx=4)
        self.save_project_button = ctk.CTkButton(button_row, text="저장", command=self.save_project)
        self.save_project_button.pack(side="left", fill="x", expand=True, padx=(4, 0))

        ctk.CTkLabel(frame, text="프로젝트명", anchor="w", text_color="#cbd5e1").pack(fill="x", padx=12)
        self.project_name_entry = ctk.CTkEntry(frame, textvariable=self.project_name_var)
        self.project_name_entry.pack(fill="x", padx=12, pady=(3, 8))
        ctk.CTkLabel(frame, text="채널명", anchor="w", text_color="#cbd5e1").pack(fill="x", padx=12)
        self.channel_name_entry = ctk.CTkEntry(frame, textvariable=self.channel_name_var)
        self.channel_name_entry.pack(fill="x", padx=12, pady=(3, 8))
        ctk.CTkLabel(frame, text="시리즈명", anchor="w", text_color="#cbd5e1").pack(fill="x", padx=12)
        self.series_name_entry = ctk.CTkEntry(frame, textvariable=self.series_name_var)
        self.series_name_entry.pack(fill="x", padx=12, pady=(3, 8))
        ctk.CTkLabel(frame, text="가사 분위기/주제", anchor="w", text_color="#cbd5e1").pack(fill="x", padx=12)
        self.lyric_mood_entry = ctk.CTkEntry(frame, textvariable=self.lyric_mood_var)
        self.lyric_mood_entry.pack(fill="x", padx=12, pady=(3, 12))

    def build_generation_group(self, parent: Any) -> None:
        frame = self.group(parent, "1. 채널 설정 / 기준 인물 / 장면")
        names = [preset.name for preset in self.generation_presets] or ["새 생성 채널"]
        self.generation_preset_menu = ctk.CTkOptionMenu(frame, variable=self.generation_preset_var, values=names, command=self.on_generation_preset_changed)
        self.generation_preset_menu.pack(fill="x", padx=12, pady=(0, 4))
        self.preset_draft_label = ctk.CTkLabel(frame, text="생성용 채널 프리셋 초안 | 기존 출력 배치 프리셋과 별도", text_color="#fbbf24", anchor="w")
        self.preset_draft_label.pack(fill="x", padx=12, pady=3)
        if self.generation_presets:
            self.preset_master_prompt_var.set(self.generation_presets[0].master_prompt)
            self.preset_negative_prompt_var.set(self.generation_presets[0].negative_prompt)
        ctk.CTkLabel(frame, text="마스터 프롬프트 (글자 없는 이미지 기본)", anchor="w").pack(fill="x", padx=12, pady=(4, 2))
        self.preset_master_prompt_entry = ctk.CTkEntry(frame, textvariable=self.preset_master_prompt_var)
        self.preset_master_prompt_entry.pack(fill="x", padx=12, pady=2)
        ctk.CTkLabel(frame, text="네거티브 프롬프트", anchor="w").pack(fill="x", padx=12, pady=(3, 2))
        self.preset_negative_prompt_entry = ctk.CTkEntry(frame, textvariable=self.preset_negative_prompt_var)
        self.preset_negative_prompt_entry.pack(fill="x", padx=12, pady=2)
        row = ctk.CTkFrame(frame, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=3)
        self.duplicate_generation_button = ctk.CTkButton(row, text="프리셋 복제", command=self.duplicate_generation_preset)
        self.duplicate_generation_button.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.save_generation_button = ctk.CTkButton(row, text="프리셋 저장", command=self.save_generation_preset)
        self.save_generation_button.pack(side="left", fill="x", expand=True, padx=(4, 0))
        ctk.CTkLabel(frame, text="기준 인물 이름 (등록 후 여러 장면에서 같은 ID 사용)", anchor="w").pack(fill="x", padx=12, pady=(7, 2))
        self.person_name_entry = ctk.CTkEntry(frame, textvariable=self.person_name_var)
        self.person_name_entry.pack(fill="x", padx=12, pady=2)
        self.add_person_button = ctk.CTkButton(frame, text="인물 등록", command=self.add_project_person)
        self.add_person_button.pack(fill="x", padx=12, pady=3)
        self.reference_role_menu = ctk.CTkOptionMenu(frame, variable=self.reference_role_var, values=["person", "style", "background_composition"])
        self.reference_role_menu.pack(fill="x", padx=12, pady=3)
        self.add_reference_button = ctk.CTkButton(frame, text="선택 인물에 참고 이미지 추가", command=self.add_project_reference)
        self.add_reference_button.pack(fill="x", padx=12, pady=3)
        self.input_file_button = ctk.CTkButton(frame, text="가사/프롬프트 TXT·JSON 불러오기", command=self.import_project_input)
        self.input_file_button.pack(fill="x", padx=12, pady=3)
        ctk.CTkLabel(frame, text="장면 설명 (사용자 작성, 자동 가사 해석 아님)", anchor="w").pack(fill="x", padx=12, pady=(7, 2))
        self.scene_description_entry = ctk.CTkEntry(frame, textvariable=self.scene_description_var)
        self.scene_description_entry.pack(fill="x", padx=12, pady=2)
        self.compose_scene_button = ctk.CTkButton(frame, text="장면 추가 / 최종 프롬프트 구성", command=self.add_project_scene)
        self.compose_scene_button.pack(fill="x", padx=12, pady=3)
        self.duplicate_scene_button = ctk.CTkButton(frame, text="현재 장면 복제", command=self.duplicate_current_scene)
        self.duplicate_scene_button.pack(fill="x", padx=12, pady=3)
        self.confirm_scene_button = ctk.CTkButton(frame, text="현재 장면 확인 완료", command=self.confirm_current_scene)
        self.confirm_scene_button.pack(fill="x", padx=12, pady=3)
        self.prompt_preview_text = ctk.CTkTextbox(frame, height=110)
        self.prompt_preview_text.pack(fill="x", padx=12, pady=3)
        self.copy_prompt_button = ctk.CTkButton(frame, text="최종 프롬프트 복사", command=self.copy_current_prompt)
        self.copy_prompt_button.pack(fill="x", padx=12, pady=(3, 12))
        ctk.CTkLabel(frame, text="3-B1 로컬 SDXL + IP-Adapter 후보 생성", text_color="#fbbf24", anchor="w").pack(fill="x", padx=12, pady=(5, 2))
        self.generation_count_entry = ctk.CTkEntry(frame, textvariable=self.generation_count_var)
        self.generation_count_entry.pack(fill="x", padx=12, pady=2)
        self.generation_ratio_menu = ctk.CTkOptionMenu(frame, variable=self.generation_ratio_var, values=list(GENERATION_RATIO_LABELS.values()), command=self.on_generation_ratio_changed)
        self.generation_ratio_menu.pack(fill="x", padx=12, pady=2)
        ctk.CTkLabel(frame, text="후보 수 | seed 시작값 | steps | guidance", anchor="w").pack(fill="x", padx=12, pady=2)
        generation_row = ctk.CTkFrame(frame, fg_color="transparent")
        generation_row.pack(fill="x", padx=12, pady=2)
        for variable in (self.generation_seed_var, self.generation_steps_var, self.generation_guidance_var):
            ctk.CTkEntry(generation_row, textvariable=variable, width=90).pack(side="left", padx=2, expand=True, fill="x")
        self.generation_model_entry = ctk.CTkEntry(frame, textvariable=self.generation_model_var)
        self.generation_model_entry.pack(fill="x", padx=12, pady=2)
        self.download_model_button = ctk.CTkButton(frame, text="SDXL 모델 명시적 다운로드/준비", command=self.download_generation_model)
        self.download_model_button.pack(fill="x", padx=12, pady=3)
        ctk.CTkLabel(frame, text="IP-Adapter Plus SDXL ViT-H | 인물 고정 보장 아님 | 스타일 참고는 인물과 구도에도 영향을 줄 수 있음", anchor="w", wraplength=350, justify="left").pack(fill="x", padx=12, pady=(5, 2))
        self.reference_mode_var.set(REFERENCE_MODE_LABELS["off"])
        self.reference_mode_menu = ctk.CTkOptionMenu(frame, variable=self.reference_mode_var, values=list(REFERENCE_MODE_LABELS.values()), command=self.on_reference_setting_changed)
        self.reference_mode_menu.pack(fill="x", padx=12, pady=2)
        self.reference_image_menu = ctk.CTkOptionMenu(frame, variable=self.reference_image_var, values=["선택 안 함"], command=self.on_reference_setting_changed)
        self.reference_image_menu.pack(fill="x", padx=12, pady=2)
        self.reference_preview_label = ctk.CTkLabel(frame, text="참고 이미지 미선택", wraplength=350, justify="left", anchor="w")
        self.reference_preview_label.pack(fill="x", padx=12, pady=2)
        ctk.CTkLabel(frame, text="참고 강도 0.0 - 1.0 (기본 0.5)", anchor="w").pack(fill="x", padx=12, pady=(2, 0))
        self.reference_strength_slider = ctk.CTkSlider(frame, from_=0.0, to=1.0, variable=self.reference_strength_var, command=lambda _value: self.on_reference_setting_changed())
        self.reference_strength_slider.pack(fill="x", padx=12, pady=2)
        self.reference_crop_entry = ctk.CTkEntry(frame, textvariable=self.reference_crop_var, placeholder_text="선택 크롭: left,top,right,bottom (비움=전체)")
        self.reference_crop_entry.pack(fill="x", padx=12, pady=2)
        self.download_adapter_button = ctk.CTkButton(frame, text="IP-Adapter 명시적 준비", command=self.download_ip_adapter)
        self.download_adapter_button.pack(fill="x", padx=12, pady=3)
        self.generate_scene_button = ctk.CTkButton(frame, text="확인된 현재 장면 후보 생성", command=self.generate_current_scene)
        self.generate_scene_button.pack(fill="x", padx=12, pady=3)
        self.retry_generation_button = ctk.CTkButton(frame, text="실패/취소 후보만 재시도", command=self.retry_current_generation)
        self.retry_generation_button.pack(fill="x", padx=12, pady=3)
        self.generation_status_label = ctk.CTkLabel(frame, text="환경 확인 전", wraplength=350, justify="left", anchor="w")
        self.generation_status_label.pack(fill="x", padx=12, pady=(2, 12))

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
        self.adopt_preview_button = ctk.CTkButton(
            frame,
            text="제거 결과를 작업 원본으로 사용",
            command=self.adopt_clean_preview,
            height=32,
            fg_color="#0f766e",
        )
        self.adopt_preview_button.pack(fill="x", padx=12, pady=4)
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
            text="1:1 글자 없는 커버 베이스\n  - 1400x1400 JPG\n  - 문구 합성 전 정사각형 이미지",
            variable=self.output_square_var,
            command=self.on_output_defaults_changed,
        )
        self.square_check.pack(anchor="w", padx=12, pady=(0, 9))
        self.thumb_check = ctk.CTkCheckBox(
            frame,
            text="16:9 Canva/CapCut 이미지\n  - 기본 1920x1080 JPG\n  - 설정에서 1280x720 선택 가능",
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
            text="AI 자연 배경 확장이 기본값입니다. 실패하면 해당 출력은 저장하지 않으며, 자연 확장·스마트 크롭·블러를 직접 선택해 다시 처리하세요.",
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
            self.new_project_button,
            self.open_project_button,
            self.save_project_button,
            self.project_name_entry,
            self.channel_name_entry,
            self.series_name_entry,
            self.lyric_mood_entry,
            self.generation_preset_menu,
            self.preset_master_prompt_entry,
            self.preset_negative_prompt_entry,
            self.duplicate_generation_button,
            self.save_generation_button,
            self.person_name_entry,
            self.add_person_button,
            self.reference_role_menu,
            self.add_reference_button,
            self.input_file_button,
            self.scene_description_entry,
            self.compose_scene_button,
            self.duplicate_scene_button,
            self.confirm_scene_button,
            self.copy_prompt_button,
            self.generation_count_entry,
            self.generation_ratio_menu,
            self.generation_model_entry,
            self.download_model_button,
            self.reference_mode_menu,
            self.reference_image_menu,
            self.reference_preview_label,
            self.reference_strength_slider,
            self.reference_crop_entry,
            self.download_adapter_button,
            self.generate_scene_button,
            self.retry_generation_button,
            self.input_type_menu,
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
            self.adopt_preview_button,
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
        for var in (
            self.project_name_var,
            self.channel_name_var,
            self.series_name_var,
            self.lyric_mood_var,
        ):
            var.trace_add("write", lambda *_args: self.on_project_field_changed())

    def setup_drag_drop(self) -> None:
        if DND_FILES is None or not getattr(self, "_covermorph_dnd_ready", False):
            self.input_status.configure(text="후보 이미지를 선택하세요. 드래그앤드롭은 tkinterdnd2 설치 시 활성화됩니다.")
            return
        try:
            self.drop_target_register(DND_FILES)
            self.dnd_bind("<<Drop>>", self.on_drop_files)
            self.input_status.configure(text="후보 이미지를 선택하거나 창 위로 드래그앤드롭하세요.")
        except tk.TclError as exc:
            write_exception(self.root_dir, "Drag and drop setup", exc)
            self.input_status.configure(text="후보 이미지를 선택하세요. 드래그앤드롭 초기화에 실패했습니다.")

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
            self.update_candidate_from_state(state)
            self.mark_project_dirty()
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
            self.update_candidate_from_state(state)
            self.mark_project_dirty()
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
        self.update_candidate_from_state(state)
        self.mark_project_dirty()
        self.draw_preview()

    def apply_saved_output_directory_notice(self) -> None:
        path_text = self.output_path_var.get().strip()
        if not path_text:
            return
        path = Path(path_text)
        self.output_dir = path
        if not path.exists() or not path.is_dir():
            self.status.configure(text="저장된 출력 폴더를 사용할 수 없습니다. 새 출력 폴더를 선택해주세요.")

    def selected_generation_preset(self) -> ChannelGenerationPreset:
        for preset in self.generation_presets:
            if preset.name == self.generation_preset_var.get():
                return preset
        return self.generation_presets[0] if self.generation_presets else ChannelGenerationPreset("custom", "새 생성 채널")

    def on_generation_preset_changed(self, _value: str = "") -> None:
        preset = self.selected_generation_preset()
        self.preset_master_prompt_var.set(preset.master_prompt)
        self.preset_negative_prompt_var.set(preset.negative_prompt)
        if self.project is None:
            return
        self.project.channel_preset_id = preset.preset_id
        self.project.channel_preset = preset.to_dict()
        self.mark_project_dirty()
        self.preset_draft_label.configure(text="생성용 채널 프리셋 초안 | 기존 확정 장면은 유지됨")

    def duplicate_generation_preset(self) -> None:
        source = self.selected_generation_preset()
        copied = ChannelGenerationPreset.from_dict({**source.to_dict(), "preset_id": "", "name": f"{source.name} 복제", "is_draft": True, "version": source.version + 1})
        self.generation_presets.append(copied)
        self.generation_preset_menu.configure(values=[item.name for item in self.generation_presets])
        self.generation_preset_var.set(copied.name)
        self.on_generation_preset_changed()

    def save_generation_preset(self) -> None:
        try:
            preset = self.selected_generation_preset()
            preset.master_prompt = self.preset_master_prompt_var.get().strip()
            preset.negative_prompt = self.preset_negative_prompt_var.get().strip()
            preset.version += 1
            if self.project is not None:
                self.project.channel_preset_id = preset.preset_id
                self.project.channel_preset = preset.to_dict()
                self.mark_project_dirty()
            save_generation_presets(self.generation_presets, self.root_dir / "config" / "channel_generation_presets.json")
            self.status.configure(text="생성용 채널 프리셋을 저장했습니다.")
        except OSError as exc:
            messagebox.showerror("프리셋 저장 실패", str(exc))

    def add_project_person(self) -> None:
        if not self.ensure_project_for_assets():
            return
        person = add_person(self.project, self.person_name_var.get().strip() or "새 인물")  # type: ignore[arg-type]
        self.person_name_var.set(f"{person.name} [{person.person_id}]")
        self.mark_project_dirty()
        self.status.configure(text=f"기준 인물을 등록했습니다: {person.name} ({person.person_id})")

    def add_project_reference(self) -> None:
        if self.project is None or not self.project.people:
            messagebox.showinfo("기준 인물", "먼저 기준 인물을 등록해 주세요.")
            return
        filename = filedialog.askopenfilename(title="참고 이미지", filetypes=[("Images", "*.png *.jpg *.jpeg *.webp"), ("All files", "*.*")])
        if not filename:
            return
        try:
            reference = add_person_reference(self.project, self.project.people[-1], Path(filename), self.reference_role_var.get(), "other")
        except ProjectAssetError as exc:
            messagebox.showerror("참고 이미지 추가 실패", str(exc))
            return
        self.mark_project_dirty()
        self.status.configure(text=f"참고 이미지를 복사했습니다. 모델 적용은 3단계에서 연결됩니다.\n{reference.path}")
        self.refresh_reference_options()

    def import_project_input(self) -> None:
        if not self.ensure_project_for_assets():
            return
        filename = filedialog.askopenfilename(title="가사 또는 프롬프트", filetypes=[("Text/JSON", "*.txt *.json"), ("All files", "*.*")])
        if not filename:
            return
        try:
            records = parse_input_file(Path(filename))
        except ProjectLoadError as exc:
            messagebox.showerror("입력 불러오기 실패", str(exc))
            return
        try:
            add_input_records(self.project, records)  # type: ignore[arg-type]
        except ProjectAssetError as exc:
            messagebox.showerror("입력 복사 실패", str(exc))
            return
        self.mark_project_dirty()
        self.status.configure(text=f"입력 자료 {len(records)}개를 추가했습니다. 가사 원문과 장면 설명은 별도로 보존됩니다.")

    def add_project_scene(self) -> None:
        if not self.ensure_project_for_assets():
            return
        preset = self.selected_generation_preset()
        scene = SceneCard(scene_id=f"scene_{len(self.project.scenes) + 1}", order=len(self.project.scenes) + 1, user_description=self.scene_description_var.get().strip())  # type: ignore[union-attr]
        configure_scene_prompt(self.project, scene, preset)  # type: ignore[arg-type]
        self.project.scenes.append(scene)  # type: ignore[union-attr]
        self.current_scene_id = scene.scene_id
        self.refresh_reference_options()
        self.show_scene_prompt(scene)
        self.mark_project_dirty()

    def current_scene(self) -> SceneCard | None:
        if self.project is None:
            return None
        return next((scene for scene in self.project.scenes if scene.scene_id == self.current_scene_id), self.project.scenes[-1] if self.project.scenes else None)

    def refresh_reference_options(self) -> None:
        self.reference_image_options = {"선택 안 함": ""}
        if self.project is not None:
            for person in self.project.people:
                for reference in person.reference_images:
                    self.reference_image_options[f"{person.name} / {reference.role} / {reference.image_id}"] = reference.image_id
        values = list(self.reference_image_options)
        self.reference_image_menu.configure(values=values)
        scene = self.current_scene()
        request = scene.structured_request if scene is not None else {}
        mode = str(request.get("reference_mode") or "off")
        if mode not in REFERENCE_MODE_LABELS:
            mode = "off"
        self.reference_mode_var.set(REFERENCE_MODE_LABELS[mode])
        reference_id = str(request.get("reference_image_id") or "")
        selected_label = next((label for label, image_id in self.reference_image_options.items() if image_id == reference_id), values[0])
        self.reference_image_var.set(selected_label)
        self.reference_strength_var.set(float(request.get("reference_strength") or 0.5))
        crop_box = request.get("reference_crop_box")
        self.reference_crop_var.set(",".join(str(value) for value in crop_box) if isinstance(crop_box, list) and len(crop_box) == 4 else "")
        ratio = scene.output_ratio if scene is not None and scene.output_ratio in GENERATION_RATIO_LABELS else "1:1"
        self.generation_ratio_var.set(GENERATION_RATIO_LABELS[ratio])
        self.refresh_reference_preview()

    def reference_mode_key(self) -> str:
        return REFERENCE_MODE_KEYS.get(self.reference_mode_var.get(), self.reference_mode_var.get())

    def generation_ratio_key(self) -> str:
        return GENERATION_RATIO_KEYS.get(self.generation_ratio_var.get(), self.generation_ratio_var.get())

    def on_generation_ratio_changed(self, *_args: Any) -> None:
        scene = self.current_scene()
        if scene is None:
            return
        scene.output_ratio = self.generation_ratio_key()
        scene.prompt_confirmed = False
        self.mark_project_dirty()
        self.status.configure(text="생성 크기가 변경되어 장면 확인 상태를 미확인으로 바꿨습니다.")

    def refresh_reference_preview(self) -> None:
        selected_id = self.reference_image_options.get(self.reference_image_var.get(), "")
        reference = None
        if self.project is not None and selected_id:
            reference = next(
                (item for person in self.project.people for item in person.reference_images if item.image_id == selected_id),
                None,
            )
        self.reference_preview_photo = None
        if reference is None or self.project is None:
            self.reference_preview_label.configure(text=f"참고 선택: {self.reference_image_var.get()}\n실제 적용은 IP-Adapter 호출 성공 시에만 기록됩니다.", image=None)
            return
        try:
            path = resolve_project_path(self.project, reference.path)
            with Image.open(path) as opened:
                preview = ImageOps.exif_transpose(opened).convert("RGB")
                preview.thumbnail((320, 180), Image.Resampling.LANCZOS)
                self.reference_preview_photo = ImageTk.PhotoImage(preview)
            self.reference_preview_label.configure(
                text=f"참고 선택: {self.reference_image_var.get()}\n실제 적용은 IP-Adapter 호출 성공 시에만 기록됩니다.",
                image=self.reference_preview_photo,
                compound="top",
            )
        except (OSError, ProjectAssetError) as exc:
            self.reference_preview_label.configure(text=f"참고 이미지 미리보기 실패: {exc}", image=None)

    def parse_reference_crop(self) -> tuple[int, int, int, int] | None:
        text = self.reference_crop_var.get().strip()
        if not text:
            return None
        try:
            values = tuple(int(item.strip()) for item in text.split(","))
        except ValueError as exc:
            raise GenerationError("크롭은 left,top,right,bottom 정수로 입력해 주세요.") from exc
        if len(values) != 4:
            raise GenerationError("크롭은 left,top,right,bottom 네 값이어야 합니다.")
        return values  # type: ignore[return-value]

    def on_reference_setting_changed(self, *_args: Any) -> None:
        scene = self.current_scene()
        if scene is None:
            return
        mode = self.reference_mode_key()
        reference_id = self.reference_image_options.get(self.reference_image_var.get(), "") if mode != "off" else ""
        scene.structured_request["reference_mode"] = mode
        scene.structured_request["reference_image_id"] = reference_id
        scene.structured_request["reference_strength"] = float(self.reference_strength_var.get())
        scene.structured_request["reference_crop_box"] = list(self.parse_reference_crop()) if self.reference_crop_var.get().strip() else None
        scene.prompt_confirmed = False
        self.mark_project_dirty()
        self.refresh_reference_preview()
        self.status.configure(text="참고 설정이 변경되어 장면 확인 상태를 미확인으로 바꿨습니다.")

    def download_ip_adapter(self) -> None:
        target = self.root_dir / "models" / "ip_adapter"

        def worker() -> None:
            SDXLTextToImageEngine.download_ip_adapter(target, DEFAULT_IP_ADAPTER, progress=lambda payload: self.worker_queue.put({"type": "generation_progress", "payload": payload}))
            self.worker_queue.put({"type": "status", "message": f"IP-Adapter 준비 완료: {target}\n{IP_ADAPTER_ENCODER}"})

        self.start_worker("IP-Adapter model download", worker)

    def generate_current_scene(self) -> None:
        scene = self.current_scene()
        if self.project is None or scene is None:
            messagebox.showinfo("장면 필요", "먼저 장면을 추가하고 프롬프트를 확인해 주세요.")
            return
        if not scene.prompt_confirmed:
            messagebox.showwarning("프롬프트 확인 필요", "미확인 장면입니다. 프롬프트를 확인 완료한 뒤 생성해 주세요.")
            return
        environment = detect_generation_environment(self.root_dir, self.generation_model_var.get().strip() or DEFAULT_SDXL_MODEL)
        reference_mode = self.reference_mode_key()
        reference_ready = bool(environment.get("ip_adapter_ready"))
        if self.generation_status_label is not None:
            self.generation_status_label.configure(
                text=(
                    f"SDXL 모델: {'준비됨' if environment.get('model_ready') else '미준비'} | "
                    f"IP-Adapter: {'준비됨' if reference_ready else '미준비'}\n"
                    f"생성 환경: {environment.get('status')} | GPU: {environment.get('gpu') or '없음'}"
                )
            )
        if environment.get("status") != "ready":
            messagebox.showwarning("SDXL 생성 불가", "CUDA GPU와 Diffusers가 준비된 환경에서만 생성합니다. CPU 자동 전환은 하지 않습니다.")
            return
        if reference_mode != "off" and not reference_ready:
            reason = environment.get("ip_adapter", {}).get("failure_reason") or "IP-Adapter 준비 상태를 확인할 수 없습니다."
            messagebox.showwarning("참고 이미지 생성 불가", f"IP-Adapter 준비가 완료되지 않았습니다.\n{reason}")
            return
        try:
            config = GenerationConfig(model_id=self.generation_model_var.get().strip() or DEFAULT_SDXL_MODEL, output_ratio=self.generation_ratio_key(), candidate_count=max(1, int(self.generation_count_var.get())), seed=int(self.generation_seed_var.get()), steps=max(1, int(self.generation_steps_var.get())), guidance_scale=float(self.generation_guidance_var.get()), reference_mode=reference_mode, reference_image_id=self.reference_image_options.get(self.reference_image_var.get(), ""), reference_strength=float(self.reference_strength_var.get()), reference_crop_box=self.parse_reference_crop(), ip_adapter_id=str(self.root_dir / "models" / "ip_adapter"))
        except (TypeError, ValueError) as exc:
            messagebox.showerror("생성 설정 오류", str(exc))
            return
        project_snapshot = self.project
        engine = SDXLTextToImageEngine(config.model_id, config.revision, config.local_files_only)

        def worker() -> None:
            result = generate_scene_candidates(project_snapshot, scene, engine, config, self.cancel_event, lambda payload: self.worker_queue.put({"type": "generation_progress", "payload": payload}))
            save_project_atomic(project_snapshot)
            self.worker_queue.put({"type": "generation_done", "result": result})
            engine.unload()

        self.start_worker("SDXL candidate generation", worker)

    def retry_current_generation(self) -> None:
        result = self.last_generation_result
        scene = self.current_scene()
        if result is None or not result.failed_indices or self.project is None or scene is None:
            messagebox.showinfo("재시도", "재시도할 실패 또는 취소 후보가 없습니다.")
            return
        config = GenerationConfig(model_id=self.generation_model_var.get().strip() or DEFAULT_SDXL_MODEL, output_ratio=self.generation_ratio_key(), candidate_count=max(1, int(self.generation_count_var.get())), seed=int(self.generation_seed_var.get()), steps=max(1, int(self.generation_steps_var.get())), guidance_scale=float(self.generation_guidance_var.get()), reference_mode=self.reference_mode_key(), reference_image_id=self.reference_image_options.get(self.reference_image_var.get(), ""), reference_strength=float(self.reference_strength_var.get()), reference_crop_box=self.parse_reference_crop(), ip_adapter_id=str(self.root_dir / "models" / "ip_adapter"))
        project_snapshot = self.project
        engine = SDXLTextToImageEngine(config.model_id, config.revision, config.local_files_only)

        def worker() -> None:
            retry = retry_failed_candidates(project_snapshot, scene, engine, config, result.failed_indices, self.cancel_event, lambda payload: self.worker_queue.put({"type": "generation_progress", "payload": payload}))
            save_project_atomic(project_snapshot)
            self.worker_queue.put({"type": "generation_done", "result": retry})
            engine.unload()

        self.start_worker("SDXL retry", worker)

    def download_generation_model(self) -> None:
        model_id = self.generation_model_var.get().strip() or DEFAULT_SDXL_MODEL
        target = self.root_dir / "models" / "sdxl_base_1.0"

        def worker() -> None:
            SDXLTextToImageEngine.download(model_id, target, lambda payload: self.worker_queue.put({"type": "generation_progress", "payload": payload}))
            self.worker_queue.put({"type": "generation_model_ready", "path": str(target)})

        self.start_worker("SDXL model download", worker)

    def handle_generation_progress(self, payload: dict[str, Any]) -> None:
        phase = payload.get("phase", "")
        if phase == "inference":
            self.top_current_label.configure(text=f"생성 추론 step {payload.get('step')}/{payload.get('steps')}")
        elif phase == "candidate_start":
            self.top_current_label.configure(text=f"장면 {payload.get('scene_id')} 후보 {payload.get('candidate')}/{payload.get('total')} seed {payload.get('seed')}")
        elif phase == "model_loading":
            self.top_current_label.configure(text=f"모델 로딩: {payload.get('model')}")
        self.status.configure(text="3-B1 SDXL + IP-Adapter 생성 중 | 실제 적용 여부는 완료 후 기록됩니다.")

    def handle_generation_done(self, result: Any) -> None:
        self.last_generation_result = result
        if self.project is not None:
            self.project_dirty = False
            self.load_project_rows(self.project)
            self.refresh_image_table()
            self.update_summary()
        run = self.project.generation_runs[-1] if self.project is not None and self.project.generation_runs else {}
        actual = bool(run.get("actual_reference_applied"))
        self.status.configure(text=f"3-B1 생성 완료: 성공 {result.completed}, 실패 {result.failed}, 취소 {result.cancelled}\n실제 참고 적용: {'예' if actual else '아니오'} | 후보 시각 품질: 미검증")
        if self.generation_status_label is not None:
            self.generation_status_label.configure(text=f"생성 결과: 성공 {result.completed} / 실패 {result.failed} / 취소 {result.cancelled}\n실제 참고 적용: {'예' if actual else '아니오'}\n실패 원인: {run.get('failure_reason') or '없음'}\n시각 품질: 미검증")

    def show_scene_prompt(self, scene: SceneCard) -> None:
        if self.prompt_preview_text is None:
            return
        self.prompt_preview_text.delete("1.0", "end")
        self.prompt_preview_text.insert("1.0", f"[자동 구성]\n{scene.prompt_auto}\n\n[네거티브]\n{scene.negative_prompt_auto}\n\n확인 상태: {'완료' if scene.prompt_confirmed else '미확인'}")

    def confirm_current_scene(self) -> None:
        scene = self.current_scene()
        if scene is None:
            return
        scene.prompt_confirmed = True
        self.show_scene_prompt(scene)
        self.mark_project_dirty()

    def duplicate_current_scene(self) -> None:
        scene = self.current_scene()
        if scene is None or self.project is None:
            return
        from .project import duplicate_scene

        copied = duplicate_scene(scene)
        copied.order = len(self.project.scenes) + 1
        self.project.scenes.append(copied)
        self.current_scene_id = copied.scene_id
        self.show_scene_prompt(copied)
        self.mark_project_dirty()

    def copy_current_prompt(self) -> None:
        scene = self.current_scene()
        if scene is None:
            return
        self.clipboard_clear()
        self.clipboard_append(scene.prompt_user or scene.prompt_auto)
        self.status.configure(text="최종 프롬프트를 클립보드에 복사했습니다.")

    def project_input_type_key(self) -> str:
        return INPUT_TYPE_LABELS.get(self.input_type_var.get(), INPUT_TYPE_TEXTLESS)

    def refresh_project_status(self) -> None:
        marker = " *" if self.project_dirty else ""
        if self.project is None:
            self.project_status.configure(text="프로젝트 없음")
            self.title(f"CoverMorph Studio v{__version__}")
            return
        self.project_status.configure(
            text=(
                f"{self.project.name}{marker}\n"
                f"{self.project.project_file}\n"
                f"후보 {len(self.project.candidates)}장 | 곡 수 {self.project.song_count}곡"
            )
        )
        self.title(f"CoverMorph Studio v{__version__} - {self.project.name}{marker}")

    def mark_project_dirty(self) -> None:
        if self._loading_project or self.project is None:
            return
        self.project_dirty = True
        self.refresh_project_status()

    def on_project_field_changed(self) -> None:
        if self.project is None or self._loading_project:
            return
        self.mark_project_dirty()

    def sync_project_from_ui(self) -> None:
        if self.project is None:
            return
        self.project.name = self.project_name_var.get().strip() or self.project.project_file.parent.name
        self.project.channel_name = self.channel_name_var.get().strip()
        self.project.series_name = self.series_name_var.get().strip()
        self.project.lyric_mood_text = self.lyric_mood_var.get().strip()
        preset = self.selected_generation_preset()
        self.project.channel_preset_id = preset.preset_id
        self.project.channel_preset = preset.to_dict()
        self.project.selected_candidate_ids = [
            state.candidate.candidate_id
            for state in self.image_states
            if state.candidate is not None and state.candidate.selected
        ]
        for state in self.image_states:
            self.update_candidate_from_state(state)

    def apply_project_to_ui(self, project: CoverMorphProject) -> None:
        self._loading_project = True
        try:
            self.project_name_var.set(project.name)
            self.channel_name_var.set(project.channel_name)
            self.series_name_var.set(project.series_name)
            self.lyric_mood_var.set(project.lyric_mood_text)
            preset_id = project.channel_preset_id
            preset = next((item for item in self.generation_presets if item.preset_id == preset_id), None)
            if preset is None and project.channel_preset:
                preset = ChannelGenerationPreset.from_dict(project.channel_preset)
            if preset is not None:
                if not any(item.preset_id == preset.preset_id for item in self.generation_presets):
                    self.generation_presets.append(preset)
                    self.generation_preset_menu.configure(values=[item.name for item in self.generation_presets])
                self.generation_preset_var.set(preset.name)
            self.current_scene_id = project.scenes[-1].scene_id if project.scenes else None
            self.refresh_reference_options()
        finally:
            self._loading_project = False
        self.refresh_project_status()

    def confirm_discard_project_changes(self) -> bool:
        if not self.project_dirty:
            return True
        result = messagebox.askyesnocancel(
            "저장되지 않은 변경",
            "현재 프로젝트에 저장되지 않은 변경이 있습니다.\n저장할까요?",
        )
        if result is None:
            return False
        if result:
            return self.save_project()
        return True

    def ensure_project_for_assets(self) -> bool:
        if self.project is not None:
            return True
        messagebox.showinfo("프로젝트 필요", "이미지를 프로젝트 내부에 보존하려면 먼저 프로젝트 폴더를 만들어주세요.")
        return self.new_project()

    def new_project(self) -> bool:
        if not self.confirm_discard_project_changes():
            return False
        directory = filedialog.askdirectory(title="새 CoverMorph 프로젝트 폴더")
        if not directory:
            return False
        project_dir = Path(directory)
        project = create_project(project_dir, project_dir.name)
        self.project = project
        self.project_dirty = False
        self.image_states.clear()
        self.selected_index = None
        self.apply_project_to_ui(project)
        self.refresh_image_table()
        self.update_summary()
        self.draw_preview()
        try:
            save_project_atomic(project)
        except OSError as exc:
            write_exception(self.root_dir, "Project create save", exc)
            messagebox.showerror("프로젝트 저장 실패", str(exc))
            self.project_dirty = True
            self.refresh_project_status()
            return False
        self.status.configure(text=f"새 프로젝트를 만들었습니다.\n{project.project_file}")
        return True

    def open_project(self) -> bool:
        if not self.confirm_discard_project_changes():
            return False
        filename = filedialog.askopenfilename(
            title="CoverMorph 프로젝트 열기",
            filetypes=PROJECT_FILETYPES,
        )
        if not filename:
            return False
        try:
            project = load_project(Path(filename))
        except ProjectLoadError as exc:
            messagebox.showerror("프로젝트 열기 실패", str(exc))
            return False
        self.project = project
        self.project_dirty = False
        self.apply_project_to_ui(project)
        self.load_project_rows(project)
        issues = validate_project_assets(project)
        if issues:
            lines = [f"{issue.candidate_id}: {issue.message} ({issue.path or '경로 없음'})" for issue in issues]
            messagebox.showwarning("프로젝트 이미지 누락", "\n".join(lines[:8]))
        self.status.configure(text=f"프로젝트를 열었습니다.\n{project.project_file}")
        return True

    def save_project(self) -> bool:
        if self.project is None:
            directory = filedialog.askdirectory(title="CoverMorph 프로젝트 폴더")
            if not directory:
                return False
            self.project = create_project(Path(directory), Path(directory).name)
        self.sync_project_from_ui()
        try:
            save_project_atomic(self.project)
        except (OSError, ValueError) as exc:
            write_exception(self.root_dir, "Project save", exc)
            messagebox.showerror("프로젝트 저장 실패", str(exc))
            return False
        self.project_dirty = False
        self.refresh_project_status()
        self.status.configure(text=f"프로젝트 저장 완료\n{self.project.project_file}")
        return True

    def on_close(self) -> None:
        if self.worker_thread is not None and self.worker_thread.is_alive():
            if not messagebox.askyesno("작업 중", "현재 작업이 실행 중입니다. 종료할까요?"):
                return
            self.cancel_event.set()
        if not self.confirm_discard_project_changes():
            return
        self.destroy()

    def candidate_working_path(self, candidate: CandidateRecord | None) -> Path | None:
        if self.project is None or candidate is None or not candidate.working_source_path:
            return None
        return resolve_project_path(self.project, candidate.working_source_path)

    def candidate_original_path(self, candidate: CandidateRecord | None) -> Path | None:
        if self.project is None or candidate is None or not candidate.original_path:
            return None
        return resolve_project_path(self.project, candidate.original_path)

    def candidate_working_status(self, candidate: CandidateRecord | None) -> str:
        if candidate is None:
            return "외부 파일"
        if candidate.working_source_approved and candidate.working_source_path:
            return "채택됨"
        if candidate.removal_preview_path:
            return "채택 대기"
        return INPUT_TYPE_STATUS.get(candidate.input_type, "확인 필요")

    def make_thumbnail_photo(self, img: Image.Image) -> ImageTk.PhotoImage:
        thumb = ImageOps.contain(img.copy(), (58, 58), Image.Resampling.LANCZOS)
        thumb_bg = Image.new("RGB", (58, 58), (15, 23, 42))
        thumb_bg.paste(thumb, ((58 - thumb.width) // 2, (58 - thumb.height) // 2))
        return ImageTk.PhotoImage(thumb_bg)

    def load_project_rows(self, project: CoverMorphProject) -> None:
        self.image_states.clear()
        self.selected_index = None
        errors: list[str] = []
        for candidate in project.candidates:
            original_path = resolve_project_path(project, candidate.original_path)
            try:
                with Image.open(original_path) as opened:
                    original = ImageOps.exif_transpose(opened).convert("RGB")
            except (OSError, UnidentifiedImageError) as exc:
                errors.append(f"{candidate.display_name}: {exc}")
                continue

            work_path = resolve_project_path(project, candidate.working_source_path) if candidate.working_source_path else original_path
            settings = candidate.settings
            job = ImageJob(
                source=work_path,
                out_square=settings.out_square,
                out_thumb=settings.out_thumb,
                out_shorts=settings.out_shorts,
                auto_remove_text=False if candidate.working_source_path else None,
                manual_boxes=tuple(settings.manual_boxes),
                ocr_boxes=tuple(settings.ocr_boxes),
                preset_name=settings.preset_name,
                ocr_languages=tuple(settings.ocr_languages),
                extension_mode=settings.extension_mode,
                subject_offset_x=settings.subject_offset_x,
                subject_offset_y=settings.subject_offset_y,
                subject_scale=settings.subject_scale,
                outpaint_prompt=settings.outpaint_prompt or DEFAULT_OUTPAINT_PROMPT,
            )
            clean_preview = None
            preview_source = candidate.working_source_path or candidate.removal_preview_path
            if preview_source:
                try:
                    with Image.open(resolve_project_path(project, preview_source)) as opened:
                        clean_preview = ImageOps.exif_transpose(opened).convert("RGB")
                except (OSError, UnidentifiedImageError):
                    clean_preview = None
            state = ImageRowState(
                job=job,
                original=original,
                thumbnail_photo=self.make_thumbnail_photo(clean_preview or original),
                candidate=candidate,
                clean_preview=clean_preview,
            )
            self.image_states.append(state)

        if self.image_states:
            self.selected_index = 0
        self.sync_controls_from_selected()
        self.refresh_image_table()
        self.update_summary()
        self.draw_preview()
        if errors:
            messagebox.showwarning("이미지 로드 오류", "\n".join(errors[:8]))

    def update_candidate_from_state(self, state: ImageRowState) -> None:
        candidate = state.candidate
        if candidate is None:
            return
        candidate.selected = bool(state.selected_var.get()) if state.selected_var is not None else candidate.selected
        candidate.settings = CandidateSettings(
            out_square=state.job.out_square,
            out_thumb=state.job.out_thumb,
            out_shorts=state.job.out_shorts,
            manual_boxes=tuple(state.job.manual_boxes),
            ocr_boxes=tuple(state.job.ocr_boxes),
            preset_name=state.job.preset_name,
            ocr_languages=tuple(state.job.ocr_languages),
            extension_mode=state.job.extension_mode,
            subject_offset_x=state.job.subject_offset_x,
            subject_offset_y=state.job.subject_offset_y,
            subject_scale=state.job.subject_scale,
            outpaint_prompt=state.job.outpaint_prompt,
        )

    def lang_codes(self) -> tuple[str, ...]:
        return OCR_LANGUAGE_LABELS.get(self.ocr_lang.get(), ("en",))

    def current_state(self) -> ImageRowState | None:
        if self.selected_index is None:
            return None
        if self.selected_index < 0 or self.selected_index >= len(self.image_states):
            return None
        return self.image_states[self.selected_index]

    def jobs(self) -> list[ImageJob]:
        return [state.job for state in self.image_states if self.state_is_selected_for_run(state)]

    def state_is_selected_for_run(self, state: ImageRowState) -> bool:
        if state.candidate is None:
            return True
        return bool(state.candidate.selected)

    def open_files(self) -> None:
        paths = filedialog.askopenfilenames(filetypes=[("Images", "*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff")])
        if paths:
            self.add_image_paths([Path(path) for path in paths])

    def add_image_paths(self, paths: list[Path]) -> None:
        if not self.ensure_project_for_assets():
            return
        candidates = [path for path in paths if path.suffix.lower() in IMAGE_EXTENSIONS]
        if not candidates:
            messagebox.showinfo("안내", "지원하는 이미지 파일을 선택해주세요.")
            return

        existing: set[Path] = set()
        if self.project is not None:
            for candidate in self.project.candidates:
                if candidate.external_source_path:
                    try:
                        existing.add(Path(candidate.external_source_path).resolve())
                    except OSError:
                        pass
        added = 0
        errors: list[str] = []
        input_type = self.project_input_type_key()
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
                settings = CandidateSettings(
                    out_square=self.output_square_var.get(),
                    out_thumb=self.output_thumb_var.get(),
                    out_shorts=self.output_shorts_var.get(),
                    manual_boxes=(),
                    ocr_boxes=(),
                    preset_name=self.preset_name.get(),
                    ocr_languages=self.lang_codes(),
                    extension_mode=self.extension_key(),
                    outpaint_prompt=self.outpaint_prompt.get("1.0", "end").strip(),
                )
                candidate = add_candidate_from_file(self.project, path, input_type, settings)  # type: ignore[arg-type]
                original_path = self.candidate_original_path(candidate)
                if original_path is None:
                    raise ProjectAssetError("프로젝트 내부 원본 경로를 만들 수 없습니다.")
                with Image.open(original_path) as opened:
                    original = ImageOps.exif_transpose(opened).convert("RGB")
                work_path = self.candidate_working_path(candidate) or original_path
                clean_preview = None
                if candidate.working_source_path:
                    with Image.open(work_path) as opened:
                        clean_preview = ImageOps.exif_transpose(opened).convert("RGB")
            except (OSError, UnidentifiedImageError, ProjectAssetError) as exc:
                write_exception(self.root_dir, f"Open image {path.name}", exc)
                errors.append(f"{path.name}: {exc}")
                continue

            photo = self.make_thumbnail_photo(clean_preview or original)
            job = ImageJob(
                source=work_path,
                out_square=self.output_square_var.get(),
                out_thumb=self.output_thumb_var.get(),
                out_shorts=self.output_shorts_var.get(),
                auto_remove_text=False if candidate.working_source_path else None,
                preset_name=self.preset_name.get(),
                ocr_languages=self.lang_codes(),
                extension_mode=self.extension_key(),
                outpaint_prompt=self.outpaint_prompt.get("1.0", "end").strip(),
            )
            self.image_states.append(
                ImageRowState(
                    job=job,
                    original=original,
                    thumbnail_photo=photo,
                    candidate=candidate,
                    clean_preview=clean_preview,
                )
            )
            existing.add(resolved)
            added += 1

        if len(self.image_states) >= MAX_IMAGES:
            self.input_status.configure(text=f"후보는 최대 {MAX_IMAGES}장까지 선택할 수 있습니다.")
        else:
            self.input_status.configure(text=f"후보 {len(self.image_states)}장")

        if errors:
            messagebox.showwarning("이미지 오류", "일부 이미지를 열 수 없습니다.\n" + "\n".join(errors[:5]))

        if added and self.selected_index is None:
            self.selected_index = 0
        if added and not self.output_path_var.get().strip():
            default_dir = self.default_output_for_current_files() or default_output_dir_for_source(self.image_states[0].job.source)
            self.output_path_var.set(str(default_dir))
            self.output_dir = default_dir

        self.sync_controls_from_selected()
        self.refresh_image_table()
        self.update_summary()
        self.save_current_settings(include_output=False)
        if added:
            self.mark_project_dirty()
        self.draw_preview()

    def remove_selected_image(self) -> None:
        if self.selected_index is None:
            return
        state = self.image_states[self.selected_index]
        if self.project is not None and state.candidate is not None:
            self.project.candidates = [
                candidate
                for candidate in self.project.candidates
                if candidate.candidate_id != state.candidate.candidate_id
            ]
            self.mark_project_dirty()
        del self.image_states[self.selected_index]
        if not self.image_states:
            self.selected_index = None
        else:
            self.selected_index = min(self.selected_index, len(self.image_states) - 1)
        self.input_status.configure(text=f"후보 {len(self.image_states)}장")
        self.sync_controls_from_selected()
        self.refresh_image_table()
        self.update_summary()
        self.draw_preview()

    def clear_images(self) -> None:
        if self.project is not None and self.image_states:
            self.project.candidates.clear()
            self.project.selected_candidate_ids.clear()
            self.mark_project_dirty()
        self.image_states.clear()
        self.selected_index = None
        self.input_status.configure(text="후보 이미지를 선택하거나 창 위로 드래그앤드롭하세요.")
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

        headers = ["번호", "선택", "미리보기", "원본 파일명", "작업 원본", "1:1", "16:9", "9:16", "확장 방식", "상태"]
        widths = [46, 54, 76, 220, 112, 54, 54, 54, 170, 86, 92, 92, 86]
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
            if state.candidate is not None:
                state.selected_var = ctk.BooleanVar(value=state.candidate.selected)
            else:
                state.selected_var = ctk.BooleanVar(value=True)
            state.extension_var = ctk.StringVar(
                value=EXTENSION_MODES.get(state.job.extension_mode, EXTENSION_MODES["ai_natural"])
            )
            state.row_widgets = []

            widgets: list[Any] = [
                ctk.CTkLabel(state.row_frame, text=str(row_index), width=widths[0]),
                ctk.CTkCheckBox(
                    state.row_frame,
                    text="",
                    width=widths[1],
                    variable=state.selected_var,
                    command=lambda item=state: self.on_candidate_selected_changed(item),
                ),
                ctk.CTkLabel(state.row_frame, image=state.thumbnail_photo, text="", width=widths[2]),
                ctk.CTkLabel(
                    state.row_frame,
                    text=state.candidate.display_name if state.candidate is not None else state.job.source.name,
                    width=widths[3],
                    anchor="w",
                ),
                ctk.CTkLabel(
                    state.row_frame,
                    text=self.candidate_working_status(state.candidate),
                    width=widths[4],
                    anchor="w",
                ),
                ctk.CTkCheckBox(
                    state.row_frame,
                    text="",
                    width=widths[5],
                    variable=state.square_var,
                    command=lambda item=state: self.on_row_output_changed(item),
                ),
                ctk.CTkCheckBox(
                    state.row_frame,
                    text="",
                    width=widths[6],
                    variable=state.thumb_var,
                    command=lambda item=state: self.on_row_output_changed(item),
                ),
                ctk.CTkCheckBox(
                    state.row_frame,
                    text="",
                    width=widths[7],
                    variable=state.shorts_var,
                    command=lambda item=state: self.on_row_output_changed(item),
                ),
                ctk.CTkOptionMenu(
                    state.row_frame,
                    variable=state.extension_var,
                    values=list(EXTENSION_MODES.values()),
                    width=widths[8],
                    command=lambda _value, item=state: self.on_row_extension_changed(item),
                ),
            ]
            widgets.extend(
                [
                    ctk.CTkButton(
                        state.row_frame,
                        text="개별 변환",
                        width=widths[10],
                        height=28,
                        command=lambda item=state: self.run_single_image(item),
                    ),
                    ctk.CTkButton(
                        state.row_frame,
                        text="결과 저장",
                        width=widths[11],
                        height=28,
                        command=lambda item=state: self.export_row_results(item),
                    ),
                    ctk.CTkButton(
                        state.row_frame,
                        text="폴더 열기",
                        width=widths[12],
                        height=28,
                        command=lambda item=state: self.open_row_folder(item),
                    ),
                ]
            )
            state.status_label = ctk.CTkLabel(state.row_frame, text=state.job.status, width=widths[9], anchor="w")
            widgets.insert(9, state.status_label)

            selectable_cols = {0, 2, 3, 4, 9}
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

    def on_candidate_selected_changed(self, state: ImageRowState) -> None:
        if state.candidate is not None and state.selected_var is not None:
            state.candidate.selected = bool(state.selected_var.get())
            self.mark_project_dirty()
        self.update_summary()

    def on_row_output_changed(self, state: ImageRowState) -> None:
        state.job.out_square = bool(state.square_var and state.square_var.get())
        state.job.out_thumb = bool(state.thumb_var and state.thumb_var.get())
        state.job.out_shorts = bool(state.shorts_var and state.shorts_var.get())
        self.update_candidate_from_state(state)
        self.mark_project_dirty()
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
        self.update_candidate_from_state(state)
        self.mark_project_dirty()
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
            self.update_candidate_from_state(state)
        self.mark_project_dirty()
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
        if self.project is not None:
            return self.project.project_dir / "outputs"
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
        if state.candidate is not None and state.candidate.input_type == INPUT_TYPE_TEXTLESS:
            self.status.configure(text="글자 없는 이미지로 입력된 후보는 OCR을 실행하지 않습니다.")
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
        if state.candidate is not None and state.candidate.input_type == INPUT_TYPE_TEXTLESS:
            self.status.configure(text="글자 없는 이미지로 입력된 후보는 글자 제거가 필요 없습니다.")
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

    def adopt_clean_preview(self) -> None:
        state = self.current_state()
        if state is None or self.project is None or state.candidate is None:
            messagebox.showinfo("안내", "먼저 프로젝트 후보를 선택해주세요.")
            return
        if state.candidate.generation_status == "succeeded":
            try:
                working_path = adopt_generated_candidate(self.project, state.candidate)
            except ProjectAssetError as exc:
                messagebox.showerror("생성 후보 채택 실패", str(exc))
                return
            state.job.source = working_path
            state.job.auto_remove_text = False
            state.job.manual_boxes = ()
            state.job.ocr_boxes = ()
            self.update_candidate_from_state(state)
            self.mark_project_dirty()
            self.refresh_image_table()
            self.draw_preview()
            self.status.configure(text=f"생성 후보를 작업 원본으로 채택했습니다.\n{working_path}")
            return
        if state.candidate.input_type == INPUT_TYPE_TEXTLESS:
            messagebox.showinfo("안내", "글자 없는 이미지 입력은 이미 작업 원본으로 채택되어 있습니다.")
            return
        if state.clean_preview is None or not state.candidate.removal_preview_path:
            messagebox.showinfo("안내", "먼저 글자 제거 미리보기를 생성하고 확인해주세요.")
            return
        try:
            working_path = adopt_removal_preview(self.project, state.candidate)
        except ProjectAssetError as exc:
            messagebox.showerror("작업 원본 채택 실패", str(exc))
            return
        state.job.source = working_path
        state.job.auto_remove_text = False
        state.job.manual_boxes = ()
        state.job.ocr_boxes = ()
        state.thumbnail_photo = self.make_thumbnail_photo(state.clean_preview)
        state.job.status = "대기"
        self.update_candidate_from_state(state)
        self.mark_project_dirty()
        self.refresh_image_table()
        self.draw_preview()
        self.status.configure(text=f"작업 원본으로 채택했습니다.\n{working_path}")

    def clear_boxes(self) -> None:
        state = self.current_state()
        if state is None:
            return
        state.job.manual_boxes = ()
        state.job.ocr_boxes = ()
        state.clean_preview = None
        state.job.status = "대기"
        self.update_candidate_from_state(state)
        self.mark_project_dirty()
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
            self.update_candidate_from_state(state)
            self.mark_project_dirty()
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
                self.update_candidate_from_state(state)
                self.mark_project_dirty()
                self.status.configure(text=f"{state.job.source.name}\n글자 영역 {len(state.job.ocr_boxes)}개 탐지")
                self.refresh_image_table()
                self.draw_preview()
        elif event_type == "preview_done":
            index = event.get("index")
            if isinstance(index, int) and 0 <= index < len(self.image_states):
                state = self.image_states[index]
                state.clean_preview = event["image"]
                state.job.status = "대기"
                quality = "미기록"
                if self.project is not None and state.candidate is not None:
                    try:
                        quality = save_removal_preview(
                            self.project,
                            state.candidate,
                            state.clean_preview,
                            str(event["engine"]),
                            expected_size=state.original.size,
                        )
                        self.mark_project_dirty()
                    except ProjectAssetError as exc:
                        quality = f"저장 실패: {exc}"
                        write_exception(self.root_dir, "Removal preview save", exc)
                    state.thumbnail_photo = self.make_thumbnail_photo(state.clean_preview)
                self.preview_mode.set("글자 제거 결과")
                self.status.configure(
                    text=f"미리보기 제거 완료: {event['engine']}\n자동 품질 상태: {quality}\n확인 후 작업 원본으로 사용을 눌러주세요."
                )
                self.refresh_image_table()
                self.draw_preview()
        elif event_type == "pipeline_progress":
            self.handle_pipeline_progress(event["payload"], event.get("row_indices"))
        elif event_type == "pipeline_done":
            self.handle_pipeline_done(event["results"], event["output_dir"], event.get("row_indices"))
        elif event_type == "generation_progress":
            self.handle_generation_progress(event["payload"])
        elif event_type == "generation_done":
            self.handle_generation_done(event["result"])
        elif event_type == "generation_model_ready":
            self.generation_model_var.set(event["path"])
            self.status.configure(text=f"SDXL 모델 준비 완료: {event['path']}")
        elif event_type == "worker_error":
            self.status.configure(text=f"오류: {event['error']}\n수동 마스크나 로그를 확인해주세요.")
            messagebox.showerror("CoverMorph 오류", event["error"])
        elif event_type == "worker_finished":
            self.worker_thread = None
            self.set_busy(False)

    def handle_pipeline_progress(self, payload: dict[str, Any], row_indices: list[int] | None = None) -> None:
        event_type = payload.get("type")
        if payload.get("filename"):
            self.top_current_label.configure(text=f"현재 처리: {payload['filename']}")
        if event_type == "status":
            self.status.configure(text=payload.get("message", "작업 중..."))
            return
        index = payload.get("index")
        state = None
        if isinstance(index, int) and 1 <= index <= len(self.image_states):
            state_index = index - 1
            if row_indices is not None and 0 <= state_index < len(row_indices):
                state_index = row_indices[state_index]
            if 0 <= state_index < len(self.image_states):
                state = self.image_states[state_index]
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
                candidate = self.image_states[row_index].candidate
                if self.project is not None and candidate is not None:
                    candidate.generated_paths = {
                        key: path_to_project_string(self.project, Path(value))
                        for key, value in result.metadata.get("output_files", {}).items()
                    }
                    self.update_candidate_from_state(self.image_states[row_index])
                    self.mark_project_dirty()
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
            message += "\n\n엔진 오류/처리 기록:\n" + "\n".join(fallback_lines[:8])
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
        environment = detect_generation_environment(self.root_dir, self.generation_model_var.get().strip() or DEFAULT_SDXL_MODEL)
        lines.extend(
            [
                f"SDXL 모델 준비: {'완료' if environment.get('model_ready') else '미완료'}",
                f"IP-Adapter 준비: {'완료' if environment.get('ip_adapter_ready') else '미완료'}",
            ]
        )
        self.ai_status.configure(text="\n".join(lines))
        if self.generation_status_label is not None:
            self.generation_status_label.configure(
                text=(
                    f"SDXL 모델: {'준비됨' if environment.get('model_ready') else '미준비'} | "
                    f"IP-Adapter: {'준비됨' if environment.get('ip_adapter_ready') else '미준비'}\n"
                    f"생성 환경: {environment.get('status')} | GPU: {environment.get('gpu') or '없음'}"
                )
            )

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

    def selected_row_indices(self) -> list[int]:
        return [index for index, state in enumerate(self.image_states) if self.state_is_selected_for_run(state)]

    def jobs_for_run(self, row_indices: list[int] | None = None) -> list[ImageJob]:
        if row_indices is None:
            row_indices = list(range(len(self.image_states)))
        jobs: list[ImageJob] = []
        for row_index in row_indices:
            state = self.image_states[row_index]
            job = state.job
            source = job.source
            auto_remove_text = job.auto_remove_text
            manual_boxes = tuple(job.manual_boxes)
            ocr_boxes = tuple(job.ocr_boxes)
            if self.project is not None and state.candidate is not None:
                working = self.candidate_working_path(state.candidate)
                if working is not None:
                    source = working
                auto_remove_text = False
                manual_boxes = ()
                ocr_boxes = ()
            jobs.append(
                ImageJob(
                    source=source,
                    out_square=job.out_square,
                    out_thumb=job.out_thumb,
                    out_shorts=job.out_shorts,
                    auto_remove_text=auto_remove_text,
                    manual_boxes=manual_boxes,
                    ocr_boxes=ocr_boxes,
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

    def validate_states_ready_for_run(self, row_indices: list[int]) -> bool:
        missing: list[str] = []
        unapproved: list[str] = []
        for row_index in row_indices:
            state = self.image_states[row_index]
            candidate = state.candidate
            if candidate is None:
                continue
            if not candidate.working_source_path or not candidate.working_source_approved:
                unapproved.append(candidate.display_name)
                continue
            working = self.candidate_working_path(candidate)
            if working is None or not working.is_file():
                missing.append(f"{candidate.display_name}: {candidate.working_source_path}")
        if unapproved:
            messagebox.showwarning(
                "작업 원본 필요",
                "기존 커버는 글자 제거 결과를 미리보기로 확인한 뒤 작업 원본으로 채택해야 합니다.\n"
                + "\n".join(unapproved[:8]),
            )
            return False
        if missing:
            messagebox.showwarning("프로젝트 이미지 누락", "\n".join(missing[:8]))
            return False
        return True

    def run_single_image(self, state: ImageRowState) -> None:
        if self.worker_thread is not None and self.worker_thread.is_alive():
            messagebox.showwarning("작업 중", "현재 작업이 끝난 뒤 개별 변환을 실행해주세요.")
            return
        if count_selected_outputs_for_jobs([state.job]) == 0:
            messagebox.showinfo("안내", "이 행에서 생성할 출력 규격을 하나 이상 선택해주세요.")
            return
        row_index = self.image_states.index(state)
        if not self.validate_states_ready_for_run([row_index]):
            return
        output_dir = self.validate_output_directory_for_run()
        if output_dir is None:
            return
        state.job.status = "처리 중"
        state.job.error = ""
        self.refresh_image_table()
        options = self.pipeline_options(output_dir)
        job = self.jobs_for_run([row_index])[0]
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
                    {"type": "pipeline_progress", "payload": payload, "row_indices": [row_index]}
                ),
                cancel_event=self.cancel_event,
            )
            self.worker_queue.put(
                {"type": "pipeline_done", "results": results, "output_dir": options.output_dir,
                 "row_indices": [row_index]}
            )

        self.start_worker("Single image", worker)

    def open_row_folder(self, state: ImageRowState) -> None:
        path = state.result.item_dir if state.result is not None else self.output_dir_for_state(state)
        self.safe_open_folder(path)

    def output_dir_for_state(self, state: ImageRowState) -> Path:
        root = self.output_path_var.get().strip()
        if self.project is not None:
            return Path(root) / state.job.source.stem if root else self.project.project_dir / "outputs" / state.job.source.stem
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
        if self.selected_index is None or self.selected_index < 0 or self.selected_index >= len(self.image_states):
            messagebox.showinfo("안내", "먼저 목록에서 이미지를 선택해주세요.")
            return
        self.run_single_image(self.image_states[self.selected_index])

    def run_pipeline(self) -> None:
        if not self.image_states:
            messagebox.showinfo("안내", "먼저 커버 이미지를 추가해주세요.")
            return
        row_indices = self.selected_row_indices()
        if not row_indices:
            messagebox.showinfo("안내", "변환할 후보를 하나 이상 선택해주세요.")
            return
        if not self.validate_states_ready_for_run(row_indices):
            return
        jobs = self.jobs_for_run(row_indices)
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
                    {"type": "pipeline_progress", "payload": payload, "row_indices": row_indices}
                ),
                cancel_event=self.cancel_event,
            )
            self.worker_queue.put(
                {
                    "type": "pipeline_done",
                    "results": results,
                    "output_dir": options.output_dir,
                    "row_indices": row_indices,
                }
            )

        self.start_worker("Pipeline", worker)

    def cancel_work(self) -> None:
        self.cancel_event.set()
        self.status.configure(text="현재 단계가 끝나면 작업을 취소합니다.")
