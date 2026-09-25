from __future__ import annotations

import copy
import json
import queue
import shutil
import threading
import time
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Any

import customtkinter as ctk
from PIL import Image

from .cover_studio import (
    TEMPLATES,
    CoverEdit,
    apply_template,
    candidate_edit,
    export_cover,
    quick_project_dir,
    render_cover,
    save_candidate_edit,
    system_font_paths,
)
from .generation import (
    DEFAULT_SDXL_MODEL,
    GenerationConfig,
    GenerationResult,
    SDXLTextToImageEngine,
    generate_scene_candidates,
    resolve_sdxl_model_path,
    retry_failed_candidates,
)
from .logger import write_exception, write_log
from .planning import (
    LlamaCppCliBackend,
    PlanningCancelled,
    PlanningError,
    make_input_snapshot,
)
from .project import (
    INPUT_TYPE_TEXTLESS,
    CandidateRecord,
    InputRecord,
    SceneCard,
    add_candidate_from_file,
    add_input_records,
    add_person,
    add_person_reference,
    create_project,
    load_generation_presets,
    load_project,
    new_id,
    parse_input_file_detailed,
    resolve_project_path,
    save_project_atomic,
)
from .quick_planning import generate_plans_for_quick_cover
from .settings import ensure_output_directory, load_settings, open_folder, save_settings
from .task_state import TaskState

DIRECT_MODE = "이미지 프롬프트 직접 사용"
LYRICS_MODE = "가사/주제로 만들기"


class QuickCoverApp(ctk.CTk):
    """One-root CoverMorph workspace for cover creation and existing-image work."""

    def __init__(self, root_dir: Path) -> None:
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        super().__init__()
        self.root_dir = root_dir
        self.settings = load_settings(root_dir)
        project_file = quick_project_dir(root_dir) / "covermorph_project.json"
        self.project = (
            load_project(project_file)
            if project_file.is_file()
            else create_project(project_file.parent, "자동 저장 간편 커버")
        )
        stored_task = self.project.input_materials.get("task_state", "")
        try:
            self.task = TaskState.restore(json.loads(stored_task))
        except (json.JSONDecodeError, TypeError):
            self.task = TaskState()
        self.records = [item for item in self.project.inputs if item.selected]
        self.selected: CandidateRecord | None = None
        self.scene: SceneCard | None = None
        self.config: GenerationConfig | None = None
        self.engine: SDXLTextToImageEngine | None = None
        self.events: queue.Queue[tuple[str, str, Any]] = queue.Queue()
        self.cancel_event = threading.Event()
        self.worker: threading.Thread | None = None
        self._worker_lock = threading.Lock()
        self.preview_photo: ctk.CTkImage | None = None
        self.thumb_photos: list[ctk.CTkImage] = []
        self._saving = False
        self._advanced_visible = False
        self._progress_indeterminate = False
        self._init_vars()
        self._build()
        self._refresh_candidates()
        self._render_task()
        self._poll_after = self.after(100, self._poll)
        self.protocol("WM_DELETE_WINDOW", self._close)

    def _init_vars(self) -> None:
        material, text = self.project.input_materials, self.project.cover_text
        self.work_type = ctk.StringVar(value=material.get("work_type", "새 커버 만들기"))
        self.mode = ctk.StringVar(value=material.get("quick_mode", LYRICS_MODE))
        self.count = ctk.StringVar(value=material.get("candidate_count", "1"))
        self.preset = ctk.StringVar(value=material.get("preset", "선택 안 함"))
        self.reference = ctk.StringVar(value=material.get("reference_path", ""))
        self.title_var = ctk.StringVar(value=text.get("title", ""))
        self.subtitle_var = ctk.StringVar(value=text.get("subtitle", ""))
        self.label_var = ctk.StringVar(value=text.get("label", ""))
        self.template = ctk.StringVar(value=text.get("template", TEMPLATES[0]))
        self.size = ctk.IntVar(value=int(text.get("font_size", 118)))
        self.color = ctk.StringVar(value=text.get("color", "#FFFFFF"))
        self.align = ctk.StringVar(value=text.get("align", "center"))
        self.x = ctk.DoubleVar(value=float(text.get("x", 0.5)))
        self.y = ctk.DoubleVar(value=float(text.get("y", 0.13)))
        self.show_title = ctk.BooleanVar(value=text.get("show_title", "1") != "0")
        self.show_subtitle = ctk.BooleanVar(value=text.get("show_subtitle", "1") != "0")
        self.show_label = ctk.BooleanVar(value=text.get("show_label", "1") != "0")
        self.stroke = ctk.BooleanVar(value=text.get("stroke", "1") != "0")
        self.shadow = ctk.BooleanVar(value=text.get("shadow", "1") != "0")
        self.output = ctk.StringVar(value=self.settings.get("output_directory", ""))
        self.stage_var = ctk.StringVar()
        self.description_var = ctk.StringVar()
        self.metrics_var = ctk.StringVar()
        self.file_var = ctk.StringVar(value=material.get("source_display", "자료를 입력해 주세요."))
        self.full_path_var = ctk.StringVar(value=material.get("source_path", ""))

    def _build(self) -> None:
        self.title("CoverMorph Studio")
        self.geometry("1366x768")
        self.minsize(1080, 680)
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=0)
        self.grid_rowconfigure(2, weight=1)

        top = ctk.CTkFrame(self, border_width=1, border_color="#2563eb")
        top.grid(row=0, column=0, columnspan=3, sticky="ew", padx=8, pady=(8, 4))
        top.grid_columnconfigure(2, weight=1)
        ctk.CTkSegmentedButton(
            top,
            values=["새 커버 만들기", "기존 이미지 변환"],
            variable=self.work_type,
            command=self._work_type_changed,
        ).grid(row=0, column=0, rowspan=2, padx=8, pady=8)
        ctk.CTkLabel(
            top, textvariable=self.stage_var, font=ctk.CTkFont(size=16, weight="bold"), width=100
        ).grid(row=0, column=1, padx=8, sticky="w")
        ctk.CTkLabel(top, textvariable=self.description_var, anchor="w").grid(
            row=0, column=2, padx=5, sticky="ew"
        )
        ctk.CTkLabel(top, textvariable=self.metrics_var, anchor="w", text_color="#bfdbfe").grid(
            row=1, column=1, columnspan=2, padx=8, sticky="ew"
        )
        buttons = ctk.CTkFrame(top, fg_color="transparent")
        buttons.grid(row=0, column=3, rowspan=2, padx=6, pady=6)
        self.generate_button = ctk.CTkButton(buttons, text="이미지 만들기", command=self.generate, width=105)
        self.generate_button.pack(side="left", padx=2)
        self.cancel_button = ctk.CTkButton(
            buttons, text="작업 중단", command=self.cancel, width=90, fg_color="#991b1b"
        )
        self.cancel_button.pack(side="left", padx=2)
        self.retry_button = ctk.CTkButton(buttons, text="남은 작업 재시작", command=self.retry, width=125)
        self.retry_button.pack(side="left", padx=2)
        self.detail_button = ctk.CTkButton(
            buttons, text="오류 상세/로그", command=self.show_error_detail, width=105
        )
        self.detail_button.pack(side="left", padx=2)

        self.progress_bar = ctk.CTkProgressBar(self, mode="determinate")
        self.progress_bar.grid(row=1, column=0, columnspan=3, sticky="ew", padx=10, pady=(0, 5))

        left = ctk.CTkScrollableFrame(self, width=345)
        left.grid(row=2, column=0, sticky="nsew", padx=(8, 4), pady=(0, 8))
        center = ctk.CTkFrame(self)
        center.grid(row=2, column=1, sticky="nsew", padx=4, pady=(0, 8))
        center.grid_columnconfigure(0, weight=1)
        center.grid_rowconfigure(1, weight=1)
        right = ctk.CTkScrollableFrame(self, width=340)
        right.grid(row=2, column=2, sticky="nsew", padx=(4, 8), pady=(0, 8))
        self._build_source(left)
        self._build_advanced(left)
        self._build_center(center)
        self._build_editor(right)

    def _build_source(self, parent: Any) -> None:
        frame = self._section(parent, "자료와 생성 설정")
        row = ctk.CTkFrame(frame, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=3)
        ctk.CTkButton(row, text="JSON/TXT 불러오기", command=self.load_material).pack(
            side="left", fill="x", expand=True, padx=(0, 2)
        )
        ctk.CTkButton(row, text="이미지 불러오기", command=self.load_existing_image).pack(
            side="left", fill="x", expand=True, padx=(2, 0)
        )
        ctk.CTkLabel(
            frame, textvariable=self.file_var, anchor="w", text_color="#e5e7eb", wraplength=310
        ).pack(fill="x", padx=9, pady=2)
        ctk.CTkButton(
            frame, text="전체 경로 확인", command=self.show_source_path, height=25, fg_color="#475569"
        ).pack(fill="x", padx=8, pady=2)
        self.input_box = ctk.CTkTextbox(frame, height=150)
        self.input_box.pack(fill="x", padx=8, pady=4)
        self.input_box.insert("1.0", self.project.input_materials.get("quick_input", ""))
        ctk.CTkSegmentedButton(frame, values=[LYRICS_MODE, DIRECT_MODE], variable=self.mode).pack(
            fill="x", padx=8, pady=4
        )
        self.presets = load_generation_presets(self.root_dir / "config" / "channel_generation_presets.json")
        names = ["선택 안 함"] + [item.name for item in self.presets]
        if self.preset.get() not in names:
            self.preset.set("선택 안 함")
        ctk.CTkOptionMenu(frame, values=names, variable=self.preset).pack(fill="x", padx=8, pady=3)
        row = ctk.CTkFrame(frame, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=3)
        ctk.CTkEntry(row, textvariable=self.reference, placeholder_text="참고 이미지 한 장 (기본 꺼짐)").pack(
            side="left", fill="x", expand=True
        )
        ctk.CTkButton(row, text="찾기", command=self.choose_reference, width=55).pack(
            side="left", padx=(4, 0)
        )
        ctk.CTkSegmentedButton(frame, values=["1", "4", "5"], variable=self.count).pack(
            fill="x", padx=8, pady=(3, 8)
        )
        ctk.CTkLabel(frame, text="입력 요약 / 실제 생성 프롬프트", anchor="w", text_color="#bfdbfe").pack(
            fill="x", padx=8
        )
        self.prompt_box = ctk.CTkTextbox(frame, height=75)
        self.prompt_box.pack(fill="x", padx=8, pady=(2, 8))
        self.prompt_box.insert("1.0", self.project.input_materials.get("actual_prompt", ""))

    def _build_advanced(self, parent: Any) -> None:
        self.advanced_button = ctk.CTkButton(
            parent, text="고급 설정 펼치기", command=self.toggle_advanced, fg_color="#334155"
        )
        self.advanced_button.pack(fill="x", padx=5, pady=5)
        self.advanced_frame = ctk.CTkFrame(parent)
        ctk.CTkLabel(self.advanced_frame, text="기존 프로젝트 호환", font=ctk.CTkFont(weight="bold")).pack(
            anchor="w", padx=8, pady=(8, 3)
        )
        ctk.CTkButton(self.advanced_frame, text="기존 프로젝트 열기", command=self.open_project).pack(
            fill="x", padx=8, pady=2
        )
        ctk.CTkButton(self.advanced_frame, text="현재 프로젝트 저장", command=self.save_project_as).pack(
            fill="x", padx=8, pady=2
        )
        ctk.CTkLabel(
            self.advanced_frame,
            text="기존 커버 글자 제거·AI 확장\n시험 기능: 결과 확인 필요",
            text_color="#fbbf24",
            justify="left",
        ).pack(fill="x", padx=8, pady=7)
        ctk.CTkLabel(
            self.advanced_frame,
            text="작업 종류를 바꿔도 현재 입력·후보·문구·출력 폴더는 유지됩니다.",
            wraplength=310,
            justify="left",
        ).pack(fill="x", padx=8, pady=(0, 8))

    def _build_center(self, parent: Any) -> None:
        ctk.CTkLabel(parent, text="후보 선택과 미리보기", font=ctk.CTkFont(weight="bold"), anchor="w").grid(
            row=0, column=0, sticky="ew", padx=10, pady=7
        )
        self.preview = ctk.CTkLabel(parent, text="생성된 후보가 없습니다")
        self.preview.grid(row=1, column=0, sticky="nsew", padx=8, pady=4)
        self.thumbs = ctk.CTkScrollableFrame(parent, orientation="horizontal", height=130)
        self.thumbs.grid(row=2, column=0, sticky="ew", padx=8, pady=8)

    def _build_editor(self, parent: Any) -> None:
        frame = self._section(parent, "문구 편집과 저장")
        for label, variable, shown in (
            ("제목", self.title_var, self.show_title),
            ("부제", self.subtitle_var, self.show_subtitle),
            ("채널명/음반사", self.label_var, self.show_label),
        ):
            row = ctk.CTkFrame(frame, fg_color="transparent")
            row.pack(fill="x", padx=8, pady=2)
            ctk.CTkCheckBox(row, text=label, variable=shown, width=115, command=self.changed).pack(
                side="left"
            )
            ctk.CTkEntry(row, textvariable=variable).pack(side="left", fill="x", expand=True)
        ctk.CTkOptionMenu(
            frame, values=list(TEMPLATES), variable=self.template, command=self.template_changed
        ).pack(fill="x", padx=8, pady=3)
        self.fonts = {path.name: str(path) for path in system_font_paths()}
        font_values = list(self.fonts) or ["지원 폰트 없음"]
        stored = Path(self.project.cover_text.get("font_path", "")).name
        self.font = ctk.StringVar(value=stored if stored in self.fonts else font_values[0])
        ctk.CTkOptionMenu(
            frame, values=font_values, variable=self.font, command=lambda _v: self.changed()
        ).pack(fill="x", padx=8, pady=3)
        row = ctk.CTkFrame(frame, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=3)
        ctk.CTkLabel(row, text="제목 크기").pack(side="left")
        ctk.CTkEntry(row, textvariable=self.size, width=60).pack(side="left", padx=4)
        ctk.CTkLabel(row, text="색상").pack(side="left", padx=(7, 0))
        ctk.CTkEntry(row, textvariable=self.color, width=85).pack(side="left", padx=4)
        ctk.CTkOptionMenu(
            row,
            values=["left", "center", "right"],
            variable=self.align,
            width=90,
            command=lambda _v: self.changed(),
        ).pack(side="right")
        for label, variable in (("가로 위치", self.x), ("세로 위치", self.y)):
            ctk.CTkLabel(frame, text=label, anchor="w").pack(fill="x", padx=8)
            ctk.CTkSlider(
                frame, from_=0.05, to=0.95, variable=variable, command=lambda _v: self.changed()
            ).pack(fill="x", padx=8, pady=2)
        row = ctk.CTkFrame(frame, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=3)
        ctk.CTkCheckBox(row, text="테두리", variable=self.stroke, command=self.changed).pack(side="left")
        ctk.CTkCheckBox(row, text="그림자", variable=self.shadow, command=self.changed).pack(
            side="left", padx=8
        )
        row = ctk.CTkFrame(frame, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=3)
        ctk.CTkEntry(row, textvariable=self.output, placeholder_text="출력 폴더").pack(
            side="left", fill="x", expand=True
        )
        ctk.CTkButton(row, text="선택", command=self.choose_output, width=55).pack(side="left", padx=4)
        ctk.CTkButton(
            frame, text="커버 저장 (1400×1400 JPG + 원본 PNG)", command=self.save_cover, height=38
        ).pack(fill="x", padx=8, pady=3)
        ctk.CTkButton(frame, text="글자 없는 배경 저장", command=self.save_textless).pack(
            fill="x", padx=8, pady=3
        )
        ctk.CTkButton(frame, text="출력 폴더 열기", command=self.open_output).pack(
            fill="x", padx=8, pady=(3, 8)
        )
        for variable in (self.title_var, self.subtitle_var, self.label_var, self.size, self.color):
            variable.trace_add("write", lambda *_args: self.changed())
        self.input_box.bind("<KeyRelease>", lambda _event: self.after(350, self.autosave))

    @staticmethod
    def _section(parent: Any, title: str) -> ctk.CTkFrame:
        frame = ctk.CTkFrame(parent, border_width=1, border_color="#334155")
        frame.pack(fill="x", padx=4, pady=5)
        ctk.CTkLabel(frame, text=title, font=ctk.CTkFont(weight="bold"), anchor="w").pack(
            fill="x", padx=8, pady=(7, 3)
        )
        return frame

    def toggle_advanced(self) -> None:
        self._advanced_visible = not self._advanced_visible
        if self._advanced_visible:
            self.advanced_frame.pack(fill="x", padx=5, pady=(0, 5))
            self.advanced_button.configure(text="고급 설정 접기")
        else:
            self.advanced_frame.pack_forget()
            self.advanced_button.configure(text="고급 설정 펼치기")

    def _work_type_changed(self, _value: str = "") -> None:
        self.autosave()
        self.description_var.set(f"{self.work_type.get()} 모드입니다. 현재 작업과 편집 내용은 유지됩니다.")

    def autosave(self) -> None:
        if self._saving:
            return
        self._saving = True
        try:
            self.project.input_materials.update(
                {
                    "quick_input": self.input_box.get("1.0", "end").strip(),
                    "quick_mode": self.mode.get(),
                    "candidate_count": self.count.get(),
                    "preset": self.preset.get(),
                    "reference_path": self.reference.get(),
                    "actual_prompt": self.prompt_box.get("1.0", "end").strip(),
                    "work_type": self.work_type.get(),
                    "source_display": self.file_var.get(),
                    "source_path": self.full_path_var.get(),
                    "task_state": json.dumps(self.task.to_dict(), ensure_ascii=False),
                }
            )
            self.project.cover_text.update(
                {
                    "title": self.title_var.get(),
                    "subtitle": self.subtitle_var.get(),
                    "label": self.label_var.get(),
                    "template": self.template.get(),
                    "font_path": self.fonts.get(self.font.get(), ""),
                    "font_size": str(self.size.get()),
                    "color": self.color.get(),
                    "align": self.align.get(),
                    "x": str(self.x.get()),
                    "y": str(self.y.get()),
                    "stroke": str(int(self.stroke.get())),
                    "shadow": str(int(self.shadow.get())),
                    "show_title": str(int(self.show_title.get())),
                    "show_subtitle": str(int(self.show_subtitle.get())),
                    "show_label": str(int(self.show_label.get())),
                }
            )
            if self.selected:
                save_candidate_edit(self.project, self.selected, self.current_edit())
            else:
                save_project_atomic(self.project)
        finally:
            self._saving = False

    @staticmethod
    def _short_path(path: Path, limit: int = 48) -> str:
        text = str(path)
        return text if len(text) <= limit else f"…{text[-limit:]}"

    def load_material(self) -> None:
        filename = filedialog.askopenfilename(
            title="JSON/TXT 자료 불러오기", filetypes=[("JSON/TXT", "*.json *.txt"), ("모든 파일", "*.*")]
        )
        if not filename:
            return
        before = self.input_box.get("1.0", "end")
        self.stage_var.set("자료 읽기")
        self.description_var.set("원본 파일을 읽고 곡 구조를 확인하고 있습니다.")
        try:
            parsed = parse_input_file_detailed(Path(filename))
            if parsed.get("needs_mapping"):
                raise ValueError("JSON 필드를 확실히 매핑할 수 없습니다. 필드 구조를 확인해 주세요.")
            records = list(parsed["records"])
            if len(records) > 1 and not self._select_records(records):
                return
            add_input_records(self.project, records)
            self.records = [item for item in records if item.selected]
            material = "\n\n".join(
                item.lyrics or item.theme_mood or item.image_prompt for item in self.records
            )
            self.input_box.delete("1.0", "end")
            self.input_box.insert("1.0", material)
            mapped = parsed.get("cover_text") or {}
            if mapped.get("title"):
                self.title_var.set(str(mapped["title"]))
            if mapped.get("subtitle"):
                self.subtitle_var.set(str(mapped["subtitle"]))
            if mapped.get("label"):
                self.label_var.set(str(mapped["label"]))
            path = Path(filename)
            self.full_path_var.set(str(path))
            self.file_var.set(f"{self._short_path(path)} · 선택 {len(self.records)}/{len(records)}곡")
            self.stage_var.set("대기")
            self.description_var.set("자료를 읽었습니다. 이미지 만들기를 누르세요.")
            self.autosave()
        except Exception as exc:
            self.input_box.delete("1.0", "end")
            self.input_box.insert("1.0", before)
            self._record_immediate_error("JSON/TXT 읽기 실패", exc)

    def _select_records(self, records: list[InputRecord]) -> bool:
        dialog = ctk.CTkToplevel(self)
        dialog.title("앨범에 사용할 곡 선택")
        dialog.geometry("500x480")
        dialog.transient(self)
        dialog.grab_set()
        variables = []
        scroll = ctk.CTkScrollableFrame(dialog)
        scroll.pack(fill="both", expand=True, padx=12, pady=12)
        for item in records:
            variable = ctk.BooleanVar(value=True)
            variables.append(variable)
            ctk.CTkCheckBox(scroll, text=item.title or "제목 없음", variable=variable).pack(
                anchor="w", padx=8, pady=4
            )
        accepted = {"value": False}

        def done() -> None:
            for record, variable in zip(records, variables, strict=True):
                record.selected = variable.get()
            accepted["value"] = any(item.selected for item in records)
            dialog.destroy()

        ctk.CTkButton(dialog, text="선택 완료", command=done).pack(pady=10)
        self.wait_window(dialog)
        return accepted["value"]

    def load_existing_image(self) -> None:
        filename = filedialog.askopenfilename(
            title="기존 이미지 불러오기", filetypes=[("이미지", "*.png *.jpg *.jpeg *.webp *.bmp")]
        )
        if not filename:
            return
        try:
            candidate = add_candidate_from_file(self.project, Path(filename), INPUT_TYPE_TEXTLESS)
            candidate.generation_status = "succeeded"
            candidate.quality_status = "existing_image_user_selected"
            self.work_type.set("기존 이미지 변환")
            self.full_path_var.set(filename)
            self.file_var.set(self._short_path(Path(filename)))
            save_project_atomic(self.project)
            self._refresh_candidates()
            self.select_candidate(candidate)
            self.autosave()
        except Exception as exc:
            self._record_immediate_error("기존 이미지 읽기 실패", exc)

    def choose_reference(self) -> None:
        path = filedialog.askopenfilename(
            title="참고 이미지", filetypes=[("이미지", "*.png *.jpg *.jpeg *.webp")]
        )
        if path:
            self.reference.set(path)
            self.autosave()

    def choose_output(self) -> None:
        path = filedialog.askdirectory(title="커버 출력 폴더")
        if path:
            self.output.set(path)
            self.settings["output_directory"] = path
            save_settings(self.root_dir, self.settings)

    def show_source_path(self) -> None:
        messagebox.showinfo("현재 자료 전체 경로", self.full_path_var.get() or "파일을 불러오지 않았습니다.")

    def _snapshot(self, source: str, count: int) -> dict[str, Any]:
        seed = int(time.time()) % 2_000_000_000
        return {
            "source": source,
            "mode": self.mode.get(),
            "count": count,
            "preset": self.preset.get(),
            "reference_path": self.reference.get(),
            "seed": seed,
            "model_id": str(resolve_sdxl_model_path(self.root_dir, DEFAULT_SDXL_MODEL)),
            "selected_input_ids": [item.input_id for item in self.records],
            "user_title": self.title_var.get(),
            "user_label": self.label_var.get(),
        }

    def generate(self) -> None:
        if not self._can_start():
            return
        source = self.input_box.get("1.0", "end").strip()
        if not source:
            messagebox.showwarning("입력 필요", "가사·주제 또는 이미지 프롬프트를 입력해 주세요.")
            return
        count = int(self.count.get())
        snapshot = self._snapshot(source, count)
        job_id = self.task.start(count, snapshot, len(self.records))
        self.cancel_event.clear()
        self.scene = None
        self.config = None
        self.autosave()
        self._start_thread(job_id, snapshot, None)

    def retry(self) -> None:
        if (
            not self.task.terminal
            or self.task.state not in {"실패", "중단됨"}
            or not self.task.input_snapshot
        ):
            return
        if not self._can_start(show_message=False):
            return
        snapshot = dict(self.task.input_snapshot)
        remaining = list(self.task.failed_indices)
        job_id = self.task.start(
            int(snapshot["count"]), snapshot, len(snapshot.get("selected_input_ids", []))
        )
        self.task.completed = int(snapshot["count"]) - len(remaining) if remaining else 0
        self.cancel_event.clear()
        self.autosave()
        self._start_thread(job_id, snapshot, remaining or None)

    def _can_start(self, show_message: bool = True) -> bool:
        with self._worker_lock:
            running = self.worker is not None and self.worker.is_alive()
        if running:
            if show_message:
                self.description_var.set("현재 작업의 실행과 정리가 끝난 뒤 다시 시도해 주세요.")
            return False
        return True

    def _start_thread(self, job_id: str, snapshot: dict[str, Any], retry_indices: list[int] | None) -> None:
        def work() -> None:
            try:
                self._run_job(job_id, snapshot, retry_indices)
            except Exception as exc:
                detail = traceback.format_exc()
                write_exception(self.root_dir, f"Quick cover {job_id}", exc)
                self.events.put((job_id, "error", (exc, detail)))

        with self._worker_lock:
            self.worker = threading.Thread(target=work, daemon=True, name=job_id)
            self.worker.start()
        self._render_task()

    def _run_job(self, job_id: str, snapshot: dict[str, Any], retry_indices: list[int] | None) -> None:
        if retry_indices and self.scene and self.config:
            engine = self._engine_for(str(snapshot["model_id"]))
            result = retry_failed_candidates(
                self.project,
                self.scene,
                engine,
                self.config,
                retry_indices,
                self.cancel_event,
                lambda data: self.events.put((job_id, "generation", data)),
            )
            self.events.put((job_id, "done", result))
            return
        prompt = str(snapshot.get("actual_prompt") or snapshot["source"])
        warning = ""
        if snapshot["mode"] == LYRICS_MODE and not snapshot.get("actual_prompt"):
            self.events.put(
                (
                    job_id,
                    "stage",
                    (
                        "가사 해석",
                        f"선택한 {max(1, len(self.records))}곡을 로컬 기획 모델로 해석하고 있습니다.",
                    ),
                )
            )
            preset = next((item for item in self.presets if item.name == snapshot["preset"]), self.presets[0])
            records = self.records or [InputRecord(new_id("direct"), "lyrics", "직접 입력", lyrics=prompt)]
            explicit = {
                "album_title": str(snapshot.get("user_title") or ""),
                "label": str(snapshot.get("user_label") or ""),
            }
            plan_input = make_input_snapshot(preset, records, prompt if not self.records else "", explicit, 4)
            backend = LlamaCppCliBackend(self.root_dir)
            plans = generate_plans_for_quick_cover(plan_input, backend, self.cancel_event)
            first = (plans.get("plans") or [{}])[0]
            prompt = str(first.get("image_prompt_en") or "").strip()
            if not prompt:
                raise PlanningError("로컬 기획이 유효한 이미지 프롬프트를 만들지 못했습니다.")
            warning = str((first.get("title") or {}).get("validation_warning") or "")
            self.events.put((job_id, "analysis_done", (len(records), warning)))
        if self.cancel_event.is_set():
            raise PlanningCancelled("가사 해석 후 이미지 생성을 시작하기 전에 중단했습니다.")
        self.events.put((job_id, "prompt", prompt))
        count = int(snapshot["count"])
        scene = SceneCard(
            new_id("quick_scene"),
            order=len(self.project.scenes) + 1,
            user_description=str(snapshot["source"]),
            prompt_user=prompt,
            negative_prompt_user="text, letters, typography, logo, watermark, distorted anatomy",
            output_ratio="1:1",
            candidate_count=count,
            prompt_confirmed=True,
            structured_request={
                "approval_source": "single_workspace",
                "textless_background": True,
                "original_input": snapshot["source"],
                "actual_generation_prompt": prompt,
                "prompt_mode": snapshot["mode"],
                "job_id": job_id,
                "planning_warning": warning,
            },
        )
        self.project.scenes.append(scene)
        reference_mode, reference_id = "off", ""
        reference_path = Path(str(snapshot.get("reference_path") or ""))
        if reference_path.is_file():
            person = add_person(self.project, "간편 참고 이미지")
            reference = add_person_reference(self.project, person, reference_path, "style")
            scene.reference_image_ids = [reference.image_id]
            scene.structured_request["reference_image_ids"] = [reference.image_id]
            reference_mode, reference_id = "style", reference.image_id
        config = GenerationConfig(
            model_id=str(snapshot["model_id"]),
            candidate_count=count,
            seed=int(snapshot["seed"]),
            local_files_only=True,
            reference_mode=reference_mode,
            reference_image_id=reference_id,
        )
        self.scene, self.config = scene, config
        engine = self._engine_for(config.model_id)
        result = generate_scene_candidates(
            self.project,
            scene,
            engine,
            config,
            self.cancel_event,
            lambda data: self.events.put((job_id, "generation", data)),
            retry_indices,
        )
        self.events.put((job_id, "done", result))

    def _engine_for(self, model_id: str) -> SDXLTextToImageEngine:
        if self.engine is None or self.engine.model_id != model_id:
            self.engine = SDXLTextToImageEngine(model_id, local_files_only=True)
        return self.engine

    def _poll(self) -> None:
        try:
            while True:
                job_id, kind, payload = self.events.get_nowait()
                if job_id != self.task.job_id:
                    continue
                if kind == "stage":
                    self.task.update(job_id, payload[0], payload[1])
                elif kind == "analysis_done":
                    count, warning = payload
                    text = warning or "가사 해석을 완료했습니다."
                    self.task.update(job_id, "모델 로딩", text, analyzed_songs=count)
                elif kind == "prompt":
                    self.prompt_box.delete("1.0", "end")
                    self.prompt_box.insert("1.0", payload)
                    self.task.input_snapshot["actual_prompt"] = payload
                elif kind == "generation":
                    self._generation_event(job_id, payload)
                elif kind == "done":
                    self._job_done(job_id, payload)
                elif kind == "error":
                    self._job_error(job_id, payload[0], payload[1])
        except queue.Empty:
            pass
        self._render_task()
        self._poll_after = self.after(100, self._poll)

    def _generation_event(self, job_id: str, data: dict[str, Any]) -> None:
        phase = data.get("phase")
        if phase == "model_loading":
            self.task.update(job_id, "모델 로딩", "SDXL 모델을 GPU에 올리고 있습니다.")
        elif phase == "inference":
            step, steps = int(data.get("step") or 0), int(data.get("steps") or 0)
            self.task.update(
                job_id,
                "이미지 생성",
                f"{self.task.current_candidate}번째 이미지 생성: {step}/{steps} step",
                step=step,
                steps=steps,
            )
        elif phase == "candidate_start":
            current = int(data.get("candidate") or 0)
            self.task.update(
                job_id,
                "이미지 생성",
                f"{current}번째 이미지를 처음부터 생성하고 있습니다.",
                current_candidate=current,
                step=0,
                steps=int(self.config.steps if self.config else 0),
            )
        elif phase == "candidate_done":
            complete = min(self.task.requested, self.task.completed + 1)
            inherited_edit = self.current_edit()
            self.task.update(
                job_id,
                "저장",
                f"후보 {complete}/{self.task.requested} 저장 및 등록 완료",
                completed=complete,
                step=self.task.steps,
            )
            self._refresh_candidates()
            candidate_id = str(data.get("candidate_id") or "")
            completed_candidate = next(
                (item for item in self.project.candidates if item.candidate_id == candidate_id), None
            )
            if completed_candidate is not None:
                save_candidate_edit(self.project, completed_candidate, inherited_edit)
                self.select_candidate(completed_candidate)
            self.autosave()

    def _job_done(self, job_id: str, result: GenerationResult) -> None:
        self._refresh_candidates()
        failed = list(result.failed_indices)
        total_completed = self.task.completed
        if result.cancelled:
            self.task.finish(
                job_id,
                "중단됨",
                f"작업을 중단했습니다. 완료 후보 {total_completed}장은 보존했습니다.",
                failed_indices=failed,
                completed=total_completed,
            )
        elif failed:
            self.task.finish(
                job_id,
                "실패",
                f"후보 {total_completed}장은 완료했고 {len(failed)}장은 실패했습니다.",
                failed_indices=failed,
                completed=total_completed,
                error_summary="일부 후보 생성 실패",
                error_detail="\n".join(result.errors),
            )
        else:
            self.task.finish(
                job_id,
                "완료",
                f"후보 {total_completed}/{self.task.requested} 저장 및 등록을 완료했습니다.",
                completed=total_completed,
                failed_indices=[],
            )
        self.autosave()

    def _job_error(self, job_id: str, exc: Exception, detail: str) -> None:
        category = self._error_category(exc)
        remaining = list(range(self.task.completed, self.task.requested))
        log_path = str(self.root_dir / "logs" / f"{time.strftime('%Y-%m-%d')}.log")
        self.task.finish(
            job_id,
            "중단됨" if isinstance(exc, PlanningCancelled) else "실패",
            f"{category}: {exc}",
            error_summary=category,
            error_detail=detail,
            log_path=log_path,
            failed_indices=remaining,
        )
        self.autosave()

    @staticmethod
    def _error_category(exc: Exception) -> str:
        text = str(exc).lower()
        if isinstance(exc, PlanningCancelled):
            return "가사 해석 중단"
        if isinstance(exc, PlanningError):
            return "가사 해석 또는 로컬 기획 실패"
        if "out of memory" in text or "cuda" in text and "memory" in text:
            return "GPU 메모리 부족"
        if "model load" in text or "pipeline" in text:
            return "SDXL 모델 로딩 실패"
        if "save" in text or "permission" in text:
            return "저장 실패"
        return "이미지 생성 실패"

    def _record_immediate_error(self, summary: str, exc: Exception) -> None:
        write_exception(self.root_dir, summary, exc)
        job_id = self.task.start(0, {})
        self.task.finish(
            job_id,
            "실패",
            f"{summary}: {exc}",
            error_summary=summary,
            error_detail=traceback.format_exc(),
            log_path=str(self.root_dir / "logs" / f"{time.strftime('%Y-%m-%d')}.log"),
        )
        self.autosave()
        self._render_task()

    def cancel(self) -> None:
        if self.task.request_cancel():
            self.cancel_event.set()
            self.autosave()
            self._render_task()

    def _render_task(self) -> None:
        now = time.monotonic()
        self.stage_var.set(self.task.state)
        idle = self.task.idle_seconds(now)
        description = self.task.description
        if self.task.active and idle >= 20:
            description = f"진행 응답 대기 · 마지막 단계: {self.task.state} · {description}"
        self.description_var.set(description)
        elapsed = self.task.elapsed(now)
        step_text = (
            f" · 현재 {self.task.current_candidate}번째 · {self.task.step}/{self.task.steps} step"
            if self.task.steps
            else ""
        )
        song_text = (
            f"선택 곡 {self.task.selected_songs} · 분석 완료 {self.task.analyzed_songs}"
            if self.task.selected_songs
            else "직접 입력"
        )
        self.metrics_var.set(
            f"{song_text} · 완료 후보 {self.task.completed}/{self.task.requested}{step_text} · 경과 {elapsed // 60:02d}:{elapsed % 60:02d} · 마지막 이벤트 {idle}초 전"
        )
        unknown_progress = self.task.active and self.task.state in {
            "자료 읽기",
            "가사 해석",
            "모델 로딩",
            "중단 요청",
        }
        if unknown_progress:
            self.progress_bar.configure(mode="indeterminate")
            if not self._progress_indeterminate:
                self.progress_bar.start()
                self._progress_indeterminate = True
        else:
            if self._progress_indeterminate:
                self.progress_bar.stop()
                self._progress_indeterminate = False
            self.progress_bar.configure(mode="determinate")
        if not unknown_progress and self.task.steps and self.task.current_candidate:
            fraction = (max(0, self.task.current_candidate - 1) + self.task.step / self.task.steps) / max(
                1, self.task.requested
            )
            self.progress_bar.set(min(1.0, fraction))
        elif not unknown_progress and self.task.requested:
            self.progress_bar.set(self.task.completed / self.task.requested)
        elif not unknown_progress:
            self.progress_bar.set(0)
        running = self.worker is not None and self.worker.is_alive()
        self.generate_button.configure(state="disabled" if running else "normal")
        self.cancel_button.configure(
            state="normal" if running and self.task.state != "중단 요청" else "disabled"
        )
        retryable = not running and self.task.state in {"실패", "중단됨"} and bool(self.task.input_snapshot)
        self.retry_button.configure(state="normal" if retryable else "disabled")
        self.detail_button.configure(
            state="normal" if self.task.error_detail or self.task.log_path else "disabled"
        )

    def show_error_detail(self) -> None:
        detail = self.task.error_detail or "원문 예외가 없습니다."
        messagebox.showinfo(
            "오류 상세/로그", f"{self.task.error_summary}\n\n{detail}\n\n로그: {self.task.log_path or '없음'}"
        )

    def _refresh_candidates(self) -> None:
        for widget in self.thumbs.winfo_children():
            widget.destroy()
        self.thumb_photos.clear()
        candidates = [item for item in self.project.candidates if item.generation_status == "succeeded"]
        for candidate in candidates:
            try:
                with Image.open(resolve_project_path(self.project, candidate.original_path)) as opened:
                    image = opened.convert("RGB")
                    image.thumbnail((115, 115))
                    photo = ctk.CTkImage(
                        light_image=image, dark_image=image, size=(image.width, image.height)
                    )
                self.thumb_photos.append(photo)
                ctk.CTkButton(
                    self.thumbs,
                    text="",
                    image=photo,
                    width=120,
                    height=120,
                    command=lambda item=candidate: self.select_candidate(item),
                ).pack(side="left", padx=3)
            except OSError:
                continue
        if not self.selected and candidates:
            self.select_candidate(
                next(
                    (item for item in candidates if item.candidate_id in self.project.selected_candidate_ids),
                    candidates[-1],
                )
            )

    def select_candidate(self, candidate: CandidateRecord) -> None:
        if self.selected:
            try:
                save_candidate_edit(self.project, self.selected, self.current_edit())
            except Exception:
                pass
        self.selected = candidate
        self._set_edit(candidate_edit(candidate))
        self.project.selected_candidate_ids = [candidate.candidate_id]
        save_project_atomic(self.project)
        self.draw_preview()

    def _set_edit(self, edit: CoverEdit) -> None:
        self._saving = True
        try:
            self.template.set(edit.template)
            self.title_var.set(edit.title.text)
            self.subtitle_var.set(edit.subtitle.text)
            self.label_var.set(edit.label.text)
            self.show_title.set(edit.title.visible)
            self.show_subtitle.set(edit.subtitle.visible)
            self.show_label.set(edit.label.visible)
            self.size.set(edit.title.size)
            self.color.set(edit.title.color)
            self.align.set(edit.title.align)
            self.x.set(edit.title.x)
            self.y.set(edit.title.y)
            self.stroke.set(edit.title.stroke_width > 0)
            self.shadow.set(edit.title.shadow)
            if Path(edit.title.font_path).name in self.fonts:
                self.font.set(Path(edit.title.font_path).name)
        finally:
            self._saving = False

    def current_edit(self) -> CoverEdit:
        edit = candidate_edit(self.selected) if self.selected else CoverEdit()
        edit.template = self.template.get()
        font = self.fonts.get(self.font.get(), "")
        for layer, text, shown in (
            (edit.title, self.title_var.get(), self.show_title.get()),
            (edit.subtitle, self.subtitle_var.get(), self.show_subtitle.get()),
            (edit.label, self.label_var.get(), self.show_label.get()),
        ):
            (
                layer.text,
                layer.visible,
                layer.font_path,
                layer.color,
                layer.align,
                layer.stroke_width,
                layer.shadow,
            ) = (
                text,
                shown,
                font,
                self.color.get(),
                self.align.get(),
                3 if self.stroke.get() else 0,
                self.shadow.get(),
            )
        edit.title.size, edit.title.x, edit.title.y = max(24, self.size.get()), self.x.get(), self.y.get()
        return edit

    def template_changed(self, value: str) -> None:
        edit = self.current_edit()
        apply_template(edit, value)
        self._set_edit(edit)
        self.changed()

    def changed(self) -> None:
        if not self._saving:
            self.draw_preview()
            self.after(300, self.autosave)

    def draw_preview(self) -> None:
        if not self.selected:
            return
        try:
            with Image.open(resolve_project_path(self.project, self.selected.original_path)) as opened:
                image, _ = render_cover(opened, self.current_edit())
            image.thumbnail((650, 540))
            self.preview_photo = ctk.CTkImage(
                light_image=image, dark_image=image, size=(image.width, image.height)
            )
            self.preview.configure(image=self.preview_photo, text="")
        except Exception as exc:
            self.preview.configure(image=None, text=f"미리보기 확인 필요\n{exc}")

    def save_cover(self) -> None:
        if not self.selected:
            messagebox.showwarning("후보 선택", "저장할 후보를 선택해 주세요.")
            return
        try:
            output = ensure_output_directory(self.output.get(), create=True)
            paths = export_cover(self.project, self.selected, self.current_edit(), output)
            self.description_var.set(f"저장 완료: {paths['cover_jpg']}")
        except Exception as exc:
            self._record_immediate_error("커버 저장 실패", exc)

    def save_textless(self) -> None:
        if not self.selected:
            return
        try:
            from .cover_studio import copy_textless

            path = copy_textless(
                self.project, self.selected, ensure_output_directory(self.output.get(), create=True)
            )
            self.description_var.set(f"글자 없는 배경 저장 완료: {path}")
        except Exception as exc:
            self._record_immediate_error("배경 저장 실패", exc)

    def open_output(self) -> None:
        try:
            open_folder(ensure_output_directory(self.output.get()))
        except Exception as exc:
            self._record_immediate_error("출력 폴더 열기 실패", exc)

    def open_project(self) -> None:
        filename = filedialog.askopenfilename(
            title="CoverMorph 프로젝트 열기",
            filetypes=[("CoverMorph 프로젝트", "covermorph_project.json"), ("JSON", "*.json")],
        )
        if not filename:
            return
        if self.worker and self.worker.is_alive():
            messagebox.showwarning("작업 중", "현재 작업이 끝난 뒤 프로젝트를 열어 주세요.")
            return
        try:
            self.project = load_project(Path(filename))
            self.records = [item for item in self.project.inputs if item.selected]
            self.selected = None
            self._refresh_candidates()
            self.description_var.set("기존 프로젝트를 같은 작업 창에서 열었습니다.")
        except Exception as exc:
            self._record_immediate_error("프로젝트 열기 실패", exc)

    def save_project_as(self) -> None:
        directory = filedialog.askdirectory(title="프로젝트 사본 저장 폴더")
        if not directory:
            return
        try:
            self.autosave()
            target = create_project(Path(directory), self.project.name)
            source_assets = self.project.project_file.parent / "assets"
            if source_assets.is_dir():
                shutil.copytree(source_assets, target.project_file.parent / "assets", dirs_exist_ok=True)
            saved_project = copy.deepcopy(self.project)
            saved_project.project_file = target.project_file
            target = saved_project
            save_project_atomic(target)
            self.description_var.set(f"프로젝트 전체 사본 저장: {target.project_file}")
        except Exception as exc:
            self._record_immediate_error("프로젝트 저장 실패", exc)

    def _close(self) -> None:
        if self.task.active:
            self.cancel_event.set()
            self.task.finish(
                self.task.job_id,
                "중단됨",
                "앱 종료로 작업이 중단되었습니다. 남은 작업을 재시작할 수 있습니다.",
                failed_indices=list(range(self.task.completed, self.task.requested)),
            )
        self.autosave()
        try:
            self.after_cancel(self._poll_after)
        except Exception:
            pass
        write_log(self.root_dir, "CoverMorph Studio closed")
        self.destroy()
