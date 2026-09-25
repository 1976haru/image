from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Any

import customtkinter as ctk
from PIL import Image, ImageTk

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
    SDXLTextToImageEngine,
    generate_scene_candidates,
    resolve_sdxl_model_path,
    retry_failed_candidates,
)
from .planning import LlamaCppCliBackend, PlanningError, generate_plans, make_input_snapshot
from .project import (
    CandidateRecord,
    InputRecord,
    SceneCard,
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
from .settings import ensure_output_directory, load_settings, open_folder, save_settings


class QuickCoverApp(ctk.CTk):
    """Project-free default UI backed by an automatically saved normal project."""

    def __init__(self, root_dir: Path) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.settings = load_settings(root_dir)
        project_file = quick_project_dir(root_dir) / "covermorph_project.json"
        self.project = (
            load_project(project_file)
            if project_file.is_file()
            else create_project(project_file.parent, "자동 저장 간편 커버")
        )
        self.records: list[InputRecord] = []
        self.selected: CandidateRecord | None = None
        self.scene: SceneCard | None = None
        self.config: GenerationConfig | None = None
        self.failed: list[int] = []
        self.events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.cancel_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.started = 0.0
        self.preview_photo: ImageTk.PhotoImage | None = None
        self.thumb_photos: list[ImageTk.PhotoImage] = []
        self._saving = False
        self._init_vars()
        self._build()
        self._refresh_candidates()
        self.after(100, self._poll)
        self.protocol("WM_DELETE_WINDOW", self._close)

    def _init_vars(self) -> None:
        material, text = self.project.input_materials, self.project.cover_text
        self.mode = ctk.StringVar(value=material.get("quick_mode", "가사/주제로 만들기"))
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
        self.status = ctk.StringVar(value="준비됨")
        self.progress = ctk.StringVar(value="후보 0/0 · 경과 00:00")
        self.summary = ctk.StringVar(value=material.get("input_summary", "자료를 입력해 주세요."))

    def _build(self) -> None:
        self.title("CoverMorph Studio - 간편 커버 제작")
        self.geometry("1460x900")
        self.minsize(980, 680)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)
        top = ctk.CTkFrame(self)
        top.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=10)
        top.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(top, textvariable=self.status, font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, padx=12, pady=8
        )
        ctk.CTkLabel(top, textvariable=self.progress, anchor="w").grid(row=0, column=1, sticky="ew")
        self.cancel_button = ctk.CTkButton(
            top, text="취소", state="disabled", command=self.cancel, fg_color="#991b1b", width=80
        )
        self.cancel_button.grid(row=0, column=2, padx=5)
        ctk.CTkButton(top, text="고급 프로젝트 화면", command=self.open_advanced, width=145).grid(
            row=0, column=3, padx=8
        )
        left = ctk.CTkScrollableFrame(self, width=430)
        left.grid(row=1, column=0, sticky="nsew", padx=(10, 6), pady=(0, 10))
        right = ctk.CTkFrame(self)
        right.grid(row=1, column=1, sticky="nsew", padx=(0, 10), pady=(0, 10))
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(1, weight=1)
        source = self._section(left, "1) 자료 입력")
        ctk.CTkButton(source, text="JSON/TXT 불러오기", command=self.load_material).pack(
            fill="x", padx=10, pady=4
        )
        self.input_box = ctk.CTkTextbox(source, height=150)
        self.input_box.pack(fill="x", padx=10, pady=4)
        self.input_box.insert("1.0", self.project.input_materials.get("quick_input", ""))
        ctk.CTkSegmentedButton(
            source, values=["가사/주제로 만들기", "이미지 프롬프트 직접 사용"], variable=self.mode
        ).pack(fill="x", padx=10, pady=4)
        self.presets = load_generation_presets(self.root_dir / "config" / "channel_generation_presets.json")
        preset_names = ["선택 안 함"] + [item.name for item in self.presets]
        if self.preset.get() not in preset_names:
            self.preset.set("선택 안 함")
        ctk.CTkOptionMenu(source, values=preset_names, variable=self.preset).pack(fill="x", padx=10, pady=3)
        row = ctk.CTkFrame(source, fg_color="transparent")
        row.pack(fill="x", padx=10, pady=3)
        ctk.CTkEntry(row, textvariable=self.reference, placeholder_text="참고 이미지 한 장 (기본 꺼짐)").pack(
            side="left", fill="x", expand=True
        )
        ctk.CTkButton(row, text="찾기", command=self.choose_reference, width=60).pack(side="left", padx=4)
        ctk.CTkSegmentedButton(source, values=["1", "4", "5"], variable=self.count).pack(
            fill="x", padx=10, pady=4
        )
        ctk.CTkButton(source, text="이미지 만들기", command=self.generate, height=42).pack(
            fill="x", padx=10, pady=5
        )
        ctk.CTkLabel(
            source,
            textvariable=self.summary,
            wraplength=390,
            justify="left",
            anchor="w",
            text_color="#fbbf24",
        ).pack(fill="x", padx=10, pady=3)
        self.prompt_box = ctk.CTkTextbox(source, height=70)
        self.prompt_box.pack(fill="x", padx=10, pady=(2, 9))
        self.prompt_box.insert("1.0", self.project.input_materials.get("actual_prompt", ""))
        edit = self._section(left, "3) 문구와 저장")
        for label, variable, shown in (
            ("제목", self.title_var, self.show_title),
            ("부제", self.subtitle_var, self.show_subtitle),
            ("채널명/음반사", self.label_var, self.show_label),
        ):
            row = ctk.CTkFrame(edit, fg_color="transparent")
            row.pack(fill="x", padx=10, pady=2)
            ctk.CTkCheckBox(row, text=label, variable=shown, width=120, command=self.changed).pack(
                side="left"
            )
            ctk.CTkEntry(row, textvariable=variable).pack(side="left", fill="x", expand=True)
        ctk.CTkOptionMenu(
            edit, values=list(TEMPLATES), variable=self.template, command=self.template_changed
        ).pack(fill="x", padx=10, pady=3)
        font_paths = system_font_paths()
        self.fonts = {path.name: str(path) for path in font_paths}
        font_values = list(self.fonts) or ["지원 폰트 없음"]
        stored_font = Path(self.project.cover_text.get("font_path", "")).name
        self.font = ctk.StringVar(value=stored_font if stored_font in self.fonts else font_values[0])
        ctk.CTkOptionMenu(
            edit, values=font_values, variable=self.font, command=lambda _v: self.changed()
        ).pack(fill="x", padx=10, pady=3)
        row = ctk.CTkFrame(edit, fg_color="transparent")
        row.pack(fill="x", padx=10, pady=3)
        ctk.CTkLabel(row, text="제목 크기").pack(side="left")
        ctk.CTkEntry(row, textvariable=self.size, width=62).pack(side="left", padx=4)
        ctk.CTkLabel(row, text="색상").pack(side="left", padx=(8, 0))
        ctk.CTkEntry(row, textvariable=self.color, width=90).pack(side="left", padx=4)
        ctk.CTkOptionMenu(
            row,
            values=["left", "center", "right"],
            variable=self.align,
            width=95,
            command=lambda _v: self.changed(),
        ).pack(side="right")
        for label, variable in (("가로 위치", self.x), ("세로 위치", self.y)):
            ctk.CTkLabel(edit, text=label, anchor="w").pack(fill="x", padx=10)
            ctk.CTkSlider(
                edit, from_=0.05, to=0.95, variable=variable, command=lambda _v: self.changed()
            ).pack(fill="x", padx=10, pady=2)
        row = ctk.CTkFrame(edit, fg_color="transparent")
        row.pack(fill="x", padx=10, pady=3)
        ctk.CTkCheckBox(row, text="테두리", variable=self.stroke, command=self.changed).pack(side="left")
        ctk.CTkCheckBox(row, text="그림자", variable=self.shadow, command=self.changed).pack(
            side="left", padx=10
        )
        row = ctk.CTkFrame(edit, fg_color="transparent")
        row.pack(fill="x", padx=10, pady=3)
        ctk.CTkEntry(row, textvariable=self.output, placeholder_text="출력 폴더").pack(
            side="left", fill="x", expand=True
        )
        ctk.CTkButton(row, text="선택", command=self.choose_output, width=60).pack(side="left", padx=4)
        ctk.CTkButton(
            edit, text="커버 저장 (1400×1400 JPG + 원본 PNG)", command=self.save_cover, height=40
        ).pack(fill="x", padx=10, pady=3)
        row = ctk.CTkFrame(edit, fg_color="transparent")
        row.pack(fill="x", padx=10, pady=(2, 10))
        ctk.CTkButton(row, text="글자 없는 배경 저장", command=self.save_textless).pack(
            side="left", fill="x", expand=True, padx=(0, 3)
        )
        ctk.CTkButton(row, text="출력 폴더 열기", command=self.open_output).pack(
            side="left", fill="x", expand=True, padx=(3, 0)
        )
        ctk.CTkLabel(
            right,
            text="2) 후보 선택 — 썸네일을 클릭하면 크게 표시됩니다",
            anchor="w",
            font=ctk.CTkFont(weight="bold"),
        ).grid(row=0, column=0, sticky="ew", padx=12, pady=8)
        self.preview = ctk.CTkLabel(right, text="생성된 후보가 없습니다")
        self.preview.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        self.thumbs = ctk.CTkScrollableFrame(right, orientation="horizontal", height=150)
        self.thumbs.grid(row=2, column=0, sticky="ew", padx=10, pady=10)
        for variable in (self.title_var, self.subtitle_var, self.label_var, self.size, self.color):
            variable.trace_add("write", lambda *_args: self.changed())
        self.input_box.bind("<KeyRelease>", lambda _event: self.after(300, self.autosave))

    @staticmethod
    def _section(parent: Any, title: str) -> ctk.CTkFrame:
        frame = ctk.CTkFrame(parent, border_width=1, border_color="#334155")
        frame.pack(fill="x", padx=5, pady=6)
        ctk.CTkLabel(frame, text=title, font=ctk.CTkFont(weight="bold"), anchor="w").pack(
            fill="x", padx=10, pady=(8, 3)
        )
        return frame

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
                    "input_summary": self.summary.get(),
                    "actual_prompt": self.prompt_box.get("1.0", "end").strip(),
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

    def load_material(self) -> None:
        filename = filedialog.askopenfilename(
            title="JSON/TXT 자료 불러오기", filetypes=[("JSON/TXT", "*.json *.txt"), ("모든 파일", "*.*")]
        )
        if not filename:
            return
        before = self.input_box.get("1.0", "end")
        try:
            parsed = parse_input_file_detailed(Path(filename))
            if parsed.get("needs_mapping"):
                raise ValueError(
                    "JSON 필드를 확실히 매핑할 수 없습니다. 고급 프로젝트 화면에서 필드를 지정해 주세요."
                )
            records = list(parsed["records"])
            if len(records) > 1 and not self._select_records(records):
                return
            add_input_records(self.project, records)
            self.records = records
            selected = [item for item in records if item.selected]
            material = "\n\n".join(item.lyrics or item.theme_mood or item.image_prompt for item in selected)
            self.input_box.delete("1.0", "end")
            self.input_box.insert("1.0", material)
            mapped_text = parsed.get("cover_text") or {}
            if mapped_text.get("title"):
                self.title_var.set(str(mapped_text["title"]))
            if mapped_text.get("subtitle"):
                self.subtitle_var.set(str(mapped_text["subtitle"]))
            if mapped_text.get("label"):
                self.label_var.set(str(mapped_text["label"]))
            self.summary.set(f"{Path(filename).name} · 전체 {len(records)}곡 · 선택 {len(selected)}곡")
            self.autosave()
        except Exception as exc:
            self.input_box.delete("1.0", "end")
            self.input_box.insert("1.0", before)
            messagebox.showerror("자료 해석 실패", f"입력은 그대로 보존했습니다.\n{exc}")

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

    def _planned_prompt(self, source: str) -> str:
        if self.mode.get() == "이미지 프롬프트 직접 사용":
            self.events.put(("summary", "직접 입력 원문을 변경 없이 실제 생성 입력으로 사용합니다."))
            return source
        self.events.put(("status", "자료 해석 중"))
        preset = next((item for item in self.presets if item.name == self.preset.get()), self.presets[0])
        records = self.records or [InputRecord(new_id("direct"), "lyrics", "직접 입력", lyrics=source)]
        snapshot = make_input_snapshot(preset, records, source if not self.records else "", {}, 4)
        backend = LlamaCppCliBackend(self.root_dir)
        try:
            result = generate_plans(snapshot, backend, self.cancel_event)
        finally:
            backend.close()
        plans = result.get("plans") or []
        if not plans or not plans[0].get("image_prompt_en"):
            raise PlanningError("로컬 기획이 유효한 이미지 프롬프트를 만들지 못했습니다.")
        self.events.put(("summary", "기존 로컬 기획 경로로 자료를 해석했습니다."))
        return str(plans[0]["image_prompt_en"])

    def generate(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        source = self.input_box.get("1.0", "end").strip()
        if not source:
            messagebox.showwarning("입력 필요", "가사·주제 또는 이미지 프롬프트를 입력해 주세요.")
            return
        count, mode = int(self.count.get()), self.mode.get()
        self.cancel_event.clear()
        self.failed = []
        self.started = time.monotonic()
        self.cancel_button.configure(state="normal")
        self.status.set("자료 해석 중" if mode == "가사/주제로 만들기" else "모델 준비 중")
        self.progress.set(f"후보 0/{count} · 경과 00:00")
        self.autosave()

        def work() -> None:
            try:
                prompt = self._planned_prompt(source)
                self.events.put(("prompt", prompt))
                scene = SceneCard(
                    new_id("quick_scene"),
                    order=len(self.project.scenes) + 1,
                    user_description=source,
                    prompt_user=prompt,
                    negative_prompt_user="text, letters, typography, logo, watermark, distorted anatomy",
                    output_ratio="1:1",
                    candidate_count=count,
                    prompt_confirmed=True,
                    structured_request={
                        "approval_source": "quick_cover_no_approval_required",
                        "textless_background": True,
                        "original_input": source,
                        "actual_generation_prompt": prompt,
                        "prompt_mode": mode,
                    },
                )
                self.project.scenes.append(scene)
                reference_mode, reference_id = "off", ""
                reference_path = Path(self.reference.get())
                if reference_path.is_file():
                    person = add_person(self.project, "간편 참고 이미지")
                    reference = add_person_reference(self.project, person, reference_path, "style")
                    scene.reference_image_ids = [reference.image_id]
                    scene.structured_request["reference_image_ids"] = [reference.image_id]
                    reference_mode, reference_id = "style", reference.image_id
                model_path = resolve_sdxl_model_path(self.root_dir, DEFAULT_SDXL_MODEL)
                config = GenerationConfig(
                    model_id=str(model_path),
                    candidate_count=count,
                    seed=int(time.time()) % 2_000_000_000,
                    local_files_only=True,
                    reference_mode=reference_mode,
                    reference_image_id=reference_id,
                )
                self.scene, self.config = scene, config
                engine = SDXLTextToImageEngine(config.model_id, local_files_only=True)
                result = generate_scene_candidates(
                    self.project,
                    scene,
                    engine,
                    config,
                    self.cancel_event,
                    lambda data: self.events.put(("generation", data)),
                )
                save_project_atomic(self.project)
                self.events.put(("done", result))
            except Exception as exc:
                self.events.put(("error", exc))

        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def retry(self) -> None:
        if not self.failed or not self.scene or not self.config:
            return
        indices = list(self.failed)
        self.cancel_event.clear()
        self.started = time.monotonic()
        self.cancel_button.configure(state="normal")

        def work() -> None:
            try:
                engine = SDXLTextToImageEngine(self.config.model_id, local_files_only=True)
                result = retry_failed_candidates(
                    self.project,
                    self.scene,
                    engine,
                    self.config,
                    indices,
                    self.cancel_event,
                    lambda data: self.events.put(("generation", data)),
                )
                save_project_atomic(self.project)
                self.events.put(("done", result))
            except Exception as exc:
                self.events.put(("error", exc))

        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def _poll(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "status":
                    self.status.set(payload)
                elif kind == "summary":
                    self.summary.set(payload)
                elif kind == "prompt":
                    self.prompt_box.delete("1.0", "end")
                    self.prompt_box.insert("1.0", payload)
                elif kind == "generation":
                    self._progress(payload)
                elif kind == "done":
                    self._done(payload)
                elif kind == "error":
                    self.cancel_button.configure(state="disabled")
                    self.status.set("실패")
                    messagebox.showerror("작업 실패", str(payload))
        except queue.Empty:
            pass
        if self.worker and self.worker.is_alive():
            elapsed = int(time.monotonic() - self.started)
            self.progress.set(
                f"{self.progress.get().split('·')[0].strip()} · 경과 {elapsed // 60:02d}:{elapsed % 60:02d}"
            )
        self.after(100, self._poll)

    def _progress(self, data: dict[str, Any]) -> None:
        phase = data.get("phase")
        if phase == "load":
            self.status.set("모델 준비 중")
        elif phase == "inference":
            self.status.set(f"이미지 생성 중 · 추론 {data.get('step')}/{data.get('steps')}")
        elif phase == "candidate_start":
            self.status.set("이미지 생성 중")
        elif phase == "candidate_done":
            self.status.set("저장 중")
            total = int(data.get("total") or self.count.get())
            complete = len(
                [item for item in self.project.candidates if item.scene_id == data.get("scene_id")]
            )
            self.progress.set(f"후보 {complete}/{total}")
            self._refresh_candidates()

    def _done(self, result: Any) -> None:
        self.cancel_button.configure(state="disabled")
        self.failed = list(result.failed_indices)
        self._refresh_candidates()
        self.status.set(
            "취소됨 · 완료 후보 보존" if result.cancelled else "완료" if not self.failed else "일부 실패"
        )
        if self.failed and messagebox.askyesno("실패 후보", "실패한 후보만 재시도할까요?"):
            self.retry()
        self.autosave()

    def cancel(self) -> None:
        self.cancel_event.set()
        self.status.set("취소 요청 중")

    def _refresh_candidates(self) -> None:
        for widget in self.thumbs.winfo_children():
            widget.destroy()
        self.thumb_photos.clear()
        candidates = [item for item in self.project.candidates if item.generation_status == "succeeded"]
        for candidate in candidates:
            try:
                with Image.open(resolve_project_path(self.project, candidate.original_path)) as opened:
                    image = opened.convert("RGB")
                    image.thumbnail((125, 125))
                    photo = ImageTk.PhotoImage(image)
                self.thumb_photos.append(photo)
                ctk.CTkButton(
                    self.thumbs,
                    text="",
                    image=photo,
                    width=130,
                    height=130,
                    command=lambda item=candidate: self.select_candidate(item),
                ).pack(side="left", padx=4)
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
        edit = candidate_edit(candidate)
        self._set_edit(edit)
        self.project.selected_candidate_ids = [candidate.candidate_id]
        save_project_atomic(self.project)
        self.draw_preview()

    def _set_edit(self, edit: CoverEdit) -> None:
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
            image.thumbnail((760, 650))
            self.preview_photo = ImageTk.PhotoImage(image)
            self.preview.configure(image=self.preview_photo, text="")
        except Exception as exc:
            self.preview.configure(image=None, text=f"미리보기 확인 필요\n{exc}")

    def save_cover(self) -> None:
        if not self.selected:
            messagebox.showwarning("후보 선택", "저장할 후보를 선택해 주세요.")
            return
        try:
            paths = export_cover(
                self.project,
                self.selected,
                self.current_edit(),
                ensure_output_directory(self.output.get(), create=True),
            )
            self.status.set(f"저장 완료: {paths['cover_jpg']}")
            messagebox.showinfo("저장 완료", "\n".join(str(path) for path in paths.values()))
        except Exception as exc:
            messagebox.showerror("저장 실패", str(exc))

    def save_textless(self) -> None:
        if not self.selected:
            messagebox.showwarning("후보 선택", "저장할 후보를 선택해 주세요.")
            return
        try:
            from .cover_studio import copy_textless

            path = copy_textless(
                self.project, self.selected, ensure_output_directory(self.output.get(), create=True)
            )
            self.status.set(f"글자 없는 배경 저장 완료: {path}")
        except Exception as exc:
            messagebox.showerror("저장 실패", str(exc))

    def open_output(self) -> None:
        try:
            open_folder(ensure_output_directory(self.output.get()))
        except Exception as exc:
            messagebox.showerror("폴더 열기 실패", str(exc))

    def open_advanced(self) -> None:
        self.autosave()
        self.destroy()
        from .gui import CoverMorphApp

        app = CoverMorphApp()
        app.mainloop()

    def _close(self) -> None:
        self.autosave()
        self.destroy()
