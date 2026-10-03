"""AI 이미지 스튜디오: YouTube / Shopify creation, reference roles, queue and candidate review.

Model names never appear as choices: the user picks a purpose, a quality (빠른 미리보기 / 일반 / 최고 품질)
and how much of the PC to use; Quality Engine V2 picks Z-Image / FLUX.2 / RealVis. All generation runs on
the studio queue thread (one job at a time); this module only builds payloads and shows results.
"""
from __future__ import annotations

import json
import os
import random
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Any

import customtkinter as ctk
from PIL import Image, ImageDraw

from .creator_jobs import export_candidate, run_creator_job, safe_name
from .creator_presets import (
    DEFAULT_MEMORY_LABEL,
    DEFAULT_QUALITY_LABEL,
    MEMORY_LABELS,
    QUALITY_LABELS,
    ROLE_LABELS,
    ROLE_NAMES,
    label_for,
    load_presets,
    validate_canvas,
)
from .creator_runner import StudioRunner, queue_path
from .creator_settings import (
    apply_backend_paths,
    engine_readiness,
    load_creator_settings,
    resolve_models_dir,
    resolve_output_dir,
    save_creator_settings,
)
from .job_queue import CANCELLED, DONE, FAILED, PENDING, RUNNING, JobQueue
from .quality_engines import resource_snapshot

STATE_LABELS = {PENDING: "대기", RUNNING: "실행 중", DONE: "완료", FAILED: "실패", CANCELLED: "취소됨"}
ENGINE_NAMES = {"zimage_turbo": "Z-Image", "flux2_klein_4b": "FLUX.2", "realvisxl_v5": "RealVis"}
KIND_LABELS = {"generated": "생성", "composite": "상품 합성(형태 보존)", "harmonized": "합성+조명 보정(AI)",
               "regenerated": "상품 참조 생성(AI)", "edit": "편집"}
MAX_REFERENCES = 4
IMAGE_TYPES = [("이미지", "*.png *.jpg *.jpeg *.webp *.bmp"), ("모든 파일", "*.*")]


def open_path(path: Path) -> None:
    try:
        os.startfile(str(path))  # noqa: S606 - user-requested folder/file open
    except OSError:
        pass


def fit_image(image: Image.Image, box: tuple[int, int]) -> ctk.CTkImage:
    copy = image.copy()
    copy.thumbnail(box, Image.Resampling.LANCZOS)
    return ctk.CTkImage(light_image=copy, dark_image=copy, size=copy.size)


class StudioWindow(ctk.CTkToplevel):
    def __init__(self, master: Any, app_root: Path):
        super().__init__(master)
        self.app_root = Path(app_root)
        self.title("AI 이미지 스튜디오 — YouTube · Shopify")
        self.geometry("1500x930")
        self.minsize(1200, 760)
        self.settings = load_creator_settings(self.app_root)
        apply_backend_paths(self.settings)
        self.purposes, self.prompt_presets, problems = load_presets(self.app_root)
        self.references: list[dict[str, Any]] = []
        self.review_job_id: str | None = None
        self.review_index = 0
        self.compare_pin: dict[str, Any] | None = None
        self._dirty = True
        self._resources_text = ""
        self._images: list[Any] = []  # keep CTkImage references alive

        output_root = resolve_output_dir(self.app_root, self.settings)
        output_root.mkdir(parents=True, exist_ok=True)
        self.queue = JobQueue(queue_path(output_root))
        self.runner = StudioRunner(self.queue, self._execute, self.settings, on_change=self._mark_dirty)

        self.tabs = ctk.CTkTabview(self)
        self.tabs.pack(fill="both", expand=True, padx=10, pady=10)
        for name in ("만들기", "대기열", "후보 비교", "설정"):
            self.tabs.add(name)
        self._build_create(self.tabs.tab("만들기"))
        self._build_queue(self.tabs.tab("대기열"))
        self._build_review(self.tabs.tab("후보 비교"))
        self._build_settings(self.tabs.tab("설정"))
        self.protocol("WM_DELETE_WINDOW", self.withdraw)  # closing the studio keeps the queue running
        if problems:
            messagebox.showwarning("프리셋", "\n".join(problems), parent=self)
        self._set_purpose_key("youtube_thumbnail")
        self.after(400, self._poll)
        threading.Thread(target=self._resource_loop, daemon=True).start()
        if self.settings.get("auto_start_when_free") and self.queue.pending() and not self.queue.paused:
            self.runner.start()

    # ================================================================== create tab
    def _build_create(self, tab: Any) -> None:
        tab.grid_columnconfigure(0, weight=3)
        tab.grid_columnconfigure(1, weight=2)
        tab.grid_rowconfigure(0, weight=1)
        form = ctk.CTkScrollableFrame(tab)
        form.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        side = ctk.CTkFrame(tab)
        side.grid(row=0, column=1, sticky="nsew")

        def heading(text: str) -> None:
            ctk.CTkLabel(form, text=text, anchor="w", font=ctk.CTkFont(size=15, weight="bold")).pack(fill="x", padx=10, pady=(12, 4))

        heading("1. 무엇을 만들까요?")
        self.purpose_var = ctk.StringVar(value=self.purposes["youtube_thumbnail"].label)
        self.family_var = ctk.StringVar(value="YouTube")
        family_row = ctk.CTkFrame(form, fg_color="transparent")
        family_row.pack(fill="x", padx=10)
        ctk.CTkSegmentedButton(family_row, values=["YouTube", "Shopify", "사용자 지정"], variable=self.family_var,
                               command=lambda _v: self._apply_family(), width=360, height=36).pack(side="left")
        self.shop_var = ctk.StringVar(value="")
        self.shop_menu = ctk.CTkOptionMenu(family_row, variable=self.shop_var, width=230, values=self._shop_labels(),
                                           command=lambda _v: self._apply_shop())
        self.shop_menu.pack(side="left", padx=10)
        size_row = ctk.CTkFrame(form, fg_color="transparent")
        size_row.pack(fill="x", padx=10, pady=6)
        ctk.CTkLabel(size_row, text="캔버스 크기").pack(side="left")
        self.width_var, self.height_var = ctk.StringVar(), ctk.StringVar()
        ctk.CTkEntry(size_row, textvariable=self.width_var, width=70).pack(side="left", padx=4)
        ctk.CTkLabel(size_row, text="×").pack(side="left")
        ctk.CTkEntry(size_row, textvariable=self.height_var, width=70).pack(side="left", padx=4)
        self.alternate_var = ctk.StringVar(value="기본 크기")
        self.alternate_menu = ctk.CTkOptionMenu(size_row, variable=self.alternate_var, values=["기본 크기"],
                                                command=self._apply_alternate, width=130)
        self.alternate_menu.pack(side="left", padx=8)
        ctk.CTkLabel(size_row, text="(테마마다 크기가 다르면 직접 입력)", text_color="#94a3b8").pack(side="left")
        for var in (self.width_var, self.height_var):
            var.trace_add("write", lambda *_a: self._draw_layout())

        heading("2. 프롬프트")
        preset_row = ctk.CTkFrame(form, fg_color="transparent")
        preset_row.pack(fill="x", padx=10)
        ctk.CTkLabel(preset_row, text="프롬프트 프리셋").pack(side="left")
        self.preset_var = ctk.StringVar(value="(직접 입력)")
        ctk.CTkOptionMenu(preset_row, variable=self.preset_var, width=260,
                          values=["(직접 입력)"] + [p.label for p in self.prompt_presets.values()],
                          command=lambda _v: self._apply_preset()).pack(side="left", padx=6)
        channel_row = ctk.CTkFrame(form, fg_color="transparent")
        channel_row.pack(fill="x", padx=10, pady=(6, 0))
        ctk.CTkLabel(channel_row, text="채널/스토어").pack(side="left")
        self.channel_var = ctk.StringVar()
        ctk.CTkEntry(channel_row, textvariable=self.channel_var, width=200).pack(side="left", padx=6)
        ctk.CTkLabel(channel_row, text="인물 수").pack(side="left", padx=(12, 0))
        self.people_var = ctk.StringVar(value="자동")
        ctk.CTkOptionMenu(channel_row, variable=self.people_var, values=["자동", "0", "1", "2"], width=80).pack(side="left", padx=6)
        self.prompt_box = ctk.CTkTextbox(form, height=90, wrap="word")
        self.prompt_box.pack(fill="x", padx=10, pady=6)
        self.prompt_hint = ctk.CTkLabel(form, text="한국어·일본어로 써도 됩니다(내부에서 영어로 번역). 글자는 이미지에 그리지 않습니다.",
                                        anchor="w", text_color="#94a3b8")
        self.prompt_hint.pack(fill="x", padx=10)
        text_row = ctk.CTkFrame(form, fg_color="transparent")
        text_row.pack(fill="x", padx=10, pady=6)
        self.title_var, self.subtitle_var, self.cta_var = ctk.StringVar(), ctk.StringVar(), ctk.StringVar()
        for label, var, width in (("제목", self.title_var, 220), ("부제", self.subtitle_var, 180), ("버튼(CTA)", self.cta_var, 120)):
            ctk.CTkLabel(text_row, text=label).pack(side="left", padx=(6, 2))
            ctk.CTkEntry(text_row, textvariable=var, width=width).pack(side="left")
        ctk.CTkLabel(form, text="제목/부제/버튼은 '텍스트 합성본' 내보내기에만 들어갑니다(무문자 원본도 항상 저장).",
                     anchor="w", text_color="#94a3b8").pack(fill="x", padx=10)

        heading("3. 레퍼런스 (역할 지정, 최대 4장)")
        ref_buttons = ctk.CTkFrame(form, fg_color="transparent")
        ref_buttons.pack(fill="x", padx=10)
        ctk.CTkButton(ref_buttons, text="레퍼런스 추가", command=self._add_references, width=130).pack(side="left")
        ctk.CTkLabel(form, text="역할: 인물=얼굴·머리·옷·자세 유지 / 상품=형태·색·로고 유지 / 스타일=빛·색감만 / "
                                "구도=배치·여백만 / 배경=장소 분위기만", anchor="w", justify="left", wraplength=620,
                     text_color="#94a3b8").pack(fill="x", padx=10, pady=(4, 0))
        self.ref_frame = ctk.CTkFrame(form)
        self.ref_frame.pack(fill="x", padx=10, pady=6)
        product_row = ctk.CTkFrame(form, fg_color="transparent")
        product_row.pack(fill="x", padx=10, pady=4)
        self.preserve_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(product_row, text="상품 형태 보존", variable=self.preserve_var,
                        command=self._update_engine_note).pack(side="left")
        ctk.CTkLabel(product_row, text="상품 크기").pack(side="left", padx=(16, 4))
        self.product_scale = ctk.CTkSlider(product_row, from_=0.25, to=0.9, number_of_steps=13, width=160)
        self.product_scale.set(0.5)
        self.product_scale.pack(side="left")
        ctk.CTkLabel(form, text="상품 형태 보존: 상품 사진의 실제 픽셀을 생성 배경에 합성한 후보가 맨 앞에 옵니다(형태 그대로). "
                                "AI가 다시 그린 후보는 '형태 확인 필요'로 표시되며 최상위로 올라가지 않습니다.",
                     anchor="w", justify="left", wraplength=620, text_color="#94a3b8").pack(fill="x", padx=10)

        heading("4. 품질과 PC 사용")
        self.quality_var = ctk.StringVar(value=self.settings.get("quality_label") or DEFAULT_QUALITY_LABEL)
        ctk.CTkSegmentedButton(form, values=list(QUALITY_LABELS), variable=self.quality_var,
                               command=lambda _v: self._update_engine_note()).pack(fill="x", padx=10, pady=2)
        self.memory_var = ctk.StringVar(value=self.settings.get("memory_label") or DEFAULT_MEMORY_LABEL)
        ctk.CTkSegmentedButton(form, values=list(MEMORY_LABELS), variable=self.memory_var).pack(fill="x", padx=10, pady=2)
        ctk.CTkLabel(form, text="작업 중 PC 우선: 한 장씩 만들고 즉시 메모리 반환(기본) · 균형: 조금 더 큰 해상도 · "
                                "자리 비움/최고 품질: 원본 해상도와 여러 엔진 순차 비교", anchor="w", wraplength=620,
                     justify="left", text_color="#94a3b8").pack(fill="x", padx=10)
        seed_row = ctk.CTkFrame(form, fg_color="transparent")
        seed_row.pack(fill="x", padx=10, pady=6)
        ctk.CTkLabel(seed_row, text="seed").pack(side="left")
        self.seed_var = ctk.StringVar(value=str(random.randint(1, 999999)))
        ctk.CTkEntry(seed_row, textvariable=self.seed_var, width=100).pack(side="left", padx=4)
        ctk.CTkButton(seed_row, text="무작위", width=70, command=lambda: self.seed_var.set(str(random.randint(1, 999999)))).pack(side="left")
        ctk.CTkLabel(seed_row, text="후보 수").pack(side="left", padx=(16, 4))
        self.count_var = ctk.StringVar(value="2")
        ctk.CTkOptionMenu(seed_row, variable=self.count_var, values=["1", "2", "3", "4"], width=70).pack(side="left")

        actions = ctk.CTkFrame(form, fg_color="transparent")
        actions.pack(fill="x", padx=10, pady=14)
        ctk.CTkButton(actions, text="대기열에 추가", height=40, command=lambda: self._enqueue(False)).pack(side="left", padx=(0, 8))
        ctk.CTkButton(actions, text="추가하고 바로 시작", height=40, fg_color="#16a34a", hover_color="#15803d",
                      command=lambda: self._enqueue(True)).pack(side="left")

        ctk.CTkLabel(side, text="레이아웃 미리보기", font=ctk.CTkFont(weight="bold")).pack(pady=(12, 4))
        self.layout_label = ctk.CTkLabel(side, text="")
        self.layout_label.pack(padx=10, pady=4)
        ctk.CTkLabel(side, text="파랑=글자 영역 · 주황=피사체 영역 · 초록=버튼 영역", text_color="#94a3b8").pack()
        self.engine_note = ctk.CTkLabel(side, text="", justify="left", anchor="w", wraplength=480)
        self.engine_note.pack(fill="x", padx=14, pady=12)

    def _shop_labels(self) -> list[str]:
        return [p.label.replace("Shopify ", "") for p in self.purposes.values() if p.key.startswith("shopify")]

    def _apply_family(self) -> None:
        family = self.family_var.get()
        if family == "YouTube":
            key = "youtube_thumbnail"
        elif family == "사용자 지정":
            key = "custom"
        else:
            label = self.shop_var.get() or self._shop_labels()[0]
            key = next(p.key for p in self.purposes.values() if p.label.replace("Shopify ", "") == label)
            self.shop_var.set(label)
        self.shop_menu.configure(state="normal" if family == "Shopify" else "disabled")
        self.purpose_var.set(self.purposes[key].label)
        self._apply_purpose()

    def _apply_shop(self) -> None:
        self.family_var.set("Shopify")
        self._apply_family()

    def _set_purpose_key(self, key: str) -> None:
        purpose = self.purposes[key]
        if key.startswith("shopify"):
            self.family_var.set("Shopify")
            self.shop_var.set(purpose.label.replace("Shopify ", ""))
        else:
            self.family_var.set("YouTube" if key == "youtube_thumbnail" else "사용자 지정")
        self.shop_menu.configure(state="normal" if key.startswith("shopify") else "disabled")
        self.purpose_var.set(purpose.label)
        self._apply_purpose()

    def _purpose(self):
        label = self.purpose_var.get()
        return next(p for p in self.purposes.values() if p.label == label)

    def _apply_purpose(self) -> None:
        purpose = self._purpose()
        self.width_var.set(str(purpose.width))
        self.height_var.set(str(purpose.height))
        self.alternate_menu.configure(values=["기본 크기", *purpose.alternates])
        self.alternate_var.set("기본 크기")
        self._draw_layout()
        self._update_engine_note()

    def _apply_alternate(self, choice: str) -> None:
        purpose = self._purpose()
        width, height = purpose.alternates.get(choice, (purpose.width, purpose.height))
        self.width_var.set(str(width))
        self.height_var.set(str(height))

    def _apply_preset(self) -> None:
        preset = next((p for p in self.prompt_presets.values() if p.label == self.preset_var.get()), None)
        if preset is None:
            return
        if preset.purpose in self.purposes:
            self._set_purpose_key(preset.purpose)
        self.channel_var.set(preset.channel)
        self.people_var.set(str(preset.people))
        self.prompt_box.delete("1.0", "end")
        self.prompt_box.insert("1.0", preset.prompt)

    def _canvas(self) -> tuple[int, int]:
        return validate_canvas(int(self.width_var.get() or 0), int(self.height_var.get() or 0))

    def _draw_layout(self) -> None:
        try:
            width, height = self._canvas()
        except (ValueError, tk.TclError):
            return
        purpose = self._purpose()
        scale = min(460 / width, 340 / height)
        size = (max(1, int(width * scale)), max(1, int(height * scale)))
        image = Image.new("RGB", size, (30, 41, 59))
        draw = ImageDraw.Draw(image)
        for region, colour in ((purpose.text_region, (59, 130, 246)), (purpose.subject_region, (249, 115, 22)),
                               (purpose.cta_region, (34, 197, 94))):
            if region:
                x, y, w, h = region
                draw.rectangle((x * size[0], y * size[1], (x + w) * size[0], (y + h) * size[1]), outline=colour, width=3)
        draw.text((6, size[1] - 16), f"{width}×{height}", fill=(226, 232, 240))
        photo = ctk.CTkImage(light_image=image, dark_image=image, size=size)
        self._layout_image = photo
        self.layout_label.configure(image=photo)

    def _update_engine_note(self) -> None:
        refs = self.references
        quality = QUALITY_LABELS.get(self.quality_var.get(), "balanced") if hasattr(self, "quality_var") else "balanced"
        if self.preserve_var.get() and any(r["role"] == "PRODUCT" for r in refs):
            text = ("엔진: 배경 Z-Image → 실제 상품 픽셀 합성"
                    + ("" if quality == "preview" else " → FLUX.2 조명 보정 후보")
                    + (" → FLUX.2 상품 참조 후보" if quality == "best" else ""))
        elif refs:
            text = "엔진: FLUX.2-klein (레퍼런스 반영)" + (" + RealVis(IP-Adapter) 비교 후보" if quality == "best" else "")
        else:
            text = {"preview": "엔진: FLUX.2-klein 1장 (약 20초)", "balanced": "엔진: Z-Image-Turbo (장당 약 35초)",
                    "best": "엔진: Z-Image-Turbo → FLUX.2-klein 순차 비교"}[quality]
        self.engine_note.configure(text=text + "\n엔진이 없거나 실패하면 RealVisXL로 자동 대체합니다.")

    # ---------------------------------------------------------------- references
    def _add_references(self) -> None:
        paths = filedialog.askopenfilenames(parent=self, title="레퍼런스 이미지", filetypes=IMAGE_TYPES)
        for path in paths:
            if len(self.references) >= MAX_REFERENCES:
                messagebox.showinfo("레퍼런스", f"레퍼런스는 최대 {MAX_REFERENCES}장입니다.", parent=self)
                break
            self.references.append({"path": path, "role": "PERSON"})
        self._render_references()

    def _render_references(self) -> None:
        for child in self.ref_frame.winfo_children():
            child.destroy()
        if not self.references:
            ctk.CTkLabel(self.ref_frame, text="레퍼런스 없음 — 텍스트만으로 생성합니다.", text_color="#94a3b8").pack(padx=8, pady=8)
        for index, ref in enumerate(self.references):
            row = ctk.CTkFrame(self.ref_frame, fg_color="transparent")
            row.pack(fill="x", padx=6, pady=3)
            try:
                with Image.open(ref["path"]) as opened:
                    thumb = fit_image(opened.convert("RGB"), (64, 64))
                self._images.append(thumb)
                ctk.CTkLabel(row, text="", image=thumb).pack(side="left")
            except OSError:
                ctk.CTkLabel(row, text="(열 수 없음)").pack(side="left")
            ctk.CTkLabel(row, text=f"{index + 1}. {Path(ref['path']).name}", width=260, anchor="w").pack(side="left", padx=6)
            var = ctk.StringVar(value=ROLE_NAMES[ref["role"]])

            def set_role(choice: str, ref=ref) -> None:
                ref["role"] = ROLE_LABELS[choice]
                if ref["role"] == "PRODUCT":
                    self.preserve_var.set(True)
                self._update_engine_note()

            ctk.CTkOptionMenu(row, values=list(ROLE_LABELS), variable=var, command=set_role, width=90).pack(side="left")
            ctk.CTkButton(row, text="삭제", width=50, fg_color="#7f1d1d",
                          command=lambda i=index: (self.references.pop(i), self._render_references())).pack(side="left", padx=6)
        self._update_engine_note()

    # ---------------------------------------------------------------- payload
    def _payload(self) -> dict[str, Any] | None:
        prompt = self.prompt_box.get("1.0", "end").strip()
        if not prompt and not self.references:
            messagebox.showwarning("만들기", "프롬프트를 입력하거나 레퍼런스를 추가하세요.", parent=self)
            return None
        try:
            canvas = self._canvas()
            seed = int(self.seed_var.get())
        except ValueError as exc:
            messagebox.showwarning("만들기", f"크기/seed를 확인하세요: {exc}", parent=self)
            return None
        purpose = self._purpose()
        preset = next((p for p in self.prompt_presets.values() if p.label == self.preset_var.get()), None)
        people = self.people_var.get()
        if people == "자동":
            from .person_quality import people_count
            people_n = preset.people if preset else people_count(prompt)
        else:
            people_n = int(people)
        composition = (preset.composition if preset and people_n == preset.people else
                       {0: "", 1: "SOLO_MEDIUM", 2: "COUPLE_MEDIUM"}.get(min(people_n, 2), ""))
        for ref in self.references:
            if not Path(ref["path"]).exists():
                messagebox.showwarning("레퍼런스", f"파일이 없습니다: {ref['path']}", parent=self)
                return None
        return {"kind": "generate", "purpose": purpose.key, "purpose_label": purpose.label, "canvas": list(canvas),
                "prompt": prompt, "prompt_preset": preset.key if preset else "", "channel": self.channel_var.get(),
                "people": people_n, "composition": composition,
                "references": [dict(r) for r in self.references],
                "product_preserve": bool(self.preserve_var.get()), "product_scale": round(float(self.product_scale.get()), 2),
                "quality": QUALITY_LABELS[self.quality_var.get()], "memory": MEMORY_LABELS[self.memory_var.get()],
                "seed": seed, "candidates": int(self.count_var.get()), "title": self.title_var.get(),
                "subtitle": self.subtitle_var.get(), "cta": self.cta_var.get()}

    def _enqueue(self, start: bool) -> None:
        payload = self._payload()
        if payload is None:
            return
        self.queue.add(payload)
        self.settings["quality_label"], self.settings["memory_label"] = self.quality_var.get(), self.memory_var.get()
        save_creator_settings(self.app_root, self.settings)
        self.seed_var.set(str(random.randint(1, 999999)))
        if start or self.settings.get("auto_start_when_free"):
            self.runner.start()
        self._mark_dirty()
        self.tabs.set("대기열")

    def _load_payload(self, payload: dict[str, Any]) -> None:
        """'프롬프트 수정': put a job's settings back into the create form."""
        if payload.get("purpose") in self.purposes:
            self._set_purpose_key(payload["purpose"])
        canvas = payload.get("canvas") or []
        if len(canvas) == 2:
            self.width_var.set(str(canvas[0]))
            self.height_var.set(str(canvas[1]))
        self.prompt_box.delete("1.0", "end")
        self.prompt_box.insert("1.0", payload.get("prompt", ""))
        self.channel_var.set(payload.get("channel", ""))
        self.people_var.set(str(payload.get("people", "자동")))
        self.references = [dict(r) for r in payload.get("references") or [] if Path(r["path"]).exists()]
        self.preserve_var.set(bool(payload.get("product_preserve")))
        self.quality_var.set(label_for(QUALITY_LABELS, payload.get("quality", "balanced")))
        self.memory_var.set(label_for(MEMORY_LABELS, payload.get("memory", "interactive_low_memory")))
        self.seed_var.set(str(payload.get("seed", "")))
        self.count_var.set(str(payload.get("candidates", 2)))
        for var, key in ((self.title_var, "title"), (self.subtitle_var, "subtitle"), (self.cta_var, "cta")):
            var.set(payload.get(key, ""))
        self._render_references()
        self.tabs.set("만들기")

    # ================================================================== queue tab
    def _build_queue(self, tab: Any) -> None:
        bar = ctk.CTkFrame(tab, fg_color="transparent")
        bar.pack(fill="x", pady=(0, 6))
        buttons = [("시작", self._start), ("현재 작업 후 일시정지", self.runner.pause_after_current), ("재개", self._start),
                   ("선택 취소", self._cancel_selected), ("대기 항목 삭제", self._delete_selected),
                   ("완료 항목 열기", self._open_selected), ("실패 재시도", self._retry_selected)]
        for text, command in buttons:
            ctk.CTkButton(bar, text=text, command=command, width=120).pack(side="left", padx=3)
        self.status_label = ctk.CTkLabel(tab, text="", anchor="w", font=ctk.CTkFont(weight="bold"))
        self.status_label.pack(fill="x", padx=4)
        self.resource_label = ctk.CTkLabel(tab, text="", anchor="w", text_color="#94a3b8")
        self.resource_label.pack(fill="x", padx=4, pady=(0, 6))
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Studio.Treeview", background="#1f2937", fieldbackground="#1f2937", foreground="#e5e7eb",
                        rowheight=28, font=("Malgun Gothic", 10))
        style.configure("Studio.Treeview.Heading", background="#334155", foreground="#f8fafc", font=("Malgun Gothic", 10, "bold"))
        style.map("Studio.Treeview", background=[("selected", "#2563eb")])
        columns = ("status", "purpose", "prompt", "engine", "quality", "progress", "elapsed", "folder")
        headings = ("상태", "목적", "프롬프트/제목", "엔진", "품질", "진행률", "경과", "출력 폴더")
        widths = (90, 160, 360, 150, 100, 220, 70, 300)
        frame = ctk.CTkFrame(tab)
        frame.pack(fill="both", expand=True)
        self.tree = ttk.Treeview(frame, columns=columns, show="headings", style="Studio.Treeview", selectmode="extended")
        for column, heading, width in zip(columns, headings, widths):
            self.tree.heading(column, text=heading)
            self.tree.column(column, width=width, anchor="w", stretch=column in ("prompt", "folder"))
        scroll = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.tree.bind("<Double-1>", lambda _e: self._open_selected())

    def _selected_jobs(self) -> list[Any]:
        return [job for job in (self.queue.get(i) for i in self.tree.selection()) if job is not None]

    def _start(self) -> None:
        self.runner.start()
        self._mark_dirty()

    def _cancel_selected(self) -> None:
        for job in self._selected_jobs():
            if job.state == RUNNING:
                self.runner.cancel_current()
            else:
                self.queue.cancel(job.id)
        self._mark_dirty()

    def _delete_selected(self) -> None:
        jobs = [job for job in self._selected_jobs() if job.state != RUNNING]
        if jobs and messagebox.askyesno("대기열", f"{len(jobs)}개 항목을 목록에서 지울까요? (만든 파일은 지우지 않습니다)", parent=self):
            self.queue.remove([job.id for job in jobs])
            self._mark_dirty()

    def _retry_selected(self) -> None:
        for job in self._selected_jobs():
            self.queue.retry(job.id)
        self.runner.poke()
        self._mark_dirty()

    def _open_selected(self) -> None:
        jobs = self._selected_jobs()
        if not jobs:
            return
        job = jobs[0]
        if job.state == DONE and job.result:
            self._show_review(job.id)
        elif job.result and job.result.get("job_dir"):
            open_path(Path(job.result["job_dir"]))

    def _engine_text(self, job: Any) -> str:
        if job.result and job.result.get("candidates"):
            engines = dict.fromkeys(ENGINE_NAMES.get(c["engine"], c["engine"]) for c in job.result["candidates"])
            return " + ".join(engines)
        payload = job.payload
        if payload.get("kind") == "edit":
            return "FLUX.2 편집"
        if payload.get("product_preserve") and any(r["role"] == "PRODUCT" for r in payload.get("references") or []):
            return "합성 + FLUX.2"
        if payload.get("references"):
            return "FLUX.2"
        return {"preview": "FLUX.2", "balanced": "Z-Image", "best": "Z-Image + FLUX.2"}.get(payload.get("quality"), "")

    def _refresh_queue(self) -> None:
        selected = set(self.tree.selection())
        self.tree.delete(*self.tree.get_children())
        now = time.time()
        for job in self.queue.jobs:
            payload = job.payload
            progress = self.runner.progress.get(job.id, {})
            if job.state == RUNNING:
                progress_text = f"{int(progress.get('fraction', 0) * 100)}% {progress.get('message', '')}"
            elif job.state == PENDING and job.waiting_reason:
                progress_text = "PC 사용 중 — 자원 대기"
            elif job.state == FAILED:
                progress_text = job.error[:80]
            else:
                progress_text = "100%" if job.state == DONE else ""
            elapsed = ""
            if job.started:
                elapsed = f"{int((job.finished or now) - job.started)}초"
            title = payload.get("title") or payload.get("prompt") or payload.get("edit_instruction") or ""
            folder = (job.result or {}).get("job_dir", "")
            self.tree.insert("", "end", iid=job.id, values=(
                STATE_LABELS.get(job.state, job.state), payload.get("purpose_label") or payload.get("purpose", ""),
                title[:80], self._engine_text(job), label_for(QUALITY_LABELS, payload.get("quality", "")),
                progress_text, elapsed, folder))
        for item in selected & set(self.tree.get_children()):
            self.tree.selection_add(item)
        paused = " · 일시정지 상태" if self.queue.paused else ""
        self.status_label.configure(text=f"대기열: {self.runner.status}{paused}")
        self.resource_label.configure(text=self._resources_text)

    def _resource_loop(self) -> None:
        while True:
            try:
                snap = resource_snapshot()
                self._resources_text = (f"GPU 사용 {snap.gpu_used_mib} / {snap.gpu_total_mib} MiB · RAM 여유 "
                                        f"{(snap.ram_available_mib or 0) / 1024:.1f} GB · 커밋 여유 "
                                        f"{(snap.commit_free_mib or 0) / 1024:.1f} GB")
                self._dirty = True
            except Exception:
                pass
            time.sleep(5)

    # ================================================================== review tab
    def _build_review(self, tab: Any) -> None:
        top = ctk.CTkFrame(tab, fg_color="transparent")
        top.pack(fill="x")
        ctk.CTkLabel(top, text="작업").pack(side="left")
        self.review_job_var = ctk.StringVar(value="")
        self.review_menu = ctk.CTkOptionMenu(top, variable=self.review_job_var, values=[""], width=520,
                                             command=self._choose_review_job)
        self.review_menu.pack(side="left", padx=6)
        ctk.CTkButton(top, text="폴더 열기", width=90, command=self._open_review_folder).pack(side="left", padx=4)
        self.thumb_row = ctk.CTkFrame(tab, fg_color="transparent")
        self.thumb_row.pack(fill="x", pady=6)
        body = ctk.CTkFrame(tab, fg_color="transparent")
        body.pack(fill="both", expand=True)
        body.grid_columnconfigure(0, weight=3)
        body.grid_columnconfigure(1, weight=2)
        body.grid_rowconfigure(0, weight=1)
        self.large_label = ctk.CTkLabel(body, text="완료된 작업을 고르세요")
        self.large_label.grid(row=0, column=0, sticky="nsew")
        side = ctk.CTkScrollableFrame(body)
        side.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        small = ctk.CTkFrame(side, fg_color="transparent")
        small.pack(fill="x")
        self.preview340 = ctk.CTkLabel(small, text="")
        self.preview340.pack(side="left", padx=4)
        self.preview180 = ctk.CTkLabel(small, text="")
        self.preview180.pack(side="left", padx=4)
        crops = ctk.CTkFrame(side, fg_color="transparent")
        crops.pack(fill="x", pady=4)
        self.face_label = ctk.CTkLabel(crops, text="")
        self.face_label.pack(side="left", padx=4)
        self.product_label = ctk.CTkLabel(crops, text="")
        self.product_label.pack(side="left", padx=4)
        self.info_box = ctk.CTkTextbox(side, height=260, wrap="word")
        self.info_box.pack(fill="x", pady=4)
        actions = ctk.CTkFrame(side, fg_color="transparent")
        actions.pack(fill="x", pady=4)
        for index, (text, command) in enumerate((("채택", self._adopt), ("다시 생성", self._regenerate),
                                                 ("seed만 변경", self._reseed), ("프롬프트 수정", self._edit_prompt),
                                                 ("편집으로 보내기", self._send_to_edit), ("기존 후보와 비교", self._compare))):
            ctk.CTkButton(actions, text=text, width=120, command=command,
                          fg_color="#16a34a" if text == "채택" else None).grid(row=index // 3, column=index % 3, padx=3, pady=3)

    def _done_jobs(self) -> list[Any]:
        return [job for job in self.queue.jobs if job.state == DONE and (job.result or {}).get("candidates")]

    def _job_label(self, job: Any) -> str:
        payload = job.payload
        title = (payload.get("title") or payload.get("prompt") or payload.get("edit_instruction") or "")[:40]
        return f"{payload.get('purpose_label') or payload.get('purpose')} · {title} · {job.id[:6]}"

    def _refresh_review_menu(self) -> None:
        labels = [self._job_label(job) for job in self._done_jobs()]
        self.review_menu.configure(values=labels or [""])

    def _choose_review_job(self, label: str) -> None:
        job = next((j for j in self._done_jobs() if self._job_label(j) == label), None)
        if job:
            self._show_review(job.id)

    def _show_review(self, job_id: str, index: int = 0) -> None:
        job = self.queue.get(job_id)
        if not job or not job.result:
            return
        self.review_job_id, self.review_index = job_id, index
        self._refresh_review_menu()
        self.review_job_var.set(self._job_label(job))
        for child in self.thumb_row.winfo_children():
            child.destroy()
        for i, candidate in enumerate(job.result["candidates"]):
            try:
                with Image.open(candidate["files"].get("preview_340") or candidate["files"]["full"]) as opened:
                    thumb = fit_image(opened.convert("RGB"), (200, 140))
            except OSError:
                continue
            self._images.append(thumb)
            label = f"{i + 1}. {KIND_LABELS.get(candidate.get('kind'), '')}\n{ENGINE_NAMES.get(candidate['engine'], candidate['engine'])}"
            if candidate.get("warnings"):
                label += " ⚠"
            ctk.CTkButton(self.thumb_row, image=thumb, text=label, compound="top", width=210,
                          fg_color="#2563eb" if i == index else "#334155",
                          command=lambda i=i: self._show_review(job_id, i)).pack(side="left", padx=4)
        self._show_candidate(job, job.result["candidates"][index])
        self.tabs.set("후보 비교")

    def _show_candidate(self, job: Any, candidate: dict[str, Any]) -> None:
        files = candidate["files"]
        with Image.open(files["full"]) as opened:
            full = opened.convert("RGB")
        large = fit_image(full, (860, 560))
        self._images.append(large)
        self.large_label.configure(image=large, text="")
        for label, key, size in ((self.preview340, "preview_340", (340, 340)), (self.preview180, "preview_180", (180, 180)),
                                 (self.face_label, "face", (160, 160)), (self.product_label, "product", (160, 160))):
            if files.get(key) and Path(files[key]).exists():
                with Image.open(files[key]) as opened:
                    photo = fit_image(opened.convert("RGB"), size)
                self._images.append(photo)
                label.configure(image=photo, text="")
            else:
                label.configure(image=None, text="")
        lines = [f"종류: {KIND_LABELS.get(candidate.get('kind'), candidate.get('kind'))}",
                 f"엔진: {candidate.get('model')} · seed {candidate['seed']} · {candidate.get('seconds')}초 · "
                 f"GPU 피크 {candidate.get('peak_vram_mib')} MiB",
                 f"라이선스: {candidate.get('license')} · 상업적 사용 {'가능' if candidate.get('commercial_use') else '확인 필요'}",
                 f"번역 적용: {'예' if candidate.get('translation_applied') else '아니오'}",
                 "", "경고:" if candidate.get("warnings") else "경고: 없음"]
        lines += [f" • {w}" for w in candidate.get("warnings") or []]
        if candidate.get("note"):
            lines += ["", candidate["note"]]
        check = candidate.get("product_check")
        if check and not check.get("exact"):
            lines += ["", f"상품 색상 차이 ΔE: {check.get('color_drift')}",
                      f"상품 글자: '{check.get('text_reference')}' → '{check.get('text_candidate')}'"]
        lines += ["", f"원본 프롬프트: {candidate.get('original_prompt')}",
                  f"번역: {candidate.get('translated_prompt')}" if candidate.get("translation_applied") else "",
                  "", f"엔진 프롬프트: {candidate.get('compiled_prompt')}"]
        self.info_box.delete("1.0", "end")
        self.info_box.insert("1.0", "\n".join(line for line in lines if line is not None))

    def _current(self) -> tuple[Any, dict[str, Any]] | None:
        job = self.queue.get(self.review_job_id or "")
        if not job or not job.result:
            messagebox.showinfo("후보 비교", "먼저 완료된 작업의 후보를 고르세요.", parent=self)
            return None
        return job, job.result["candidates"][self.review_index]

    def _open_review_folder(self) -> None:
        current = self._current()
        if current:
            open_path(Path(current[0].result["job_dir"]))

    def _adopt(self) -> None:
        current = self._current()
        if not current:
            return
        job, candidate = current
        payload = job.payload
        purpose = self.purposes.get(payload.get("purpose", "")) or next(iter(self.purposes.values()))
        name = payload.get("title") or payload.get("prompt") or "image"
        written = export_candidate(candidate, purpose, export_dir=Path(job.result["job_dir"]) / "export",
                                   name=f"{safe_name(name, 30)}_{candidate['index']:02d}",
                                   jpg_quality=int(self.settings.get("jpg_quality") or 92),
                                   title=payload.get("title", ""), subtitle=payload.get("subtitle", ""),
                                   cta=payload.get("cta", ""), canvas=tuple(payload.get("canvas") or ()) or None)
        adopted = Path(job.result["job_dir"]) / "adopted.json"
        adopted.write_text(json.dumps({"candidate": candidate, "exports": written}, ensure_ascii=False, indent=2), encoding="utf-8")
        messagebox.showinfo("채택", "내보냈습니다:\n" + "\n".join(Path(p).name for p in written), parent=self)
        open_path(Path(job.result["job_dir"]) / "export")

    def _requeue(self, payload: dict[str, Any]) -> None:
        payload.pop("job_dir", None)
        self.queue.add(payload)
        self.runner.start()
        self._mark_dirty()
        self.tabs.set("대기열")

    def _regenerate(self) -> None:
        current = self._current()
        if current:
            self._requeue({**current[0].payload, "seed": random.randint(1, 999999)})

    def _reseed(self) -> None:
        current = self._current()
        if current:
            value = simpledialog.askinteger("seed만 변경", "새 seed", parent=self,
                                            initialvalue=int(current[1]["seed"]) + 1, minvalue=0)
            if value is not None:
                self._requeue({**current[0].payload, "seed": value})

    def _edit_prompt(self) -> None:
        current = self._current()
        if current:
            self._load_payload(current[0].payload)

    def _send_to_edit(self) -> None:
        current = self._current()
        if not current:
            return
        job, candidate = current
        instruction = simpledialog.askstring("편집으로 보내기", "어떻게 바꿀까요? (예: 인물은 그대로 두고 배경을 밤거리로)", parent=self)
        if not instruction:
            return
        payload = {k: v for k, v in job.payload.items() if k not in ("references", "product_preserve")}
        self._requeue({**payload, "kind": "edit", "edit_image": candidate["files"]["full"], "edit_instruction": instruction,
                       "prompt": "", "candidates": 1, "seed": random.randint(1, 999999)})

    def _compare(self) -> None:
        current = self._current()
        if not current:
            return
        _job, candidate = current
        if self.compare_pin is None or self.compare_pin is candidate:
            self.compare_pin = candidate
            messagebox.showinfo("기존 후보와 비교", "이 후보를 기준(A)으로 고정했습니다. 다른 후보를 고른 뒤 다시 누르세요.", parent=self)
            return
        window = ctk.CTkToplevel(self)
        window.title("후보 비교 A / B")
        for column, (name, item) in enumerate((("A", self.compare_pin), ("B", candidate))):
            with Image.open(item["files"]["full"]) as opened:
                photo = fit_image(opened.convert("RGB"), (700, 520))
            self._images.append(photo)
            ctk.CTkLabel(window, text=f"{name}: {ENGINE_NAMES.get(item['engine'], item['engine'])} · seed {item['seed']}",
                         font=ctk.CTkFont(weight="bold")).grid(row=0, column=column, pady=4)
            ctk.CTkLabel(window, text="", image=photo).grid(row=1, column=column, padx=6, pady=6)
        self.compare_pin = None

    # ================================================================== settings tab
    def _build_settings(self, tab: Any) -> None:
        self.setting_vars: dict[str, Any] = {}

        def path_row(label: str, key: str, hint: str) -> None:
            row = ctk.CTkFrame(tab, fg_color="transparent")
            row.pack(fill="x", padx=10, pady=6)
            ctk.CTkLabel(row, text=label, width=140, anchor="w").pack(side="left")
            var = ctk.StringVar(value=self.settings.get(key) or "")
            self.setting_vars[key] = var
            ctk.CTkEntry(row, textvariable=var, width=640, placeholder_text=hint).pack(side="left", padx=4)
            ctk.CTkButton(row, text="찾아보기", width=90,
                          command=lambda: var.set(filedialog.askdirectory(parent=self) or var.get())).pack(side="left")

        path_row("모델 폴더", "models_dir", f"비우면 {self.app_root / 'models'}")
        path_row("sd.cpp 폴더", "sdcpp_dir", "비우면 <모델 폴더>\\quality_v2\\sdcpp")
        path_row("출력 폴더", "output_dir", f"비우면 {self.app_root / 'creator_output'}")
        row = ctk.CTkFrame(tab, fg_color="transparent")
        row.pack(fill="x", padx=10, pady=6)
        ctk.CTkLabel(row, text="JPG 품질", width=140, anchor="w").pack(side="left")
        self.jpg_var = ctk.StringVar(value=str(self.settings.get("jpg_quality", 92)))
        ctk.CTkEntry(row, textvariable=self.jpg_var, width=60).pack(side="left")
        self.auto_var = ctk.BooleanVar(value=bool(self.settings.get("auto_start_when_free")))
        ctk.CTkCheckBox(tab, text="PC가 여유 있을 때 자동 시작 (작업을 추가하거나 앱을 열면 대기열을 자동으로 돌립니다)",
                        variable=self.auto_var).pack(anchor="w", padx=10, pady=6)
        row = ctk.CTkFrame(tab, fg_color="transparent")
        row.pack(fill="x", padx=10, pady=6)
        ctk.CTkLabel(row, text="자리 비움 모드: 키보드·마우스 입력이 없을 때만 시작 (분, 0=사용 안 함)").pack(side="left")
        self.idle_var = ctk.StringVar(value=str(self.settings.get("require_idle_minutes", 0)))
        ctk.CTkEntry(row, textvariable=self.idle_var, width=60).pack(side="left", padx=6)
        row = ctk.CTkFrame(tab, fg_color="transparent")
        row.pack(fill="x", padx=10, pady=6)
        ctk.CTkLabel(row, text="자원 부족 시 재확인 간격(초)").pack(side="left")
        self.recheck_var = ctk.StringVar(value=str(self.settings.get("recheck_seconds", 20)))
        ctk.CTkEntry(row, textvariable=self.recheck_var, width=60).pack(side="left", padx=6)
        buttons = ctk.CTkFrame(tab, fg_color="transparent")
        buttons.pack(fill="x", padx=10, pady=10)
        ctk.CTkButton(buttons, text="저장", command=self._save_settings).pack(side="left", padx=4)
        ctk.CTkButton(buttons, text="엔진 상태 확인", command=self._check_engines).pack(side="left", padx=4)
        ctk.CTkButton(buttons, text="출력 폴더 열기", command=lambda: open_path(resolve_output_dir(self.app_root, self.settings))).pack(side="left", padx=4)
        self.engine_status = ctk.CTkTextbox(tab, height=260)
        self.engine_status.pack(fill="both", expand=True, padx=10, pady=6)

    def _save_settings(self) -> None:
        try:
            jpg = max(50, min(100, int(self.jpg_var.get())))
            idle = max(0, int(self.idle_var.get()))
            recheck = max(5, int(self.recheck_var.get()))
        except ValueError:
            messagebox.showwarning("설정", "숫자를 확인하세요.", parent=self)
            return
        output_before = resolve_output_dir(self.app_root, self.settings)
        for key, var in self.setting_vars.items():
            self.settings[key] = var.get().strip()
        self.settings.update(jpg_quality=jpg, require_idle_minutes=idle, recheck_seconds=recheck,
                             auto_start_when_free=bool(self.auto_var.get()))
        save_creator_settings(self.app_root, self.settings)
        apply_backend_paths(self.settings)
        note = ""
        if resolve_output_dir(self.app_root, self.settings) != output_before:
            note = "\n출력 폴더를 바꾸면 새 대기열은 앱을 다시 열 때 적용됩니다."
        self._check_engines()
        messagebox.showinfo("설정", "저장했습니다. 환경변수 없이 다음 실행에도 이 경로를 씁니다." + note, parent=self)

    def _check_engines(self) -> None:
        models = resolve_models_dir(self.app_root, self.settings)
        status = engine_readiness(models)
        names = {"zimage_turbo": "Z-Image-Turbo (기본 생성)", "flux2_klein_4b": "FLUX.2-klein (레퍼런스/편집)",
                 "realvisxl_v5": "RealVisXL (대체)"}
        lines = [f"모델 폴더: {models}", ""]
        for key, info in status["engines"].items():
            lines.append(f"{'준비됨' if info['ready'] else '없음'} — {names[key]}")
            lines += [f"    없음: {m}" for m in info["missing"]]
        lines.append(f"{'준비됨' if status['translator'] else '없음'} — 한국어/일본어 프롬프트 번역(Qwen3)")
        if not all(info["ready"] for info in status["engines"].values()):
            lines += ["", "모델 준비: scripts\\prepare_quality_v2_models.py --models-dir <모델 폴더> (약 16 GB)"]
        self.engine_status.delete("1.0", "end")
        self.engine_status.insert("1.0", "\n".join(lines))

    # ================================================================== plumbing
    def _execute(self, payload: dict[str, Any], cancel: threading.Event, progress) -> dict[str, Any]:
        return run_creator_job(payload, cancel, models_dir=resolve_models_dir(self.app_root, self.settings),
                               output_root=resolve_output_dir(self.app_root, self.settings), progress=progress,
                               app_root=self.app_root)

    def _mark_dirty(self) -> None:
        self._dirty = True

    def _poll(self) -> None:
        if self._dirty:
            self._dirty = False
            try:
                self._refresh_queue()
                self._refresh_review_menu()
            except tk.TclError:
                return
        self.after(400, self._poll)

    def shutdown(self) -> None:
        self.runner.shutdown()


_studio: StudioWindow | None = None


def app_root_dir() -> Path:
    import sys
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def open_studio(master: Any, app_root: Path) -> StudioWindow:
    global _studio
    if _studio is None or not _studio.winfo_exists():
        _studio = StudioWindow(master, app_root)
    _studio.deiconify()
    _studio.lift()
    _studio.focus_force()
    return _studio


def shutdown_studio() -> None:
    if _studio is not None:
        _studio.shutdown()


def run_selftest(app_root: Path, out_dir: Path, timeout: float = 900.0) -> int:
    """Packaged-app acceptance: open the real studio, fill the form, enqueue like the button, wait, screenshot.

        CoverMorphStudio.exe --studio-selftest <out_dir>
    Uses only config/creator_settings.json for paths (no environment variables).
    """
    from PIL import ImageGrab

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ctk.set_appearance_mode("dark")
    root = ctk.CTk()
    root.withdraw()
    studio = StudioWindow(root, app_root)
    studio.geometry("1500x930+20+20")
    report: dict[str, Any] = {"models_dir": str(resolve_models_dir(studio.app_root, studio.settings)),
                              "env_models_dir": os.environ.get("COVERMORPH_MODELS_DIR", "")}
    started = time.time()

    def shot(name: str) -> None:
        studio.update()
        studio.lift()
        studio.attributes("-topmost", True)
        studio.update()
        x, y = studio.winfo_rootx(), studio.winfo_rooty()
        ImageGrab.grab((x, y, x + studio.winfo_width(), y + studio.winfo_height())).save(out_dir / f"{name}.png")

    def begin() -> None:
        studio._set_purpose_key("youtube_thumbnail")
        studio.preset_var.set(studio.prompt_presets["tc_solo_woman"].label)
        studio._apply_preset()
        studio.quality_var.set("빠른 미리보기")
        studio.count_var.set("1")
        studio._update_engine_note()
        shot("01_create")
        before = len(studio.queue.jobs)
        studio._enqueue(True)
        report["job_id"] = studio.queue.jobs[before].id if len(studio.queue.jobs) > before else ""
        root.after(1000, wait)

    def wait() -> None:
        job = studio.queue.get(report.get("job_id", ""))
        if job is not None and job.state == RUNNING and "queue_shot" not in report:
            studio._refresh_queue()
            shot("02_queue_running")
            report["queue_shot"] = True
        if job is not None and job.state in (DONE, FAILED, CANCELLED):
            report.update(state=job.state, error=job.error, seconds=round(time.time() - started, 1),
                          result_dir=(job.result or {}).get("job_dir"))
            studio._refresh_queue()
            shot("03_queue_done")
            if job.state == DONE:
                studio._show_review(job.id)
                shot("04_review")
            studio.tabs.set("설정")
            studio._check_engines()
            shot("05_settings")
            finish()
            return
        if time.time() - started > timeout:
            report.update(state="timeout")
            finish()
            return
        root.after(1000, wait)

    def finish() -> None:
        (out_dir / "selftest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        studio.shutdown()
        root.after(500, root.destroy)

    root.after(1500, begin)
    root.mainloop()
    return 0 if report.get("state") == DONE else 1
