from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple
import json, sys
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
from PIL import Image, ImageTk

from .processor import (
    detect_text_boxes_easyocr, inpaint_text_opencv, mask_pil_from_boxes,
    mild_enhance, make_square, make_text_safe_landscape, make_shorts, save_jpg,
    build_outpaint_canvas, restore_protected_pixels
)
from .presets import PRESETS, preset_to_dict
from .ai_plugins import AIBackends
from .logger import write_log, write_exception

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

def app_root():
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]

class CoverMorphApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.root_dir=app_root()
        self.ai=AIBackends(self.root_dir)

        self.title("CoverMorph Studio v0.5")
        self.geometry("1460x900")
        self.minsize(1220,760)

        self.files: List[Path] = []
        self.original: Optional[Image.Image] = None
        self.current: Optional[Image.Image] = None
        self.before_image: Optional[Image.Image] = None
        self.after_image: Optional[Image.Image] = None
        self.showing_after=True

        self.preview_tk=None
        self.text_boxes: List[Tuple[int,int,int,int]]=[]
        self.preview_scale=1.0
        self.preview_offset=(0,0)
        self.drag_start=None
        self.output_dir: Optional[Path]=None

        self.build_ui()
        self.refresh_ai_status()
        write_log(self.root_dir, "CoverMorph Studio v0.5 started")

    def build_ui(self):
        self.grid_columnconfigure(1,weight=1)
        self.grid_rowconfigure(0,weight=1)

        left=ctk.CTkScrollableFrame(self,width=360)
        left.grid(row=0,column=0,sticky="nsew",padx=10,pady=10)

        right=ctk.CTkFrame(self)
        right.grid(row=0,column=1,sticky="nsew",padx=(0,10),pady=10)
        right.grid_columnconfigure(0,weight=1)
        right.grid_rowconfigure(0,weight=1)

        ctk.CTkLabel(left,text="CoverMorph Studio",font=ctk.CTkFont(size=25,weight="bold")).pack(pady=(12,2))
        ctk.CTkLabel(left,text="v0.5 · Protected Outpaint Edition",text_color="#9ca3af").pack(pady=(0,12))

        ctk.CTkButton(left,text="① 커버 이미지 불러오기",command=self.open_files,height=40).pack(fill="x",padx=10,pady=5)
        ctk.CTkButton(left,text="② 출력 폴더 지정",command=self.choose_output,height=38).pack(fill="x",padx=10,pady=5)

        ctk.CTkLabel(left,text="채널 프리셋",anchor="w",font=ctk.CTkFont(weight="bold")).pack(fill="x",padx=12,pady=(16,5))
        self.preset_name=ctk.StringVar(value="OldPopLounge")
        ctk.CTkOptionMenu(left,variable=self.preset_name,values=list(PRESETS.keys()),command=self.update_preset_desc).pack(fill="x",padx=10)
        self.preset_desc=ctk.CTkLabel(left,text=PRESETS["OldPopLounge"].description,wraplength=315,justify="left",text_color="#cbd5e1")
        self.preset_desc.pack(fill="x",padx=12,pady=6)

        ctk.CTkLabel(left,text="OCR 언어",anchor="w",font=ctk.CTkFont(weight="bold")).pack(fill="x",padx=12,pady=(12,5))
        self.ocr_lang=ctk.StringVar(value="영어")
        ctk.CTkOptionMenu(left,variable=self.ocr_lang,values=["영어","한국어+영어","일본어+영어"]).pack(fill="x",padx=10)

        ctk.CTkLabel(left,text="AI 엔진",anchor="w",font=ctk.CTkFont(weight="bold")).pack(fill="x",padx=12,pady=(15,5))
        self.ai_status=ctk.CTkLabel(left,text="",wraplength=315,justify="left")
        self.ai_status.pack(fill="x",padx=12,pady=4)
        ctk.CTkButton(left,text="AI 상태 새로고침",command=self.refresh_ai_status,height=30,fg_color="#4b5563").pack(fill="x",padx=10,pady=4)

        self.prefer_lama=ctk.BooleanVar(value=True)
        self.prefer_esrgan=ctk.BooleanVar(value=True)
        self.auto_remove=ctk.BooleanVar(value=True)
        self.enhance=ctk.BooleanVar(value=True)
        self.out_square=ctk.BooleanVar(value=True)
        self.out_thumb=ctk.BooleanVar(value=True)
        self.out_shorts=ctk.BooleanVar(value=True)
        self.use_outpaint=ctk.BooleanVar(value=False)
        self.protect_person=ctk.BooleanVar(value=True)

        ctk.CTkCheckBox(left,text="LaMa 우선 글자 제거",variable=self.prefer_lama).pack(anchor="w",padx=13,pady=(10,3))
        ctk.CTkCheckBox(left,text="Real-ESRGAN 우선 업스케일",variable=self.prefer_esrgan).pack(anchor="w",padx=13,pady=3)
        ctk.CTkCheckBox(left,text="SDXL AI 배경 확장 (고성능 PC)",variable=self.use_outpaint).pack(anchor="w",padx=13,pady=3)
        ctk.CTkCheckBox(left,text="사람 원본 픽셀 보호",variable=self.protect_person).pack(anchor="w",padx=13,pady=3)
        ctk.CTkCheckBox(left,text="글자 자동삭제",variable=self.auto_remove).pack(anchor="w",padx=13,pady=3)
        ctk.CTkCheckBox(left,text="기본 화질 보정",variable=self.enhance).pack(anchor="w",padx=13,pady=3)
        ctk.CTkCheckBox(left,text="1:1 클린커버",variable=self.out_square).pack(anchor="w",padx=13,pady=3)
        ctk.CTkCheckBox(left,text="16:9 썸네일",variable=self.out_thumb).pack(anchor="w",padx=13,pady=3)
        ctk.CTkCheckBox(left,text="9:16 숏츠",variable=self.out_shorts).pack(anchor="w",padx=13,pady=3)

        ctk.CTkLabel(left,text="AI 배경 프롬프트",anchor="w",font=ctk.CTkFont(weight="bold")).pack(fill="x",padx=12,pady=(12,4))
        self.outpaint_prompt=ctk.CTkTextbox(left,height=72)
        self.outpaint_prompt.pack(fill="x",padx=10)
        self.outpaint_prompt.insert("1.0","natural photographic continuation of the existing background, seamless lighting, realistic detail, no text")

        ctk.CTkButton(left,text="글자 자동 탐지",command=self.detect_text,height=34,fg_color="#4b5563").pack(fill="x",padx=10,pady=(13,4))
        ctk.CTkButton(left,text="미리보기 글자 제거",command=self.preview_remove,height=34,fg_color="#4b5563").pack(fill="x",padx=10,pady=4)
        ctk.CTkButton(left,text="Before / After 전환",command=self.toggle_before_after,height=34,fg_color="#374151").pack(fill="x",padx=10,pady=4)
        ctk.CTkButton(left,text="마스크 초기화",command=self.clear_boxes,height=32,fg_color="#374151").pack(fill="x",padx=10,pady=4)

        self.status=ctk.CTkLabel(left,text="커버 이미지를 불러오세요.",wraplength=315,justify="left")
        self.status.pack(fill="x",padx=12,pady=(15,10))

        ctk.CTkButton(left,text="▶ 원클릭 AI 자동 변환",command=self.run_pipeline,height=52,
                      font=ctk.CTkFont(size=16,weight="bold"),
                      fg_color="#198754",hover_color="#146c43").pack(fill="x",padx=10,pady=(8,16))

        self.canvas=tk.Canvas(right,bg="#0f172a",highlightthickness=0,cursor="cross")
        self.canvas.grid(row=0,column=0,sticky="nsew",padx=8,pady=8)
        self.canvas.bind("<ButtonPress-1>",self.on_press)
        self.canvas.bind("<ButtonRelease-1>",self.on_release)
        self.canvas.bind("<Configure>",lambda e:self.draw_preview())

        footer=ctk.CTkFrame(right)
        footer.grid(row=1,column=0,sticky="ew",padx=8,pady=(0,8))
        ctk.CTkLabel(footer,text="자동 탐지에서 빠진 글자는 미리보기에서 직접 드래그해 마스크를 추가하세요.").pack(side="left",padx=12,pady=8)
        self.view_label=ctk.CTkLabel(footer,text="VIEW: AFTER")
        self.view_label.pack(side="right",padx=12)

    def refresh_ai_status(self):
        st=self.ai.status()
        txt=(
            f"LaMa: {'사용 가능' if st['lama'] else '미설치'}\n"
            f"Real-ESRGAN: {'사용 가능' if st['realesrgan'] else '실행파일 없음'}\n"
            f"CUDA: {'감지됨' if st['cuda'] else '없음/미사용'}"
            f"\n사람 분리: {'사용 가능' if st['rembg'] else '미설치'}"
            f"\nSDXL: {'사용 가능' if st['sdxl'] else '미설치'}"
        )
        self.ai_status.configure(text=txt)

    def lang_codes(self):
        v=self.ocr_lang.get()
        if "한국어" in v:return ("ko","en")
        if "일본어" in v:return ("ja","en")
        return ("en",)

    def update_preset_desc(self,*_):
        self.preset_desc.configure(text=PRESETS[self.preset_name.get()].description)

    def open_files(self):
        paths=filedialog.askopenfilenames(filetypes=[("Images","*.jpg *.jpeg *.png *.webp *.bmp")])
        if not paths:return
        self.files=[Path(p) for p in paths]
        self.original=Image.open(self.files[0]).convert("RGB")
        self.current=self.original.copy()
        self.before_image=self.original.copy()
        self.after_image=self.current.copy()
        self.text_boxes=[]
        self.status.configure(text=f"{len(self.files)}장 선택\n첫 이미지: {self.files[0].name}")
        self.draw_preview()

    def choose_output(self):
        d=filedialog.askdirectory(title="자동 저장 폴더")
        if d:
            self.output_dir=Path(d)
            self.status.configure(text=f"출력 폴더:\n{self.output_dir}")

    def detect_text(self):
        if not self.current:return
        self.status.configure(text="글자 탐지 중...")
        self.update_idletasks()
        try:
            self.text_boxes=detect_text_boxes_easyocr(self.current,self.lang_codes())
            self.status.configure(text=f"글자 영역 {len(self.text_boxes)}개 탐지\n빠진 글자는 직접 드래그하세요.")
            self.draw_preview()
        except Exception as e:
            write_exception(self.root_dir, "OCR", e)
            messagebox.showerror("OCR 오류",str(e))

    def remove_with_best_backend(self, img, boxes):
        if not boxes:
            return img.copy(), "No text mask"
        if self.prefer_lama.get() and self.ai.lama_available():
            try:
                return self.ai.inpaint(img,mask_pil_from_boxes(img.size,boxes))
            except Exception as e:
                write_exception(self.root_dir,"LaMa fallback",e)
        return inpaint_text_opencv(img,boxes), "OpenCV Telea"

    def preview_remove(self):
        if not self.current or not self.text_boxes:return
        self.before_image=self.current.copy()
        out,engine=self.remove_with_best_backend(self.current,self.text_boxes)
        self.current=out
        self.after_image=out.copy()
        self.showing_after=True
        self.text_boxes=[]
        self.status.configure(text=f"미리보기 제거 완료 · {engine}")
        self.draw_preview()

    def toggle_before_after(self):
        if self.before_image is None or self.after_image is None:return
        self.showing_after=not self.showing_after
        self.current=(self.after_image if self.showing_after else self.before_image).copy()
        self.view_label.configure(text="VIEW: AFTER" if self.showing_after else "VIEW: BEFORE")
        self.draw_preview()

    def clear_boxes(self):
        self.text_boxes=[]
        self.draw_preview()

    def on_press(self,e):
        if self.current:self.drag_start=(e.x,e.y)

    def on_release(self,e):
        if not self.current or not self.drag_start:return
        x1,y1=self.drag_start;x2,y2=e.x,e.y
        ox,oy=self.preview_offset;s=self.preview_scale
        ix1=int((min(x1,x2)-ox)/s); iy1=int((min(y1,y2)-oy)/s)
        ix2=int((max(x1,x2)-ox)/s); iy2=int((max(y1,y2)-oy)/s)
        ix1=max(0,min(ix1,self.current.width)); ix2=max(0,min(ix2,self.current.width))
        iy1=max(0,min(iy1,self.current.height)); iy2=max(0,min(iy2,self.current.height))
        if ix2-ix1>5 and iy2-iy1>5:self.text_boxes.append((ix1,iy1,ix2,iy2))
        self.drag_start=None
        self.draw_preview()

    def draw_preview(self):
        self.canvas.delete("all")
        if not self.current:return
        cw=max(100,self.canvas.winfo_width());ch=max(100,self.canvas.winfo_height())
        s=min((cw-30)/self.current.width,(ch-30)/self.current.height)
        pw=int(self.current.width*s);ph=int(self.current.height*s)
        p=self.current.resize((pw,ph),Image.Resampling.LANCZOS)
        self.preview_tk=ImageTk.PhotoImage(p)
        ox=(cw-pw)//2;oy=(ch-ph)//2
        self.preview_scale=s;self.preview_offset=(ox,oy)
        self.canvas.create_image(ox,oy,image=self.preview_tk,anchor="nw")
        for a,b,c,d in self.text_boxes:
            self.canvas.create_rectangle(ox+a*s,oy+b*s,ox+c*s,oy+d*s,outline="#ff416c",width=2)

    def upscale_best(self,img):
        if self.prefer_esrgan.get() and self.ai.realesrgan_available():
            try:
                return self.ai.upscale(img,scale=2,model="realesrgan-x4plus")
            except Exception as e:
                write_exception(self.root_dir,"Real-ESRGAN fallback",e)
        return mild_enhance(img,1.0), "Local sharpen"

    def make_ai_format(self, img, size, anchor):
        person_mask=None; seg_engine="whole image protect"
        if self.protect_person.get() and self.ai.rembg_available():
            try:
                person_mask,seg_engine=self.ai.person_mask(img)
            except Exception as e:
                write_exception(self.root_dir,"Person segmentation fallback",e)
        canvas,mask,protect=build_outpaint_canvas(img,size,person_mask,anchor=anchor)
        prompt=self.outpaint_prompt.get("1.0","end").strip()
        negative="text, letters, logo, watermark, duplicate person, extra limbs, distorted face, oversaturated"
        generated,engine=self.ai.outpaint(canvas,mask,prompt,negative)
        return restore_protected_pixels(generated,protect), f"{engine} + {seg_engine}"

    def run_pipeline(self):
        if not self.files:
            messagebox.showinfo("안내","먼저 커버 이미지를 불러오세요.")
            return
        if not self.output_dir:
            self.choose_output()
            if not self.output_dir:return

        preset=PRESETS[self.preset_name.get()]
        total=0; errors=[]
        for idx,p in enumerate(self.files,1):
            try:
                img=Image.open(p).convert("RGB")
                clean=img
                boxes=[]
                remove_engine="Skipped"

                if self.auto_remove.get():
                    boxes=detect_text_boxes_easyocr(img,self.lang_codes())
                    if idx==1 and self.text_boxes:
                        boxes=list({tuple(x) for x in (boxes+self.text_boxes)})
                    clean,remove_engine=self.remove_with_best_backend(img,boxes)

                if self.enhance.get():
                    clean=mild_enhance(clean,preset.enhance_strength)

                upscale_engine="Skipped"
                if self.prefer_esrgan.get() and self.ai.realesrgan_available():
                    try:
                        clean,upscale_engine=self.ai.upscale(clean,scale=2,model="realesrgan-x4plus")
                    except Exception as e:
                        write_exception(self.root_dir,"Real-ESRGAN pipeline fallback",e)
                        upscale_engine="Fallback local"

                item_dir=self.output_dir/p.stem
                item_dir.mkdir(parents=True,exist_ok=True)

                if self.out_square.get():
                    save_jpg(make_square(clean),item_dir/f"{p.stem}_clean_1x1_1400x1400.jpg");total+=1
                outpaint_engine="Disabled"
                if self.out_thumb.get():
                    if self.use_outpaint.get() and self.ai.sdxl_available():
                        thumb,outpaint_engine=self.make_ai_format(clean,(1920,1080),preset.person_anchor_16x9)
                    else:
                        thumb=make_text_safe_landscape(clean,preset)
                        if self.use_outpaint.get(): outpaint_engine="Fallback blur canvas"
                    save_jpg(thumb,item_dir/f"{p.stem}_thumb_16x9_1920x1080.jpg");total+=1
                if self.out_shorts.get():
                    if self.use_outpaint.get() and self.ai.sdxl_available():
                        shorts,outpaint_engine=self.make_ai_format(clean,(1080,1920),preset.person_anchor_9x16)
                    else:
                        shorts=make_shorts(clean,preset)
                        if self.use_outpaint.get(): outpaint_engine="Fallback blur canvas"
                    save_jpg(shorts,item_dir/f"{p.stem}_shorts_9x16_1080x1920.jpg");total+=1

                meta={
                    "version":"0.5.0",
                    "source":p.name,
                    "preset":preset_to_dict(preset),
                    "ocr_languages":self.lang_codes(),
                    "text_boxes":len(boxes),
                    "inpaint_engine":remove_engine,
                    "upscale_engine":upscale_engine,
                    "outpaint_engine":outpaint_engine,
                    "person_pixel_protection":self.protect_person.get(),
                    "outpaint_prompt":self.outpaint_prompt.get("1.0","end").strip(),
                    "outputs":{
                        "square":self.out_square.get(),
                        "thumbnail":self.out_thumb.get(),
                        "shorts":self.out_shorts.get()
                    }
                }
                (item_dir/"covermorph_job.json").write_text(
                    json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8"
                )
                write_log(self.root_dir,f"OK {p.name} | inpaint={remove_engine} | upscale={upscale_engine}")
                self.status.configure(text=f"처리 중 {idx}/{len(self.files)}\n{p.name}\n제거: {remove_engine}\n업스케일: {upscale_engine}")
                self.update_idletasks()
            except Exception as e:
                errors.append(f"{p.name}: {e}")
                write_exception(self.root_dir,f"FAILED {p.name}",e)

        self.status.configure(text=f"완료: {total}개 결과물\n{self.output_dir}\n오류: {len(errors)}")
        messagebox.showinfo("CoverMorph 완료",f"{total}개 이미지 생성 완료\n오류 {len(errors)}건")
