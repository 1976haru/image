from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps, ImageEnhance

Rect = Tuple[int, int, int, int]

def pil_to_cv(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def cv_to_pil(arr: np.ndarray) -> Image.Image:
    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(arr)

def detect_text_boxes_easyocr(img: Image.Image, langs=("en",), min_conf=0.22) -> List[Rect]:
    try:
        import easyocr
    except Exception:
        return []
    reader = easyocr.Reader(list(langs), gpu=False, verbose=False)
    arr = np.array(img.convert("RGB"))
    results = reader.readtext(arr, detail=1, paragraph=False)
    boxes: List[Rect] = []
    for quad, text, conf in results:
        if conf < min_conf:
            continue
        xs = [p[0] for p in quad]
        ys = [p[1] for p in quad]
        x1, x2 = int(min(xs)), int(max(xs))
        y1, y2 = int(min(ys)), int(max(ys))
        pad = max(6, int(min(img.size) * 0.008))
        boxes.append((max(0,x1-pad), max(0,y1-pad),
                      min(img.width,x2+pad), min(img.height,y2+pad)))
    return boxes

def mask_from_boxes(size, boxes: List[Rect], dilation=9) -> np.ndarray:
    w, h = size
    mask = np.zeros((h, w), dtype=np.uint8)
    for x1,y1,x2,y2 in boxes:
        cv2.rectangle(mask, (x1,y1), (x2,y2), 255, -1)
    if dilation > 0:
        k = max(3, dilation | 1)
        mask = cv2.dilate(mask, np.ones((k,k), np.uint8), iterations=1)
    return mask

def mask_pil_from_boxes(size, boxes: List[Rect]) -> Image.Image:
    return Image.fromarray(mask_from_boxes(size, boxes)).convert("L")

def inpaint_text_opencv(img: Image.Image, boxes: List[Rect], radius=7) -> Image.Image:
    if not boxes:
        return img.copy()
    arr = pil_to_cv(img)
    mask = mask_from_boxes(img.size, boxes, dilation=9)
    result = cv2.inpaint(arr, mask, radius, cv2.INPAINT_TELEA)
    return cv_to_pil(result)

def detect_largest_face(img: Image.Image) -> Optional[Rect]:
    if not hasattr(cv2, "CascadeClassifier") or not hasattr(cv2, "data"):
        return None
    gray = cv2.cvtColor(pil_to_cv(img), cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    faces = cascade.detectMultiScale(gray, 1.08, 5, minSize=(48,48))
    if len(faces) == 0:
        return None
    x,y,w,h = max(faces, key=lambda f:f[2]*f[3])
    return (x,y,x+w,y+h)

def mild_enhance(img: Image.Image, strength=1.0) -> Image.Image:
    arr = pil_to_cv(img)
    blur = cv2.GaussianBlur(arr, (0,0), 1.0)
    alpha = 1.0 + 0.16 * strength
    sharp = cv2.addWeighted(arr, alpha, blur, -(alpha-1.0), 0)
    sharp = np.clip(sharp,0,255).astype(np.uint8)
    out = cv_to_pil(sharp)
    out = ImageEnhance.Contrast(out).enhance(1.02)
    return out

def blurred_background(img: Image.Image, size, blur=36):
    bg = ImageOps.fit(img, size, method=Image.Resampling.LANCZOS)
    bg = bg.filter(ImageFilter.GaussianBlur(blur))
    bg = ImageEnhance.Brightness(bg).enhance(0.88)
    return bg

def build_outpaint_canvas(img: Image.Image, size, person_mask: Optional[Image.Image] = None,
                          anchor="center", margin=0.04):
    """Create an outpaint canvas and a white mask only where pixels may be generated.

    The returned protect layer is pasted back after AI generation, guaranteeing that
    the source photograph/person pixels are not changed.
    """
    tw, th = size
    scale = min((tw * (1 - 2*margin)) / img.width,
                (th * (1 - 2*margin)) / img.height)
    nw, nh = max(1, int(img.width*scale)), max(1, int(img.height*scale))
    fg = img.resize((nw, nh), Image.Resampling.LANCZOS)
    if anchor == "right" and tw > th:
        x = tw - nw - int(tw*margin)
    elif anchor == "left" and tw > th:
        x = int(tw*margin)
    else:
        x = (tw-nw)//2
    y = (th-nh)//2
    base = blurred_background(img, size, blur=max(20, int(min(size)*0.025)))
    base.paste(fg, (x, y))
    gen_mask = Image.new("L", size, 255)
    # Slight overlap lets the model blend seams while keeping the core unchanged.
    inset = max(8, int(min(size)*0.012))
    gen_mask.paste(0, (x+inset, y+inset, x+nw-inset, y+nh-inset))
    gen_mask = gen_mask.filter(ImageFilter.GaussianBlur(max(4, inset//2)))
    protect = Image.new("RGBA", size, (0,0,0,0))
    if person_mask is not None:
        pm = person_mask.resize((nw,nh), Image.Resampling.LANCZOS)
        protect.paste(fg.convert("RGBA"), (x,y), pm)
    else:
        hard = Image.new("L", (nw,nh), 255)
        protect.paste(fg.convert("RGBA"), (x,y), hard)
    return base, gen_mask, protect

def restore_protected_pixels(generated: Image.Image, protect: Image.Image):
    out = generated.convert("RGBA")
    out.alpha_composite(protect)
    return out.convert("RGB")

def crop_subject_region(img: Image.Image):
    face = detect_largest_face(img)
    if not face:
        return img.copy()
    x1,y1,x2,y2 = face
    fw,fh=x2-x1,y2-y1
    cx=(x1+x2)/2
    cy=(y1+y2)/2
    left=max(0, int(cx-fw*3.1))
    right=min(img.width, int(cx+fw*3.1))
    top=max(0, int(cy-fh*2.35))
    bottom=min(img.height, int(cy+fh*4.8))
    if right-left < img.width*0.45:
        return img.copy()
    return img.crop((left,top,right,bottom))

def paste_subject_on_canvas(img: Image.Image, target_size, anchor="right",
                            subject_ratio=0.42, safe_ratio=0.35):
    tw,th=target_size
    bg=blurred_background(img,(tw,th),blur=max(24,int(min(tw,th)*0.028)))
    subj=crop_subject_region(img)

    if tw > th:
        max_w=int(tw*subject_ratio)
        max_h=int(th*0.96)
    else:
        max_w=int(tw*0.94)
        max_h=int(th*subject_ratio)

    scale=min(max_w/subj.width,max_h/subj.height)
    nw=max(1,int(subj.width*scale))
    nh=max(1,int(subj.height*scale))
    fg=subj.resize((nw,nh),Image.Resampling.LANCZOS)

    if tw > th:
        if anchor=="right":
            x=max(int(tw*safe_ratio), tw-int(tw*0.035)-nw)
        elif anchor=="left":
            x=int(tw*0.035)
        else:
            x=(tw-nw)//2
        y=(th-nh)//2
    else:
        x=(tw-nw)//2
        y=max(int(th*0.05),(th-nh)//2)

    mask=Image.new("L",(nw,nh),255).filter(
        ImageFilter.GaussianBlur(max(6,int(min(nw,nh)*0.018)))
    )
    bg.paste(fg,(x,y),mask)
    return bg

def make_text_safe_landscape(img: Image.Image, preset):
    out=paste_subject_on_canvas(
        img,(1920,1080),
        anchor=preset.person_anchor_16x9,
        subject_ratio=preset.person_ratio_16x9,
        safe_ratio=preset.text_safe_ratio_16x9
    )
    safe_w=int(out.width*preset.text_safe_ratio_16x9)
    zone=out.crop((0,0,safe_w,out.height)).filter(ImageFilter.GaussianBlur(3))
    zone=ImageEnhance.Contrast(zone).enhance(0.88)
    out.paste(zone,(0,0))
    return out

def make_shorts(img: Image.Image, preset):
    return paste_subject_on_canvas(
        img,(1080,1920),
        anchor=preset.person_anchor_9x16,
        subject_ratio=preset.person_ratio_9x16,
        safe_ratio=0.0
    )

def make_square(img: Image.Image):
    return ImageOps.fit(img,(1400,1400),method=Image.Resampling.LANCZOS)

def save_jpg(img: Image.Image, path: Path, quality=96):
    path.parent.mkdir(parents=True,exist_ok=True)
    img.convert("RGB").save(path,"JPEG",quality=quality,subsampling=0,optimize=True)
