# CoverMorph Studio v0.5.4

CoverMorph Studio는 음원 커버 이미지를 `1:1` 클린 커버, `16:9` 유튜브 썸네일, `9:16` 숏츠/Reels/TikTok 이미지로 변환하는 Windows용 로컬 프로그램입니다. 기본 기능은 외부 API 없이 동작하며, 선택형 AI 도구를 설치하면 글자 제거, 업스케일, 자연 배경 확장을 더 고급 방식으로 처리합니다.

## 주요 기능

- 최대 5장 다중 이미지 선택 및 일괄 변환
- 이미지별 `1:1`, `16:9`, `9:16` 출력 규격 개별 선택
- `1400x1400` 클린 커버 JPG 생성
- `1920x1080` 또는 `1280x720` 유튜브 썸네일 JPG 생성
- `1080x1920` 세로 숏츠 이미지 JPG 생성
- 기본값인 `AI 자연 배경 확장`으로 16:9와 9:16 화면 전체 채우기
- SDXL 실패 시 손상 가능성이 있는 대체 이미지를 자동 저장하지 않고 해당 출력만 실패 처리
- EasyOCR 언어별 Reader 캐시: 영어, 한국어+영어, 일본어+영어
- LaMa 선택형 글자 제거, 실패 시 OpenCV NS/Telea 품질 비교 fallback
- Real-ESRGAN 선택형 업스케일, 미설치 또는 실패 시 로컬 선명도 보정 fallback
- 원본 핵심 영역 보호 및 SDXL 생성 후 보호 픽셀 복원
- 출력 폴더, 출력 규격, 해상도, 프리셋, OCR 언어를 `config/settings.json`에 저장
- 파일별 `covermorph_job.json`과 날짜별 로그 저장

## 기본 설치

1. Python 3.11 또는 3.12를 설치합니다.
2. 저장소 폴더에서 `INSTALL.bat`을 실행합니다.
3. 설치가 끝나면 `RUN.bat`을 실행합니다.

`INSTALL.bat`은 Python 3.11을 먼저 찾고, 없으면 3.12를 찾습니다. 지원하지 않는 Python 버전만 있으면 명확한 오류를 표시하고 중단합니다.

## 선택형 AI 설치

기본 설치만으로도 OpenCV 글자 제거, 로컬 선명도 보정, 로컬 배경 확장, 1:1/16:9/9:16 저장이 가능합니다. 아래 AI 기능은 필요할 때만 추가로 설치하세요.

### LaMa 글자 제거

```bat
INSTALL_AI_LAMA.bat
```

LaMa가 설치되어 있으면 글자 제거에 우선 사용합니다. 실행 중 오류가 나면 프로그램은 멈추지 않고 OpenCV NS와 Telea 결과를 경계 품질로 비교해 선택합니다.

### SDXL AI 자연 배경 확장

```bat
INSTALL_AI_OUTPAINT.bat
```

SDXL 모델은 프로그램에서 처음 사용할 때 자동 다운로드됩니다. 첫 다운로드는 시간이 오래 걸릴 수 있고, 인터넷 연결과 저장 공간이 필요합니다. 모델 다운로드 실패, CUDA 오류, 메모리 부족, 추론 오류가 발생하면 해당 출력은 실패 처리하고 저장하지 않습니다. `자연 배경 확장`, `스마트 크롭`, `블러 배경`을 직접 선택해 다시 처리하면 선택한 로컬 엔진만 사용합니다. 실패 원인은 로그와 작업 JSON에 기록됩니다.

### Real-ESRGAN 설치 위치

Real-ESRGAN NCNN-Vulkan은 직접 내려받아 아래 위치 중 하나에 넣습니다.

```text
tools/
  realesrgan-ncnn-vulkan.exe
  models/
    realesrgan-x4plus.param
    realesrgan-x4plus.bin
```

프로그램은 다음 위치를 순서대로 찾습니다.

- 프로그램 루트의 `tools`
- EXE 옆의 `tools`
- PyInstaller 실행 중 `sys._MEIPASS`
- 시스템 `PATH`

실행 파일이나 모델이 없으면 작업은 중단되지 않고 로컬 선명도 보정으로 전환됩니다.

## 실행 방법

1. `RUN.bat`을 실행합니다.
2. `이미지 추가` 버튼으로 1~5장을 선택하거나, 지원 환경에서는 창으로 드래그앤드롭합니다.
3. 출력 폴더를 확인하거나 `찾아보기`로 변경합니다.
4. 이미지 목록에서 각 행의 `1:1`, `16:9`, `9:16` 체크박스를 선택합니다.
5. 필요한 경우 이미지 행을 클릭하고 프리셋, OCR 언어, 확장 방식, 위치, 확대축소를 조절합니다.
6. `원클릭 자동 변환`을 누릅니다.

출력 규격이 하나도 선택되지 않으면 변환은 시작되지 않고 안내창이 표시됩니다.

## 처음 사용할 때 권장 설정

- 출력 규격: 필요한 규격만 선택
- 중복 파일 처리: `새 번호 붙이기`
- OCR 언어: 영어 커버는 `영어`, 한글 커버는 `한국어+영어`
- 이미지 확장 방식: `AI 자연 배경 확장`
- 원본 핵심 영역 보호: 켜기
- Real-ESRGAN: 설치하지 않았다면 꺼도 됩니다.

## 일반 PC용 설정

NVIDIA GPU가 없거나 AI 모델을 설치하지 않은 PC에서는 다음 조합이 안정적입니다.

- 이미지 확장 방식: `자연 배경 확장`
- LaMa: 미설치 가능
- Real-ESRGAN: 미설치 가능
- SDXL: 미설치 가능

이 경우 16:9는 좌우 배경을, 9:16은 위아래 배경을 로컬 방식으로 확장해 화면을 채웁니다.

## NVIDIA GPU PC용 설정

NVIDIA GPU와 충분한 VRAM이 있으면 다음 조합을 권장합니다.

- `INSTALL_AI_OUTPAINT.bat` 실행
- 이미지 확장 방식: `AI 자연 배경 확장`
- 원본 핵심 영역 보호: 켜기
- 사람 분할 마스크로 얼굴/인물 보호: 켜기
- 필요 시 Real-ESRGAN 설치

AI 자연 배경 확장은 원본 핵심 피사체를 보호하고, 부족한 배경 영역만 생성한 뒤 보호 영역을 원본 픽셀로 다시 합성합니다.

## 이미지 확장 방식

- `AI 자연 배경 확장`: 기본값입니다. SDXL로 부족한 배경을 만들고 원본 핵심 영역을 보호합니다.
- `스마트 크롭`: 원본을 확대해 화면 전체를 채웁니다. 일부 가장자리는 잘릴 수 있습니다.
- `자연 배경 확장`: AI 없이 가장자리 미러링, 색상 연장, 블렌딩을 조합합니다.
- `블러 배경`: 기존 방식입니다. 사용자가 직접 선택했을 때만 사용합니다.
- `원본 전체 맞춤`: 원본 전체를 표시합니다. 여백이 생길 수 있습니다.

OldPopLounge 프리셋의 16:9 출력은 인물이나 주요 피사체를 오른쪽에 배치하고 왼쪽 약 35~40%를 문구 공간으로 확보합니다. 9:16 출력은 주요 피사체를 중앙에 유지하고 위아래 배경을 확장합니다.

## 출력 폴더와 설정

설정 파일은 프로그램 폴더 아래에 자동 생성됩니다.

```text
config/settings.json
```

예시:

```json
{
  "output_directory": "D:\\03_image\\CoverMorph_Output",
  "output_square": true,
  "output_thumbnail": true,
  "output_shorts": true,
  "thumbnail_resolution": "1920x1080",
  "extension_mode": "ai_natural",
  "protect_core": true,
  "duplicate_policy": "new_number",
  "last_preset": "OldPopLounge",
  "last_ocr_language": "영어"
}
```

출력 폴더를 아직 지정하지 않았다면 첫 번째 원본 이미지 폴더 아래 `CoverMorph_Output`을 기본값으로 사용합니다.

```text
D:\음원커버\19 음악커버 1.png
D:\음원커버\CoverMorph_Output
```

저장된 출력 폴더가 삭제되었거나 사용할 수 없으면 프로그램은 종료되지 않고 새 출력 폴더 선택을 안내합니다. 직접 입력한 경로는 변환 시작 전에 존재 여부, 생성 가능 여부, 쓰기 권한을 검사합니다. 한글과 공백이 포함된 Windows 경로도 지원합니다.

## 출력 폴더 구조

기본 출력 폴더 아래에 원본 파일명별 하위 폴더가 만들어집니다.

```text
D:\03_image\CoverMorph_Output
  cover01
    cover01_thumbnail_16x9_1920x1080.jpg
    cover01_shorts_9x16_1080x1920.jpg
    covermorph_job.json
  cover02
    cover02_shorts_9x16_1080x1920.jpg
    covermorph_job.json
```

같은 파일명이 이미 있으면 기본값은 원본 보호를 위해 `새 번호 붙이기`입니다.

```text
image_thumbnail_16x9_1920x1080_02.jpg
```

## 로그와 작업 JSON

오류 상세 내용은 다음 위치에 저장됩니다.

```text
logs/YYYY-MM-DD.log
```

EXE가 쓰기 권한이 없는 위치에서 실행되면 로그는 `%LOCALAPPDATA%\CoverMorphStudio\logs`로 자동 우회됩니다. 각 이미지 폴더의 `covermorph_job.json`에는 프로그램 버전, 원본 파일명, 프리셋, OCR 언어, 탐지된 글자 영역 수, 글자 제거 엔진, 업스케일 엔진, 16:9/9:16 생성 엔진, SDXL fallback 여부, 원본 핵심 영역 보호 여부, 출력 파일 경로, 처리 시간, 성공/실패 상태가 기록됩니다.

## 테스트 실행

Windows에서 `RUN_TESTS.bat`을 더블클릭하면 다음 검사를 실행합니다.

1. 테스트 의존성 설치
2. `pip check`
3. `ruff check .`
4. `pytest`

기본 pytest는 실제 AI 모델 다운로드가 필요 없는 테스트만 실행합니다. SDXL, LaMa, Real-ESRGAN 실사용 테스트는 `optional_ai`로 표시되어 기본 실행에서는 skip됩니다.

## EXE 만들기

```bat
BUILD_EXE.bat
```

PyInstaller onedir 형태로 `dist\CoverMorphStudio` 폴더(`CoverMorphStudio.exe`)가 생성됩니다. SDXL 브리지용 torch/diffusers/transformers도 함께 포함되어 폴더가 큽니다. `customtkinter`, `tkinterdnd2`, 프리셋, `hub_manifest.json`, `tools` 안내 파일은 포함하지만, SDXL/LaMa 같은 대용량 선택형 AI 모델은 EXE에 강제로 포함하지 않습니다. EXE에서 Real-ESRGAN을 쓰려면 EXE 폴더 옆에 `tools` 폴더를 만들고 실행 파일과 모델을 넣으세요.

## 썸네일 브리지 (YouTube Dynamic Thumbnail Studio 연동)

CoverMorph는 YouTube Dynamic Thumbnail Studio v0.6(`1976haru/youtubesum`)의 배경 생성/편집 백엔드로 headless 실행할 수 있습니다. 계약 문서는 `THUMBNAIL_BRIDGE_CONTRACT.md`입니다. 인수 없이 더블클릭하면 기존 GUI가 그대로 열립니다.

| 호출 형태 | 입력 | 용도 |
| --- | --- | --- |
| `CoverMorphStudio.exe --thumbnail-bridge-json` | stdin: 계약 v1 JSON 1개 | 계약 기본 형태 |
| `CoverMorphStudio.exe --image-bridge` | stdin: youtubesum `json-stdin` 요청 | youtubesum `IMAGE_BRIDGE_MODE=json-stdin` |
| `CoverMorphStudio.exe --action generate --project-dir ... --options-json ...` | 명령줄 인수 | youtubesum 기본 `IMAGE_BRIDGE_MODE=cli` |

세 형태 모두 stdout에는 JSON 응답 1개만 쓰고(라이브러리 출력은 파일 디스크립터 수준에서 stderr로 돌립니다), `ok=false`이면 종료 코드 2를 반환합니다. 진행 로그는 stderr와 `logs\`에 남습니다.

- `status`: GPU 이름, VRAM 바이트, torch/diffusers/transformers/accelerate/safetensors 버전, SDXL·IP-Adapter 준비 상태, 선택된 메모리 프로필, 사용 가능한 편집 단계, 프로젝트 폴더 쓰기 가능 여부. 다운로드는 하지 않습니다.
- `generate`: 기존 `SDXLTextToImageEngine`으로 글자 없는 16:9 배경을 1344x768로 생성하고, 중앙 16:9 크롭 + Lanczos로 1280x720 `canvas_clean.png`를 만듭니다(늘리지 않음). 제목/부제는 프롬프트에 넣지 않으며 text/logo/watermark/signage를 negative prompt로 억제합니다. `preview_reference.png`는 원본 생성 해상도 이미지입니다.
- `edit`: 지시문(한/일/영)을 세 단계로 분류합니다.
  - `recompose` (로컬, 모델 불필요): 인물 좌우 이동, 지정한 쪽 문구 공간 단순화(예: `왼쪽 40%`), 인물을 보호한 채 배경 어둡게/밝게.
  - `regenerate`: 이전 프롬프트 + 지시문의 장면어로 SDXL 새 배경. 인물은 유지되지 않습니다.
  - `reference_regenerate`: 현재 캔버스의 인물을 IP-Adapter Plus 참조로 넣어 새 배경 생성. 인물은 비슷하게 유지되지만 픽셀/포즈가 그대로 보존되지는 않습니다.
  - 물체 추가/삭제, 표정·옷·머리 변경, 글자/로고 넣기, 인식되지 않는 지시는 `UNSUPPORTED_EDIT`로 거절하고 기존 파일을 바꾸지 않습니다. 요청한 단계의 모델이 준비되지 않은 경우도 `UNSUPPORTED_EDIT`입니다.

출력: `canvas_clean.png`, `preview_reference.png`, `subject_boxes.json`, `safe_zones.json`, `palette.json`, `composition.json`, `project_manifest.json`. 좌표는 0..1 정규화 값입니다. 인물 역할(protagonist/counterpart/other)은 크기·위치·요청 의도로 정하며 성별을 추정하지 않습니다. youtubesum은 `safe_zones`를 "글자를 피해야 할 영역"으로 읽기 때문에 `avoid` 키에는 얼굴/인물 영역만 넣고, 글자 추천 영역은 `preferred_text_regions`에만 씁니다. `palette.json`의 `fill_color`/`stroke_color`/`highlight_color`와 `composition.json`의 `positions`는 youtubesum이 바로 읽는 키입니다.

모든 결과는 프로젝트 폴더 안의 staging 폴더에 먼저 쓰고 검증한 뒤 교체합니다. 교체 중 하나라도 실패하면 이전 파일을 모두 되돌리고, 생성/편집 실패 시 이전 결과는 그대로 남습니다.

### 모델 위치와 GPU 메모리

모델은 EXE에 포함하지 않습니다. `<EXE 폴더>\models\sdxl_base_1.0`, `models\ip_adapter`를 쓰거나 `COVERMORPH_MODELS_DIR`(또는 요청 `options.models_dir`)로 지정하세요. 브리지는 오프라인(`HF_HUB_OFFLINE=1`, `local_files_only`)으로만 모델을 읽습니다.

메모리 프로필은 감지된 VRAM으로 정합니다. 14 GB 이상 `standard`, 8~14 GB `balanced`(VAE slicing+tiling), 8 GB 미만 `conservative`(model CPU offload, 1024x576 생성 후 확대). RTX 3060 12 GB에서 같은 seed로 측정하면 `balanced`가 `standard`보다 빠르고(26.5초 대 28.4초) 최대 메모리도 낮았으며 결과는 같았습니다. CUDA OOM이 나면 파이프라인을 해제하고 한 단계 낮은 프로필로 딱 한 번만 다시 시도합니다. 생성 시간, 최대 할당/예약 메모리, 프로필, 재시도 여부는 `project_manifest.json`의 `generation`에 기록됩니다.

### v1.0.0-rc1 — 설치와 설정 유지

- 압축을 풀고 `CoverMorphStudio.exe`를 더블클릭하면 첫 실행 설정 마법사가 열립니다(모델 폴더 한 번 지정 → 용도 → 품질 → PC 사용 방식). 자세한 순서: `QUICKSTART_KO.txt`.
- 설정·대기열·로그는 `%LOCALAPPDATA%\CoverMorphStudio`에 저장되어 EXE를 새로 빌드하거나 새 버전으로 바꿔도 유지됩니다. 기본 출력 폴더는 `문서\CoverMorphStudio`입니다.
- Shopify 상품: '상품 처리'에서 원본 그대로 합성(권장) / 자연광 보정 합성 / AI 재구성을 고릅니다. 처음 한 번 상품 마스크를 확인·수정하고, 결과에서 '상품 위치·크기 조정'으로 AI 없이 다시 배치할 수 있습니다.
- 문제가 생기면 설정 탭의 '진단 정보 복사'(프롬프트·이미지 미포함). 릴리스 검증 결과: `docs/V1_RELEASE_VALIDATION.md`. 릴리스 빌드: `python scriptsuild_release.py`.

### AI 이미지 스튜디오 (YouTube · Shopify)

EXE를 더블클릭하면 'AI 이미지 스튜디오' 창이 함께 열립니다(상단 초록 버튼으로 다시 열기). 모델 이름을 고를 필요 없이 목적(YouTube / Shopify 히어로·컬렉션·상품 라이프스타일·프로모션·모바일 / 사용자 지정), 품질(빠른 미리보기·일반·최고 품질), PC 사용(작업 중 PC 우선·균형·자리 비움)을 고르고 대기열에 넣으면 됩니다.

- 레퍼런스는 역할(인물·상품·스타일·구도·배경)을 지정해 최대 4장, '상품 형태 보존'을 켜면 실제 상품 픽셀 합성 후보가 맨 앞에 옵니다(AI가 다시 그린 후보는 '형태 확인 필요' 표시).
- 대기열은 앱을 다시 열어도 유지되고, 현재 작업 후 일시정지/재개, 자원 부족 시 'PC 사용 중 — 자원 대기'로 기다립니다. 작업마다 엔진 프로세스가 종료되어 메모리를 돌려줍니다.
- 모델·출력 폴더는 '설정' 탭에서 저장합니다(환경변수 불필요). 자세한 내용과 RTX 3060 검증 결과: `docs/SHOPIFY_WORKFLOW.md`.

### Quality Engine V2 (Z-Image-Turbo / FLUX.2-klein)

RTX 3060에서 같은 장면·시드로 비교한 결과(`docs/QUALITY_ENGINE_V2.md`)로 기본 엔진을 정했습니다.

| 역할 | 엔진 | 비고 |
| --- | --- | --- |
| 기본 텍스트→이미지 | Z-Image-Turbo Q6_K (Apache-2.0) | 1280x720 약 35초, GPU 피크 7.3 GiB |
| 기본 참조/편집 | FLUX.2-klein-4B Q8_0 (Apache-2.0) | 참조 1장 약 27초, 인물/상품 유지 + 배경 변경 |
| 대체(fallback) | RealVisXL V5 (기존 SDXL 경로) | V2 모델이 없거나 실패하면 자동 사용 |

- 준비: `python scripts\prepare_quality_v2_models.py --models-dir <models>` (약 16 GB, SHA-256 검증, `--verify`로 재확인). 파일은 `<models>\quality_v2`에 들어가며 git에 넣지 않습니다.
- 두 엔진은 stable-diffusion.cpp(`sd-cli.exe`, CUDA) 하위 프로세스로 실행되고 작업이 끝나면 프로세스가 종료되어 GPU/RAM을 즉시 반환합니다. 한 번에 한 모델만 메모리에 올립니다.
- 브리지 옵션: `engine` = `auto`(기본, V2 설치 시 사용) | `zimage_turbo` | `flux2_klein_4b` | `legacy`, `quality_mode` = `preview`(FLUX.2 1장) | `balanced`(기본) | `best`(엔진 2개 순차), `candidates` 1~4(브리지 기본 1, 추가 후보는 `<project>\candidates\`), `memory_policy` = `interactive_low_memory` | `balanced_idle` | `night_best`.
- `project_manifest.json`에 backend, model, quantization, model_license, commercial_use_flag, original_prompt, compiled_prompt, reference_roles, quality_mode, memory_profile, timing, peak_vram_mib, system_commit_before/after가 기록됩니다.
- 작업 대기열(`covermorph/job_queue.py`): 저장(원자적 교체), 현재 작업 후 일시정지, 재개, 대기 작업 취소, 앱 재시작 후 복구, GPU/commit/RAM이 기준보다 낮으면 시작하지 않고 대기. 실제 GPU 검증: `python scripts\validate_quality_queue.py --models-dir <models> --output-dir validation_results\quality_queue`.
- 한국어/일본어 프롬프트와 편집 지시는 같은 Qwen3-4B로 영어 번역 후 사용합니다(llama.cpp CPU, 약 9초, GPU 미사용). Z-Image가 한글/일본어 단어를 이미지에 글자로 그리는 문제(원문 2/3, 번역 0/3)를 막기 위한 것이며, 원문·번역·최종 프롬프트가 manifest에 남습니다.
- 비교 재현: `python scripts\benchmark_quality_v2.py --models-dir <models> --output-dir validation_results\quality_v2 --make-refs`.

### youtubesum 설정

```bat
set IMAGE_PROGRAM_EXE=D:\...\dist\CoverMorphStudio\CoverMorphStudio.exe
set COVERMORPH_MODELS_DIR=D:\...\models
set IMAGE_BRIDGE_MODE=cli
set IMAGE_BRIDGE_TIMEOUT=900
```

### 브리지 테스트

`pytest tests\test_thumbnail_bridge_runtime.py`는 가짜 엔진으로 동작하며(manifest에 `real_ai: false`) 모델을 다운로드하지 않습니다. 실제 GPU 생성은 `set COVERMORPH_REAL_AI=1`과 `COVERMORPH_MODELS_DIR`을 지정한 뒤 `pytest -m optional_ai`로, 패키지 EXE 확인은 `set COVERMORPH_BRIDGE_EXE=<exe 경로>`로 실행합니다.

## Playlist Studio Hub 연결

`hub_manifest.json`의 버전은 `0.5.4`입니다. Playlist Studio Hub에서 로컬 앱을 등록할 때 이 저장소 폴더를 앱 경로로 지정하고, entrypoint가 `RUN.bat`인지 확인하세요.

## v0.5.4 안정화 변경

- OCR 기본값은 `자동 전체 언어 탐지`이며 영어, 일본어, 한국어, 프랑스어 Reader를 조합별로 캐시합니다. 일본어가 포함되면 일본어 보조 Reader도 실행하고 겹친 박스를 병합합니다.
- 글자 제거는 글자 크기 기반 마스크 패딩을 사용하고, LaMa -> OpenCV NS/Telea 비교 -> 주변 패치 fallback 순서로 처리합니다. 마스크 경계에는 feather를 적용합니다.
- 9:16은 원본을 가로 기준으로 비율 유지 배치하고 위아래만 확장합니다. 전체 배경을 `ImageOps.fit`으로 늘리지 않으므로 원본 인물과 직선이 세로로 찌그러지지 않습니다. 16:9는 세로 기준으로 배치하고 좌우만 확장합니다.
- 미리보기와 로컬 저장은 공통 `render_full_frame_format` 렌더러를 사용합니다. AI Outpainting 실패 시 손상 가능성이 있는 로컬 결과를 자동으로 성공 저장하지 않고, 선택한 로컬 모드로 다시 처리해야 합니다.
- 목록의 각 행에서 선택한 규격만 `개별 변환`할 수 있습니다. 완료 후 `결과 저장`으로 해당 행 결과만 다른 폴더로 내보내고 `폴더 열기`로 이미지별 결과 폴더를 열 수 있습니다.
- 선택형 EasyOCR 일본어 가중치, LaMa, rembg, SDXL, Real-ESRGAN 실모델은 기본 설치에 포함되지 않습니다. 미설치/오류 시 기본 로컬 처리로 계속 진행합니다.
