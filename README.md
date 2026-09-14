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

PyInstaller onedir 형태로 `dist\CoverMorphStudio_v0.5.4` 폴더가 생성됩니다. `customtkinter`, `tkinterdnd2`, 프리셋, `hub_manifest.json`, `tools` 안내 파일은 포함하지만, SDXL/LaMa 같은 대용량 선택형 AI 모델은 EXE에 강제로 포함하지 않습니다. EXE에서 Real-ESRGAN을 쓰려면 EXE 폴더 옆에 `tools` 폴더를 만들고 실행 파일과 모델을 넣으세요.

## Playlist Studio Hub 연결

`hub_manifest.json`의 버전은 `0.5.4`입니다. Playlist Studio Hub에서 로컬 앱을 등록할 때 이 저장소 폴더를 앱 경로로 지정하고, entrypoint가 `RUN.bat`인지 확인하세요.

## v0.5.4 안정화 변경

- OCR 기본값은 `자동 전체 언어 탐지`이며 영어, 일본어, 한국어, 프랑스어 Reader를 조합별로 캐시합니다. 일본어가 포함되면 일본어 보조 Reader도 실행하고 겹친 박스를 병합합니다.
- 글자 제거는 글자 크기 기반 마스크 패딩을 사용하고, LaMa -> OpenCV NS/Telea 비교 -> 주변 패치 fallback 순서로 처리합니다. 마스크 경계에는 feather를 적용합니다.
- 9:16은 원본을 가로 기준으로 비율 유지 배치하고 위아래만 확장합니다. 전체 배경을 `ImageOps.fit`으로 늘리지 않으므로 원본 인물과 직선이 세로로 찌그러지지 않습니다. 16:9는 세로 기준으로 배치하고 좌우만 확장합니다.
- 미리보기와 로컬 저장은 공통 `render_full_frame_format` 렌더러를 사용합니다. AI Outpainting 실패 시 손상 가능성이 있는 로컬 결과를 자동으로 성공 저장하지 않고, 선택한 로컬 모드로 다시 처리해야 합니다.
- 목록의 각 행에서 선택한 규격만 `개별 변환`할 수 있습니다. 완료 후 `결과 저장`으로 해당 행 결과만 다른 폴더로 내보내고 `폴더 열기`로 이미지별 결과 폴더를 열 수 있습니다.
- 선택형 EasyOCR 일본어 가중치, LaMa, rembg, SDXL, Real-ESRGAN 실모델은 기본 설치에 포함되지 않습니다. 미설치/오류 시 기본 로컬 처리로 계속 진행합니다.
