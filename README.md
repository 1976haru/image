# CoverMorph Studio v0.5.1

CoverMorph Studio는 음원 커버 이미지 한 장을 플레이리스트 운영에 필요한 이미지 3종으로 변환하는 Windows용 로컬 프로그램입니다. 기본 기능은 외부 API 없이 동작하며, 선택형 AI 도구가 설치되어 있으면 자동으로 더 높은 품질의 엔진을 사용합니다.

## 주요 기능

- 1:1 클린 커버 생성: `1400x1400`
- YouTube용 16:9 썸네일 생성: `1920x1080`
- Shorts/Reels/TikTok용 9:16 세로 이미지 생성: `1080x1920`
- EasyOCR 기반 글자 영역 탐지
- LaMa 선택형 글자 제거, 실패 시 OpenCV Telea 자동 fallback
- OpenCV 기반 기본 글자 제거와 선명도 보정
- Real-ESRGAN NCNN-Vulkan 선택형 업스케일, 미설치/실패 시 로컬 선명도 보정
- SDXL 선택형 AI 배경 확장, 다운로드/메모리/CUDA/추론 실패 시 Blur Canvas 자동 fallback
- rembg 선택형 인물 분리와 원본 인물 픽셀 보호
- 파일별 작업 JSON과 날짜별 로그 저장
- 배치 처리 중 진행률, 현재 파일명, 성공/실패 개수 표시

## 기본 설치 방법

1. Python 3.11 또는 3.12를 설치합니다.
2. 이 폴더에서 `INSTALL.bat`을 실행합니다.
3. 설치가 끝나면 `RUN.bat`을 실행합니다.

`INSTALL.bat`은 Python 3.11을 먼저 찾고, 없으면 3.12를 찾습니다. Python 3.14처럼 지원하지 않는 버전만 있으면 명확한 오류를 표시하고 멈춥니다.

## 선택형 AI 설치

기본 설치만으로도 1:1, 16:9, 9:16 생성과 OpenCV 글자 제거가 동작합니다. 더 높은 품질이 필요할 때만 아래 설치를 추가하세요.

### LaMa 글자 제거

`INSTALL_AI_LAMA.bat`을 실행합니다.

설치 후 프로그램을 다시 실행하면 LaMa가 자동 감지됩니다. LaMa 실행 중 오류가 나면 OpenCV Telea 방식으로 자동 전환됩니다.

### SDXL AI 배경 확장

`INSTALL_AI_OUTPAINT.bat`을 실행합니다.

NVIDIA GPU가 있으면 CUDA용 PyTorch를 설치하고, 없으면 CPU용 PyTorch를 설치합니다. SDXL 모델은 설치 배치 파일에서 바로 받지 않고, 프로그램에서 처음 사용할 때 자동으로 다운로드됩니다. 첫 다운로드는 오래 걸릴 수 있고, 저장 공간과 메모리가 충분해야 합니다.

SDXL 다운로드 실패, 메모리 부족, CUDA 오류, 추론 오류가 발생해도 전체 작업은 멈추지 않습니다. 실패 원인은 `logs`와 `covermorph_job.json`에 기록되고, 해당 출력은 Blur Canvas 방식으로 생성됩니다.

### Real-ESRGAN

Real-ESRGAN NCNN-Vulkan은 직접 내려받아 아래 위치 중 하나에 넣습니다.

- 프로그램 루트의 `tools`
- EXE 옆의 `tools`
- PyInstaller 실행 중 `sys._MEIPASS` 안의 `tools`
- 시스템 `PATH`

권장 구조:

```text
tools/
  realesrgan-ncnn-vulkan.exe
  models/
    realesrgan-x4plus.param
    realesrgan-x4plus.bin
```

실행 파일이나 모델 파일이 없으면 프로그램은 멈추지 않고 로컬 선명도 보정으로 전환됩니다.

## 실행 방법

1. `RUN.bat` 실행
2. `커버 이미지 불러오기` 선택
3. `출력 폴더 지정` 선택
4. 프리셋과 OCR 언어 선택
5. 필요하면 `글자 자동 탐지` 후 마우스로 마스크 영역을 추가
6. `원클릭 AI 자동 변환` 실행

작업 중에는 같은 작업을 중복 실행할 수 없습니다. `작업 취소`를 누르면 현재 진행 중인 파일 단계가 끝난 뒤 안전하게 중단됩니다.

## 처음 사용할 때 권장 설정

- OCR 언어: 영어 커버는 `영어`, 한국어가 섞이면 `한국어+영어`, 일본어가 섞이면 `일본어+영어`
- `글자 자동 삭제`: 켜기
- `기본 화질 보정`: 켜기
- `사람 원본 픽셀 보호`: 켜기
- `SDXL AI 배경 확장`: 처음에는 끄고 기본 결과를 먼저 확인

EasyOCR은 언어 조합별 Reader를 캐시해서 재사용합니다. 첫 실행 때 모델 다운로드가 필요하면 화면에 준비 안내가 표시됩니다. OCR이 실패해도 마우스로 직접 마스크를 그려 계속 작업할 수 있습니다.

## 일반 PC용 설정

- `SDXL AI 배경 확장`: 끄기
- `Real-ESRGAN 우선 업스케일`: Real-ESRGAN을 설치하지 않았다면 꺼도 됩니다.
- 기본 OpenCV/Blur Canvas 방식으로 빠르게 생성합니다.

## NVIDIA GPU PC용 설정

- `INSTALL_AI_OUTPAINT.bat` 실행
- `SDXL AI 배경 확장`: 켜기
- `사람 원본 픽셀 보호`: 켜기
- Real-ESRGAN을 설치했다면 `Real-ESRGAN 우선 업스케일`: 켜기

GPU 메모리가 부족하면 SDXL 출력만 Blur Canvas로 전환되고, 나머지 출력은 계속 처리됩니다.

## 출력 폴더 구조

입력 파일이 `EP014_cover.jpg`이고 출력 폴더가 `D:\covers_out`이면 다음처럼 저장됩니다.

```text
D:\covers_out\
  EP014_cover\
    EP014_cover_clean_1x1_1400x1400.jpg
    EP014_cover_thumb_16x9_1920x1080.jpg
    EP014_cover_shorts_9x16_1080x1920.jpg
    covermorph_job.json
```

작업 JSON에는 프로그램 버전, 원본 파일명, 프리셋, OCR 언어, 탐지 글자 영역 수, 글자 제거 엔진, 업스케일 엔진, 16:9/9:16 생성 엔진, SDXL fallback 여부, 인물 보호 여부, 출력 파일 경로, 처리 시간, 성공/실패 상태가 기록됩니다.

## 오류 로그 위치

기본 로그 위치:

```text
logs/YYYY-MM-DD.log
```

EXE가 쓰기 권한이 없는 위치에서 실행되면 다음 위치로 자동 우회됩니다.

```text
%LOCALAPPDATA%\CoverMorphStudio\logs
```

읽을 수 없는 이미지, 손상된 이미지, 출력 폴더 생성 실패, 저장 실패, 선택형 AI fallback 원인은 로그와 작업 JSON에서 확인할 수 있습니다.

## 테스트 실행

Windows에서 `RUN_TESTS.bat`을 더블클릭하면 다음 검사를 실행합니다.

1. 테스트 요구 패키지 설치
2. `pip check`
3. `ruff check .`
4. `pytest`

기본 pytest는 실제 AI 모델 다운로드가 필요 없는 테스트만 수행합니다. SDXL, LaMa, Real-ESRGAN 실제 실행 테스트는 선택형 테스트로 표시되어 기본 실행에서는 skip됩니다.

## EXE 만들기

`BUILD_EXE.bat`을 실행합니다.

PyInstaller onedir 형태로 `dist\CoverMorphStudio_v0.5.1` 폴더가 생성됩니다. `customtkinter`, 프리셋, `hub_manifest.json`, `tools` 안내 파일은 포함되지만, SDXL/LaMa 같은 대용량 선택형 모델은 EXE에 강제로 포함하지 않습니다. Real-ESRGAN을 EXE에서 쓰려면 EXE 폴더 옆에 `tools` 폴더를 만들고 실행 파일과 모델을 넣으세요.

## Playlist Studio Hub 연결

`hub_manifest.json`의 버전은 `0.5.1`입니다. Playlist Studio Hub에서 로컬 앱을 등록할 때 이 저장소 폴더를 앱 경로로 지정하고, entrypoint가 `RUN.bat`인지 확인하세요. Hub가 앱 목록을 새로 읽으면 CoverMorph Studio가 이미지 도구로 표시됩니다.
