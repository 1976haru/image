# CoverMorph Studio v0.5.2

CoverMorph Studio는 음원 커버 이미지 한 장을 플레이리스트 운영에 필요한 이미지로 변환하는 Windows용 로컬 프로그램입니다. 기본 기능은 외부 API 없이 동작하며, 선택형 AI 도구가 설치되어 있으면 자동으로 더 높은 품질의 엔진을 사용합니다.

## 주요 기능

- 1:1 클린 커버 생성: `1400x1400` JPG
- YouTube용 16:9 썸네일 생성: `1920x1080` 또는 `1280x720` JPG
- Shorts/Reels/TikTok용 9:16 세로 이미지 생성: `1080x1920` JPG
- 원하는 출력 규격만 선택해서 변환
- 출력 폴더와 마지막 UI 설정을 `config/settings.json`에 저장
- 동일 파일명 처리 방식 선택: 덮어쓰기, 새 번호 붙이기, 건너뛰기
- EasyOCR 기반 글자 영역 탐지
- LaMa 선택형 글자 제거, 실패 시 OpenCV Telea 자동 fallback
- Real-ESRGAN NCNN-Vulkan 선택형 업스케일, 미설치/실패 시 로컬 선명도 보정
- SDXL 선택형 AI 배경 확장, 다운로드/메모리/CUDA/추론 실패 시 Blur Canvas 자동 fallback
- rembg 선택형 인물 분리와 원본 인물 픽셀 보호
- 파일별 작업 JSON과 날짜별 로그 저장

## 기본 설치 방법

1. Python 3.11 또는 3.12를 설치합니다.
2. 이 폴더에서 `INSTALL.bat`을 실행합니다.
3. 설치가 끝나면 `RUN.bat`을 실행합니다.

`INSTALL.bat`은 Python 3.11을 먼저 찾고, 없으면 3.12를 찾습니다. 지원하지 않는 Python만 있으면 명확한 오류를 표시하고 멈춥니다.

## 선택형 AI 설치

기본 설치만으로도 OpenCV 글자 제거, Blur Canvas 확장, 1:1/16:9/9:16 생성이 동작합니다.

### LaMa 글자 제거

`INSTALL_AI_LAMA.bat`을 실행합니다.

설치 후 프로그램을 다시 실행하면 LaMa가 자동 감지됩니다. LaMa 실행 중 오류가 나면 OpenCV Telea 방식으로 자동 전환됩니다.

### SDXL AI 배경 확장

`INSTALL_AI_OUTPAINT.bat`을 실행합니다.

SDXL 모델은 프로그램에서 처음 사용할 때 자동으로 다운로드됩니다. 첫 다운로드는 오래 걸릴 수 있고, 저장 공간과 메모리가 충분해야 합니다. 다운로드 실패, CUDA 오류, 메모리 부족, 추론 오류가 발생해도 전체 작업은 멈추지 않고 해당 출력만 Blur Canvas로 전환됩니다.

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
3. 출력 폴더 확인 또는 `찾아보기`로 변경
4. 프리셋과 OCR 언어 선택
5. 필요한 출력 이미지 규격 선택
6. 필요하면 `글자 자동 탐지` 후 마우스로 마스크 영역 추가
7. `원클릭 AI 자동 변환` 실행

출력 규격은 1개 이상 선택해야 합니다. 예를 들어 16:9만, 9:16만, 16:9+9:16만, 세 규격 전체를 자유롭게 선택할 수 있습니다.

## 출력 폴더와 설정

설정 파일은 프로그램 폴더 안의 다음 위치에 자동 생성됩니다.

```text
config/settings.json
```

저장 항목 예:

```json
{
  "output_directory": "D:\\03_image\\CoverMorph_Output",
  "output_square": true,
  "output_thumbnail": true,
  "output_shorts": true,
  "thumbnail_resolution": "1920x1080",
  "duplicate_policy": "new_number",
  "last_preset": "OldPopLounge",
  "last_ocr_language": "영어"
}
```

출력 폴더를 아직 설정하지 않았다면 첫 번째 원본 이미지 폴더 아래 `CoverMorph_Output`을 기본값으로 사용합니다.

```text
D:\음원커버\19 음악커버 1.png
D:\음원커버\CoverMorph_Output
```

저장된 출력 폴더가 삭제됐거나 사용할 수 없으면 프로그램은 종료되지 않고 새 출력 폴더 선택을 안내합니다. 직접 입력한 경로는 변환 시작 전에 검사하며, 없는 폴더는 생성 여부를 확인합니다.

## 출력 폴더 구조

기본 출력 폴더 아래에 원본 파일명별 하위 폴더가 만들어집니다.

```text
D:\03_image\CoverMorph_Output
  19 음악커버 1
    19 음악커버 1_clean_1x1_1400x1400.jpg
    19 음악커버 1_thumbnail_16x9_1920x1080.jpg
    19 음악커버 1_shorts_9x16_1080x1920.jpg
    covermorph_job.json
```

동일한 파일명이 이미 있을 때 기본값은 `새 번호 붙이기`입니다.

```text
image_thumbnail_16x9_1920x1080_02.jpg
```

## 작업 요약과 완료 창

변환 버튼 위에는 현재 출력 폴더, 생성 예정 규격, 선택 이미지 수, 예상 결과물 수가 표시됩니다. 체크박스나 출력 폴더가 바뀌면 즉시 갱신됩니다.

완료 창에는 저장된 출력 폴더, 생성된 이미지 개수, 성공/실패 원본 개수, 생성된 규격이 표시됩니다. `출력 폴더 열기` 버튼으로 결과 폴더를 바로 열 수 있습니다.

## 오류 로그 위치

기본 로그 위치:

```text
logs/YYYY-MM-DD.log
```

EXE가 쓰기 권한이 없는 위치에서 실행되면 다음 위치로 자동 우회됩니다.

```text
%LOCALAPPDATA%\CoverMorphStudio\logs
```

작업 JSON에는 프로그램 버전, 원본 파일명, 프리셋, OCR 언어, 탐지 글자 영역 수, 글자 제거 엔진, 업스케일 엔진, 16:9/9:16 생성 엔진, SDXL fallback 여부, 인물 보호 여부, 출력 파일 경로, 처리 시간, 성공/실패 상태가 기록됩니다.

## 테스트 실행

Windows에서 `RUN_TESTS.bat`을 더블클릭하면 다음 검사를 실행합니다.

1. 테스트 요구 패키지 설치
2. `pip check`
3. `ruff check .`
4. `pytest`

기본 pytest는 실제 AI 모델 다운로드가 필요 없는 테스트만 수행합니다. SDXL, LaMa, Real-ESRGAN 실제 실행 테스트는 선택형 테스트로 표시되어 기본 실행에서는 skip됩니다.

## EXE 만들기

`BUILD_EXE.bat`을 실행합니다.

PyInstaller onedir 형태로 `dist\CoverMorphStudio_v0.5.2` 폴더가 생성됩니다. `customtkinter`, 프리셋, `hub_manifest.json`, `tools` 안내 파일은 포함되지만, SDXL/LaMa 같은 대용량 선택형 모델은 EXE에 강제로 포함하지 않습니다. Real-ESRGAN을 EXE에서 쓰려면 EXE 폴더 옆에 `tools` 폴더를 만들고 실행 파일과 모델을 넣으세요.

## Playlist Studio Hub 연결

`hub_manifest.json`의 버전은 `0.5.2`입니다. Playlist Studio Hub에서 로컬 앱을 등록할 때 이 저장소 폴더를 앱 경로로 지정하고, entrypoint가 `RUN.bat`인지 확인하세요.
