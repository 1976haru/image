# CoverMorph Studio v0.5

음원 커버 한 장을 플레이리스트 운영에 필요한 3종 이미지로 자동 변환하는 Windows용 로컬 프로그램입니다.

## 원클릭 파이프라인

1400×1400 커버
→ OCR 글자 탐지
→ LaMa 우선 / OpenCV fallback 글자 제거
→ 얼굴·인물 보호
→ 화질 향상
→ Real-ESRGAN 우선 / 로컬 fallback
→ 1:1 클린커버
→ 16:9 유튜브 썸네일
→ 9:16 Shorts/Reels/TikTok
→ 지정 폴더 자동 저장

## v0.5 신규

- SDXL 인페인팅 모델을 이용한 16:9 / 9:16 선택형 AI 배경 확장
- rembg 사람 분할 마스크
- 생성 후 사람 픽셀을 원본에서 다시 합성하는 원본 보호 구조
- AI가 없거나 실패할 때 기존 Blur Canvas로 안전하게 자동 대체
- 배경 생성용 프롬프트 직접 입력
- 엔진 사용 내역을 작업 JSON에 기록

## 설치

1. `INSTALL.bat` 실행
2. 평소에는 `RUN.bat` 실행
3. 글자 제거 AI가 필요하면 `INSTALL_AI_LAMA.bat` 실행
4. AI 배경 확장이 필요하면 `INSTALL_AI_OUTPAINT.bat` 실행

SDXL은 첫 실행 때 모델을 내려받으며, NVIDIA GPU와 충분한 여유 공간을 권장합니다.
일반 PC에서는 `SDXL AI 배경 확장`을 끄면 무료 로컬 확장 방식으로 작동합니다.

## 권장 사용 순서

이미지 불러오기 → 출력 폴더 지정 → 프리셋 선택 → 글자 자동 탐지 → 필요한 영역 수동 드래그 → 원클릭 자동 변환

`사람 원본 픽셀 보호`를 켜면 SDXL이 배경을 생성한 뒤 분리된 사람 영역을 원본 픽셀로 복원합니다.

## v0.4 기능 유지

- LaMa 실제 연동
- Real-ESRGAN NCNN-Vulkan 실제 연동
- GPU/CPU/AI Backend 상태 자동 진단
- Before / After 비교 보기
- 작업 로그 기록
- AI 실패 시 자동 fallback
- EXE 빌드 스크립트
- optional AI 설치 스크립트
- 작업별 manifest 생성
- OldPopLounge / Tokyo ChillRap / Generic Playlist 프리셋 유지
- 좌측 텍스트 세이프존 + 인물 우측 배치
- 9:16 얼굴 보호

## 빠른 시작

### 기본 설치
1. `INSTALL.bat`
2. `RUN.bat`

기본 설치만으로도 동작합니다.

### 고품질 글자 제거
`INSTALL_AI_LAMA.bat`

설치 후 프로그램을 다시 실행하면 LaMa를 자동 감지합니다.

### Real-ESRGAN
`tools` 폴더 안에 아래 파일을 넣으세요.

- `realesrgan-ncnn-vulkan.exe`
- Real-ESRGAN 모델 파일/폴더

프로그램이 실행파일을 자동 감지합니다.

권장 모델:
- 사진/실사: `realesrgan-x4plus`
- 일반 장면 경량: `realesr-general-x4v3`

## 출력 예

입력:
`EP014_cover.jpg`

출력:
- `EP014_cover_clean_1x1_1400x1400.jpg`
- `EP014_cover_thumb_16x9_1920x1080.jpg`
- `EP014_cover_shorts_9x16_1080x1920.jpg`
- `covermorph_job.json`

## Before / After

첫 번째 이미지에 대해:
- 자동 글자 탐지
- 수동 마스크 보정
- `미리보기 글자 제거`
- Before / After 토글

로 품질을 확인할 수 있습니다.

## EXE 만들기

`BUILD_EXE.bat`

PyInstaller로 onedir EXE를 만듭니다.
AI 모델 자체는 용량이 매우 크기 때문에 EXE에 강제로 묶지 않고 외부 `tools` / 모델 구조를 권장합니다.

## 설계 원칙

1. 외부 API 비용 없음
2. AI 기능이 없어도 기본 기능은 실행
3. AI 기능이 있으면 자동으로 고품질 엔진 사용
4. 얼굴/인물 원본을 가능한 유지
5. 반복 업로드 작업에 적합한 배치 처리
