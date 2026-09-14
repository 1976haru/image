# CoverMorph 3-B1: 단일 참고 이미지

기준 원격은 6fadb90 이후 이미 추가된 1053724였습니다. 이번 변경은 그 초안을 보완합니다. 기존 processor.py, LaMa 환경, 설치 요구사항은 변경하지 않습니다.

## 모델과 출처

한 가지 조합: `h94/IP-Adapter`의 **IP-Adapter Plus SDXL ViT-H**, 인물 외모 참고 또는 스타일 참고에 같은 어댑터를 사용합니다. 패치 특징을 사용하는 Plus를 선택했고, 기본 SDXL 어댑터의 bigG 인코더 대신 대응하는 ViT-H를 명시적으로 로드합니다. FaceID/InsightFace는 사용하지 않습니다.

고정 revision: `9bf28b38530e55ffa91c6d82e5161a982c22f284` (safetensors 추가 커밋).

필요 파일은 아래 세 개만 다운로드합니다.

- `sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors` (847,517,512 bytes)
- `models/image_encoder/config.json`
- `models/image_encoder/model.safetensors` (2,528,373,448 bytes)

두 가중치의 공식 LFS SHA-256도 코드에서 검증합니다. 다운로드 완료 후 safetensors 열기, 파일 해시, revision을 확인한 `adapter_ready.json`을 마지막에 저장합니다. 중단된 준비 작업은 완료 표시가 없습니다. 이전 기본 어댑터를 다운로드했다면 Plus 준비 버튼을 다시 누르세요.

출처(2026-09-14 확인):

- [Diffusers IP-Adapter 공식 사용법](https://huggingface.co/docs/diffusers/en/using-diffusers/ip_adapter)
- [모델 카드](https://huggingface.co/h94/IP-Adapter/blob/main/README.md)
- [Plus 가중치와 SHA-256](https://huggingface.co/h94/IP-Adapter/commit/9bf28b38530e55ffa91c6d82e5161a982c22f284)
- [ViT-H 인코더와 SHA-256](https://huggingface.co/h94/IP-Adapter/commit/0859e809306db97aa2338e370587ab284e8a754f)

IP-Adapter 저장소 라이선스는 Apache-2.0입니다. SDXL 기본 모델은 별도 [모델 카드와 OpenRAIL++ 라이선스](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)를 따릅니다. 참고 사진의 사용 권리는 별개입니다.

## 준비 및 클릭 순서

1. 기존 `D:\03_image\image_github\RUN.bat` 실행. 앱 시작 시 모델 다운로드는 없습니다.
2. GPU 도착 후 기본 `.venv`에서 CUDA PyTorch를 준비하고 `torch.cuda.is_available()`를 확인합니다. `.venv_lama`를 변경하지 마세요. 3-A 설치에 사용한 diffusers, transformers, accelerate, safetensors를 재사용합니다. 이번 기능 때문에 기본 numpy/Pillow를 다시 설치하지 않습니다.
3. 프로젝트 열기 → 등록 인물의 참고 이미지 추가. 용도는 `person` 또는 `style`로 등록합니다. 배경·구도 자료는 보존되지만 이 단계 생성용으로 선택할 수 없습니다.
4. SDXL 모델 준비 버튼 → 모델 경로 확인. IP-Adapter 명시적 준비 버튼 → 완료 메시지 확인. 준비 완료는 파일 확인이며 추론 성공을 뜻하지 않습니다.
5. 장면에서 참고 사용 `off` / `person` / `style` 선택 → 해당 용도의 **한 장** 선택 → 이미지 미리보기 확인.
6. 필요하면 **참고 영역 직접 크롭** → 얼굴·상반신 영역 드래그. 전체 사용도 가능합니다. EXIF 회전 후 좌표를 사용하고, 투명 영역은 흰색에 합성합니다. 원본 대신 파생 PNG를 만듭니다.
7. 강도 설정(기본 0.5, UI 범위 0~1) → 생성 크기와 프롬프트 확인 완료 → 현재 장면 후보 생성.
8. 후보를 검토한 뒤 사용자가 작업 원본으로 승인하고 기존 출력 변환을 실행합니다.

공식 안내의 균형값 0.5를 기본으로 사용합니다. 0~1은 이 앱의 제한 범위이며 모델의 모든 가능한 강도 범위를 뜻하지 않습니다. 인물 참고는 외모 특징 참고이며 동일 인물을 보장하지 않습니다. 스타일 참고도 인물·구도에 영향을 줄 수 있습니다. 강도 0은 어댑터가 로드된 상태일 수 있으므로 텍스트 전용은 `off`로 선택하세요.

## 실제 연결 및 보존

`CLIPVisionModelWithProjection`으로 ViT-H를 등록하고 `load_ip_adapter(..., image_encoder_folder=None)`로 Plus 가중치를 로드합니다. 매 후보 호출에 `ip_adapter_image`와 `set_ip_adapter_scale`의 실제 설정을 사용합니다. 성공한 호출만 참고 적용으로 기록합니다. 로드 실패, 파일 손상, 미적용 결과는 저장하지 않습니다.

후보는 한 장씩 처리합니다. FP16, VAE slicing/tiling, model CPU offload를 사용합니다. GPU에서 추론하고 일부 모듈을 시스템 RAM으로 옮기는 방식이며 CPU 전용 추론으로 전환하지 않습니다. RTX 3060 12GB 목표 설정이지만 이 환경에서 실제 VRAM 사용량/성능은 검증하지 않았습니다. 메모리 부족 시 해상도나 모델을 자동 변경하지 않습니다.

모드 또는 어댑터 변경 시 이전 파이프라인을 해제하고 재생성합니다. 같은 모드에서 A→B는 새 PIL 이미지를 매번 전달합니다. 임베딩 캐시는 사용하지 않습니다. 전처리 PNG 캐시는 원본 해시·크롭·전처리 버전으로 구분합니다. 어댑터/인코더 revision은 생성 기록에 별도 저장합니다.

작업 시작 시 설정과 장면을 복사하고 참고 이미지를 메모리로 읽습니다. 재시도는 원래 설정과 해시로 검증한 파생 참고 PNG를 사용합니다. 완료 후보 메타데이터는 후속 후보의 실패로 바뀌지 않습니다. 실패 원인과 실행 상태를 프로젝트 JSON에 저장하며 시각 품질은 계속 `unverified`입니다.

## RTX 3060 도착 후 실제 비교

저장소 폴더의 PowerShell에서(사진 경로는 실제 파일로 변경):

```powershell
.\.venv\Scripts\python.exe scripts\validate_reference_generation.py --model models/sdxl_base_1.0 --adapter models/ip_adapter --person "D:\참고\인물.jpg" --style "D:\참고\카페.jpg" --seed 12345 --ratio 1:1
```

각 용도를 따로 off/0.5/0.8로 비교합니다. prompt, seed, steps, 크기를 고정합니다. 다른 프롬프트는 `--prompt "..."`, 가로/세로는 `--ratio 16:9` 또는 `--ratio 9:16`입니다. 실제 생성 크기는 각각 1344×768, 768×1344이며 최종 출력 비율과 다릅니다.

`validation_results/3b1_<시간>/`에 PNG, 프로젝트, 설정, 장치, 시간, CUDA 최대 할당/예약 메모리를 남깁니다. Git에서는 제외됩니다. GPU/모델이 없으면 진단만 기록합니다. `--diagnose-only`는 생성하지 않습니다. 보고서의 참고 반영, 얼굴 왜곡, 구도 평가는 사람이 결과를 보고 각각 작성해야 합니다. 자동 테스트 성공을 시각 품질 성공으로 해석하지 않습니다.

이번 범위에는 두 인물 독립 유지, ControlNet, 학습, 문구 합성, 유료 API가 없습니다.
