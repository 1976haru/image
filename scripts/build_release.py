"""Build the release folder + ZIP: python scripts/build_release.py

1. writes covermorph/_build_info.py (commit, time) — generated, not committed
2. runs the same PyInstaller command as BUILD_EXE.bat (one-folder build)
3. adds QUICKSTART_KO.txt, checks that no user data / logs / outputs / models ended up in the folder
4. zips dist/CoverMorphStudio as CoverMorphStudio-v<version>-win64.zip and prints SHA-256 of EXE and ZIP
Models stay external; user settings live in %LOCALAPPDATA%\\CoverMorphStudio and are never packaged.
"""
from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist" / "CoverMorphStudio"
FORBIDDEN = ("config", "creator_output", "logs", "models", "validation_results", "studio_queue.json", "settings.json")

QUICKSTART = """CoverMorph Studio {version} — 빠른 시작

1. 압축을 원하는 폴더에 풉니다 (한글 경로 가능).
2. CoverMorphStudio.exe 를 더블클릭합니다.
3. 처음 실행하면 설정 마법사가 열립니다.
   - 모델 폴더(quality_v2 폴더가 들어 있는 폴더)를 한 번만 지정합니다.
   - 주로 만들 것(YouTube / Shopify / 둘 다), 기본 품질, PC 사용 방식을 고릅니다.
   설정은 %LOCALAPPDATA%\\CoverMorphStudio 에 저장되어 새 버전으로 바꿔도 유지됩니다.
4. 'AI 이미지 스튜디오' 창에서
   - 목적(YouTube 썸네일 / Shopify 히어로·컬렉션·상품 라이프스타일·프로모션·모바일)을 고르고
   - 프롬프트(한국어·일본어 가능)를 쓰고, 필요하면 레퍼런스를 역할(인물/상품/스타일/구도/배경)과 함께 추가합니다.
   - 상품 사진은 '상품 처리: 원본 그대로 합성'을 쓰면 상품 모양·로고가 바뀌지 않습니다(처음 한 번 마스크 확인).
   - '대기열에 추가' 또는 '추가하고 바로 시작'을 누릅니다.
5. '대기열' 탭: 현재 작업 후 일시정지 / 재개 / 실패 재시도. 앱을 닫았다 열어도 대기열이 이어집니다.
   PC 자원이 부족하면 작업을 멈추지 않고 'PC 사용 중 — 자원 대기'로 기다립니다.
6. '후보 비교' 탭: 후보를 비교하고 '채택'을 누르면 원본 크기 JPG/PNG(글자 없는 버전 + 글자 합성 버전)로 내보냅니다.
   결과는 기본적으로 문서\\CoverMorphStudio 에 저장됩니다.

문제가 생기면: 설정 탭 → '진단 정보 복사' (프롬프트·이미지는 포함되지 않습니다).
필요한 것: NVIDIA GPU(검증: RTX 3060 12GB), Windows 10/11, 모델 파일(약 16 GB, 별도 제공).
빌드: {build}
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    sys.path.insert(0, str(ROOT))
    from covermorph import __version__
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    build = f"{commit}{'-dirty' if dirty else ''}"
    stamp = time.strftime("%Y-%m-%d %H:%M")
    info = ROOT / "covermorph" / "_build_info.py"
    info.write_text(f'BUILD_COMMIT = "{build}"\nBUILD_TIME = "{stamp}"\n', encoding="utf-8")
    line = next(l for l in (ROOT / "BUILD_EXE.bat").read_text(encoding="utf-8", errors="replace").splitlines()
                if "pyinstaller.exe" in l and "app.py" in l)
    args = shlex.split(line.strip(), posix=True)
    args[0] = str(ROOT / ".venv" / "Scripts" / "pyinstaller.exe")
    args[-1:-1] = ["--hidden-import", "covermorph._build_info"]
    print("building", __version__, build, flush=True)
    if subprocess.call(args, cwd=ROOT) != 0:
        return 1
    (DIST / "QUICKSTART_KO.txt").write_text(QUICKSTART.format(version=__version__, build=f"{build} ({stamp})"), encoding="utf-8")
    leaked = [name for name in FORBIDDEN if (DIST / name).exists()]
    if leaked:
        print("refusing to package user data:", leaked)
        return 2
    archive = ROOT / "dist" / f"CoverMorphStudio-v{__version__}-win64.zip"
    archive.unlink(missing_ok=True)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(DIST.rglob("*")):
            if path.is_file():
                zf.write(path, Path("CoverMorphStudio") / path.relative_to(DIST))
    exe = DIST / "CoverMorphStudio.exe"
    result = {"version": __version__, "build": build, "built": stamp, "exe": str(exe), "exe_sha256": sha256(exe),
              "zip": str(archive), "zip_sha256": sha256(archive), "zip_mib": round(archive.stat().st_size / 2 ** 20, 1)}
    (ROOT / "dist" / f"release-v{__version__}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
