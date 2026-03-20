"""
FID 테스트용 데이터 준비 스크립트

reference_images와 vision_ai_generated에서 이미지를 복사해서
FID_test/{subtype}/{original_image, normal_00, ai_generated} 폴더를 채웁니다.

- original_image/ : ref_*.jpg (real reference images)
- normal_00/      : normal_00 이미지를 N장 복사 (파일명 normal_00_001.ext ~ )
- ai_generated/   : AI 생성 이미지 최대 N장 복사
"""

import shutil
from pathlib import Path

# ===== 경로 설정 =====
BASE_DIR      = Path(__file__).parent
AUG_DIR       = BASE_DIR.parent / "augmentation"
REF_DIR       = AUG_DIR / "reference_images"
AI_DIR        = AUG_DIR / "vision_ai_generated"

N_NORMAL      = 100  # normal_00 복사 횟수
N_AI          = 100  # AI 이미지 최대 복사 수
IMAGE_EXTS    = {".jpg", ".jpeg", ".png", ".PNG", ".JPG", ".JPEG"}

# 서브타입 매핑: (ref 폴더, ai 폴더)
SUBTYPES = [
    ("casting_Inclusoes",       REF_DIR / "casting_Inclusoes",            AI_DIR / "casting_Inclusoes"),
    ("casting_Rechupe",         REF_DIR / "casting_Rechupe",              AI_DIR / "casting_Rechupe"),
    ("Console_Collision",       REF_DIR / "Console" / "Console_Collision",AI_DIR / "Console" / "Console_Collision"),
    ("Console_Dirty",           REF_DIR / "Console" / "Console_Dirty",    AI_DIR / "Console" / "Console_Dirty"),
    ("Console_Gap",             REF_DIR / "Console" / "Console_Gap",      AI_DIR / "Console" / "Console_Gap"),
    ("Console_Scratch",         REF_DIR / "Console" / "Console_Scratch",  AI_DIR / "Console" / "Console_Scratch"),
    ("Cylinder_Chip",           REF_DIR / "Cylinder" / "Cylinder_Chip",   AI_DIR / "Cylinder" / "Cylinder_Chip"),
    ("Cylinder_PistonMiss",     REF_DIR / "Cylinder" / "Cylinder_PistonMiss", AI_DIR / "Cylinder" / "Cylinder_PistonMiss"),
    ("Cylinder_Porosity",       REF_DIR / "Cylinder" / "Cylinder_Porosity",   AI_DIR / "Cylinder" / "Cylinder_Porosity"),
    ("Cylinder_RCS",            REF_DIR / "Cylinder" / "Cylinder_RCS",    AI_DIR / "Cylinder" / "Cylinder_RCS"),
    ("screw_defect",            REF_DIR / "screw_defect",                 AI_DIR / "screw_defect"),
    ("Wood_impurities",         REF_DIR / "Wood" / "Wood_impurities",     AI_DIR / "Wood" / "Wood_impurities"),
    ("Wood_pits",               REF_DIR / "Wood" / "Wood_pits",           AI_DIR / "Wood" / "Wood_pits"),
]


def get_images(folder: Path, exclude_name: str = None) -> list:
    files = sorted(f for f in folder.iterdir() if f.suffix in IMAGE_EXTS)
    if exclude_name:
        files = [f for f in files if not f.stem.startswith(exclude_name)]
    return files


def get_normal00(folder: Path) -> Path | None:
    for f in folder.iterdir():
        if f.stem == "normal_00" and f.suffix in IMAGE_EXTS:
            return f
    return None


def prepare_subtype(name: str, ref_folder: Path, ai_folder: Path):
    print(f"\n{'='*60}")
    print(f"  준비 중: {name}")
    print(f"{'='*60}")

    dst_base    = BASE_DIR / name
    dst_orig    = dst_base / "original_image"
    dst_normal  = dst_base / "normal_00"
    dst_ai      = dst_base / "ai_generated"

    for d in [dst_orig, dst_normal, dst_ai]:
        d.mkdir(parents=True, exist_ok=True)

    if not ref_folder.exists():
        print(f"  [SKIP] ref 폴더 없음: {ref_folder}")
        return
    if not ai_folder.exists():
        print(f"  [SKIP] ai 폴더 없음: {ai_folder}")
        return

    # --- original_image: ref_*.* (normal_00 제외) ---
    ref_imgs = get_images(ref_folder, exclude_name="normal_00")
    for src in ref_imgs:
        dst = dst_orig / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
    print(f"  original_image: {len(ref_imgs)}장 복사 완료")

    # --- normal_00: normal_00 이미지를 N장 복사 ---
    normal_src = get_normal00(ref_folder)
    if normal_src is None:
        print(f"  [SKIP] normal_00 파일 없음")
    else:
        ext = normal_src.suffix
        for i in range(1, N_NORMAL + 1):
            dst = dst_normal / f"normal_00_{i:03d}{ext}"
            if not dst.exists():
                shutil.copy2(normal_src, dst)
        print(f"  normal_00: {N_NORMAL}장 복사 완료 ({normal_src.name} → normal_00_001~{N_NORMAL:03d}{ext})")

    # --- ai_generated: 최대 N장 ---
    ai_imgs = sorted(f for f in ai_folder.iterdir() if f.suffix in IMAGE_EXTS)[:N_AI]
    for src in ai_imgs:
        dst = dst_ai / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
    print(f"  ai_generated: {len(ai_imgs)}장 복사 완료")


def main():
    print("FID 데이터 준비 시작")
    print(f"  REF   : {REF_DIR}")
    print(f"  AI    : {AI_DIR}")
    print(f"  출력  : {BASE_DIR}")
    print(f"  normal_00 복사 수: {N_NORMAL}")
    print(f"  AI 이미지 최대  : {N_AI}")

    for name, ref_folder, ai_folder in SUBTYPES:
        prepare_subtype(name, ref_folder, ai_folder)

    print(f"\n{'='*60}")
    print("  데이터 준비 완료!")
    print(f"{'='*60}")

    # 준비된 폴더 요약
    for name, _, _ in SUBTYPES:
        dst_base = BASE_DIR / name
        if dst_base.exists():
            orig_n   = len(list((dst_base / "original_image").iterdir())) if (dst_base / "original_image").exists() else 0
            normal_n = len(list((dst_base / "normal_00").iterdir())) if (dst_base / "normal_00").exists() else 0
            ai_n     = len(list((dst_base / "ai_generated").iterdir())) if (dst_base / "ai_generated").exists() else 0
            print(f"  {name:<25}  orig={orig_n:>3}  normal={normal_n:>3}  ai={ai_n:>3}")


if __name__ == "__main__":
    main()
