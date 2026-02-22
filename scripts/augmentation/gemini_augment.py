"""
Gemini API - Defect Image Augmentation
casting_Inclusoes (150), casting_Rechupe (150), screw_defect (300)

결함 위치만 변경하면서 증강 생성 (결함 크기는 원본과 동일하게 유지)

사용법:
  cd /home/jjh0709/gitrepo/VISION-Instance-Seg/scripts/augmentation
  python gemini_augment.py casting_Inclusoes
  python gemini_augment.py casting_Rechupe
  python gemini_augment.py screw_defect
  python gemini_augment.py all

백그라운드 (scripts/augmentation/ 디렉토리에서 실행):
  cd /home/jjh0709/gitrepo/VISION-Instance-Seg/scripts/augmentation
  nohup python -u gemini_augment.py casting_Inclusoes > casting_Inclusoes.log 2>&1 &
  nohup python -u gemini_augment.py casting_Rechupe   > casting_Rechupe.log   2>&1 &
  nohup python -u gemini_augment.py screw_defect      > screw_defect.log      2>&1 &

  # PID 확인
  jobs -l
  tail -f casting_Inclusoes.log
"""

from google import genai
from google.genai import types
from pathlib import Path
import time
from PIL import Image
import io
from datetime import datetime, timedelta
import sys
import json
import argparse

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# ===== Gemini API 설정 =====
# 환경변수로 관리: export GEMINI_API_KEY="your_key"
import os
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyBlPF2KZXmvcHG_3IqKyugCym46-86dirY")

BASE_REFERENCE_DIR = "reference_images"
BASE_OUTPUT_DIR = "vision_ai_generated"

DELAY_BETWEEN_IMAGES = 35  # 35초
MAX_RETRIES = 3
RATE_LIMIT_BACKOFF = 600  # 10분

# ===== 결함 유형별 설정 =====
DEFECT_CONFIGS = {
    "casting_Inclusoes": {
        "total_images": 150,
        "description": "Casting inclusion defect - non-metallic foreign material trapped inside a metal casting",
        "prompt_base": (
            "Generate a new image of a metal casting part with an inclusion defect. "
            "An inclusion defect means non-metallic foreign material (such as sand, slag, or oxide) "
            "is trapped inside the metal casting surface, appearing as a small dark spot or discoloration. "
            "The FIRST image is a NORMAL casting without any defect — use it as the base appearance reference. "
            "The REMAINING images are DEFECTIVE castings with inclusion defects "
            "(defect areas are highlighted with a BLUE BORDER in the reference images — "
            "the blue marking is only for reference, do NOT include it in the output). "
        ),
        "prompt_key_instruction": (
            "MANDATORY: The output image MUST contain exactly one clearly visible inclusion defect "
            "directly on the METAL CASTING SURFACE. "
            "Do NOT generate a defect-free image — a clean output with no defect is WRONG and unacceptable. "
            "The defect MUST be placed ON the metallic surface of the casting part, "
            "not on the background or any non-metallic area. "
            "Generate a realistic casting image WITH an inclusion defect, but place the defect "
            "at a DIFFERENT POSITION than shown in the reference defect images. "
            "CRITICAL — DEFECT SIZE: "
            "Look at the blue-bordered rectangles in the reference defect images. "
            "The defect you generate must be THE EXACT SAME SIZE as those blue boxes, or SMALLER. "
            "Those blue boxes are already very small relative to the full image — keep it that way. "
            "Do NOT enlarge the defect. A correct defect is a tiny, barely-noticeable dark spot "
            "or discoloration that blends into the casting surface. "
            "If the defect takes up more than 2% of the image width, it is TOO LARGE — make it smaller. "
            "Do NOT include any blue markings or highlights in the generated image. "
            "The defect should look like a real inclusion: a tiny dark or discolored patch on the metal, "
            "not artificially marked. "
        ),
        "prompt_variations": [
            "Place the inclusion defect in the upper-left quadrant of the casting surface.",
            "Place the inclusion defect slightly left of center on the casting surface.",
            "Place the inclusion defect in the lower-right area of the casting surface.",
            "Place the inclusion defect near the top-center of the casting surface.",
            "Place the inclusion defect in the upper-right area of the casting surface.",
            "Place the inclusion defect near the bottom-center of the casting surface.",
            "Place the inclusion defect on the left side of the casting surface.",
            "Place the inclusion defect in the lower-left area of the casting surface.",
            "Place the inclusion defect slightly right of center on the casting surface.",
            "Place the inclusion defect in the middle-right area of the casting surface.",
        ],
        "prompt_style": (
            "Industrial inspection photography, even and consistent lighting, sharp focus on the casting surface. "
            "Maintain the exact same casting part shape, material texture, color, and background as the FIRST (normal) reference. "
            "CRITICAL: Do NOT change the casting shape, size, or overall composition. "
            "Only add one TINY, SUBTLE defect at a new position on the METAL SURFACE — the same size as the blue boxes in the references. "
            "Do NOT add blue markings. The defect must be nearly invisible at first glance, "
            "matching the size seen inside the blue reference boxes. "
            "REMINDER: The final image MUST have a defect on the metal casting surface — "
            "do not output a clean, defect-free image."
        ),
    },

    "casting_Rechupe": {
        "total_images": 150,
        "description": "Casting rechupe (shrinkage/porosity) defect - cavities or voids formed during metal solidification",
        "prompt_base": (
            "Generate a new image of a metal casting part with a rechupe (shrinkage) defect. "
            "A rechupe defect is a small cavity, sunken area, or void formed when the metal shrinks "
            "during solidification, leaving a small depression or hole on the casting surface. "
            "The FIRST image is a NORMAL casting without any defect — use it as the base appearance reference. "
            "The REMAINING images are DEFECTIVE castings with rechupe/shrinkage defects "
            "(defect areas are highlighted with a BLUE BORDER in the reference images — "
            "the blue marking is only for reference, do NOT include it in the output). "
        ),
        "prompt_key_instruction": (
            "MANDATORY: The output image MUST contain exactly one visible rechupe/shrinkage defect. "
            "CRITICAL PLACEMENT RULE: The defect MUST appear ONLY on the METAL CASTING SURFACE "
            "(the metallic object itself) — absolutely NOT on the background. "
            "Look at the FIRST (normal) reference image carefully: "
            "the casting part is the metallic/grey object occupying the main area of the image; "
            "the background is the non-metallic surrounding area (floor, wall, or backdrop). "
            "The defect MUST be placed strictly within the visible boundaries of the metallic casting part. "
            "The background area must remain completely clean and identical to the normal reference image — "
            "no defects, marks, or changes in the background. "
            "Generate a realistic casting image WITH a rechupe/shrinkage defect, but place the defect "
            "at a DIFFERENT POSITION than shown in the reference defect images. "
            "CRITICAL — DEFECT SIZE: "
            "Look at the blue-bordered rectangles in the reference defect images. "
            "The defect you generate must be THE EXACT SAME SIZE as those blue boxes, or SMALLER. "
            "Those blue boxes are already very small relative to the full image — keep it that way. "
            "Do NOT enlarge the defect. A correct defect is a small cavity, pit, or sunken spot "
            "that is barely noticeable on the casting surface. "
            "If the defect takes up more than 4% of the image width, it is TOO LARGE — make it smaller. "
            "Do NOT include any blue markings or highlights in the generated image. "
            "The defect should look like a real shrinkage void: a small pit or depression on the metal surface, "
            "not artificially marked, and never in the background. "
        ),
        "prompt_variations": [
            "Place the shrinkage defect in the upper-left area of the metal casting part surface (not the background).",
            "Place the shrinkage defect slightly left of center on the metal casting part surface (not the background).",
            "Place the shrinkage defect in the lower-right area of the metal casting part surface (not the background).",
            "Place the shrinkage defect near the top-center of the metal casting part surface (not the background).",
            "Place the shrinkage defect in the upper-right area of the metal casting part surface (not the background).",
            "Place the shrinkage defect near the bottom-center of the metal casting part surface (not the background).",
            "Place the shrinkage defect on the left side of the metal casting part surface (not the background).",
            "Place the shrinkage defect in the lower-left area of the metal casting part surface (not the background).",
            "Place the shrinkage defect slightly right of center on the metal casting part surface (not the background).",
            "Place the shrinkage defect in the middle-right area of the metal casting part surface (not the background).",
        ],
        "prompt_style": (
            "Industrial inspection photography, even and consistent lighting, sharp focus on the casting surface. "
            "Maintain the exact same casting part shape, material texture, color, and background as the FIRST (normal) reference. "
            "CRITICAL: Do NOT change the casting shape, size, or overall composition. "
            "Only add one TINY, SUBTLE defect on the METAL CASTING PART SURFACE — the same size as the blue boxes in the references. "
            "The defect must be ON the metallic object, NOT anywhere in the background. "
            "The background must look exactly as in the normal reference image. "
            "Do NOT add blue markings. The defect must be nearly invisible at first glance, "
            "matching the size seen inside the blue reference boxes."
        ),
    },

    "screw_defect": {
        "total_images": 300,
        "description": "Screw defect - various manufacturing defects on screws",
        "prompt_base": (
            "Generate a new image of a screw with a manufacturing defect. "
            "The FIRST image is a NORMAL screw without any defect — use it as the base appearance reference. "
            "The REMAINING images are DEFECTIVE screws with visible manufacturing defects. "
            "WARNING: The reference defect images contain BLUE RECTANGLE BORDERS drawn around the defect areas. "
            "These blue rectangles are annotation markers added for reference ONLY — "
            "they are NOT part of the real screw image and must NEVER appear in the output. "
        ),
        "prompt_key_instruction": (
            "ABSOLUTE RULE — NO BLUE MARKINGS: "
            "The output image must contain ZERO blue rectangles, blue borders, blue boxes, "
            "blue outlines, or any blue annotation markings of any kind. "
            "The blue rectangles visible in the reference images are annotation tools used to mark defect locations — "
            "they do not exist on the actual screw. Generate the screw as it would appear in reality, "
            "with NO overlaid graphics, NO colored boxes, NO highlighted regions. "
            "Generate a realistic screw image WITH a manufacturing defect, but place the defect "
            "at a DIFFERENT POSITION along the screw than shown in the reference defect images. "
            "CRITICAL — DEFECT SIZE: "
            "Use the blue-bordered rectangles in the reference images only to judge the SIZE of the defect — "
            "not to copy their appearance. The defect must be the same size as those boxes, or SMALLER. "
            "The defect should be subtle: a small crack, burr, nick, or surface damage on one section of thread. "
            "If the defect spans more than 20% of the screw length, it is TOO LARGE — make it smaller. "
            "The defect must look like a real manufacturing flaw on the screw surface, "
            "with no blue color, no box, no border, no marking of any kind around it. "
        ),
        "prompt_variations": [
            "Place the defect on the upper portion of the screw shaft/threads.",
            "Place the defect near the middle of the screw shaft/threads.",
            "Place the defect on the lower portion of the screw shaft near the tip.",
            "Place the defect near the screw head area.",
            "Place the defect on the upper-middle section of the threads.",
            "Place the defect near the transition between head and shaft.",
            "Place the defect on the lower-middle section of the threads.",
            "Place the defect near the tip of the screw.",
            "Place the defect slightly below the head on the shaft.",
            "Place the defect across multiple threads in the middle section.",
        ],
        "prompt_style": (
            "Industrial inspection photography, consistent lighting, sharp focus on the screw surface. "
            "Maintain the exact same screw type, size, thread pattern, and overall appearance as the FIRST (normal) reference. "
            "CRITICAL: Keep the same screw shape, material, and background. "
            "Only add one SMALL physical defect on the screw surface at a new position. "
            "The final image must look like a real photograph with absolutely no overlaid graphics, "
            "no blue rectangles, no colored borders, no annotation marks of any kind. "
            "The defect must be subtle and small, matching the SIZE (not the appearance) "
            "of what was shown inside the blue reference boxes."
        ),
    },
}


# ===== 진행상황 관리 =====
def get_progress_file(defect_type):
    return f"progress_{defect_type}.json"


def load_progress(defect_type):
    progress_file = get_progress_file(defect_type)
    if Path(progress_file).exists():
        with open(progress_file, 'r') as f:
            return json.load(f)
    return {
        'completed': [],
        'failed': [],
        'last_successful_index': -1,
        'start_time': None
    }


def save_progress(progress, defect_type):
    progress_file = get_progress_file(defect_type)
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2)


# ===== 이미지 로드 =====
def load_reference_images(defect_type):
    """레퍼런스 이미지 로드 - 정렬하여 첫 번째가 정상, 나머지가 결함 (전부 로드)"""
    ref_dir = Path(BASE_REFERENCE_DIR) / defect_type

    reference_images = sorted(
        list(ref_dir.glob("*.jpg")) +
        list(ref_dir.glob("*.JPG")) +
        list(ref_dir.glob("*.jpeg")) +
        list(ref_dir.glob("*.JPEG")) +
        list(ref_dir.glob("*.png")) +
        list(ref_dir.glob("*.PNG"))
    )

    if not reference_images:
        print(f"ERROR: No images in '{ref_dir}/'", flush=True)
        print(f"Please add reference images:", flush=True)
        print(f"  - 1st image: Normal (no defect) — named so it sorts first (e.g. normal_00.jpg)", flush=True)
        print(f"  - 2nd~10th images: Defect images with blue border bbox (ref_01_*.jpg ...)", flush=True)
        exit(1)

    if len(reference_images) < 2:
        print(f"ERROR: Need at least 2 images (1 normal + 1 defect) in '{ref_dir}/'", flush=True)
        exit(1)

    ref_data_list = []
    for idx, ref_path in enumerate(reference_images):
        img = Image.open(ref_path)
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        role = "NORMAL (no defect)" if idx == 0 else f"DEFECT sample #{idx}"
        ref_data_list.append({
            'name': ref_path.name,
            'bytes': img_byte_arr.getvalue(),
            'role': role,
            'is_normal': idx == 0
        })
        print(f"  Loaded: {ref_path.name} [{role}]", flush=True)

    return ref_data_list


# ===== 프롬프트 생성 =====
def generate_prompt(config, index):
    """결함 유형별 프롬프트 생성 - 위치 변화 중심, 크기 강제 제약 포함"""
    variation = config["prompt_variations"][index % len(config["prompt_variations"])]

    prompt = (
        config["prompt_base"] +
        config["prompt_key_instruction"] +
        variation + " " +
        config["prompt_style"]
    )
    return prompt


# ===== 단일 결함 유형 생성 =====
def run_generation(defect_type, count_override=None):
    if defect_type not in DEFECT_CONFIGS:
        print(f"ERROR: Unknown defect type '{defect_type}'", flush=True)
        print(f"Available types: {list(DEFECT_CONFIGS.keys())}", flush=True)
        exit(1)

    config = DEFECT_CONFIGS[defect_type]
    total_images = count_override if count_override else config["total_images"]
    output_dir = Path(BASE_OUTPUT_DIR) / defect_type
    output_dir.mkdir(parents=True, exist_ok=True)

    # 진행상황 로드
    progress = load_progress(defect_type)
    if progress['start_time'] is None:
        progress['start_time'] = time.time()
        save_progress(progress, defect_type)

    start_time = progress['start_time']
    start_index = progress['last_successful_index'] + 1

    print("=" * 80, flush=True)
    print(f"GEMINI API - {defect_type} ({total_images} Images)", flush=True)
    print("=" * 80, flush=True)
    print(f"Description:      {config['description']}", flush=True)
    print(f"Started at:       {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"Current time:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"Target:           {total_images} images", flush=True)
    print(f"Output:           {output_dir}/", flush=True)
    print(f"Progress:         {len(progress['completed'])}/{total_images} completed", flush=True)

    if start_index > 0:
        print(f"Resuming from:    Index {start_index}", flush=True)

    remaining = total_images - len(progress['completed'])
    eta_seconds = remaining * DELAY_BETWEEN_IMAGES
    eta_time = datetime.now() + timedelta(seconds=eta_seconds)
    print(f"ETA:              {eta_time.strftime('%Y-%m-%d %H:%M:%S')} (~{int(eta_seconds/3600)}h {int((eta_seconds%3600)/60)}m)", flush=True)
    print("=" * 80, flush=True)

    # API 클라이언트
    if not GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY 환경변수가 설정되지 않았습니다.", flush=True)
        print("  export GEMINI_API_KEY='your_new_api_key'", flush=True)
        exit(1)
    client = genai.Client(api_key=GEMINI_API_KEY)

    # 레퍼런스 이미지 로드 (전부)
    print(f"\n[1/2] Loading reference images for {defect_type}...", flush=True)
    ref_data_list = load_reference_images(defect_type)
    print(f"  Total {len(ref_data_list)} reference images loaded", flush=True)
    print(f"  (1 normal + {len(ref_data_list)-1} defect samples)", flush=True)

    # 생성 루프
    print(f"\n[2/2] Starting generation...", flush=True)
    print("=" * 80, flush=True)

    for i in range(start_index, total_images):
        # 이미 완료된 인덱스 스킵
        if i in progress['completed']:
            continue

        retry_count = 0

        while retry_count < MAX_RETRIES:
            try:
                # 진행률
                completed_count = len(progress['completed'])
                progress_pct = (completed_count / total_images) * 100

                # ETA
                if completed_count > 0:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / completed_count
                    remaining_imgs = total_images - completed_count
                    remaining_secs = avg_time * remaining_imgs
                    eta = datetime.now() + timedelta(seconds=remaining_secs)
                    rh = int(remaining_secs / 3600)
                    rm = int((remaining_secs % 3600) / 60)
                else:
                    remaining_secs = total_images * DELAY_BETWEEN_IMAGES
                    eta = datetime.now() + timedelta(seconds=remaining_secs)
                    rh = int(remaining_secs / 3600)
                    rm = int((remaining_secs % 3600) / 60)

                print(f"\n{'='*80}", flush=True)
                print(f"[{i+1:3d}/{total_images}] {defect_type} | {progress_pct:.1f}% | {datetime.now().strftime('%m/%d %H:%M:%S')}", flush=True)
                print(f"ETA: {eta.strftime('%m/%d %H:%M:%S')} (~{rh}h {rm}m left)", flush=True)
                print(f"{'='*80}", flush=True)

                # 프롬프트 생성
                prompt = generate_prompt(config, i)

                # ── API contents 구성 ──────────────────────────────────────────────
                # 순서: [정상 이미지] [결함 이미지 1~9장] [텍스트 프롬프트]
                # 결함 이미지는 매 호출마다 회전(rotating) 방식으로 일부 선택
                contents = []

                # 1) 정상 이미지 (항상 포함)
                normal_ref = ref_data_list[0]
                contents.append(types.Part.from_bytes(
                    data=normal_ref['bytes'],
                    mime_type='image/png'
                ))

                # 2) 결함 이미지 - 전체를 순환하며 최대 4장 선택
                #    (토큰 절약: 4장으로 제한, 매 호출마다 다른 조합)
                defect_refs = ref_data_list[1:]
                n_defect = len(defect_refs)
                MAX_DEFECT_REFS = min(4, n_defect)  # 최대 4장

                # 시작 인덱스를 i에 따라 이동 → 매 호출마다 다른 조합
                start_ref = i % n_defect
                selected_indices = [(start_ref + k) % n_defect for k in range(MAX_DEFECT_REFS)]

                for ref_idx in selected_indices:
                    ref = defect_refs[ref_idx]
                    contents.append(types.Part.from_bytes(
                        data=ref['bytes'],
                        mime_type='image/png'
                    ))

                # 3) 텍스트 프롬프트
                contents.append(prompt)

                print(f"  Refs: normal + {len(selected_indices)} defect refs (indices {selected_indices})", flush=True)

                # Gemini API 호출
                response = client.models.generate_content(
                    model='gemini-2.5-flash-image',
                    contents=contents,
                    config=types.GenerateContentConfig(
                        temperature=0.3,
                        response_modalities=["Image"]
                    )
                )

                saved = False
                if response.candidates:
                    for part in response.candidates[0].content.parts:
                        if part.inline_data:
                            image_data = part.inline_data.data
                            save_path = output_dir / f"{defect_type}_{i:03d}.png"

                            with open(save_path, 'wb') as f:
                                f.write(image_data)

                            progress['completed'].append(i)
                            progress['last_successful_index'] = i
                            save_progress(progress, defect_type)
                            saved = True

                            print(f"  SAVED: {save_path.name}", flush=True)
                            print(f"  Success: {len(progress['completed'])}/{total_images} | Failed: {len(progress['failed'])}", flush=True)
                            break

                if saved:
                    break
                else:
                    retry_count += 1
                    print(f"  No image in response (attempt {retry_count}/{MAX_RETRIES})", flush=True)
                    time.sleep(30)

            except Exception as e:
                error_msg = str(e)

                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                    retry_count += 1
                    print(f"  RATE LIMIT! Waiting {RATE_LIMIT_BACKOFF}s...", flush=True)
                    print(f"  Error: {error_msg[:150]}", flush=True)
                    save_progress(progress, defect_type)
                    time.sleep(RATE_LIMIT_BACKOFF)

                elif "quota" in error_msg.lower() or "daily" in error_msg.lower():
                    print(f"  QUOTA EXCEEDED!", flush=True)
                    print(f"  Error: {error_msg}", flush=True)
                    print(f"  Progress: {len(progress['completed'])}/{total_images}", flush=True)
                    save_progress(progress, defect_type)
                    exit(1)

                else:
                    retry_count += 1
                    print(f"  ERROR: {error_msg[:200]}", flush=True)
                    time.sleep(60)

        if retry_count >= MAX_RETRIES:
            progress['failed'].append(i)
            save_progress(progress, defect_type)
            print(f"  FAILED after {MAX_RETRIES} retries. Skipping...", flush=True)

        # 다음 이미지 전 대기
        if i < total_images - 1:
            next_time = datetime.now() + timedelta(seconds=DELAY_BETWEEN_IMAGES)
            print(f"  Waiting {DELAY_BETWEEN_IMAGES}s until {next_time.strftime('%H:%M:%S')}...", flush=True)
            time.sleep(DELAY_BETWEEN_IMAGES)

    # 완료
    total_time = time.time() - start_time

    print("\n" + "=" * 80, flush=True)
    print(f"GENERATION COMPLETE: {defect_type}", flush=True)
    print("=" * 80, flush=True)
    print(f"Total images:     {total_images}", flush=True)
    print(f"Successful:       {len(progress['completed'])} ({len(progress['completed'])/total_images*100:.1f}%)", flush=True)
    print(f"Failed:           {len(progress['failed'])} ({len(progress['failed'])/total_images*100:.1f}%)", flush=True)
    print(f"Total time:       {int(total_time/3600)}h {int((total_time%3600)/60)}m {int(total_time%60)}s", flush=True)
    print(f"Output directory: {output_dir}/", flush=True)
    print("=" * 80, flush=True)

    if progress['failed']:
        print(f"\n  Failed indices: {sorted(progress['failed'])}", flush=True)

    return len(progress['completed']), len(progress['failed'])


# ===== 메인 =====
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemini Defect Image Augmentation")
    parser.add_argument(
        "defect_type",
        choices=["casting_Inclusoes", "casting_Rechupe", "screw_defect", "all"],
        help="Type of defect to generate"
    )
    parser.add_argument(
        "--count", type=int, default=None,
        help="Override image count (e.g. --count 5 for test run)"
    )
    args = parser.parse_args()

    if args.defect_type == "all":
        results = {}
        for dtype in ["casting_Inclusoes", "casting_Rechupe", "screw_defect"]:
            print(f"\n{'#'*80}", flush=True)
            print(f"# Starting: {dtype}", flush=True)
            print(f"{'#'*80}\n", flush=True)
            success, failed = run_generation(dtype, count_override=args.count)
            results[dtype] = {"success": success, "failed": failed}

        print("\n" + "=" * 80, flush=True)
        print("ALL DEFECT TYPES COMPLETE", flush=True)
        print("=" * 80, flush=True)
        for dtype, r in results.items():
            total = args.count if args.count else DEFECT_CONFIGS[dtype]["total_images"]
            print(f"  {dtype}: {r['success']}/{total} success, {r['failed']} failed", flush=True)
        print("=" * 80, flush=True)
    else:
        run_generation(args.defect_type, count_override=args.count)
