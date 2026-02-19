"""
Gemini API - Defect Image Augmentation
casting_Inclusoes (50), casting_Rechupe (50), screw_defect (100)

결함 위치만 변경하면서 증강 생성

사용법:
  python generate_defects.py casting_Inclusoes
  python generate_defects.py casting_Rechupe
  python generate_defects.py screw_defect
  python generate_defects.py all

백그라운드:
  nohup python -u generate_defects.py casting_Inclusoes > casting_Inclusoes.log 2>&1 &
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
GEMINI_API_KEY = "AIzaSyBy1c9mJQfBWOgJW3Nn8MTuMxVWtfwCqP8"

BASE_REFERENCE_DIR = "reference_images"
BASE_OUTPUT_DIR = "vision_ai_generated"

DELAY_BETWEEN_IMAGES = 35  # 35초
MAX_RETRIES = 3
RATE_LIMIT_BACKOFF = 600  # 10분

# ===== 결함 유형별 설정 =====
DEFECT_CONFIGS = {
    "casting_Inclusoes": {
        "total_images": 50,
        "description": "Casting inclusion defect - non-metallic foreign material trapped inside a metal casting",
        "prompt_base": (
            "Generate a new image of a metal casting part with an inclusion defect. "
            "An inclusion defect means non-metallic foreign material (such as sand, slag, or oxide) "
            "is trapped inside the metal casting surface. "
            "The first reference image shows a NORMAL casting without any defect. "
            "The other reference images show DEFECTIVE castings with inclusion defects "
            "(the defect areas are highlighted in blue in the references). "
        ),
        "prompt_key_instruction": (
            "Generate a realistic casting image WITH an inclusion defect, but place the defect "
            "at a DIFFERENT POSITION than shown in the reference defect images. "
            "The defect should look natural - a small dark spot, discoloration, or rough patch "
            "where foreign material is embedded in the metal surface. "
            "Do NOT include any blue markings or highlights in the generated image. "
            "The defect should look like a real inclusion, not artificially marked. "
        ),
        "prompt_variations": [
            "Place the inclusion defect slightly to the upper-left area of the casting surface.",
            "Place the inclusion defect near the center of the casting surface.",
            "Place the inclusion defect slightly to the lower-right area of the casting surface.",
            "Place the inclusion defect near the top edge of the casting surface.",
            "Place the inclusion defect slightly to the upper-right area of the casting surface.",
            "Place the inclusion defect near the bottom area of the casting surface.",
            "Place the inclusion defect slightly to the left side of the casting surface.",
            "Place the inclusion defect near the lower-left area of the casting surface.",
            "Place the inclusion defect slightly off-center to the right.",
            "Place the inclusion defect near the middle-left area of the casting surface.",
        ],
        "prompt_style": (
            "Industrial inspection photography, consistent lighting, sharp focus on the casting surface. "
            "Maintain the same casting part type, material, and overall appearance as the references. "
            "CRITICAL: Keep the same casting shape and material. Only change the defect position. "
            "Do NOT add blue markings. The defect must look completely natural and realistic."
        ),
    },

    "casting_Rechupe": {
        "total_images": 50,
        "description": "Casting rechupe (shrinkage/porosity) defect - cavities or voids formed during metal solidification",
        "prompt_base": (
            "Generate a new image of a metal casting part with a rechupe (shrinkage) defect. "
            "A rechupe defect is a cavity, void, or sunken area formed when the metal shrinks "
            "during solidification, leaving holes or depressions on or inside the casting. "
            "The first reference image shows a NORMAL casting without any defect. "
            "The other reference images show DEFECTIVE castings with rechupe/shrinkage defects "
            "(the defect areas are highlighted in blue in the references). "
        ),
        "prompt_key_instruction": (
            "Generate a realistic casting image WITH a rechupe/shrinkage defect, but place the defect "
            "at a DIFFERENT POSITION than shown in the reference defect images. "
            "The defect should look natural - a small cavity, depression, sunken area, or porosity "
            "where the metal failed to fill properly during casting. "
            "Do NOT include any blue markings or highlights in the generated image. "
            "The defect should look like a real shrinkage void, not artificially marked. "
        ),
        "prompt_variations": [
            "Place the shrinkage defect slightly to the upper-left area of the casting surface.",
            "Place the shrinkage defect near the center of the casting surface.",
            "Place the shrinkage defect slightly to the lower-right area of the casting surface.",
            "Place the shrinkage defect near the top edge of the casting surface.",
            "Place the shrinkage defect slightly to the upper-right area of the casting surface.",
            "Place the shrinkage defect near the bottom area of the casting surface.",
            "Place the shrinkage defect slightly to the left side of the casting surface.",
            "Place the shrinkage defect near the lower-left area of the casting surface.",
            "Place the shrinkage defect slightly off-center to the right.",
            "Place the shrinkage defect near the middle-left area of the casting surface.",
        ],
        "prompt_style": (
            "Industrial inspection photography, consistent lighting, sharp focus on the casting surface. "
            "Maintain the same casting part type, material, and overall appearance as the references. "
            "CRITICAL: Keep the same casting shape and material. Only change the defect position. "
            "Do NOT add blue markings. The defect must look completely natural and realistic."
        ),
    },

    "screw_defect": {
        "total_images": 100,
        "description": "Screw defect - various manufacturing defects on screws",
        "prompt_base": (
            "Generate a new image of a screw with a manufacturing defect. "
            "The first reference image shows a NORMAL screw without any defect. "
            "The other reference images show DEFECTIVE screws with visible manufacturing defects "
            "(the defect areas are highlighted in blue in the references). "
        ),
        "prompt_key_instruction": (
            "Generate a realistic screw image WITH a manufacturing defect, but place the defect "
            "at a DIFFERENT POSITION than shown in the reference defect images. "
            "The defect should look natural - a crack, burr, deformation, missing thread, "
            "surface damage, or other manufacturing flaw on the screw. "
            "Do NOT include any blue markings or highlights in the generated image. "
            "The defect should look like a real manufacturing defect, not artificially marked. "
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
            "Maintain the same screw type, size, and overall appearance as the references. "
            "CRITICAL: Keep the same screw shape, thread pattern, and material. Only change the defect position. "
            "Do NOT add blue markings. The defect must look completely natural and realistic."
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
    """레퍼런스 이미지 로드 - 정렬하여 첫 번째가 정상, 나머지가 결함"""
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
        print(f"  - 1st image: Normal (no defect)", flush=True)
        print(f"  - 2nd~5th images: Defect images (blue-marked)", flush=True)
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
    """결함 유형별 프롬프트 생성 - 위치 변화 중심"""
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
    client = genai.Client(api_key=GEMINI_API_KEY)

    # 레퍼런스 이미지 로드
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

                # 모든 레퍼런스 이미지를 함께 전송
                # 순서: normal image, defect images, text prompt
                contents = []

                # 정상 이미지 (첫 번째)
                normal_ref = ref_data_list[0]
                contents.append(types.Part.from_bytes(
                    data=normal_ref['bytes'],
                    mime_type='image/png'
                ))

                # 결함 이미지들 (순환하며 1~2장 선택)
                defect_refs = ref_data_list[1:]  # 결함 이미지만
                # 매 생성마다 다른 결함 레퍼런스 조합 사용
                defect_idx = i % len(defect_refs)
                selected_defect = defect_refs[defect_idx]
                contents.append(types.Part.from_bytes(
                    data=selected_defect['bytes'],
                    mime_type='image/png'
                ))

                # 추가 결함 레퍼런스 (있으면 하나 더)
                if len(defect_refs) > 1:
                    extra_idx = (i + 1) % len(defect_refs)
                    if extra_idx != defect_idx:
                        extra_defect = defect_refs[extra_idx]
                        contents.append(types.Part.from_bytes(
                            data=extra_defect['bytes'],
                            mime_type='image/png'
                        ))

                # 텍스트 프롬프트
                contents.append(prompt)

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
        help="Override image count (e.g. --count 10 for test run)"
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
