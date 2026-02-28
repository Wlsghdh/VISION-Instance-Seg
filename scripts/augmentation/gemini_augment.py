"""
Gemini API - Defect Image Augmentation
casting_Inclusoes (150), casting_Rechupe (150), screw_defect (300)
Console (300 = Collision×75 + Dirty×75 + Gap×75 + Scratch×75)
Cylinder (300 = Chip×75 + PistonMiss×75 + Porosity×75 + RCS×75)
Wood    (300 = impurities×150 + pits×150)

결함 위치만 변경하면서 증강 생성 (결함 크기는 원본과 동일하게 유지)
Console/Cylinder/Wood: bbox 없는 원본 data 이미지를 defect 레퍼런스로 사용

사용법:
  cd /home/jjh0709/gitrepo/VISION-Instance-Seg/scripts/augmentation
  python gemini_augment.py Console
  python gemini_augment.py Cylinder
  python gemini_augment.py Wood
  python gemini_augment.py all

백그라운드:
  nohup python -u gemini_augment.py Console  > Console.log  2>&1 &
  nohup python -u gemini_augment.py Cylinder > Cylinder.log 2>&1 &
  nohup python -u gemini_augment.py Wood     > Wood.log     2>&1 &
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

import os
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

BASE_REFERENCE_DIR = "reference_images"
BASE_OUTPUT_DIR    = "vision_ai_generated"
DATA_ROOT          = "/home/jjh0709/gitrepo/VISION-Instance-Seg/data"

DELAY_BETWEEN_IMAGES = 10   # 초 (기존 35 → 10)
MAX_RETRIES          = 3
RATE_LIMIT_BACKOFF   = 600  # 10분

# ===== 결함 유형별 설정 =====
DEFECT_CONFIGS = {

    # ── 기존 (bbox 레퍼런스 방식 유지) ──────────────────────────────
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
            "Generate a realistic casting image WITH an inclusion defect, but place the defect "
            "at a DIFFERENT POSITION than shown in the reference defect images. "
            "CRITICAL — DEFECT SIZE: the defect must be THE EXACT SAME SIZE as the blue boxes, or SMALLER. "
            "Do NOT include any blue markings or highlights in the generated image. "
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
            "Industrial inspection photography, even lighting, sharp focus on the casting surface. "
            "Maintain the exact same casting part shape, material texture, color, and background as the FIRST (normal) reference. "
            "Only add one TINY, SUBTLE defect at a new position. Do NOT add blue markings. "
            "REMINDER: The final image MUST have a defect on the metal casting surface."
        ),
    },

    "casting_Rechupe": {
        "total_images": 150,
        "description": "Casting rechupe (shrinkage/porosity) defect",
        "prompt_base": (
            "Generate a new image of a metal casting part with a rechupe (shrinkage) defect. "
            "A rechupe defect is a small cavity, sunken area, or void on the casting surface. "
            "The FIRST image is a NORMAL casting — use it as the base appearance reference. "
            "The REMAINING images are DEFECTIVE castings with rechupe defects "
            "(defect areas highlighted with a BLUE BORDER for reference only — do NOT include in output). "
        ),
        "prompt_key_instruction": (
            "MANDATORY: Output MUST contain exactly one visible rechupe/shrinkage defect ON THE METAL SURFACE. "
            "The background must remain identical to the normal reference. "
            "Place the defect at a DIFFERENT POSITION than shown in references. "
            "DEFECT SIZE: same as the blue boxes or SMALLER. Do NOT include blue markings. "
        ),
        "prompt_variations": [
            "Place the shrinkage defect in the upper-left area of the casting surface.",
            "Place the shrinkage defect slightly left of center on the casting surface.",
            "Place the shrinkage defect in the lower-right area of the casting surface.",
            "Place the shrinkage defect near the top-center of the casting surface.",
            "Place the shrinkage defect in the upper-right area of the casting surface.",
            "Place the shrinkage defect near the bottom-center of the casting surface.",
            "Place the shrinkage defect on the left side of the casting surface.",
            "Place the shrinkage defect in the lower-left area of the casting surface.",
            "Place the shrinkage defect slightly right of center on the casting surface.",
            "Place the shrinkage defect in the middle-right area of the casting surface.",
        ],
        "prompt_style": (
            "Industrial inspection photography, consistent lighting, sharp focus on the casting. "
            "Maintain exact casting shape, texture, color, background as the FIRST (normal) reference. "
            "Only add one TINY, SUBTLE defect. Do NOT add blue markings."
        ),
    },

    "screw_defect": {
        "total_images": 300,
        "description": "Screw defect - various manufacturing defects on screws",
        "prompt_base": (
            "Generate a new image of a screw with a manufacturing defect. "
            "The FIRST image is a NORMAL screw — use it as the base reference. "
            "The REMAINING images are DEFECTIVE screws. "
            "WARNING: Reference images contain BLUE RECTANGLE BORDERS — annotation markers ONLY. "
            "NEVER include blue rectangles in the output. "
        ),
        "prompt_key_instruction": (
            "ABSOLUTE RULE — NO BLUE MARKINGS in output. "
            "Place the defect at a DIFFERENT POSITION along the screw than in references. "
            "DEFECT SIZE: same size as blue boxes or SMALLER. No blue color/boxes/borders in output. "
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
            "Industrial inspection photography, consistent lighting, sharp focus on the screw. "
            "Keep the same screw type, material, background. Only add one SMALL defect at a new position. "
            "No overlaid graphics, no blue rectangles, no colored borders."
        ),
    },

    # ── Console (bbox 없는 원본 이미지 방식) ─────────────────────────
    "Console_Collision": {
        "total_images": 75,
        "ref_dir": "Console/Console_Collision",   # normal_00.png 위치
        "out_dir": "Console/Console_Collision",
        "data_ref": {                             # defect 레퍼런스: bbox 없는 원본
            "data_dir": f"{DATA_ROOT}/Console/train",
            "annotation": f"{DATA_ROOT}/Console/train/_annotations.coco.json",
            "cat_id": 0,
            "n_samples": 9,
        },
        "description": "Console collision defect - extremely subtle micro-crack or impact chip on console surface",
        "prompt_base": (
            "Generate a new image of an electronic console/control panel part with a collision defect. "
            "The FIRST image is a NORMAL console — use it as the base appearance reference (keep everything identical). "
            "The REMAINING images are DEFECTIVE consoles showing real collision defects: "
            "very subtle micro-cracks, tiny chips, or barely-visible impact marks on the panel surface. "
            "Study these defect examples carefully to understand the look, size, and texture of the defect. "
        ),
        "prompt_key_instruction": (
            "MANDATORY: Output MUST contain exactly one collision defect — "
            "a very subtle micro-crack, tiny chip, or barely-visible impact mark. "
            "CRITICAL — EXTREME SUBTLETY: The defect must be nearly invisible at first glance, "
            "detectable only under close inspection. "
            "Place the defect at a DIFFERENT POSITION than in the reference images. "
            "Keep defect SIZE similar to what is shown in the reference images. "
            "Do NOT generate a defect-free image. "
            "Do NOT add any colored boxes, borders, or annotation marks — the output must look like a real photo. "
        ),
        "prompt_variations": [
            "Place the tiny collision mark in the upper-left area of the console panel.",
            "Place the micro-crack or chip slightly left of center on the console panel.",
            "Place the subtle collision defect in the lower-right area of the console surface.",
            "Place the tiny impact mark near the top-center of the console panel.",
            "Place the micro collision defect in the upper-right area of the console surface.",
            "Place the subtle impact chip near the lower-center of the console panel.",
            "Place the tiny collision mark on the left-side area of the console.",
            "Place the micro-crack in the lower-left area of the console panel.",
            "Place the subtle collision defect slightly right of center on the console.",
            "Place the tiny impact mark in the middle-right area of the console panel.",
        ],
        "prompt_style": (
            "Industrial inspection photography with slightly varied lighting. "
            "Maintain exact same console shape, panel layout, color, and composition as the FIRST (normal) image. "
            "Only add one EXTREMELY SUBTLE collision defect — nearly invisible, same size as in reference defect images. "
            "Output must be a clean, realistic photo with no overlaid graphics or annotation marks."
        ),
    },

    "Console_Dirty": {
        "total_images": 75,
        "ref_dir": "Console/Console_Dirty",
        "out_dir": "Console/Console_Dirty",
        "data_ref": {
            "data_dir": f"{DATA_ROOT}/Console/train",
            "annotation": f"{DATA_ROOT}/Console/train/_annotations.coco.json",
            "cat_id": 1,
            "n_samples": 9,
        },
        "description": "Console dirty defect - smudge, fingerprint, or contamination mark on console surface",
        "prompt_base": (
            "Generate a new image of an electronic console/control panel part with a dirty/contamination defect. "
            "The FIRST image is a NORMAL console — use it as the base appearance reference. "
            "The REMAINING images are DEFECTIVE consoles showing real dirty defects: "
            "smudge marks, fingerprint impressions, grease stains, or contamination on the panel surface. "
            "Study these examples to understand the look and texture of the dirty defect. "
        ),
        "prompt_key_instruction": (
            "MANDATORY: Output MUST contain exactly one dirty/contamination defect — "
            "a visible smudge, fingerprint mark, grease stain, or dust accumulation on the console surface. "
            "The dirty mark should look like genuine contamination: irregular in shape, with natural smear texture. "
            "Place the dirty mark at a DIFFERENT POSITION than in the reference images. "
            "Keep defect SIZE similar to the references. "
            "Do NOT generate a defect-free image. "
            "Do NOT add any colored boxes, borders, or annotation marks. "
        ),
        "prompt_variations": [
            "Place the dirty smudge mark in the upper-left area of the console panel.",
            "Place the contamination spot slightly left of center on the console panel.",
            "Place the dirty mark in the lower-right area of the console surface.",
            "Place the smudge near the top-center of the console panel.",
            "Place the dirty contamination spot in the upper-right area of the console.",
            "Place the smudge near the lower-center of the console panel.",
            "Place the dirty spot along the left-side of the console surface.",
            "Place the contamination smear in the lower-left area of the console panel.",
            "Place the dirty mark slightly right of center on the console surface.",
            "Place the smudge or grease mark in the middle-right area of the console.",
        ],
        "prompt_style": (
            "Industrial inspection photography with slightly varied lighting. "
            "Maintain exact same console shape, layout, color as the FIRST (normal) image. "
            "Only add one realistic dirty/contamination mark — look of a genuine fingerprint, grease smear, or dust smudge. "
            "Output must be a clean, realistic photo with no overlaid graphics or annotation marks."
        ),
    },

    "Console_Gap": {
        "total_images": 75,
        "ref_dir": "Console/Console_Gap",
        "out_dir": "Console/Console_Gap",
        "data_ref": {
            "data_dir": f"{DATA_ROOT}/Console/train",
            "annotation": f"{DATA_ROOT}/Console/train/_annotations.coco.json",
            "cat_id": 2,
            "n_samples": 9,
        },
        "description": "Console gap defect - visible gap/separation between console components",
        "prompt_base": (
            "Generate a new image of an electronic console/control panel part with a gap defect. "
            "The FIRST image is a NORMAL console — use it as the base appearance reference. "
            "The REMAINING images are DEFECTIVE consoles showing real gap defects: "
            "visible separations or openings between two console components or panel sections "
            "that should be tightly fitted together. "
            "Study these examples to understand the look of the gap defect. "
        ),
        "prompt_key_instruction": (
            "MANDATORY: Output MUST contain exactly one gap defect — "
            "a visible separation or opening between two console parts/sections that should be touching. "
            "The gap should look physically real: a dark opening or misalignment between panel edges. "
            "Place the gap at a DIFFERENT LOCATION than in the reference images "
            "(still at a joint, seam, or edge between components). "
            "Keep gap SIZE similar to the references. "
            "Do NOT generate a defect-free image. "
            "Do NOT add any colored boxes, borders, or annotation marks. "
        ),
        "prompt_variations": [
            "Show the gap at a joint in the upper-left area of the console.",
            "Show the gap/separation at a seam slightly left of center on the console.",
            "Show the gap at a component edge in the lower-right area of the console.",
            "Show the gap near the top-center seam of the console panel.",
            "Show the gap at a panel joint in the upper-right area of the console.",
            "Show the gap near a lower-center edge of the console panel.",
            "Show the gap at a seam on the left-side of the console.",
            "Show the gap at a panel joint in the lower-left area of the console.",
            "Show the gap at a component seam slightly right of center on the console.",
            "Show the gap at a panel edge in the middle-right area of the console.",
        ],
        "prompt_style": (
            "Industrial inspection photography with slightly varied lighting. "
            "Maintain exact same console shape, layout, color as the FIRST (normal) image. "
            "Only introduce one realistic gap/separation between existing component edges. "
            "The gap should cast a natural shadow and look physically real. "
            "Output must be a clean, realistic photo with no overlaid graphics or annotation marks."
        ),
    },

    "Console_Scratch": {
        "total_images": 75,
        "ref_dir": "Console/Console_Scratch",
        "out_dir": "Console/Console_Scratch",
        "data_ref": {
            "data_dir": f"{DATA_ROOT}/Console/train",
            "annotation": f"{DATA_ROOT}/Console/train/_annotations.coco.json",
            "cat_id": 3,
            "n_samples": 9,
        },
        "description": "Console scratch defect - linear scratch mark on the console surface",
        "prompt_base": (
            "Generate a new image of an electronic console/control panel part with a scratch defect. "
            "The FIRST image is a NORMAL console — use it as the base appearance reference. "
            "The REMAINING images are DEFECTIVE consoles showing real scratch defects: "
            "linear marks scratched into the panel surface, leaving a visible groove. "
            "Study these examples to understand the look, length, and texture of the scratch. "
        ),
        "prompt_key_instruction": (
            "MANDATORY: Output MUST contain exactly one scratch defect — "
            "a realistic linear scratch mark on the console panel surface. "
            "The scratch should look genuine: a thin, elongated groove with slightly lighter coloring. "
            "The scratch can be straight or slightly curved, at any angle. "
            "Place the scratch at a DIFFERENT POSITION/ANGLE than in the reference images. "
            "Keep scratch SIZE and LENGTH similar to the references. "
            "Do NOT generate a defect-free image. "
            "Do NOT add any colored boxes, borders, or annotation marks. "
        ),
        "prompt_variations": [
            "Place the scratch diagonally across the upper-left area of the console surface.",
            "Place the scratch running horizontally slightly left of center on the console panel.",
            "Place the scratch in the lower-right area at a diagonal angle.",
            "Place the scratch running vertically near the top-center of the console panel.",
            "Place the scratch diagonally across the upper-right area of the console surface.",
            "Place the scratch running horizontally near the lower-center of the console panel.",
            "Place the scratch vertically along the left-side of the console surface.",
            "Place the scratch diagonally in the lower-left area of the console panel.",
            "Place the scratch at a slight angle slightly right of center on the console.",
            "Place the scratch horizontally in the middle-right area of the console surface.",
        ],
        "prompt_style": (
            "Industrial inspection photography with slightly varied lighting. "
            "Maintain exact same console shape, layout, color as the FIRST (normal) image. "
            "Only add one realistic linear scratch — natural depth cues, slight shadow along one edge. "
            "Output must be a clean, realistic photo with no overlaid graphics or annotation marks."
        ),
    },

    # ── Cylinder (bbox 없는 원본 이미지 방식) ─────────────────────────
    "Cylinder_Chip": {
        "total_images": 75,
        "ref_dir": "Cylinder/Cylinder_Chip",
        "out_dir": "Cylinder/Cylinder_Chip",
        "data_ref": {
            "data_dir": f"{DATA_ROOT}/Cylinder/train",
            "annotation": f"{DATA_ROOT}/Cylinder/train/_annotations.coco.json",
            "cat_id": 0,
            "n_samples": 9,
        },
        "description": "Cylinder chip defect - chip or scratch at the BOTTOM edge/rim of the cylinder",
        "prompt_base": (
            "Generate a new image of a precision-machined cylinder part with a chip defect. "
            "The FIRST image is a NORMAL cylinder — use it as the base appearance reference. "
            "The REMAINING images are DEFECTIVE cylinders showing real chip defects: "
            "small chips, nicks, or scratch marks specifically at or near the BOTTOM RIM/EDGE of the cylinder. "
            "Study these examples to understand the location, size, and appearance of the chip. "
        ),
        "prompt_key_instruction": (
            "MANDATORY: Output MUST contain exactly one chip defect — "
            "a small chip, nick, or scratch mark at the BOTTOM EDGE of the cylinder. "
            "CRITICAL PLACEMENT: The chip MUST be at or very near the bottom rim/edge — not in the middle or top. "
            "The chip should look like material has broken away or been scratched from the lower rim. "
            "Vary its exact position along the bottom rim relative to the references. "
            "Keep defect SIZE similar to the references. "
            "Do NOT generate a defect-free image. "
            "Do NOT add any colored boxes, borders, or annotation marks. "
        ),
        "prompt_variations": [
            "Place the chip at the bottom-left portion of the cylinder's lower rim.",
            "Place the chip at the bottom-front area of the cylinder's lower edge.",
            "Place the chip on the right side of the cylinder's bottom rim.",
            "Place the chip at the bottom-right of the cylinder's lower edge.",
            "Place the chip at the front-center of the cylinder's bottom rim.",
            "Place the chip slightly left of center on the cylinder's lower edge.",
            "Place the chip on the far left of the cylinder's bottom rim.",
            "Place the chip near the bottom-back area of the cylinder's lower edge.",
            "Place the chip slightly right of center along the cylinder's bottom rim.",
            "Place the chip at the back-left of the cylinder's lower edge.",
        ],
        "prompt_style": (
            "Industrial inspection photography with slightly varied lighting. "
            "Maintain exact same cylinder shape, material finish, color as the FIRST (normal) image. "
            "Only add a chip at the bottom edge — looks like a small broken or scratched area on the lower rim. "
            "Output must be a clean, realistic photo with no overlaid graphics or annotation marks."
        ),
    },

    "Cylinder_PistonMiss": {
        "total_images": 75,
        "ref_dir": "Cylinder/Cylinder_PistonMiss",
        "out_dir": "Cylinder/Cylinder_PistonMiss",
        "data_ref": {
            "data_dir": f"{DATA_ROOT}/Cylinder/train",
            "annotation": f"{DATA_ROOT}/Cylinder/train/_annotations.coco.json",
            "cat_id": 1,
            "n_samples": 9,
        },
        "description": "Cylinder piston-miss defect - missing step/groove between stages; appears as single layer",
        "prompt_base": (
            "Generate a new image of a precision-machined cylinder part with a piston-miss defect. "
            "The FIRST image is a NORMAL cylinder — use it as the base appearance reference, "
            "which should show clearly distinct stepped/layered sections. "
            "The REMAINING images are DEFECTIVE cylinders showing real piston-miss defects: "
            "the clear step or groove that should separate cylinder stages is MISSING or poorly defined, "
            "making distinct stages appear merged into one undifferentiated surface. "
            "Study these examples carefully to understand where and how the step disappears. "
        ),
        "prompt_key_instruction": (
            "MANDATORY: Output MUST show a piston-miss defect — "
            "the boundary/groove/step separating cylinder stages is missing or blurred at one location. "
            "Where there should be a clear step between sections, it appears merged or absent. "
            "Vary which section boundary is affected relative to the references. "
            "Do NOT generate a defect-free cylinder. "
            "Do NOT add any colored boxes, borders, or annotation marks. "
        ),
        "prompt_variations": [
            "Show the missing step between the upper and middle sections of the cylinder.",
            "Show the missing groove between the top-left stage and the section below.",
            "Show the merged/absent groove between the middle and lower sections.",
            "Show the missing step definition near the upper portion of the cylinder.",
            "Show the blurred step boundary in the upper-right area of the cylinder.",
            "Show the missing groove between two stages in the central region.",
            "Show the absent step boundary between sections on the left side.",
            "Show the merged cylinder stages in the lower-upper transition area.",
            "Show the missing step on the right side of the cylinder.",
            "Show the blurred stage boundary near the mid-height of the cylinder.",
        ],
        "prompt_style": (
            "Industrial inspection photography with slightly varied lighting. "
            "Maintain exact same cylinder shape, material, color as the FIRST (normal) image. "
            "Only introduce the missing-step defect at a different location than in references. "
            "Output must be a clean, realistic photo with no overlaid graphics or annotation marks."
        ),
    },

    "Cylinder_Porosity": {
        "total_images": 75,
        "ref_dir": "Cylinder/Cylinder_Porosity",
        "out_dir": "Cylinder/Cylinder_Porosity",
        "data_ref": {
            "data_dir": f"{DATA_ROOT}/Cylinder/train",
            "annotation": f"{DATA_ROOT}/Cylinder/train/_annotations.coco.json",
            "cat_id": 2,
            "n_samples": 9,
        },
        "description": "Cylinder porosity defect - small pitting or surface peeling/flaking marks",
        "prompt_base": (
            "Generate a new image of a precision-machined cylinder part with a porosity defect. "
            "The FIRST image is a NORMAL cylinder — use it as the base appearance reference. "
            "The REMAINING images are DEFECTIVE cylinders showing real porosity defects: "
            "small pits, craters, or surface peeling/flaking where material has been damaged or removed. "
            "Study these examples to understand the size, texture, and look of the porosity defect. "
        ),
        "prompt_key_instruction": (
            "MANDATORY: Output MUST contain a porosity defect — "
            "small pits, craters, or peeled/flaked areas on the cylinder surface. "
            "The defect should look like small craters or patches where surface material has pitted or peeled. "
            "Place the defect at a DIFFERENT POSITION than in the references. "
            "Keep defect SIZE similar to the references. "
            "Do NOT generate a defect-free image. "
            "Do NOT add any colored boxes, borders, or annotation marks. "
        ),
        "prompt_variations": [
            "Place the porosity pitting in the upper-left area of the cylinder body.",
            "Place the surface pitting/peeling slightly left of center on the cylinder.",
            "Place the porosity defect in the lower-right area of the cylinder surface.",
            "Place the pitting/flaking near the top-center of the cylinder body.",
            "Place the porosity defect in the upper-right area of the cylinder.",
            "Place the surface pitting near the lower-center of the cylinder body.",
            "Place the porosity/peeling defect on the left-side of the cylinder.",
            "Place the pitting marks in the lower-left area of the cylinder body.",
            "Place the porosity defect slightly right of center on the cylinder.",
            "Place the surface peeling/pitting in the middle-right area of the cylinder.",
        ],
        "prompt_style": (
            "Industrial inspection photography with slightly varied lighting. "
            "Maintain exact same cylinder shape, material, color as the FIRST (normal) image. "
            "Only add porosity/pitting defect — small craters or peeled areas with natural texture. "
            "Output must be a clean, realistic photo with no overlaid graphics or annotation marks."
        ),
    },

    "Cylinder_RCS": {
        "total_images": 75,
        "ref_dir": "Cylinder/Cylinder_RCS",
        "out_dir": "Cylinder/Cylinder_RCS",
        "data_ref": {
            "data_dir": f"{DATA_ROOT}/Cylinder/train",
            "annotation": f"{DATA_ROOT}/Cylinder/train/_annotations.coco.json",
            "cat_id": 3,
            "n_samples": 9,
        },
        "description": "Cylinder RCS defect - multiple parallel scratch marks running simultaneously",
        "prompt_base": (
            "Generate a new image of a precision-machined cylinder part with an RCS defect. "
            "The FIRST image is a NORMAL cylinder — use it as the base appearance reference. "
            "The REMAINING images are DEFECTIVE cylinders showing real RCS defects: "
            "multiple parallel linear scratches occurring simultaneously on the cylinder surface, "
            "like marks from a multi-point contact dragging across the surface at once. "
            "Study these examples to understand the pattern, spacing, and size of these scratches. "
        ),
        "prompt_key_instruction": (
            "MANDATORY: Output MUST contain an RCS defect — "
            "multiple parallel scratches (2–4 lines) running in the same direction simultaneously. "
            "The scratches should be closely spaced parallel grooves, as if made by multiple contact points at once. "
            "Place them at a DIFFERENT POSITION/ANGLE than in the references. "
            "Keep the scratch group SIZE similar to the references. "
            "Do NOT generate a defect-free image. "
            "Do NOT add any colored boxes, borders, or annotation marks. "
        ),
        "prompt_variations": [
            "Place the multiple parallel scratches running vertically in the upper-left area of the cylinder.",
            "Place the concurrent scratch marks running horizontally slightly left of center.",
            "Place the parallel scratches diagonally in the lower-right area of the cylinder.",
            "Place the concurrent scratches running vertically near the top-center of the cylinder.",
            "Place the parallel scratches diagonally in the upper-right area of the cylinder.",
            "Place the concurrent scratches running horizontally near the lower-center.",
            "Place the parallel scratches vertically on the left-side of the cylinder.",
            "Place the concurrent scratches diagonally in the lower-left area.",
            "Place the parallel scratches slightly right of center on the cylinder.",
            "Place the concurrent scratches running vertically in the middle-right area.",
        ],
        "prompt_style": (
            "Industrial inspection photography with slightly varied lighting. "
            "Maintain exact same cylinder shape, material, color as the FIRST (normal) image. "
            "Only add the multiple parallel scratches — thin parallel grooves with natural depth and shadow. "
            "Output must be a clean, realistic photo with no overlaid graphics or annotation marks."
        ),
    },

    # ── Wood (bbox 없는 원본 이미지 방식) ────────────────────────────
    "Wood_impurities": {
        "total_images": 150,
        "ref_dir": "Wood/Wood_impurities",
        "out_dir": "Wood/Wood_impurities",
        "data_ref": {
            "data_dir": f"{DATA_ROOT}/Wood/train",
            "annotation": f"{DATA_ROOT}/Wood/train/_annotations.coco.json",
            "cat_id": 0,
            "n_samples": 9,
        },
        "description": "Wood impurities defect - white/bright marks on wood surface, like scorched or burned spots",
        "prompt_base": (
            "Generate a new image of a wood material surface with an impurities defect. "
            "The FIRST image is a NORMAL wood surface — use it as the base appearance reference. "
            "The REMAINING images are DEFECTIVE wood surfaces showing real impurity defects: "
            "WHITE or BRIGHT-colored spots/patches on the wood — marks like scorched/burned areas, "
            "mineral deposits, or white discoloration contrasting against the natural wood grain. "
            "Study these examples to understand the color, size, and texture of the impurity mark. "
        ),
        "prompt_key_instruction": (
            "MANDATORY: Output MUST contain an impurities defect — "
            "a white or bright-colored mark on the wood surface that contrasts with the natural wood. "
            "The mark looks like: a scorched/burned white spot, mineral deposit, bleached area, "
            "or white powder/ash-like contamination on the wood. "
            "KEY CHARACTERISTIC: WHITE or LIGHT-COLORED appearance against the wood background. "
            "Place the mark at a DIFFERENT POSITION than in the references. "
            "Keep SIZE similar to the references. "
            "Do NOT generate a defect-free image. "
            "Do NOT add any colored boxes, borders, or annotation marks. "
        ),
        "prompt_variations": [
            "Place the white impurity mark in the upper-left area of the wood surface.",
            "Place the bright/white spot slightly left of center on the wood.",
            "Place the impurity mark in the lower-right area of the wood surface.",
            "Place the white defect mark near the top-center of the wood surface.",
            "Place the bright impurity spot in the upper-right area of the wood.",
            "Place the white/scorched mark near the lower-center of the wood.",
            "Place the impurity mark (white) along the left-side of the wood surface.",
            "Place the bright white spot in the lower-left area of the wood surface.",
            "Place the white impurity mark slightly right of center on the wood.",
            "Place the bright/scorched mark in the middle-right area of the wood.",
        ],
        "prompt_style": (
            "Industrial inspection photography with slightly varied lighting. "
            "Maintain exact same wood grain pattern, color, texture as the FIRST (normal) image. "
            "Only add the white/bright impurity mark — naturally embedded in the wood surface "
            "like a genuine scorched spot or mineral deposit. "
            "Output must be a clean, realistic photo with no overlaid graphics or annotation marks."
        ),
    },

    "Wood_pits": {
        "total_images": 150,
        "ref_dir": "Wood/Wood_pits",
        "out_dir": "Wood/Wood_pits",
        "data_ref": {
            "data_dir": f"{DATA_ROOT}/Wood/train",
            "annotation": f"{DATA_ROOT}/Wood/train/_annotations.coco.json",
            "cat_id": 1,
            "n_samples": 9,
        },
        "description": "Wood pits defect - scratch marks or pit indentations on wood surface",
        "prompt_base": (
            "Generate a new image of a wood material surface with a pits/scratch defect. "
            "The FIRST image is a NORMAL wood surface — use it as the base appearance reference. "
            "The REMAINING images are DEFECTIVE wood surfaces showing real pits defects: "
            "scratch marks, linear grooves, or pit indentations on the wood surface where the "
            "material has been scratched or gouged, disrupting the smooth wood grain. "
            "Study these examples to understand the look, length, and depth of the scratch/pit. "
        ),
        "prompt_key_instruction": (
            "MANDATORY: Output MUST contain a pits/scratch defect — "
            "visible scratch marks or pit indentations on the wood surface. "
            "The scratch should look genuine: a linear groove cutting across the grain "
            "with raw wood exposed at the scratch path. Can be single prominent scratch or cluster of pits. "
            "Place the defect at a DIFFERENT POSITION than in the references. "
            "Keep SIZE and LENGTH similar to the references. "
            "Do NOT generate a defect-free image. "
            "Do NOT add any colored boxes, borders, or annotation marks. "
        ),
        "prompt_variations": [
            "Place the scratch diagonally across the upper-left area of the wood surface.",
            "Place the scratch groove running horizontally slightly left of center on the wood.",
            "Place the pit/scratch marks in the lower-right area of the wood surface.",
            "Place the scratch running vertically near the top-center of the wood.",
            "Place the scratch groove diagonally across the upper-right area of the wood.",
            "Place the pit marks near the lower-center of the wood surface.",
            "Place the scratch running along the left-side of the wood surface.",
            "Place the scratch groove diagonally in the lower-left area of the wood.",
            "Place the pit/scratch marks slightly right of center on the wood surface.",
            "Place the scratch horizontally across the middle-right area of the wood.",
        ],
        "prompt_style": (
            "Industrial inspection photography with slightly varied lighting. "
            "Maintain exact same wood grain pattern, color, texture as the FIRST (normal) image. "
            "Only add the scratch/pit marks — a groove with natural shadow and exposed raw wood. "
            "Output must be a clean, realistic photo with no overlaid graphics or annotation marks."
        ),
    },
}

# ===== 클래스 그룹 =====
CLASS_GROUPS = {
    "Console":  ["Console_Collision", "Console_Dirty", "Console_Gap", "Console_Scratch"],
    "Cylinder": ["Cylinder_Chip", "Cylinder_PistonMiss", "Cylinder_Porosity", "Cylinder_RCS"],
    "Wood":     ["Wood_impurities", "Wood_pits"],
}


# ===== 진행상황 관리 =====
def get_progress_file(defect_type):
    return f"progress_{defect_type}.json"


def load_progress(defect_type):
    pf = get_progress_file(defect_type)
    if Path(pf).exists():
        with open(pf) as f:
            return json.load(f)
    return {'completed': [], 'failed': [], 'last_successful_index': -1, 'start_time': None}


def save_progress(progress, defect_type):
    with open(get_progress_file(defect_type), 'w') as f:
        json.dump(progress, f, indent=2)


# ===== 이미지 로드 =====
def load_reference_images(defect_type, config):
    """
    레퍼런스 이미지 로드.
    - data_ref 있음: normal_00은 ref_dir에서, defect 샘플은 원본 data에서 (bbox 없음)
    - data_ref 없음: ref_dir의 모든 이미지 사용 (기존 방식, 첫 번째=정상)
    """
    if "data_ref" in config:
        return _load_from_data(defect_type, config)
    else:
        return _load_from_ref_dir(defect_type, config)


def _load_from_ref_dir(defect_type, config):
    """기존 방식: ref_dir의 모든 이미지 사용"""
    if "ref_dir" in config:
        ref_dir = Path(BASE_REFERENCE_DIR) / config["ref_dir"]
    else:
        ref_dir = Path(BASE_REFERENCE_DIR) / defect_type

    exts = ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG"]
    imgs = sorted(sum([list(ref_dir.glob(e)) for e in exts], []))

    if len(imgs) < 2:
        print(f"ERROR: Need ≥2 images in '{ref_dir}/'", flush=True)
        exit(1)

    result = []
    for idx, p in enumerate(imgs):
        img = Image.open(p)
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        role = "NORMAL (no defect)" if idx == 0 else f"DEFECT sample #{idx}"
        result.append({'name': p.name, 'bytes': buf.getvalue(),
                       'role': role, 'is_normal': idx == 0})
        print(f"  Loaded: {p.name} [{role}]", flush=True)
    return result


def _load_from_data(defect_type, config):
    """
    새 방식: normal_00.png는 ref_dir에서, defect 샘플은 원본 data에서 (bbox 없음)
    """
    ref_dir  = Path(BASE_REFERENCE_DIR) / config["ref_dir"]
    dr       = config["data_ref"]
    data_dir = Path(dr["data_dir"])
    ann_path = Path(dr["annotation"])
    cat_id   = dr["cat_id"]
    n_samp   = dr.get("n_samples", 9)

    # 1) normal_00 로드
    normal_candidates = sorted(
        list(ref_dir.glob("normal_00.*")) +
        list(ref_dir.glob("Normal_00.*"))
    )
    if not normal_candidates:
        # fallback: 이름순 첫 번째 파일
        all_files = sorted(p for p in ref_dir.iterdir() if p.is_file())
        normal_candidates = all_files[:1]
    if not normal_candidates:
        print(f"ERROR: normal_00 not found in '{ref_dir}'", flush=True)
        exit(1)

    result = []
    p = normal_candidates[0]
    img = Image.open(p)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    result.append({'name': p.name, 'bytes': buf.getvalue(),
                   'role': 'NORMAL (no defect)', 'is_normal': True})
    print(f"  Loaded: {p.name} [NORMAL (no defect)]", flush=True)

    # 2) COCO annotation에서 해당 cat_id 이미지 찾기
    with open(ann_path) as f:
        coco = json.load(f)
    id2name = {img_info["id"]: img_info["file_name"] for img_info in coco["images"]}
    cat_img_ids = set()
    for ann in coco["annotations"]:
        if ann["category_id"] == cat_id:
            cat_img_ids.add(ann["image_id"])

    selected_ids = sorted(cat_img_ids)[:n_samp]

    for idx, img_id in enumerate(selected_ids, start=1):
        fname = id2name[img_id]
        img_path = data_dir / fname
        img = Image.open(img_path).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        role = f"DEFECT sample #{idx} (no bbox)"
        result.append({'name': fname, 'bytes': buf.getvalue(),
                       'role': role, 'is_normal': False})
        print(f"  Loaded: {fname} [{role}]", flush=True)

    print(f"  → {len(result)-1} defect samples from original data (no bbox)", flush=True)
    return result


# ===== 프롬프트 생성 =====
def generate_prompt(config, index):
    variation = config["prompt_variations"][index % len(config["prompt_variations"])]
    return config["prompt_base"] + config["prompt_key_instruction"] + variation + " " + config["prompt_style"]


# ===== 단일 결함 유형 생성 =====
def run_generation(defect_type, count_override=None):
    if defect_type not in DEFECT_CONFIGS:
        print(f"ERROR: Unknown defect type '{defect_type}'", flush=True)
        exit(1)

    config = DEFECT_CONFIGS[defect_type]
    total_images = count_override if count_override else config["total_images"]

    if "out_dir" in config:
        output_dir = Path(BASE_OUTPUT_DIR) / config["out_dir"]
    else:
        output_dir = Path(BASE_OUTPUT_DIR) / defect_type
    output_dir.mkdir(parents=True, exist_ok=True)

    progress = load_progress(defect_type)
    if progress['start_time'] is None:
        progress['start_time'] = time.time()
        save_progress(progress, defect_type)

    start_time  = progress['start_time']
    start_index = progress['last_successful_index'] + 1

    print("=" * 80, flush=True)
    print(f"GEMINI API - {defect_type} ({total_images} Images)", flush=True)
    print("=" * 80, flush=True)
    print(f"Description: {config['description']}", flush=True)
    print(f"Started at:  {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"Output:      {output_dir}/", flush=True)
    print(f"Progress:    {len(progress['completed'])}/{total_images} completed", flush=True)
    print(f"Delay:       {DELAY_BETWEEN_IMAGES}s between images", flush=True)

    remaining    = total_images - len(progress['completed'])
    eta_seconds  = remaining * DELAY_BETWEEN_IMAGES
    eta_time     = datetime.now() + timedelta(seconds=eta_seconds)
    print(f"ETA:         {eta_time.strftime('%Y-%m-%d %H:%M:%S')} (~{int(eta_seconds/3600)}h {int((eta_seconds%3600)/60)}m)", flush=True)
    print("=" * 80, flush=True)

    if not GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY not set. Run: export GEMINI_API_KEY='your_key'", flush=True)
        exit(1)
    client = genai.Client(api_key=GEMINI_API_KEY)

    print(f"\n[1/2] Loading reference images...", flush=True)
    ref_data_list = load_reference_images(defect_type, config)
    print(f"  Total: 1 normal + {len(ref_data_list)-1} defect samples", flush=True)

    print(f"\n[2/2] Starting generation...", flush=True)
    print("=" * 80, flush=True)

    for i in range(start_index, total_images):
        if i in progress['completed']:
            continue

        retry_count = 0

        while retry_count < MAX_RETRIES:
            try:
                completed_count = len(progress['completed'])
                progress_pct    = (completed_count / total_images) * 100

                if completed_count > 0:
                    elapsed        = time.time() - start_time
                    avg_time       = elapsed / completed_count
                    remaining_imgs = total_images - completed_count
                    remaining_secs = avg_time * remaining_imgs
                    eta = datetime.now() + timedelta(seconds=remaining_secs)
                    rh  = int(remaining_secs / 3600)
                    rm  = int((remaining_secs % 3600) / 60)
                else:
                    remaining_secs = total_images * DELAY_BETWEEN_IMAGES
                    eta = datetime.now() + timedelta(seconds=remaining_secs)
                    rh  = int(remaining_secs / 3600)
                    rm  = int((remaining_secs % 3600) / 60)

                print(f"\n[{i+1:3d}/{total_images}] {defect_type} | {progress_pct:.1f}% | {datetime.now().strftime('%m/%d %H:%M:%S')}", flush=True)
                print(f"ETA: {eta.strftime('%m/%d %H:%M:%S')} (~{rh}h {rm}m left)", flush=True)

                prompt   = generate_prompt(config, i)
                contents = []

                # 정상 이미지 (항상 포함)
                normal_ref = ref_data_list[0]
                contents.append(types.Part.from_bytes(data=normal_ref['bytes'], mime_type='image/png'))

                # 결함 이미지 - 최대 4장 순환 선택
                defect_refs    = ref_data_list[1:]
                n_defect       = len(defect_refs)
                MAX_DEFECT_REFS = min(4, n_defect)
                start_ref      = i % n_defect
                selected_indices = [(start_ref + k) % n_defect for k in range(MAX_DEFECT_REFS)]

                for ref_idx in selected_indices:
                    ref = defect_refs[ref_idx]
                    contents.append(types.Part.from_bytes(data=ref['bytes'], mime_type='image/png'))

                contents.append(prompt)
                print(f"  Refs: normal + {len(selected_indices)} defect (indices {selected_indices})", flush=True)

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
                            save_path = output_dir / f"{defect_type}_{i:03d}.png"
                            with open(save_path, 'wb') as f:
                                f.write(part.inline_data.data)
                            progress['completed'].append(i)
                            progress['last_successful_index'] = i
                            save_progress(progress, defect_type)
                            saved = True
                            print(f"  SAVED: {save_path.name} | {len(progress['completed'])}/{total_images}", flush=True)
                            break

                if saved:
                    break
                else:
                    retry_count += 1
                    print(f"  No image in response (attempt {retry_count}/{MAX_RETRIES})", flush=True)
                    time.sleep(10)

            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                    retry_count += 1
                    print(f"  RATE LIMIT! Waiting {RATE_LIMIT_BACKOFF}s...", flush=True)
                    save_progress(progress, defect_type)
                    time.sleep(RATE_LIMIT_BACKOFF)
                elif "quota" in error_msg.lower() or "daily" in error_msg.lower():
                    print(f"  QUOTA EXCEEDED! {error_msg}", flush=True)
                    save_progress(progress, defect_type)
                    exit(1)
                else:
                    retry_count += 1
                    print(f"  ERROR: {error_msg[:200]}", flush=True)
                    time.sleep(30)

        if retry_count >= MAX_RETRIES:
            progress['failed'].append(i)
            save_progress(progress, defect_type)
            print(f"  FAILED after {MAX_RETRIES} retries. Skipping.", flush=True)

        if i < total_images - 1:
            next_t = datetime.now() + timedelta(seconds=DELAY_BETWEEN_IMAGES)
            print(f"  Waiting {DELAY_BETWEEN_IMAGES}s → {next_t.strftime('%H:%M:%S')}", flush=True)
            time.sleep(DELAY_BETWEEN_IMAGES)

    total_time = time.time() - start_time
    print(f"\nCOMPLETE: {defect_type} | {len(progress['completed'])}/{total_images} success | "
          f"{int(total_time/3600)}h {int((total_time%3600)/60)}m", flush=True)
    return len(progress['completed']), len(progress['failed'])


# ===== 메인 =====
if __name__ == "__main__":
    all_choices = list(DEFECT_CONFIGS.keys()) + list(CLASS_GROUPS.keys()) + ["all"]
    parser = argparse.ArgumentParser(description="Gemini Defect Image Augmentation")
    parser.add_argument("defect_type", choices=all_choices,
                        help="Defect type, class group (Console/Cylinder/Wood), or 'all'")
    parser.add_argument("--count", type=int, default=None,
                        help="Override image count (e.g. --count 5 for test)")
    args = parser.parse_args()

    if args.defect_type in CLASS_GROUPS:
        results = {}
        sub_defects = CLASS_GROUPS[args.defect_type]
        print(f"\n### Class: {args.defect_type} → {sub_defects}\n", flush=True)
        for dtype in sub_defects:
            s, f = run_generation(dtype, count_override=args.count)
            results[dtype] = (s, f)
        print(f"\n=== {args.defect_type} DONE ===", flush=True)
        total_s = sum(v[0] for v in results.values())
        total_f = sum(v[1] for v in results.values())
        for dtype, (s, f) in results.items():
            tgt = args.count or DEFECT_CONFIGS[dtype]["total_images"]
            print(f"  {dtype}: {s}/{tgt} success, {f} failed", flush=True)
        print(f"  TOTAL: {total_s} success, {total_f} failed", flush=True)

    elif args.defect_type == "all":
        run_order = (
            ["casting_Inclusoes", "casting_Rechupe", "screw_defect"] +
            CLASS_GROUPS["Console"] + CLASS_GROUPS["Cylinder"] + CLASS_GROUPS["Wood"]
        )
        for dtype in run_order:
            run_generation(dtype, count_override=args.count)

    else:
        run_generation(args.defect_type, count_override=args.count)
