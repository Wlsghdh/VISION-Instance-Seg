"""
results/training/<exp>/... 의 결과를 results_github/<exp>/... 로 복사 (git 공유).
기존 팀 관행(ldy)과 동일한 구조: results_github/<exp>/<cond>/<cat>/<model>/seed42/...

복사 대상:
  - eval_results/results.json  (필수)
  - config.yaml / config.py    (재현성)
  - metrics.json               (학습 로그, 기본 포함 — 논문 분석용)
  - events.out.tfevents.*      (TensorBoard, --include-tb 지정 시만)
  - inference/*                (제외, 용량 큼)
  - *.pth                      (제외, 용량 큼)

집계 JSON: results/evaluation/results.json → results_github/evaluation/results.json

사용법:
    python scripts/sync_results_to_github.py
    python scripts/sync_results_to_github.py --dry-run
    python scripts/sync_results_to_github.py --include-tb    # TensorBoard 포함
    python scripts/sync_results_to_github.py --exp exp1_3cls  # 특정 실험만
"""
import argparse
import shutil
from pathlib import Path

ROOT = Path('/home/jjh0709/gitrepo/VISION-Instance-Seg')
SRC = ROOT / 'results' / 'training'
DST = ROOT / 'results_github'
AGG_SRC = ROOT / 'results' / 'evaluation' / 'results.json'
AGG_DST = DST / 'evaluation' / 'results.json'

# 복사할 파일 패턴 (run 디렉토리 기준)
INCLUDE_PATTERNS = [
    'eval_results/results.json',
    'eval_results/*.json',
    'config.yaml',
    'config.py',
    'metrics.json',  # 기본 포함
]
TB_PATTERN = 'events.out.tfevents.*'
EXCLUDE_DIRS = ['inference']


def copy_if_newer(src: Path, dst: Path, dry: bool = False) -> bool:
    if not src.exists() or not src.is_file():
        return False
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return False
    if not dry:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    return True


def sync_run_dir(run_dir: Path, dst_root: Path, patterns, dry: bool, include_tb: bool):
    """run_dir = results/training/<exp>/<cond>/<cat>/<model>/seed42"""
    rel = run_dir.relative_to(SRC)  # exp/cond/cat/model/seedN
    dst_dir = dst_root / rel

    n_copy = 0
    # 명시 패턴
    all_patterns = list(patterns)
    if include_tb:
        all_patterns.append(TB_PATTERN)

    for pat in all_patterns:
        for src_file in run_dir.glob(pat):
            if any(d in src_file.parts for d in EXCLUDE_DIRS):
                continue
            rel_file = src_file.relative_to(run_dir)
            if copy_if_newer(src_file, dst_dir / rel_file, dry):
                n_copy += 1
    return n_copy


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--include-tb', action='store_true',
                   help='TensorBoard events.tfevents 파일도 포함 (용량 큼)')
    p.add_argument('--exp', type=str, default=None,
                   help='특정 실험만 (예: exp1_3cls)')
    args = p.parse_args()

    total = 0
    if not SRC.exists():
        print(f'❌ {SRC} 없음')
        return

    # 모든 run 디렉토리 순회 (seed로 끝나는 폴더)
    exp_filter = args.exp
    for seed_dir in SRC.glob('*/*/*/*/seed*'):
        if not seed_dir.is_dir():
            continue
        if exp_filter and seed_dir.parts[-5] != exp_filter:
            continue
        n = sync_run_dir(seed_dir, DST, INCLUDE_PATTERNS, args.dry_run, args.include_tb)
        if n > 0:
            rel = seed_dir.relative_to(SRC)
            print(f'  ✅ {rel}  ({n}파일)')
            total += n

    # 집계 JSON
    if copy_if_newer(AGG_SRC, AGG_DST, args.dry_run):
        print(f'  ✅ evaluation/results.json')
        total += 1

    print(f'\n총 복사/갱신: {total}파일')
    print(f'저장 위치: {DST}')
    if args.dry_run:
        print('(dry-run)')
    else:
        print('다음 단계: git add results_github/ && git commit && git push')


if __name__ == '__main__':
    main()
