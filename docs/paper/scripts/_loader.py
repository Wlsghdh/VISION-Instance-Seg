"""results/evaluation/results.json 통합 로더."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
AGG = ROOT / 'results' / 'evaluation' / 'results.json'


def load_all():
    if not AGG.exists():
        raise FileNotFoundError(AGG)
    return json.loads(AGG.read_text())


def flat(records, category='Exp2_3cls', require_segm=True):
    """flat dict list로 변환. 각 원소: exp, cond, model, seed, segm_AP, ..."""
    out = []
    for r in records:
        if r.get('category') != category:
            continue
        ev = r.get('eval') or {}
        if require_segm and ev.get('segm_AP') is None:
            continue
        row = {
            'exp': r.get('experiment'),
            'cond': r.get('condition'),
            'model': r.get('model'),
            'seed': r.get('seed'),
        }
        for k in ('segm_AP', 'segm_AP50', 'segm_AP75',
                  'segm_AP-Dirty', 'segm_AP-Inclusoes', 'segm_AP-impurities',
                  'bbox_AP'):
            row[k] = ev.get(k)
        if 'smoke' in (row['model'] or ''):
            continue
        out.append(row)
    return out


def pick(rows, exp=None, cond=None, model=None, seed=42):
    for r in rows:
        if exp and r['exp'] != exp: continue
        if cond and r['cond'] != cond: continue
        if model and r['model'] != model: continue
        if seed and r.get('seed') != seed: continue
        return r
    return None
