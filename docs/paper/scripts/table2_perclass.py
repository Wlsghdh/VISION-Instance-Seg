"""Table 2 — Per-class breakdown (cond1~cond4_6x × 2 모델)."""
from pathlib import Path
import _loader

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / 'tables'
OUT.mkdir(parents=True, exist_ok=True)

CONDS = ['cond1', 'cond2', 'cond3', 'cond4_4x', 'cond4_5x', 'cond4_6x']
MODELS = ['mask_rcnn', 'cascade_mask_rcnn']
CLASSES = ['Dirty', 'Inclusoes', 'impurities']


def main():
    rows = _loader.flat(_loader.load_all())
    lines_md = ['## Table 2. Per-class AP breakdown\n']

    for m in MODELS:
        lines_md.append(f'### {_model_name(m)}\n')
        header = ['| 조건 |'] + [f' {c} |' for c in CLASSES] + [' mean |']
        lines_md.append(''.join(header))
        lines_md.append('|------|' + ':---:|' * (len(CLASSES) + 1))
        for cond in CONDS:
            r = _loader.pick(rows, exp='exp2', cond=cond, model=m)
            if not r:
                continue
            vals = [r.get(f'segm_AP-{c}') for c in CLASSES]
            mean = r['segm_AP']
            row = f"| {cond} | " + ' | '.join(f'{v:.2f}' if v is not None else '—' for v in vals) + f" | **{mean:.2f}** |"
            lines_md.append(row)
        lines_md.append('')
    (OUT / 'table2_perclass.md').write_text('\n'.join(lines_md) + '\n')
    print('✅ saved: tables/table2_perclass.md')
    print('\n'.join(lines_md))


def _model_name(m):
    return {'mask_rcnn': 'Mask R-CNN', 'cascade_mask_rcnn': 'Cascade Mask R-CNN'}[m]


if __name__ == '__main__':
    main()
