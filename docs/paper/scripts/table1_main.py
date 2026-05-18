"""Table 1 — Main result (cond1/2/3 + cond4_6x). .tex + .md 동시 생성."""
from pathlib import Path
import _loader

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / 'tables'
OUT.mkdir(parents=True, exist_ok=True)

ROWS = [
    ('baseline', 'cond1', '원본 20/cls'),
    ('+전통 125', 'cond2', '원본 20 + 전통 125/cls'),
    ('+GenAI 125', 'cond3', '원본 20 + GenAI 125/cls'),
    ('+GenAI+전통×6', 'cond4_6x', '(원본 20 + GenAI 125) + 전통 870/cls'),
]


def main():
    rows = _loader.flat(_loader.load_all())

    # baseline 값 (cond1)
    base_m = _loader.pick(rows, exp='exp2', cond='cond1', model='mask_rcnn')
    base_c = _loader.pick(rows, exp='exp2', cond='cond1', model='cascade_mask_rcnn')

    lines_md = [
        '| 조건 | 구성 | Mask R-CNN ↑ | Cascade Mask R-CNN ↑ | Δ vs baseline |',
        '|------|------|:---:|:---:|:---:|',
    ]
    lines_tex = [
        r'\begin{tabular}{llcccc}',
        r'\toprule',
        r'조건 & 구성 & Mask R-CNN $\uparrow$ & Cascade $\uparrow$ & $\Delta$ vs baseline (M/C) \\',
        r'\midrule',
    ]
    for label, cond, cfg in ROWS:
        m = _loader.pick(rows, exp='exp2', cond=cond, model='mask_rcnn')
        c = _loader.pick(rows, exp='exp2', cond=cond, model='cascade_mask_rcnn')
        m_ap = m['segm_AP'] if m else None
        c_ap = c['segm_AP'] if c else None
        delta = '—' if cond == 'cond1' else \
            f"{m_ap - base_m['segm_AP']:+.2f} / {c_ap - base_c['segm_AP']:+.2f}" \
            if (m_ap and c_ap) else '—'
        lines_md.append(f"| **{label}** | {cfg} | "
                        f"{m_ap:.2f} | {c_ap:.2f} | {delta} |")
        lines_tex.append(f"{label} & {cfg} & {m_ap:.2f} & {c_ap:.2f} & {delta} \\\\")
    lines_tex.append(r'\bottomrule')
    lines_tex.append(r'\end{tabular}')

    (OUT / 'table1_main.md').write_text('\n'.join(lines_md) + '\n')
    (OUT / 'table1_main.tex').write_text('\n'.join(lines_tex) + '\n')
    print('✅ saved: tables/table1_main.md / .tex')
    print('\n'.join(lines_md))


if __name__ == '__main__':
    main()
