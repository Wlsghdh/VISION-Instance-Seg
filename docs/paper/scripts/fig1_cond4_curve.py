"""Figure 1 — Condition 4 N 스윕 곡선 (GenAI 125 고정 + 전통 N×145 스윕)."""
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import _style
import _loader

HERE = Path(__file__).resolve().parent
FIGS = HERE.parent / 'figs'
FIGS.mkdir(parents=True, exist_ok=True)


def main(lang='en'):
    _style.apply(lang)
    rows = _loader.flat(_loader.load_all())
    # cond4_Nx 뽑기
    ns = list(range(1, 11))
    per_model = {'mask_rcnn': {}, 'cascade_mask_rcnn': {}}
    for n in ns:
        for m in per_model.keys():
            r = _loader.pick(rows, exp='exp2', cond=f'cond4_{n}x', model=m)
            if r:
                per_model[m][n] = r['segm_AP']

    # baseline reference (cond1)
    base = {m: _loader.pick(rows, exp='exp2', cond='cond1', model=m) for m in per_model}

    fig, ax = plt.subplots()
    for m, data in per_model.items():
        if not data:
            continue
        xs = sorted(data.keys())
        ys = [data[x] for x in xs]
        ax.plot(xs, ys, marker=_style.MARKERS[m], color=_style.COLORS[m],
                label=_style.DISPLAY[m])
        if base[m]:
            ax.axhline(base[m]['segm_AP'], color=_style.COLORS[m],
                       linestyle=':', alpha=0.5, linewidth=1.0)

    if lang == 'kr':
        ax.set_xlabel('전통 증강 배수 N (각 단위 145/class)')
        ax.set_ylabel('segm AP (IoU 0.50:0.95)')
        ax.set_title('GenAI 125 + 전통 N×145 스윕')
    else:
        ax.set_xlabel('Traditional Augmentation Multiplier N (×145/class)')
        ax.set_ylabel('segm AP (IoU 0.50:0.95)')
        ax.set_title('cond4: GenAI 125 + Traditional N×145')
    ax.set_xticks(range(1, 11))
    ax.legend(loc='best')

    out = FIGS / f'fig1_cond4_curve{"_kr" if lang=="kr" else ""}'
    _style.save(fig, str(out))
    print(f'✅ saved: {out}.pdf / .png')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--lang', choices=['en', 'kr', 'both'], default='en')
    args = p.parse_args()
    if args.lang == 'both':
        main('en'); main('kr')
    else:
        main(args.lang)
