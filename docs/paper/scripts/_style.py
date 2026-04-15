"""Publication-quality matplotlib 공통 스타일."""
import matplotlib as mpl
import matplotlib.pyplot as plt
import subprocess

def has_nanum():
    try:
        out = subprocess.check_output(['fc-list'], text=True, stderr=subprocess.DEVNULL)
        return 'Nanum' in out
    except Exception:
        return False


def apply(lang='en'):
    font_family = 'NanumGothic' if (lang in ('kr', 'both') and has_nanum()) else 'DejaVu Sans'
    mpl.rcParams.update({
        'font.family': font_family,
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'legend.fontsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'figure.figsize': (5.5, 3.5),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'axes.unicode_minus': False,
    })


COLORS = {
    'mask_rcnn': '#1f77b4',           # blue
    'cascade_mask_rcnn': '#d62728',   # red
    'maskdino': '#2ca02c',            # green
    'mask2former': '#9467bd',         # purple
    'cascade_rcnn': '#ff7f0e',        # orange
    'solov2': '#8c564b',              # brown
    'rtmdet_ins': '#17becf',          # cyan
}
MARKERS = {
    'mask_rcnn': 'o',
    'cascade_mask_rcnn': 's',
    'maskdino': '^',
    'mask2former': 'D',
}
DISPLAY = {
    'mask_rcnn': 'Mask R-CNN',
    'cascade_mask_rcnn': 'Cascade Mask R-CNN',
    'maskdino': 'MaskDINO',
    'mask2former': 'Mask2Former',
    'cascade_rcnn': 'Cascade R-CNN',
    'solov2': 'SOLOv2',
    'rtmdet_ins': 'RTMDet-Ins',
}


def save(fig, path_base):
    """path_base='fig1' → fig1.pdf + fig1.png 동시 저장."""
    fig.savefig(f'{path_base}.pdf')
    fig.savefig(f'{path_base}.png')
