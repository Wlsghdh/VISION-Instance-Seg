"""
MaskDINO 실험 결과 분석 및 시각화
- Detection (bbox) + Segmentation (mask) 성능 모두 분석
- 13개 실험 결과 비교
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 한글 폰트 설정 (선택사항)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# Paths
BASE_DIR = Path('/data2/project/2026winter/jjh0709/AA_CV_R')
RESULTS_DIR = BASE_DIR / 'results' / 'maskdino'
PLOTS_DIR = BASE_DIR / 'plots' / 'maskdino'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 결과 로드
# ============================================================================

def load_experiment_results():
    """모든 실험 결과 로드"""
    results = {}

    # 각 실험 폴더에서 results.json 로드
    for exp_dir in RESULTS_DIR.iterdir():
        if not exp_dir.is_dir():
            continue

        exp_name = exp_dir.name
        results_file = exp_dir / 'results.json'

        if results_file.exists():
            with open(results_file, 'r') as f:
                data = json.load(f)
                results[exp_name] = data
                print(f"✓ Loaded: {exp_name}")
        else:
            print(f"⚠️  Results not found: {exp_name}")

    return results


def parse_results(results):
    """
    MaskDINO 결과를 DataFrame으로 변환
    MaskDINO outputs: {'bbox': {...}, 'segm': {...}}
    """
    data = []

    for exp_name, exp_results in results.items():
        if 'error' in exp_results:
            print(f"⚠️  Skipping {exp_name}: {exp_results['error']}")
            continue

        # 실험 유형 파싱
        if 'exp_1' in exp_name:
            exp_group = 'Exp1: GenAI Amount'
            genai_amount = int(exp_name.split('genai')[-1])
            aug_type = 'GenAI'
            data_amount = genai_amount
        elif 'exp_2' in exp_name:
            exp_group = 'Exp2: Augmentation Method'

            if 'only' in exp_name:
                aug_type = 'Original Only'
                data_amount = 0
            elif 'traditional' in exp_name and 'genai' not in exp_name:
                aug_type = 'Traditional'
                data_amount = int(exp_name.split('traditional')[-1])
            elif 'genai' in exp_name and 'traditional' in exp_name:
                aug_type = 'GenAI + Traditional'
                parts = exp_name.split('genai')[1].split('_')
                data_amount = int(parts[0])
            elif 'genai' in exp_name:
                aug_type = 'GenAI'
                data_amount = int(exp_name.split('genai')[-1])
            else:
                aug_type = 'Unknown'
                data_amount = 0
        else:
            exp_group = 'Unknown'
            aug_type = 'Unknown'
            data_amount = 0

        # BBox 결과
        bbox_results = exp_results.get('bbox', {})
        bbox_ap = bbox_results.get('AP', 0.0)
        bbox_ap50 = bbox_results.get('AP50', 0.0)
        bbox_ap75 = bbox_results.get('AP75', 0.0)

        # Segmentation 결과
        segm_results = exp_results.get('segm', {})
        segm_ap = segm_results.get('AP', 0.0)
        segm_ap50 = segm_results.get('AP50', 0.0)
        segm_ap75 = segm_results.get('AP75', 0.0)

        data.append({
            'exp_name': exp_name,
            'exp_group': exp_group,
            'aug_type': aug_type,
            'data_amount': data_amount,
            'bbox_AP': bbox_ap,
            'bbox_AP50': bbox_ap50,
            'bbox_AP75': bbox_ap75,
            'segm_AP': segm_ap,
            'segm_AP50': segm_ap50,
            'segm_AP75': segm_ap75,
        })

    return pd.DataFrame(data)


# ============================================================================
# 시각화
# ============================================================================

def plot_exp1_genai_amount(df):
    """
    실험 1: GenAI 증강 데이터 양에 따른 성능 변화
    """
    exp1_df = df[df['exp_group'] == 'Exp1: GenAI Amount'].copy()
    exp1_df = exp1_df.sort_values('data_amount')

    # Baseline 추가
    baseline = df[df['aug_type'] == 'Original Only']
    if not baseline.empty:
        baseline_row = baseline.iloc[0].copy()
        baseline_row['data_amount'] = 0
        exp1_df = pd.concat([pd.DataFrame([baseline_row]), exp1_df], ignore_index=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # BBox metrics
    bbox_metrics = [('bbox_AP50', 'BBox AP@0.5'), ('bbox_AP75', 'BBox AP@0.75'), ('bbox_AP', 'BBox AP@0.5:0.95')]
    for ax, (metric, title) in zip(axes[0], bbox_metrics):
        ax.plot(exp1_df['data_amount'], exp1_df[metric], marker='o', linewidth=2, markersize=8, label='Detection')
        ax.set_xlabel('GenAI Augmentation Amount', fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        for _, row in exp1_df.iterrows():
            ax.annotate(f"{row[metric]:.2f}",
                       xy=(row['data_amount'], row[metric]),
                       xytext=(0, 10),
                       textcoords='offset points',
                       ha='center',
                       fontsize=8,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))

    # Segmentation metrics
    segm_metrics = [('segm_AP50', 'Segm AP@0.5'), ('segm_AP75', 'Segm AP@0.75'), ('segm_AP', 'Segm AP@0.5:0.95')]
    for ax, (metric, title) in zip(axes[1], segm_metrics):
        ax.plot(exp1_df['data_amount'], exp1_df[metric], marker='s', linewidth=2, markersize=8,
                color='green', label='Segmentation')
        ax.set_xlabel('GenAI Augmentation Amount', fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        for _, row in exp1_df.iterrows():
            ax.annotate(f"{row[metric]:.2f}",
                       xy=(row['data_amount'], row[metric]),
                       xytext=(0, 10),
                       textcoords='offset points',
                       ha='center',
                       fontsize=8,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.5))

    plt.suptitle('Experiment 1: GenAI Augmentation Amount Effect (MaskDINO)',
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'exp1_genai_amount.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {PLOTS_DIR / 'exp1_genai_amount.png'}")
    plt.close()


def plot_exp2_aug_method_comparison(df):
    """
    실험 2: 증강 방법별 비교
    """
    exp2_df = df[df['exp_group'] == 'Exp2: Augmentation Method'].copy()
    data_amounts = sorted(exp2_df[exp2_df['data_amount'] > 0]['data_amount'].unique())

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, amount in enumerate(data_amounts):
        if idx >= len(axes):
            break

        ax = axes[idx]

        subset = exp2_df[exp2_df['data_amount'] == amount].copy()
        baseline = exp2_df[exp2_df['aug_type'] == 'Original Only']
        if not baseline.empty:
            subset = pd.concat([baseline, subset], ignore_index=True)

        subset = subset.sort_values('bbox_AP50')

        x = np.arange(len(subset))
        width = 0.35

        # BBox vs Segm 비교
        ax.bar(x - width/2, subset['bbox_AP50'], width, label='Detection AP@0.5', alpha=0.8)
        ax.bar(x + width/2, subset['segm_AP50'], width, label='Segmentation AP@0.5', alpha=0.8, color='green')

        ax.set_xlabel('Augmentation Method', fontsize=11)
        ax.set_ylabel('AP@0.5', fontsize=11)
        ax.set_title(f'Data Amount: {amount if amount > 0 else "Baseline"}',
                    fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(subset['aug_type'], rotation=15, ha='right', fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Experiment 2: Augmentation Method Comparison (MaskDINO)',
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'exp2_aug_method.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {PLOTS_DIR / 'exp2_aug_method.png'}")
    plt.close()


def plot_bbox_vs_segm_correlation(df):
    """
    Detection vs Segmentation 성능 상관관계
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    aug_types = df['aug_type'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(aug_types)))
    color_map = dict(zip(aug_types, colors))

    for aug_type in aug_types:
        subset = df[df['aug_type'] == aug_type]
        ax.scatter(subset['bbox_AP50'], subset['segm_AP50'],
                  label=aug_type, s=100, alpha=0.7, color=color_map[aug_type])

    # 대각선 (bbox_AP = segm_AP)
    min_val = min(df['bbox_AP50'].min(), df['segm_AP50'].min())
    max_val = max(df['bbox_AP50'].max(), df['segm_AP50'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, label='y=x')

    ax.set_xlabel('Detection AP@0.5', fontsize=12)
    ax.set_ylabel('Segmentation AP@0.5', fontsize=12)
    ax.set_title('Detection vs Segmentation Performance', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'bbox_vs_segm.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {PLOTS_DIR / 'bbox_vs_segm.png'}")
    plt.close()


def plot_overall_heatmap(df):
    """전체 성능 히트맵 (BBox + Segm)"""
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # BBox AP50 heatmap
    pivot_bbox = df.pivot_table(
        index='aug_type',
        columns='data_amount',
        values='bbox_AP50',
        aggfunc='mean'
    )

    sns.heatmap(pivot_bbox, annot=True, fmt='.2f', cmap='YlOrRd',
                cbar_kws={'label': 'Detection AP@0.5'}, ax=axes[0], linewidths=0.5)
    axes[0].set_title('Detection Performance (AP@0.5)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Augmentation Data Amount', fontsize=11)
    axes[0].set_ylabel('Augmentation Type', fontsize=11)

    # Segm AP50 heatmap
    pivot_segm = df.pivot_table(
        index='aug_type',
        columns='data_amount',
        values='segm_AP50',
        aggfunc='mean'
    )

    sns.heatmap(pivot_segm, annot=True, fmt='.2f', cmap='YlGnBu',
                cbar_kws={'label': 'Segmentation AP@0.5'}, ax=axes[1], linewidths=0.5)
    axes[1].set_title('Segmentation Performance (AP@0.5)', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Augmentation Data Amount', fontsize=11)
    axes[1].set_ylabel('Augmentation Type', fontsize=11)

    plt.suptitle('Overall Performance Heatmap (MaskDINO)', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'overall_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {PLOTS_DIR / 'overall_heatmap.png'}")
    plt.close()


# ============================================================================
# 통계 분석
# ============================================================================

def print_statistics(df):
    """통계 분석 출력"""
    print(f"\n{'='*80}")
    print("Statistical Analysis for MaskDINO")
    print(f"{'='*80}")

    # 증강 타입별 평균 성능
    print("\n1. Average Performance by Augmentation Type:")
    print("-" * 80)
    aug_stats = df.groupby('aug_type')[['bbox_AP50', 'bbox_AP', 'segm_AP50', 'segm_AP']].mean()
    print(aug_stats.to_string())

    # 최고 성능 실험
    print("\n2. Best Performing Experiments:")
    print("-" * 80)

    best_bbox_idx = df['bbox_AP50'].idxmax()
    best_bbox = df.loc[best_bbox_idx]
    print(f"\nBest Detection (AP@0.5): {best_bbox['exp_name']}")
    print(f"  BBox AP@0.5: {best_bbox['bbox_AP50']:.2f}")
    print(f"  BBox AP@0.75: {best_bbox['bbox_AP75']:.2f}")
    print(f"  BBox AP: {best_bbox['bbox_AP']:.2f}")
    print(f"  Augmentation: {best_bbox['aug_type']}, Amount: {best_bbox['data_amount']}")

    best_segm_idx = df['segm_AP50'].idxmax()
    best_segm = df.loc[best_segm_idx]
    print(f"\nBest Segmentation (AP@0.5): {best_segm['exp_name']}")
    print(f"  Segm AP@0.5: {best_segm['segm_AP50']:.2f}")
    print(f"  Segm AP@0.75: {best_segm['segm_AP75']:.2f}")
    print(f"  Segm AP: {best_segm['segm_AP']:.2f}")
    print(f"  Augmentation: {best_segm['aug_type']}, Amount: {best_segm['data_amount']}")

    # Baseline 대비 성능 향상
    print("\n3. Performance Improvement over Baseline:")
    print("-" * 80)
    baseline = df[df['aug_type'] == 'Original Only']
    if not baseline.empty:
        baseline_bbox = baseline.iloc[0]['bbox_AP50']
        baseline_segm = baseline.iloc[0]['segm_AP50']
        print(f"Baseline Detection AP@0.5: {baseline_bbox:.2f}")
        print(f"Baseline Segmentation AP@0.5: {baseline_segm:.2f}")

        print("\nImprovement (Detection):")
        for _, row in df.iterrows():
            if row['aug_type'] != 'Original Only':
                improvement = ((row['bbox_AP50'] - baseline_bbox) / baseline_bbox) * 100
                print(f"  {row['exp_name']:<45s}: {improvement:+6.2f}%")

        print("\nImprovement (Segmentation):")
        for _, row in df.iterrows():
            if row['aug_type'] != 'Original Only':
                improvement = ((row['segm_AP50'] - baseline_segm) / baseline_segm) * 100
                print(f"  {row['exp_name']:<45s}: {improvement:+6.2f}%")

    print("\n" + "="*80)


# ============================================================================
# Main
# ============================================================================

def main():
    print(f"\n{'='*80}")
    print("Analyzing MaskDINO Results")
    print(f"{'='*80}")

    # 결과 로드
    results = load_experiment_results()
    if not results:
        print("❌ No results found. Please run training first.")
        return

    # DataFrame 변환
    df = parse_results(results)

    print(f"\nTotal experiments loaded: {len(df)}")
    print(f"\nExperiment groups:")
    print(df['exp_group'].value_counts())
    print(f"\nAugmentation types:")
    print(df['aug_type'].value_counts())

    # 통계 분석
    print_statistics(df)

    # 시각화
    print(f"\n{'='*80}")
    print("Generating Plots...")
    print(f"{'='*80}")

    plot_exp1_genai_amount(df)
    plot_exp2_aug_method_comparison(df)
    plot_bbox_vs_segm_correlation(df)
    plot_overall_heatmap(df)

    # CSV 저장
    csv_file = PLOTS_DIR / 'results_summary.csv'
    df.to_csv(csv_file, index=False)
    print(f"\nSaved CSV: {csv_file}")

    print(f"\n{'='*80}")
    print(f"✅ Analysis Complete!")
    print(f"Plots saved to: {PLOTS_DIR}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
