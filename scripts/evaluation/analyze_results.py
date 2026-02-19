"""
실험 결과 분석 및 시각화
- 13개 실험 결과 비교
- 증강 방법별 성능 그래프
- 데이터 양에 따른 성능 변화
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# Paths
BASE_DIR = Path('/data2/project/2026winter/jjh0709/AA_CV_R')
RESULTS_DIR = BASE_DIR / 'results'
PLOTS_DIR = BASE_DIR / 'plots'
PLOTS_DIR.mkdir(exist_ok=True)

# ============================================================================
# 결과 로드
# ============================================================================

def load_results(model_type):
    """
    모델의 모든 실험 결과 로드
    """
    results_file = RESULTS_DIR / model_type / 'all_results.json'

    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        return None

    with open(results_file, 'r') as f:
        results = json.load(f)

    return results


def results_to_dataframe(results):
    """
    결과를 DataFrame으로 변환
    """
    data = []

    for res in results:
        exp_name = res['exp_name']

        # 실험 유형 파싱
        if 'exp_1' in exp_name:
            exp_group = 'Exp1: GenAI Amount'
            # exp_1_original26_genai50 -> 50
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
                # exp_2_original26_traditional50 -> 50
                data_amount = int(exp_name.split('traditional')[-1])
            elif 'genai' in exp_name and 'traditional' in exp_name:
                aug_type = 'GenAI + Traditional'
                # exp_2_original26_genai50_traditional -> 50
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

        data.append({
            'exp_name': exp_name,
            'exp_group': exp_group,
            'aug_type': aug_type,
            'data_amount': data_amount,
            'mAP50': res['mAP50'],
            'mAP75': res['mAP75'],
            'mAP': res['mAP'],
            'precision': res['precision'],
            'recall': res['recall'],
        })

    return pd.DataFrame(data)


# ============================================================================
# 시각화
# ============================================================================

def plot_exp1_genai_amount(df, model_type):
    """
    실험 1: GenAI 증강 데이터 양에 따른 성능 변화
    """
    exp1_df = df[df['exp_group'] == 'Exp1: GenAI Amount'].copy()
    exp1_df = exp1_df.sort_values('data_amount')

    # Baseline 추가 (원본만)
    baseline = df[df['aug_type'] == 'Original Only']
    if not baseline.empty:
        baseline_row = baseline.iloc[0].copy()
        baseline_row['data_amount'] = 0
        exp1_df = pd.concat([pd.DataFrame([baseline_row]), exp1_df], ignore_index=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    metrics = ['mAP50', 'mAP75', 'mAP']
    titles = ['mAP@0.5', 'mAP@0.75', 'mAP@0.5:0.95']

    for ax, metric, title in zip(axes, metrics, titles):
        ax.plot(exp1_df['data_amount'], exp1_df[metric], marker='o', linewidth=2, markersize=8)
        ax.set_xlabel('GenAI Augmentation Amount', fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(f'{title} vs GenAI Amount', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Annotate points
        for _, row in exp1_df.iterrows():
            ax.annotate(f"{row[metric]:.3f}",
                       xy=(row['data_amount'], row[metric]),
                       xytext=(0, 10),
                       textcoords='offset points',
                       ha='center',
                       fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))

    plt.suptitle(f'Experiment 1: GenAI Augmentation Amount Effect ({model_type})',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f'{model_type}_exp1_genai_amount.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {PLOTS_DIR / f'{model_type}_exp1_genai_amount.png'}")
    plt.close()


def plot_exp2_aug_method_comparison(df, model_type):
    """
    실험 2: 증강 방법별 비교 (각 데이터 양별로)
    """
    exp2_df = df[df['exp_group'] == 'Exp2: Augmentation Method'].copy()

    # 데이터 양별로 그룹화
    data_amounts = sorted(exp2_df[exp2_df['data_amount'] > 0]['data_amount'].unique())

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, amount in enumerate(data_amounts):
        if idx >= len(axes):
            break

        ax = axes[idx]

        # 해당 데이터 양의 실험들
        subset = exp2_df[exp2_df['data_amount'] == amount].copy()

        # Baseline 추가
        baseline = exp2_df[exp2_df['aug_type'] == 'Original Only']
        if not baseline.empty:
            subset = pd.concat([baseline, subset], ignore_index=True)

        # mAP50 기준으로 정렬
        subset = subset.sort_values('mAP50')

        # Bar plot
        x = range(len(subset))
        width = 0.25

        ax.bar([i - width for i in x], subset['mAP50'], width, label='mAP@0.5', alpha=0.8)
        ax.bar(x, subset['mAP75'], width, label='mAP@0.75', alpha=0.8)
        ax.bar([i + width for i in x], subset['mAP'], width, label='mAP@0.5:0.95', alpha=0.8)

        ax.set_xlabel('Augmentation Method', fontsize=11)
        ax.set_ylabel('mAP', fontsize=11)
        ax.set_title(f'Augmentation Amount: {amount if amount > 0 else "Baseline"}',
                    fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(subset['aug_type'], rotation=15, ha='right', fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle(f'Experiment 2: Augmentation Method Comparison ({model_type})',
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f'{model_type}_exp2_aug_method.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {PLOTS_DIR / f'{model_type}_exp2_aug_method.png'}")
    plt.close()


def plot_overall_comparison(df, model_type):
    """
    전체 실험 비교 (Heatmap)
    """
    # Pivot table 생성
    pivot_df = df.pivot_table(
        index='aug_type',
        columns='data_amount',
        values='mAP50',
        aggfunc='mean'
    )

    fig, ax = plt.subplots(figsize=(12, 6))

    sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='YlGnBu',
                cbar_kws={'label': 'mAP@0.5'}, ax=ax, linewidths=0.5)

    ax.set_xlabel('Augmentation Data Amount', fontsize=12)
    ax.set_ylabel('Augmentation Type', fontsize=12)
    ax.set_title(f'Overall Performance Heatmap ({model_type})', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f'{model_type}_overall_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {PLOTS_DIR / f'{model_type}_overall_heatmap.png'}")
    plt.close()


def plot_precision_recall(df, model_type):
    """
    Precision-Recall 비교
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # 증강 타입별로 색상 지정
    aug_types = df['aug_type'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(aug_types)))
    color_map = dict(zip(aug_types, colors))

    for aug_type in aug_types:
        subset = df[df['aug_type'] == aug_type]
        ax.scatter(subset['recall'], subset['precision'],
                  label=aug_type, s=100, alpha=0.7, color=color_map[aug_type])

    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(f'Precision-Recall Comparison ({model_type})', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f'{model_type}_precision_recall.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {PLOTS_DIR / f'{model_type}_precision_recall.png'}")
    plt.close()


# ============================================================================
# 통계 분석
# ============================================================================

def print_statistics(df, model_type):
    """
    통계 분석 출력
    """
    print(f"\n{'='*80}")
    print(f"Statistical Analysis for {model_type}")
    print(f"{'='*80}")

    # 증강 타입별 평균 성능
    print("\n1. Average Performance by Augmentation Type:")
    print("-" * 80)
    aug_stats = df.groupby('aug_type')[['mAP50', 'mAP75', 'mAP']].mean()
    print(aug_stats.to_string())

    # 최고 성능 실험
    print("\n2. Best Performing Experiments:")
    print("-" * 80)
    best_idx = df['mAP50'].idxmax()
    best_exp = df.loc[best_idx]
    print(f"Best mAP@0.5: {best_exp['exp_name']}")
    print(f"  mAP@0.5: {best_exp['mAP50']:.4f}")
    print(f"  mAP@0.75: {best_exp['mAP75']:.4f}")
    print(f"  mAP@0.5:0.95: {best_exp['mAP']:.4f}")
    print(f"  Augmentation: {best_exp['aug_type']}")
    print(f"  Data Amount: {best_exp['data_amount']}")

    # Baseline 대비 성능 향상
    print("\n3. Performance Improvement over Baseline:")
    print("-" * 80)
    baseline = df[df['aug_type'] == 'Original Only']
    if not baseline.empty:
        baseline_map50 = baseline.iloc[0]['mAP50']
        print(f"Baseline mAP@0.5: {baseline_map50:.4f}")
        print("\nImprovement:")
        for _, row in df.iterrows():
            if row['aug_type'] != 'Original Only':
                improvement = ((row['mAP50'] - baseline_map50) / baseline_map50) * 100
                print(f"  {row['exp_name']:<45s}: {improvement:+6.2f}%")

    print("\n" + "="*80)


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Analyze experiment results')
    parser.add_argument('--model', type=str, required=True,
                        help='Model type to analyze (yolo, maskdino, faster_rcnn)')

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"Analyzing Results for {args.model}")
    print(f"{'='*80}")

    # 결과 로드
    results = load_results(args.model)
    if results is None:
        print("No results found. Please run training first.")
        return

    # DataFrame 변환
    df = results_to_dataframe(results)

    print(f"\nTotal experiments: {len(df)}")
    print(f"\nExperiment groups:")
    print(df['exp_group'].value_counts())
    print(f"\nAugmentation types:")
    print(df['aug_type'].value_counts())

    # 통계 분석
    print_statistics(df, args.model)

    # 시각화
    print(f"\n{'='*80}")
    print("Generating Plots...")
    print(f"{'='*80}")

    plot_exp1_genai_amount(df, args.model)
    plot_exp2_aug_method_comparison(df, args.model)
    plot_overall_comparison(df, args.model)
    plot_precision_recall(df, args.model)

    print(f"\n{'='*80}")
    print(f"✅ Analysis Complete!")
    print(f"Plots saved to: {PLOTS_DIR}")
    print(f"{'='*80}")


if __name__ == '__main__':
    # 사용 예시:
    # python analyze_results.py --model yolo
    # python analyze_results.py --model maskdino

    main()
