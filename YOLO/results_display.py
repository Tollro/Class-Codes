import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

# ==================== 数据加载与预处理 ====================
def load_and_merge_epochs(file_path):
    """
    读取CSV，处理重复表头，合并两阶段epoch（将第二阶段epoch加上第一阶段最大epoch值）
    """
    # 先用pandas读取，跳过重复表头（rows where first column is 'epoch'）
    raw_data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('epoch'):   # 跳过表头行
                continue
            if line.strip():               # 非空行
                raw_data.append(line.strip().split(','))

    # 转换为DataFrame，指定列名
    columns = ['epoch','time','train/box_loss','train/cls_loss','train/dfl_loss',
               'metrics/precision(B)','metrics/recall(B)','metrics/mAP50(B)','metrics/mAP50-95(B)',
               'val/box_loss','val/cls_loss','val/dfl_loss','lr/pg0','lr/pg1','lr/pg2']
    df = pd.DataFrame(raw_data, columns=columns)

    # 转换为数值类型
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 检测第二阶段起始（epoch突然从大值变回1）
    epoch_col = df['epoch']
    split_idx = None
    for i in range(1, len(epoch_col)):
        if epoch_col.iloc[i] == 1 and epoch_col.iloc[i-1] > 1:
            split_idx = i
            break

    if split_idx is None:
        print("未检测到第二阶段，数据可能已连续或只有一个阶段")
        return df

    # 第一阶段
    first_stage = df.iloc[:split_idx].copy()
    # 第二阶段
    second_stage = df.iloc[split_idx:].copy()

    # 计算偏移量：第一阶段最大epoch
    offset = int(first_stage['epoch'].max())
    second_stage['epoch'] = second_stage['epoch'] + offset

    # 合并
    full_df = pd.concat([first_stage, second_stage], ignore_index=True)
    print(f"数据合并完成：第一阶段 epochs {first_stage['epoch'].min()}–{first_stage['epoch'].max()}, "
          f"第二阶段 epochs {second_stage['epoch'].min()}–{second_stage['epoch'].max()}, "
          f"总计 {len(full_df)} 行。")
    return full_df

# ==================== 指标分析 ====================
def analyze_metrics(df):
    """打印关键指标统计，并找出最佳epoch"""
    print("\n========== 关键指标统计 ==========")
    # 关注验证集指标
    metrics_of_interest = {
        'mAP50 (B)': 'metrics/mAP50(B)',
        'mAP50-95 (B)': 'metrics/mAP50-95(B)',
        'Precision (B)': 'metrics/precision(B)',
        'Recall (B)': 'metrics/recall(B)',
        'val/box_loss': 'val/box_loss',
        'val/cls_loss': 'val/cls_loss',
        'val/dfl_loss': 'val/dfl_loss',
        'train/box_loss': 'train/box_loss',
        'train/cls_loss': 'train/cls_loss',
        'train/dfl_loss': 'train/dfl_loss'
    }

    for name, col in metrics_of_interest.items():
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if len(series) == 0:
            continue
        # 对于损失，最低为好；对于精度/mAP，最高为好
        if 'loss' in col.lower():
            best_idx = series.idxmin()
            best_val = series.min()
            best_epoch = df.loc[best_idx, 'epoch']
            trend = "最小值"
        else:
            best_idx = series.idxmax()
            best_val = series.max()
            best_epoch = df.loc[best_idx, 'epoch']
            trend = "最大值"
        last_val = series.iloc[-1]
        print(f"{name:20s} : 最终值 = {last_val:.4f}  |  {trend} = {best_val:.4f} @ epoch {int(best_epoch)}")

    # 额外找出最佳mAP50-95对应的epoch
    map95_col = 'metrics/mAP50-95(B)'
    if map95_col in df.columns:
        best_map95_idx = df[map95_col].idxmax()
        best_map95_epoch = df.loc[best_map95_idx, 'epoch']
        best_map95_val = df.loc[best_map95_idx, map95_col]
        print(f"\n🏆 最佳 mAP50-95 = {best_map95_val:.4f} 出现在 epoch {int(best_map95_epoch)}")

    # 学习率变化
    lr_col = 'lr/pg0'
    if lr_col in df.columns:
        lr_series = df[lr_col].dropna()
        print(f"\n学习率范围: {lr_series.min():.2e} → {lr_series.max():.2e}")

# ==================== 可视化 ====================
def plot_metrics(df, save_path='training_analysis.png'):
    """绘制训练/验证损失和关键指标曲线"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    ax1, ax2, ax3 = axes[0,0], axes[0,1], axes[0,2]
    ax4, ax5, ax6 = axes[1,0], axes[1,1], axes[1,2]

    epochs = df['epoch'].values

    # 第一行：损失函数
    ax1.plot(epochs, df['train/box_loss'], label='train/box_loss', color='blue')
    ax1.plot(epochs, df['val/box_loss'], label='val/box_loss', color='orange')
    ax1.set_title('Box Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2.plot(epochs, df['train/cls_loss'], label='train/cls_loss', color='blue')
    ax2.plot(epochs, df['val/cls_loss'], label='val/cls_loss', color='orange')
    ax2.set_title('Class Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    ax3.plot(epochs, df['train/dfl_loss'], label='train/dfl_loss', color='blue')
    ax3.plot(epochs, df['val/dfl_loss'], label='val/dfl_loss', color='orange')
    ax3.set_title('DFL Loss')
    ax3.set_xlabel('Epoch')
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.6)

    # 第二行：检测指标
    ax4.plot(epochs, df['metrics/precision(B)'], label='Precision', color='green')
    ax4.plot(epochs, df['metrics/recall(B)'], label='Recall', color='red')
    ax4.set_title('Precision & Recall')
    ax4.set_xlabel('Epoch')
    ax4.legend()
    ax4.grid(True, linestyle='--', alpha=0.6)

    ax5.plot(epochs, df['metrics/mAP50(B)'], label='mAP50', color='purple')
    ax5.plot(epochs, df['metrics/mAP50-95(B)'], label='mAP50-95', color='brown')
    ax5.set_title('mAP')
    ax5.set_xlabel('Epoch')
    ax5.legend()
    ax5.grid(True, linestyle='--', alpha=0.6)

    # 学习率（可选）
    if 'lr/pg0' in df.columns:
        ax6.plot(epochs, df['lr/pg0'], label='lr/pg0', color='black')
        ax6.set_title('Learning Rate')
        ax6.set_xlabel('Epoch')
        ax6.set_yscale('log')
        ax6.legend()
        ax6.grid(True, linestyle='--', alpha=0.6)
    else:
        ax6.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\n图表已保存至 {save_path}")
    plt.show()

# ==================== 主程序 ====================
if __name__ == "__main__":
    # 请将您的日志文件命名为 'results.csv' 并放在当前目录，或修改为实际路径
    file_path = 'yolo11nresults.csv'   # 您可以修改文件名
    try:
        df = load_and_merge_epochs(file_path)
        analyze_metrics(df)
        plot_metrics(df)
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到，请检查路径或将日志内容保存为 CSV 文件。")