"""
YOLO训练结果分析脚本
作用：读取训练产生的results.csv，绘制损失变化、精度/召回、mAP、学习率等曲线，
      并打印最优指标。
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

def load_results(csv_path='results.csv'):
    """读取CSV文件，处理列名空格，返回DataFrame。"""
    if not os.path.exists(csv_path):
        print(f"错误：文件 '{csv_path}' 不存在。")
        sys.exit(1)
    df = pd.read_csv(csv_path)
    # 去除列名前后的空格
    df.columns = df.columns.str.strip()
    print(f"成功加载数据，共 {len(df)} 个epoch，列名：{list(df.columns)}")
    return df

def plot_losses(df, save_path='loss_curves.png'):
    """绘制训练和验证的 box_loss, cls_loss, dfl_loss 以及总损失趋势。"""
    # 检查必要的列是否存在
    required_loss_cols = [
        'train/box_loss', 'train/cls_loss', 'train/dfl_loss',
        'val/box_loss', 'val/cls_loss', 'val/dfl_loss'
    ]
    if not all(col in df.columns for col in required_loss_cols):
        print("警告：损失列不全，跳过损失曲线绘制。")
        return

    epochs = df['epoch']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    # 1) 训练损失
    axes[0,0].plot(epochs, df['train/box_loss'], label='Train Box Loss')
    axes[0,0].plot(epochs, df['train/cls_loss'], label='Train Cls Loss')
    axes[0,0].plot(epochs, df['train/dfl_loss'], label='Train DFL Loss')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].set_title('Training Losses')
    axes[0,0].legend()
    axes[0,0].grid(True)

    # 2) 验证损失
    axes[0,1].plot(epochs, df['val/box_loss'], label='Val Box Loss')
    axes[0,1].plot(epochs, df['val/cls_loss'], label='Val Cls Loss')
    axes[0,1].plot(epochs, df['val/dfl_loss'], label='Val DFL Loss')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('Loss')
    axes[0,1].set_title('Validation Losses')
    axes[0,1].legend()
    axes[0,1].grid(True)

    # 3) 训练 vs 验证总损失 (三个分量求和)
    df['train/total_loss'] = df['train/box_loss'] + df['train/cls_loss'] + df['train/dfl_loss']
    df['val/total_loss'] = df['val/box_loss'] + df['val/cls_loss'] + df['val/dfl_loss']
    axes[1,0].plot(epochs, df['train/total_loss'], label='Train Total Loss')
    axes[1,0].plot(epochs, df['val/total_loss'], label='Val Total Loss')
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].set_ylabel('Total Loss')
    axes[1,0].set_title('Total Loss (sum of components)')
    axes[1,0].legend()
    axes[1,0].grid(True)

    # 4) 损失比 (val/train) 可观察过拟合
    df['loss_ratio'] = df['val/total_loss'] / (df['train/total_loss'] + 1e-8)
    axes[1,1].plot(epochs, df['loss_ratio'], color='purple')
    axes[1,1].axhline(y=1.0, color='gray', linestyle='--', label='Ratio=1')
    axes[1,1].set_xlabel('Epoch')
    axes[1,1].set_ylabel('Val/Train Total Loss Ratio')
    axes[1,1].set_title('Val/Train Loss Ratio')
    axes[1,1].legend()
    axes[1,1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"损失曲线已保存至 '{save_path}'。")

def plot_metrics(df, save_path='metrics_curves.png'):
    """绘制准确率、召回率、mAP50 和 mAP50-95 曲线。"""
    metric_cols = ['metrics/precision(B)', 'metrics/recall(B)',
                   'metrics/mAP50(B)', 'metrics/mAP50-95(B)']
    if not all(col in df.columns for col in metric_cols):
        print("警告：评估指标列不全，跳过指标曲线绘制。")
        return

    epochs = df['epoch']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0,0].plot(epochs, df['metrics/precision(B)'], color='blue')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Precision')
    axes[0,0].set_title('Precision (B)')
    axes[0,0].grid(True)

    axes[0,1].plot(epochs, df['metrics/recall(B)'], color='orange')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('Recall')
    axes[0,1].set_title('Recall (B)')
    axes[0,1].grid(True)

    axes[1,0].plot(epochs, df['metrics/mAP50(B)'], color='green')
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].set_ylabel('mAP@0.5')
    axes[1,0].set_title('mAP@0.5')
    axes[1,0].grid(True)

    axes[1,1].plot(epochs, df['metrics/mAP50-95(B)'], color='red')
    axes[1,1].set_xlabel('Epoch')
    axes[1,1].set_ylabel('mAP@0.5:0.95')
    axes[1,1].set_title('mAP@0.5:0.95')
    axes[1,1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"评估指标曲线已保存至 '{save_path}'。")

def plot_lr(df, save_path='lr_curves.png'):
    """绘制不同参数组的学习率变化。"""
    lr_cols = ['lr/pg0', 'lr/pg1', 'lr/pg2']
    existing_lr = [col for col in lr_cols if col in df.columns]
    if not existing_lr:
        print("警告：学习率列未找到，跳过学习率曲线。")
        return

    epochs = df['epoch']
    plt.figure(figsize=(10, 5))
    for col in existing_lr:
        plt.plot(epochs, df[col], label=col)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedules')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"学习率曲线已保存至 '{save_path}'。")

def print_best_metrics(df):
    """打印每个指标的最佳值及对应epoch。"""
    metric_cols = {
        'metrics/precision(B)': 'max',
        'metrics/recall(B)': 'max',
        'metrics/mAP50(B)': 'max',
        'metrics/mAP50-95(B)': 'max',
        'val/box_loss': 'min',
        'val/cls_loss': 'min',
        'val/dfl_loss': 'min',
        'train/box_loss': 'min',
        'train/cls_loss': 'min',
        'train/dfl_loss': 'min'
    }
    print("\n========== 训练最优指标 ==========")
    for col, mode in metric_cols.items():
        if col not in df.columns:
            continue
        if mode == 'max':
            best_row = df.loc[df[col].idxmax()]
            best_val = best_row[col]
        else:
            best_row = df.loc[df[col].idxmin()]
            best_val = best_row[col]
        print(f"{col}: {best_val:.6f}  于 Epoch {int(best_row['epoch'])}")

    # 打印最终epoch的指标
    final_epoch = df['epoch'].max()
    final_row = df[df['epoch'] == final_epoch].iloc[0]
    print(f"\n--- 最终 Epoch {int(final_epoch)} 指标 ---")
    for col in metric_cols.keys():
        if col in df.columns:
            print(f"{col}: {final_row[col]:.6f}")
    # 总训练时长
    if 'time' in df.columns:
        total_time_hours = df['time'].max() / 3600  # 假设time是累计秒数
        print(f"\n总训练时长：{total_time_hours:.2f} 小时")

def main():
    # 如果命令行提供了CSV路径，则使用；否则默认当前目录下的results.csv
    csv_path = "D:\\Git\\my_respositories\\Class-Codes\\runs\\detect\\fruit_detect_train\\yolo11n_fruit_exp1-2-2\\results.csv"
    df = load_results(csv_path)

    # 绘制并保存所有曲线
    plot_losses(df, save_path='loss_curves.png')
    plot_metrics(df, save_path='metrics_curves.png')
    plot_lr(df, save_path='lr_curves.png')

    # 打印最优指标
    print_best_metrics(df)

if __name__ == "__main__":
    main()