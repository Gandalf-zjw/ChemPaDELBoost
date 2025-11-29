import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Wedge
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib import colormaps
import pandas as pd
import numpy as np
import matplotlib.patches as patches
import os
import glob

from matplotlib.colors import LinearSegmentedColormap

FS = 1.50  # 字号整体放大 30%，按需改成 1.2、1.5 等
def fz(x):  # 统一把字号转成整数
    return int(round(x * FS))

# 色条（右侧图例）样式
CB_LABEL_FZ = fz(18)   # 色条标题字号（原来相当于 fz(12~14)）
CB_TICK_FZ  = fz(14)   # 色条刻度字号（原来是 10~11）
CB_LINE_W   = 2.2      # 色条边框/刻度线粗细
CB_PAD      = 0.06     # 色条与主图间距，稍加大避免挤


def create_macaron_cmap(percentages):
    """
    马卡龙渐变 - 从浅色开始但不是白色
    """
    macaron_colors = [
        "#f0f8ff",  # 爱丽丝蓝 - 非常浅的蓝色但不是白色
        "#fde2e4",  # pastel pink
        "#fad2e1",  # light rose
        "#fff1c1",  # pastel lemon
        "#e2f0cb",  # pastel mint
        "#bde0fe"   # pastel blue
    ]
    cmap = LinearSegmentedColormap.from_list("macaron", macaron_colors, N=256)
    norm_percentages = {k: v / 100 for k, v in percentages.items()}
    return cmap, norm_percentages

def setup_plot_style():
    plt.rcParams.update({
        "font.family": "Calibri",        # 全局用 Calibri
        "font.weight": "bold",           # 全局加粗
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "font.size": fz(12),
        "axes.edgecolor": "black",
        "axes.linewidth": 1.8,           # 轴线更粗
        "xtick.color": "black",
        "ytick.color": "black",
        "xtick.major.width": 1.6,
        "ytick.major.width": 1.6,
        "savefig.dpi": 300,
    })



def calculate_percentages_and_radii(tp, fp, fn, tn, max_r):
    """
    按预测类别归一化：
      预测为正：TP + FP = 100%
      预测为负：FN + TN = 100%
    """
    percentages = {
        "TP": tp / (tp + fp) * 100 if (tp + fp) != 0 else 0,
        "FP": fp / (tp + fp) * 100 if (tp + fp) != 0 else 0,
        "FN": fn / (fn + tn) * 100 if (fn + tn) != 0 else 0,
        "TN": tn / (fn + tn) * 100 if (fn + tn) != 0 else 0
    }

    max_val = max(percentages.values()) if max(percentages.values()) > 0 else 1
    radii = {k: max_r * np.sqrt(v / max_val) for k, v in percentages.items()}
    return percentages, radii


def create_colormap_option8(percentages):
    cmap = colormaps["inferno"]  # 可换成 "plasma"、"coolwarm_r"、"magma" ,"RdYlGn_r",等
    norm_percentages = {k: v / 100 for k, v in percentages.items()}
    return cmap, norm_percentages


def draw_sectors(ax, center, radii, cmap, norm_percentages):
    cx, cy = center
    angles = {
        "TP": (90, 180),
        "FN": (0, 90),
        "FP": (180, 270),
        "TN": (270, 360)
    }
    for key, (theta1, theta2) in angles.items():
        color = cmap(norm_percentages[key])
        # 增强边框，特别是对于浅色区域
        edge_width = 1.5 if norm_percentages[key] < 0.3 else 1.2
        wedge = Wedge(center=(cx, cy), r=radii[key],
                      theta1=theta1, theta2=theta2,
                      facecolor=color, edgecolor="gray", lw=edge_width)  # 使用灰色边框增强可见性
        ax.add_patch(wedge)


def add_quadrant_borders(ax, max_r):
    ax.plot([0.5 - max_r, 0.5 + max_r], [0.5, 0.5], color="black", lw=1.8)
    ax.plot([0.5, 0.5], [0.5 - max_r, 0.5 + max_r], color="black", lw=1.8)
    rect = plt.Rectangle((0.5 - max_r, 0.5 - max_r), 2 * max_r, 2 * max_r,
                         fill=False, edgecolor="black", lw=1.8)
    ax.add_patch(rect)



def add_sector_center_labels(ax, max_r):
    labels = {
        "TP": (0.5 - max_r * 0.35, 0.5 + max_r * 0.35),
        "FN": (0.5 + max_r * 0.35, 0.5 + max_r * 0.35),
        "FP": (0.5 - max_r * 0.35, 0.5 - max_r * 0.35),
        "TN": (0.5 + max_r * 0.35, 0.5 - max_r * 0.35)
    }
    for k, (x, y) in labels.items():
        ax.text(x, y, k, ha="center", va="center",
                fontsize=fz(14), color="black", fontweight="bold", fontfamily="Calibri")



def add_labels(ax, max_r, percentages):
    label_positions = {
        "TP": (0.5 - max_r * 0.35, 0.5 + max_r * 0.7),
        "FN": (0.5 + max_r * 0.35, 0.5 + max_r * 0.7),
        "FP": (0.5 - max_r * 0.35, 0.5 - max_r * 0.7),
        "TN": (0.5 + max_r * 0.35, 0.5 - max_r * 0.7)
    }
    for k, (x, y) in label_positions.items():
        ax.text(x, y, f"{percentages[k]:.1f}%", ha="center", va="center",
                fontsize=fz(14), color="black", fontweight="bold", fontfamily="Calibri")



def add_axis_labels(ax, max_r, line_width=1.6, font_size=fz(12)):
    """
    添加带灰色框的坐标轴标签（Predicted/Actual Positive/Negative）
    """
    label_w = max_r * 0.2  # 左侧框宽度（相对于主图）
    label_h = max_r * 0.2  # 顶部框高度

    # === 左侧 Actual Positive ===
    ax.add_patch(patches.Rectangle(
        (0.5 - max_r - label_w, 0.5),
        label_w, max_r, facecolor='#F2F2F2',
        edgecolor='black', linewidth=line_width))
    ax.text(0.5 - max_r - label_w / 2, 0.5 + max_r / 2,
            'Positive', rotation=90, ha='center', va='center',
            fontsize=font_size, weight='bold', color='black', fontfamily='Calibri')

    # === 左侧 Actual Negative ===
    ax.add_patch(patches.Rectangle(
        (0.5 - max_r - label_w, 0.5 - max_r),
        label_w, max_r, facecolor='#F2F2F2',
        edgecolor='black', linewidth=line_width))
    ax.text(0.5 - max_r - label_w / 2, 0.5 - max_r / 2,
            'Negative', rotation=90, ha='center', va='center',
            fontsize=font_size, weight='bold', color='black', fontfamily='Arial')

    # === 顶部 Predicted Positive ===
    ax.add_patch(patches.Rectangle(
        (0.5 - max_r, 0.5 + max_r),
        max_r, label_h, facecolor='#F2F2F2',
        edgecolor='black', linewidth=line_width))
    ax.text(0.5 - max_r / 2, 0.5 + max_r + label_h / 2,
            'Positive', ha='center', va='center',
            fontsize=font_size, weight='bold', color='black', fontfamily='Arial')

    # === 顶部 Predicted Negative ===
    ax.add_patch(patches.Rectangle(
        (0.5, 0.5 + max_r),
        max_r, label_h, facecolor='#F2F2F2',
        edgecolor='black', linewidth=line_width))
    ax.text(0.5 + max_r / 2, 0.5 + max_r + label_h / 2,
            'Negative', ha='center', va='center',
            fontsize=font_size, weight='bold', color='black', fontfamily='Arial')

    # === 添加总体标题 ===
    ax.text(0.5, 0.5 + max_r + label_h * 1.6, "Predicted Class",
            ha='center', va='center', fontsize=font_size, fontweight='bold')
    ax.text(0.5 - max_r - label_w * 1.6, 0.5, "Actual Class",
            ha='center', va='center', rotation=90,
            fontsize=font_size, fontweight='bold')


def add_metric_grid(ax, max_r, tp, fp, fn, tn, line_width=1.6, font_size=fz(12)):
    """
    底部两行灰框标签：
    第1行：左 Precision | 右 Recall（上边框与混淆矩阵底边重合）
    第2行：整行 F1-Score（紧贴其下）
    风格与 add_axis_labels 一致（灰底 + 黑边 + 加粗字体）
    """
    # === 计算指标 ===
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) != 0 else 0

    # === 文本 ===
    t_prec = f"Precision={precision * 100:.1f}%"
    t_recal = f"Recall={recall * 100:.1f}%"
    t_f1 = f"F1-Score={f1 * 100:.1f}%"

    # === 尺寸参数 ===
    row_h = max_r * 0.12  # 0.048, 两行刚好落在 0~1 里
    x_left = 0.5 - max_r
    mid_x = 0.5
    y_bottom_matrix = 0.5 - max_r  # 主矩阵底边 y 坐标

    # === 第1行 Precision/Recall 的 y 坐标 ===
    # 使其上边框正好与矩阵底边对齐
    y_row1_top = y_bottom_matrix
    y_row1_center = y_row1_top - row_h / 2

    # === 第2行 F1 的 y 坐标（紧贴第1行）===
    y_row2_top = y_row1_top - row_h
    y_row2_center = y_row2_top - row_h / 2

    # ---------- 第1行：Precision ----------
    ax.add_patch(patches.Rectangle(
        (x_left, y_row1_top - row_h),  # 左下角
        max_r, row_h,
        facecolor="#F2F2F2", edgecolor="black", linewidth=line_width))
    ax.text(x_left + max_r / 2, y_row1_center, t_prec,
            ha="center", va="center", fontsize=font_size, weight='bold', color='black', fontfamily='Calibri')

    # ---------- 第1行：Recall ----------
    ax.add_patch(patches.Rectangle(
        (mid_x, y_row1_top - row_h),
        max_r, row_h,
        facecolor="#F2F2F2", edgecolor="black", linewidth=line_width))
    ax.text(mid_x + max_r / 2, y_row1_center, t_recal,
            ha="center", va="center", fontsize=font_size,
            weight="bold", color="black", fontfamily="Arial")

    # ---------- 第2行：F1 ----------
    ax.add_patch(patches.Rectangle(
        (x_left, y_row2_top - row_h),  # 左下角
        2 * max_r, row_h,
        facecolor="#F2F2F2", edgecolor="black", linewidth=line_width))
    ax.text(mid_x, y_row2_center, t_f1,
            ha="center", va="center", fontsize=font_size,
            weight="bold", color="black", fontfamily="Arial")


def create_confusion_matrix_plot(tp, fp, fn, tn, title="Confusion Matrix Visualization"):
    setup_plot_style()

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_aspect('equal')

    max_r = 0.4

    # 百分比 + 半径
    percentages, radii = calculate_percentages_and_radii(tp, fp, fn, tn, max_r)

    # === 使用马卡龙渐变 ===
    cmap, norm_percentages = create_macaron_cmap(percentages)

    # 扇形
    draw_sectors(ax, (0.5, 0.5), radii, cmap, norm_percentages)

    # 边框、文字、指标条
    add_quadrant_borders(ax, max_r)
    add_sector_center_labels(ax, max_r)
    add_labels(ax, max_r, percentages)
    add_axis_labels(ax, max_r)
    add_metric_grid(ax, max_r, tp, fp, fn, tn)

    sm = ScalarMappable(norm=Normalize(vmin=0, vmax=100), cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.050, pad=CB_PAD)

    # 标题（色条标签）
    cbar.set_label("Percentage (%)", fontsize=CB_LABEL_FZ,
                   fontweight="bold", labelpad=10)

    # 刻度（数字 0–100）
    cbar.set_ticks([0, 20, 40, 60, 80, 100])
    cbar.ax.tick_params(labelsize=CB_TICK_FZ, width=CB_LINE_W, length=6)

    # 边框加粗
    cbar.outline.set_linewidth(CB_LINE_W)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.title(title, fontsize=fz(20), fontweight='bold', pad=20)
    plt.tight_layout()
    return fig, ax



def process_single_csv(file_path, output_dir, label_name=None):
    """
    处理单个CSV文件
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)

        # 获取文件名（不含扩展名）
        file_name = os.path.splitext(os.path.basename(file_path))[0]

        if label_name:
            # 处理单个标签
            if label_name not in df['label'].values:
                print(f"警告：标签 '{label_name}' 在文件 {file_path} 中不存在")
                return

            label_data = df[df['label'] == label_name].iloc[0]
            tp = label_data['tp']
            fp = label_data['fp']
            fn = label_data['fn']
            tn = label_data['tn']

            # 创建图表
            title = f"{label_name} - {file_name}"
            fig, ax = create_confusion_matrix_plot(tp, fp, fn, tn, title)

            # 保存图表
            output_path = os.path.join(output_dir, f"{file_name}_{label_name}_confusion_matrix.png")
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"已保存: {output_path}")

        else:
            # 处理所有标签
            for _, row in df.iterrows():
                label_name = row['label']
                tp = row['tp']
                fp = row['fp']
                fn = row['fn']
                tn = row['tn']

                # 创建图表
                title = f"{label_name} - {file_name}"
                fig, ax = create_confusion_matrix_plot(tp, fp, fn, tn, title)

                # 保存图表
                output_path = os.path.join(output_dir, f"{file_name}_{label_name}_confusion_matrix.png")
                fig.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f"已保存: {output_path}")

    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")


def process_all_folds(base_path, output_dir, specific_label=None):
    """
    处理所有fold的CSV文件
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 查找所有CSV文件
    csv_files = glob.glob(os.path.join(base_path, "*.csv"))

    if not csv_files:
        print(f"在路径 {base_path} 中未找到CSV文件")
        return

    print(f"找到 {len(csv_files)} 个CSV文件")

    # 处理每个文件
    for csv_file in csv_files:
        print(f"\n处理文件: {csv_file}")
        process_single_csv(csv_file, output_dir, specific_label)


def create_summary_plot(csv_files, output_dir, label_name):
    """
    为指定标签创建所有fold的汇总图
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))  # 2行3列，最后一个位置空着或放汇总
    axes = axes.flatten()

    all_data = []

    for i, csv_file in enumerate(csv_files):
        if i >= 5:  # 最多显示5个fold
            break

        try:
            df = pd.read_csv(csv_file)
            file_name = os.path.splitext(os.path.basename(csv_file))[0]

            if label_name in df['label'].values:
                label_data = df[df['label'] == label_name].iloc[0]
                tp = label_data['tp']
                fp = label_data['fp']
                fn = label_data['fn']
                tn = label_data['tn']

                # 在子图中绘制
                ax = axes[i]
                ax.set_aspect('equal')
                max_r = 0.4

                percentages, radii = calculate_percentages_and_radii(tp, fp, fn, tn, max_r)
                cmap, norm_percentages = create_macaron_cmap(percentages)


                draw_sectors(ax, (0.5, 0.5), radii, cmap, norm_percentages)
                add_quadrant_borders(ax, max_r)
                add_sector_center_labels(ax, max_r)
                add_labels(ax, max_r, percentages)

                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
                ax.set_title(f"{file_name}\nTP:{tp}, FP:{fp}, FN:{fn}, TN:{tn}", fontsize=fz(12))

                all_data.append((tp, fp, fn, tn))

        except Exception as e:
            print(f"处理文件 {csv_file} 时出错: {e}")

    # 计算平均值并在最后一个子图中显示
    if all_data:
        avg_tp = np.mean([d[0] for d in all_data])
        avg_fp = np.mean([d[1] for d in all_data])
        avg_fn = np.mean([d[2] for d in all_data])
        avg_tn = np.mean([d[3] for d in all_data])

        ax = axes[5]
        ax.set_aspect('equal')
        max_r = 0.4

        percentages, radii = calculate_percentages_and_radii(avg_tp, avg_fp, avg_fn, avg_tn, max_r)
        cmap, norm_percentages = create_macaron_cmap(percentages)


        draw_sectors(ax, (0.5, 0.5), radii, cmap, norm_percentages)
        add_quadrant_borders(ax, max_r)
        add_sector_center_labels(ax, max_r)
        add_labels(ax, max_r, percentages)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title(f"Average across all folds\nTP:{avg_tp:.1f}, FP:{avg_fp:.1f}, FN:{avg_fn:.1f}, TN:{avg_tn:.1f}",
                     fontsize=fz(10))

    plt.suptitle(f"Confusion Matrix Visualization for '{label_name}'", fontsize=fz(18), fontweight='bold')
    plt.tight_layout()

    # 保存汇总图
    output_path = os.path.join(output_dir, f"summary_{label_name}_confusion_matrices.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"已保存汇总图: {output_path}")


def calculate_fold_metrics_average(csv_files, label_name):
    """
    计算五折的平均指标，保持TP、FP、FN、TN为整数
    """
    all_tp, all_fp, all_fn, all_tn = [], [], [], []
    all_precision, all_recall, all_f1 = [], [], []

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if label_name in df['label'].values:
                label_data = df[df['label'] == label_name].iloc[0]
                tp = label_data['tp']
                fp = label_data['fp']
                fn = label_data['fn']
                tn = label_data['tn']

                all_tp.append(tp)
                all_fp.append(fp)
                all_fn.append(fn)
                all_tn.append(tn)

                # 计算每个fold的指标
                precision = tp / (tp + fp) if (tp + fp) != 0 else 0
                recall = tp / (tp + fn) if (tp + fn) != 0 else 0
                f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) != 0 else 0

                all_precision.append(precision)
                all_recall.append(recall)
                all_f1.append(f1)

        except Exception as e:
            print(f"处理文件 {csv_file} 时出错: {e}")

    if not all_tp:
        raise ValueError(f"在提供的CSV文件中未找到标签 '{label_name}'")

    # 计算平均值和标准差
    avg_tp = int(np.round(np.mean(all_tp)))
    avg_fp = int(np.round(np.mean(all_fp)))
    avg_fn = int(np.round(np.mean(all_fn)))
    avg_tn = int(np.round(np.mean(all_tn)))

    avg_precision = np.mean(all_precision)
    avg_recall = np.mean(all_recall)
    avg_f1 = np.mean(all_f1)

    std_precision = np.std(all_precision)
    std_recall = np.std(all_recall)
    std_f1 = np.std(all_f1)

    metrics_summary = {
        'tp': avg_tp, 'fp': avg_fp, 'fn': avg_fn, 'tn': avg_tn,
        'precision': {'mean': avg_precision, 'std': std_precision},
        'recall': {'mean': avg_recall, 'std': std_recall},
        'f1': {'mean': avg_f1, 'std': std_f1},
        'n_folds': len(all_tp)
    }

    return metrics_summary


def create_average_confusion_matrix_plot(csv_files, label_name, output_dir):
    """
    创建五折平均结果的混淆矩阵图
    """
    # 计算平均指标
    metrics = calculate_fold_metrics_average(csv_files, label_name)

    # 设置绘图样式
    setup_plot_style()

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_aspect('equal')

    max_r = 0.4

    # 使用平均的TP、FP、FN、TN计算百分比和半径
    percentages, radii = calculate_percentages_and_radii(
        metrics['tp'], metrics['fp'], metrics['fn'], metrics['tn'], max_r
    )

    # 使用修改后的马卡龙渐变
    cmap, norm_percentages = create_macaron_cmap(percentages)

    # 绘制扇形
    draw_sectors(ax, (0.5, 0.5), radii, cmap, norm_percentages)

    # 添加边框、文字、指标条
    add_quadrant_borders(ax, max_r)
    add_sector_center_labels(ax, max_r)
    add_labels(ax, max_r, percentages)
    add_axis_labels(ax, max_r)

    # 修改指标显示，添加标准差
    add_metric_grid_with_std(ax, max_r, metrics)

    sm = ScalarMappable(norm=Normalize(vmin=0, vmax=100), cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.050, pad=CB_PAD)
    cbar.set_label("Percentage (%)", fontsize=CB_LABEL_FZ,
                   fontweight="bold", labelpad=10)
    cbar.set_ticks([0, 20, 40, 60, 80, 100])
    cbar.ax.tick_params(labelsize=CB_TICK_FZ, width=CB_LINE_W, length=6)
    cbar.outline.set_linewidth(CB_LINE_W)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # 标题包含标准差信息
    title = (f"{label_name} - 5-Fold Average\n"
)

    plt.title(title, fontsize=fz(14), fontweight='bold', pad=20)
    plt.tight_layout()

    # 保存图像
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"average_{label_name}_confusion_matrix.png")
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"已保存平均结果: {output_path}")

    return metrics

def get_union_labels(csv_files):
    """
    从五折 CSV 中汇总全部出现过的标签名（去重、排序）。
    要求每个 CSV 至少有列: label, tp, fp, fn, tn
    """
    labels = set()
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if 'label' not in df.columns:
                print(f"[warn] {csv_file} 缺少 'label' 列，已跳过")
                continue
            # 防止空值/非字符串
            labels.update(df['label'].dropna().astype(str).tolist())
        except Exception as e:
            print(f"[warn] 读取 {csv_file} 出错: {e}")
    return sorted(labels)


def create_average_plots_for_all_labels(csv_files, output_dir):
    """
    为“所有标签”批量生成五折平均图，并输出一个汇总CSV（均值±标准差）。
    - 图像逐标签保存在 {output_dir}/avg_all_labels 下
    - 指标汇总表保存为 {output_dir}/avg_all_labels/fivefold_avg_metrics.csv
    """
    # 1) 收集标签
    all_labels = get_union_labels(csv_files)
    if not all_labels:
        print("[error] 没找到任何标签。请检查CSV是否包含 'label' 列。")
        return

    # 2) 输出目录
    out_dir = os.path.join(output_dir, "avg_all_labels")
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for i, label in enumerate(all_labels, 1):
        try:
            metrics = create_average_confusion_matrix_plot(csv_files, label, out_dir)
            rows.append({
                "label": label,
                "tp": metrics["tp"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "tn": metrics["tn"],
                "precision_mean": metrics["precision"]["mean"],
                "precision_std":  metrics["precision"]["std"],
                "recall_mean":    metrics["recall"]["mean"],
                "recall_std":     metrics["recall"]["std"],
                "f1_mean":        metrics["f1"]["mean"],
                "f1_std":         metrics["f1"]["std"],
                "n_folds":        metrics["n_folds"],
            })
            print(f"[{i}/{len(all_labels)}] Done: {label}")
        except Exception as e:
            print(f"[warn] 处理标签 {label} 出错: {e}")

    # 3) 汇总表
    if rows:
        df_sum = pd.DataFrame(rows)
        # 百分比更友好：也可以另存一个“百分数形式”的备份
        df_sum.to_csv(os.path.join(out_dir, "fivefold_avg_metrics.csv"), index=False)
        print(f"已输出指标汇总: {os.path.join(out_dir, 'fivefold_avg_metrics.csv')}")
    else:
        print("[warn] 没有任何标签成功输出。")


def add_metric_grid_with_std(ax, max_r, metrics, line_width=1.6, font_size=fz(12)):
    """
    修改后的指标显示函数，包含标准差
    """
    precision = metrics['precision']['mean']
    recall = metrics['recall']['mean']
    f1 = metrics['f1']['mean']

    precision_std = metrics['precision']['std']
    recall_std = metrics['recall']['std']
    f1_std = metrics['f1']['std']

    # 文本（包含标准差）
    t_prec = f"Precision={precision * 100:.1f}% ± {precision_std * 100:.1f}%"
    t_recal = f"Recall={recall * 100:.1f}% ± {recall_std * 100:.1f}%"
    t_f1 = f"F1-Score={f1 * 100:.1f}% ± {f1_std * 100:.1f}%"

    # 尺寸参数
    row_h = max_r * 0.12  # 0.048, 两行刚好落在 0~1 里
    x_left = 0.5 - max_r
    mid_x = 0.5
    y_bottom_matrix = 0.5 - max_r

    # 第1行 Precision/Recall
    y_row1_top = y_bottom_matrix
    y_row1_center = y_row1_top - row_h / 2

    # 第2行 F1
    y_row2_top = y_row1_top - row_h
    y_row2_center = y_row2_top - row_h / 2

    # Precision
    ax.add_patch(patches.Rectangle(
        (x_left, y_row1_top - row_h),
        max_r, row_h,
        facecolor="#F2F2F2", edgecolor="black", linewidth=line_width))
    ax.text(x_left + max_r / 2, y_row1_center, t_prec,
            ha="center", va="center", fontsize=font_size - 1, weight='bold',
            color='black', fontfamily='Calibri')

    # Recall
    ax.add_patch(patches.Rectangle(
        (mid_x, y_row1_top - row_h),
        max_r, row_h,
        facecolor="#F2F2F2", edgecolor="black", linewidth=line_width))
    ax.text(mid_x + max_r / 2, y_row1_center, t_recal,
            ha="center", va="center", fontsize=font_size - 1,
            weight="bold", color="black", fontfamily="Arial")

    # F1
    ax.add_patch(patches.Rectangle(
        (x_left, y_row2_top - row_h),
        2 * max_r, row_h,
        facecolor="#F2F2F2", edgecolor="black", linewidth=line_width))
    ax.text(mid_x, y_row2_center, t_f1,
            ha="center", va="center", fontsize=font_size - 1,
            weight="bold", color="black", fontfamily="Arial")


# 主程序
if __name__ == "__main__":
    # 设置路径
    base_path = r"..."
    output_dir = r"..."

    # 获取所有CSV文件
    csv_files = [
        r"...\fold0_labelwise.csv",
        r"...\fold1_labelwise.csv",
        r"...\fold2_labelwise.csv",
        r"...\fold3_labelwise.csv",
        r"...\fold4_labelwise.csv"
    ]

    # 选择处理模式
    print("请选择处理模式:")
    print("1. 处理所有标签的所有fold")
    print("2. 处理特定标签的所有fold")
    print("3. 为特定标签创建汇总图")
    print("4. 为特定标签创建五折平均图（推荐）")
    print("5. 为全部标签创建五折平均图（批量，自动生成汇总表）")

    choice = input("请输入选择 (1/2/3/4/5): ").strip()

    if choice == "1":
        # 处理所有标签的所有fold
        process_all_folds(base_path, output_dir)

    elif choice == "2":
        # 处理特定标签的所有fold
        label_name = input("请输入标签名称 (例如: alcoholic, aldehydic等): ").strip()
        process_all_folds(base_path, output_dir, label_name)

    elif choice == "3":
        # 为特定标签创建汇总图
        label_name = input("请输入标签名称 (例如: alcoholic, aldehydic等): ").strip()
        create_summary_plot(csv_files, output_dir, label_name)

    elif choice == "4":
        # 为特定标签创建五折平均图
        label_name = input("请输入标签名称 (例如: alcoholic, aldehydic等): ").strip()
        metrics = create_average_confusion_matrix_plot(csv_files, label_name, output_dir)
        print(f"\n五折平均结果统计:")
        print(f"TP: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']}, TN: {metrics['tn']}")
        print(f"Precision: {metrics['precision']['mean'] * 100:.1f}% ± {metrics['precision']['std'] * 100:.1f}%")
        print(f"Recall: {metrics['recall']['mean'] * 100:.1f}% ± {metrics['recall']['std'] * 100:.1f}%")
        print(f"F1-Score: {metrics['f1']['mean'] * 100:.1f}% ± {metrics['f1']['std'] * 100:.1f}%")

    elif choice == "5":
        # 为全部标签批量创建五折平均图 + 汇总表
        create_average_plots_for_all_labels(csv_files, output_dir)

    else:
        print("无效选择")

    print("\n处理完成！")
