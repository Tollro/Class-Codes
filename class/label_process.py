import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

# ==================== 配置区 ====================
# 目标类别列表（顺序决定 class_id）
TARGET_CLASSES = [
    'Apple',
    'Watermelon',
    'Orange',
    'Banana',
    'Strawberry',
    'Kiwifruit',
    'Pineapple',
    'Durian',
    'Pitaya',
]

# 类别名归一化映射（处理大小写、别名）
CANONICAL_MAP = {name.lower(): name for name in TARGET_CLASSES}
CANONICAL_MAP.update({
    'kiwi': 'Kiwifruit',
    'dragonfruit': 'Pitaya',
    'pitaya': 'Pitaya',
    'strawberry': 'Strawberry',
    'water melon': 'Watermelon',
    'watermelon': 'Watermelon',
})

# class_id 查询字典
CLASS2ID = {name: idx for idx, name in enumerate(TARGET_CLASSES)}


def normalize_label(name: str) -> str:
    """将原始标签名归一化到目标类别，无法匹配返回 None"""
    if not name:
        return None
    text = name.strip()
    return CANONICAL_MAP.get(text.lower())


def convert_voc_to_yolo(xml_path: Path, img_width: int, img_height: int):
    """
    解析 VOC XML，返回 YOLO 格式的标注列表。
    每项格式：class_id x_center y_center width height（归一化）
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    yolo_anns = []

    # 如果 XML 内无 size 信息，则使用传入的尺寸
    size_node = root.find('size')
    if size_node is not None:
        w_node = size_node.find('width')
        h_node = size_node.find('height')
        if w_node is not None and h_node is not None:
            img_width = int(w_node.text)
            img_height = int(h_node.text)

    for obj in root.findall('object'):
        name_el = obj.find('name')
        if name_el is None or not name_el.text:
            continue

        canonical = normalize_label(name_el.text)
        if canonical is None:
            print(f"  警告：未知类别 '{name_el.text}' 在文件 {xml_path.name}，已跳过")
            continue

        class_id = CLASS2ID[canonical]

        bndbox = obj.find('bndbox')
        if bndbox is None:
            continue

        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)

        # 转换为 YOLO 格式（中心点 + 宽高，归一化）
        x_center = (xmin + xmax) / 2.0 / img_width
        y_center = (ymin + ymax) / 2.0 / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height

        # 防止越界
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        width = max(0.0, min(1.0, width))
        height = max(0.0, min(1.0, height))

        yolo_anns.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    return yolo_anns


def process_dataset(xml_root: Path, yolo_root: Path):
    """
    处理 xml_root 下的 train 和 val 子文件夹，
    将转换后的 txt 保存到 yolo_root/labels/train 和 yolo_root/labels/val
    """
    subsets = ['train', 'val']
    unknown_labels = set()
    total_files = 0

    for subset in subsets:
        xml_subdir = xml_root / subset
        if not xml_subdir.exists():
            print(f"跳过不存在的子目录: {xml_subdir}")
            continue

        txt_outdir = yolo_root / 'labels' / subset
        txt_outdir.mkdir(parents=True, exist_ok=True)

        xml_files = list(xml_subdir.glob('*.xml'))
        print(f"\n处理 {subset} 集：找到 {len(xml_files)} 个 XML 文件")

        for xml_path in xml_files:
            total_files += 1
            # 默认尺寸（若 XML 中缺失则使用）
            default_w, default_h = 800, 768
            yolo_lines = convert_voc_to_yolo(xml_path, default_w, default_h)

            # 写入对应的 txt 文件
            txt_path = txt_outdir / (xml_path.stem + '.txt')
            with open(txt_path, 'w', encoding='utf-8') as f:
                if yolo_lines:
                    f.write('\n'.join(yolo_lines))

    # 输出未知类别汇总
    if unknown_labels:
        print("\n⚠️ 发现未在目标类别中的标签名：")
        for name in sorted(unknown_labels):
            print(f"  - {name}")
    else:
        print("\n✅ 所有类别均已成功映射。")

    print(f"\n转换完成！共处理 {total_files} 个 XML 文件。")


def main():
    # parser = argparse.ArgumentParser(description="VOC XML 转 YOLO txt 格式")
    # parser.add_argument('--xml_root', type=str, required=True,
    #                     help='包含 train/ 和 val/ 子文件夹的 VOC XML 根目录')
    # parser.add_argument('--yolo_root', type=str, required=True,
    #                     help='YOLO 数据集根目录（例如 D:/AI_train_Datasets/FruitsDataset/combine）')
    # args = parser.parse_args()

    # xml_root = Path(args.xml_root)
    # yolo_root = Path(args.yolo_root)

    xml_root = Path("D:/AI_train_Datasets/FruitsDataset/combine/labels")  # 存放 VOC XML 的根目录
    yolo_root = Path("D:/AI_train_Datasets/FruitsDataset/combine")

    if not xml_root.exists():
        raise FileNotFoundError(f"XML 根目录不存在: {xml_root}")

    print("类别名称列表（用于 data.yaml 的 names 字段）：")
    for idx, name in enumerate(TARGET_CLASSES):
        print(f"  {idx}: {name}")
    print(f"类别总数 nc: {len(TARGET_CLASSES)}\n")

    process_dataset(xml_root, yolo_root)


if __name__ == '__main__':
    main()