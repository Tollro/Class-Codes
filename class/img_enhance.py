import random
import os
from pathlib import Path
import cv2
import numpy as np
import xml.etree.ElementTree as ET

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

# 硬编码路径，请根据需要修改
IMAGE_DIR = Path(r"D:\AI_train_Datasets\FruitsDataset\Crawler\select")
LABEL_DIR = Path(r"D:\AI_train_Datasets\FruitsDataset\Crawler\train_labels")
OUTPUT_IMAGE_DIR = Path(r"D:\AI_train_Datasets\FruitsDataset\Crawler\aug_images")
OUTPUT_LABEL_DIR = LABEL_DIR  # 如果想分开，请修改这里
AUG_SUFFIX = "_aug3"  # 变换后图像的后缀


def find_images(folder: Path):
    """遍历文件夹，返回所有图片文件的路径"""
    for path in folder.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def load_xml(xml_path: Path):
    """加载 XML 文件"""
    return ET.parse(xml_path)


def update_xml_for_image(tree: ET.ElementTree, new_filename: str, new_path: str,
                         new_width: int, new_height: int):
    """更新 XML 中的 filename、path 和 size 信息"""
    root = tree.getroot()
    filename = root.find('filename')
    if filename is not None:
        filename.text = new_filename
    path_el = root.find('path')
    if path_el is not None:
        path_el.text = new_path

    size = root.find('size')
    if size is not None:
        width_el = size.find('width')
        height_el = size.find('height')
        if width_el is not None:
            width_el.text = str(new_width)
        if height_el is not None:
            height_el.text = str(new_height)

    return root


def transform_bndbox(bndbox: ET.Element, width: int, height: int, transform_type: str):
    """根据变换类型调整边界框坐标"""
    xmin = int(bndbox.find('xmin').text)
    ymin = int(bndbox.find('ymin').text)
    xmax = int(bndbox.find('xmax').text)
    ymax = int(bndbox.find('ymax').text)

    if transform_type == 'contrast':
        new_coords = (xmin, ymin, xmax, ymax)
    elif transform_type == 'flip':
        new_coords = (width - xmax, ymin, width - xmin, ymax)
    elif transform_type == 'rotate90':
        # 顺时针旋转90度
        new_coords = (height - ymax, xmin, height - ymin, xmax)
    elif transform_type == 'rotate180':
        new_coords = (width - xmax, height - ymax, width - xmin, height - ymin)
    elif transform_type == 'rotate270':
        # 顺时针旋转270度（或逆时针90度）
        new_coords = (ymin, width - xmax, ymax, width - xmin)
    else:
        raise ValueError(f'Unsupported transform type: {transform_type}')

    bndbox.find('xmin').text = str(new_coords[0])
    bndbox.find('ymin').text = str(new_coords[1])
    bndbox.find('xmax').text = str(new_coords[2])
    bndbox.find('ymax').text = str(new_coords[3])


def apply_contrast(image: np.ndarray, factor: float) -> np.ndarray:
    """调整图像对比度，factor > 1 增强对比度，< 1 减弱对比度"""
    # 转换为 float 进行计算，避免溢出
    img_float = image.astype(np.float32)
    # 对比度调整公式：new = factor * (img - mean) + mean，这里简单使用 factor * img
    # 实际效果类似 PIL 的 ImageEnhance.Contrast
    mean = np.mean(img_float, axis=(0, 1), keepdims=True)
    adjusted = factor * (img_float - mean) + mean
    adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
    return adjusted


def apply_flip(image: np.ndarray) -> np.ndarray:
    """水平翻转图像"""
    return cv2.flip(image, 1)  # 1 表示水平翻转


def apply_rotate(image: np.ndarray, angle: int) -> np.ndarray:
    """旋转图像，angle 为 90、180、270（顺时针）"""
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        raise ValueError("仅支持 90、180、270 度旋转")


def save_xml(tree: ET.ElementTree, output_xml_path: Path):
    """保存 XML 文件"""
    tree.write(output_xml_path, encoding='utf-8', xml_declaration=True)


def augment_image(image_path: Path, xml_path: Path, output_image_dir: Path, output_label_dir: Path):
    """对单张图片进行随机增强，并生成新的图片和 XML"""
    # 使用 OpenCV 读取图片（BGR 格式，不影响后续处理）
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"无法读取图片：{image_path}")
        return

    # 获取原始尺寸
    orig_height, orig_width = image.shape[:2]

    image_name = image_path.stem
    image_ext = image_path.suffix
    xml_tree = load_xml(xml_path)

    # 三种变换方式：对比度增强、水平翻转、顺时针旋转90度
    transformations = [
        ('contrast', apply_contrast(image, 1.5)),
        ('flip', apply_flip(image)),
        ('rotate90', apply_rotate(image, 90)),
    ]

    # 随机选择一种变换
    transform_name, result_image = random.choice(transformations)

    # 生成新的文件名
    new_stem = f'{image_name}{AUG_SUFFIX}'
    new_filename = new_stem + image_ext
    output_image_path = output_image_dir / new_filename
    output_xml_path = output_label_dir / f'{new_stem}.xml'

    # 获取新图像的尺寸
    if transform_name == 'contrast':
        new_width, new_height = orig_width, orig_height
    else:
        new_height, new_width = result_image.shape[:2]  # 注意 OpenCV 形状是 (h, w)

    # 更新 XML 中的图像信息
    new_root = update_xml_for_image(xml_tree, new_filename, str(output_image_path.resolve()),
                                    new_width, new_height)

    # 更新所有目标的边界框
    for obj in new_root.findall('object'):
        bndbox = obj.find('bndbox')
        if bndbox is not None:
            transform_bndbox(bndbox, orig_width, orig_height, transform_name)

    # 保存图像与 XML
    cv2.imwrite(str(output_image_path), result_image)
    save_xml(xml_tree, output_xml_path)


def main():
    image_dir = IMAGE_DIR.resolve()
    label_dir = LABEL_DIR.resolve()
    output_image_dir = OUTPUT_IMAGE_DIR.resolve()
    output_label_dir = OUTPUT_LABEL_DIR.resolve()

    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    for image_path in find_images(image_dir):
        xml_path = label_dir / (image_path.stem + '.xml')
        if not xml_path.exists():
            print(f'警告：未找到标签文件 {xml_path}, 已跳过 {image_path.name}')
            continue
        augment_image(image_path, xml_path, output_image_dir, output_label_dir)
        print(f'已处理：{image_path.name}')

    print('增强完成。输出图片目录：', output_image_dir)
    print('输出XML目录：', output_label_dir)


if __name__ == '__main__':
    main()