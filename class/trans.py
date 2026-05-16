import json
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom

def prettify(elem):
    """返回格式化的 XML 字符串"""
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="\t")

def json_to_voc_xml(json_path, output_xml_path, folder="train", base_path="D:\\AIModels\\FruitDataset_original"):
    """
    将单个 JSON 标注文件转换为 VOC XML 格式

    :param json_path: 输入的 JSON 文件路径
    :param output_xml_path: 输出的 XML 文件路径
    :param folder: 数据集文件夹名（如 train/val）
    :param base_path: 图像完整路径的前缀
    """
    # 读取 JSON 文件
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 创建根元素
    root = ET.Element('annotation')

    # folder
    folder_elem = ET.SubElement(root, 'folder')
    folder_elem.text = folder

    # filename
    filename_elem = ET.SubElement(root, 'filename')
    filename_elem.text = data.get('imagePath', '')

    # path
    path_elem = ET.SubElement(root, 'path')
    path_elem.text = os.path.join(base_path, folder, data.get('imagePath', ''))

    # source
    source_elem = ET.SubElement(root, 'source')
    database_elem = ET.SubElement(source_elem, 'database')
    database_elem.text = 'Unknown'

    # size
    size_elem = ET.SubElement(root, 'size')
    width_elem = ET.SubElement(size_elem, 'width')
    width_elem.text = str(data.get('imageWidth', 0))
    height_elem = ET.SubElement(size_elem, 'height')
    height_elem.text = str(data.get('imageHeight', 0))
    depth_elem = ET.SubElement(size_elem, 'depth')
    depth_elem.text = '3'  # 默认 RGB 图像深度

    # segmented
    segmented_elem = ET.SubElement(root, 'segmented')
    segmented_elem.text = '0'

    # 处理每一个标注对象
    for shape in data.get('shapes', []):
        # 只处理矩形框
        if shape.get('shape_type') != 'rectangle':
            continue

        label = shape.get('label', '')
        points = shape.get('points', [])

        if len(points) != 2:
            continue

        # 左上角和右下角坐标
        (x1, y1), (x2, y2) = points
        xmin = int(round(min(x1, x2)))
        ymin = int(round(min(y1, y2)))
        xmax = int(round(max(x1, x2)))
        ymax = int(round(max(y1, y2)))

        # 创建 object 元素
        obj_elem = ET.SubElement(root, 'object')

        name_elem = ET.SubElement(obj_elem, 'name')
        name_elem.text = label

        pose_elem = ET.SubElement(obj_elem, 'pose')
        pose_elem.text = 'Unspecified'

        truncated_elem = ET.SubElement(obj_elem, 'truncated')
        truncated_elem.text = '0'

        difficult_elem = ET.SubElement(obj_elem, 'difficult')
        difficult_elem.text = '0'

        bndbox_elem = ET.SubElement(obj_elem, 'bndbox')
        xmin_elem = ET.SubElement(bndbox_elem, 'xmin')
        xmin_elem.text = str(xmin)
        ymin_elem = ET.SubElement(bndbox_elem, 'ymin')
        ymin_elem.text = str(ymin)
        xmax_elem = ET.SubElement(bndbox_elem, 'xmax')
        xmax_elem.text = str(xmax)
        ymax_elem = ET.SubElement(bndbox_elem, 'ymax')
        ymax_elem.text = str(ymax)

    # 生成格式化的 XML 并写入文件
    xml_str = prettify(root)
    with open(output_xml_path, 'w', encoding='utf-8') as f:
        f.write(xml_str)

def batch_convert(input_dir, output_dir, folder="train", base_path="D:\\AIModels\\FruitDataset_original"):
    """
    批量转换文件夹内所有 JSON 文件为 VOC XML

    :param input_dir: 存放 JSON 标注文件的文件夹路径
    :param output_dir: 输出 XML 文件的文件夹路径
    :param folder: 数据集子目录名称（如 train/val）
    :param base_path: 图像根目录前缀
    """
    if not os.path.exists(input_dir):
        print(f"错误：输入目录不存在 - {input_dir}")
        return

    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有 JSON 文件
    json_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.json')]
    total = len(json_files)

    if total == 0:
        print(f"警告：在 {input_dir} 中没有找到 JSON 文件")
        return

    print(f"找到 {total} 个 JSON 文件，开始转换...")

    for idx, filename in enumerate(json_files, 1):
        json_path = os.path.join(input_dir, filename)
        xml_filename = os.path.splitext(filename)[0] + '.xml'
        xml_path = os.path.join(output_dir, xml_filename)

        try:
            json_to_voc_xml(json_path, xml_path, folder, base_path)
            print(f"[{idx}/{total}] 转换成功: {filename} -> {xml_filename}")
        except Exception as e:
            print(f"[{idx}/{total}] 转换失败: {filename} - 错误: {e}")

    print("批量转换完成！")

# 使用示例
if __name__ == "__main__":
    # ========== 请根据实际情况修改以下路径 ==========
    input_directory = r"D:\AI_train_Datasets\FruitsDataset\json"   # JSON 文件所在文件夹
    output_directory = r"D:\AI_train_Datasets\FruitsDataset\Crawler"           # 输出 XML 的文件夹
    dataset_folder = "train"                           # train / val
    image_root_path = r"D:\AI_train_Datasets\FruitsDataset\Crawler"  # 图像根目录
    # =============================================

    batch_convert(
        input_dir=input_directory,
        output_dir=output_directory,
        folder=dataset_folder,
        base_path=image_root_path
    )