import os
import xml.etree.ElementTree as ET
from xml.dom import minidom

from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont


def _get_class_name(model, cls_id, class_names):
    if class_names is not None and 0 <= int(cls_id) < len(class_names):
        return class_names[int(cls_id)]
    names = getattr(model, "names", None)
    if names is not None:
        if isinstance(names, dict):
            return str(names.get(int(cls_id), f"class_{int(cls_id)}"))
        return str(names[int(cls_id)])
    return f"class_{int(cls_id)}"


def _build_voc_xml(folder_name, filename, abs_img_path, width, height, depth, objects):
    """objects: list of dicts with keys name, xmin, ymin, xmax, ymax, truncated, difficult"""
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = folder_name
    ET.SubElement(root, "filename").text = filename
    ET.SubElement(root, "path").text = abs_img_path
    src = ET.SubElement(root, "source")
    ET.SubElement(src, "database").text = "Unknown"
    size_el = ET.SubElement(root, "size")
    ET.SubElement(size_el, "width").text = str(int(width))
    ET.SubElement(size_el, "height").text = str(int(height))
    ET.SubElement(size_el, "depth").text = str(int(depth))
    ET.SubElement(root, "segmented").text = "0"

    for obj in objects:
        ob = ET.SubElement(root, "object")
        ET.SubElement(ob, "name").text = obj["name"]
        ET.SubElement(ob, "pose").text = "Unspecified"
        ET.SubElement(ob, "truncated").text = str(int(obj.get("truncated", 0)))
        ET.SubElement(ob, "difficult").text = str(int(obj.get("difficult", 0)))
        bb = ET.SubElement(ob, "bndbox")
        ET.SubElement(bb, "xmin").text = str(int(obj["xmin"]))
        ET.SubElement(bb, "ymin").text = str(int(obj["ymin"]))
        ET.SubElement(bb, "xmax").text = str(int(obj["xmax"]))
        ET.SubElement(bb, "ymax").text = str(int(obj["ymax"]))

    rough = ET.tostring(root, encoding="unicode")
    reparsed = minidom.parseString(rough)
    return reparsed.toprettyxml(indent="\t")


def _truncated_flag(x1, y1, x2, y2, w, h, tol=1):
    return 1 if (x1 <= tol or y1 <= tol or x2 >= w - tol or y2 >= h - tol) else 0


def _draw_preview(im, boxes_xyxy, class_ids, name_fn, colors):
    """boxes_xyxy: list of (x1,y1,x2,y2); name_fn(cls_id) -> str"""
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    for (x1, y1, x2, y2), cls_id in zip(boxes_xyxy, class_ids):
        x1, y1, x2, y2 = map(float, (x1, y1, x2, y2))
        cid = int(cls_id)
        color = colors[cid % len(colors)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = name_fn(cid)
        tb = draw.textbbox((x1, y1), label, font=font)
        draw.rectangle(tb, fill=color)
        draw.text((x1, y1), label, fill=(255, 255, 255), font=font)
    return im


def auto_label_images(
    model_path,
    image_dir,
    output_label_dir,
    class_names=None,
    img_exts=(".jpg", ".jpeg", ".png"),
    save_yolo_txt=True,
    save_voc_xml=True,
    output_xml_dir=None,
    preview_num=5,
    preview_dir=None,
):
    """
    使用 YOLOv8 为文件夹中的图片生成标签：可选 YOLO txt 与 Pascal VOC XML；
    前 preview_num 张保存带框预览图。

    Args:
        model_path: 训练好的 YOLOv8 权重路径。
        image_dir: 图片文件夹。
        output_label_dir: YOLO txt 输出目录（save_yolo_txt 为 True 时使用）。
        class_names: 类别名列表；为 None 时用 model.names。
        img_exts: 支持的扩展名。
        save_yolo_txt: 是否写入 .txt。
        save_voc_xml: 是否写入 VOC 风格 .xml。
        output_xml_dir: XML 目录；默认与 output_label_dir 相同。
        preview_num: 生成预览图的张数（0 表示不生成）。
        preview_dir: 预览图目录；默认在 image_dir 同级的 preview_auto。
    """
    os.makedirs(output_label_dir, exist_ok=True)
    if output_xml_dir is None:
        output_xml_dir = output_label_dir
    if save_voc_xml:
        os.makedirs(output_xml_dir, exist_ok=True)

    if preview_num > 0:
        if preview_dir is None:
            parent = os.path.dirname(os.path.abspath(image_dir)) or os.getcwd()
            preview_dir = os.path.join(parent, "preview_auto")
        preview_dir = os.path.normpath(preview_dir)
        os.makedirs(preview_dir, exist_ok=True)

    model = YOLO(model_path)
    folder_name = os.path.basename(os.path.normpath(image_dir)) or "images"

    image_files = sorted(
        f for f in os.listdir(image_dir) if os.path.splitext(f)[1].lower() in img_exts
    )
    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 128, 255),
        (255, 128, 0),
        (255, 0, 255),
        (0, 255, 255),
        (128, 0, 255),
        (255, 255, 0),
    ]

    for idx, img_name in enumerate(image_files):
        img_path = os.path.join(image_dir, img_name)
        abs_path = os.path.abspath(img_path)

        result = model(img_path)[0]
        boxes = result.boxes

        with Image.open(img_path).convert("RGB") as im:
            w, h = im.size
            depth = 3

        label_lines = []
        objects_xml = []
        boxes_xyxy = []
        class_ids_list = []

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1 = max(0, min(w, x1))
                y1 = max(0, min(h, y1))
                x2 = max(0, min(w, x2))
                y2 = max(0, min(h, y2))
                if x2 <= x1 or y2 <= y1:
                    continue

                cx = (x1 + x2) / 2 / w
                cy = (y1 + y2) / 2 / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                label_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

                cname = _get_class_name(model, cls_id, class_names)
                objects_xml.append(
                    {
                        "name": cname,
                        "xmin": int(round(x1)),
                        "ymin": int(round(y1)),
                        "xmax": int(round(x2)),
                        "ymax": int(round(y2)),
                        "truncated": _truncated_flag(x1, y1, x2, y2, w, h),
                        "difficult": 0,
                    }
                )
                boxes_xyxy.append((x1, y1, x2, y2))
                class_ids_list.append(cls_id)

        stem = os.path.splitext(img_name)[0]

        if save_yolo_txt:
            label_path = os.path.join(output_label_dir, stem + ".txt")
            with open(label_path, "w", encoding="utf-8") as f:
                for ln in label_lines:
                    f.write(ln + "\n")

        if save_voc_xml:
            xml_str = _build_voc_xml(
                folder_name, img_name, abs_path, w, h, depth, objects_xml
            )
            xml_path = os.path.join(output_xml_dir, stem + ".xml")
            with open(xml_path, "w", encoding="utf-8") as f:
                f.write(xml_str)

        if preview_num > 0 and idx < preview_num:
            with Image.open(img_path).convert("RGB") as im_prev:
                name_fn = lambda cid: _get_class_name(model, cid, class_names)
                out_prev = _draw_preview(im_prev.copy(), boxes_xyxy, class_ids_list, name_fn, colors)
                prev_path = os.path.join(preview_dir, f"preview_{stem}.jpg")
                out_prev.save(prev_path, quality=95)
            print(f"Preview -> {prev_path}")

        parts = []
        if save_yolo_txt:
            parts.append("txt")
        if save_voc_xml:
            parts.append("xml")
        fmt = "+".join(parts) if parts else "none"
        print(f"Processed {img_name} -> {fmt}: {len(label_lines)} objects.")


if __name__ == "__main__":
    model_path = "./Fruit-Detection-Model-using-YOLOv8-main/weights/best.pt"
    image_dir = "./watermelon"
    output_label_dir = "./watermelon_labels"
    class_names = None

    auto_label_images(
        model_path,
        image_dir,
        output_label_dir,
        class_names,
        save_yolo_txt=True,
        save_voc_xml=True,
        output_xml_dir="./watermelon_labels",
        preview_num=5,
        preview_dir=None,
    )
