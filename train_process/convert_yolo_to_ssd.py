import os
import yaml
from PIL import Image

def convert_yolo_to_ssd(yaml_path, output_train_txt, output_val_txt):
    # 读取 YAML 配置文件
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    path = config['path']
    print(f"Dataset root path: {path}")
    train_images = config['train'][0]  # 'images/train'
    val_images = config['val']  # 'images/val'
    train_labels = train_images.replace('images', 'labels')
    val_labels = val_images.replace('images', 'labels')

    print(f"Train images dir: {os.path.join(path, train_images)}")
    print(f"Train labels dir: {os.path.join(path, train_labels)}")
    print(f"Val images dir: {os.path.join(path, val_images)}")
    print(f"Val labels dir: {os.path.join(path, val_labels)}")

    # 获取类别数量和名称（虽然不需要，但可以验证）
    nc = config['nc']
    names = config['names']

    def process_dataset(image_dir, label_dir, output_txt):
        print(f"Processing {output_txt}...")
        if not os.path.exists(label_dir):
            print(f"Error: Label directory {label_dir} does not exist.")
            return
        if not os.path.exists(image_dir):
            print(f"Error: Image directory {image_dir} does not exist.")
            return
        
        label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
        print(f"Found {len(label_files)} label files in {label_dir}")
        
        with open(output_txt, 'w', encoding='gbk') as out_f:
            for label_file in label_files:
                label_path = os.path.join(label_dir, label_file)
                image_file = label_file.replace('.txt', '.jpg')  # 假设图片是 .jpg，如果是 .png 需调整
                image_path = os.path.join(image_dir, image_file)

                if not os.path.exists(image_path):
                    print(f"Warning: Image {image_path} not found, skipping.")
                    continue

                # 获取图片尺寸
                with Image.open(image_path) as img:
                    img_width, img_height = img.size

                # 读取标签
                with open(label_path, 'r') as f:
                    lines = f.readlines()

                boxes = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])

                    # 转换为绝对坐标
                    x1 = int((x_center - width / 2) * img_width)
                    y1 = int((y_center - height / 2) * img_height)
                    x2 = int((x_center + width / 2) * img_width)
                    y2 = int((y_center + height / 2) * img_height)

                    boxes.append(f"{x1},{y1},{x2},{y2},{class_id}")

                if boxes:
                    # 写入一行：图片路径 空格分隔的boxes
                    out_f.write(f"{image_path} {' '.join(boxes)}\n")
                    print(f"Processed {label_file}: {len(boxes)} boxes")

    # 处理训练集
    train_image_dir = os.path.join(path, train_images)
    train_label_dir = os.path.join(path, train_labels)
    process_dataset(train_image_dir, train_label_dir, output_train_txt)

    # 处理验证集
    val_image_dir = os.path.join(path, val_images)
    val_label_dir = os.path.join(path, val_labels)
    process_dataset(val_image_dir, val_label_dir, output_val_txt)

    print("Conversion completed.")
    print(f"Train annotations saved to: {output_train_txt}")
    print(f"Val annotations saved to: {output_val_txt}")

    # 处理训练集
    train_image_dir = os.path.join(path, train_images)
    train_label_dir = os.path.join(path, train_labels)
    process_dataset(train_image_dir, train_label_dir, output_train_txt)

    # 处理验证集
    val_image_dir = os.path.join(path, val_images)
    val_label_dir = os.path.join(path, val_labels)
    process_dataset(val_image_dir, val_label_dir, output_val_txt)

    print("Conversion completed.")
    print(f"Train annotations saved to: {output_train_txt}")
    print(f"Val annotations saved to: {output_val_txt}")

if __name__ == "__main__":
    yaml_path = 'train_process/fruit_detect.yaml'  # 当前目录下的 YAML 文件
    output_train_txt = '2007_train.txt'
    output_val_txt = '2007_val.txt'
    convert_yolo_to_ssd(yaml_path, output_train_txt, output_val_txt)