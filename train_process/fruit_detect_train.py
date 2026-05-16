from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    # model = YOLO("yolo11n.yaml")  # build a new model from YAML
    model = YOLO("yolo11n.pt")
    # model = YOLO("runs/detect/fruit_detect_train/yolo11n_fruit_exp1-2/weights/last.pt")  # load a pretrained model (recommended for training)
    # model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML and transfer weights

    # Train the model
    results = model.train(
        data="train_process/fruit_detect.yaml",
        epochs=170,          # 总训练轮次
        # patience=80,         # 早停，连续50轮无提升则停止
        imgsz=640,           # 输入图像尺寸
        batch=24,            # 手动设置批次大小
        device=0,            # GPU设备号
        workers=4,           # 数据加载进程数
        project='fruit_detect_train',
        name='yolo11n_fruit_exp1-2-2',
        # 数据增强参数 (示例：略微调整HSV增强)
        hsv_h=0.02,
        hsv_s=0.6,
        hsv_v=0.4,
        fliplr=0.5,
        # mosaic=1.0,        # 默认为1.0，可注释掉
        # close_mosaic=10,   # 最后10轮关闭马赛克增强，防止波动'
    )
