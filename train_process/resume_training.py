from ultralytics import YOLO

print("Starting resume training...")

if __name__ == '__main__':
    print("Loading model...")
    model = YOLO("runs/detect/fruit_detect_train/yolo11n_fruit_exp1-2-2/weights/last.pt")
    print("Model loaded, starting training...")
    results = model.train(
        resume=True,
        epochs=200,
        data="train_process/fruit_detect.yaml",
        batch=12,          # 从 24 降低到 12（可先试 16，不行再降到 12 或 8）
        workers=4,         # 手动限制为 4，避免自动分配过多
        device=0           # 确保使用单张 GPU（如果有多张）
    )
    print("Training completed.")