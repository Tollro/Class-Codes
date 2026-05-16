from ultralytics import YOLO

model = YOLO("myfish_training_docs/epoches100/weights/last.pt")

model.predict(source="D:/AIModels/myfish_rect_Dataset/images/test/image00700.png", save=True, project=r'D:\AIModels\ultralytics-main\runs\detect\myfish_rect_epoches100', stream=False)
