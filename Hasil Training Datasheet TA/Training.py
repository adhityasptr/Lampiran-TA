from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(
    data="C:/File Personal/Documents/kebutuhan TA/kodingan/Datasheet TA/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device="cpu",                   
    workers=2,
    patience=20,
    optimizer="SGD",
    verbose=True,
    save=True,
    save_period=10,
    cache=True
)