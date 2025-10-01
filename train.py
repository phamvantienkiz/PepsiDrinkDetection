from ultralytics import YOLO



if __name__ == '__main__':
    # Load a model
    model = YOLO("yolo11n.pt")

    # Train the model
    results = model.train(data="data.yaml", epochs=40, imgsz=640, batch=8)