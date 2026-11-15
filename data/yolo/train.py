from ultralytics import YOLO

if __name__ == '__main__':
    # Load a pretrained YOLOv8 model
    model = YOLO('yolo11m.pt')

    # Train the model on a custom dataset
    model.train(data='data_json/dataset.yaml', epochs=400, imgsz=1024,)

    # Save the trained model