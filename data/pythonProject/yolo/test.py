from ultralytics import YOLO

if __name__ == "__main__":
    # Load a pretrained YOLOv8 model
    model = YOLO("best.pt")

    # Perform inference on an image
    results = model(r"C:\Users\admin\Desktop\fushi_data\IMG_20250101_093521.jpg")

    results[0].show()
    pass
