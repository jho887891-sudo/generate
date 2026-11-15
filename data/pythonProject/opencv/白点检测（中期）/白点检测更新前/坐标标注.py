import os
import glob
import json
import cv2

def load_roi_file(filepath):
    if not os.path.exists(filepath):
        return {"regions": []}  # 初始化为空的结构
    with open(filepath, 'r') as f:
        return json.load(f)

def save_roi_file(filepath, roi_data):
    with open(filepath, 'w') as f:
        json.dump(roi_data, f, indent=4)

def edit_roi(roi_data, new_region):
    if 'regions' not in roi_data:
        roi_data['regions'] = []
    roi_data['regions'].append(new_region)
    return roi_data

def manual_annotation(image_path):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    resized_image = cv2.resize(image, (int(width * 0.3), int(height * 0.3)))

    roi = cv2.selectROI("Select Region", resized_image, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()

    return {
        "x": int(roi[0]),
        "y": int(roi[1]),
        "width": int(roi[2]),
        "height": int(roi[3])
    }

if __name__ == "__main__":
    roi_file = 'roi.json'
    roi_data = load_roi_file(roi_file)

    image_folder = 'image'
    image_files = glob.glob(os.path.join(image_folder, '*.jpg'))

    for image_file in image_files:
        new_region = manual_annotation(image_file)
        roi_data = edit_roi(roi_data, new_region)

        print("按任意键继续到下一张...")
        cv2.waitKey(0)

    save_roi_file(roi_file, roi_data)
