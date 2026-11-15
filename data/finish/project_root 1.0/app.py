from flask import Flask, request, jsonify, render_template
from functions.image_processing import process_images, find_and_process_ok_folders
from functions.make_report import generate_report

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process_images', methods=['POST'])
def process_images_route():
    base_folder = request.json.get('base_folder')
    output_base_folder = request.json.get('output_base_folder')
    # 调用处理函数
    find_and_process_ok_folders(base_folder, output_base_folder)
    return jsonify({"status": "处理完成"})

@app.route('/generate_report', methods=['POST'])
def generate_report_route():
    base_image_folder = request.json.get('base_image_folder')
    template_path = request.json.get('template_path')
    output_folder = request.json.get('output_folder')
    aspect_ratios = request.json.get('aspect_ratios')  # 获取比例数据

    report = generate_report(base_image_folder, template_path, output_folder, aspect_ratios)  # 传递比例
    return jsonify({"report": report})

if __name__ == '__main__':
    app.run(debug=True)
