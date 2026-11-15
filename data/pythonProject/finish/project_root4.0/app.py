from flask import Flask, request, jsonify, render_template
from functions.image_processing import find_and_process_ok_folders
from functions.make_report import generate_report
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process_images', methods=['POST'])
def process_images_route():
    data = request.json
    base_folder = data.get('base_folder')
    output_base_folder = data.get('output_base_folder')
    threshold2 = data.get('threshold2')  # 获取阈值2

    # 确保输出文件夹存在
    if not os.path.exists(output_base_folder):
        os.makedirs(output_base_folder)

    # 调用处理函数
    try:
        find_and_process_ok_folders(base_folder, output_base_folder, threshold2)  # 传递阈值2
        return jsonify({"status": "处理完成"})
    except Exception as e:
        return jsonify({"status": "处理失败", "error": str(e)})

@app.route('/generate_report', methods=['POST'])
def generate_report_route():
    base_image_folder = request.json.get('base_image_folder')
    template_path = request.json.get('template_path')
    output_folder = request.json.get('output_folder')
    aspect_ratios = request.json.get('aspect_ratios')

    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 调用生成报告函数
    try:
        report = generate_report(base_image_folder, template_path, output_folder, aspect_ratios)
        return jsonify({"report": report})
    except Exception as e:
        return jsonify({"status": "生成报告失败", "error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
