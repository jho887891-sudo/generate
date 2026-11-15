from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import os
import json
import logging
import time
import threading
from functions.func1 import process_pdf_to_excel
from functions.func2 import reconcile_excel_files
from functions.func3 import negative_quantity  # 导入新功能

app = Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'
app.config['CONFIG_PATH'] = 'Extracted_Info.json'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB限制

# 确保上传目录和输出目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 全局进度字典
progress_data = {}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process_pdf', methods=['POST'])
def process_pdf():
    if 'pdf_file' not in request.files:
        return jsonify({'error': '没有选择文件'}), 400

    pdf_file = request.files['pdf_file']
    if pdf_file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400

    if pdf_file and pdf_file.filename.lower().endswith('.pdf'):
        # 保存上传的PDF文件
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_file.filename)
        pdf_file.save(pdf_path)

        # 生成唯一任务ID
        task_id = str(int(time.time() * 1000))
        progress_data[task_id] = {'progress': 0, 'message': '开始处理...'}

        # 启动处理线程
        thread = threading.Thread(
            target=process_pdf_thread,
            args=(pdf_path, task_id)
        )
        thread.start()

        return jsonify({
            'success': True,
            'message': '处理已开始',
            'task_id': task_id
        })

    return jsonify({'error': '请上传PDF文件'}), 400


def process_pdf_thread(pdf_path, task_id):
    """在后台线程中处理PDF"""
    try:
        # 进度回调函数
        def progress_callback(progress, message):
            progress_data[task_id] = {'progress': progress, 'message': message}

        # 处理PDF
        output_path = process_pdf_to_excel(
            pdf_path=pdf_path,
            output_dir=app.config['OUTPUT_FOLDER'],
            config_path=app.config['CONFIG_PATH'],
            progress_callback=progress_callback
        )

        # 更新进度为完成
        progress_data[task_id] = {
            'progress': 100,
            'message': '处理完成',
            'download_url': f'/download/{os.path.basename(output_path)}'
        }

    except PermissionError as pe:
        logger.error(f"文件权限错误: {str(pe)}", exc_info=True)
        progress_data[task_id] = {
            'progress': -1,
            'message': f'文件被占用，请关闭后重试: {str(pe)}'
        }
    except Exception as e:
        logger.error(f"PDF处理失败: {str(e)}", exc_info=True)
        progress_data[task_id] = {
            'progress': -1,
            'message': f'处理失败: {str(e)}'
        }


@app.route('/progress/<task_id>')
def get_progress(task_id):
    """获取处理进度"""
    if task_id in progress_data:
        return jsonify(progress_data[task_id])
    return jsonify({'error': '任务ID不存在'}), 404


@app.route('/reconcile_excel', methods=['POST'])
def reconcile_excel():
    if 'receipt_file' not in request.files or 'total_file' not in request.files:
        return jsonify({'error': '请选择两个Excel文件'}), 400

    receipt_file = request.files['receipt_file']
    total_file = request.files['total_file']

    if receipt_file.filename == '' or total_file.filename == '':
        return jsonify({'error': '请选择两个Excel文件'}), 400

    if (receipt_file and receipt_file.filename.lower().endswith(('.xlsx', '.xls')) and
            total_file and total_file.filename.lower().endswith(('.xlsx', '.xls'))):

        # 保存上传的文件
        receipt_path = os.path.join(app.config['UPLOAD_FOLDER'], receipt_file.filename)
        total_path = os.path.join(app.config['UPLOAD_FOLDER'], total_file.filename)

        receipt_file.save(receipt_path)
        total_file.save(total_path)

        # 处理对账
        try:
            output_path, log_path = reconcile_excel_files(
                receipt_path=receipt_path,
                total_path=total_path,
                output_dir=app.config['OUTPUT_FOLDER']
            )
            return jsonify({
                'success': True,
                'message': '对账完成',
                'download_url': f'/download/{os.path.basename(output_path)}',
                'log_url': f'/download/{os.path.basename(log_path)}'
            })
        except Exception as e:
            logger.error(f"Excel对账失败: {str(e)}", exc_info=True)
            return jsonify({'error': f'对账失败: {str(e)}'}), 500

    return jsonify({'error': '请上传Excel文件'}), 400


# 新增路由：处理Excel数量取负
@app.route('/negative_excel', methods=['POST'])
def negative_excel():
    if 'negative_file' not in request.files:
        return jsonify({'error': '没有选择文件'}), 400

    excel_file = request.files['negative_file']
    if excel_file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400

    if excel_file and excel_file.filename.lower().endswith(('.xlsx', '.xls')):
        # 保存上传的Excel文件
        excel_path = os.path.join(app.config['UPLOAD_FOLDER'], excel_file.filename)
        excel_file.save(excel_path)

        # 处理Excel
        try:
            output_path = negative_quantity(
                input_path=excel_path,
                output_dir=app.config['OUTPUT_FOLDER']
            )
            return jsonify({
                'success': True,
                'message': '处理成功',
                'download_url': f'/download/{os.path.basename(output_path)}'
            })
        except PermissionError as pe:
            logger.error(f"文件权限错误: {str(pe)}")
            return jsonify({'error': f'文件被占用，请关闭后重试: {str(pe)}'}), 403
        except Exception as e:
            logger.error(f"Excel处理失败: {str(e)}", exc_info=True)
            return jsonify({'error': f'处理失败: {str(e)}'}), 500

    return jsonify({'error': '请上传Excel文件'}), 400


@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return jsonify({'error': '文件不存在'}), 404


if __name__ == '__main__':
    # 创建配置文件示例（如果不存在）
    config_path = app.config['CONFIG_PATH']
    if not os.path.exists(config_path):
        default_config = {
            "company_list": ["众诺", "其他公司"],
            "company_titles": {
                "众诺": ["品号", "品名", "规格", "单位", "数量", "批号"],
                "其他公司": ["物料编码", "物料名称", "规格型号", "单位", "数量", "批号"]
            },
            "years": [2023, 2024, 2025]
        }
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, ensure_ascii=False, indent=2)
            logger.info(f"已创建默认配置文件: {config_path}")
        except Exception as e:
            logger.error(f"创建配置文件失败: {str(e)}")

    app.run(debug=True, host='0.0.0.0', port=5000)