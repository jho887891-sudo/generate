from flask import Flask, render_template, request, jsonify, send_file, g, current_app
from flask_cors import CORS
import os
import json
import logging
import time
import threading
import uuid
import queue
from functions.func1 import process_pdf_to_excel  # 修改为委外单OCR识别
from functions.func2 import reconcile_excel_files
from functions.func3 import negative_quantity
from functions.func4 import reconcile_delivery_files
from functions.func5 import reconcile_purchase_files
from functions.func6 import process_purchase_ocr  # 新增采购单OCR识别

app = Flask(__name__)
CORS(app)

# 配置
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'
app.config['WEIWAI_CONFIG_PATH'] = 'Extracted_Info.json'  # 委外单配置文件
app.config['PURCHASE_CONFIG_PATH'] = 'Extracted_Info_caigou.json'  # 采购单配置文件
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB限制
app.config['TASK_TIMEOUT'] = 3600  # 任务超时时间（秒）
app.config['MAX_CONCURRENT_PROCESSING'] = 1  # 最大并发处理数

# 确保目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 全局任务字典
tasks = {}
active_tasks = {}  # 用于跟踪活动任务
processing_queue = queue.Queue()  # 处理队列
processing_lock = threading.Lock()  # 处理锁
processing_thread = None  # 后台处理线程


# 后台处理线程
def processing_worker():
    """后台处理线程，按顺序处理队列中的任务"""
    while True:
        task_id, file_path, task_type = processing_queue.get()
        if task_id is None:  # 终止信号
            break

        try:
            # 创建应用上下文
            with app.app_context():
                task = tasks[task_id]
                task['status'] = 'processing'
                task['current_file'] = os.path.basename(file_path)

                # 进度回调函数
                def progress_callback(progress, message):
                    if task_id in tasks and not tasks[task_id].get('cancelled', False):
                        tasks[task_id]['progress'] = progress
                        tasks[task_id]['message'] = message

                # 根据任务类型处理文件
                if task_type == 'weiwai':
                    # 处理委外单PDF
                    output_path = process_pdf_to_excel(
                        pdf_path=file_path,
                        output_dir=app.config['OUTPUT_FOLDER'],
                        config_path=app.config['WEIWAI_CONFIG_PATH'],
                        progress_callback=progress_callback
                    )
                elif task_type == 'purchase':
                    # 处理采购单PDF
                    output_path = process_purchase_ocr(
                        pdf_path=file_path,
                        output_dir=app.config['OUTPUT_FOLDER'],
                        config_path=app.config['PURCHASE_CONFIG_PATH'],
                        progress_callback=progress_callback
                    )
                else:
                    raise ValueError(f"未知的任务类型: {task_type}")

                # 提取文件名
                filename = os.path.basename(output_path)

                # 更新任务状态
                task['processed_files'] += 1
                task['progress'] = (task['processed_files'] / task['total_files']) * 100
                task['results'].append({
                    'success': True,
                    'filename': filename,
                    'download_url': f'/download/{filename}'
                })

                # 如果所有文件处理完成
                if task['processed_files'] >= task['total_files']:
                    task['status'] = 'completed'
                    task[
                        'message'] = f'批量处理完成! 成功: {len([r for r in task["results"] if r["success"]])}/{task["total_files"]}'
                    task['progress'] = 100

                    # 添加下载链接
                    task['download_links'] = []
                    for result in task['results']:
                        if result['success']:
                            task['download_links'].append({
                                'filename': result['filename'],
                                'url': result['download_url']
                            })

        except Exception as e:
            logger.error(f"处理文件失败: {file_path} - {str(e)}", exc_info=True)
            if task_id in tasks:
                task['processed_files'] += 1
                task['progress'] = (task['processed_files'] / task['total_files']) * 100
                task['results'].append({
                    'success': False,
                    'filename': os.path.basename(file_path),
                    'error': str(e)
                })

                # 如果所有文件处理完成
                if task['processed_files'] >= task['total_files']:
                    task['status'] = 'completed'
                    task[
                        'message'] = f'批量处理完成! 成功: {len([r for r in task["results"] if r["success"]])}/{task["total_files"]}'
                    task['progress'] = 100
        finally:
            processing_queue.task_done()


# 启动处理线程
def start_processing_thread():
    global processing_thread
    if processing_thread is None:
        processing_thread = threading.Thread(target=processing_worker, daemon=True)
        processing_thread.start()


# 在应用启动时启动处理线程
start_processing_thread()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process_weiwai_pdf', methods=['POST'])
def process_weiwai_pdf():
    """处理委外单PDF文件"""
    if 'pdf_file' not in request.files:
        return jsonify({'error': '没有选择文件'}), 400

    pdf_files = request.files.getlist('pdf_file')
    if not pdf_files or any(file.filename == '' for file in pdf_files):
        return jsonify({'error': '没有选择文件'}), 400

    # 验证所有文件都是PDF
    if any(not file.filename.lower().endswith('.pdf') for file in pdf_files):
        return jsonify({'error': '请上传PDF文件'}), 400

    # 创建任务
    task_id = str(uuid.uuid4())

    # 保存文件并添加到处理队列
    file_paths = []
    for file in pdf_files:
        # 保存上传的PDF文件
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        file_paths.append(file_path)

    # 创建任务对象
    tasks[task_id] = {
        'type': 'weiwai_pdf_batch',
        'status': 'pending',
        'total_files': len(file_paths),
        'processed_files': 0,
        'results': [],
        'start_time': time.time(),
        'progress': 0,
        'message': '等待处理...',
        'file_paths': file_paths,
        'cancelled': False
    }

    # 将文件添加到处理队列
    for file_path in file_paths:
        processing_queue.put((task_id, file_path, 'weiwai'))

    return jsonify({
        'success': True,
        'message': '委外单批量处理已开始',
        'task_id': task_id
    })


@app.route('/process_purchase_ocr', methods=['POST'])
def process_purchase_ocr_route():
    """处理采购单OCR识别"""
    if 'purchase_pdf_file' not in request.files:
        return jsonify({'error': '没有选择文件'}), 400

    pdf_files = request.files.getlist('purchase_pdf_file')
    if not pdf_files or any(file.filename == '' for file in pdf_files):
        return jsonify({'error': '没有选择文件'}), 400

    # 验证所有文件都是PDF
    if any(not file.filename.lower().endswith('.pdf') for file in pdf_files):
        return jsonify({'error': '请上传PDF文件'}), 400

    # 创建任务
    task_id = str(uuid.uuid4())

    # 保存文件并添加到处理队列
    file_paths = []
    for file in pdf_files:
        # 保存上传的PDF文件
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        file_paths.append(file_path)

    # 创建任务对象
    tasks[task_id] = {
        'type': 'purchase_ocr_batch',
        'status': 'pending',
        'total_files': len(file_paths),
        'processed_files': 0,
        'results': [],
        'start_time': time.time(),
        'progress': 0,
        'message': '等待处理...',
        'file_paths': file_paths,
        'cancelled': False
    }

    # 将文件添加到处理队列
    for file_path in file_paths:
        processing_queue.put((task_id, file_path, 'purchase'))

    return jsonify({
        'success': True,
        'message': '采购单批量处理已开始',
        'task_id': task_id
    })


@app.route('/progress/<task_id>')
def get_progress(task_id):
    """获取处理进度"""
    if task_id in tasks:
        task = tasks[task_id]
        response = {
            'task_id': task_id,
            'status': task['status'],
            'progress': task.get('progress', 0),
            'message': task.get('message', ''),
            'total_files': task.get('total_files', 1),
            'processed_files': task.get('processed_files', 0),
            'current_file': task.get('current_file', ''),
            'results': task.get('results', [])
        }

        # 计算当前处理中的文件
        if task['status'] == 'processing' and task['processed_files'] < task['total_files']:
            response['current_file'] = os.path.basename(task['file_paths'][task['processed_files']])

        # 添加下载链接
        if task['status'] == 'completed' and 'download_links' in task:
            response['download_links'] = task['download_links']

        return jsonify(response)
    return jsonify({'error': '任务ID不存在'}), 404


@app.route('/cancel_task/<task_id>', methods=['POST'])
def cancel_task(task_id):
    """取消处理任务"""
    if task_id in tasks:
        tasks[task_id]['cancelled'] = True
        tasks[task_id]['status'] = 'cancelled'
        tasks[task_id]['message'] = '处理已取消'

        # 从队列中移除该任务的所有文件
        with processing_lock:
            # 创建一个新队列，只保留不属于该任务的项目
            new_queue = queue.Queue()
            while not processing_queue.empty():
                t_id, file_path, task_type = processing_queue.get()
                if t_id != task_id:
                    new_queue.put((t_id, file_path, task_type))
            processing_queue.queue = new_queue.queue

        return jsonify({'success': True, 'message': '取消请求已发送'})
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

            # 提取文件名
            output_filename = os.path.basename(output_path)
            log_filename = os.path.basename(log_path)

            return jsonify({
                'success': True,
                'message': '对账完成',
                'download_url': f'/download/{output_filename}',
                'log_url': f'/download/{log_filename}'
            })
        except Exception as e:
            logger.error(f"Excel对账失败: {str(e)}", exc_info=True)
            return jsonify({'error': f'对账失败: {str(e)}'}), 500

    return jsonify({'error': '请上传Excel文件'}), 400


@app.route('/reconcile_delivery', methods=['POST'])
def reconcile_delivery():
    """送货单对账功能"""
    if 'delivery_receipt_file' not in request.files or 'delivery_total_file' not in request.files:
        return jsonify({'error': '请选择两个Excel文件'}), 400

    receipt_file = request.files['delivery_receipt_file']
    total_file = request.files['delivery_total_file']

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
            result_file, log_file = reconcile_delivery_files(
                receipt_path=receipt_path,
                total_path=total_path,
                output_dir=app.config['OUTPUT_FOLDER']
            )

            if not result_file or not log_file:
                return jsonify({'error': '对账处理失败'}), 500

            # 提取文件名
            result_filename = os.path.basename(result_file)
            log_filename = os.path.basename(log_file)

            return jsonify({
                'success': True,
                'message': '送货单对账完成',
                'download_url': f'/download/{result_filename}',
                'log_url': f'/download/{log_filename}'
            })
        except Exception as e:
            logger.error(f"送货单对账失败: {str(e)}", exc_info=True)
            return jsonify({'error': f'送货单对账失败: {str(e)}'}), 500

    return jsonify({'error': '请上传Excel文件'}), 400


@app.route('/reconcile_purchase', methods=['POST'])
def reconcile_purchase():
    """采购对账功能"""
    if 'purchase_receipt_file' not in request.files or 'purchase_statement_file' not in request.files:
        return jsonify({'error': '请选择两个Excel文件'}), 400

    purchase_receipt_file = request.files['purchase_receipt_file']
    purchase_statement_file = request.files['purchase_statement_file']

    if purchase_receipt_file.filename == '' or purchase_statement_file.filename == '':
        return jsonify({'error': '请选择两个Excel文件'}), 400

    if (purchase_receipt_file and purchase_receipt_file.filename.lower().endswith(('.xlsx', '.xls')) and
            purchase_statement_file and purchase_statement_file.filename.lower().endswith(('.xlsx', '.xls'))):

        # 保存上传的文件
        purchase_receipt_path = os.path.join(app.config['UPLOAD_FOLDER'], purchase_receipt_file.filename)
        purchase_statement_path = os.path.join(app.config['UPLOAD_FOLDER'], purchase_statement_file.filename)

        purchase_receipt_file.save(purchase_receipt_path)
        purchase_statement_file.save(purchase_statement_path)

        # 处理对账
        try:
            result_file, log_file = reconcile_purchase_files(
                purchase_receipt_path=purchase_receipt_path,
                purchase_statement_path=purchase_statement_path,
                output_dir=app.config['OUTPUT_FOLDER']
            )

            if not result_file or not log_file:
                return jsonify({'error': '采购对账处理失败'}), 500

            # 提取文件名
            result_filename = os.path.basename(result_file)
            log_filename = os.path.basename(log_file)

            return jsonify({
                'success': True,
                'message': '采购对账完成',
                'download_url': f'/download/{result_filename}',
                'log_url': f'/download/{log_filename}'
            })
        except Exception as e:
            logger.error(f"采购对账失败: {str(e)}", exc_info=True)
            return jsonify({'error': f'采购对账失败: {str(e)}'}), 500

    return jsonify({'error': '请上传Excel文件'}), 400


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

            # 提取文件名
            filename = os.path.basename(output_path)

            return jsonify({
                'success': True,
                'message': '处理成功',
                'download_url': f'/download/{filename}'
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


# 清理过期任务
def cleanup_tasks():
    """定期清理过期任务"""
    while True:
        try:
            current_time = time.time()
            expired_tasks = []

            for task_id, task in list(tasks.items()):
                # 任务超时或已完成超过1小时
                if (current_time - task['start_time'] > app.config['TASK_TIMEOUT'] or
                        (task['status'] in ['completed', 'failed', 'cancelled'] and
                         current_time - task['start_time'] > 3600)):
                    expired_tasks.append(task_id)

            for task_id in expired_tasks:
                if task_id in tasks:
                    del tasks[task_id]
                if task_id in active_tasks:
                    del active_tasks[task_id]

        except Exception as e:
            logger.error(f"清理任务失败: {str(e)}", exc_info=True)

        # 每小时清理一次
        time.sleep(3600)


# 启动任务清理线程
cleanup_thread = threading.Thread(target=cleanup_tasks, daemon=True)
cleanup_thread.start()


# 在应用启动时创建配置文件示例（如果不存在）
def create_default_configs():
    # 委外单默认配置
    weiwai_config_path = app.config['WEIWAI_CONFIG_PATH']
    if not os.path.exists(weiwai_config_path):
        default_config = {
            "company_list": ["众诺", "其他公司"],
            "company_titles": {
                "众诺": ["品号", "品名", "规格", "单位", "数量", "批号"],
                "其他公司": ["物料编码", "物料名称", "规格型号", "单位", "数量", "批号"]
            },
            "years": [2023, 2024, 2025]
        }
        try:
            with open(weiwai_config_path, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, ensure_ascii=False, indent=2)
            logger.info(f"已创建默认委外单配置文件: {weiwai_config_path}")
        except Exception as e:
            logger.error(f"创建委外单配置文件失败: {str(e)}")

    # 采购单默认配置
    purchase_config_path = app.config['PURCHASE_CONFIG_PATH']
    if not os.path.exists(purchase_config_path):
        default_config = {
            "company_list": ["供应商A", "供应商B"],
            "company_titles": {
                "供应商A": ["采购单号", "物料编码", "物料名称", "规格", "单位", "数量", "单价", "金额"],
                "供应商B": ["PO号", "物料号", "物料描述", "规格", "单位", "数量", "单价", "总价"]
            },
            "years": [2023, 2024, 2025]
        }
        try:
            with open(purchase_config_path, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, ensure_ascii=False, indent=2)
            logger.info(f"已创建默认采购单配置文件: {purchase_config_path}")
        except Exception as e:
            logger.error(f"创建采购单配置文件失败: {str(e)}")


# 在应用启动时创建默认配置文件
create_default_configs()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)