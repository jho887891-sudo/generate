from flask import Flask, render_template, request, jsonify, send_file, g
from flask_cors import CORS
import os
import json
import logging
import time
import threading
import sqlite3
import datetime
from functions.func1 import process_pdf_to_excel
from functions.func2 import reconcile_excel_files
from functions.func3 import negative_quantity

app = Flask(__name__)
CORS(app)

# 配置
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'
app.config['CONFIG_PATH'] = 'Extracted_Info.json'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB限制
app.config['DATABASE'] = 'reconciliation.db'  # SQLite数据库文件

# 确保目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 全局进度字典
progress_data = {}


# 初始化数据库
def init_db():
    db_path = app.config['DATABASE']
    if not os.path.exists(db_path):
        logger.info(f"创建数据库文件: {db_path}")

    # 直接连接数据库
    db = sqlite3.connect(db_path)
    cursor = db.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS reconciliation_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_type TEXT NOT NULL,
            upload_time DATETIME DEFAULT CURRENT_TIMESTAMP,
            company_name TEXT,
            delivery_date TEXT,
            delivery_no TEXT,
            additional_info TEXT
        )
    ''')
    db.commit()
    db.close()
    logger.info("数据库初始化完成")


# 获取数据库连接
def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(app.config['DATABASE'])
        g.db.row_factory = sqlite3.Row
    return g.db


# 关闭数据库连接
@app.teardown_appcontext
def close_db(exception):
    db = getattr(g, 'db', None)
    if db is not None:
        db.close()


# 保存文件信息到数据库
def save_file_to_db(filename, file_path, file_type, company_name=None, delivery_date=None, delivery_no=None,
                    additional_info=None):
    try:
        db = get_db()
        cursor = db.cursor()
        cursor.execute('''
            INSERT INTO reconciliation_files 
            (filename, file_path, file_type, company_name, delivery_date, delivery_no, additional_info)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (filename, file_path, file_type, company_name, delivery_date, delivery_no,
              json.dumps(additional_info) if additional_info else None))
        db.commit()
        return cursor.lastrowid
    except Exception as e:
        logger.error(f"保存文件信息到数据库失败: {str(e)}", exc_info=True)
        return None


# 获取历史文件列表
def get_history_files(file_type=None, limit=10):
    try:
        db = get_db()
        cursor = db.cursor()

        if file_type:
            cursor.execute('''
                SELECT id, filename, file_path, file_type, upload_time, company_name, delivery_date, delivery_no
                FROM reconciliation_files
                WHERE file_type = ?
                ORDER BY upload_time DESC
                LIMIT ?
            ''', (file_type, limit))
        else:
            cursor.execute('''
                SELECT id, filename, file_path, file_type, upload_time, company_name, delivery_date, delivery_no
                FROM reconciliation_files
                ORDER BY upload_time DESC
                LIMIT ?
            ''', (limit,))

        return cursor.fetchall()
    except Exception as e:
        logger.error(f"获取历史文件列表失败: {str(e)}", exc_info=True)
        return []


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/history')
def history():
    # 获取最近的历史文件
    pdf_files = get_history_files('pdf_excel', 10)
    excel_files = get_history_files('excel_reconciliation', 10)
    negative_files = get_history_files('negative_excel', 10)

    return render_template('history.html',
                           pdf_files=pdf_files,
                           excel_files=excel_files,
                           negative_files=negative_files)


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
    # 创建应用上下文
    with app.app_context():
        try:
            # 进度回调函数
            def progress_callback(progress, message):
                progress_data[task_id]['progress'] = progress
                progress_data[task_id]['message'] = message

            # 处理PDF
            output_path = process_pdf_to_excel(
                pdf_path=pdf_path,
                output_dir=app.config['OUTPUT_FOLDER'],
                config_path=app.config['CONFIG_PATH'],
                progress_callback=progress_callback
            )

            # 提取文件名
            filename = os.path.basename(output_path)

            # 尝试从文件名中提取公司名称、日期和单号
            company_name = None
            delivery_date = None
            delivery_no = None

            # 保存到数据库
            file_id = save_file_to_db(
                filename=filename,
                file_path=output_path,
                file_type='pdf_excel',
                company_name=company_name,
                delivery_date=delivery_date,
                delivery_no=delivery_no
            )

            # 更新进度为完成
            progress_data[task_id] = {
                'progress': 100,
                'message': '处理完成',
                'download_url': f'/download/{filename}',
                'file_id': file_id
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

            # 提取文件名
            output_filename = os.path.basename(output_path)
            log_filename = os.path.basename(log_path)

            # 保存到数据库
            file_id = save_file_to_db(
                filename=output_filename,
                file_path=output_path,
                file_type='excel_reconciliation',
                additional_info={
                    'log_filename': log_filename,
                    'log_path': log_path
                }
            )

            return jsonify({
                'success': True,
                'message': '对账完成',
                'download_url': f'/download/{output_filename}',
                'log_url': f'/download/{log_filename}',
                'file_id': file_id
            })
        except Exception as e:
            logger.error(f"Excel对账失败: {str(e)}", exc_info=True)
            return jsonify({'error': f'对账失败: {str(e)}'}), 500

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

            # 保存到数据库
            file_id = save_file_to_db(
                filename=filename,
                file_path=output_path,
                file_type='negative_excel'
            )

            return jsonify({
                'success': True,
                'message': '处理成功',
                'download_url': f'/download/{filename}',
                'file_id': file_id
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


@app.route('/file/<int:file_id>')
def file_detail(file_id):
    try:
        db = get_db()
        cursor = db.cursor()
        cursor.execute('''
            SELECT id, filename, file_path, file_type, upload_time, company_name, delivery_date, delivery_no, additional_info
            FROM reconciliation_files
            WHERE id = ?
        ''', (file_id,))
        file_info = cursor.fetchone()

        if file_info:
            # 转换附加信息
            additional_info = json.loads(file_info['additional_info']) if file_info['additional_info'] else None

            return render_template('file_detail.html', file_info={
                'id': file_info['id'],
                'filename': file_info['filename'],
                'file_type': file_info['file_type'],
                'upload_time': file_info['upload_time'],
                'company_name': file_info['company_name'],
                'delivery_date': file_info['delivery_date'],
                'delivery_no': file_info['delivery_no'],
                'additional_info': additional_info
            })
        else:
            return render_template('error.html', message='文件不存在'), 404
    except Exception as e:
        logger.error(f"获取文件详情失败: {str(e)}", exc_info=True)
        return render_template('error.html', message='服务器错误'), 500


@app.route('/history/files')
def get_history_files_api():
    file_type = request.args.get('type', default=None, type=str)
    limit = request.args.get('limit', default=10, type=int)

    try:
        files = get_history_files(file_type, limit)
        result = []
        for file in files:
            result.append({
                'id': file['id'],
                'filename': file['filename'],
                'file_type': file['file_type'],
                'upload_time': file['upload_time'],
                'company_name': file['company_name'],
                'delivery_date': file['delivery_date'],
                'delivery_no': file['delivery_no']
            })
        return jsonify(result)
    except Exception as e:
        logger.error(f"获取历史文件列表失败: {str(e)}", exc_info=True)
        return jsonify({'error': '服务器错误'}), 500


if __name__ == '__main__':
    # 初始化数据库
    init_db()

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