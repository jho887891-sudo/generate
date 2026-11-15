import os

os.environ['FLASK_SKIP_DOTENV'] = '1'  # 彻底禁用Flask加载环境变量文件

import re
import uuid
import threading
import sqlite3
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify, send_file
from datetime import datetime, timedelta
from func1 import main_calculation

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'
app.config['DATABASE'] = 'tasks.db'

# 确保文件夹存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)


# 初始化数据库 - 使用本地时间
def init_db():
    conn = sqlite3.connect(app.config['DATABASE'])
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS tasks
                 (id TEXT PRIMARY KEY,
                  created_at DATETIME DEFAULT (datetime('now', 'localtime')),  -- 使用本地时间
                  status TEXT,
                  progress INTEGER,
                  message TEXT,
                  result_path TEXT,
                  raise_path TEXT,
                  done_path TEXT,
                  raw_filename TEXT,
                  template_filename TEXT,
                  baseline_filename TEXT)''')
    conn.commit()
    conn.close()


init_db()

# 任务状态字典（用于存储运行中的任务状态）
tasks = {}


# 将时间字符串增加8小时
def add_hours_to_time(time_str, hours=8):
    """将时间字符串增加指定小时数"""
    if not time_str:
        return ""

    try:
        # 解析时间
        dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M")
        # 增加小时数
        adjusted_dt = dt + timedelta(hours=hours)
        return adjusted_dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return time_str  # 如果解析失败，返回原始字符串


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/history')
def history():
    conn = sqlite3.connect(app.config['DATABASE'])
    c = conn.cursor()
    # 在查询时直接格式化时间
    c.execute("""
        SELECT 
            id,
            strftime('%Y-%m-%d %H:%M', created_at) AS formatted_time,
            status,
            progress,
            message,
            result_path,
            raise_path,
            done_path,
            raw_filename,
            template_filename,
            baseline_filename
        FROM tasks 
        ORDER BY created_at DESC
    """)
    tasks = c.fetchall()
    conn.close()

    # 转换时间格式
    formatted_tasks = []
    for task in tasks:
        # 将时间增加8小时
        adjusted_time = add_hours_to_time(task[1])

        formatted_tasks.append({
            'id': task[0],
            'created_at': adjusted_time,  # 使用调整后的时间
            'status': task[2],
            'progress': task[3],
            'message': task[4],
            'result_path': task[5],
            'raise_path': task[6],
            'done_path': task[7],
            'raw_filename': task[8],
            'template_filename': task[9],
            'baseline_filename': task[10]
        })

    return render_template('history.html', tasks=formatted_tasks)


@app.route('/tasks')
def get_tasks():
    conn = sqlite3.connect(app.config['DATABASE'])
    c = conn.cursor()
    # 在查询时直接格式化时间
    c.execute("""
        SELECT 
            id,
            strftime('%Y-%m-%d %H:%M', created_at) AS formatted_time,
            status,
            progress,
            message,
            result_path,
            raise_path,
            done_path,
            raw_filename,
            template_filename,
            baseline_filename
        FROM tasks 
        ORDER BY created_at DESC
    """)
    tasks = c.fetchall()
    conn.close()

    formatted_tasks = []
    for task in tasks:
        # 将时间增加8小时
        adjusted_time = add_hours_to_time(task[1])

        formatted_tasks.append({
            'id': task[0],
            'created_at': adjusted_time,  # 使用调整后的时间
            'status': task[2],
            'progress': task[3],
            'message': task[4],
            'result_path': task[5],
            'raise_path': task[6],
            'done_path': task[7],
            'raw_filename': task[8],
            'template_filename': task[9],
            'baseline_filename': task[10]
        })

    return jsonify(formatted_tasks)


@app.route('/upload', methods=['POST'])
def upload_files():
    raw_file = request.files['raw_file']
    template_file = request.files['template_file']
    baseline_file = request.files['baseline_file']

    if not raw_file or not template_file or not baseline_file:
        return jsonify({'error': '请上传所有必需文件'}), 400

    task_id = str(uuid.uuid4())
    task_upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], task_id)
    task_result_dir = os.path.join(app.config['RESULT_FOLDER'], task_id)
    os.makedirs(task_upload_dir, exist_ok=True)
    os.makedirs(task_result_dir, exist_ok=True)

    # 保存原始文件名
    raw_filename = raw_file.filename
    template_filename = template_file.filename
    baseline_filename = baseline_file.filename

    # 保存文件
    raw_path = os.path.join(task_upload_dir, 'raw.xlsx')
    template_path = os.path.join(task_upload_dir, 'template.xlsx')
    baseline_path = os.path.join(task_upload_dir, 'baseline.xlsx')
    raw_file.save(raw_path)
    template_file.save(template_path)
    baseline_file.save(baseline_path)

    # 创建数据库记录
    conn = sqlite3.connect(app.config['DATABASE'])
    c = conn.cursor()
    c.execute(
        "INSERT INTO tasks (id, status, progress, message, raw_filename, template_filename, baseline_filename) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (task_id, 'pending', 0, '任务已创建', raw_filename, template_filename, baseline_filename))
    conn.commit()
    conn.close()

    # 初始化任务状态
    tasks[task_id] = {
        'status': 'pending',
        'progress': 0,
        'message': '任务已创建',
        'result_files': None,
        'error': None
    }

    # 后台执行计算任务
    thread = threading.Thread(
        target=process_reduction_calculation,
        args=(task_id, raw_path, template_path, baseline_path, task_result_dir)
    )
    thread.start()

    return jsonify({'task_id': task_id})


def process_reduction_calculation(task_id, raw_path, template_path, baseline_path, result_dir):
    try:
        # 更新任务状态为运行中
        update_task_status(task_id, 'running', 10, '开始处理数据...')

        # 调用计算函数
        result_files = main_calculation(
            raw_path=raw_path,
            template_path=template_path,
            baseline_path=baseline_path,
            output_dir=result_dir,
            progress_callback=lambda p, msg: update_task_status(task_id, 'running', p, msg)
        )

        # 更新任务状态为完成
        update_task_status(task_id, 'completed', 100, '处理完成')

        # 更新数据库记录
        conn = sqlite3.connect(app.config['DATABASE'])
        c = conn.cursor()
        c.execute(
            "UPDATE tasks SET status=?, progress=?, message=?, result_path=?, raise_path=?, done_path=? WHERE id=?",
            (
                'completed', 100, '处理完成', result_files['result'], result_files['raise'], result_files['done'],
                task_id))
        conn.commit()
        conn.close()

    except Exception as e:
        error_msg = f'处理错误: {str(e)}'
        update_task_status(task_id, 'error', 0, error_msg)

        # 更新数据库记录
        conn = sqlite3.connect(app.config['DATABASE'])
        c = conn.cursor()
        c.execute("UPDATE tasks SET status=?, message=? WHERE id=?",
                  ('error', error_msg, task_id))
        conn.commit()
        conn.close()


def update_task_status(task_id, status, progress, message):
    if task_id in tasks:
        tasks[task_id]['status'] = status
        tasks[task_id]['progress'] = progress
        tasks[task_id]['message'] = message


@app.route('/task/<task_id>')
def get_task_status(task_id):
    # 首先检查内存中的任务状态
    if task_id in tasks:
        return jsonify(tasks[task_id])

    # 如果不在内存中，从数据库查询
    conn = sqlite3.connect(app.config['DATABASE'])
    c = conn.cursor()
    c.execute("SELECT status, progress, message, result_path, raise_path, done_path FROM tasks WHERE id=?", (task_id,))
    task = c.fetchone()
    conn.close()

    if task:
        return jsonify({
            'status': task[0],
            'progress': task[1],
            'message': task[2],
            'result_path': task[3],
            'raise_path': task[4],
            'done_path': task[5]
        })
    else:
        return jsonify({'error': '任务不存在'}), 404


@app.route('/download/<task_id>/<file_type>')
def download_file(task_id, file_type):
    conn = sqlite3.connect(app.config['DATABASE'])
    c = conn.cursor()
    c.execute("SELECT result_path, raise_path, done_path FROM tasks WHERE id=?", (task_id,))
    task = c.fetchone()
    conn.close()

    if not task:
        return '任务不存在', 404

    file_path = None
    filename = ""

    if file_type == 'result' and task[0]:
        file_path = task[0]
        filename = os.path.basename(file_path)
    elif file_type == 'raise' and task[1]:
        file_path = task[1]
        filename = os.path.basename(file_path)
    elif file_type == 'done' and task[2]:
        file_path = task[2]
        filename = os.path.basename(file_path)

    if file_path and os.path.exists(file_path):
        return send_file(file_path, as_attachment=True, download_name=filename)
    else:
        return '文件不存在', 404


if __name__ == '__main__':
    # 完全禁用Flask的环境变量加载功能
    from flask.cli import load_dotenv

    load_dotenv = lambda: None  # 覆盖load_dotenv函数为空操作

    # 修改这里：监听所有网段
    app.run(host='0.0.0.0', port=5000, debug=True)