from flask import Flask, render_template, request, jsonify, send_file, session, redirect, url_for
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from apscheduler.schedulers.background import BackgroundScheduler
import os
import json
import logging
import time
import threading
import uuid
import queue
import sqlite3
from datetime import datetime, timedelta
import fitz  # PyMuPDF
from paddleocr import PaddleOCR
from typing import List, Tuple, Dict, Any, Optional, Callable
import urllib.parse
from werkzeug.utils import secure_filename
import hashlib
from functools import wraps

app = Flask(__name__)
CORS(app)

# 配置
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'
app.config['WEIWAI_CONFIG_PATH'] = 'Extracted_Info.json'  # 委外OCR配置文件
app.config['PURCHASE_CONFIG_PATH'] = 'Extracted_Info_caigou.json'  # 采购OCR配置文件
app.config['INVOICE_PRICING_CONFIG_PATH'] = 'invoice_pricing_config.json'  # 新增：单价处理配置文件
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB限制
app.config['TASK_TIMEOUT'] = 3600  # 任务超时时间（秒）
app.config['MAX_CONCURRENT_PROCESSING'] = 1  # 最大并发处理数
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///history.db'  # 历史记录数据库
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'your-secret-key-change-this-in-production'  # 请在生产环境中更改
app.config['USERS_CONFIG_PATH'] = 'users.json'  # 用户配置文件
app.config['SESSION_TIMEOUT'] = 3600  # 会话超时时间（秒）

# 确保目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 初始化数据库
db = SQLAlchemy(app)


# 用户模型
class User:
    def __init__(self, username, password, role, display_name):
        self.username = username
        self.password = password  # 存储明文密码
        self.role = role
        self.display_name = display_name

    def verify_password(self, password):
        """验证密码 - 直接比较明文"""
        return self.password == password


# 用户管理类
class UserManager:
    def __init__(self, config_path):
        self.config_path = config_path
        self.users = self.load_users()
        self.login_attempts = {}  # 记录登录尝试次数

    def load_users(self):
        """加载用户配置 - 使用明文密码"""
        try:
            if not os.path.exists(self.config_path):
                # 如果用户配置文件不存在，创建默认用户
                self.create_default_users()

            with open(self.config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                users = {}
                for user_data in data.get('users', []):
                    user = User(
                        username=user_data['username'],
                        password=user_data['password'],  # 直接使用明文密码
                        role=user_data.get('role', 'user'),
                        display_name=user_data.get('display_name', user_data['username'])
                    )
                    users[user_data['username']] = user
                return users
        except Exception as e:
            logger.error(f"加载用户配置失败: {str(e)}")
            return {}

    def create_default_users(self):
        """创建默认用户配置"""
        default_users = {
            "users": [
                {
                    "username": "admin",
                    "password": "admin132",  # 明文密码
                    "role": "admin",
                    "display_name": "系统管理员"
                }
            ]

        }

        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(default_users, f, ensure_ascii=False, indent=2)
            logger.info(f"已创建默认用户配置文件: {self.config_path}")
        except Exception as e:
            logger.error(f"创建用户配置文件失败: {str(e)}")

    def authenticate(self, username, password):
        """用户认证 - 直接比较明文密码"""
        # 检查登录尝试次数
        if username in self.login_attempts:
            if self.login_attempts[username] >= 5:  # 最多尝试5次
                return False, "登录尝试次数过多，请稍后再试"

        if username in self.users:
            user = self.users[username]
            # 直接比较明文密码
            if user.verify_password(password):
                # 重置登录尝试次数
                if username in self.login_attempts:
                    del self.login_attempts[username]
                return True, user
            else:
                # 增加登录尝试次数
                self.login_attempts[username] = self.login_attempts.get(username, 0) + 1
                logger.warning(f"密码不匹配: 用户={username}")
                return False, "用户名或密码错误"
        else:
            logger.warning(f"用户不存在: {username}")
            return False, "用户不存在"

    def get_user(self, username):
        """获取用户信息"""
        return self.users.get(username)


# 初始化用户管理器
user_manager = UserManager(app.config['USERS_CONFIG_PATH'])


# 登录装饰器
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not is_logged_in():
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({'success': False, 'error': '未登录', 'redirect': url_for('login')}), 401
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)

    return decorated_function


def is_logged_in():
    """检查用户是否已登录"""
    if 'user_id' in session and 'logged_in' in session:
        # 检查会话是否过期
        last_activity = session.get('last_activity')
        if last_activity:
            try:
                last_activity = datetime.fromisoformat(last_activity)
                if datetime.now() - last_activity > timedelta(seconds=app.config['SESSION_TIMEOUT']):
                    logout_user()
                    return False
            except:
                # 如果时间格式错误，清除会话
                logout_user()
                return False

        # 更新最后活动时间
        session['last_activity'] = datetime.now().isoformat()
        return True
    return False


def get_current_user():
    """获取当前用户"""
    if is_logged_in():
        return user_manager.get_user(session.get('user_id'))
    return None


def login_user(username):
    """用户登录"""
    session['user_id'] = username
    session['logged_in'] = True
    session['last_activity'] = datetime.now().isoformat()
    session.permanent = True


def logout_user():
    """用户登出"""
    session.clear()


# 历史记录模型
class OperationHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    task_id = db.Column(db.String(36), unique=True, nullable=False)
    function_code = db.Column(db.String(50), nullable=False)
    operation_name = db.Column(db.String(100), nullable=False)
    files = db.Column(db.JSON, nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now())  # 改为本地时间
    user_id = db.Column(db.Integer)  # 如果有用户系统可以关联

    def to_dict(self):
        return {
            'id': self.id,
            'task_id': self.task_id,
            'function_code': self.function_code,
            'operation_name': self.operation_name,
            'files': self.files,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'is_expired': datetime.now() - self.created_at > timedelta(days=15)  # 使用本地时间计算
        }


# 创建数据库表
with app.app_context():
    db.create_all()

# 全局任务字典
tasks = {}
active_tasks = {}  # 用于跟踪活动任务
processing_queue = queue.Queue()  # 处理队列
processing_lock = threading.Lock()  # 处理锁
processing_thread = None  # 后台处理线程

# 启动定时清理任务
scheduler = BackgroundScheduler()


def cleanup_old_history():
    """清理超过15天的历史记录 - 使用本地时间"""
    try:
        cutoff_date = datetime.now() - timedelta(days=15)  # 使用本地时间
        old_records = OperationHistory.query.filter(OperationHistory.created_at < cutoff_date).all()

        for record in old_records:
            db.session.delete(record)

        db.session.commit()
        logger.info(f"清理了 {len(old_records)} 条过期历史记录")
    except Exception as e:
        logger.error(f"清理历史记录失败: {str(e)}")


scheduler.add_job(cleanup_old_history, 'cron', hour=1)
scheduler.start()


# 启动处理线程
def start_processing_thread():
    global processing_thread
    if processing_thread is None:
        processing_thread = threading.Thread(target=processing_worker, daemon=True)
        processing_thread.start()


# 后台处理线程
def processing_worker():
    """后台处理线程，按顺序处理队列中的任务"""
    while True:
        task_data = processing_queue.get()
        if task_data is None:  # 终止信号
            break

        task_id, file_path, task_type = task_data

        try:
            # 创建应用上下文
            with app.app_context():
                task = tasks[task_id]

                # 检查任务是否已取消
                if task.get('cancelled', False):
                    logger.info(f"任务 {task_id} 已取消，跳过处理")
                    continue

                task['status'] = 'processing'
                task['current_file'] = os.path.basename(file_path)

                # 根据任务类型选择处理函数和配置文件
                if task_type == 'weiwai_ocr':
                    from functions.func1 import process_pdf_to_excel as process_func
                    config_path = app.config['WEIWAI_CONFIG_PATH']
                    task_type_name = '委外OCR识别'
                elif task_type == 'purchase_ocr':
                    from functions.func6 import process_pdf_to_excel as process_func
                    config_path = app.config['PURCHASE_CONFIG_PATH']
                    task_type_name = '采购OCR识别'
                else:
                    # 默认使用委外OCR识别
                    from functions.func1 import process_pdf_to_excel as process_func
                    config_path = app.config['WEIWAI_CONFIG_PATH']
                    task_type_name = '委外OCR识别'

                # 进度回调函数
                def progress_callback(progress, message):
                    if task_id in tasks and not tasks[task_id].get('cancelled', False):
                        tasks[task_id]['progress'] = progress
                        tasks[task_id]['message'] = message

                # 处理PDF
                output_path = process_func(
                    pdf_path=file_path,
                    output_dir=app.config['OUTPUT_FOLDER'],
                    config_path=config_path,  # 使用特定任务的配置文件
                    progress_callback=progress_callback
                )

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
                        'message'] = f'{task_type_name}批量处理完成! 成功: {len([r for r in task["results"] if r["success"]])}/{task["total_files"]}'
                    task['progress'] = 100

                    # 添加下载链接
                    task['download_links'] = []
                    for result in task['results']:
                        if result['success']:
                            task['download_links'].append({
                                'filename': result['filename'],
                                'url': result['download_url']
                            })

                    # 保存历史记录
                    save_history(
                        task_id=task_id,
                        function_code=task_type,
                        operation_name=task_type_name,
                        files=[{'name': r['filename'], 'path': r['download_url'].replace('/download/', '')}
                               for r in task['results'] if r['success']]
                    )

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


# 保存历史记录
def save_history(task_id, function_code, operation_name, files):
    """保存操作历史记录"""
    try:
        history = OperationHistory(
            task_id=task_id,
            function_code=function_code,
            operation_name=operation_name,
            files=files
        )
        db.session.add(history)
        db.session.commit()
        logger.info(f"已保存历史记录: {task_id}")
    except Exception as e:
        logger.error(f"保存历史记录失败: {str(e)}")


# 在应用启动时启动处理线程
start_processing_thread()


# 登录路由
@app.route('/login', methods=['GET', 'POST'])
def login():
    """登录页面 - 使用明文密码验证"""
    # 如果已登录，重定向到首页
    if is_logged_in():
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        remember_me = request.form.get('remember_me') == 'on'

        logger.info(f"登录请求: 用户名={username}")

        if not username or not password:
            logger.warning("用户名或密码为空")
            return jsonify({
                'success': False,
                'error': '用户名和密码不能为空'
            }), 401

        # 调试：打印所有可用用户
        logger.info(f"可用用户: {list(user_manager.users.keys())}")

        success, result = user_manager.authenticate(username, password)

        if success:
            user = result
            login_user(username)

            # 设置会话过期时间
            if remember_me:
                session.permanent = True
                app.permanent_session_lifetime = timedelta(days=7)  # 记住我一周
            else:
                session.permanent = False
                app.permanent_session_lifetime = timedelta(hours=1)  # 1小时

            next_url = request.args.get('next', url_for('index'))
            logger.info(f"用户登录成功: {username}, 角色: {user.role}")

            return jsonify({
                'success': True,
                'message': '登录成功',
                'redirect': next_url,
                'user': {
                    'username': user.username,
                    'display_name': user.display_name,
                    'role': user.role
                }
            })
        else:
            logger.warning(f"登录失败: {username}, 原因: {result}")
            return jsonify({
                'success': False,
                'error': result
            }), 401

    return render_template('login.html')


@app.route('/logout')
def logout():
    """退出登录"""
    logout_user()
    return redirect(url_for('login'))


@app.route('/')
def index():
    if is_logged_in():
        return render_template('index.html')
    else:
        return redirect(url_for('login'))


@app.route('/api/user/info')
@login_required
def get_user_info():
    """获取当前用户信息"""
    user = get_current_user()
    if user:
        return jsonify({
            'success': True,
            'data': {
                'username': user.username,
                'display_name': user.display_name,
                'role': user.role
            }
        })
    return jsonify({'success': False, 'error': '用户未登录'}), 401


@app.route('/api/history', methods=['GET'])
@login_required
def get_history():
    """获取历史记录"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        function_code = request.args.get('function_code')
        date_filter = request.args.get('date_filter')
        task_id = request.args.get('task_id')

        query = OperationHistory.query

        # 按任务ID筛选
        if task_id:
            query = query.filter_by(task_id=task_id)
        else:
            # 按功能代码筛选
            if function_code and function_code != 'all':
                query = query.filter_by(function_code=function_code)

            # 按日期筛选 - 使用本地时间
            now = datetime.now()  # 使用本地时间
            if date_filter:
                if date_filter == 'today':
                    query = query.filter(OperationHistory.created_at >= now.date())
                elif date_filter == '3days':
                    query = query.filter(OperationHistory.created_at >= now - timedelta(days=3))
                elif date_filter == '6days':
                    query = query.filter(OperationHistory.created_at >= now - timedelta(days=6))
                elif date_filter == '15days':
                    query = query.filter(OperationHistory.created_at >= now - timedelta(days=15))

        paginated = query.order_by(OperationHistory.created_at.desc()).paginate(
            page=page, per_page=per_page, error_out=False)

        history_list = [item.to_dict() for item in paginated.items]

        return jsonify({
            'success': True,
            'data': history_list,
            'pagination': {
                'total': paginated.total,
                'pages': paginated.pages,
                'current_page': paginated.page,
                'per_page': paginated.per_page,
                'has_next': paginated.has_next,
                'has_prev': paginated.has_prev
            }
        })
    except Exception as e:
        logger.error(f"获取历史记录失败: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500



@app.route('/api/history/function_codes', methods=['GET'])
@login_required
def get_function_codes():
    """获取所有功能代码选项"""
    try:
        current_user = get_current_user()
        query = db.session.query(
            OperationHistory.function_code,
            OperationHistory.operation_name
        ).distinct()

        # 非管理员只能看到自己的功能代码
        if current_user.role != 'admin':
            query = query.filter(OperationHistory.user_id == current_user.username)

        codes = query.all()

        return jsonify({
            'success': True,
            'data': [{'code': c[0], 'name': c[1]} for c in codes]
        })
    except Exception as e:
        logger.error(f"获取功能代码失败: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/process_pdf_weiwai', methods=['POST'])
@login_required
def process_pdf_weiwai():
    """委外OCR识别处理"""
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
        'type': 'weiwai_ocr',
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
        processing_queue.put((task_id, file_path, 'weiwai_ocr'))

    return jsonify({
        'success': True,
        'message': '委外OCR识别批量处理已开始',
        'task_id': task_id
    })


@app.route('/process_pdf_purchase', methods=['POST'])
@login_required
def process_pdf_purchase():
    """采购OCR识别处理"""
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
        'type': 'purchase_ocr',
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
        processing_queue.put((task_id, file_path, 'purchase_ocr'))

    return jsonify({
        'success': True,
        'message': '采购OCR识别批量处理已开始',
        'task_id': task_id
    })


@app.route('/progress/<task_id>')
@login_required
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
@login_required
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
                task_data = processing_queue.get()
                if task_data[0] != task_id:  # task_data是元组 (task_id, file_path, task_type)
                    new_queue.put(task_data)
            # 将新队列赋值回原队列
            processing_queue.queue = new_queue.queue

        return jsonify({'success': True, 'message': '取消请求已发送'})
    return jsonify({'error': '任务ID不存在'}), 404


@app.route('/reconcile_weiwai', methods=['POST'])
@login_required
def reconcile_weiwai():
    """委外对账功能 - 替换原有的Excel对账"""
    if 'weiwai_receipt_file' not in request.files or 'weiwai_total_file' not in request.files:
        return jsonify({'error': '请选择两个Excel文件'}), 400

    receipt_file = request.files['weiwai_receipt_file']
    total_file = request.files['weiwai_total_file']

    if receipt_file.filename == '' or total_file.filename == '':
        return jsonify({'error': '请选择两个Excel文件'}), 400

    if (receipt_file and receipt_file.filename.lower().endswith(('.xlsx', '.xls')) and
            total_file and total_file.filename.lower().endswith(('.xlsx', '.xls'))):

        # 保存上传的文件
        receipt_path = os.path.join(app.config['UPLOAD_FOLDER'], receipt_file.filename)
        total_path = os.path.join(app.config['UPLOAD_FOLDER'], total_file.filename)

        receipt_file.save(receipt_path)
        total_file.save(total_path)

        # 处理对账 - 使用原有的func2功能
        try:
            from functions.func2 import reconcile_excel_files
            output_path, log_path = reconcile_excel_files(
                receipt_path=receipt_path,
                total_path=total_path,
                output_dir=app.config['OUTPUT_FOLDER']
            )

            # 提取文件名
            output_filename = os.path.basename(output_path)
            log_filename = os.path.basename(log_path)

            # 保存历史记录
            save_history(
                task_id=str(uuid.uuid4()),
                function_code='weiwai_reconcile',
                operation_name='委外对账',
                files=[
                    {'name': '对账结果.xlsx', 'path': output_filename},
                    {'name': '处理日志.txt', 'path': log_filename}
                ]
            )

            return jsonify({
                'success': True,
                'message': '委外对账完成',
                'download_url': f'/download/{output_filename}',
                'log_url': f'/download/{log_filename}'
            })
        except Exception as e:
            logger.error(f"委外对账失败: {str(e)}", exc_info=True)
            return jsonify({'error': f'委外对账失败: {str(e)}'}), 500

    return jsonify({'error': '请上传Excel文件'}), 400


@app.route('/reconcile_weiwai_receipt', methods=['POST'])
@login_required
def reconcile_weiwai_receipt():
    """委外收据对账功能 - 替换原有的送货单对账"""
    if 'weiwai_receipt_input_file' not in request.files or 'weiwai_delivery_file' not in request.files:
        return jsonify({'error': '请选择两个Excel文件'}), 400

    receipt_file = request.files['weiwai_receipt_input_file']
    delivery_file = request.files['weiwai_delivery_file']

    if receipt_file.filename == '' or delivery_file.filename == '':
        return jsonify({'error': '请选择两个Excel文件'}), 400

    if (receipt_file and receipt_file.filename.lower().endswith(('.xlsx', '.xls')) and
            delivery_file and delivery_file.filename.lower().endswith(('.xlsx', '.xls'))):

        # 保存上传的文件
        receipt_path = os.path.join(app.config['UPLOAD_FOLDER'], receipt_file.filename)
        delivery_path = os.path.join(app.config['UPLOAD_FOLDER'], delivery_file.filename)

        receipt_file.save(receipt_path)
        delivery_file.save(delivery_path)

        # 处理对账 - 使用原有的func4功能
        try:
            from functions.func4 import reconcile_delivery_files
            result_file, log_file = reconcile_delivery_files(
                receipt_path=receipt_path,
                total_path=delivery_path,  # 注意：这里保持参数名不变，但实际是送货单
                output_dir=app.config['OUTPUT_FOLDER']
            )

            if not result_file or not log_file:
                return jsonify({'error': '委外收据对账处理失败'}), 500

            # 提取文件名
            result_filename = os.path.basename(result_file)
            log_filename = os.path.basename(log_file)

            # 保存历史记录
            save_history(
                task_id=str(uuid.uuid4()),
                function_code='weiwai_receipt_reconcile',
                operation_name='委外收据对账',
                files=[
                    {'name': '对账结果.xlsx', 'path': result_filename},
                    {'name': '处理日志.txt', 'path': log_filename}
                ]
            )

            return jsonify({
                'success': True,
                'message': '委外收据对账完成',
                'download_url': f'/download/{result_filename}',
                'log_url': f'/download/{log_filename}'
            })
        except Exception as e:
            logger.error(f"委外收据对账失败: {str(e)}", exc_info=True)
            return jsonify({'error': f'委外收据对账失败: {str(e)}'}), 500

    return jsonify({'error': '请上传Excel文件'}), 400


@app.route('/reconcile_purchase', methods=['POST'])
@login_required
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
            from functions.func5 import reconcile_purchase_files
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

            # 保存历史记录
            save_history(
                task_id=str(uuid.uuid4()),
                function_code='purchase_reconcile',
                operation_name='采购对账',
                files=[
                    {'name': '对账结果.xlsx', 'path': result_filename},
                    {'name': '处理日志.txt', 'path': log_filename}
                ]
            )

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


@app.route('/reconcile_purchase_receipts', methods=['POST'])
@login_required
def reconcile_purchase_receipts():
    """采购收据对账功能 - func7"""
    if 'delivery_file' not in request.files or 'inventory_file' not in request.files:
        return jsonify({'error': '请选择两个Excel文件'}), 400

    delivery_file = request.files['delivery_file']
    inventory_file = request.files['inventory_file']

    if delivery_file.filename == '' or inventory_file.filename == '':
        return jsonify({'error': '请选择两个Excel文件'}), 400

    if (delivery_file and delivery_file.filename.lower().endswith(('.xlsx', '.xls')) and
            inventory_file and inventory_file.filename.lower().endswith(('.xlsx', '.xls'))):

        # 保存上传的文件
        delivery_path = os.path.join(app.config['UPLOAD_FOLDER'], delivery_file.filename)
        inventory_path = os.path.join(app.config['UPLOAD_FOLDER'], inventory_file.filename)

        delivery_file.save(delivery_path)
        inventory_file.save(inventory_path)

        # 处理对账
        try:
            from functions.func7 import reconcile_purchase_receipts
            # 修正参数名
            result_file, log_file, error_msg = reconcile_purchase_receipts(
                delivery_file=delivery_path,  # 修正参数名
                inventory_file=inventory_path,  # 修正参数名
                output_dir=app.config['OUTPUT_FOLDER']
            )

            if error_msg:
                return jsonify({'error': error_msg}), 500

            if not result_file or not log_file:
                return jsonify({'error': '采购收据对账处理失败'}), 500

            # 提取文件名
            result_filename = os.path.basename(result_file)
            log_filename = os.path.basename(log_file)

            # 保存历史记录
            save_history(
                task_id=str(uuid.uuid4()),
                function_code='purchase_receipt_reconcile',
                operation_name='采购收据对账',
                files=[
                    {'name': '对账结果.xlsx', 'path': result_filename},
                    {'name': '处理日志.txt', 'path': log_filename}
                ]
            )

            return jsonify({
                'success': True,
                'message': '采购收据对账完成',
                'download_url': f'/download/{result_filename}',
                'log_url': f'/download/{log_filename}'
            })
        except Exception as e:
            logger.error(f"采购收据对账失败: {str(e)}", exc_info=True)
            return jsonify({'error': f'采购收据对账失败: {str(e)}'}), 500

    return jsonify({'error': '请上传Excel文件'}), 400


@app.route('/negative_excel', methods=['POST'])
@login_required
def negative_excel():
    """数量取负功能"""
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
            from functions.func3 import negative_quantity
            output_path = negative_quantity(
                input_path=excel_path,
                output_dir=app.config['OUTPUT_FOLDER']
            )

            # 提取文件名
            filename = os.path.basename(output_path)

            # 保存历史记录
            save_history(
                task_id=str(uuid.uuid4()),
                function_code='negative_excel',
                operation_name='数量取负',
                files=[{'name': '处理结果.xlsx', 'path': filename}]
            )

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


@app.route('/process_invoice_pricing', methods=['POST'])
@login_required
def process_invoice_pricing():
    """单价处理功能 - func8"""
    if 'price_list_file' not in request.files or 'invoice_file' not in request.files:
        return jsonify({'error': '请选择两个Excel文件'}), 400

    price_list_file = request.files['price_list_file']
    invoice_file = request.files['invoice_file']

    if price_list_file.filename == '' or invoice_file.filename == '':
        return jsonify({'error': '请选择两个Excel文件'}), 400

    if (price_list_file and price_list_file.filename.lower().endswith(('.xlsx', '.xls')) and
            invoice_file and invoice_file.filename.lower().endswith(('.xlsx', '.xls'))):

        # 保存上传的文件
        price_list_path = os.path.join(app.config['UPLOAD_FOLDER'], price_list_file.filename)
        invoice_path = os.path.join(app.config['UPLOAD_FOLDER'], invoice_file.filename)

        price_list_file.save(price_list_path)
        invoice_file.save(invoice_path)

        # 处理单价
        try:
            from functions.func8 import process_invoice_pricing
            # 传递配置文件路径
            result_file, error_msg = process_invoice_pricing(
                price_list_file=price_list_path,
                invoice_file=invoice_path,
                output_dir=app.config['OUTPUT_FOLDER'],
                config_file=app.config['INVOICE_PRICING_CONFIG_PATH']  # 新增：传递配置文件
            )

            if error_msg:
                return jsonify({'error': error_msg}), 500

            if not result_file:
                return jsonify({'error': '单价处理失败'}), 500

            # 提取文件名
            result_filename = os.path.basename(result_file)

            # 保存历史记录
            save_history(
                task_id=str(uuid.uuid4()),
                function_code='invoice_pricing',
                operation_name='单价处理',
                files=[{'name': '单价处理结果.xlsx', 'path': result_filename}]
            )

            return jsonify({
                'success': True,
                'message': '单价处理完成',
                'download_url': f'/download/{result_filename}'
            })
        except Exception as e:
            logger.error(f"单价处理失败: {str(e)}", exc_info=True)
            return jsonify({'error': f'单价处理失败: {str(e)}'}), 500

    return jsonify({'error': '请上传Excel文件'}), 400


@app.route('/download/<filename>')
@login_required
def download_file(filename):
    """文件下载功能"""
    try:
        # 解码URL编码的文件名
        decoded_filename = urllib.parse.unquote(filename)
    except:
        decoded_filename = filename

    # 直接使用解码后的文件名构造路径
    file_path = os.path.join(app.config['OUTPUT_FOLDER'], decoded_filename)

    # 安全检查：防止路径遍历攻击
    # 确保文件路径在输出目录内
    if not os.path.abspath(file_path).startswith(os.path.abspath(app.config['OUTPUT_FOLDER'])):
        logger.error(f"非法文件路径: {file_path}")
        return jsonify({'error': '非法文件路径'}), 403

    if os.path.exists(file_path):
        # 使用原始文件名作为下载文件名
        return send_file(
            file_path,
            as_attachment=True,
            download_name=decoded_filename  # 保留原始文件名
        )

    # 如果文件不存在，尝试使用安全文件名查找
    safe_filename = secure_filename(decoded_filename)
    safe_file_path = os.path.join(app.config['OUTPUT_FOLDER'], safe_filename)

    if os.path.exists(safe_file_path):
        logger.warning(f"使用安全文件名访问文件: {safe_filename}")
        return send_file(
            safe_file_path,
            as_attachment=True,
            download_name=decoded_filename  # 保留原始文件名
        )

    logger.error(f"文件不存在: {decoded_filename} (安全文件名: {safe_filename})")
    return jsonify({'error': '文件不存在'}), 404


@app.route('/health')
def health_check():
    """健康检查端点"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),  # 使用本地时间
        'active_tasks': len([t for t in tasks.values() if t.get('status') == 'processing']),
        'pending_tasks': processing_queue.qsize()
    })


def cleanup_old_files():
    """定期清理旧文件"""
    while True:
        try:
            time.sleep(3600)  # 每小时清理一次

            now = time.time()
            cutoff_time = now - (360 * 3600)  # 24小时前

            for folder in [app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER']]:
                if not os.path.exists(folder):
                    continue

                for filename in os.listdir(folder):
                    filepath = os.path.join(folder, filename)
                    if os.path.isfile(filepath):
                        # 删除超过24小时的文件
                        file_time = os.path.getmtime(filepath)
                        if file_time < cutoff_time:
                            try:
                                os.remove(filepath)
                                logger.info(f"清理旧文件: {filename}")
                            except Exception as e:
                                logger.warning(f"无法删除文件 {filename}: {str(e)}")
        except Exception as e:
            logger.error(f"文件清理任务错误: {str(e)}")
            time.sleep(3600)  # 出错后等待一小时再重试


def ensure_config_files():
    """确保配置文件存在"""
    configs = {
        app.config['WEIWAI_CONFIG_PATH']: {
            "company_list": ["众诺", "其他公司"],
            "company_titles": {
                "众诺": ["品号", "品名", "规格", "单位", "数量", "批号"],
                "其他公司": ["物料编码", "物料名称", "规格型号", "单位", "数量", "批号"]
            },
            "years": [2023, 2024, 2025]
        },
        app.config['PURCHASE_CONFIG_PATH']: {
            "company_list": ["供应商A", "供应商B"],
            "company_titles": {
                "供应商A": ["产品编码", "产品名称", "规格", "单位", "数量", "批次"],
                "供应商B": ["物料代码", "物料描述", "规格", "计量单位", "数量", "批号"]
            },
            "years": [2023, 2024, 2025]
        },
        app.config['INVOICE_PRICING_CONFIG_PATH']: {  # 新增：单价处理配置文件
            "price_list": {
                "code_column": "价表编码",
                "price_column": "单价"
            },
            "invoice": {
                "code_column": "物料编码",
                "price_column": "单价",
                "quantity_column": "数量",
                "amount_column": "金额"
            }
        }
    }

    for config_path, default_config in configs.items():
        if not os.path.exists(config_path):
            try:
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(default_config, f, ensure_ascii=False, indent=2)
                logger.info(f"已创建默认配置文件: {config_path}")
            except Exception as e:
                logger.error(f"创建配置文件失败: {config_path} - {str(e)}")


@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': '文件大小超过限制(50MB)'}), 413


@app.errorhandler(500)
def internal_error(e):
    logger.error(f"服务器内部错误: {str(e)}")
    return jsonify({'error': '服务器内部错误'}), 500


@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': '请求的资源不存在'}), 404


@app.errorhandler(401)
def unauthorized(e):
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify({'success': False, 'error': '未登录', 'redirect': url_for('login')}), 401
    return redirect(url_for('login', next=request.url))


if __name__ == '__main__':
    # 确保配置文件存在
    ensure_config_files()

    # 确保用户配置文件存在
    user_manager.load_users()  # 这会自动创建默认用户配置文件

    # 启动文件清理线程
    cleanup_thread = threading.Thread(target=cleanup_old_files, daemon=True)
    cleanup_thread.start()

    logger.info("财务对账系统启动成功")
    logger.info(f"上传目录: {app.config['UPLOAD_FOLDER']}")
    logger.info(f"输出目录: {app.config['OUTPUT_FOLDER']}")
    logger.info(f"默认用户: admin/admin123, user1/user123, operator/operator123")

    app.run(debug=True, host='0.0.0.0', port=5000)