import sqlite3
import os
import json
import uuid

# 数据库文件路径
DB_PATH = 'file_database.db'


def init_db():
    """初始化数据库"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # 创建文件表
    c.execute('''CREATE TABLE IF NOT EXISTS files (
        id TEXT PRIMARY KEY,
        filename TEXT NOT NULL,
        file_path TEXT NOT NULL,
        file_type TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        company_name TEXT,
        delivery_date TEXT,
        delivery_no TEXT,
        additional_info TEXT
    )''')

    conn.commit()
    conn.close()


def save_file_to_db(filename: str, file_path: str, file_type: str,
                    company_name: str = None, delivery_date: str = None,
                    delivery_no: str = None, additional_info: dict = None) -> str:
    """保存文件信息到数据库"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # 生成唯一ID
    file_id = str(uuid.uuid4())

    # 转换附加信息为JSON字符串
    add_info_str = json.dumps(additional_info) if additional_info else None

    # 插入数据
    c.execute('''INSERT INTO files 
              (id, filename, file_path, file_type, company_name, delivery_date, delivery_no, additional_info)
              VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
              (file_id, filename, file_path, file_type, company_name, delivery_date, delivery_no, add_info_str))

    conn.commit()
    conn.close()
    return file_id


def get_file_by_id(file_id: str) -> dict:
    """根据ID获取文件信息"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute('''SELECT * FROM files WHERE id = ?''', (file_id,))
    row = c.fetchone()
    conn.close()

    if not row:
        return None

    # 解析结果
    file_info = {
        'id': row[0],
        'filename': row[1],
        'file_path': row[2],
        'file_type': row[3],
        'created_at': row[4],
        'company_name': row[5],
        'delivery_date': row[6],
        'delivery_no': row[7],
        'additional_info': json.loads(row[8]) if row[8] else None
    }

    return file_info


def delete_file_by_id(file_id: str):
    """根据ID删除文件记录"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute('''DELETE FROM files WHERE id = ?''', (file_id,))

    conn.commit()
    conn.close()


# 初始化数据库
init_db()