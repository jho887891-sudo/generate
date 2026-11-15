import json
import hashlib

def hash_password(password):
    """密码加密"""
    return hashlib.sha256(password.encode()).hexdigest()

# 创建默认用户
users_data = {
    "users": [
        {
            "username": "admin",
            "password": hash_password("admin123"),  # 加密存储
            "role": "admin",
            "display_name": "系统管理员"
        }
    ],
    "settings": {
        "max_login_attempts": 5,
        "session_timeout": 3600,
        "require_strong_password": False
    }
}

# 保存到文件
with open('users.json', 'w', encoding='utf-8') as f:
    json.dump(users_data, f, ensure_ascii=False, indent=2)

print("用户配置文件已创建！")
print("默认账号: admin")
print("默认密码: admin123")
print("请及时修改默认密码！")