import os
import json

# 设置要更改的路径
old_path = "..\\..\\D:\\work\\python study\\pythonProject\\json\\E（1）\\E\\1\\"
new_path = ""
folder_path = "D:\\work\\python study\\pythonProject\\json\\E（1）\\E\\1"

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    if filename.endswith('.json'):
        file_path = os.path.join(folder_path, filename)

        # 读取 JSON 文件
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)


        # 递归查找并替换路径
        def replace_paths(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(value, str) and old_path in value:
                        obj[key] = value.replace(old_path, new_path)
                    else:
                        replace_paths(value)
            elif isinstance(obj, list):
                for item in obj:
                    replace_paths(item)


        replace_paths(data)

        # 将修改后的数据写回 JSON 文件
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

print("路径更改完成。")