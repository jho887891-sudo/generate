import os
import sys
from PIL import Image
from pathlib import Path


def check_image_size(filepath):
    """检查图片尺寸的增强函数"""
    try:
        # 清理文件路径
        filepath = str(filepath).strip().strip('"').strip("'")

        # 处理可能的转义字符
        if not os.path.isabs(filepath):
            # 如果是相对路径，转为绝对路径
            filepath = os.path.abspath(filepath)

        print(f"正在检查文件: {filepath}")
        print(f"文件是否存在: {os.path.exists(filepath)}")

        if not os.path.exists(filepath):
            # 尝试使用原始字符串
            filepath = r"{}".format(filepath)
            print(f"尝试使用原始字符串路径: {filepath}")
            print(f"现在文件是否存在: {os.path.exists(filepath)}")

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"文件不存在: {filepath}")

        # 检查文件扩展名
        ext = os.path.splitext(filepath)[1].lower()
        print(f"文件扩展名: {ext}")

        # 打开图片
        with Image.open(filepath) as img:
            width, height = img.size

            # 获取更多信息
            info = {
                '文件名': os.path.basename(filepath),
                '完整路径': filepath,
                '尺寸': f"{width} × {height} 像素",
                '宽度': f"{width} 像素",
                '高度': f"{height} 像素",
                '格式': img.format,
                '颜色模式': img.mode,
                '文件大小': f"{os.path.getsize(filepath) / 1024:.2f} KB",
                'DPI': img.info.get('dpi', '未知'),
                '位深度': img.mode  # 如 RGB=24位, RGBA=32位
            }

            # 输出结果
            print("\n" + "=" * 60)
            print("图片详细信息:")
            print("=" * 60)
            for key, value in info.items():
                print(f"{key:>10}: {value}")
            print("=" * 60)

            return width, height

    except FileNotFoundError as e:
        print(f"❌ 错误: 文件不存在")
        print(f"请检查路径是否正确: {filepath}")
        print("提示:")
        print("1. 确保文件路径没有中文字符转义问题")
        print("2. 尝试将路径用双引号括起来")
        print("3. 尝试使用正斜杠 (/) 代替反斜杠")
        return None, None

    except Image.UnidentifiedImageError:
        print(f"❌ 错误: 无法识别的图片格式")
        print("这可能不是有效的图片文件，或者文件已损坏")
        return None, None

    except PermissionError:
        print(f"❌ 错误: 没有权限访问文件")
        print("请确保文件没有被其他程序占用")
        return None, None

    except Exception as e:
        print(f"❌ 未知错误: {type(e).__name__}")
        print(f"错误详情: {str(e)}")
        return None, None


def main():
    """主函数"""
    print("图片尺寸检查工具")
    print("版本: 2.0")
    print("-" * 60)

    if len(sys.argv) > 1:
        # 从命令行参数获取文件路径
        for filepath in sys.argv[1:]:
            check_image_size(filepath)
            print("\n")
    else:
        # 交互式输入
        print("请输入图片文件路径（可以直接拖拽文件到窗口中）:")
        filepath = input("> ").strip()

        # 处理拖拽文件可能带有的引号
        if filepath.startswith('"') and filepath.endswith('"'):
            filepath = filepath[1:-1]
        elif filepath.startswith("'") and filepath.endswith("'"):
            filepath = filepath[1:-1]

        check_image_size(filepath)


if __name__ == "__main__":
    # 检查是否安装了PIL
    try:
        from PIL import Image

        main()
    except ImportError:
        print("❌ 需要安装Pillow库")
        print("请运行: pip install Pillow")
        print("或者: pip3 install Pillow")

        # 询问是否自动安装
        response = input("是否立即安装？(y/n): ").lower()
        if response == 'y':
            import subprocess

            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow"])
                print("✅ Pillow安装成功！请重新运行程序")
            except Exception as e:
                print(f"安装失败: {e}")