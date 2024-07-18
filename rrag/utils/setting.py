import os
from rrag.constants import RESULT_DIR
# 指定要创建的文件夹路径


def setting_env_init():
    # print(RESULT_DIR)
    folder_path = os.path.join(RESULT_DIR, "exp_{}".format(len(os.listdir(RESULT_DIR))))

    # 创建文件夹
    try:
        os.makedirs(folder_path, exist_ok=True)  # exist_ok=True表示如果文件夹已存在则不报错
        print(f"文件夹 '{folder_path}' 创建成功。")
    except Exception as e:
        print(f"创建文件夹时出错: {e}")
    return folder_path
