import os
import glob

# 当前目录下查找所有以 "_4k.json" 结尾的 .json 文件
pattern = "*_1000k.json"
files = glob.glob(pattern)
length = len(pattern) -1
if not files:
    print("未找到匹配 '_1000k.json' 的文件。")
else:
    print(f"找到 {len(files)} 个文件，正在重命名...")
    for old_name in files:
        # 确保是文件（而非目录）
        if os.path.isfile(old_name):
            # 构造新文件名：将 _4k.json 替换为 _200k.json
            new_name = old_name[:-length] + "_200k.json"  # 去掉 '_4k.json' (8字符) 加上新后缀
            # 或者使用更灵活的方式：
            # new_name = old_name.replace("_4k.json", "_200k.json", 1)
            
            os.rename(old_name, new_name)
            print(f"重命名: {old_name} → {new_name}")
