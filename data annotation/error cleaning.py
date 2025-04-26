import json
import os

def extract_error_files(folder_path, error_message, output_file):
    """
    扫描指定文件夹中的所有JSON文件，提取包含特定错误信息的文件
    
    参数:
    folder_path (str): JSON文件所在的文件夹路径
    error_message (str): 要查找的错误信息
    output_file (str): 输出结果的TXT文件路径
    """
    # 创建一个列表来存储匹配的文件名
    matching_files = []
    
    # 获取文件夹中的所有文件
    all_files = os.listdir(folder_path)
    json_files = [f for f in all_files if f.endswith('.json')]
    
    print(f"开始扫描文件夹: {folder_path}")
    print(f"找到 {len(json_files)} 个JSON文件")
    
    # 处理每个JSON文件
    for index, json_file in enumerate(json_files):
        file_path = os.path.join(folder_path, json_file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # 读取文件内容
                content = f.read()
                
                # 检查是否包含指定的错误信息字符串
                if error_message in content:
                    matching_files.append(json_file)
                    print(f"找到匹配文件: {json_file}")
        except Exception as e:
            print(f"无法处理文件 {json_file}: {str(e)}")
        
        # 显示进度
        if (index + 1) % 20 == 0 or index + 1 == len(json_files):
            print(f"已处理 {index + 1}/{len(json_files)} 个文件...")
    
    # 将匹配的文件名写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for file_name in matching_files:
            f.write(file_name + '\n')
    
    print(f"\n扫描完成！")
    print(f"总共找到 {len(matching_files)} 个包含错误信息 '{error_message}' 的文件")
    print(f"结果已保存至: {output_file}")

# 设置参数
folder_path = "/Users/jacquesqiu/Downloads/label final"  # 当前目录
error_message = "Max retries exceeded"  # 要查找的错误信息
output_file = "error_files_list.txt"  # 输出文件名

# 执行提取操作
extract_error_files(folder_path, error_message, output_file)
