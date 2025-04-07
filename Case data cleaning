import os

def find_small_files(directory_path, max_size_bytes=100):
    """
    遍历指定目录，找出所有小于指定大小的文件
    
    Args:
        directory_path: 要遍历的目录路径
        max_size_bytes: 最大文件大小（字节）
        
    Returns:
        小文件列表，每项包含文件路径和大小
    """
    small_files = []
    
    # 遍历目录
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            
            # 检查文件大小
            try:
                file_size = os.path.getsize(file_path)
                if file_size < max_size_bytes:
                    # 文件大小小于指定值，添加到列表
                    relative_path = os.path.relpath(file_path, directory_path)
                    small_files.append({
                        "文件路径": relative_path,
                        "绝对路径": file_path,
                        "文件大小": f"{file_size} bytes"
                    })
            except Exception as e:
                print(f"无法获取文件 {file_path} 的大小: {e}")
    
    return small_files

def save_to_text_file(small_files, output_file="小文件清单.txt"):
    """
    将小文件列表保存为文本文件
    
    Args:
        small_files: 小文件列表
        output_file: 输出文本文件名
    """
    if not small_files:
        print("未找到小文件")
        return
    
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(f"共找到 {len(small_files)} 个小文件（小于100字节）：\n\n")
        for i, file_info in enumerate(small_files, 1):
            file.write(f"{i}. {file_info['绝对路径']} - {file_info['文件大小']}\n")
    
    print(f"已将小文件清单保存至 {output_file}")
    print(f"共找到 {len(small_files)} 个小文件（小于100字节）")

def main():
    directory_name = "Data"
    
    # 获取当前工作目录
    current_dir = os.getcwd()
    directory_path = os.path.join(current_dir, directory_name)
    
    if not os.path.exists(directory_path):
        print(f"错误: 目录 '{directory_name}' 不存在")
        return
    
    print(f"正在遍历目录: {directory_path}")
    small_files = find_small_files(directory_path, max_size_bytes=100)
    
    # 按文件大小排序
    small_files.sort(key=lambda x: int(x["文件大小"].split()[0]))
    
    # 打印结果
    if small_files:
        print("\n找到的小文件（小于100字节）:")
        print(f"共找到 {len(small_files)} 个小文件\n")
        for i, file_info in enumerate(small_files, 1):
            print(f"{i}. {file_info['绝对路径']} - {file_info['文件大小']}")
        
        # 询问是否保存到文件
        save_option = input("\n是否需要将清单保存到文本文件？(y/n): ")
        if save_option.lower() == 'y':
            save_to_text_file(small_files)
    else:
        print("未找到小于100字节的文件")

if __name__ == "__main__":
    main()
