import os


# 1. 读取 .txt 文件中的编号
def read_indices(file_path):
    with open(file_path, 'r') as file:
        # 读取所有编号，并将其转换为整数列表
        indices = [int(line.strip()) for line in file]
    return indices


# 2. 删除文件夹中的对应文件
def delete_files_by_position(folder_path, positions_to_delete):
    # 获取文件夹中的所有文件名
    all_files = sorted(os.listdir(folder_path))

    # 删除指定位置的文件
    for pos in positions_to_delete:
        # 位置从1开始，但文件夹中的文件是从0开始索引的
        index_to_delete = pos - 1
        if 0 <= index_to_delete < len(all_files):
            file_path = os.path.join(folder_path, all_files[index_to_delete])
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f'Deleted: {file_path}')
            else:
                print(f'File not found: {file_path}')
        else:
            print(f'Position out of range: {pos}')


# 主程序
def main():
    # 设置 .txt 文件路径和文件夹路径
    indices_file_path = 'disease_indices.txt'
    folder_path = 'D:\\ic\\rotate\\disease_t'

    # 读取编号
    positions = read_indices(indices_file_path)

    # 获取最后36个编号
    positions_to_delete = positions[-36:]

    # 删除文件
    delete_files_by_position(folder_path, positions_to_delete)


if __name__ == "__main__":
    main()

