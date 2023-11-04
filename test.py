import argparse

def parse_arguments():
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description='Process a list of integers.')

    # 添加参数配置，使用 '--integers' 作为可选参数，而不是位置参数
    parser.add_argument('--integers', type=int, nargs='+',
                        help='an integer for the accumulator', required=True)
    
    # 解析传入的参数
    args = parser.parse_args()
    return args.integers

if __name__ == "__main__":
    # 解析命令行输入的数字列表
    numbers = parse_arguments()
    
    # 打印数字列表
    print("Received list of numbers:", numbers)
