import json
import numpy as np
import matplotlib.pyplot as plt

# 读取 JSON 文件
def load_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# 提取输入长度和输出长度
def extract_lengths(data):
    input_lengths = [item['prompt_len'] for item in data]
    output_lengths = [item['output_len'] for item in data]
    return input_lengths, output_lengths

# 计算 CDF
def compute_cdf(lengths):
    hist, bin_edges = np.histogram(lengths, bins=100, density=True)
    cdf = np.cumsum(hist * np.diff(bin_edges))
    return bin_edges[1:], cdf

# 绘制 CDF 图
def plot_cdf(input_edges, input_cdf, output_edges, output_cdf):
    plt.figure(figsize=(10, 6))
    plt.plot(input_edges, input_cdf, label='Input Length CDF', color='blue')
    plt.plot(output_edges, output_cdf, label='Output Length CDF', color='orange')
    plt.xlabel('Length')
    plt.ylabel('CDF')
    plt.title('CDF of Input and Output Lengths')
    plt.legend()
    plt.grid(True)
    plt.savefig("./cdf_sg_90k_part1.png")

# 主函数
def main():
    file_path = '/share/dataset/ShareGPT52K/replicated_data.json'  # 替换为你的 JSON 文件路径
    file_path = '/share/dataset/ShareGPT52K/sg_90k_part1.json'  # 替换为你的 JSON 文件路径
    data = load_json_file(file_path)
    input_lengths, output_lengths = extract_lengths(data)
    
    input_edges, input_cdf = compute_cdf(input_lengths)
    output_edges, output_cdf = compute_cdf(output_lengths)
    
    plot_cdf(input_edges, input_cdf, output_edges, output_cdf)

if __name__ == "__main__":
    main()