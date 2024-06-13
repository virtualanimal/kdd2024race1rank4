import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--first_json', default="title_10000/result-checkpoint-2000.json")
parser.add_argument('--second_json',default='title_venue/result-checkpoint-2000.json')
parser.add_argument('--merge_llm_name',default='../dataset/result/merge_llm.json')
args = parser.parse_args()

# 读取第一个 JSON 文件
with open(args.first_json, 'r') as f:
    data1 = json.load(f)

# 读取第二个 JSON 文件
with open(args.second_json, 'r') as f:
    data2 = json.load(f)

# with open('/data/lyp/whoiswho-top-solutions/incorrect_assignment_detection/ChatGLM/title_venu_auther/result-checkpoint-2000_0.7618.json', 'r') as f:
#     data3 = json.load(f)


# 初始化一个空字典用于存储加和后的值
summed_data = {}

# 遍历第一个 JSON 文件的键值对
for key, value in data1.items():
    if key in data2:  # 检查第二个 JSON 文件是否有相同的键
        # 将对应的值相加，并存储到新字典中
        author = {}
        auther1 =  value
        auther2 = data2[key]
        # auther3 = data3[key]
        for key_2,value_2 in auther1.items():
            # author[key_2] = value_2*0.5 + auther2[key_2]*0.5 + auther3[key_2]*0.0
            author[key_2] = value_2 * 0.5 + auther2[key_2] * 0.5
        summed_data[key] = author

# 将结果保存到新的 JSON 文件中
with open(args.merge_llm_name, 'w') as f:
    json.dump(summed_data, f)