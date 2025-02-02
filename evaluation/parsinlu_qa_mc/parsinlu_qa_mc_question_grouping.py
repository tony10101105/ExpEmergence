import os
import glob
import json
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from tqdm import tqdm
import argparse


def load_csv(file_path):
    data = pd.read_csv(file_path)
    return data

def traverse_directory(top_directory_path):
    all_data = {}
    for root, dirs, files in os.walk(top_directory_path):
        for file in files:
            if file.endswith('.jsonl') and 'results_' not in file:
                file_path = os.path.join(root, file)
                data = read_jsonl(file_path)
                all_data[root] = data
    return all_data

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
        return data


parser = argparse.ArgumentParser()
parser.add_argument("--redist", default=True, help='whether to redistribute probs')
parser.add_argument("--group_num", default=10, help='group number')
parser.add_argument("--model_size_uplimit", default=199.52623149688787) # 199.52623149688787: 2.3, 125.89254117941675: 2.1, 79.43282347242814: 1.9
parser.add_argument("--model_size_downlimit", default=-1000)
args = parser.parse_args()

removed_models = ['microsoft__phi-1_5', 'microsoft__phi-2', 'Qwen__Qwen1.5-0.5B', 'Qwen__Qwen1.5-1.8B',\
                'Qwen__Qwen1.5-4B', 'stabilityai__stablelm-2-1_6b', 'facebook/xglm-564M']

csv_file = 'base_llm_benchmark_eval.csv'
eval_data = load_csv(csv_file)
eval_data.dropna(subset=['FLOPs (1E21)'])

directory_path = 'parsinlu_qa_mc'
all_data = traverse_directory(directory_path)

all_brier_scores, all_sanities, all_accs = {}, {}, {}
for model_name, model_data in all_data.items():
    model_name = model_name.split('/')[1]
    brier_scores, sanities, accs = [], [], []
    for d in model_data:
        label = int(d['target'])
        ans = [0, 0, 0, 0]
        ans[label] = 1
        logprobs = [float(item[0][0]) for item in d['resps']]
        linearprobs = [math.e**i for i in logprobs]
        
        idx = linearprobs.index(max(linearprobs))
        acc = label == idx        
        accs.append(acc)        

        redist_linearprobs = [i / sum(linearprobs) for i in linearprobs]
        if args.redist:
            brier_score = (np.array(redist_linearprobs)[label]-1)**2
        else:
            brier_score = (np.array(linearprobs)[label]-1)**2
        
        # assert sum(linearprobs) <= 1
        brier_scores.append((d["doc_id"], brier_score))
        sanities.append((d["doc_id"], sum(linearprobs)))

    brier_scores = sorted(brier_scores, key=lambda x: x[0])
    sanities = sorted(sanities, key=lambda x: x[0])
    all_brier_scores[model_name] = brier_scores
    all_sanities[model_name] = sanities
    all_accs[model_name] = sum(accs) / len(accs)

### calculate the averaged brier score of each question
question_num = 1050
ques_briers = []
for i in tqdm(range(question_num)):
    value = []
    init_key = -1
    for model_name, sub_list in all_brier_scores.items():
        model_size = eval_data.loc[eval_data['Model'] == model_name.replace('__', '/'), 'FLOPs (1E21)'].iloc[0]
        if model_size > args.model_size_uplimit or model_size < args.model_size_downlimit or model_name in removed_models:
            continue
        key = [sub_list[i][0]]
        if init_key != -1:
            assert key == init_key
        else:
            init_key = key
        value.append(sub_list[i][-1])
    key.append(sum(value) / len(value))
    ques_briers.append(key)

ques_sanities = []
for i in tqdm(range(question_num)):
    value = []
    init_key = -1
    for model_name, sub_list in all_sanities.items():
        model_size = eval_data.loc[eval_data['Model'] == model_name.replace('__', '/'), 'FLOPs (1E21)'].iloc[0]
        if model_size > args.model_size_uplimit or model_size < args.model_size_downlimit or model_name in removed_models:
            continue
        key = [sub_list[i][0]]
        if init_key != -1:
            assert key == init_key
        else:
            init_key = key
        value.append(sub_list[i][-1])
    key.append(sum(value) / len(value))
    ques_sanities.append(key)

ques = [ques_briers[i][:2]+[ques_sanities[i][-1]] for i in range(len(ques_briers))]

ques = sorted(ques, key=lambda x: x[1])
ques_briers = [i[:2] for i in ques]
ques_sanities = [[i[0]]+[i[-1]] for i in ques]

eval_data.set_index('Model', inplace=True, drop=False)
saved_question_idx = np.linspace(0, question_num, args.group_num+1)
saved_question_idx = [[int(math.floor(saved_question_idx[i])), int(math.floor(saved_question_idx[i+1]))] for i in range(len(saved_question_idx)-1)]

for j, idx in tqdm(enumerate(saved_question_idx)):
    saved_question_brier = {}
    for model_name, model_data in all_brier_scores.items():
        val, cnt = 0, 0
        for i in range(idx[0], idx[1]):
            ques = ques_briers[i]
            for d in model_data:
                if d[0] == ques[0]:
                    val += d[1]
                    cnt += 1

        assert cnt == idx[1] - idx[0]
        saved_question_brier[model_name] = val / cnt

    for model_name, brier in saved_question_brier.items():
        model_name = model_name.replace('__', '/')
        eval_data.loc[model_name, f'{str(idx[0])}_{str(idx[1])}_brier'] = brier
    
    saved_question_san = {}
    for model_name, model_data in all_sanities.items():
        val, cnt = 0, 0
        for i in range(idx[0], idx[1]):
            ques = ques_sanities[i]
            for d in model_data:
                if d[0] == ques[0]:
                    val += d[1]
                    cnt += 1

        assert cnt == idx[1] - idx[0]
        saved_question_san[model_name] = val / cnt
        
    for model_name, san in saved_question_san.items():
        model_name = model_name.replace('__', '/')
        eval_data.loc[model_name, f'{str(idx[0])}_{str(idx[1])}_san'] = san

# save acc
for model_name, acc in all_accs.items():
    model_name = model_name.replace('__', '/')
    eval_data.loc[model_name, 'acc'] = acc

# save brier
saved_question_brier = {}
for model_name, model_data in all_brier_scores.items():
    val, cnt = 0, 0
    for d in model_data:
        val += d[1]
        cnt += 1
    saved_question_brier[model_name] = val / cnt

for model_name, brier in saved_question_brier.items():
    model_name = model_name.replace('__', '/')
    eval_data.loc[model_name, 'brier'] = brier

# save san
saved_question_san = {}
for model_name, model_data in all_sanities.items():
    val, cnt = 0, 0
    for d in model_data:
        val += d[1]
        cnt += 1
    saved_question_san[model_name] = val / cnt

for model_name, san in saved_question_san.items():
    model_name = model_name.replace('__', '/')
    eval_data.loc[model_name, 'san'] = san

if args.redist:
    eval_data.to_csv(f'parsinlu_qa_mc_instance_brier_{args.model_size_downlimit}_{args.model_size_uplimit}_{args.group_num}_redist.csv', index=False)
else:
    eval_data.to_csv(f'parsinlu_qa_mc_instance_brier_{args.model_size_downlimit}_{args.model_size_uplimit}_{args.group_num}_undist.csv', index=False)