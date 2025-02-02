import os
import glob
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from tqdm import tqdm
import argparse
from mmlu_metadata import group_subtasks, subject_to_macro, subject_names


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

                this_sn = ''
                for sn in subject_names:
                    if sn in file_path:
                        this_sn = sn
                        break
                if root not in all_data.keys():
                    all_data[root] = {}
                if this_sn == '':
                    raise Exception('error')
                all_data[root].update({this_sn: data})
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
parser.add_argument("--model_size_uplimit", default=31.622776601683793)
parser.add_argument("--model_size_downlimit", default=-1000)
args = parser.parse_args()

removed_models = ['microsoft__phi-1_5', 'microsoft__phi-2', 'Qwen__Qwen1.5-0.5B', 'Qwen__Qwen1.5-1.8B',\
                'Qwen__Qwen1.5-4B', 'stabilityai__stablelm-2-1_6b', 'facebook/xglm-564M']

csv_file = 'base_llm_benchmark_eval.csv'
eval_data = load_csv(csv_file)
eval_data.dropna(subset=['FLOPs (1E21)'])

directory_path = 'mmlu'
all_data = traverse_directory(directory_path)

all_brier_scores, all_sanities, all_accs, all_probs = {}, {}, {}, {}
for model_name, model_data in all_data.items():
    model_name = model_name.split('/')[1]
    brier_scores, sanities, accs, probs = [], [], [], []
    for task, data in model_data.items():
        for d in data:
            ans = [0, 0, 0, 0]
            label = int(d['target'])
            ans[label] = 1
            logprobs = [float(item[0][0]) for item in d['resps']]
            linearprobs = [math.e**i for i in logprobs]
            
            idx = linearprobs.index(max(linearprobs))
            acc = label == idx        
            accs.append((task, d["doc_id"], acc))
            
            redist_linearprobs = [i / sum(linearprobs) for i in linearprobs]
            if args.redist:
                brier_score = (np.array(redist_linearprobs)[label]-1)**2
            else:
                brier_score = (np.array(linearprobs)[label]-1)**2
            # assert sum(linearprobs) <= 1
            
            brier_scores.append((task, d["doc_id"], brier_score))
            sanities.append((task, d["doc_id"], sum(linearprobs)))
            probs.append((task, d["doc_id"], redist_linearprobs))

    brier_scores = sorted(brier_scores, key=lambda x: x[0]+str(x[1]))
    sanities = sorted(sanities, key=lambda x: x[0]+str(x[1]))
    accs = sorted(accs, key=lambda x: x[0]+str(x[1]))
    all_brier_scores[model_name] = brier_scores
    all_sanities[model_name] = sanities
    all_accs[model_name] = accs
    probs = sorted(probs, key=lambda x: x[0]+str(x[1]))
    all_probs[model_name] = probs


### calculate the averaged brier score of each question
question_num = 14042
ques_briers = []
for i in tqdm(range(question_num)):
    value = []
    init_key = -1
    for model_name, sub_list in all_accs.items():
        model_size = eval_data.loc[eval_data['Model'] == model_name.replace('__', '/'), 'FLOPs (1E21)'].iloc[0]
        if model_size > args.model_size_uplimit or model_size < args.model_size_downlimit or model_name in removed_models:
            continue
        key = list(sub_list[i][:2])
        if init_key != -1:
            assert key == init_key
        else:
            init_key = key
        value.append(sub_list[i][-1])
    key.append(sum(value) / len(value))
    ques_briers.append(key)

## for table 2
# ques_probs = []
# for i in tqdm(range(question_num)):
#     prob_dists = []
#     init_key = -1
#     for model_name, sub_list in all_probs.items():
#         model_size = eval_data.loc[eval_data['Model'] == model_name.replace('__', '/'), 'FLOPs (1E21)'].iloc[0]
#         if model_size > args.model_size_uplimit or model_size < args.model_size_downlimit or model_name in removed_models:
#             continue
#         key = list(sub_list[i][:2])
#         if init_key != -1:
#             assert key == init_key
#         else:
#             init_key = key
#         prob_dists.append(sub_list[i][-1])
        
#     averaged_probs = np.mean(np.array(prob_dists), axis=0)
#     key.append(averaged_probs)
#     ques_probs.append(key)

# for q in ques_probs:
#     if q[0] == 'mmlu_global_facts' and q[1] == 66:
#         print('find easy!')
#         print(q[2])
#     if q[0] == 'mmlu_conceptual_physics' and q[1] == 44:
#         print('find hard!')
#         print(q[2])
# exit(0)
##

ques_sanities = []
for i in tqdm(range(question_num)):
    value = []
    init_key = -1
    for model_name, sub_list in all_sanities.items():
        model_size = eval_data.loc[eval_data['Model'] == model_name.replace('__', '/'), 'FLOPs (1E21)'].iloc[0]
        if model_size > args.model_size_uplimit or model_size < args.model_size_downlimit or model_name in removed_models:
            continue
        key = list(sub_list[i][:2])
        if init_key != -1:
            assert key == init_key
        else:
            init_key = key
        value.append(sub_list[i][-1])
    key.append(sum(value) / len(value))
    ques_sanities.append(key)

ques = [ques_briers[i][:3]+[ques_sanities[i][-1]] for i in range(len(ques_briers))]

ques = sorted(ques, key=lambda x: x[2])
ques_briers = [i[:3] for i in ques]
ques_sanities = [i[:2]+[i[-1]] for i in ques]

eval_data.set_index('Model', inplace=True, drop=False)
saved_question_idx = np.linspace(0, question_num, args.group_num+1)
saved_question_idx = [[int(math.floor(saved_question_idx[i])), int(math.floor(saved_question_idx[i+1]))] for i in range(len(saved_question_idx)-1)]

for idx in tqdm(saved_question_idx):
    saved_question_brier = {}
    for model_name, model_data in all_accs.items():
        val, cnt = 0, 0
        for i in range(idx[0], idx[1]):
            ques = ques_briers[i]
            for (subject_name, ques_id, brier) in model_data:
                if subject_name == ques[0] and ques_id == ques[1]:
                    val += brier
                    cnt += 1

        assert cnt == idx[1] - idx[0]
        saved_question_brier[model_name] = val / cnt

    for model_name, qs_brier in saved_question_brier.items():
        model_name = model_name.replace('__', '/')
        eval_data.loc[model_name, f'{str(idx[0])}_{str(idx[1])}_acc'] = qs_brier

    saved_question_san = {}
    for model_name, model_data in all_sanities.items():
        val, cnt = 0, 0
        for i in range(idx[0], idx[1]):
            ques = ques_sanities[i]
            for (subject_name, ques_id, san) in model_data:
                if subject_name == ques[0] and ques_id == ques[1]:
                    val += san
                    cnt += 1

        assert cnt == idx[1] - idx[0]
        saved_question_san[model_name] = val / cnt
        
    for model_name, san in saved_question_san.items():
        model_name = model_name.replace('__', '/')
        eval_data.loc[model_name, f'{str(idx[0])}_{str(idx[1])}_san'] = san

# save brier
saved_question_brier = {}
for model_name, model_data in all_brier_scores.items():
    val, cnt = 0, 0
    for d in model_data:
        val += d[2]
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
    eval_data.to_csv(f'acc_mmlu_instance_brier_{args.model_size_downlimit}_{args.model_size_uplimit}_{args.group_num}_redist.csv', index=False)
else:
    eval_data.to_csv(f'acc_mmlu_instance_brier_{args.model_size_downlimit}_{args.model_size_uplimit}_{args.group_num}_undist.csv', index=False)