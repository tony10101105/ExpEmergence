# U-shaped-and-Inverted-U-Scaling-behind-Emergent-Abilities-of-Large-Language-Models
Official code of the paper [U-shaped and Inverted-U Scaling behind Emergent Abilities of Large Language Models](https://arxiv.org/abs/2410.01692)

## News
- (Nov. 2024) The paper has been accepted to the NeurIPS'24 ATTRIB workshop as **oral presentation**.

## Overview
This paper explains why LLMs sometimes experience emergent abilities. In short, [deep double descent](https://arxiv.org/abs/1912.02292) on easy questions and [U-shaped scaling](https://arxiv.org/abs/2211.02011) on hard questions offset each other, leading to initially flat overall performance. The performance soar occurs around the second descent on easy questions. We further provide a simple pipeline to forecast the occurrence of emergent abilities.

<p float="left">
  <img src="vis/mmlu_spectro_gn_10_d_7_redist.png" width="375" />
  <img src="vis/persian_qa_spectro_gn_10_d_5_redist.png" width="375" /> 
</p>

## Setup
Download and set up the repository:
```bash
git clone https://github.com/tony10101105/U-shaped-and-Inverted-U-Scaling-behind-Emergent-Abilities-of-Large-Language-Models.git
cd U-shaped-and-Inverted-U-Scaling-behind-Emergent-Abilities-of-Large-Language-Models
```
```bash
conda env create -name ExpEmergence -file requirements.txt
conda activate ExpEmergence
```

## :rocket: Usage
We include six scripts to replicate the main results of this work:

**Plot model overall performance measured by accuracy or TC Brier Score:**
```bash
python plot_overall_performance.py
```
**Plot model performance on each question group measured by TC Brier Score:**
```bash
python plot_question_group_tc_brier.py
```
**Plot model performance on each question group measured by accuracy:**
```bash
python plot_question_group_acc.py
```
**Perform Slice-and-Sandwich to construct the accuracy-based scaling law using models before the emergence threshold:**
```bash
python fit_cluster.py.py
```
**Slice-and-Sandwich's robustness analysis regarding polynomial degree:**
```bash
python fit_cluster_robustness_degree.py
```
**Slice-and-Sandwich's robustness analysis regarding emergence threshold:**
```bash
python fit_cluster_robustness_threshold.py
```

We welcome you to use our code and data to exploit new insights and methods. NOTE: the data *base_llm_benchmark_eval - base_llm_benchmark_eval.csv* that we built upon is from the paper [Observational Scaling Laws](https://github.com/ryoungj/ObsScaling). Thanks for their excellent work!

## Replication of LLM Evaluation
You can evaluate any LLMs and run our experiments (note that replicating our csv files, including over 50 LLMs on 9 datasets, can be computationally expensive). Three steps are required to replicate a dataset (use mmlu as the example): 

1. Install the [lm-eval package](https://github.com/EleutherAI/lm-evaluation-harness).

```bash
cd lm-evaluation-harness
mkdir eval_out
cd eval_out
```
2. *Put mmlu.sh* and *all_models.txt* provided in our *evaluation/mmlu* to *lm-evaluation-harness/* and run it:
```bash
sh mmlu.sh
```
3. Put *base_llm_benchmark_eval - base_llm_benchmark_eval.csv*, *mmlu_question_grouping.py* and *mmlu_metadata.py* provided in our *evaluation/mmlu* to *lm-evaluation-harness/eval_out* and run it to generate the csv file as those in our *data/*:
```bash
python mmlu_question_grouping.py
```  

Generate the accuracy-based version by placing and running:  

```bash
python mmlu_question_grouping_acc.py
```

## Citation
```
@article{wu2024u,
  title={U-shaped and Inverted-U Scaling behind Emergent Abilities of Large Language Models},
  author={Wu, Tung-Yu and Lo, Pei-Yu},
  journal={arXiv preprint arXiv:2410.01692},
  year={2024}
}
```
