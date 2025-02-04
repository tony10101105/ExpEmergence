# U-shaped and Inverted-U Scaling behind Emergent Abilities of Large Language Models
Official code of the paper [U-shaped and Inverted-U Scaling behind Emergent Abilities of Large Language Models](https://arxiv.org/abs/2410.01692)

## News
- (Jan. 2025) The paper has been accepted to the ICLR'25 as a main conference paper.
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
git clone https://github.com/tony10101105/ExpEmergence.git
cd ExpEmergence
```
```bash
conda env create -n ExpEmergence -f requirements.txt
conda activate ExpEmergence
```

## :rocket: Usage
Six scripts to replicate the main results:

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
python fit_cluster.py
```
**Slice-and-Sandwich's robustness analysis regarding polynomial degree:**
```bash
python fit_cluster_robustness_degree.py
```
**Slice-and-Sandwich's robustness analysis regarding emergence threshold:**
```bash
python fit_cluster_robustness_threshold.py
```

The data `base_llm_benchmark_eval.csv` we built upon is from the paper [Observational Scaling Laws](https://github.com/ryoungj/ObsScaling). Thanks for their excellent work!

## Replication of LLM Evaluation
Below we describe how to evaluate LLMs on a dataset to generate csv files as in our */data*. Three steps are required (use MMLU as the example): 

1. Install the [lm-eval package](https://github.com/EleutherAI/lm-evaluation-harness).
3. Put `all_models.txt` and `mmlu.sh` provided in our *evaluation/* and *evaluation/mmlu* to *lm-evaluation-harness/* and run it:
```bash
cd lm-evaluation-harness
cp $ROOT_DIR$/ExpEmergence/evaluation/all_models.txt all_models.txt
cp $ROOT_DIR$/ExpEmergence/evaluation/mmlu/mmlu.sh mmlu.sh
mkdir eval_out
bash mmlu.sh
```
3. Put `base_llm_benchmark_eval.csv`, `mmlu_question_grouping.py` and `mmlu_metadata.py` provided in our *evaluation/mmlu* to *lm-evaluation-harness/eval_out* and run it to generate the csv file as those in our *data/*:
```bash
cd eval_out
cp $ROOT_DIR$/ExpEmergence/evaluation/base_llm_benchmark_eval.csv base_llm_benchmark_eval.csv
cp $ROOT_DIR$/ExpEmergence/evaluation/mmlu/mmlu_question_grouping.py mmlu_question_grouping.py
cp $ROOT_DIR$/ExpEmergence/evaluation/mmlu/mmlu_metadata.py mmlu_metadata.py
python mmlu_question_grouping.py
```  

Generate the accuracy-based version by instead placing and running:  

```bash
python mmlu_question_grouping_acc.py
```

## Citation
```
@inproceedings{wu2024u,
  title={U-shaped and Inverted-U Scaling behind Emergent Abilities of Large Language Models},
  author={Wu, Tung-Yu and Lo, Pei-Yu},
  booktitle={Proceedings of the International Conference on Learning Representations (ICLR)},
  year={2025}
}
```
