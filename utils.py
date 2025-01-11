import numpy as np


basic_parameter = {
    'mmlu': {'type': 'Emergence', 'threshold': 1.5, 'question_num': 14042, 'random_guess_acc': 0.25},
    'parsinlu_qa_mc': {'type': 'Emergence', 'threshold': 2.3, 'question_num': 1050, 'random_guess_acc': 0.25},
    'arithmetic': {'type': 'Emergence', 'threshold': 1.8, 'question_num': 15023, 'random_guess_acc': 0.145},
    'hindu_knowledge': {'type': 'Emergence', 'threshold': 1, 'question_num': 175, 'random_guess_acc': 0.25},
    'analogical_similarity': {'type': 'Emergence', 'threshold': 2, 'question_num': 323, 'random_guess_acc': 0.145},
    'conceptual_combinations': {'type': 'Emergence', 'threshold': 1.5, 'question_num': 103, 'random_guess_acc': 0.25},
    'hellaswag': {'type': 'No Emergence', 'threshold': 2.3, 'question_num': 10042, 'random_guess_acc': 0.25},
    'arc': {'type': 'No Emergence', 'threshold': 2.3, 'question_num': 3548, 'random_guess_acc': 0.25},
    'abstract_narrative_understanding': {'type': 'No Emergence', 'threshold': 2.3, 'question_num': 3000, 'random_guess_acc': 0.1}
}

group_plot_parameter = {
    'mmlu': {'illustration_degree': 7, 'y_loc_brier': -0.1, 'y_loc_acc': 0.8},
    'parsinlu_qa_mc': {'illustration_degree': 5, 'y_loc_brier': -0.1, 'y_loc_acc': 1},
    'arithmetic': {'illustration_degree': 10, 'y_loc_brier': 0.2, 'y_loc_acc': 1.2},
    'hindu_knowledge': {'illustration_degree': 5, 'y_loc_brier': -0.2},
    'analogical_similarity': {'illustration_degree': 5, 'y_loc_brier': -0.4},
    'conceptual_combinations': {'illustration_degree': 5, 'y_loc_brier': 0},
    'hellaswag': {'illustration_degree': 5, 'y_loc_brier': -0.4},
    'arc': {'illustration_degree': 5, 'y_loc_brier': -0.4},
    'abstract_narrative_understanding': {'illustration_degree': 5, 'y_loc_brier': 0.05}
}

overall_plot_parameter = {
    'mmlu': {'y_loc_brier': -0.3, 'y_loc_acc': 0.6},
    'parsinlu_qa_mc': {'y_loc_brier': -0.5, 'y_loc_acc': 0.35},
    'arithmetic': {'y_loc_brier': -0.2, 'y_loc_acc': 0.8},
    'hindu_knowledge': {'y_loc_brier': -0.2, 'y_loc_acc': 0.6},
    'analogical_similarity': {'y_loc_brier': -0.7, 'y_loc_acc': 0.25},
    'conceptual_combinations': {'y_loc_brier': -0.3, 'y_loc_acc': 0.6},
    'hellaswag': {'y_loc_brier': -0.35, 'y_loc_acc': 0.6},
    'arc': {'y_loc_brier': -0.25, 'y_loc_acc': 0.7},
    'abstract_narrative_understanding': {'y_loc_brier': -0.6, 'y_loc_acc': 0.35}
}

fit_parameter = {
    'mmlu': {'clusters': ['0_4680_brier', '9361_14042_brier'], 'robust_analysis_thresholds': [1.5, 1.3, 1.1]},
    'parsinlu_qa_mc': {'clusters': ['0_350_brier', '700_1050_brier'], 'robust_analysis_thresholds': [2.3, 2.1, 1.9]},
    'arithmetic': {'clusters': ['0_5007_brier', '10015_15023_brier'], 'robust_analysis_thresholds': [1.8, 1.6, 1.4]},
    'hindu_knowledge': {'clusters': ['0_58_brier', '116_175_brier']},
    'analogical_similarity': {'clusters': ['0_107_brier', '215_323_brier']},
    'conceptual_combinations': {'clusters': ['0_34_brier', '68_103_brier']},
    'hellaswag': {'clusters': ['0_3347_brier', '6694_10042_brier']},
    'arc': {'clusters': ['0_1182_brier', '2365_3548_brier']},
    'abstract_narrative_understanding': {'clusters': ['0_1000_brier', '2000_3000_brier']}
}

def model_filter(df):
    """filter out probably over-trained small models"""
    df = df[df['Model'] != 'microsoft/phi-1_5']
    df = df[df['Model'] != 'microsoft/phi-2']
    df = df[df['Model'] != 'Qwen/Qwen1.5-0.5B']
    df = df[df['Model'] != 'Qwen/Qwen1.5-1.8B']
    df = df[df['Model'] != 'Qwen/Qwen1.5-4B']
    df = df[df['Model'] != 'stabilityai/stablelm-2-1_6b']
    return df

def whiten(hex_color, k):
    """Convert hex to rgba format manually"""
    r,g,b = hex_color.lstrip('#').strip('rgb()').split(',')
    return f'rgb({int(r)+k},{int(g)+k},{int(b)+k})'  # Create rgba string with transparency

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def inverse_sigmoid(y, esp=1e-6):
    y = np.maximum(y, 0.0)
    y = np.minimum(y, 1.0)
    return np.log((y + esp) / (1 - y + esp))

def brier2acc_ols_model(brier, intercept, slope):
    return (brier - intercept) / slope

def clip_value(x):
    return x if x < 1 else 1