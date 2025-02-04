import os
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
from utils import basic_parameter, fit_parameter, model_filter


### parameters
dataset = 'mmlu' # must be in task_parameter
save = True # whether to save figures as pdf
use_hard_lift = False # whether use hard_lift in Appendix.G.2
plot_mse = False # whether to plot mse loss
clip_scaling_law = False # whether to clip values beyond min & max
plot_oracle = False # oracle is to let the baseline see both train and test split
###


os.makedirs(f'figure/{dataset}/fit', exist_ok=True)
threshold = 10**basic_parameter[dataset]['threshold']
random_guess_acc = basic_parameter[dataset]['random_guess_acc']
clusters = fit_parameter[dataset]['clusters']

csv_file = f'data/{dataset}/{dataset}_instance_brier_-1000_{threshold}_3_redist.csv'
df = pd.read_csv(csv_file)
df.rename(columns={'FLOPs (1E21)': 'FLOPs'}, inplace=True)
df = df.sort_values(by='FLOPs')
df = model_filter(df)

fitting_split = df['FLOPs'] < threshold
testing_split = df['FLOPs'] >= threshold

df = df.dropna(subset=['acc'])
for cluster in clusters:
    df = df.dropna(subset=[cluster])
    df[cluster] = df[cluster].apply(lambda x: float(x)*(-1))

df = df.dropna(subset=['FLOPs'])
df['brier'] = df['brier'].apply(lambda x: float(x)*(-1))
df['FLOPs'] = df['FLOPs'].apply(lambda x: np.log10(float(x)))
colors = ['red' if size < np.log10(threshold) else 'blue' for size in df['FLOPs']]


### fit the simple cluster with polynomial regression
df['FLOPs2'] = df['FLOPs']**2
df['FLOPs3'] = df['FLOPs']**3
df['FLOPs4'] = df['FLOPs']**4
df['FLOPs5'] = df['FLOPs']**5
df['FLOPs6'] = df['FLOPs']**6
df['FLOPs7'] = df['FLOPs']**7
fit_df = df[df['FLOPs'] < np.log10(threshold)]
for i, cluster in enumerate(clusters):
    fit_color = 'red' if i == 0 else 'orange'
    test_color = 'blue' if i == 0 else 'lightblue'
    fit_label = 'easy' if i == 0 else 'hard'
    
    plt.scatter(df['FLOPs'][fitting_split], df[cluster][fitting_split], color=fit_color, label=f'{fit_label} training split')
    plt.scatter(df['FLOPs'][testing_split], df[cluster][testing_split], color=test_color, label=f'{fit_label} testing split')

    if i == 0:
        model = smf.ols('Q(cluster) ~ Q("FLOPs") + Q("FLOPs2") + Q("FLOPs3")', data=fit_df).fit()
        df[f'{cluster}_pred_3'] = model.predict(df)
        plt.ylim(-1, 0)
        plt.plot(df['FLOPs'], df[f'{cluster}_pred_3'], color='black', label=f'{fit_label}, order=3')

        model = smf.ols('Q(cluster) ~ Q("FLOPs") + Q("FLOPs2") + Q("FLOPs3") + Q("FLOPs4") + Q("FLOPs5")', data=fit_df).fit()
        df[f'{cluster}_pred_5'] = model.predict(df)
        plt.ylim(-1, 0)
        plt.plot(df['FLOPs'], df[f'{cluster}_pred_5'], linestyle='--', color='black', label=f'{fit_label}, order=5')
        
        model = smf.ols('Q(cluster) ~ Q("FLOPs") + Q("FLOPs2") + Q("FLOPs3") + Q("FLOPs4") + Q("FLOPs5") + Q("FLOPs6") + Q("FLOPs7")', data=fit_df).fit()
        df[f'{cluster}_pred_7'] = model.predict(df)
        plt.ylim(-1, 0)
        plt.plot(df['FLOPs'], df[f'{cluster}_pred_7'], linestyle=':', color='black', label=f'{fit_label}, order=7')
    else:
        model = smf.ols('Q(cluster) ~ Q("FLOPs") + Q("FLOPs2")', data=fit_df).fit()
        plt.ylim(-1, 0)
        df[f'{cluster}_pred_2'] = model.predict(df)
        plt.plot(df['FLOPs'], df[f'{cluster}_pred_2'], color='black', label=f'{fit_label}, order=2')
        
        model = smf.ols('Q(cluster) ~ Q("FLOPs") + Q("FLOPs2") + Q("FLOPs3") + Q("FLOPs4")', data=fit_df).fit()
        plt.ylim(-1, 0)
        df[f'{cluster}_pred_4'] = model.predict(df)
        plt.plot(df['FLOPs'], df[f'{cluster}_pred_4'], linestyle='--', color='black', label=f'{fit_label}, order=4')
        
        model = smf.ols('Q(cluster) ~ Q("FLOPs") + Q("FLOPs2") + Q("FLOPs3") + Q("FLOPs4") + Q("FLOPs5") + Q("FLOPs6")', data=fit_df).fit()
        plt.ylim(-1, 0)
        df[f'{cluster}_pred_6'] = model.predict(df)
        plt.plot(df['FLOPs'], df[f'{cluster}_pred_6'], linestyle=':', color='black', label=f'{fit_label}, order=6')

    print(model.summary())
    
plt.ylim([-1, 0])
plt.xlabel('log compute (M)', fontsize=18)
plt.ylabel('TC Brier Score', fontsize=18)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
if save:
    plt.savefig(f'figure/{dataset}/fit/{dataset}_easy_hard_brier_fit_robustness_degree.pdf', bbox_inches='tight', pad_inches=0)
plt.show()
plt.clf()