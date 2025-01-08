import os
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
from utils import basic_parameter, fit_parameter, model_filter, sigmoid, inverse_sigmoid, clip_value, brier2acc_ols_model


dataset = 'mmlu' # must be in task_parameter
save = True # whether to save figures as pdf
use_hard_lift = False # whether use hard_lift in Appendix.G.2
plot_mse = False # whether to plot mse loss
clip_scaling_law = False # whether to clip values beyond min & max
plot_oracle = False # oracle is to let the baseline see both train and test split


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
    assert 'brier' in cluster
    df = df.dropna(subset=[cluster])
    df[cluster] = df[cluster].apply(lambda x: float(x)*(-1))

df = df.dropna(subset=['FLOPs'])
df['brier'] = df['brier'].apply(lambda x: float(x)*(-1))
df['FLOPs'] = df['FLOPs'].apply(lambda x: np.log10(float(x)))
colors = ['red' if size < np.log10(threshold) else 'blue' for size in df['FLOPs']]


### acc-brier relation
fit_df = df[df['FLOPs'] < np.log10(threshold)]
X = fit_df['acc']
X = sm.add_constant(X)
y = fit_df['brier']

acc2brier_ols_model = sm.OLS(y, X).fit()
print('acc2brier_ols_model summary: ', acc2brier_ols_model.summary())

df['ols_pred_brier'] = acc2brier_ols_model.predict(sm.add_constant(df['acc']))

plt.scatter(df['acc'][fitting_split], df['brier'][fitting_split], color='red', label='training split')
plt.scatter(df['acc'][testing_split], df['brier'][testing_split], color='blue', label='testing split')

pearson = df['acc'].corr(df['brier'], method='pearson')
spearman = df['acc'].corr(df['brier'], method='spearman')
kendall = df['acc'].corr(df['brier'], method='kendall')
print('pearson coefficient: ', pearson)
print('spearman coefficient: ', spearman)
print('kendall coefficient: ', kendall)

plt.plot(df['acc'], df['ols_pred_brier'], color='black', label='OLS')
plt.xticks(fontsize=12)
plt.yticks(np.arange(-0.6, -0.1, 0.1), fontsize=12)
plt.xlabel('accuracy', fontsize=18)
plt.ylabel('TC Brier Score', fontsize=18)
plt.legend(fontsize=12)
if save:
    plt.savefig(f'figure/{dataset}/fit/{dataset}_acc_redist_brier_relation.pdf', bbox_inches='tight', pad_inches=0)
plt.show()


### fit the simple cluster with polynomial regression
df['FLOPs2'] = df['FLOPs']**2
df['FLOPs3'] = df['FLOPs']**3
df['FLOPs4'] = df['FLOPs']**4
df['FLOPs5'] = df['FLOPs']**5
fit_df = df[df['FLOPs'] < np.log10(threshold)]
for i, cluster in enumerate(clusters):
    fit_color = 'red' if i == 0 else 'orange'
    test_color = 'blue' if i == 0 else 'lightblue'
    fit_label = 'easy' if i == 0 else 'hard'
    
    plt.scatter(df['FLOPs'][fitting_split], df[cluster][fitting_split], color=fit_color, label=f'{fit_label} training split')
    plt.scatter(df['FLOPs'][testing_split], df[cluster][testing_split], color=test_color, label=f'{fit_label} testing split')

    if i == 0:
        model = smf.ols('Q(cluster) ~ Q("FLOPs") + Q("FLOPs2") + Q("FLOPs3") + Q("FLOPs4") + Q("FLOPs5")', data=fit_df).fit()
    else:
        model = smf.ols('Q(cluster) ~ Q("FLOPs") + Q("FLOPs2")', data=fit_df).fit()
        
    print('regression summary: ', model.summary())
    df[f'{cluster}_pred'] = model.predict(df)
    plt.ylim(-1, 0)
    plt.plot(df['FLOPs'], df[f'{cluster}_pred'], color='black', label=f'{fit_label} polynomial regression')
    
plt.ylim([-1, 0])
plt.xlabel('log compute (M)', fontsize=18)
plt.ylabel('TC Brier Score', fontsize=18)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
if save:
    plt.savefig(f'figure/{dataset}/fit/{dataset}_redist_easy_hard_brier_fit.pdf', bbox_inches='tight', pad_inches=0)
plt.show()
plt.clf()


### predict brier scores
plt.scatter(df['FLOPs'][fitting_split], df['brier'][fitting_split], color='red', label='training split')
plt.scatter(df['FLOPs'][testing_split], df['brier'][testing_split], color='blue', label='testing split')

if use_hard_lift: # in hard_lift, we also do calibration on brier
    fit_brier_avg = df['brier'][fitting_split].mean()
    our_brier_avg = df[f'{clusters[1]}_pred'][fitting_split].mean()
    df['brier_pred'] = df[f'{clusters[1]}_pred'].apply(lambda x: x + fit_brier_avg - our_brier_avg)
else:
    df['brier_pred'] = (df[f'{clusters[0]}_pred'] + df[f'{clusters[1]}_pred']) / 2

plt.plot(df['FLOPs'], df['brier_pred'], color='black', label='ours')
plt.ylim([-1, 0])
plt.xlabel('log compute (M)', fontsize=18)
plt.ylabel('TC Brier Score', fontsize=18)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
if save:
    plt.savefig(f'figure/{dataset}/fit/{dataset}_redist_brier_scaling_law.pdf', bbox_inches='tight', pad_inches=0)
plt.show()
plt.clf()


### fit our acc prediction
coefficients = acc2brier_ols_model.params
intercept = coefficients[0]
slope = coefficients[1]

df['acc_pred'] = df['brier_pred'].apply(lambda x: brier2acc_ols_model(x, intercept, slope))

fit_acc_avg = df['acc'][fitting_split].mean()
our_acc_avg = df['acc_pred'][fitting_split].mean()
df['acc_pred'] = df['acc_pred'].apply(lambda x: x + fit_acc_avg - our_acc_avg)

if clip_scaling_law:
    df['acc_pred'] = df['acc_pred'].apply(lambda x: clip_value(x))

plt.scatter(df['FLOPs'][fitting_split], df['acc'][fitting_split], color='red', label='training split')
plt.scatter(df['FLOPs'][testing_split], df['acc'][testing_split], color='blue', label='testing split')
if plot_mse:
    our_mse = np.mean((df['acc'][testing_split] - df['acc_pred'][testing_split]) ** 2)
    plt.plot(df['FLOPs'], df['acc_pred'], color='black', label=f'ours. MSE={our_mse:.2e}')
else:
    plt.plot(df['FLOPs'], df['acc_pred'], color='black', label='ours')


### fit baseline acc predcition
df['logit_acc'] = df['acc'].apply(lambda x: inverse_sigmoid(x - random_guess_acc))

fit_df = df[df['FLOPs'] < np.log10(threshold)]
model = smf.ols('logit_acc ~ FLOPs', data=fit_df).fit()

a_estimated = model.params['FLOPs']
b_estimated = model.params['Intercept']

df['acc_base_pred'] = df['FLOPs'].apply(lambda x: sigmoid(a_estimated * x + b_estimated) + random_guess_acc)
if plot_mse:
    baseline_mse = np.mean((df['acc'][testing_split] - df['acc_base_pred'][testing_split]) ** 2)
    plt.plot(df['FLOPs'], df['acc_base_pred'], color='orange', label=f'baseline. MSE={baseline_mse:.2e}')
else:
    plt.plot(df['FLOPs'], df['acc_base_pred'], color='orange', label='baseline')

if plot_oracle:
    model = smf.ols('logit_acc ~ FLOPs', data=df).fit()

    a_estimated = model.params['FLOPs']
    b_estimated = model.params['Intercept']

    df['acc_oracle_pred'] = df['FLOPs'].apply(lambda x: sigmoid(a_estimated * x + b_estimated) + random_guess_acc)
    if plot_mse:
        baseline_mse = np.mean((df['acc'][testing_split] - df['acc_oracle_pred'][testing_split]) ** 2)
        plt.plot(df['FLOPs'], df['acc_oracle_pred'], color='green', label=f'oracle. MSE={baseline_mse:.2e}')
    else:
        plt.plot(df['FLOPs'], df['acc_oracle_pred'], color='green', label=f'oracle')


plt.ylim([0, 1])
plt.xlabel('log compute (M)', fontsize=18)
plt.ylabel('accuracy', fontsize=18)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
if save:
    if not use_hard_lift:
        plt.savefig(f'figure/{dataset}/fit/{dataset}_redist_acc_scaling_law.pdf', bbox_inches='tight', pad_inches=0)
    else:
        plt.savefig(f'figure/{dataset}/fit/{dataset}_redist_acc_scaling_law_hardlift.pdf', bbox_inches='tight', pad_inches=0)
plt.show()