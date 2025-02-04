import os
import pandas as pd
import plotly.express as px
import plotly.io as pio
import numpy as np
from utils import basic_parameter, overall_plot_parameter, model_filter


### parameters
dataset = 'mmlu' # must be in basic_parameter
mode = 'acc' # 'brier' or 'acc'
brier_mode = 'redist' # if mode == 'acc', brier_mode will have no impact
###


dataset_type = basic_parameter[dataset]['type']
threshold = basic_parameter[dataset]['threshold']
threshold_in_flops = 10**threshold
y_loc = overall_plot_parameter[dataset][f'y_loc_{mode}']
y_title = 'TC Brier Score' if mode == 'brier' else 'accuracy'

csv_file = f'data/{dataset}/{dataset}_instance_brier_-1000_{threshold_in_flops}_3_{brier_mode}.csv' # threshold and group number do not affect overall performance
df = pd.read_csv(csv_file)
df = model_filter(df)

df = df.dropna(subset=[mode])
df = df.dropna(subset=['FLOPs (1E21)'])
df['FLOPs (1E21)'] = df['FLOPs (1E21)'].apply(lambda x: np.log10(float(x)))
df['brier'] = df['brier'].apply(lambda x: float(x)*(-1))

fig = px.scatter(df, x='FLOPs (1E21)', y=mode, color='Model', hover_data=['Release Date', 'Model Size (B)', 'Pretraining Data Size (T)'],
                    hover_name='Model Family')
fig.update_traces(marker=dict(size=10))

if dataset_type == 'Emergence':
    fig.add_vline(x=threshold, line_dash='dash', line_color='red', line_width=2)
fig.add_annotation(
    x=threshold,  # Position of the annotation on the x-axis
    y=y_loc,    # Position on the y-axis, adjust according to your data
    text=dataset_type,  # The label text
    showarrow=dataset_type == 'Emergence',  # Show an arrow pointing to the line
    arrowhead=2,  # Arrow style
    arrowsize=1,  # Size of the arrowhead
    arrowwidth=2,  # Thickness of the arrow
    ax=-100,  # Position of the annotation relative to the arrow
    ay=0,
    font=dict(size=24, color='black'),  # Customize text font
    bgcolor='white',  # Background color for better visibility
    bordercolor='black',  # Border color around the annotation
)


fig.update_layout(transition_duration=500, xaxis_title='log compute (M)', yaxis_title=y_title,
                    xaxis=dict(title_font=dict(size=28), tickfont=dict(size=18)),
                    yaxis=dict(title_font=dict(size=28), tickfont=dict(size=18)),
                    showlegend=False,
                    margin=dict(l=10, r=10, t=10, b=10))


# fig.show()
pio.kaleido.scope.mathjax = None
os.makedirs(f'figure/{dataset}', exist_ok=True)
if mode == 'brier':
    fig.write_image(f'figure/{dataset}/{dataset}_{mode}_{brier_mode}.pdf')
else:
    fig.write_image(f'figure/{dataset}/{dataset}_{mode}.pdf')