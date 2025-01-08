import os
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
import math
from utils import basic_parameter, group_plot_parameter, model_filter, whiten
import plotly.express.colors as pec


# parameters
dataset = 'parsinlu_qa_mc' # must be mmlu, arithmetic, or parsinlu_qa_mc
brier_mode = 'redist' # redist or undist. The former is w/ conditionality and the latter is w/o conditionality
group_num = 10 # number of question groups
show_color_bar = False # whether to show the color bar

dataset_type = basic_parameter[dataset]['type'] # 'Emergence' or 'No Emergence'
threshold = basic_parameter[dataset]['threshold'] # emergent threshold and also the threshold for question difficulty calculation
question_num = basic_parameter[dataset]['question_num'] # number of questions in the dataset
threshold_in_flops = 10**threshold
degree = group_plot_parameter[dataset]['illustration_degree'] # for visulaization
y_loc = group_plot_parameter[dataset]['y_loc_acc'] # for visulaization

difficulty_levels = list(range(1, group_num+1))
color_scale = pec.diverging.RdYlBu
if group_num == 3:
    color_scale = [color_scale[0], color_scale[3], color_scale[9]]

saved_question_idx = np.linspace(0, question_num, group_num+1)
saved_question_idx = [[int(math.floor(saved_question_idx[i])), int(math.floor(saved_question_idx[i+1]))] for i in range(len(saved_question_idx)-1)]
modes = [f'{i}_{j}_acc' for i, j in saved_question_idx]

csv_file = f'data/{dataset}/acc_{dataset}_instance_brier_-1000_{threshold_in_flops}_{group_num}_{brier_mode}.csv'
df = pd.read_csv(csv_file)
df = model_filter(df)

for m in modes:
    df = df.dropna(subset=[m])
df = df.dropna(subset=['FLOPs (1E21)'])
df['FLOPs (1E21)'] = df['FLOPs (1E21)'].apply(lambda x: np.log10(float(x))) # convert to effective model size

for m in modes:
    coefficients = np.polyfit(df['FLOPs (1E21)'], df[m], degree)
    polynomial = np.poly1d(coefficients)
    df[f'{m}_pred'] = polynomial(df['FLOPs (1E21)'])

def hex_to_rgba(hex_color, alpha=1.0):
    # Convert hex to rgba format manually
    hex_color = hex_color.lstrip('#')
    hex_color = hex_color.strip('rgb').strip('(').strip(')').split(',')
    r, g, b = tuple(int(hex, 16) for hex in hex_color)  # Convert hex to RGB
    return f'rgba({r},{g},{b},{alpha})'  # Create rgba string with transparency

fig = go.Figure()
df.sort_values(by='FLOPs (1E21)', ascending=True, inplace=True)
for i, m in enumerate(modes):
    line_color = color_scale[i]
    
    residuals = df[m] - df[f'{m}_pred']
    residual_std_error = np.std(residuals)
    ci_factor = 1.96
    y_upper = df[f'{m}_pred'] + ci_factor * residual_std_error
    y_lower = df[f'{m}_pred'] - ci_factor * residual_std_error

    fig.add_scatter(x=np.concatenate([df['FLOPs (1E21)'], df['FLOPs (1E21)'][::-1]]),
                    y=np.concatenate([y_upper, y_lower[::-1]]),
                    fill='toself', fillcolor=hex_to_rgba(line_color, 0.3),
                    line=dict(color='rgba(255,255,255,0)'),  # Hide the line
                    showlegend=False)

if show_color_bar:
    color_scale_white = [whiten(c, 25) for c in color_scale]
    fig.add_trace(go.Scatter(x=[None], y=[None],
                            mode='markers',
                            marker=dict(
                            size=10,
                            color=difficulty_levels,  # Map difficulty levels (1–10)
                            colorscale=color_scale_white[::-1],  # Use your custom color scale
                            cmin=1,  # Minimum difficulty level
                            cmax=group_num,  # Maximum difficulty level (e.g., 10)
                            colorbar=dict(
                                title="Difficulty<br>Level",  # Title for the colorbar
                                titleside="top",  # Position of the title
                                titlefont=dict(size=19),  # Font size for the colorbar title
                                tickfont=dict(size=16),  # Font size for the colorbar ticks
                                tickvals=list(range(1, group_num + 1)),  # Show ticks from 1 to 10
                                ticktext=list(range(group_num, 0, -1)),  # Reverse the tick labels (10 -> 1)
                                ticks="outside",
                                tickmode="array",
                                tickwidth=2,  # Optionally, change the width of the ticks
                                ticklen=5,  # Length of each tick
                            ),
                            showscale=True  # Display the colorbar
                            )))

for i, m in enumerate(modes):
    line_color = color_scale[i]
    fig.add_trace(go.Scatter(x=df['FLOPs (1E21)'], y=df[m], mode='markers', name=m, line=dict(color=line_color)))
    fig.add_scatter(x=df['FLOPs (1E21)'], y=df[f'{m}_pred'], mode='lines', name=f'{m}_fit', line=dict(color=line_color))

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

fig.update_layout(transition_duration=500, xaxis_title='log compute (M)', yaxis_title='accuracy',
                    xaxis=dict(title_font=dict(size=24), tickfont=dict(size=18)),
                    yaxis=dict(title_font=dict(size=24), tickfont=dict(size=18)),
                    showlegend=False,
                    # legend=dict(
                    #     font=dict(size=16)
                    # ),
                    margin=dict(l=10, r=10, t=10, b=10),
                    width=1000,
                    height=600)
# fig.update_yaxes(range=[0, 1])

fig.show()
pio.kaleido.scope.mathjax = None
os.makedirs(f'figure/{dataset}', exist_ok=True)
fig.write_image(f'figure/{dataset}/acc_{dataset}_spectro_gn_{group_num}_d_{degree}_{brier_mode}.pdf', engine='kaleido')