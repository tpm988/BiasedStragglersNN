import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

# Define your file names
batch = '10' #["10", "32"]
file1 = f'Results_for_plot_b{batch}_a0.5.xlsx'
file2 = f'Results_for_plot_b{batch}_a2.xlsx'

# Load your data
df1_acc = pd.read_excel(file1, sheet_name='acc')
df1_fairness = pd.read_excel(file1, sheet_name='fairness')

df2_acc = pd.read_excel(file2, sheet_name='acc')
df2_fairness = pd.read_excel(file2, sheet_name='fairness')

# Define color mapping for columns: the order matters (for legend 2 by 3 layout)
color_map = {
    'Fair-Fate(SP)': 'blue',
    'Fair-Fate-VC(SP)': 'purple',
    'Fair-Fate(EO)': 'green',
    'Fair-Fate-VC(EO)': 'orange',
    'Fair-Fate(EQO)': 'red',
    'Fair-Fate-VC(EQO)': 'brown',
}

# Create subplots
fig, axs = plt.subplots(2, 4, figsize=(20, 10))

# Add data to the subplots
for i, df_acc in enumerate([df1_acc, df2_acc]):
    for col in df_acc.columns:
        axs[i, 0].plot(df_acc[col], color=color_map[col], label=col)
    axs[i, 0].set_title('acc for ' + ['a = 0.5', 'a = 2'][i])
    axs[i, 0].set_ylim([0.0, 1.0])
    axs[i, 0].set_yticks([i * 0.2 for i in range(6)])

for i, df_fairness in enumerate([df1_fairness, df2_fairness]):
    for col in ['Fair-Fate(SP)', 'Fair-Fate-VC(SP)']:
        axs[i, 1].plot(df_fairness[col], color=color_map[col], label=col)
    axs[i, 1].set_title('SP for ' + ['a = 0.5', 'a = 2'][i])
    axs[i, 1].set_ylim([0.0, 1.0])
    axs[i, 1].set_yticks([i * 0.2 for i in range(6)])

    for col in ['Fair-Fate(EO)', 'Fair-Fate-VC(EO)']:
        axs[i, 2].plot(df_fairness[col], color=color_map[col], label=col)
    axs[i, 2].set_title('EO for ' + ['a = 0.5', 'a = 2'][i])
    axs[i, 2].set_ylim([0.0, 1.0])
    axs[i, 2].set_yticks([i * 0.2 for i in range(6)])

    for col in ['Fair-Fate(EQO)', 'Fair-Fate-VC(EQO)']:
        axs[i, 3].plot(df_fairness[col], color=color_map[col], label=col)
    axs[i, 3].set_title('EQO for ' + ['a = 0.5', 'a = 2'][i])
    axs[i, 3].set_ylim([0.0, 1.0])
    axs[i, 3].set_yticks([i * 0.2 for i in range(6)])

# # Create custom legend
# legend_lines = [mlines.Line2D([0], [0], color=color, linewidth=1, linestyle='-') for color in color_map.values()]
# plt.figlegend(legend_lines, color_map.keys(), loc='lower right')

# Create a grid legend
legend_labels = np.array(list(color_map.keys())).reshape(2, 3)
legend_lines = np.array([mlines.Line2D([0], [0], color=color, linewidth=1, linestyle='-') for color in color_map.values()]).reshape(2, 3)

# Place legend at bottom right
leg = plt.legend(legend_lines.ravel(), legend_labels.ravel(), ncol=3, bbox_to_anchor=(1.0, -0.1, 0.0, 0.0), loc='upper right', fontsize='large')

# plt.tight_layout()
plt.subplots_adjust(bottom=0.2, right=0.8)  # To ensure the legend does not overlap with subplots
plt.savefig(f'Exp_Result_b{batch}.png', dpi=300, pad_inches=0.1, bbox_inches='tight')
plt.show()