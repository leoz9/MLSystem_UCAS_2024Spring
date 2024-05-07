import pandas as pd
import matplotlib.pyplot as plt

# Read data
sklearn_data = pd.read_csv('sklearn_performance.csv')
torch_data = pd.read_csv('torch_performance.csv')
gpu_data = pd.read_csv('gpu_performance.csv')
mutil_gup = pd.read_csv('mutil_gpu_performance')
# Add 'Model' column to each dataframe
sklearn_data['Model'] = 'sklearn'
torch_data['Model'] = 'torch'
gpu_data['Model'] = 'torch_gpu'
mutil_gup['Model'] = 'mutil_gpu'
# Rearrange columns to have 'Model' as the first column
sklearn_data = sklearn_data[['Model'] + [col for col in sklearn_data.columns if col != 'Model']]
torch_data = torch_data[['Model'] + [col for col in torch_data.columns if col != 'Model']]
gpu_data = gpu_data[['Model'] + [col for col in gpu_data.columns if col != 'Model']]
mutil_gup = mutil_gup[['Model'] + [col for col in gpu_data.columns if col != 'Model']]
# Merge dataframes
merged_data = pd.concat([sklearn_data, torch_data, gpu_data,mutil_gup], ignore_index=True)

# Create table image
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('tight')
ax.axis('off')

table = ax.table(cellText=merged_data.values,
                 colLabels=merged_data.columns,
                 loc='center',
                 cellLoc='center',
                 colColours=['lightblue'] * len(merged_data.columns))

table.auto_set_font_size(True)
table.set_fontsize(10)
table.scale(1.2, 1.2)

# Save image
plt.savefig('performance_comparison_table.png', bbox_inches='tight', pad_inches=0.1)
plt.show()
