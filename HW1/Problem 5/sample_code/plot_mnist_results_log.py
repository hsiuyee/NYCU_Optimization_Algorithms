import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import json
from matplotlib.ticker import FuncFormatter

experiments = ["(b)_SGD_lr_0.001", "(b)_SVRG_lr_0.025"]
data_dir = "outputs/a/(b)"

# load data
all_stats = []
for exp in experiments:
    folders = [x for x in os.listdir(data_dir) if x.startswith(exp)]
    
    for folder in folders:
        stats = {}
        args_file = os.path.join(data_dir, folder, "args.json")
        with open(args_file, "r") as f:
            args = json.load(f)

        npz_file = os.path.join(data_dir, folder, "train_stats.npz")
        npz = np.load(npz_file)
        stats['epoch'] = pd.Series(np.arange(len(npz['train_loss'])))
        stats['train_loss'] = pd.Series(np.log(npz['train_loss'] + 1e-10))
        stats['train_acc'] = pd.Series(npz['train_acc'])
        stats['val_loss'] = pd.Series(npz['val_loss'])
        stats['val_acc'] = pd.Series(npz['val_acc'])
        stats = pd.DataFrame(stats)
        stats['optimizer'] = args['optimizer']
        stats['learning_rate'] = args['lr']
        stats['batch_size'] = args['batch_size']
        all_stats.append(stats)

stats_df = pd.concat(all_stats)
stats_df = stats_df[stats_df['learning_rate'] >= 0.001]

# plot results
plt.figure(figsize=(12, 8))
for name, group in stats_df.groupby(['optimizer', 'learning_rate']):
    optimizer, lr = name
    label = f"{optimizer}, LR: {lr}"
    plt.plot(group['epoch'], group['train_loss'], label=label)

plt.ylim(-7, 0)

def scientific_notation(x, pos):
    return f'{10 ** x:.0e}'

plt.gca().yaxis.set_major_formatter(FuncFormatter(scientific_notation))

plt.title("A(b) MNIST convex: training loss residual P(w)-P(w*)", fontsize=15)
plt.xlabel("Epoch")
plt.ylabel("Training Loss - Optimum (Log Scale)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()


output_dir = "figures"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
plt.savefig(os.path.join(output_dir, "A(b).png"))
plt.show()
