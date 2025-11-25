import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_two_models_comparison(model1_df, model2_df, model1_name, model2_name, figname):
    """
    Plot: Compare balanced accuracy between two models.
    
    Parameters:
    -----------
    model1_df : DataFrame
        Data for first model (with Balanced_Accuracy column)
    model2_df : DataFrame
        Data for second model (with Balanced_Accuracy column)
    model1_name : str
        Display name for first model
    model2_name : str
        Display name for second model
    figname : str
        Output figure filename
    """
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams.update({'font.size': 14})
    
    # Color scheme
    box_color = ['#ABCFE3', '#F3CC99']
    edge_color = ['#459BB8', '#E58A43']
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Extract balanced accuracy scores
    model1_scores = model1_df['Balanced_Accuracy'].values
    model2_scores = model2_df['Balanced_Accuracy'].values
    
    # Calculate means and stds
    model1_mean = np.mean(model1_scores)
    model1_std = np.std(model1_scores)
    model2_mean = np.mean(model2_scores)
    model2_std = np.std(model2_scores)
    
    # Bar positions
    bar_width = 0.30
    x = np.array([0, 1])
    
    # Plot bars
    means = [model1_mean, model2_mean]
    stds = [model1_std, model2_std]
    
    for i, (mean, std, color, edge) in enumerate(zip(means, stds, box_color, edge_color)):
        error_kw = {'ecolor': edge, 'elinewidth': 2}
        ax.bar(x[i], mean, width=bar_width,
               linewidth=1.5, zorder=2,
               facecolor=color,
               edgecolor=edge,
               yerr=std, capsize=4, alpha=0.7,
               error_kw=error_kw)
        
        # Plot individual data points
        scores = model1_scores if i == 0 else model2_scores
        jitter = np.random.uniform(-bar_width/4, bar_width/4, size=len(scores))
        ax.scatter(np.full_like(scores, x[i]) + jitter,
                  scores,
                  facecolors=edge,
                  edgecolors='black',
                  s=50,
                  marker='o',
                  zorder=3,
                  linewidths=0.5)
    
    # Formatting
    ax.set_ylabel('Balanced Accuracy', fontsize=16)
#     ax.set_xlabel('Models', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels([model1_name, model2_name], fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=16, width=2, length=6)
    ax.set_ylim(0, 1.0)
    ax.set_xlim(-0.5, 1.5)
    
    # Set spine width
    bwith = 2
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    
    plt.tight_layout()
    plt.savefig(figname, dpi=300)
    plt.close()
    print(f"Saved comparison plot to {figname}")

# Example usage:
# Assuming you have two DataFrames for two different models
model1_df = pd.read_csv('./pred_Fus_FusC_ds/CV/Fus_FusC_summary_metrics.csv')  # Your provided data
model2_df = pd.read_csv('./fine_tune_Fus_FusC_ds/CV/Fus_FusC_summary_statistics.csv')  # Load second model data

plot_two_models_comparison(
    model1_df, 
    model2_df,
    'Original Model',
    'Fine-tuned Model',
    'model_comparison.png'
)