# analysis/utils/visualization_fn.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_age_histogram(ages, title="Age Distribution", bins=30, save_path=None):
    """Plot histogram of ages."""
    plt.figure(figsize=(10, 6))
    sns.histplot(ages, bins=bins, kde=True, color='skyblue')
    plt.xlabel("Age (years)")
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_age_by_group(df, age_col='age', group_col='split', title="Age by Group", save_path=None):
    """Boxplot of ages across groups."""
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x=group_col, y=age_col, hue=group_col, palette='Set2', legend=False)
    sns.swarmplot(data=df, x=group_col, y=age_col, color='black', alpha=0.5, size=3)
    plt.title(title)
    plt.grid(alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_age_vs_sex(ages, sexes, title="Age by Sex", save_path=None):
    """Boxplot of age separated by sex."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Prepare data
    unique_sexes = np.unique(sexes)
    data_by_sex = [ages[sexes == s] for s in unique_sexes]
    
    plt.figure(figsize=(10, 6))
    bp = plt.boxplot(data_by_sex, labels=unique_sexes, patch_artist=True,
                     boxprops=dict(facecolor='lightblue', alpha=0.7),
                     medianprops=dict(color='red', linewidth=2))
    # Add swarmplot-like points (jitter)
    for i, s in enumerate(unique_sexes):
        y = ages[sexes == s]
        x = np.random.normal(i+1, 0.04, size=len(y))
        plt.plot(x, y, 'k.', alpha=0.3, markersize=4)
    
    plt.title(title)
    plt.ylabel("Age (years)")
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_intensity_histogram(image_data, title="Voxel Intensity Distribution", bins=50, save_path=None):
    """Plot histogram of image intensities."""
    plt.figure(figsize=(10, 6))
    sns.histplot(image_data.flatten(), bins=bins, color='green', kde=True)
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")
    plt.title(title)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def show_mid_slice(image_data, slice_axis=2, slice_idx=None, title="Mid Slice", cmap='gray', save_path=None):
    """Display a middle slice of a 3D volume."""
    if slice_idx is None:
        slice_idx = image_data.shape[slice_axis] // 2
    if slice_axis == 0:
        slice_img = image_data[slice_idx, :, :]
    elif slice_axis == 1:
        slice_img = image_data[:, slice_idx, :]
    else:
        slice_img = image_data[:, :, slice_idx]
    plt.figure(figsize=(8, 8))
    plt.imshow(slice_img, cmap=cmap)
    plt.axis('off')
    plt.title(title)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()