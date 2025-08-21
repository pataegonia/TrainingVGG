import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_combined_results(csv_files, save_path=None):
    """
    Plots combined training and validation loss, and validation accuracy from multiple CSV files.

    Args:
        csv_files (list): A list of paths to the CSV files.
        save_path (str, optional): The path to save the combined plot. If None, the plot is displayed.
    """
    if not csv_files:
        print("Error: No CSV files provided.")
        return

    all_data = {}
    for file in csv_files:
        if not os.path.exists(file):
            print(f"Warning: The file '{file}' was not found. Skipping.")
            continue
        
        try:
            df = pd.read_csv(file)
            # Use the filename as the key
            file_name = os.path.basename(file).split('.')[0]
            all_data[file_name] = df
        except pd.errors.EmptyDataError:
            print(f"Warning: The file '{file}' is empty. Skipping.")
            continue
        except Exception as e:
            print(f"An error occurred while reading '{file}': {e}. Skipping.")
            continue

    if not all_data:
        print("Error: No valid data found in the provided CSV files.")
        return

    # Create a figure with 3 subplots for the different metrics
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    fig.suptitle("Comparison of Training Metrics from Multiple Runs", fontsize=20)

    # Plot 1: Train Loss
    for name, data in all_data.items():
        if 'train_loss' in data.columns and 'epoch' in data.columns:
            axes[0].plot(data['epoch'], data['train_loss'], label=f'{name} - Train Loss')
    axes[0].set_title('Training Loss over Epochs', fontsize=16)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend()
    axes[0].grid(True)

    # Plot 2: Validation Loss
    for name, data in all_data.items():
        if 'val_loss' in data.columns and 'epoch' in data.columns:
            axes[1].plot(data['epoch'], data['val_loss'], label=f'{name} - Val Loss')
    axes[1].set_title('Validation Loss over Epochs', fontsize=16)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend()
    axes[1].grid(True)

    # Plot 3: Validation Accuracy
    for name, data in all_data.items():
        if 'val_accuracy' in data.columns and 'epoch' in data.columns:
            axes[2].plot(data['epoch'], data['val_accuracy'], label=f'{name} - Val Accuracy')
    axes[2].set_title('Validation Accuracy over Epochs', fontsize=16)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Accuracy (%)', fontsize=12)
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path)
        print(f"Combined plot saved to {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    # Provide a list of specific file paths
    files_to_plot = ['training_log.csv', 'ResNet_result.csv']
    plot_combined_results(files_to_plot, save_path='combined_results1.png')