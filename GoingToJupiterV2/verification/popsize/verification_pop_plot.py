import os
import glob
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_second_column_per_file(folder_path):
    """
    Reads all CSV files in folder and returns a list of pandas Series,
    each Series contains the second column (index 1) of one file,
    preserving the original row index.
    """
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in '{folder_path}'")
    csv_files.sort()

    series_list = []
    for file in csv_files:
        try:
            df = pd.read_csv(file, header=None, usecols=[1])
            col = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna()
            series_list.append(col)
        except Exception as e:
            print(f"Warning: Could not read file '{file}'. Error: {e}")

    if not series_list:
        raise ValueError("No valid numeric data found in any CSV file.")
    return series_list

def compute_rowwise_stats(series_list):
    """
    Aligns all series by row index (outer join) and computes the mean and
    standard deviation for each row across files.
    Returns:
        row_indices: list of row indices
        means: array of mean values
        stds: array of standard deviation values
    """
    df = pd.concat(series_list, axis=1, ignore_index=True)
    means = df.mean(axis=1, skipna=True)
    stds = df.std(axis=1, ddof=1, skipna=True)
    row_indices = df.index.values

    valid_rows = ~(means.isna() & stds.isna())
    row_indices = row_indices[valid_rows]
    means = means[valid_rows]
    stds = stds[valid_rows]

    return row_indices, means.values, stds.values

def plot_rowwise_stats(row_indices, means, stds):
    """
    Plots the mean and standard deviation, and adds a note about how many
    std values are below 0.1.
    """
    valid_stds = stds[np.isfinite(stds)]
    count_below_0_1 = np.sum(valid_stds < 0.1)

    plt.figure(figsize=(12, 6))

    plt.plot(row_indices, means, 'o-', color='blue', linewidth=2, markersize=4, label='Mean')
    plt.plot(row_indices, stds, 's-', color='red', linewidth=2, markersize=4, label='Std Dev')

    note_text = f"Number of months with std < 0.1: {count_below_0_1}"
    plt.annotate(note_text, xy=(0.02, 0.95), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black"),
                 fontsize=10, verticalalignment='top')

    plt.xlabel('2036-42 launch dates in months', fontsize=12)
    plt.ylabel('Delta V', fontsize=12)
    plt.title('Effects of different seeds on results of VEE optimisations', fontsize=14)  # Updated title
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    try:
        series_list = load_second_column_per_file(folder)
        print(f"Loaded {len(series_list)} CSV files.")

        row_idx, means, stds = compute_rowwise_stats(series_list)

        print(f"\nNumber of rows with data: {len(row_idx)}")
        print("\nFirst 10 rows (index, mean, std):")
        for i in range(min(10, len(row_idx))):
            print(f"  Row {row_idx[i]:3d}: mean = {means[i]:10.4f}, std = {stds[i]:10.4f}")

        plot_rowwise_stats(row_idx, means, stds)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()