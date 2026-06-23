import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from itertools import cycle

def read_and_combine(folder):
    """
    Reads all CSV files in `folder`, adds a column with the file's base name,
    and returns a single DataFrame.
    """
    all_data = []
    csv_files = [f for f in os.listdir(folder) if f.lower().endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in folder: {folder}")

    for filename in csv_files:
        filepath = os.path.join(folder, filename)
        df = pd.read_csv(filepath)

        # Title = filename without extension
        raw_title = os.path.splitext(filename)[0]
        # Strip the last two characters (e.g., "v2") – use only if filename length > 2
        title = raw_title[:-2] if len(raw_title) > 2 else raw_title
        df.insert(0, 'source_file', title)

        # ---- Column checks and data conversion ----
        required_cols = ['launch_date', 'delta_v_ms', 'total_tof_days']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(
                    f"Missing column '{col}' in {filename}. "
                    f"Available columns: {list(df.columns)}"
                )

        # Convert launch_date to datetime (and drop rows where conversion fails)
        df['launch_date'] = pd.to_datetime(df['launch_date'], errors='coerce')
        df.dropna(subset=['launch_date'], inplace=True)

        all_data.append(df)

    combined = pd.concat(all_data, ignore_index=True)
    return combined

def plot_total_deltav_vs_time(df):
    """
    Plots 'delta_v_ms' vs 'launch_date'.
    Marker shape distinguishes 'source_file'.
    Colour distinguishes 'total_tof_days'.
    Only points with delta_v_ms <= 6000 are shown.
    X-axis labelled with years.
    """
    # Apply cutoff
    df = df[df['delta_v_ms'] <= 7500]
    if df.empty:
        print("No data points with delta_v_ms <= 6000 m/s to plot.")
        return

    # Unique file titles -> assign distinct markers
    sources = df['source_file'].unique()
    markers = cycle(['o', 's', 'D', '^', 'v', '<', '>', '*', 'x', 'h', 'H', '+', 'p', '|', '_'])
    source_to_marker = {src: next(markers) for src in sources}

    fig, ax = plt.subplots(figsize=(10, 6))

    vmin = df['total_tof_days'].min()/365
    vmax = df['total_tof_days'].max()/365
    print(vmin,vmax)

    # Scatter plot: each source gets its own scatter call for the legend
    for src in sources:
        subset = df[df['source_file'] == src]
        sc = ax.scatter(
            subset['launch_date'],
            subset['delta_v_ms'],
            c=subset['total_tof_days']/365,
            marker=source_to_marker[src],
            label=src,
            cmap='coolwarm',
            edgecolor='k',
            linewidth=0.5,
            s=60,
            vmin=vmin,
            vmax=vmax,
        )

    # Format x-axis to show years
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    # Colour bar for time of flight
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Total Time of Flight (years)')

    ax.set_xlabel('Launch Date (year)')
    ax.set_ylabel('Total Delta‑V (m/s)')
    #ax.set_title('Total Delta‑V vs Launch Date (≤ 6000 m/s)')
    #ax.legend(title='File (marker shape)',loc='upper center')
    ax.grid(True, linestyle='--', alpha=0.6)

    legend = ax.legend(title='Gravity Assist type',loc='upper center')
    for handle in legend.legend_handles:
        handle.set_facecolor('none')
        handle.set_edgecolor('black')

    plt.axhline(y=6000, color='r', linestyle='-')
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(
        description="Combine CSVs and plot total delta‑v against launch date."
    )
    parser.add_argument(
        'folder', nargs='?', default='.',
        help='Folder containing CSV files (default: current directory)'
    )
    args = parser.parse_args()

    combined_df = read_and_combine(args.folder)
    plot_total_deltav_vs_time(combined_df)

if __name__ == "__main__":
    main()