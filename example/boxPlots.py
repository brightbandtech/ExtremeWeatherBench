import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re

# ---------------------------------------------------------
# Step 1: Load the combined results CSV file
# ---------------------------------------------------------
csv_filename = "results.csv"  # Update path if needed
df = pd.read_csv(csv_filename)

# Verify required columns exist in the dataframe
required_cols = ['value', 'target_variable', 'metric', 'lead_time', 'event_type']
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' missing in CSV file!")

# Helper function to clean lead time strings (e.g., "0 days 06:00:00" -> "6 hr")
def clean_lead_time(val):
    val_str = str(val).strip()
    if "days" in val_str or ":" in val_str:
        try:
            parts = val_str.split()
            for p in parts:
                if ":" in p:
                    hours = int(p.split(":")[0])
                    return f"{hours} hr"
        except Exception:
            pass
    try:
        hours = int(float(val_str))
        return f"{hours} hr"
    except ValueError:
        return f"{val_str} hr"

def extract_hour(val):
    try:
        return int(val.replace(" hr", ""))
    except:
        return 0

# Mapping dictionary for short meteorological feature names
feature_short_names = {
    "surface_air_temperature": "T2m",
    "surface_eastward_wind": "U10",
    "surface_northward_wind": "V10",
    "mean_sea_level_pressure": "MSLP",
}

def get_short_name(name):
    return feature_short_names.get(name, name[:8] if len(str(name)) > 8 else str(name))

# ---------------------------------------------------------
# Step 2: Extract all unique metrics from the 'metric' column
# ---------------------------------------------------------
unique_metrics = df['metric'].dropna().unique()
print(f"Found {len(unique_metrics)} unique metric(s): {list(unique_metrics)}")

# ---------------------------------------------------------
# Step 3: Loop through each metric and generate separate plots
# ---------------------------------------------------------
for target_metric in unique_metrics:
    print(f"\nProcessing metric: {target_metric}...")
    
    subset_df = df[df['metric'] == target_metric].copy()
    if subset_df.empty:
        continue

    # Format lead times and feature names
    subset_df['lead_time_formatted'] = subset_df['lead_time'].apply(clean_lead_time)
    subset_df['lead_time_sort'] = subset_df['lead_time_formatted'].apply(extract_hour)
    subset_df['short_feature'] = subset_df['target_variable'].apply(get_short_name)

    # Sort data logically to maintain clean hierarchical order
    subset_df = subset_df.sort_values(by=['event_type', 'short_feature', 'lead_time_sort'])
    subset_df['group_key'] = subset_df['event_type'] + " | " + subset_df['short_feature'] + " | " + subset_df['lead_time_formatted']

    # Initialize figure canvas
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
    sns.set_theme(style="whitegrid")

    # Generate box plot with a very subtle gap (e.g., gap=0.1) between adjacent boxes
    sns.boxplot(
        data=subset_df,
        x="group_key",
        y="value",
        palette="Set2",
        width=0.85,  # Balanced width
        gap=0.01,     # Very minor/subtle gap between boxes
        fliersize=2,
        ax=ax
    )

    # Configure background grid lines
    ax.grid(True, which='major', axis='y', linestyle='-', linewidth=0.7, alpha=0.7)
    ax.grid(True, which='major', axis='x', linestyle=':', linewidth=0.5, alpha=0.5)

    # Set titles and labels
    plt.title(f"{target_metric} Distribution Across Features and Lead Times", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("", fontsize=12)
    plt.ylabel(f"{target_metric} Value", fontsize=12)

    # Fix tick warning by explicitly setting tick positions and labels
    xticks_positions = ax.get_xticks()
    xtick_labels = []
    for item in ax.get_xticklabels():
        parts = item.get_text().split(" | ")
        if len(parts) == 3:
            xtick_labels.append(parts[2])
        else:
            xtick_labels.append(item.get_text())

    ax.set_xticks(xticks_positions)
    ax.set_xticklabels(xtick_labels, rotation=0, ha='center', fontsize=10)

    # Place feature names once centered above their corresponding lead times
    unique_groups = subset_df[['event_type', 'short_feature']].drop_duplicates()
    current_pos = 0
    for idx, row in unique_groups.iterrows():
        ev = row['event_type']
        feat = row['short_feature']
        
        count = subset_df[(subset_df['event_type'] == ev) & (subset_df['short_feature'] == feat)]['lead_time_formatted'].nunique()
        center_pos = current_pos + (count - 1) / 2.0
        
        ax.text(center_pos, -0.12, f"{feat}", transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=10, fontweight='bold', color='black')
        
        current_pos += count

    # Place event type labels cleanly at the bottom, adding vertical separators
    unique_events = subset_df['event_type'].unique()
    current_pos = 0
    for i, event in enumerate(unique_events):
        event_data = subset_df[subset_df['event_type'] == event]
        total_boxes = event_data['group_key'].nunique()
        center_pos = current_pos + (total_boxes - 1) / 2.0
        
        # Display event name cleanly in capitalized format
        ax.text(center_pos, -0.22, f"{str(event).upper()}", transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=11, fontweight='bold', color='darkblue')
        
        # Draw vertical separator line between major event types
        if i > 0:
            ax.axvline(x=current_pos - 0.5, color='gray', linestyle='--', linewidth=1.2)
            
        current_pos += total_boxes

    # Adjust layout spacing to fit multi-tier custom axis labels
    plt.subplots_adjust(bottom=0.28)

    # Standardize output filenames based on metric names
    metric_str = str(target_metric).lower().strip()
    if "crps" in metric_str:
        plot_filename = "crps.png"
    elif "rootmeansquarederror" in metric_str or "rmse" in metric_str:
        plot_filename = "rmse.png"
    else:
        clean_name = re.sub(r'[^\w\-]', '_', metric_str)
        plot_filename = f"{clean_name}.png"

    # Save the figure tightly with minimal padding
    plt.savefig(plot_filename, bbox_inches='tight', pad_inches=0.1)
    print(f"Saved plot successfully: {plot_filename}")

    plt.close()