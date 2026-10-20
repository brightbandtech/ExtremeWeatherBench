import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------
# Step 1: Load and inspect dataset
# ---------------------------------------------------------
csv_filename = "example/csv/heatwave_evaluation_results_1.csv"  # Update path if needed
df = pd.read_csv(csv_filename)

required_cols = [
    "value",
    "target_variable",
    "metric",
    "lead_time",
    "event_type",
    "speds_component",
]
for col in required_cols:
    if col not in df.columns:
        if col == "speds_component":
            df["speds_component"] = "None"
        else:
            raise ValueError(f"Column '{col}' missing in CSV file!")

df["speds_component"] = df["speds_component"].fillna("None")


# ---------------------------------------------------------
# Step 2: Helper Functions & Component Mapping
# ---------------------------------------------------------
def parse_lead_time(val):
    val_str = str(val).strip()
    try:
        td = pd.to_timedelta(val_str)
        total_hours = int(td.total_seconds() // 3600)
        return f"{total_hours} hr", total_hours
    except Exception:
        pass
    try:
        hours = int(float(val_str))
        return f"{hours} hr", hours
    except ValueError:
        pass
    days_match = re.search(r"(\d+)\s*day", val_str)
    hours_match = re.search(r"(\d+):", val_str)
    days = int(days_match.group(1)) if days_match else 0
    hours = int(hours_match.group(1)) if hours_match else 0
    total = days * 24 + hours
    return f"{total} hr", total


feature_short_names = {
    "surface_air_temperature": "T2m",
    "surface_eastward_wind": "U10",
    "surface_northward_wind": "V10",
    "mean_sea_level_pressure": "MSLP",
}

def get_short_name(name):
    return feature_short_names.get(
        name, name[:8] if len(str(name)) > 8 else str(name)
    )

def assign_component_code(val):
    val_clean = str(val).strip().lower()
    if "tail" in val_clean:
        return "1"  # 1: Tail Score
    elif "grad" in val_clean:
        return "2"  # 2: Gradient
    elif "sharp" in val_clean or "spr" in val_clean:
        return "3"  # 3: Sharpness / Spread
    elif any(k in val_clean for k in ["tot", "all", "full", "speds", "score"]):
        return "4"  # 4: Total Score
    return "4"


# ---------------------------------------------------------
# Step 3: Filter and Prepare SPEDS Data
# ---------------------------------------------------------
speds_df = df[
    df["metric"].astype(str).str.contains("speds", case=False, na=False)
].copy()

if speds_df.empty:
    raise ValueError("No data found matching 'speds'. Check CSV entries!")

speds_df["comp_code"] = speds_df["speds_component"].apply(assign_component_code)
parsed_lead = speds_df["lead_time"].apply(parse_lead_time)
speds_df["lead_time_formatted"] = [p[0] for p in parsed_lead]
speds_df["lead_time_sort"] = [p[1] for p in parsed_lead]
speds_df["short_feature"] = speds_df["target_variable"].apply(get_short_name)

speds_df = speds_df.sort_values(by=["event_type", "short_feature", "lead_time_sort"])
speds_df["group_key"] = (
    speds_df["event_type"] + " | " + speds_df["short_feature"] + " | " + speds_df["lead_time_formatted"]
)

# Common Configurations
unique_ticks = speds_df["group_key"].nunique()
fig_width = max(16, unique_ticks * 0.45)
fig_height = 10  # Height for individual graphs

palette_by_number = {
    "1": "#D95F02",  # Tail (Orange)
    "2": "#1B9E77",  # Gradient (Green)
    "3": "#7570B3",  # Sharpness (Purple)
    "4": "#E7298A",  # Total (Magenta)
}

present_codes = [c for c in ["1", "2", "3", "4"] if c in speds_df["comp_code"].unique()]
label_descriptions = {
    "1": "1: Tail Score",
    "2": "2: Gradient",
    "3": "3: Sharpness",
    "4": "4: Total Score",
}

# ---------------------------------------------------------
# Helper Function for Multi-tier X-Axis & Separators
# ---------------------------------------------------------
def apply_custom_x_axis(ax, data_df):
    """Applies lead times, features, event types, and vertical dividers to any axis."""
    # 1. Primary x-axis ticks (Lead times)
    xticks_positions = ax.get_xticks()
    xtick_labels = []
    for item in ax.get_xticklabels():
        parts = item.get_text().split(" | ")
        xtick_labels.append(parts[2] if len(parts) == 3 else item.get_text())
    ax.set_xticks(xticks_positions)
    ax.set_xticklabels(xtick_labels, rotation=45, ha="right", fontsize=9)

    # 2. Feature level (T2m, U10, MSLP, etc.)
    unique_groups = data_df[["event_type", "short_feature"]].drop_duplicates()
    current_pos = 0
    for _, row in unique_groups.iterrows():
        ev = row["event_type"]
        feat = row["short_feature"]
        count = data_df[(data_df["event_type"] == ev) & (data_df["short_feature"] == feat)]["group_key"].nunique()
        center_pos = current_pos + (count - 1) / 2.0
        ax.text(center_pos, -0.16, f"{feat}", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=10, fontweight="bold", color="black")
        current_pos += count

    # 3. Event type level & Dividers
    unique_events = data_df["event_type"].unique()
    current_pos = 0
    for i, event in enumerate(unique_events):
        event_data = data_df[data_df["event_type"] == event]
        total_boxes = event_data["group_key"].nunique()
        center_pos = current_pos + (total_boxes - 1) / 2.0
        ax.text(center_pos, -0.26, f"{str(event).upper()}", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=11, fontweight="bold", color="darkblue")
        if i > 0:
            ax.axvline(x=current_pos - 0.5, color="gray", linestyle="--", linewidth=1.2)
        current_pos += total_boxes

def add_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    new_labels = [label_descriptions.get(lbl, lbl) for lbl in labels]
    ax.legend(handles=handles, labels=new_labels, title="SPEDS Component", 
              title_fontsize=11, fontsize=10, loc="upper right", frameon=True, framealpha=0.95)


# =========================================================
# GRAPH 1: BOX PLOT
# =========================================================
print("\nGenerating Box Plot...")
fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300)
sns.set_theme(style="whitegrid")

box_width = 0.8
sns.boxplot(
    data=speds_df, x="group_key", y="value", hue="comp_code",
    hue_order=present_codes, palette=palette_by_number,
    width=box_width, gap=0.08, fliersize=2, ax=ax,
)
ax.set_title("SPEDS Score Box Plot Distribution Across Features & Lead Times\n"
             "[ 1: Tail  |  2: Gradient  |  3: Sharpness  |  4: Total ]", 
             fontsize=15, fontweight="bold", pad=20)
ax.set_xlabel("")
ax.set_ylabel("SPEDS Value", fontsize=12)
ax.grid(True, which="major", axis="y", linestyle="-", linewidth=0.7, alpha=0.7)
ax.grid(True, which="major", axis="x", linestyle=":", linewidth=0.5, alpha=0.5)

# Numbers over boxes
n_boxes = len(present_codes)
slot_width = box_width / n_boxes
code_to_offset = dict(zip(present_codes, [-box_width / 2 + (k + 0.5) * slot_width for k in range(n_boxes)]))
y_min, y_max = ax.get_ylim()
y_span = y_max - y_min
ax.set_ylim(y_min, y_max + y_span * 0.12)

for tick_idx, g_key in enumerate(speds_df["group_key"].unique()):
    for code in present_codes:
        b_data = speds_df[(speds_df["group_key"] == g_key) & (speds_df["comp_code"] == code)]["value"].dropna()
        if not b_data.empty:
            ax.text(tick_idx + code_to_offset[code], b_data.max() + (y_span * 0.02),
                    str(code), ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=palette_by_number[code])

apply_custom_x_axis(ax, speds_df)
add_legend(ax)
plt.subplots_adjust(bottom=0.28)
plt.savefig("example/figs/spedsscore_boxplot.png", bbox_inches="tight", pad_inches=0.15)
plt.close()
print(" -> Saved: spedsscore_boxplot.png")


# =========================================================
# GRAPH 2: LINE CHART
# =========================================================
print("Generating Line Chart...")
fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300)
sns.set_theme(style="whitegrid")

sns.lineplot(
    data=speds_df, x="group_key", y="value", hue="comp_code",
    hue_order=present_codes, palette=palette_by_number,
    estimator="mean", errorbar=None, marker="o", markersize=6, linewidth=2, ax=ax,
)
ax.set_title("Mean SPEDS Components Trend Evolution Across Lead Times", 
             fontsize=15, fontweight="bold", pad=20)
ax.set_xlabel("")
ax.set_ylabel("Mean SPEDS Value", fontsize=12)
ax.grid(True, which="major", axis="y", linestyle="-", linewidth=0.7, alpha=0.7)
ax.grid(True, which="major", axis="x", linestyle=":", linewidth=0.5, alpha=0.5)

apply_custom_x_axis(ax, speds_df)
add_legend(ax)
plt.subplots_adjust(bottom=0.28)
plt.savefig("example/figs/spedsscore_linechart.png", bbox_inches="tight", pad_inches=0.15)
plt.close()
print(" -> Saved: spedsscore_linechart.png")


# =========================================================
# GRAPH 3: COLUMN / BAR GRAPH
# =========================================================
print("Generating Column Graph...")
fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300)
sns.set_theme(style="whitegrid")

sns.barplot(
    data=speds_df, x="group_key", y="value", hue="comp_code",
    hue_order=present_codes, palette=palette_by_number,
    estimator="mean", errorbar=None, width=0.8, ax=ax,
)
ax.set_title("Mean SPEDS Components Column Comparison", 
             fontsize=15, fontweight="bold", pad=20)
ax.set_xlabel("")
ax.set_ylabel("Mean SPEDS Value", fontsize=12)
ax.grid(True, which="major", axis="y", linestyle="-", linewidth=0.7, alpha=0.7)
ax.grid(True, which="major", axis="x", linestyle=":", linewidth=0.5, alpha=0.5)

apply_custom_x_axis(ax, speds_df)
add_legend(ax)
plt.subplots_adjust(bottom=0.28)
plt.savefig("example/figs/spedsscore_barchart.png", bbox_inches="tight", pad_inches=0.15)
plt.close()
print(" -> Saved: spedsscore_barchart.png\n[ALL DONE]")