#Grand Ensemble Mean Track
import matplotlib.pyplot as plt
import pandas as pd

# 1. Load the cyclone track dataset
csv_path = "example/csv/tc_track_event_255_tracks.csv"
df = pd.read_csv(csv_path)

# Parse valid_time as datetime for proper chronological sequencing
df["valid_time"] = pd.to_datetime(df["valid_time"])

# ==========================================================
# 2. GENERATE CONTINUOUS UNIQUE TRACK IDS
# ==========================================================
# Assign a unique continuous identifier across all (init_time, ensemble) combinations
# This prevents pandas from mixing up "Member 1" from different initialization times
df["unique_track_id"] = df.groupby(["init_time", "ensemble"]).ngroup() + 1
print(f"Total individual tracks found: {df['unique_track_id'].max()}")

# ==========================================================
# 3. COMPUTE THE GRAND ENSEMBLE MEAN TRACK
# ==========================================================
# Aggregate mean coordinates across all active members at each valid_time
grand_mean_track = (
    df.groupby("valid_time")[["latitude", "longitude"]]
    .mean()
    .reset_index()
    .sort_values("valid_time")
)

# ==========================================================
# 4. PLOTTING SETUP
# ==========================================================
plt.figure(figsize=(14, 9), dpi=150)

# Plot individual continuous tracks with low opacity
for unique_id, track_data in df.groupby("unique_track_id"):
    track_data = track_data.sort_values(by="valid_time")
    plt.plot(
        track_data["longitude"],
        track_data["latitude"],
        color="steelblue",
        linewidth=0.8,
        alpha=0.25,  # Low alpha to reveal spatial track density without clutter
    )

# ==========================================================
# 5. PLOT GRAND MEAN TRACK & GROUND TRUTH
# ==========================================================
# Plot the grand mean trajectory across all runs
plt.plot(
    grand_mean_track["longitude"],
    grand_mean_track["latitude"],
    color="crimson",
    linestyle="--",
    linewidth=3.0,
    marker="o",
    markersize=6,
    label="Grand Ensemble Mean Track",
    zorder=9,  # Render above individual tracks
)

# Deduplicate and plot the actual ground-truth path
actual_track = df.drop_duplicates(subset=["valid_time"]).sort_values("valid_time")

plt.plot(
    actual_track["actual_longitude"],
    actual_track["actual_latitude"],
    color="black",
    marker="X",
    markersize=9,
    linewidth=2.5,
    label="Actual Track (Ground Truth)",
    zorder=10,  # Keep ground truth on the absolute top layer
)

# ==========================================================
# 6. DISPLAY CONFIGURATION & SAVE
# ==========================================================
plt.title(
    "Tropical Cyclone Tracks: All Runs, Grand Mean & Ground Truth",
    fontsize=16,
    fontweight="bold",
    pad=15,
)
plt.xlabel("Longitude (°)", fontsize=12)
plt.ylabel("Latitude (°)", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.5)

# Place legend in the upper left, avoiding the messy individual track labels
plt.legend(loc="upper left", fontsize=11)

plt.tight_layout()

# Save high-resolution figure to disk
output_file = "example/figs/all_inits_with_grand_mean.png"
plt.savefig(output_file, dpi=300, bbox_inches="tight")
print(f"Plot saved to: {output_file}")

plt.show()