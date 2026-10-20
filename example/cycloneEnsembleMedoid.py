import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ==========================================================
# 1. ENSEMBLE MEDOID FUNCTION (WITH MINIMUM LENGTH CHECK)
# ==========================================================
def find_ensemble_medoid(tracks):
  """tracks: list of arrays, each of shape (n_timesteps, 2) [lat, lon]

  Returns the index of the most representative real member.
  """
  n_members = len(tracks)
  distance_matrix = np.zeros((n_members, n_members))

  for i in range(n_members):
    for j in range(i + 1, n_members):
      min_len = min(len(tracks[i]), len(tracks[j]))

      if min_len == 0:
        continue

      # Mean Euclidean distance across common overlapping lead times
      diff = tracks[i][:min_len] - tracks[j][:min_len]
      dist = np.mean(np.linalg.norm(diff, axis=1))

      distance_matrix[i, j] = dist
      distance_matrix[j, i] = dist

  # Member with minimum sum of distances to all other members
  medoid_idx = np.argmin(distance_matrix.sum(axis=1))
  return medoid_idx, distance_matrix


# ==========================================================
# 2. LOAD DATASET
# ==========================================================
csv_path = "example/csv/tc_track_event_255_tracks.csv"
df = pd.read_csv(csv_path)
df["valid_time"] = pd.to_datetime(df["valid_time"])

# Group by unique track combinations
df["unique_track_id"] = df.groupby(["init_time", "ensemble"]).ngroup() + 1

# ==========================================================
# 3. FILTER OUT SHORT/COLLAPSED TRACKS (CRITICAL FIX)
# ==========================================================
# Require tracks to persist for at least 8 timesteps (e.g., 48-96 hours)
# This prevents dead/spurious tracks with only 1-2 points from gaming the distance metric
MIN_TIMESTEPS_REQUIRED = 8

track_lengths = df.groupby("unique_track_id")["valid_time"].count()
valid_track_ids = track_lengths[
    track_lengths >= MIN_TIMESTEPS_REQUIRED
].index.tolist()

print(
    f"Total unique tracks: {len(track_lengths)}, Qualified tracks (>="
    f" {MIN_TIMESTEPS_REQUIRED} steps): {len(valid_track_ids)}"
)

# Build tracks list ONLY from qualified long-lasting trajectories
tracks = []
track_id_map = []

for track_id in valid_track_ids:
  track_data = df[df["unique_track_id"] == track_id].sort_values(
      by="valid_time"
  )
  coords = track_data[["latitude", "longitude"]].to_numpy()
  tracks.append(coords)
  track_id_map.append(track_id)

# ==========================================================
# 4. CALCULATE MEDOID ACROSS QUALIFIED MEMBERS
# ==========================================================
medoid_idx, distance_matrix = find_ensemble_medoid(tracks)
selected_track_id = track_id_map[medoid_idx]
avg_dist_to_peers = distance_matrix[medoid_idx].sum() / (len(tracks) - 1)

print("\n=== Robust Medoid Identified ===")
print(f"Medoid Unique Track ID: {selected_track_id}")
print(f"Total Points in Medoid Track: {len(tracks[medoid_idx])}")
print(f"Average Euclidean Distance: {avg_dist_to_peers:.3f}°")
print("================================\n")

# Extract the medoid data for plotting
medoid_df = df[df["unique_track_id"] == selected_track_id].sort_values(
    "valid_time"
)

# ==========================================================
# 5. VISUALIZATION
# ==========================================================
plt.figure(figsize=(13, 8), dpi=150)

# Plot all ensemble members as thin background lines
for track_id, track_data in df.groupby("unique_track_id"):
  track_data = track_data.sort_values(by="valid_time")
  plt.plot(
      track_data["longitude"],
      track_data["latitude"],
      color="steelblue",
      linewidth=0.8,
      alpha=0.2,
  )

# Plot the real Medoid track
plt.plot(
    medoid_df["longitude"],
    medoid_df["latitude"],
    color="#0066cc",
    linestyle="-",
    linewidth=3.5,
    marker="o",
    markersize=6,
    label=(
        f"Ensemble Medoid (Track ID: {selected_track_id}, Points:"
        f" {len(medoid_df)})"
    ),
    zorder=9,
)

# Plot Ground Truth
actual_track = df.drop_duplicates(subset=["valid_time"]).sort_values(
    "valid_time"
)
plt.plot(
    actual_track["actual_longitude"],
    actual_track["actual_latitude"],
    color="black",
    marker="X",
    markersize=9,
    linewidth=2.8,
    label="Actual Track (Ground Truth)",
    zorder=10,
)

plt.title(
    "Ensemble Medoid Selection (Filtered for Complete Trajectories)",
    fontsize=15,
    fontweight="bold",
    pad=15,
)
plt.xlabel("Longitude (°)", fontsize=12)
plt.ylabel("Latitude (°)", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(loc="upper left", fontsize=10)
plt.tight_layout()

plt.savefig("example/figs/emedoid_fixed.png", dpi=300, bbox_inches="tight")
print("Plot saved successfully as medoid_fixed.png")
plt.show()