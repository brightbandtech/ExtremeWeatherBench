# Standard K-Mean Clustering of Cyclone Tracks with Point-Wise Probabilities, Spreadness, and Distance to Overall Mean
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# ==========================================================
# 0. HAVERSINE DISTANCE HELPER FOR KILOMETERS
# ==========================================================
def haversine_np(lat1, lon1, lat2, lon2):
  R = 6371.0  # Earth radius in km
  lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
  dlat = lat2 - lat1
  dlon = lon2 - lon1
  a = (
      np.sin(dlat / 2.0) ** 2
      + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
  )
  return 2 * R * np.arcsin(np.sqrt(a))


# ==========================================================
# 1. LOAD AND PREPARE DATASET
# ==========================================================
csv_path = "example/csv/tc_track_event_255_tracks.csv"
df = pd.read_csv(csv_path)

df["valid_time"] = pd.to_datetime(df["valid_time"])

# Generate continuous track IDs across all (init_time, ensemble) combinations
df["unique_track_id"] = df.groupby(["init_time", "ensemble"]).ngroup() + 1
total_unique_tracks = df["unique_track_id"].nunique()
print(f"Total unique tracks identified: {total_unique_tracks}")

# ==========================================================
# 2. CONSTRUCT TRAJECTORY FEATURE MATRIX FOR K-MEANS
# ==========================================================
# Pivot latitudes and longitudes with tracks as rows and timesteps as columns
lat_matrix = df.pivot(
    index="unique_track_id", columns="valid_time", values="latitude"
)
lon_matrix = df.pivot(
    index="unique_track_id", columns="valid_time", values="longitude"
)

# Handle storm dissipation with forward and backward fill
lat_matrix = lat_matrix.ffill(axis=1).bfill(axis=1)
lon_matrix = lon_matrix.ffill(axis=1).bfill(axis=1)

# Ensure strictly unique column names to avoid validation errors
lat_matrix.columns = [f"lat_{t}" for t in lat_matrix.columns]
lon_matrix.columns = [f"lon_{t}" for t in lon_matrix.columns]

features = pd.concat([lat_matrix, lon_matrix], axis=1)

# ==========================================================
# 3. K-MEANS CLUSTERING & OVERALL PROBABILITIES
# ==========================================================
NUM_CLUSTERS = 3
kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(features.values)

track_to_cluster = pd.Series(cluster_labels, index=features.index)
df["cluster"] = df["unique_track_id"].map(track_to_cluster)

# Calculate overall track proportions
overall_cluster_counts = df.groupby("cluster")["unique_track_id"].nunique()
overall_probabilities = (overall_cluster_counts / total_unique_tracks) * 100

# ==========================================================
# 4. COMPUTE POINT-BY-POINT COORDINATES, PROBABILITIES & METRICS
# ==========================================================
# Group by cluster and valid_time to compute mean coordinates, active counts, and spread (std dev)
point_stats = (
    df.groupby(["cluster", "valid_time"])
    .agg(
        mean_lat=("latitude", "mean"),
        mean_lon=("longitude", "mean"),
        active_members=("unique_track_id", "nunique"),
        lat_std=("latitude", "std"),
        lon_std=("longitude", "std"),
    )
    .reset_index()
)

# 1. Calculate total active members per valid_time across all clusters
total_active_per_time = df.groupby("valid_time")["unique_track_id"].nunique()

# 2. Map this total active count back to point_stats using valid_time directly
point_stats["total_active_all"] = point_stats["valid_time"].map(
    total_active_per_time
)

# 3. Active ratio probability at each specific point (%)
point_stats["point_prob_active_pct"] = (
    point_stats["active_members"] / point_stats["total_active_all"]
) * 100

# 4. Internal Dispersion / Spreadness (magnitude of lat/lon std dev)
point_stats["spread_deg"] = np.sqrt(
    point_stats["lat_std"] ** 2 + point_stats["lon_std"] ** 2
).fillna(0.0)

# 5. Fixed scenario probability (relative to total ensemble size)
point_stats["cluster_total_members"] = point_stats["cluster"].map(
    overall_cluster_counts
)
point_stats["overall_prob_pct"] = point_stats["cluster"].map(
    overall_probabilities
)

# 6. Compute Overall Ensemble Mean Track (For plotting and distance calculation)
overall_mean = (
    df.groupby("valid_time")[["latitude", "longitude"]]
    .mean()
    .reset_index()
    .sort_values("valid_time")
)
overall_mean_renamed = overall_mean.rename(
    columns={"latitude": "mean_lat_all", "longitude": "mean_lon_all"}
)

# 7. Calculate Distance from Cluster Mean to Overall Mean at each timestep (in km)
point_stats = pd.merge(point_stats, overall_mean_renamed, on="valid_time", how="left")
point_stats["dist_to_overall_mean_km"] = haversine_np(
    point_stats["mean_lat"],
    point_stats["mean_lon"],
    point_stats["mean_lat_all"],
    point_stats["mean_lon_all"],
)

# ==========================================================
# 5. PRINT POINT-WISE PROBABILITIES & METRICS TO CONSOLE
# ==========================================================
print("\n" + "=" * 105)
print("POINT-BY-POINT SCENARIO PROBABILITIES, SPREADNESS & DISTANCE TO MEAN")
print("=" * 105)

for c_id in range(NUM_CLUSTERS):
  cluster_df = point_stats[point_stats["cluster"] == c_id].sort_values(
      "valid_time"
  )
  print(
      f"\n--- Cluster {c_id + 1} (Overall Scenario Probability:"
      f" {overall_probabilities[c_id]:.1f}%, Total Members:"
      f" {overall_cluster_counts[c_id]}) ---"
  )

  display_table = cluster_df[[
      "valid_time",
      "mean_lat",
      "mean_lon",
      "active_members",
      "point_prob_active_pct",
      "spread_deg",
      "dist_to_overall_mean_km",
  ]].copy()

  display_table.columns = [
      "Valid Time",
      "Mean Lat (°)",
      "Mean Lon (°)",
      "Active Count",
      "Point Prob (%)",
      "Spread (±°)",
      "Dist to Mean (km)",
  ]
  display_table["Mean Lat (°)"] = display_table["Mean Lat (°)"].round(2)
  display_table["Mean Lon (°)"] = display_table["Mean Lon (°)"].round(2)
  display_table["Point Prob (%)"] = display_table["Point Prob (%)"].round(1)
  display_table["Spread (±°)"] = display_table["Spread (±°)"].round(2)
  display_table["Dist to Mean (km)"] = display_table["Dist to Mean (km)"].round(
      1
  )

  print(display_table.to_string(index=False))

print("=" * 105 + "\n")

# ==========================================================
# 6. VISUALIZATION
# ==========================================================
plt.figure(figsize=(16, 10), dpi=150)
palette = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]

# 1. Plot thin background spaghetti tracks
for unique_id, track_data in df.groupby("unique_track_id"):
  c_id = track_data["cluster"].iloc[0]
  track_data = track_data.sort_values("valid_time")
  plt.plot(
      track_data["longitude"],
      track_data["latitude"],
      color=palette[c_id % len(palette)],
      linewidth=0.8,
      alpha=0.2,
  )

# 2. Plot Cluster Mean Tracks with Annotated Point Stats
for c_id in range(NUM_CLUSTERS):
  c_data = point_stats[point_stats["cluster"] == c_id].sort_values("valid_time")
  cluster_color = palette[c_id % len(palette)]

  plt.plot(
      c_data["mean_lon"],
      c_data["mean_lat"],
      color=cluster_color,
      linestyle="-",
      linewidth=3.0,
      marker="o",
      markersize=6,
      label=(
          f"Cluster {c_id + 1} ({overall_probabilities[c_id]:.1f}%,"
          f" N={overall_cluster_counts[c_id]})"
      ),
      zorder=8,
  )

  # Annotate probability, spreadness, and distance to overall mean near every point
  for _, row in c_data.iterrows():
    annotation_text = (
        f"{row['point_prob_active_pct']:.0f}%"
        f" (±{row['spread_deg']:.1f}°,"
        f" {row['dist_to_overall_mean_km']:.0f}km)"
    )
    plt.text(
        row["mean_lon"] + 0.05,
        row["mean_lat"] + 0.05,
        annotation_text,
        fontsize=6.5,
        color='black',
        fontweight="bold",
        zorder=9,
    )

# ==========================================
# PLOT OVERALL ENSEMBLE MEAN
# ==========================================
plt.plot(
    overall_mean["longitude"],
    overall_mean["latitude"],
    color="crimson",
    linestyle="--",
    linewidth=3.0,
    marker="s",
    markersize=6,
    label="Overall Ensemble Mean",
    zorder=9,
)

# 3. Plot Actual Ground Truth Track
actual_track = df.drop_duplicates(subset=["valid_time"]).sort_values("valid_time")
plt.plot(
    actual_track["actual_longitude"],
    actual_track["actual_latitude"],
    color="black",
    marker="X",
    markersize=9,
    linewidth=3.0,
    label="Actual Track (Ground Truth)",
    zorder=10,
)

# Configuration & layout
plt.title(
    "Cyclone Ensemble Clustering with Point-Wise Metrics\nProbability,"
    " Internal Spread (±°), & Distance to Overall Mean (km)",
    fontsize=15,
    fontweight="bold",
    pad=15,
)
plt.xlabel("Longitude (°)", fontsize=12)
plt.ylabel("Latitude (°)", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(loc="upper left", fontsize=9)
plt.tight_layout()

output_plot = "kmeans_pointwise_probabilities.png"
plt.savefig(output_plot, dpi=300, bbox_inches="tight")
print(f"Plot saved to: {output_plot}")

plt.show()

