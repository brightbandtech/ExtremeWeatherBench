# Cyclone Track Clustering and Medoid Selection, DTW Distance Matrix
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics import r2_score
from sklearn.cluster import DBSCAN

# NOTE ON NORMALIZATION:
# Raw DTW cost is a SUM over the warping path, so a short track naturally
# accumulates a smaller total cost than a long track -- even if it is not
# genuinely a better match. This biases medoid/clustering toward short-lived
# members for the wrong reason (fewer points to sum, not better tracking).
# We divide by (len_a + len_b) so every pair is compared on a fair,
# per-step basis regardless of how long each member's track is.

try:
    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean

    def dtw_distance(seq_a, seq_b):
        raw_d, _ = fastdtw(seq_a, seq_b, dist=euclidean)
        return raw_d / (len(seq_a) + len(seq_b))

    print("Using fastdtw (fast approximate DTW), length-normalized.")
except ImportError:
    print("fastdtw not installed (pip install fastdtw) -> falling back to exact O(n*m) DTW. "
          "This may be slow for large ensembles with long tracks.")

    def dtw_distance(seq_a, seq_b):
        """Length-normalized Dynamic Time Warping distance between two (N,2)
        lat/lon sequences. Unlike Euclidean-on-padded-vectors, this correctly
        compares tracks of different lengths without needing to crop or pad
        them -- and dividing by (n+m) removes the bias toward short tracks
        that raw summed DTW cost would otherwise introduce."""
        n, m = len(seq_a), len(seq_b)
        D = np.full((n + 1, m + 1), np.inf)
        D[0, 0] = 0.0
        point_dist = cdist(seq_a, seq_b, metric="euclidean")
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                D[i, j] = point_dist[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
        return D[n, m] / (n + m)

from sklearn.cluster import AgglomerativeClustering

# ==========================================================
# 1. LOAD AND PREPARE DATASET
# ==========================================================
csv_path = "csv/tc_track_event_256_tracks.csv"
df_raw = pd.read_csv(csv_path)
df_raw["valid_time"] = pd.to_datetime(df_raw["valid_time"])

# Get a sorted list of all unique init_times and select the target run
unique_init_times = sorted(df_raw["init_time"].unique())
first_run = unique_init_times[0]

print(f"Available init_times: {unique_init_times}")
print(f"Selected init_time: {first_run}")

# Filter the dataset to keep ONLY the selected run
df_raw = df_raw[df_raw["init_time"] == first_run].copy()
df_raw["unique_track_id"] = df_raw.groupby(["init_time", "ensemble"]).ngroup() + 1
all_tracks = sorted(df_raw["unique_track_id"].unique())
print(f"Total Ensemble Members (before filtering): {len(all_tracks)}")

# ==========================================================
# 2. NOISE-FLOOR FILTERING -- ONLY DROP UNUSABLE TRACKS
# ==========================================================
# A short track is NOT automatically a bad/unreliable track -- if the real
# storm itself dissipated quickly, the short members are exactly the ones
# that got it right. A hard "must survive N steps" cutoff would wrongly
# punish them for being correct. So we no longer filter on "long enough to
# trust"; length-normalized DTW (above) already makes short and long tracks
# fairly comparable. We only drop tracks that are too short to define any
# meaningful trajectory shape at all (a single point / 2-point line has no
# real path to compare).
MIN_TRACK_STEPS = 3  # noise floor only: below this, there's no real "path" to compare

track_lengths = df_raw.groupby("unique_track_id").size()
long_lived_tracks = track_lengths[track_lengths >= MIN_TRACK_STEPS].index.tolist()
dropped_tracks = track_lengths[track_lengths < MIN_TRACK_STEPS].index.tolist()

print(f"Dropping {len(dropped_tracks)} member(s) with < {MIN_TRACK_STEPS} steps (noise floor only).")
print(f"Remaining members after filtering: {len(long_lived_tracks)} / {len(all_tracks)}")

df = df_raw[df_raw["unique_track_id"].isin(long_lived_tracks)].copy()

tracks = sorted(df["unique_track_id"].unique())
valid_times_sorted = sorted(df["valid_time"].unique())
print(f"Total Timesteps Available (post-filter): {len(valid_times_sorted)}")

# ==========================================================
# 3. ACTIVE-MEMBER COUNT PER TIMESTEP (ON FILTERED DATA)
# ==========================================================
# Any per-timestep statistic (mean, median, mode) is only meaningful once
# enough members are still alive. Below MIN_ACTIVE_N we mark the dynamic
# mean as low-confidence instead of presenting it as a real estimate.
# This count is now taken over the ALREADY-FILTERED df, so short-lived
# members never contribute to it.
MIN_ACTIVE_N = max(10, int(0.3 * len(tracks)))  # 30% of filtered ensemble, floor of 10

active_counts = df.groupby("valid_time").size()
dynamic_mean = df.groupby("valid_time")[["latitude", "longitude"]].mean().reset_index()
dynamic_mean["active_n"] = dynamic_mean["valid_time"].map(active_counts)
dynamic_mean["confident"] = dynamic_mean["active_n"] >= MIN_ACTIVE_N

# Same idea, median instead of mean:
dynamic_median = df.groupby("valid_time")[["latitude", "longitude"]].median().reset_index()
dynamic_median["active_n"] = dynamic_median["valid_time"].map(active_counts)
dynamic_median["confident"] = dynamic_median["active_n"] >= MIN_ACTIVE_N

actual_df = df.drop_duplicates(subset=["valid_time"]).sort_values("valid_time")
actual_lats = actual_df["actual_latitude"].values
actual_lons = actual_df["actual_longitude"].values

print(f"\nMinimum active members required for a 'confident' aggregate: {MIN_ACTIVE_N}")
print(dynamic_mean[["valid_time", "active_n", "confident"]].to_string(index=False))

# ==========================================================
# 4. FULL-LENGTH SEQUENCES PER MEMBER (FOR DTW / MEDOID / CLUSTERING)
# ==========================================================
# Each member's own natural lifespan is preserved (no truncation to a
# shared window, no padding). All members here already passed the
# MIN_TRACK_STEPS filter in Section 2.
member_sequences = {}
for tid in tracks:
    t_df = df[df["unique_track_id"] == tid].sort_values("valid_time")
    member_sequences[tid] = t_df[["latitude", "longitude"]].to_numpy()

eligible_tracks = list(member_sequences.keys())
n_eligible = len(eligible_tracks)

# ==========================================================
# 5. DTW DISTANCE MATRIX (HANDLES UNEQUAL-LENGTH TRACKS PROPERLY)
# ==========================================================
print(f"\nComputing DTW distance matrix for {n_eligible} eligible tracks "
      f"({n_eligible * (n_eligible - 1) // 2} pairs)...")

dtw_matrix = np.zeros((n_eligible, n_eligible))
for i in range(n_eligible):
    for j in range(i + 1, n_eligible):
        d = dtw_distance(member_sequences[eligible_tracks[i]], member_sequences[eligible_tracks[j]])
        dtw_matrix[i, j] = d
        dtw_matrix[j, i] = d

# ==========================================================
# 6. TRAJECTORY-LEVEL MEDOID (SOLVES THE "FRANKEN-TRACK" PROBLEM)
# ==========================================================
# Instead of picking the most central POINT at each timestep independently
# (which can stitch together a path no real storm ever flew), pick the
# single REAL member whose WHOLE TRACK is closest to all the others.
medoid_row = np.argmin(dtw_matrix.sum(axis=1))
trajectory_medoid_id = eligible_tracks[medoid_row]

# ==========================================================
# 7. SCENARIO CLUSTERING (HANDLES BIMODAL ENSEMBLES, E.G. RECURVE VS NO-RECURVE)
# ==========================================================
# If the ensemble genuinely splits into distinct scenarios, a single "central"
# track sits in the no-man's-land between clusters and looks outlier-driven
# even when no individual member is an outlier.
N_CLUSTERS = 3
cluster_model = AgglomerativeClustering(
    n_clusters=N_CLUSTERS, metric="precomputed", linkage="average"
)
cluster_labels = cluster_model.fit_predict(dtw_matrix)
# dbscan = DBSCAN(eps=0.05, min_samples=2, metric='precomputed')
# cluster_labels = dbscan.fit_predict(dtw_matrix)
cluster_sizes = pd.Series(cluster_labels).value_counts()
dominant_cluster = cluster_sizes.idxmax()
dominant_frac = cluster_sizes.max() / n_eligible
print(f"\nCluster sizes: {cluster_sizes.to_dict()}")
print(f"Dominant cluster: {dominant_cluster} ({dominant_frac:.0%} of eligible members)")


def cluster_medoid(cluster_id):
    idx = np.where(cluster_labels == cluster_id)[0]
    sub_matrix = dtw_matrix[np.ix_(idx, idx)]
    local_medoid = idx[np.argmin(sub_matrix.sum(axis=1))]
    return eligible_tracks[local_medoid]


dominant_medoid_id = cluster_medoid(dominant_cluster)

# Track IDs belonging to the dominant cluster -- needed to compute the
# per-timestep spread (uncertainty) of that scenario for plotting.
dominant_cluster_ids = [eligible_tracks[i] for i in np.where(cluster_labels == dominant_cluster)[0]]

# Only report a secondary scenario if it's non-trivial (not just 1-2 stray members)
secondary_scenario = None
if len(cluster_sizes) > 1:
    minority_cluster = cluster_sizes.idxmin()
    minority_frac = cluster_sizes.min() / n_eligible
    if minority_frac >= 0.15:
        secondary_scenario = cluster_medoid(minority_cluster)
        print(f"Secondary scenario cluster present ({minority_frac:.0%} of members) "
              f"-> Track ID {secondary_scenario}")

# ==========================================================
# 7b. DOMINANT-CLUSTER UNCERTAINTY (SPREAD AROUND THE MEDOID LINE)
# ==========================================================
# At each valid_time, look at ONLY the dominant-cluster members that are
# still active, and measure how spread out they are (std of lat, std of
# lon). This is the "how confident is the ensemble at this point in time"
# signal -- narrow where members agree, wide where they diverge.
dom_df = df[df["unique_track_id"].isin(dominant_cluster_ids)]
dom_spread = dom_df.groupby("valid_time").agg(
    mean_lat=("latitude", "mean"),
    mean_lon=("longitude", "mean"),
    std_lat=("latitude", "std"),
    std_lon=("longitude", "std"),
    n_members=("latitude", "size"),
).reset_index().sort_values("valid_time")

# std is NaN when only 1 member is present at that timestep -- treat as 0
dom_spread[["std_lat", "std_lon"]] = dom_spread[["std_lat", "std_lon"]].fillna(0.0)




# ==========================================================
# 9. PLOTTING THE RESULTS (CLUSTER-COLORED SPAGHETTI + DYNAMIC MEAN)
# ==========================================================
plt.figure(figsize=(16, 10), dpi=150)
ax = plt.gca()

# 1. Plot background tracks colored by their Cluster ID
# Map each eligible track ID to its cluster label
track_to_cluster = {}
for i, tid in enumerate(eligible_tracks):
    track_to_cluster[tid] = cluster_labels[i]

# Define distinct colors for clusters (e.g., Cluster 0 = Blue, Cluster 1 = Orange, etc.)
cluster_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

for track_id in all_tracks:
    t_df = df_raw[df_raw["unique_track_id"] == track_id].sort_values("valid_time")
    # If the track was eligible and clustered, color it by cluster; otherwise gray
    if track_id in track_to_cluster:
        c_id = track_to_cluster[track_id]
        col = cluster_colors[c_id % len(cluster_colors)]
        plt.plot(t_df["longitude"], t_df["latitude"], color=col, linewidth=0.8, alpha=0.3)
    else:
        plt.plot(t_df["longitude"], t_df["latitude"], color="gray", linewidth=0.4, alpha=0.1)

# 2. Ground Truth Actual Track
plt.plot(actual_lons, actual_lats, color="black", marker="X", markersize=8,
          linewidth=3, label="Actual Track (Ground Truth)", zorder=12)

# 3. Dynamic Mean (Restored)
conf = dynamic_mean["confident"].values
lons_m, lats_m = dynamic_mean["longitude"].values, dynamic_mean["latitude"].values
plt.plot(lons_m[conf], lats_m[conf], color="crimson", linestyle="--", linewidth=2.5,
          label=f"Dynamic Mean (active N >= {MIN_ACTIVE_N}, filtered)", zorder=10)
plt.plot(lons_m[~conf], lats_m[~conf], color="crimson", linestyle=":", linewidth=1.5,
          alpha=0.5, label="Dynamic Mean (low-confidence)", zorder=9)

# 4. Scenario Medoids
selected = {
    "Dominant-Cluster Medoid": (dominant_medoid_id, "#2ca02c", "s"),
}
if secondary_scenario is not None:
    selected["Alternative-Scenario Medoid"] = (secondary_scenario, "#ff7f0e", "^")

for i, (label, (tid, color, marker)) in enumerate(selected.items()):
    t_df = df[df["unique_track_id"] == tid].sort_values("valid_time")
    plt.plot(t_df["longitude"], t_df["latitude"], color=color, linewidth=3.0,
              marker=marker, markersize=7, label=f"{label} (ID {tid})", zorder=11 + i)

plt.title("Ensemble Member Clusters & Dynamic Mean vs Ground Truth\n"
          "Length-Normalized DTW Clustering",
          fontsize=15, fontweight="bold", pad=15)
plt.xlabel("Longitude (°)", fontsize=12)
plt.ylabel("Latitude (°)", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.5)

handles, labels_ = ax.get_legend_handles_labels()
plt.legend(handles, labels_, loc="upper left", fontsize=9, framealpha=0.9, edgecolor="black")
plt.tight_layout()

output_plot = "trajectory_clustered2_colored.png"
plt.savefig(output_plot, dpi=300, bbox_inches="tight")
plt.show()
print(f"\nPlot saved to: {output_plot}")

print("\n=== SELECTED OPERATIONAL TRACKS ===")
for label, (tid, _, _) in selected.items():
    print(f"- {label}: Track ID {tid}")

# ==========================================================
# 10. CALCULATE R2 SCORE: SELECTED PATH & MEAN vs ACTUAL
# ==========================================================

# 1. Prepare Ground Truth (Real) Track
real_track = actual_df[["valid_time", "actual_latitude", "actual_longitude"]].dropna()

# 2. Prepare Selected Path (Dominant-Cluster Medoid)
selected_track = df[df["unique_track_id"] == dominant_medoid_id][
    ["valid_time", "latitude", "longitude"]
]

# 3. Prepare Dynamic Mean Path
mean_track = dynamic_mean[["valid_time", "latitude", "longitude"]].rename(
    columns={"latitude": "mean_lat", "longitude": "mean_lon"}
)

# Merge datasets on valid_time to ensure time-aligned comparison
merged_selected = pd.merge(real_track, selected_track, on="valid_time", how="inner")
merged_mean = pd.merge(real_track, mean_track, on="valid_time", how="inner")

print("\n" + "="*60)
print("PERFORMANCE METRICS (R² SCORE) vs ACTUAL TRACK")
print("="*60)

# --- A. Selected Path (Medoid) Metrics ---
if not merged_selected.empty:
    r2_lat_sel = r2_score(merged_selected["actual_latitude"], merged_selected["latitude"])
    r2_lon_sel = r2_score(merged_selected["actual_longitude"], merged_selected["longitude"])
    
    print(f"1. Selected Path (Dominant Medoid ID: {dominant_medoid_id})")
    print(f"   - Aligned Timesteps : {len(merged_selected)}")
    print(f"   - R² Score (Lat)    : {r2_lat_sel:.4f}")
    print(f"   - R² Score (Lon)    : {r2_lon_sel:.4f}")
else:
    print("1. Selected Path: No matching timesteps found with Actual Track.")

print("-" * 60)

# --- B. Dynamic Mean Metrics ---
if not merged_mean.empty:
    r2_lat_mean = r2_score(merged_mean["actual_latitude"], merged_mean["mean_lat"])
    r2_lon_mean = r2_score(merged_mean["actual_longitude"], merged_mean["mean_lon"])
    
    print(f"2. Dynamic Mean Path")
    print(f"   - Aligned Timesteps : {len(merged_mean)}")
    print(f"   - R² Score (Lat)    : {r2_lat_mean:.4f}")
    print(f"   - R² Score (Lon)    : {r2_lon_mean:.4f}")
else:
    print("2. Dynamic Mean Path: No matching timesteps found with Actual Track.")

print("="*60 + "\n")