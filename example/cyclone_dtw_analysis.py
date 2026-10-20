import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics import r2_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture

# ==========================================================
# HELPER FUNCTIONS
# ==========================================================
def calc_distances_km(lat1, lon1, lat2, lon2):
    """Calculates Haversine physical distance in kilometers."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    return 2 * R * np.arcsin(np.sqrt(a))

try:
    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean

    def dtw_distance(seq_a, seq_b):
        raw_d, _ = fastdtw(seq_a, seq_b, dist=euclidean)
        return raw_d / (len(seq_a) + len(seq_b))

    def dtw_path(seq_a, seq_b):
        _, path = fastdtw(seq_a, seq_b, dist=euclidean)
        return path

    print("Using fastdtw (fast approximate DTW), length-normalized.")
except ImportError:
    print("fastdtw not installed (pip install fastdtw) -> falling back to exact O(n*m) DTW.")

    def _dtw_full(seq_a, seq_b):
        n, m = len(seq_a), len(seq_b)
        D = np.full((n + 1, m + 1), np.inf)
        D[0, 0] = 0.0
        point_dist = cdist(seq_a, seq_b, metric="euclidean")
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                D[i, j] = point_dist[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
        return D

    def dtw_distance(seq_a, seq_b):
        n, m = len(seq_a), len(seq_b)
        D = _dtw_full(seq_a, seq_b)
        return D[n, m] / (n + m)

    def dtw_path(seq_a, seq_b):
        n, m = len(seq_a), len(seq_b)
        D = _dtw_full(seq_a, seq_b)
        i, j = n, m
        path = []
        while i > 0 and j > 0:
            path.append((i - 1, j - 1))
            step = np.argmin([D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]])
            if step == 0:
                i, j = i - 1, j - 1
            elif step == 1:
                i -= 1
            else:
                j -= 1
        path.reverse()
        return path

# ==========================================================
# 1. LOAD AND PREPARE DATASET
# ==========================================================
csv_path = "csv/tc_track_event_256_tracks.csv"
df_raw = pd.read_csv(csv_path)
df_raw["valid_time"] = pd.to_datetime(df_raw["valid_time"])

# Get a sorted list of all unique init_times and select the target run
unique_init_times = sorted(df_raw["init_time"].unique())
first_run = unique_init_times[4]

print(f"Available init_times: {unique_init_times}")
print(f"Selected init_time: {first_run}")

# Filter the dataset to keep ONLY the selected run
df_raw = df_raw[df_raw["init_time"] == first_run].copy()
df_raw["unique_track_id"] = df_raw.groupby(["init_time", "ensemble"]).ngroup() + 1
all_tracks = sorted(df_raw["unique_track_id"].unique())
print(f"Total Ensemble Members (before filtering): {len(all_tracks)}")

# ==========================================================
# 2. NOISE-FLOOR FILTERING
# ==========================================================
MIN_TRACK_STEPS = 3  

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
# 3. ACTIVE-MEMBER COUNT PER TIMESTEP & DYNAMIC MEAN
# ==========================================================
MIN_ACTIVE_N = max(10, int(0.3 * len(tracks)))  

active_counts = df.groupby("valid_time").size()
dynamic_mean = df.groupby("valid_time")[["latitude", "longitude"]].mean().reset_index()
dynamic_mean["active_n"] = dynamic_mean["valid_time"].map(active_counts)
dynamic_mean["confident"] = dynamic_mean["active_n"] >= MIN_ACTIVE_N

actual_df = df.drop_duplicates(subset=["valid_time"]).sort_values("valid_time")
actual_lats = actual_df["actual_latitude"].values
actual_lons = actual_df["actual_longitude"].values
actual_times = actual_df["valid_time"].values

# ==========================================================
# 4. FULL-LENGTH SEQUENCES PER MEMBER
# ==========================================================
member_sequences = {}
for tid in tracks:
    t_df = df[df["unique_track_id"] == tid].sort_values("valid_time")
    member_sequences[tid] = t_df[["latitude", "longitude"]].to_numpy()

eligible_tracks = list(member_sequences.keys())
n_eligible = len(eligible_tracks)

# ==========================================================
# 5. DTW DISTANCE MATRIX
# ==========================================================
print(f"\nComputing DTW distance matrix for {n_eligible} eligible tracks...")
dtw_matrix = np.zeros((n_eligible, n_eligible))
for i in range(n_eligible):
    for j in range(i + 1, n_eligible):
        d = dtw_distance(member_sequences[eligible_tracks[i]], member_sequences[eligible_tracks[j]])
        dtw_matrix[i, j] = d
        dtw_matrix[j, i] = d

# ==========================================================
# 6. TRAJECTORY-LEVEL MEDOID
# ==========================================================
medoid_row = np.argmin(dtw_matrix.sum(axis=1))
trajectory_medoid_id = eligible_tracks[medoid_row]

# ==========================================================
# 7. SCENARIO CLUSTERING & DBA CONSENSUS
# ==========================================================
N_CLUSTERS = 2
cluster_model = AgglomerativeClustering(n_clusters=N_CLUSTERS, metric="precomputed", linkage="average")
cluster_labels = cluster_model.fit_predict(dtw_matrix)

cluster_sizes = pd.Series(cluster_labels).value_counts()
dominant_cluster = cluster_sizes.idxmax()
dominant_medoid_id = eligible_tracks[np.where(cluster_labels == dominant_cluster)[0][np.argmin(dtw_matrix[np.ix_(np.where(cluster_labels == dominant_cluster)[0], np.where(cluster_labels == dominant_cluster)[0])].sum(axis=1))]]
dominant_cluster_ids = [eligible_tracks[i] for i in np.where(cluster_labels == dominant_cluster)[0]]

DBA_ITERATIONS = 8

def dba_update(center, sequences):
    T = len(center)
    assoc = [[] for _ in range(T)]
    for seq in sequences:
        path = dtw_path(center, seq)
        for (ci, si) in path:
            assoc[ci].append(seq[si])
    return np.array([np.mean(pts, axis=0) if pts else center[t] for t, pts in enumerate(assoc)])

dominant_sequences = [member_sequences[tid] for tid in dominant_cluster_ids]
dba_center = member_sequences[dominant_medoid_id].copy() 

print(f"\nRunning DBA ({DBA_ITERATIONS} iterations) over {len(dominant_sequences)} dominant-cluster members...")
for it in range(DBA_ITERATIONS):
    new_center = dba_update(dba_center, dominant_sequences)
    shift = np.mean(np.linalg.norm(new_center - dba_center, axis=1))
    dba_center = new_center
    if shift < 1e-4: 
        print(f"  Converged after {it + 1} iterations (avg shift {shift:.5f}).")
        break
dba_track = dba_center 

# ==========================================================
# 8. FIND THE "ORACLE" BEST TRACK & EARLY-OBS MATCH
# ==========================================================
actual_seq = np.stack([actual_lats, actual_lons], axis=1)

oracle_distances = {}
for tid in eligible_tracks:
    oracle_distances[tid] = dtw_distance(member_sequences[tid], actual_seq)
oracle_track_id = min(oracle_distances, key=oracle_distances.get)
print(f"\nOracle Best Track (Closest to Ground Truth): ID {oracle_track_id}")

early_steps = min(4, len(actual_lats))
early_actual = np.concatenate([actual_lats[:early_steps], early_actual[:early_steps] if 'early_actual' in locals() else actual_lons[:early_steps]]) # safely handled below
early_obs_match_id = None
best_mse = np.inf
for tid in eligible_tracks:
    seq = member_sequences[tid]
    if len(seq) < early_steps: continue
    seq_early = np.concatenate([seq[:early_steps, 0], seq[:early_steps, 1]])
    mse = np.mean((seq_early - early_actual) ** 2)
    if mse < best_mse:
        best_mse = mse
        early_obs_match_id = tid

# ==========================================================
# 8b. ROBUST OUTLIER-RESILIENT GMM MODAL TRACKING
# ==========================================================
print("\nCalculating Robust Core-Filtered GMM Modal Track...")
gmm_track_records = []

for vt in valid_times_sorted:
    t_df = df[df["valid_time"] == vt]
    n_active = len(t_df)
    
    if n_active < MIN_ACTIVE_N:
        continue  
        
    X = t_df[["latitude", "longitude"]].to_numpy()
    
    # --- ROBUSTNESS STEP: REMOVE SPATIAL OUTLIERS ---
    # Calculate median position and Euclidean distance of each track from the median
    median_lat = np.median(X[:, 0])
    median_lon = np.median(X[:, 1])
    distances = np.sqrt((X[:, 0] - median_lat)**2 + (X[:, 1] - median_lon)**2)
    
    # Drop points that are extreme spatial outliers (beyond 1.5 * Interquartile Range)
    q75, q25 = np.percentile(distances, [75, 25])
    iqr = q75 - q25
    outlier_threshold = q75 + 1.5 * iqr
    
    clean_mask = distances <= outlier_threshold
    X_clean = X[clean_mask]
    
    # Fallback if too many points got filtered
    if len(X_clean) < 5:
        X_clean = X 

    # Fit GMM strictly on the outlier-free core data
    n_components = min(2, max(1, len(X_clean) // 15))
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    gmm.fit(X_clean)
    
    dominant_mode_idx = np.argmax(gmm.weights_)
    mode_lat, mode_lon = gmm.means_[dominant_mode_idx]
    
    gmm_track_records.append({
        "valid_time": vt,
        "gmm_lat": mode_lat,
        "gmm_lon": mode_lon
    })

gmm_track = pd.DataFrame(gmm_track_records)
gmm_seq = gmm_track[["gmm_lat", "gmm_lon"]].to_numpy()
print(f"Robust GMM Modal Track generated with {len(gmm_track)} valid timesteps.")
# ==========================================================
# ROBUST WHOLE-TRACK DENSITY SCORING (KDE-MEDOID)
# ==========================================================
print("\nCalculating Density-Weighted Best Track...")

# 1. Calculate spatial median position at each timestep to define the "core"
core_df = df.groupby("valid_time")[["latitude", "longitude"]].median().reset_index()
core_df = core_df.rename(columns={"latitude": "core_lat", "longitude": "core_lon"})

# 2. Merge every member's points with the timestep's core position
df_density = pd.merge(df, core_df, on="valid_time", how="left")

# 3. Calculate the spatial distance of each member from the ensemble core at every timestep
df_density["core_distance"] = np.sqrt(
    (df_density["latitude"] - df_density["core_lat"])**2 + 
    (df_density["longitude"] - df_density["core_lon"])**2
)

# 4. Score each track by its average distance to the core (lower distance = hugs the high-density center)
track_density_scores = df_density.groupby("unique_track_id")["core_distance"].agg(["mean", "count"])

# Filter out tracks that didn't survive long enough
valid_density_tracks = track_density_scores[track_density_scores["count"] >= MIN_TRACK_STEPS]

# 5. Select the single track that stays closest to the density core overall
best_density_track_id = valid_density_tracks["mean"].idxmax() if valid_density_tracks.empty else valid_density_tracks["mean"].idxmin()

print(f"Density-Core Best Track ID: {best_density_track_id}")

# Extract the sequence for plotting and evaluation
density_track_df = df[df["unique_track_id"] == best_density_track_id].sort_values("valid_time")
density_seq = density_track_df[["latitude", "longitude"]].to_numpy()
# ==========================================================
# 9. 3D SPACE-TIME CUBE PLOTTING (Z-AXIS = VALID TIME)
# ==========================================================
fig = plt.figure(figsize=(14, 10), dpi=150)
ax = fig.add_subplot(projection='3d')

# Convert datetime to matplotlib date numbers for clean 3D Z-axis mapping
actual_z = mdates.date2num(actual_times)

# 1. Background Spaghetti Tracks in 3D
for track_id in all_tracks:
    t_df = df_raw[df_raw["unique_track_id"] == track_id].sort_values("valid_time")
    z_vals = mdates.date2num(t_df["valid_time"])
    ax.plot3D(t_df["longitude"], t_df["latitude"], z_vals, color="gray", linewidth=0.5, alpha=0.15)

# 2. Actual Ground Truth in 3D
ax.plot3D(actual_lons, actual_lats, actual_z, color="black", marker="X", markersize=6,
          linewidth=3, label="Actual Track (Ground Truth)", zorder=12)

# 3. Dynamic Mean in 3D
mean_times = mdates.date2num(dynamic_mean["valid_time"].values)
ax.plot3D(dynamic_mean["longitude"], dynamic_mean["latitude"], mean_times, color="crimson", linestyle="--", linewidth=2.5,
          label="Dynamic Mean", zorder=6)

# 4. Selected Tracks (Medoids, Oracle, Early Match) in 3D
selected = {
    "Trajectory Medoid": (trajectory_medoid_id, "#1f77b4", "o"),
    "Dominant-Cluster Medoid": (dominant_medoid_id, "#2ca02c", "s"),
    "Oracle Best Track": (oracle_track_id, "#ffb400", "*"),
    "Density-Core Best Track": (best_density_track_id, "#e377c2", "v"),
}
if early_obs_match_id is not None:
    selected["Early Obs Match"] = (early_obs_match_id, "#9467bd", "D")

for i, (label, (tid, color, marker)) in enumerate(selected.items()):
    t_df = df[df["unique_track_id"] == tid].sort_values("valid_time")
    z_vals = mdates.date2num(t_df["valid_time"])
    lw = 3.5 if "Oracle" in label else 2.5
    mz = 8 if "Oracle" in label else 4
    ax.plot3D(t_df["longitude"], t_df["latitude"], z_vals, color=color, linewidth=lw,
              marker=marker, markersize=mz, label=f"{label} (ID {tid})", zorder=9 + i)

# 5. DBA Consensus Track in 3D
medoid_df = df[df["unique_track_id"] == dominant_medoid_id].sort_values("valid_time")
dba_z = mdates.date2num(medoid_df["valid_time"].values)
ax.plot3D(dba_track[:, 1], dba_track[:, 0], dba_z, color="#d62728", linestyle="-", 
          linewidth=2.5, marker="*", markersize=6, label="DBA Consensus", zorder=16)

# 6. GMM Modal Track in 3D
gmm_z = mdates.date2num(gmm_track["valid_time"].values)
ax.plot3D(gmm_track["gmm_lon"], gmm_track["gmm_lat"], gmm_z, color="#9467bd", linestyle="-", 
          linewidth=2.5, marker="P", markersize=6, label="GMM Modal Track", zorder=17)

# Formatting 3D Axes
ax.set_title("3D Space-Time Cube: Cyclone Trajectories Over Valid Time", fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("Longitude (°)", fontsize=11, labelpad=10)
ax.set_ylabel("Latitude (°)", fontsize=11, labelpad=10)
ax.set_zlabel("Valid Time", fontsize=11, labelpad=10)

# Format Z-axis ticks to display readable dates/times instead of raw float datenums
ax.zaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)

ax.legend(loc="upper left", fontsize=8, framealpha=0.9, edgecolor="black")
plt.tight_layout()

output_plot = "trajectory_3d_spacetime.png"
plt.savefig(output_plot, dpi=300, bbox_inches="tight")
plt.show()
print(f"\n3D Plot saved to: {output_plot}")

# ==========================================================
# 10. VERIFICATION & R² SCORES
# ==========================================================
print("\n=== DISTANCE TO ACTUAL TRACK (Length-Normalized DTW) ===")
verify_candidates = {label: member_sequences[tid] for label, (tid, _, _) in selected.items() if "Early" not in label}
verify_candidates["DBA Consensus"] = dba_track
verify_candidates["GMM Modal Track"] = gmm_seq

for label, seq in verify_candidates.items():
    d = dtw_distance(seq, actual_seq)
    print(f"- {label}: {d:.4f}")

real_track = actual_df[["valid_time", "actual_latitude", "actual_longitude"]].dropna()

mean_track = dynamic_mean[["valid_time", "latitude", "longitude"]].rename(columns={"latitude": "mean_lat", "longitude": "mean_lon"})
merged_mean = pd.merge(real_track, mean_track, on="valid_time", how="inner")

dba_df = pd.DataFrame({
    "valid_time": medoid_df["valid_time"].values,
    "dba_lat": dba_track[:, 0],
    "dba_lon": dba_track[:, 1]
})
merged_dba = pd.merge(real_track, dba_df, on="valid_time", how="inner")
merged_gmm = pd.merge(real_track, gmm_track, on="valid_time", how="inner")

def calc_r2(merged_df, lat_col, lon_col):
    if merged_df.empty: return np.nan, np.nan
    r2_lat = r2_score(merged_df["actual_latitude"], merged_df[lat_col])
    r2_lon = r2_score(merged_df["actual_longitude"], merged_df[lon_col])
    return r2_lat, r2_lon

r2_mean_lat, r2_mean_lon = calc_r2(merged_mean, "mean_lat", "mean_lon")
r2_dba_lat, r2_dba_lon = calc_r2(merged_dba, "dba_lat", "dba_lon")
r2_gmm_lat, r2_gmm_lon = calc_r2(merged_gmm, "gmm_lat", "gmm_lon")

print("\n=== R² SCORES (Time-Aligned) ===")
print(f"- Dynamic Mean    -> R² Lat: {r2_mean_lat:.4f} | R² Lon: {r2_mean_lon:.4f}")
print(f"- DBA Consensus   -> R² Lat: {r2_dba_lat:.4f} | R² Lon: {r2_dba_lon:.4f}")
print(f"- GMM Modal Track -> R² Lat: {r2_gmm_lat:.4f} | R² Lon: {r2_gmm_lon:.4f}")

print("\n=== SPATIAL ERRORS (KM) ===")
def print_spatial_metrics(name, merged_df, pred_lat_col, pred_lon_col):
    if merged_df.empty: return
    distances = calc_distances_km(
        merged_df["actual_latitude"], merged_df["actual_longitude"],
        merged_df[pred_lat_col], merged_df[pred_lon_col]
    )
    print(f"- {name:15}: Mean Error = {np.mean(distances):.1f} km | RMSE = {np.sqrt(np.mean(distances**2)):.1f} km | Max = {np.max(distances):.1f} km")

print_spatial_metrics("Dynamic Mean", merged_mean, "mean_lat", "mean_lon")
print_spatial_metrics("DBA Consensus", merged_dba, "dba_lat", "dba_lon")
print_spatial_metrics("GMM Modal", merged_gmm, "gmm_lat", "gmm_lon")