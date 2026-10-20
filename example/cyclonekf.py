import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics import r2_score
from sklearn.cluster import AgglomerativeClustering

# ==========================================================
# 0. PHYSICS-INFORMED KALMAN FILTER CLASS
# ==========================================================
class PhysicsInformedKF:
    def __init__(self, dt_hours=6.0, steering_pull=0.3):
        self.dt_hours = dt_hours
        self.dt_seconds = dt_hours * 3600.0
        self.meters_per_deg = 111000.0
        self.steering_pull = steering_pull
        
        self.beta_meridional_ms = 1.0  
        self.beta_zonal_ms = -0.5      
        
        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        self.Q = np.eye(4) * 0.5  # Default Process Noise

    def _get_beta_drift(self, lat):
        lat_direction = 1.0 if lat >= 0 else -1.0
        beta_lat_deg = (self.beta_meridional_ms * lat_direction * self.dt_seconds) / self.meters_per_deg
        cos_lat = max(np.cos(np.radians(lat)), 0.1)
        beta_lon_deg = (self.beta_zonal_ms * self.dt_seconds) / (self.meters_per_deg * cos_lat)
        return beta_lat_deg, beta_lon_deg

    def run_kf_smoother(self, ensemble_lats, ensemble_lons, u_wind_data, v_wind_data, confidence_matrix=None, thickness_data=None):
        n_timesteps = ensemble_lats.shape[0]
        x_estimates, P_estimates = np.zeros((n_timesteps, 4)), np.zeros((n_timesteps, 4, 4))
        x_predictions, P_predictions = np.zeros((n_timesteps, 4)), np.zeros((n_timesteps, 4, 4))
        
        # 1. SMART VELOCITY INITIALIZATION (Fixes cold-start lag)
        init_lat = np.nanmean(ensemble_lats[0])
        init_lon = np.nanmean(ensemble_lons[0])
        
        if n_timesteps > 1:
            init_lat_2 = np.nanmean(ensemble_lats[1])
            init_lon_2 = np.nanmean(ensemble_lons[1])
            init_v_lat = init_lat_2 - init_lat
            init_v_lon = (init_lon_2 - init_lon + 180) % 360 - 180
        else:
            init_v_lat, init_v_lon = 0.0, 0.0

        x = np.array([init_lat, init_lon, init_v_lat, init_v_lon])
        P = np.eye(4) * 0.5  
        
        # --- FORWARD PASS ---
        for t in range(n_timesteps):
            
            # 2. CONFIDENCE & MEASUREMENT (Z) AND NOISE (R)
            if confidence_matrix is not None:
                w = np.exp(confidence_matrix[t])
                valid = ~np.isnan(ensemble_lats[t]) & ~np.isnan(w)
                
                if valid.any():
                    z_lat = np.average(ensemble_lats[t][valid], weights=w[valid])
                    z_lon = np.average(ensemble_lons[t][valid], weights=w[valid])
                    
                    raw_var_lat = np.average((ensemble_lats[t][valid] - z_lat)**2, weights=w[valid])
                    raw_var_lon = np.average((ensemble_lons[t][valid] - z_lon)**2, weights=w[valid])
                    
                    n_valid = valid.sum()
                    mean_conf = np.mean(w[valid]) if len(w[valid]) > 0 else 0.5
                    confidence_scaling = 1.0 / max(mean_conf, 0.1)
                    
                    var_lat = ((raw_var_lat / n_valid) * confidence_scaling) + 0.0001 if n_valid > 1 else 0.1
                    var_lon = ((raw_var_lon / n_valid) * confidence_scaling) + 0.0001 if n_valid > 1 else 0.1
                else:
                    z_lat, z_lon = np.nan, np.nan
                    var_lat, var_lon = 1.0, 1.0
            else:
                z_lat, z_lon = np.nanmean(ensemble_lats[t]), np.nanmean(ensemble_lons[t])
                var_lat = np.nanvar(ensemble_lats[t]) + 0.001
                var_lon = np.nanvar(ensemble_lons[t]) + 0.001

            # 3. PHYSICS PROCESS MODEL (Prediction with Dynamic Q)
            if t > 0:
                u_ms, v_ms = u_wind_data[t], v_wind_data[t]
                cos_lat = max(np.cos(np.radians(x[0])), 0.1)
                steer_lat_deg = (v_ms * self.dt_seconds) / self.meters_per_deg
                steer_lon_deg = (u_ms * self.dt_seconds) / (self.meters_per_deg * cos_lat)
                beta_lat_deg, beta_lon_deg = self._get_beta_drift(x[0])
                
                x_pred = self.F @ x
                x_pred[2] = (1 - self.steering_pull) * x[2] + (self.steering_pull * steer_lat_deg) + beta_lat_deg
                x_pred[3] = (1 - self.steering_pull) * x[3] + (self.steering_pull * steer_lon_deg) + beta_lon_deg
                
                # --- DYNAMIC PROCESS NOISE (Q) USING THICKNESS ---
                if thickness_data is not None and not np.isnan(thickness_data[t]):
                    thick_val = thickness_data[t]
                    # Higher geopotential thickness indicates a stronger thermal core -> lower process noise (higher stability)
                    q_scale = max(0.1, 1.0 - min(0.9, thick_val / 10000.0))
                    self.Q = np.eye(4) * (0.05 * q_scale)
                else:
                    self.Q = np.eye(4) * 0.05 
                # ------------------------------------------------
                
                P_pred = self.F @ P @ self.F.T + self.Q
            else:
                x_pred, P_pred = x.copy(), P.copy()
            
            x_predictions[t], P_predictions[t] = x_pred, P_pred
            
            # 4. MEASUREMENT UPDATE
            if not np.isnan(z_lat):
                z = np.array([z_lat, z_lon])
                R = np.array([[var_lat, 0], [0, var_lon]])
                
                S = self.H @ P_pred @ self.H.T + R
                K = P_pred @ self.H.T @ np.linalg.inv(S)
                y = z - (self.H @ x_pred)
                x = x_pred + K @ y
                
                P = (np.eye(4) - K @ self.H) @ P_pred 
            else:
                x, P = x_pred, P_pred 
                
            x_estimates[t], P_estimates[t] = x, P

        # --- BACKWARD PASS (RTS Smoother) ---
        x_smooth, P_smooth = np.zeros_like(x_estimates), np.zeros_like(P_estimates)
        x_smooth[-1], P_smooth[-1] = x_estimates[-1], P_estimates[-1]
        
        for t in range(n_timesteps - 2, -1, -1):
            P_inv = np.linalg.inv(P_predictions[t + 1])
            C = P_estimates[t] @ self.F.T @ P_inv
            x_smooth[t] = x_estimates[t] + C @ (x_smooth[t + 1] - x_predictions[t + 1])
            P_smooth[t] = P_estimates[t] + C @ (P_smooth[t + 1] - P_predictions[t + 1]) @ C.T

        return x_smooth[:, 0], x_smooth[:, 1]

# ==========================================================
# 1. LOAD AND PREPARE DATASET
# ==========================================================
target_case_id = 255
csv_path = f"example/csv/tc_track_event_{target_case_id}_tracks.csv"
df_raw = pd.read_csv(csv_path)
df_raw["valid_time"] = pd.to_datetime(df_raw["valid_time"])

first_run = sorted(df_raw["init_time"].unique())[0]
df_raw = df_raw[df_raw["init_time"] == first_run].copy()
df_raw["unique_track_id"] = df_raw.groupby(["init_time", "ensemble"]).ngroup() + 1
all_tracks = sorted(df_raw["unique_track_id"].unique())

# ==========================================================
# 2. FILTERING
# ==========================================================
MIN_TRACK_STEPS = 3  
track_lengths = df_raw.groupby("unique_track_id").size()
long_lived_tracks = track_lengths[track_lengths >= MIN_TRACK_STEPS].index.tolist()
df = df_raw[df_raw["unique_track_id"].isin(long_lived_tracks)].copy()
tracks = sorted(df["unique_track_id"].unique())

# ==========================================================
# 3. ACTIVE-MEMBER COUNT PER TIMESTEP & DYNAMIC MEAN
# ==========================================================
MIN_ACTIVE_N = 2
active_counts = df.groupby("valid_time").size()
dynamic_mean = df.groupby("valid_time")[["latitude", "longitude"]].mean().reset_index()
dynamic_mean["active_n"] = dynamic_mean["valid_time"].map(active_counts)
dynamic_mean["confident"] = dynamic_mean["active_n"] >= MIN_ACTIVE_N

actual_df = df.drop_duplicates(subset=["valid_time"]).sort_values("valid_time")
actual_lats = actual_df["actual_latitude"].values
actual_lons = actual_df["actual_longitude"].values

# ==========================================================
# 4. RUN PHYSICS-INFORMED KALMAN FILTER
# ==========================================================
print("\nRunning Physics-Informed Kalman Filter & RTS Smoother...")

pivot_lat = df.pivot(index="valid_time", columns="unique_track_id", values="latitude")
pivot_lon = df.pivot(index="valid_time", columns="unique_track_id", values="longitude")

kf_valid_times = pivot_lat.index.values
ensemble_lats_matrix = pivot_lat.to_numpy()
ensemble_lons_matrix = pivot_lon.to_numpy()

dt_hours_calc = (kf_valid_times[1] - kf_valid_times[0]) / np.timedelta64(1, 'h')

wind_means = df.groupby("valid_time")[["u_wind", "v_wind"]].mean().loc[kf_valid_times]
real_u_wind = wind_means["u_wind"].to_numpy()
real_v_wind = wind_means["v_wind"].to_numpy()

# Extract mean geopotential thickness per timestep aligned with kf_valid_times
if "geopotential_thickness" in df.columns:
    thickness_means = df.groupby("valid_time")["geopotential_thickness"].mean().loc[kf_valid_times]
    real_thickness = thickness_means.to_numpy()
else:
    real_thickness = None

# ==========================================================
# CREATE CONFIDENCE COLUMN (Updated with Geopotential Thickness)
# ==========================================================
slp_min, slp_max = df["air_pressure_at_mean_sea_level"].min(), df["air_pressure_at_mean_sea_level"].max()
slp_norm = (slp_max - df["air_pressure_at_mean_sea_level"]) / (slp_max - slp_min) if slp_max > slp_min else 1.0

wind_min, wind_max = df["surface_wind_speed"].min(), df["surface_wind_speed"].max()
wind_norm = (df["surface_wind_speed"] - wind_min) / (wind_max - wind_min) if wind_max > wind_min else 1.0

if "geopotential_thickness" in df.columns:
    thick_min, thick_max = df["geopotential_thickness"].min(), df["geopotential_thickness"].max()
    thick_norm = (df["geopotential_thickness"] - thick_min) / (thick_max - thick_min) if thick_max > thick_min else 1.0
    df["confidence"] = (slp_norm + wind_norm + thick_norm) / 3.0
else:
    df["confidence"] = (slp_norm + wind_norm) / 2.0

pivot_conf = df.pivot(index="valid_time", columns="unique_track_id", values="confidence")
confidence_matrix = pivot_conf.to_numpy()

kf = PhysicsInformedKF(dt_hours=dt_hours_calc, steering_pull=0.1)
kf_lats, kf_lons = kf.run_kf_smoother(
    ensemble_lats_matrix, 
    ensemble_lons_matrix, 
    real_u_wind, 
    real_v_wind,
    confidence_matrix=confidence_matrix,
    thickness_data=real_thickness
)

kf_track_df = pd.DataFrame({"valid_time": kf_valid_times, "kf_lat": kf_lats, "kf_lon": kf_lons})

# ==========================================================
# 5. PLOTTING THE RESULTS
# ==========================================================
plt.figure(figsize=(16, 10), dpi=150)
ax = plt.gca()

for track_id in all_tracks:
    t_df = df_raw[df_raw["unique_track_id"] == track_id].sort_values("valid_time")
    plt.plot(t_df["longitude"], t_df["latitude"], color="gray", linewidth=0.6, alpha=0.2)

plt.plot(actual_lons, actual_lats, color="black", marker="X", markersize=8,
          linewidth=3, label="Actual Track (Ground Truth)", zorder=12)

conf = dynamic_mean["confident"].values
lons_m, lats_m = dynamic_mean["longitude"].values, dynamic_mean["latitude"].values
plt.plot(lons_m[conf], lats_m[conf], color="crimson", linestyle="--", linewidth=2.5,
          label=f"Dynamic Mean (Active >= {MIN_ACTIVE_N})", zorder=10)

plt.plot(kf_lons, kf_lats, color="cyan", linewidth=3.5, linestyle="-", 
         label="Physics-Informed KF & Smoother Track", zorder=13)

plt.title("Physics-Informed Kalman Filter vs Dynamic Mean & Ground Truth", fontsize=15, fontweight="bold", pad=15)
plt.xlabel("Longitude (°)", fontsize=12)
plt.ylabel("Latitude (°)", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.5)

handles, labels_ = ax.get_legend_handles_labels()
plt.legend(handles, labels_, loc="upper left", fontsize=10, framealpha=0.9, edgecolor="black")
plt.tight_layout()
os.makedirs("example/figs", exist_ok=True)
plt.savefig(f"example/figs/trajectory_kf_vs_mean_{target_case_id}.png", dpi=300, bbox_inches="tight")
plt.show()

# ==========================================================
# 6. CALCULATE R2 SCORE: MEAN & KF vs ACTUAL
# ==========================================================
real_track = actual_df[["valid_time", "actual_latitude", "actual_longitude"]].dropna()

mean_track = dynamic_mean[["valid_time", "latitude", "longitude"]].rename(columns={"latitude": "mean_lat", "longitude": "mean_lon"})
merged_mean = pd.merge(real_track, mean_track, on="valid_time", how="inner")
merged_kf = pd.merge(real_track, kf_track_df, on="valid_time", how="inner")

print("\n" + "="*60)
print("PERFORMANCE METRICS (R² SCORE) vs ACTUAL TRACK")
print("="*60)

if not merged_mean.empty:
    r2_lat_mean = r2_score(merged_mean["actual_latitude"], merged_mean["mean_lat"])
    r2_lon_mean = r2_score(merged_mean["actual_longitude"], merged_mean["mean_lon"])
    print(f"1. Dynamic Mean Path")
    print(f"   - Aligned Timesteps : {len(merged_mean)}")
    print(f"   - R² Score (Lat)    : {r2_lat_mean:.4f}")
    print(f"   - R² Score (Lon)    : {r2_lon_mean:.4f}")

print("-" * 60)

if not merged_kf.empty:
    r2_lat_kf = r2_score(merged_kf["actual_latitude"], merged_kf["kf_lat"])
    r2_lon_kf = r2_score(merged_kf["actual_longitude"], merged_kf["kf_lon"])
    print(f"2. Physics-Informed KF Path")
    print(f"   - Aligned Timesteps : {len(merged_kf)}")
    print(f"   - R² Score (Lat)    : {r2_lat_kf:.4f}")
    print(f"   - R² Score (Lon)    : {r2_lon_kf:.4f}")
    
print("="*60 + "\n")