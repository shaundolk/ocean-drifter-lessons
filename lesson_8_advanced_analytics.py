 
#!/usr/bin/env python3
"""
LESSON 8: ADVANCED ANALYSIS TECHNIQUES
=====================================

Learning Goals:
- Apply advanced statistical methods to oceanographic data
- Learn time series analysis and spectral methods
- Practice machine learning techniques for pattern recognition
- Understand uncertainty quantification and error analysis
- Explore comparative analysis with multiple datasets

What you'll learn about advanced data science:
- How to identify periodic patterns in time series data
- Principal Component Analysis (PCA) for dimension reduction
- Clustering techniques for finding similar patterns
- Bootstrap methods for uncertainty estimation
- Cross-correlation analysis for comparing datasets

This lesson bridges the gap between basic analysis and research-level
data science techniques used by professional oceanographers!

Prerequisites: Complete Lessons 1-7 first!
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy import stats, signal
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def create_multi_drifter_dataset():
    """
    Create a dataset with multiple drifters for comparative analysis.
    
    This simulates having multiple drifters in different regions,
    like real oceanographic studies!
    """
    print("🌊 Creating multi-drifter dataset for advanced analysis...")
    
    # Create 3 different drifters with distinct characteristics
    drifter_configs = [
        {'id': 11111, 'region': 'Gulf Stream', 'start_lat': 35.0, 'start_lon': -75.0, 
         'temp_base': 20, 'current_strength': 'strong', 'color': 'red'},
        {'id': 22222, 'region': 'Sargasso Sea', 'start_lat': 28.0, 'start_lon': -65.0, 
         'temp_base': 25, 'current_strength': 'moderate', 'color': 'blue'},
        {'id': 33333, 'region': 'Labrador Current', 'start_lat': 45.0, 'start_lon': -50.0, 
         'temp_base': 8, 'current_strength': 'variable', 'color': 'green'}
    ]
    
    all_data = []
    
    for config in drifter_configs:
        print(f"   📍 Creating drifter {config['id']} in {config['region']}...")
        
        # Create 4 months of data for each drifter
        hours = 120 * 24  # 4 months
        times = pd.date_range('2023-01-01', periods=hours, freq='H')
        
        # Initialize position
        lats = [config['start_lat']]
        lons = [config['start_lon']]
        
        # Generate trajectory based on region characteristics
        for i in range(1, hours):
            current_lat = lats[-1]
            current_lon = lons[-1]
            
            # Different movement patterns by region
            if config['current_strength'] == 'strong':
                # Gulf Stream: fast northeast flow with high variability
                east_vel = 60 + 40 * np.sin(2 * np.pi * i / (15 * 24)) + np.random.normal(0, 20)
                north_vel = 40 + 20 * np.cos(2 * np.pi * i / (10 * 24)) + np.random.normal(0, 15)
            elif config['current_strength'] == 'moderate':
                # Sargasso: slow anticyclonic circulation
                east_vel = 10 + 15 * np.sin(2 * np.pi * i / (30 * 24)) + np.random.normal(0, 8)
                north_vel = -5 + 20 * np.cos(2 * np.pi * i / (25 * 24)) + np.random.normal(0, 10)
            else:  # variable
                # Labrador: strong seasonal variation
                seasonal_factor = 1 + 0.5 * np.sin(2 * np.pi * i / (365 * 24))
                east_vel = 25 * seasonal_factor * np.sin(2 * np.pi * i / (20 * 24)) + np.random.normal(0, 15)
                north_vel = -15 * seasonal_factor + np.random.normal(0, 12)
            
            # Convert to position changes
            dt = 1  # hour
            dlat = (north_vel * dt * 3600 / 100000) / 111.0
            dlon = (east_vel * dt * 3600 / 100000) / (111.0 * np.cos(np.radians(current_lat)))
            
            lats.append(current_lat + dlat)
            lons.append(current_lon + dlon)
        
        # Generate temperature data with regional characteristics
        temps = []
        for i in range(hours):
            # Base temperature varies with latitude and region
            base_temp = config['temp_base'] - 0.3 * abs(lats[i] - config['start_lat'])
            
            # Seasonal cycle (different phases for different regions)
            seasonal_phase = {'Gulf Stream': 0, 'Sargasso Sea': np.pi/4, 'Labrador Current': np.pi/2}
            seasonal = 4 * np.sin(2 * np.pi * i / (365 * 24) + seasonal_phase[config['region']])
            
            # Daily cycle
            daily = 1.2 * np.sin(2 * np.pi * i / 24 - np.pi/4)
            
            # Regional variability
            if config['region'] == 'Gulf Stream':
                variability = np.random.normal(0, 1.5)  # High variability
            elif config['region'] == 'Sargasso Sea':
                variability = np.random.normal(0, 0.8)  # Low variability
            else:
                variability = np.random.normal(0, 2.0)  # Very high variability
            
            temps.append(base_temp + seasonal + daily + variability)
        
        # Create DataFrame for this drifter
        drifter_data = pd.DataFrame({
            'time': times,
            'latitude': lats,
            'longitude': lons,
            'sst': temps,
            'drifter_id': config['id'],
            'region': config['region'],
            'color': config['color']
        })
        
        all_data.append(drifter_data)
    
    # Combine all drifters
    combined_data = pd.concat(all_data, ignore_index=True)
    
    print(f"✅ Created dataset with {len(combined_data)} total observations")
    print(f"📊 {len(drifter_configs)} drifters across different regions")
    print(f"🌡️  Temperature range: {combined_data['sst'].min():.1f}°C to {combined_data['sst'].max():.1f}°C")
    
    return combined_data

def spectral_analysis(data, variable='sst'):
    """
    Perform spectral analysis to identify periodic patterns.
    
    This teaches students how to find cycles in time series data!
    """
    print(f"\n🌊 Performing spectral analysis on {variable}...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Spectral Analysis: {variable.upper()} Periodicity', fontsize=16, fontweight='bold')
    
    colors = ['red', 'blue', 'green']
    results = {}
    
    for i, (drifter_id, group) in enumerate(data.groupby('drifter_id')):
        color = colors[i % len(colors)]
        region = group['region'].iloc[0]
        
        # Ensure regular time spacing
        group = group.sort_values('time').reset_index(drop=True)
        time_series = group[variable].dropna()
        
        if len(time_series) < 100:  # Need sufficient data
            continue
        
        # Remove trend (detrend)
        detrended = signal.detrend(time_series)
        
        # Calculate power spectral density
        sampling_freq = 24  # samples per day (hourly data)
        frequencies, power = signal.periodogram(detrended, fs=sampling_freq)
        
        # Convert frequency to periods in days
        periods = 1 / frequencies[1:]  # Skip zero frequency
        power = power[1:]
        
        # Plot 1: Time series (detrended)
        axes[0,0].plot(group['time'], detrended, color=color, alpha=0.7, 
                      linewidth=0.8, label=f'{region}')
        
        # Plot 2: Power spectral density
        axes[0,1].loglog(periods, power, color=color, alpha=0.7, 
                        linewidth=1.5, label=f'{region}')
        
        # Find dominant periods
        # Focus on periods between 1 day and 60 days
        period_mask = (periods >= 1) & (periods <= 60)
        relevant_periods = periods[period_mask]
        relevant_power = power[period_mask]
        
        # Find peaks
        peak_indices = signal.find_peaks(relevant_power, height=np.max(relevant_power) * 0.1)[0]
        dominant_periods = relevant_periods[peak_indices]
        
        results[drifter_id] = {
            'region': region,
            'dominant_periods': dominant_periods,
            'max_power_period': relevant_periods[np.argmax(relevant_power)]
        }
        
        print(f"   📊 {region} (ID {drifter_id}):")
        if len(dominant_periods) > 0:
            print(f"      • Dominant periods: {dominant_periods[:3]:.1f} days")
            print(f"      • Strongest signal: {relevant_periods[np.argmax(relevant_power)]:.1f} days")
        else:
            print(f"      • No clear dominant periods found")
    
    # Formatting
    axes[0,0].set_title('A) Detrended Time Series')
    axes[0,0].set_ylabel(f'{variable.upper()} Anomaly')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].tick_params(axis='x', rotation=45)
    
    axes[0,1].set_title('B) Power Spectral Density')
    axes[0,1].set_xlabel('Period (days)')
    axes[0,1].set_ylabel('Power')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].axvline(x=1, color='gray', linestyle='--', alpha=0.5, label='Daily cycle')
    axes[0,1].axvline(x=365.25, color='gray', linestyle=':', alpha=0.5, label='Annual cycle')
    
    # Plot 3: Wavelet-like analysis (using windowed FFT)
    print("\n🌊 Creating time-frequency analysis...")
    
    # Choose one drifter for detailed time-frequency analysis
    main_drifter = data[data['drifter_id'] == data['drifter_id'].iloc[0]].copy()
    main_drifter = main_drifter.sort_values('time').reset_index(drop=True)
    
    # Window parameters
    window_size = 30 * 24  # 30 days
    overlap = window_size // 2
    
    time_windows = []
    period_windows = []
    power_matrix = []
    
    for start in range(0, len(main_drifter) - window_size, overlap):
        end = start + window_size
        window_data = main_drifter[variable].iloc[start:end].dropna()
        
        if len(window_data) < window_size * 0.8:  # Skip if too much missing data
            continue
        
        # Detrend window
        detrended_window = signal.detrend(window_data)
        
        # Calculate periodogram for this window
        freqs, powers = signal.periodogram(detrended_window, fs=24)
        periods_window = 1 / freqs[1:]
        powers_window = powers[1:]
        
        # Store results
        time_windows.append(main_drifter['time'].iloc[start + window_size//2])
        if len(power_matrix) == 0:
            period_windows = periods_window[(periods_window >= 1) & (periods_window <= 30)]
        
        # Interpolate to common period grid
        power_interp = np.interp(period_windows, periods_window, powers_window)
        power_matrix.append(power_interp)
    
    if len(power_matrix) > 0:
        power_matrix = np.array(power_matrix).T
        
        im = axes[1,0].pcolormesh(time_windows, period_windows, power_matrix, 
                                 shading='auto', cmap='viridis')
        axes[1,0].set_title('C) Time-Frequency Analysis')
        axes[1,0].set_xlabel('Time')
        axes[1,0].set_ylabel('Period (days)')
        axes[1,0].set_yscale('log')
        plt.colorbar(im, ax=axes[1,0], label='Power')
        axes[1,0].tick_params(axis='x', rotation=45)
    
    # Plot 4: Autocorrelation analysis
    print("🔄 Computing autocorrelation functions...")
    
    max_lag_days = 30
    max_lag_hours = max_lag_days * 24
    
    for i, (drifter_id, group) in enumerate(data.groupby('drifter_id')):
        color = colors[i % len(colors)]
        region = group['region'].iloc[0]
        
        time_series = group[variable].dropna()
        if len(time_series) < max_lag_hours * 2:
            continue
        
        # Calculate autocorrelation
        autocorr = []
        lags = range(0, min(max_lag_hours, len(time_series)//2))
        
        for lag in lags:
            if lag == 0:
                autocorr.append(1.0)
            else:
                corr = np.corrcoef(time_series[:-lag], time_series[lag:])[0,1]
                autocorr.append(corr if not np.isnan(corr) else 0)
        
        lag_days = np.array(lags) / 24
        axes[1,1].plot(lag_days, autocorr, color=color, linewidth=2, 
                      label=f'{region}', alpha=0.8)
        
        # Find decorrelation timescale (first zero crossing or e-folding time)
        decorr_time = None
        for j, corr in enumerate(autocorr[1:], 1):
            if corr <= np.exp(-1):  # e-folding time
                decorr_time = lag_days[j]
                break
        
        if decorr_time:
            axes[1,1].axvline(x=decorr_time, color=color, linestyle='--', alpha=0.5)
            print(f"   🕒 {region}: Decorrelation time ≈ {decorr_time:.1f} days")
    
    axes[1,1].set_title('D) Autocorrelation Functions')
    axes[1,1].set_xlabel('Time Lag (days)')
    axes[1,1].set_ylabel('Correlation Coefficient')
    axes[1,1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[1,1].axhline(y=np.exp(-1), color='red', linestyle=':', alpha=0.5, label='e⁻¹')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].set_ylim(-0.5, 1)
    
    plt.tight_layout()
    plt.savefig('spectral_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results

def principal_component_analysis(data):
    """
    Perform PCA to identify dominant patterns of variability.
    
    This teaches students dimension reduction and pattern recognition!
    """
    print("\n🔍 Performing Principal Component Analysis (PCA)...")
    
    # Prepare data matrix
    # Each row is a time point, each column is a different variable/drifter
    
    # Create a matrix with temperature from each drifter
    drifter_ids = data['drifter_id'].unique()
    time_index = pd.date_range(data['time'].min(), data['time'].max(), freq='6H')
    
    # Create temperature matrix (time x drifters)
    temp_matrix = []
    drifter_names = []
    
    for did in drifter_ids:
        drifter_data = data[data['drifter_id'] == did].copy()
        region = drifter_data['region'].iloc[0]
        drifter_names.append(f"{region}\n(ID: {did})")
        
        # Interpolate to common time grid
        temp_series = drifter_data.set_index('time')['sst'].reindex(time_index)
        temp_series = temp_series.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
        temp_matrix.append(temp_series.values)
    
    temp_matrix = np.array(temp_matrix).T  # Transpose so time is rows, drifters are columns
    
    # Remove any remaining NaN values
    valid_mask = ~np.isnan(temp_matrix).any(axis=1)
    temp_matrix = temp_matrix[valid_mask]
    valid_times = time_index[valid_mask]
    
    print(f"   📊 Data matrix shape: {temp_matrix.shape} (time x drifters)")
    
    # Standardize the data (important for PCA)
    scaler = StandardScaler()
    temp_standardized = scaler.fit_transform(temp_matrix)
    
    # Perform PCA
    pca = PCA()
    pca_result = pca.fit_transform(temp_standardized)
    
    # Create comprehensive PCA visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Principal Component Analysis: Temperature Patterns', fontsize=16, fontweight='bold')
    
    # Plot 1: Explained variance
    explained_var = pca.explained_variance_ratio_ * 100
    cumulative_var = np.cumsum(explained_var)
    
    axes[0,0].bar(range(1, len(explained_var)+1), explained_var, alpha=0.7, color='skyblue')
    axes[0,0].plot(range(1, len(cumulative_var)+1), cumulative_var, 'ro-', linewidth=2)
    axes[0,0].set_title('A) Explained Variance by Component')
    axes[0,0].set_xlabel('Principal Component')
    axes[0,0].set_ylabel('Explained Variance (%)')
    axes[0,0].grid(True, alpha=0.3)
    
    # Add text annotations
    for i in range(min(3, len(explained_var))):
        axes[0,0].text(i+1, explained_var[i] + 1, f'{explained_var[i]:.1f}%', 
                      ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: PC loadings (how much each drifter contributes to each PC)
    pc_loadings = pca.components_[:3].T  # First 3 PCs
    
    x = np.arange(len(drifter_names))
    width = 0.25
    
    for i in range(3):
        axes[0,1].bar(x + i*width, pc_loadings[:, i], width, 
                     label=f'PC{i+1} ({explained_var[i]:.1f}%)', alpha=0.7)
    
    axes[0,1].set_title('B) Principal Component Loadings')
    axes[0,1].set_xlabel('Drifter Region')
    axes[0,1].set_ylabel('Loading')
    axes[0,1].set_xticks(x + width)
    axes[0,1].set_xticklabels(drifter_names, rotation=45, ha='right')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Plot 3: PC time series
    for i in range(min(3, pca_result.shape[1])):
        axes[0,2].plot(valid_times, pca_result[:, i], label=f'PC{i+1}', linewidth=1.5, alpha=0.8)
    
    axes[0,2].set_title('C) Principal Component Time Series')
    axes[0,2].set_xlabel('Time')
    axes[0,2].set_ylabel('PC Score')
    axes[0,2].legend()
    axes[0,2].grid(True, alpha=0.3)
    axes[0,2].tick_params(axis='x', rotation=45)
    
    # Plot 4: Biplot (PC1 vs PC2)
    colors = ['red', 'blue', 'green']
    for i, (did, group) in enumerate(data.groupby('drifter_id')):
        region = group['region'].iloc[0]
        axes[1,0].scatter(pc_loadings[i, 0], pc_loadings[i, 1], 
                         s=200, c=colors[i], alpha=0.7, label=region)
        axes[1,0].annotate(region, (pc_loadings[i, 0], pc_loadings[i, 1]), 
                          xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    axes[1,0].set_title('D) PC Biplot (PC1 vs PC2)')
    axes[1,0].set_xlabel(f'PC1 ({explained_var[0]:.1f}%)')
    axes[1,0].set_ylabel(f'PC2 ({explained_var[1]:.1f}%)')
    axes[1,0].axhline(y=0, color='black', linestyle='--', alpha=0.3)
    axes[1,0].axvline(x=0, color='black', linestyle='--', alpha=0.3)
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].legend()
    
    # Plot 5: Reconstruction quality
    # Show how well PC1 alone can reconstruct the original data
    pc1_reconstruction = np.outer(pca_result[:, 0], pca.components_[0])
    pc1_reconstruction = scaler.inverse_transform(pc1_reconstruction)
    
    # Compare original vs reconstructed for first drifter
    original_temp = temp_matrix[:, 0]
    reconstructed_temp = pc1_reconstruction[:, 0]
    
    axes[1,1].plot(valid_times, original_temp, 'b-', linewidth=2, alpha=0.7, label='Original')
    axes[1,1].plot(valid_times, reconstructed_temp, 'r--', linewidth=2, alpha=0.7, label='PC1 Reconstruction')
    
    # Calculate R²
    r_squared = 1 - np.var(original_temp - reconstructed_temp) / np.var(original_temp)
    
    axes[1,1].set_title(f'E) Reconstruction Quality (R² = {r_squared:.3f})')
    axes[1,1].set_xlabel('Time')
    axes[1,1].set_ylabel('Temperature (°C)')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].tick_params(axis='x', rotation=45)
    
    # Plot 6: PC interpretation
    interpretation_text = f"""
PCA INTERPRETATION:

PC1 ({explained_var[0]:.1f}% of variance):
• Represents the dominant pattern
• Likely captures seasonal/regional differences
• High positive loading = warmer than average
• High negative loading = cooler than average

PC2 ({explained_var[1]:.1f}% of variance):
• Secondary pattern of variation
• May capture different seasonal phases
• Or regional circulation differences

PC3 ({explained_var[2]:.1f}% of variance):
• Third-order effects
• Local variations or noise

PRACTICAL MEANING:
• First 2 PCs capture {cumulative_var[1]:.1f}% of variance
• Most temperature variability can be
  explained by just 2 patterns!
• This simplifies complex 3D data into
  understandable 2D patterns
"""
    
    axes[1,2].text(0.05, 0.95, interpretation_text, transform=axes[1,2].transAxes, 
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
    axes[1,2].set_xlim(0, 1)
    axes[1,2].set_ylim(0, 1)
    axes[1,2].axis('off')
    axes[1,2].set_title('F) Interpretation Guide')
    
    plt.tight_layout()
    plt.savefig('pca_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print(f"\n📊 PCA Results Summary:")
    print(f"   • PC1 explains {explained_var[0]:.1f}% of temperature variance")
    print(f"   • PC2 explains {explained_var[1]:.1f}% of temperature variance")
    print(f"   • First 2 PCs together explain {cumulative_var[1]:.1f}% of variance")
    print(f"   • Dominant pattern (PC1) successfully captures major temperature variations")
    
    return pca, temp_matrix, valid_times

def clustering_analysis(data):
    """
    Perform clustering to identify similar oceanographic regimes.
    
    This teaches students unsupervised machine learning for pattern recognition!
    """
    print("\n🔍 Performing clustering analysis to identify oceanographic regimes...")
    
    # Prepare features for clustering
    # Use multiple variables: temperature, location, and time
    
    features_list = []
    labels_list = []
    
    for drifter_id, group in data.groupby('drifter_id'):
        region = group['region'].iloc[0]
        
        # Create features for each observation
        features = pd.DataFrame({
            'temperature': group['sst'],
            'latitude': group['latitude'],
            'longitude': group['longitude'],
            'month': group['time'].dt.month,
            'day_of_year': group['time'].dt.dayofyear,
        })
        
        # Add derived features
        features['temp_gradient'] = features['temperature'].diff()
        features['lat_gradient'] = features['latitude'].diff()
        
        # Remove NaN values
        features = features.dropna()
        
        features_list.append(features)
        labels_list.extend([region] * len(features))
    
    # Combine all features
    all_features = pd.concat(features_list, ignore_index=True)
    all_labels = labels_list
    
    print(f"   📊 Feature matrix shape: {all_features.shape}")
    print(f"   🏷️  Features: {list(all_features.columns)}")
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(all_features)
    
    # Perform K-means clustering with different numbers of clusters
    cluster_range = range(2, 8)
    inertias = []
    silhouette_scores = []
    
    from sklearn.metrics import silhouette_score
    
    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_scaled)
        
        inertias.append(kmeans.inertia_)
        if len(np.unique(cluster_labels)) > 1:  # Need at least 2 clusters for silhouette
            silhouette_avg = silhouette_score(features_scaled, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        else:
            silhouette_scores.append(0)
    
    # Choose optimal number of clusters (elbow method + silhouette)
    # Find elbow point
    def find_elbow(inertias):
        npoints = len(inertias)
        allCoord = np.vstack((range(npoints), inertias)).T
        firstPoint = allCoord[0]
        lineVec = allCoord[-1] - allCoord[0]
        lineVecNorm = lineVec / np.sqrt(np.sum(lineVec**2))
        
        vecFromFirst = allCoord - firstPoint
        scalarProduct = np.sum(vecFromFirst * lineVecNorm, axis=1)
        vecFromFirstParallel = np.outer(scalarProduct, lineVecNorm)
        vecToLine = vecFromFirst - vecFromFirstParallel
        distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))
        
        return np.argmax(distToLine)
    
    elbow_idx = find_elbow(inertias)
    optimal_clusters = cluster_range[elbow_idx]
    
    print(f"   🎯 Optimal number of clusters: {optimal_clusters}")
    
    # Perform final clustering
    final_kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
    final_cluster_labels = final_kmeans.fit_predict(features_scaled)
    
    # Create comprehensive clustering visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Clustering Analysis: Oceanographic Regimes', fontsize=16, fontweight='bold')
    
    # Plot 1: Elbow method
    axes[0,0].plot(cluster_range, inertias, 'bo-', linewidth=2, markersize=8)
    axes[0,0].axvline(x=optimal_clusters, color='red', linestyle='--', alpha=0.7, 
                     label=f'Optimal: {optimal_clusters}')
    axes[0,0].set_title('A) Elbow Method')
    axes[0,0].set_xlabel('Number of Clusters')
    axes[0,0].set_ylabel('Within-Cluster Sum of Squares')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].legend()
    
    # Plot 2: Silhouette scores
    axes[0,1].plot(cluster_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
    axes[0,1].axvline(x=optimal_clusters, color='red', linestyle='--', alpha=0.7)
    axes[0,1].set_title('B) Silhouette Analysis')
    axes[0,1].set_xlabel('Number of Clusters')
    axes[0,1].set_ylabel('Average Silhouette Score')
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Clusters in temperature-latitude space
    scatter = axes[0,2].scatter(all_features['latitude'], all_features['temperature'], 
                               c=final_cluster_labels, cmap='viridis', alpha=0.6, s=10)
    axes[0,2].set_title('C) Clusters in Temperature-Latitude Space')
    axes[0,2].set_xlabel('Latitude (°N)')
    axes[0,2].set_ylabel('Temperature (°C)')
    axes[0,2].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[0,2], label='Cluster')
    
    # Plot 4: Cluster centers in feature space
    cluster_centers = final_kmeans.cluster_centers_
    cluster_centers_original = scaler.inverse_transform(cluster_centers)
    
    # Create heatmap of cluster characteristics
    cluster_df = pd.DataFrame(cluster_centers_original, 
                             columns=all_features.columns,
                             index=[f'Cluster {i}' for i in range(optimal_clusters)])
    
    im = axes[1,0].imshow(cluster_df.values, cmap='RdYlBu_r', aspect='auto')
    axes[1,0].set_xticks(range(len(cluster_df.columns)))
    axes[1,0].set_xticklabels(cluster_df.columns, rotation=45, ha='right')
    axes[1,0].set_yticks(range(len(cluster_df.index)))
    axes[1,0].set_yticklabels(cluster_df.index)
    axes[1,0].set_title('D) Cluster Characteristics')
    plt.colorbar(im, ax=axes[1,0], label='Feature Value')
    
    # Plot 5: Geographic distribution of clusters
    axes[1,1].scatter(all_features['longitude'], all_features['latitude'], 
                     c=final_cluster_labels, cmap='viridis', alpha=0.6, s=10)
    axes[1,1].set_title('E) Geographic Distribution of Clusters')
    axes[1,1].set_xlabel('Longitude (°W)')
    axes[1,1].set_ylabel('Latitude (°N)')
    axes[1,1].grid(True, alpha=0.3)
    
    # Plot 6: Seasonal distribution of clusters
    seasonal_data = pd.DataFrame({
        'month': all_features['month'],
        'cluster': final_cluster_labels
    })
    
    # Create heatmap of cluster occurrence by month
    cluster_month = seasonal_data.groupby(['month', 'cluster']).size().unstack(fill_value=0)
    cluster_month_pct = cluster_month.div(cluster_month.sum(axis=1), axis=0) * 100
    
    im2 = axes[1,2].imshow(cluster_month_pct.T.values, cmap='YlOrRd', aspect='auto')
    axes[1,2].set_xticks(range(len(cluster_month_pct.index)))
    axes[1,2].set_xticklabels(cluster_month_pct.index)
    axes[1,2].set_yticks(range(len(cluster_month_pct.columns)))
    axes[1,2].set_yticklabels([f'Cluster {i}' for i in cluster_month_pct.columns])
    axes[1,2].set_title('F) Seasonal Distribution (%)')
    axes[1,2].set_xlabel('Month')
    plt.colorbar(im2, ax=axes[1,2], label='Percentage')
    
    plt.tight_layout()
    plt.savefig('clustering_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Interpret clusters
    print(f"\n🔍 Cluster Interpretation:")
    for i in range(optimal_clusters):
        cluster_mask = final_cluster_labels == i
        cluster_data = all_features[cluster_mask]
        
        print(f"\n   📊 CLUSTER {i}:")
        print(f"      • Size: {cluster_mask.sum()} observations ({cluster_mask.mean()*100:.1f}%)")
        print(f"      • Avg Temperature: {cluster_data['temperature'].mean():.1f}°C")
        print(f"      • Avg Latitude: {cluster_data['latitude'].mean():.1f}°N")
        print(f"      • Dominant Months: {cluster_data['month'].mode().values}")
        
        # Identify regime characteristics
        if cluster_data['temperature'].mean() > all_features['temperature'].mean() + all_features['temperature'].std():
            regime_type = "Warm Water Regime"
        elif cluster_data['temperature'].mean() < all_features['temperature'].mean() - all_features['temperature'].std():
            regime_type = "Cold Water Regime"
        else:
            regime_type = "Temperate Regime"
        
        print(f"      • Regime Type: {regime_type}")
    
    return final_cluster_labels, final_kmeans, all_features

def uncertainty_analysis(data):
    """
    Perform bootstrap analysis to quantify uncertainty in estimates.
    
    This teaches students about uncertainty quantification!
    """
    print("\n📊 Performing uncertainty analysis using bootstrap methods...")
    
    def bootstrap_statistic(data_sample, statistic_func, n_bootstrap=1000):
        """Calculate bootstrap confidence intervals for a statistic."""
        n = len(data_sample)
        bootstrap_stats = []
        
        np.random.seed(42)  # For reproducibility
        
        for i in range(n_bootstrap):
            # Resample with replacement
            bootstrap_sample = np.random.choice(data_sample, size=n, replace=True)
            stat = statistic_func(bootstrap_sample)
            bootstrap_stats.append(stat)
        
        bootstrap_stats = np.array(bootstrap_stats)
        
        # Calculate confidence intervals
        ci_lower = np.percentile(bootstrap_stats, 2.5)
        ci_upper = np.percentile(bootstrap_stats, 97.5)
        
        return bootstrap_stats, ci_lower, ci_upper
    
    # Analyze different statistics for each drifter
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Uncertainty Analysis: Bootstrap Confidence Intervals', fontsize=16, fontweight='bold')
    
    statistics_results = {}
    
    colors = ['red', 'blue', 'green']
    
    for i, (drifter_id, group) in enumerate(data.groupby('drifter_id')):
        region = group['region'].iloc[0]
        color = colors[i % len(colors)]
        
        temp_data = group['sst'].dropna().values
        
        # Bootstrap different statistics
        
        # 1. Mean temperature
        mean_boots, mean_ci_low, mean_ci_high = bootstrap_statistic(temp_data, np.mean)
        
        # 2. Standard deviation
        std_boots, std_ci_low, std_ci_high = bootstrap_statistic(temp_data, np.std)
        
        # 3. Temperature range
        range_func = lambda x: np.max(x) - np.min(x)
        range_boots, range_ci_low, range_ci_high = bootstrap_statistic(temp_data, range_func)
        
        # 4. 90th percentile
        p90_func = lambda x: np.percentile(x, 90)
        p90_boots, p90_ci_low, p90_ci_high = bootstrap_statistic(temp_data, p90_func)
        
        statistics_results[drifter_id] = {
            'region': region,
            'mean': {'value': np.mean(temp_data), 'ci': (mean_ci_low, mean_ci_high), 'boots': mean_boots},
            'std': {'value': np.std(temp_data), 'ci': (std_ci_low, std_ci_high), 'boots': std_boots},
            'range': {'value': range_func(temp_data), 'ci': (range_ci_low, range_ci_high), 'boots': range_boots},
            'p90': {'value': p90_func(temp_data), 'ci': (p90_ci_low, p90_ci_high), 'boots': p90_boots}
        }
        
        # Plot bootstrap distributions
        
        # Plot 1: Mean temperature bootstrap
        axes[0,0].hist(mean_boots, bins=50, alpha=0.7, color=color, label=region, density=True)
        axes[0,0].axvline(np.mean(temp_data), color=color, linestyle='-', linewidth=2)
        axes[0,0].axvline(mean_ci_low, color=color, linestyle='--', alpha=0.7)
        axes[0,0].axvline(mean_ci_high, color=color, linestyle='--', alpha=0.7)
    
    axes[0,0].set_title('A) Bootstrap Distribution: Mean Temperature')
    axes[0,0].set_xlabel('Mean Temperature (°C)')
    axes[0,0].set_ylabel('Density')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Confidence intervals comparison
    regions = []
    means = []
    ci_lows = []
    ci_highs = []
    
    for drifter_id, results in statistics_results.items():
        regions.append(results['region'])
        means.append(results['mean']['value'])
        ci_lows.append(results['mean']['ci'][0])
        ci_highs.append(results['mean']['ci'][1])
    
    y_pos = np.arange(len(regions))
    
    axes[0,1].errorbar(means, y_pos, 
                      xerr=[np.array(means) - np.array(ci_lows), 
                           np.array(ci_highs) - np.array(means)],
                      fmt='o', markersize=8, capsize=5, capthick=2)
    
    for i, (mean, region) in enumerate(zip(means, regions)):
        axes[0,1].plot(mean, i, 'o', markersize=10, color=colors[i], alpha=0.8)
    
    axes[0,1].set_yticks(y_pos)
    axes[0,1].set_yticklabels(regions)
    axes[0,1].set_title('B) Mean Temperature with 95% CI')
    axes[0,1].set_xlabel('Temperature (°C)')
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Multiple statistics comparison
    stats_names = ['Mean', 'Std Dev', 'Range', '90th %ile']
    x_pos = np.arange(len(stats_names))
    width = 0.25
    
    for i, (drifter_id, results) in enumerate(statistics_results.items()):
        values = [results['mean']['value'], results['std']['value'], 
                 results['range']['value'], results['p90']['value']]
        errors_low = [results['mean']['value'] - results['mean']['ci'][0],
                     results['std']['value'] - results['std']['ci'][0],
                     results['range']['value'] - results['range']['ci'][0],
                     results['p90']['value'] - results['p90']['ci'][0]]
        errors_high = [results['p90']['ci'][1] - results['p90']['value'],
                      results['range']['ci'][1] - results['range']['value'],
                      results['std']['ci'][1] - results['std']['value'],
                      results['mean']['ci'][1] - results['mean']['value']]
        
        axes[1,0].bar(x_pos + i*width, values, width, 
                     yerr=[errors_low, errors_high], 
                     label=results['region'], alpha=0.7, color=colors[i],
                     capsize=3)
    
    axes[1,0].set_title('C) Statistics Comparison with Uncertainty')
    axes[1,0].set_xlabel('Statistic')
    axes[1,0].set_ylabel('Value')
    axes[1,0].set_xticks(x_pos + width)
    axes[1,0].set_xticklabels(stats_names)
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot 4: Uncertainty summary table
    summary_text = "UNCERTAINTY ANALYSIS SUMMARY\n" + "="*40 + "\n\n"
    
    for drifter_id, results in statistics_results.items():
        region = results['region']
        mean_val = results['mean']['value']
        mean_uncertainty = (results['mean']['ci'][1] - results['mean']['ci'][0]) / 2
        
        summary_text += f"{region}:\n"
        summary_text += f"  Mean Temperature: {mean_val:.2f} ± {mean_uncertainty:.2f}°C\n"
        summary_text += f"  95% CI: [{results['mean']['ci'][0]:.2f}, {results['mean']['ci'][1]:.2f}]\n"
        summary_text += f"  Relative Uncertainty: {(mean_uncertainty/mean_val)*100:.1f}%\n\n"
    
    summary_text += "KEY INSIGHTS:\n"
    summary_text += "• Bootstrap provides robust uncertainty estimates\n"
    summary_text += "• Confidence intervals show statistical reliability\n"
    summary_text += "• Larger samples → smaller uncertainty\n"
    summary_text += "• Important for comparing regions objectively"
    
    axes[1,1].text(0.05, 0.95, summary_text, transform=axes[1,1].transAxes, 
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    axes[1,1].set_xlim(0, 1)
    axes[1,1].set_ylim(0, 1)
    axes[1,1].axis('off')
    axes[1,1].set_title('D) Summary Results')
    
    plt.tight_layout()
    plt.savefig('uncertainty_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n📊 Bootstrap Results:")
    for drifter_id, results in statistics_results.items():
        region = results['region']
        mean_val = results['mean']['value']
        mean_ci = results['mean']['ci']
        uncertainty = (mean_ci[1] - mean_ci[0]) / 2
        
        print(f"   🌡️  {region}:")
        print(f"      Mean temperature: {mean_val:.2f} ± {uncertainty:.2f}°C")
        print(f"      95% confidence interval: [{mean_ci[0]:.2f}, {mean_ci[1]:.2f}]°C")
        print(f"      Relative uncertainty: {(uncertainty/mean_val)*100:.1f}%")
    
    return statistics_results

def comparative_analysis(data):
    """
    Compare different regions using advanced statistical tests.
    
    This teaches students hypothesis testing and comparative methods!
    """
    print("\n🔬 Performing comparative statistical analysis...")
    
    # Prepare data for comparison
    regions_data = {}
    for drifter_id, group in data.groupby('drifter_id'):
        region = group['region'].iloc[0]
        regions_data[region] = group['sst'].dropna().values
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Comparative Analysis: Statistical Tests Between Regions', fontsize=16, fontweight='bold')
    
    # Statistical tests
    from scipy import stats as scipy_stats
    
    regions = list(regions_data.keys())
    n_regions = len(regions)
    
    # 1. ANOVA test
    print("   🧪 Performing Analysis of Variance (ANOVA)...")
    
    anova_data = list(regions_data.values())
    f_stat, p_value = scipy_stats.f_oneway(*anova_data)
    
    print(f"      F-statistic: {f_stat:.3f}")
    print(f"      p-value: {p_value:.2e}")
    
    if p_value < 0.05:
        print(f"      Result: Significant differences between regions (p < 0.05)")
    else:
        print(f"      Result: No significant differences between regions (p ≥ 0.05)")
    
    # 2. Post-hoc pairwise comparisons
    print("\n   🔍 Pairwise comparisons (t-tests with Bonferroni correction)...")
    
    pairwise_results = []
    n_comparisons = n_regions * (n_regions - 1) // 2
    alpha_corrected = 0.05 / n_comparisons  # Bonferroni correction
    
    for i in range(n_regions):
        for j in range(i+1, n_regions):
            region1, region2 = regions[i], regions[j]
            data1, data2 = regions_data[region1], regions_data[region2]
            
            # Two-sample t-test
            t_stat, p_val = scipy_stats.ttest_ind(data1, data2)
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(data1) - 1) * np.var(data1, ddof=1) + 
                                 (len(data2) - 1) * np.var(data2, ddof=1)) / 
                                (len(data1) + len(data2) - 2))
            cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std
            
            significant = p_val < alpha_corrected
            
            pairwise_results.append({
                'region1': region1, 'region2': region2,
                't_stat': t_stat, 'p_value': p_val,
                'significant': significant, 'cohens_d': cohens_d
            })
            
            print(f"      {region1} vs {region2}:")
            print(f"         t-statistic: {t_stat:.3f}")
            print(f"         p-value: {p_val:.2e} ({'significant' if significant else 'not significant'})")
            print(f"         Effect size (Cohen's d): {cohens_d:.3f}")
    
    # Plot 1: Box plots with statistical annotations
    box_data = [regions_data[region] for region in regions]
    box_plot = axes[0,0].boxplot(box_data, labels=regions, patch_artist=True)
    
    colors = ['lightcoral', 'lightblue', 'lightgreen']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    axes[0,0].set_title('A) Temperature Distributions by Region')
    axes[0,0].set_ylabel('Temperature (°C)')
    axes[0,0].grid(True, alpha=0.3)
    
    # Add significance annotations
    y_max = max([np.max(data) for data in box_data])
    y_offset = (y_max - min([np.min(data) for data in box_data])) * 0.1
    
    for i, result in enumerate(pairwise_results):
        if result['significant']:
            y_pos = y_max + y_offset * (i + 1)
            region_idx1 = regions.index(result['region1'])
            region_idx2 = regions.index(result['region2'])
            
            axes[0,0].plot([region_idx1 + 1, region_idx2 + 1], [y_pos, y_pos], 'k-', linewidth=1)
            axes[0,0].text((region_idx1 + region_idx2 + 2) / 2, y_pos + y_offset/4, 
                          f"p={result['p_value']:.1e}", ha='center', fontsize=8)
    
    # Plot 2: Effect sizes heatmap
    effect_size_matrix = np.zeros((n_regions, n_regions))
    p_value_matrix = np.ones((n_regions, n_regions))
    
    for result in pairwise_results:
        i = regions.index(result['region1'])
        j = regions.index(result['region2'])
        effect_size_matrix[i, j] = result['cohens_d']
        effect_size_matrix[j, i] = -result['cohens_d']  # Symmetric but with opposite sign
        p_value_matrix[i, j] = result['p_value']
        p_value_matrix[j, i] = result['p_value']
    
    im1 = axes[0,1].imshow(effect_size_matrix, cmap='RdBu_r', vmin=-2, vmax=2)
    axes[0,1].set_xticks(range(n_regions))
    axes[0,1].set_yticks(range(n_regions))
    axes[0,1].set_xticklabels(regions, rotation=45)
    axes[0,1].set_yticklabels(regions)
    axes[0,1].set_title('B) Effect Sizes (Cohen\'s d)')
    
    # Add text annotations
    for i in range(n_regions):
        for j in range(n_regions):
            if i != j:
                text = f'{effect_size_matrix[i, j]:.2f}'
                if p_value_matrix[i, j] < alpha_corrected:
                    text += '*'
                axes[0,1].text(j, i, text, ha='center', va='center', 
                              color='white' if abs(effect_size_matrix[i, j]) > 0.5 else 'black',
                              fontweight='bold')
    
    plt.colorbar(im1, ax=axes[0,1], label='Effect Size')
    
    # Plot 3: Statistical power analysis
    print("\n   ⚡ Computing statistical power...")
    
    # Calculate power for detecting different effect sizes
    from scipy.stats import norm
    
    effect_sizes = np.linspace(0, 3, 100)
    sample_size = min([len(data) for data in regions_data.values()])  # Use smallest sample
    alpha = 0.05
    
    # Power calculation for two-sample t-test
    def calculate_power(effect_size, n, alpha=0.05):
        # Critical value
        t_crit = scipy_stats.t.ppf(1 - alpha/2, 2*n - 2)
        
        # Non-centrality parameter
        ncp = effect_size * np.sqrt(n/2)
        
        # Power = P(|T| > t_crit | H1 is true)
        power = 1 - scipy_stats.nct.cdf(t_crit, 2*n - 2, ncp) + scipy_stats.nct.cdf(-t_crit, 2*n - 2, ncp)
        return power
    
    powers = [calculate_power(es, sample_size) for es in effect_sizes]
    
    axes[1,0].plot(effect_sizes, powers, 'b-', linewidth=2)
    axes[1,0].axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='80% Power')
    axes[1,0].axvline(x=0.8, color='orange', linestyle=':', alpha=0.7, label='Large Effect')
    axes[1,0].set_title(f'C) Statistical Power Analysis (n={sample_size})')
    axes[1,0].set_xlabel('Effect Size (Cohen\'s d)')
    axes[1,0].set_ylabel('Statistical Power')
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].legend()
    axes[1,0].set_ylim(0, 1)
    
    # Plot 4: Summary of findings
    summary_text = f"""
COMPARATIVE ANALYSIS RESULTS
{'='*35}

ANOVA Test:
F({n_regions-1}, {sum(len(data) for data in regions_data.values()) - n_regions}) = {f_stat:.3f}
p-value = {p_value:.2e}
Result: {'Significant' if p_value < 0.05 else 'Not significant'} differences

Pairwise Comparisons:
(Bonferroni corrected α = {alpha_corrected:.4f})

"""
    
    for result in pairwise_results:
        summary_text += f"{result['region1']} vs {result['region2']}:\n"
        summary_text += f"  p = {result['p_value']:.2e} ({'*' if result['significant'] else 'ns'})\n"
        summary_text += f"  Cohen's d = {result['cohens_d']:.3f} "
        
        # Interpret effect size
        abs_d = abs(result['cohens_d'])
        if abs_d < 0.2:
            effect_interpretation = "(negligible)"
        elif abs_d < 0.5:
            effect_interpretation = "(small)"
        elif abs_d < 0.8:
            effect_interpretation = "(medium)"
        else:
            effect_interpretation = "(large)"
        
        summary_text += effect_interpretation + "\n\n"
    
    summary_text += "INTERPRETATION:\n"
    summary_text += "• * = statistically significant\n"
    summary_text += "• ns = not significant\n"
    summary_text += "• Effect sizes: 0.2=small, 0.5=medium, 0.8=large"
    
    axes[1,1].text(0.05, 0.95, summary_text, transform=axes[1,1].transAxes, 
                   fontsize=9, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.8))
    axes[1,1].set_xlim(0, 1)
    axes[1,1].set_ylim(0, 1)
    axes[1,1].axis('off')
    axes[1,1].set_title('D) Statistical Summary')
    
    plt.tight_layout()
    plt.savefig('comparative_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return pairwise_results, f_stat, p_value

def create_advanced_analysis_summary():
    """
    Create a comprehensive summary of all advanced analysis techniques.
    
    This helps students understand when and how to use each method!
    """
    summary = """
🎯 ADVANCED ANALYSIS TECHNIQUES SUMMARY
======================================

🌊 SPECTRAL ANALYSIS:
Purpose: Find periodic patterns in time series data
When to use: Looking for cycles (daily, seasonal, etc.)
Key outputs: Dominant periods, power spectral density
Applications: Climate cycles, tidal analysis, seasonal patterns

📊 PRINCIPAL COMPONENT ANALYSIS (PCA):
Purpose: Reduce dimensionality and find dominant patterns
When to use: Multiple variables, complex datasets
Key outputs: Principal components, explained variance
Applications: Pattern recognition, data compression, noise reduction

🔍 CLUSTERING ANALYSIS:
Purpose: Group similar observations together
When to use: Exploring natural groupings in data
Key outputs: Cluster assignments, regime identification
Applications: Water mass classification, ecosystem regimes

📈 UNCERTAINTY ANALYSIS:
Purpose: Quantify confidence in estimates
When to use: All scientific analyses!
Key outputs: Confidence intervals, error bounds
Applications: Model validation, hypothesis testing

🧪 COMPARATIVE ANALYSIS:
Purpose: Test for significant differences between groups
When to use: Comparing regions, times, conditions
Key outputs: P-values, effect sizes, statistical power
Applications: Climate change detection, regional comparisons

ANALYSIS WORKFLOW DECISION TREE:
================================

1. Start with exploratory analysis (Lessons 1-7)
2. Ask: "What patterns might exist?"
   → Use spectral analysis for time patterns
   → Use PCA for spatial/multivariate patterns
3. Ask: "Are there natural groupings?"
   → Use clustering analysis
4. Ask: "How certain are my results?"
   → Use uncertainty analysis
5. Ask: "Are differences significant?"
   → Use comparative analysis

PROFESSIONAL TIPS:
==================

✅ BEST PRACTICES:
• Always visualize your data first
• Check assumptions (normality, independence)
• Use multiple methods to confirm findings
• Report uncertainty and effect sizes
• Consider practical vs statistical significance

❌ COMMON MISTAKES:
• Applying methods without understanding assumptions
• Over-interpreting statistical significance
• Ignoring multiple comparison corrections
• Not checking for outliers or data quality issues
• Confusing correlation with causation

🎓 ADVANCED TOPICS TO EXPLORE:
• Wavelet analysis for time-frequency decomposition
• Machine learning classification methods
• Non-parametric statistical tests
• Bayesian analysis and model comparison
• Cross-validation and model selection
• Time series forecasting methods

💼 CAREER APPLICATIONS:
• Research scientist positions
• Data analyst/scientist roles
• Environmental consulting
• Climate modeling
• Oceanographic research
• Government policy analysis

Remember: Advanced methods are powerful tools, but they require
careful application and interpretation. Always start with simple
methods and build complexity as needed!
"""
    
    print(summary)
    
    # Save to file
    with open('advanced_analysis_guide.txt', 'w') as f:
        f.write(summary)
    
    print("\n💾 Advanced analysis guide saved as 'advanced_analysis_guide.txt'")

def main():
    """
    Main function that demonstrates advanced analysis techniques.
    
    This capstone lesson brings together sophisticated methods!
    """
    print("🔬 LESSON 8: ADVANCED ANALYSIS TECHNIQUES")
    print("=" * 50)
    print()
    print("Welcome to the final lesson! 🎉")
    print("You're about to learn advanced data science techniques")
    print("used by professional oceanographers and data scientists.")
    print()
    print("Today's advanced methods:")
    print("• Spectral analysis for finding periodic patterns")
    print("• Principal Component Analysis (PCA) for pattern recognition")
    print("• Clustering analysis for regime identification")
    print("• Uncertainty quantification using bootstrap methods")
    print("• Comparative analysis with hypothesis testing")
    print()
    print("These techniques will take your data analysis to the next level!")
    print()
    
    # Step 1: Create multi-drifter dataset
    print("STEP 1: Creating multi-drifter dataset...")
    data = create_multi_drifter_dataset()
    
    # Step 2: Spectral analysis
    print("\nSTEP 2: Spectral analysis - Finding periodic patterns...")
    spectral_results = spectral_analysis(data)
    
    # Step 3: Principal Component Analysis
    print("\nSTEP 3: Principal Component Analysis - Pattern recognition...")
    pca_model, temp_matrix, times = principal_component_analysis(data)
    
    # Step 4: Clustering analysis
    print("\nSTEP 4: Clustering analysis - Regime identification...")
    cluster_labels, kmeans_model, features = clustering_analysis(data)
    
    # Step 5: Uncertainty analysis
    print("\nSTEP 5: Uncertainty analysis - Quantifying confidence...")
    bootstrap_results = uncertainty_analysis(data)
    
    # Step 6: Comparative analysis
    print("\nSTEP 6: Comparative analysis - Statistical hypothesis testing...")
    comparison_results, f_stat, p_value = comparative_analysis(data)
    
    # Step 7: Create summary guide
    print("\nSTEP 7: Creating advanced analysis guide...")
    create_advanced_analysis_summary()
    
    print("\n" + "=" * 50)
    print("🎉 LESSON 8 COMPLETE!")
    print("🎓 CONGRATULATIONS - YOU'VE COMPLETED ALL 8 LESSONS!")
    print("\nWhat you've mastered:")
    print("• Spectral analysis and frequency domain methods")
    print("• Dimensionality reduction with PCA")
    print("• Unsupervised learning with clustering")
    print("• Uncertainty quantification with bootstrap methods")
    print("• Hypothesis testing and statistical comparisons")
    print("• Professional data science workflows")
    print("\n📊 Files created:")
    print("   📈 spectral_analysis.png")
    print("   📈 pca_analysis.png")
    print("   📈 clustering_analysis.png")
    print("   📈 uncertainty_analysis.png")
    print("   📈 comparative_analysis.png")
    print("   📋 advanced_analysis_guide.txt")
    print("\n🌟 YOU ARE NOW READY FOR REAL OCEANOGRAPHIC RESEARCH!")
    print("\nYour journey from basic plotting to advanced analysis is complete.")
    print("You have learned:")
    print("   🌊 Ocean physics and processes")
    print("   📊 Professional data analysis techniques")
    print("   📝 Scientific communication skills")
    print("   🔬 Research-level statistical methods")
    print("   💻 Industry-standard programming practices")
    print("\n🚀 NEXT STEPS:")
    print("• Apply these skills to real GDP drifter data")
    print("• Contribute to open-source oceanography projects")
    print("• Consider careers in oceanography, data science, or climate research")
    print("• Share your knowledge by teaching others!")
    print("\n🌍 The ocean is waiting for your discoveries!")
    print("=" * 50)

# Educational extensions for advanced techniques
"""
🎓 ADVANCED EDUCATIONAL EXTENSIONS:

RESEARCH CONNECTIONS:
1. Real Applications:
   - El Niño/La Niña detection using spectral analysis
   - Ocean heat content using PCA
   - Water mass classification using clustering
   - Climate model validation using comparative analysis

2. Professional Development:
   - Graduate school preparation
   - Research internship opportunities
   - Scientific conference presentation skills
   - Peer review and publication process

3. Interdisciplinary Applications:
   - Marine biology and ecosystem analysis
   - Climate science and paleoclimatology
   - Fisheries science and management
   - Coastal engineering and hazards

ADVANCED PROGRAMMING CONCEPTS:
- Object-oriented programming for analysis workflows
- Version control with Git for reproducible research
- High-performance computing for large datasets
- Cloud computing and distributed analysis
- API integration for real-time data access

CAREER PATHWAYS:
- Oceanographic research scientist
- Climate data analyst
- Environmental consultant
- Data scientist in tech industry
- Government research positions
- Academic research and teaching
- Science communication and outreach

CONTINUING EDUCATION:
- Graduate courses in physical oceanography
- Machine learning and AI applications
- Advanced statistical methods
- Scientific computing and modeling
- Research ethics and data management
"""

if __name__ == "__main__":
    main()
