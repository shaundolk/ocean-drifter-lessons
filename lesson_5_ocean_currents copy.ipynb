 
#!/usr/bin/env python3
"""
LESSON 5: OCEAN CURRENT ANALYSIS
================================

Learning Goals:
- Understand what ocean currents are and why they matter
- Learn to calculate velocity from position data
- Visualize current speed and direction
- Discover how currents transport drifters (and everything else!)

What you'll learn about the ocean:
- Ocean currents are like rivers in the sea
- Currents transport heat, nutrients, and marine life around the globe
- Current speed and direction change with location and time
- Major current systems like the Gulf Stream shape our climate

Ocean currents are invisible highways that move water, heat, nutrients, 
marine life, and unfortunately pollution around our planet!

Prerequisites: Complete Lessons 1-4 first!
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import FancyArrowPatch
import warnings
warnings.filterwarnings('ignore')

def calculate_velocity_from_positions(data):
    """
    Calculate ocean current velocity from drifter position changes.
    
    This is the core of current analysis! We look at how the drifter
    moves between measurements to understand the ocean current.
    
    Args:
        data: DataFrame with time, latitude, longitude columns
    
    Returns:
        DataFrame with added velocity columns
    """
    print("🧮 Calculating ocean current velocities from drifter movement...")
    
    # Make a copy to avoid modifying original data
    df = data.copy()
    
    # Calculate time differences (in hours)
    df['time_diff'] = df['time'].diff().dt.total_seconds() / 3600  # hours
    
    # Calculate position differences
    # Latitude difference in degrees
    df['lat_diff'] = df['latitude'].diff()
    
    # Longitude difference in degrees
    df['lon_diff'] = df['longitude'].diff()
    
    # Convert lat/lon differences to distances in kilometers
    # 1 degree of latitude ≈ 111 km everywhere on Earth
    df['north_distance'] = df['lat_diff'] * 111.0  # km
    
    # 1 degree of longitude varies with latitude: 111 * cos(latitude)
    df['east_distance'] = (df['lon_diff'] * 111.0 * 
                          np.cos(np.radians(df['latitude'])))  # km
    
    # Calculate velocities in km/hour, then convert to cm/s (oceanographic standard)
    # 1 km/h = 27.778 cm/s
    df['u_velocity'] = (df['east_distance'] / df['time_diff']) * 27.778  # cm/s eastward
    df['v_velocity'] = (df['north_distance'] / df['time_diff']) * 27.778  # cm/s northward
    
    # Calculate total speed and direction
    df['current_speed'] = np.sqrt(df['u_velocity']**2 + df['v_velocity']**2)  # cm/s
    df['current_direction'] = np.degrees(np.arctan2(df['v_velocity'], df['u_velocity']))
    
    # Clean up first row (NaN values from diff calculation)
    df = df.dropna().reset_index(drop=True)
    
    print(f"✅ Calculated velocities for {len(df)} time steps")
    print(f"🏃 Average current speed: {df['current_speed'].mean():.1f} cm/s")
    print(f"⚡ Maximum current speed: {df['current_speed'].max():.1f} cm/s")
    
    # Add some context for students
    avg_speed_kmh = df['current_speed'].mean() / 27.778
    print(f"💡 That's about {avg_speed_kmh:.2f} km/hour - like a slow bicycle!")
    
    return df

def load_drifter_data_with_currents():
    """
    Load drifter data and simulate realistic ocean current patterns.
    
    We'll create data that mimics real ocean currents like the Gulf Stream!
    """
    print("🌊 Loading drifter data with realistic current patterns...")
    
    # Create a more realistic track that follows ocean current patterns
    hours = 120 * 24  # 4 months of hourly data
    times = pd.date_range('2023-01-01', periods=hours, freq='H')
    
    # Start in the western North Atlantic (like many real GDP drifters)
    start_lat, start_lon = 25.0, -80.0  # Near Florida
    
    # Simulate a drifter caught in Gulf Stream-like current
    # Gulf Stream moves northeast at varying speeds
    
    lats = [start_lat]
    lons = [start_lon]
    
    for i in range(1, len(times)):
        current_lat = lats[-1]
        current_lon = lons[-1]
        
        # Simulate Gulf Stream: strong eastward and northward flow
        # Speed decreases as we move away from the coast
        distance_from_coast = current_lon + 75  # Distance from ~75°W
        
        if distance_from_coast < 10:  # Close to coast - strong current
            east_velocity = 80 + np.random.normal(0, 20)  # cm/s
            north_velocity = 40 + np.random.normal(0, 15)  # cm/s
        elif distance_from_coast < 20:  # Mid-stream
            east_velocity = 60 + np.random.normal(0, 15)  # cm/s
            north_velocity = 30 + np.random.normal(0, 10)  # cm/s
        else:  # Far from coast - weaker current
            east_velocity = 20 + np.random.normal(0, 10)  # cm/s
            north_velocity = 10 + np.random.normal(0, 8)   # cm/s
        
        # Add some meandering (currents aren't perfectly straight!)
        meander = 0.1 * np.sin(2 * np.pi * i / (30 * 24))  # 30-day cycle
        north_velocity += meander * 20
        
        # Convert velocity to position change (1 hour timestep)
        # cm/s * 3600 s/hour * 1 km/100000 cm = km/hour
        east_km_per_hour = east_velocity * 3600 / 100000
        north_km_per_hour = north_velocity * 3600 / 100000
        
        # Convert km to degrees (approximately)
        lat_change = north_km_per_hour / 111.0  # degrees
        lon_change = east_km_per_hour / (111.0 * np.cos(np.radians(current_lat)))
        
        new_lat = current_lat + lat_change
        new_lon = current_lon + lon_change
        
        lats.append(new_lat)
        lons.append(new_lon)
    
    # Create DataFrame
    data = pd.DataFrame({
        'time': times,
        'latitude': lats,
        'longitude': lons,
        'drifter_id': 23456
    })
    
    # Add some realistic temperature data based on Gulf Stream
    # Gulf Stream carries warm water north
    base_temp = 24 - 0.2 * np.abs(np.array(lats) - 30)  # Warmer in Gulf Stream core
    seasonal = 3 * np.sin(2 * np.pi * np.arange(len(times)) / (365*24))
    noise = np.random.normal(0, 0.5, len(times))
    data['sst'] = base_temp + seasonal + noise
    
    print(f"✅ Created {len(data)} data points")
    print(f"📍 Track spans {data['latitude'].min():.1f}°N to {data['latitude'].max():.1f}°N")
    print(f"📍 Track spans {data['longitude'].max():.1f}°W to {data['longitude'].min():.1f}°W")
    
    return data

def plot_current_speed_timeseries(data):
    """
    Plot how current speed changes over time.
    
    This shows students that ocean currents aren't constant!
    """
    print("\n📈 Creating current speed time series...")
    
    plt.figure(figsize=(12, 8))
    
    # Main speed plot
    plt.subplot(2, 1, 1)
    plt.plot(data['time'], data['current_speed'], 'b-', linewidth=0.8, alpha=0.7)
    
    # Add rolling average to show trends
    smooth_speed = data['current_speed'].rolling(window=48, center=True).mean()  # 2-day average
    plt.plot(data['time'], smooth_speed, 'r-', linewidth=2, label='2-day average')
    
    plt.title('Ocean Current Speed Over Time', fontsize=14, fontweight='bold')
    plt.ylabel('Current Speed (cm/s)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add speed categories
    plt.axhline(y=25, color='orange', linestyle='--', alpha=0.7, label='Moderate current')
    plt.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='Strong current')
    
    # Direction plot
    plt.subplot(2, 1, 2)
    plt.scatter(data['time'], data['current_direction'], c=data['current_speed'], 
               s=10, cmap='viridis', alpha=0.6)
    plt.colorbar(label='Speed (cm/s)')
    plt.title('Current Direction Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Direction (degrees)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add direction labels
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    plt.axhline(y=90, color='gray', linestyle='-', alpha=0.3)
    plt.text(data['time'].iloc[len(data)//4], 0, 'East', ha='center', va='bottom')
    plt.text(data['time'].iloc[len(data)//4], 90, 'North', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('current_speed_timeseries.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print interesting statistics
    print(f"🏃 Average current speed: {data['current_speed'].mean():.1f} cm/s")
    print(f"⚡ Fastest current encountered: {data['current_speed'].max():.1f} cm/s")
    print(f"🐌 Slowest current encountered: {data['current_speed'].min():.1f} cm/s")
    
    # Convert to everyday units
    avg_kmh = data['current_speed'].mean() / 27.778
    max_kmh = data['current_speed'].max() / 27.778
    print(f"💡 Average speed in km/h: {avg_kmh:.2f} km/h")
    print(f"💡 Maximum speed in km/h: {max_kmh:.2f} km/h")

def plot_current_vectors(data):
    """
    Create a map showing current vectors (arrows showing speed and direction).
    
    This is one of the most important oceanographic visualizations!
    """
    print("\n🗺️  Creating current vector map...")
    
    # Subsample data for cleaner visualization (every 24th point = daily)
    step = 24
    plot_data = data[::step].copy()
    
    fig = plt.figure(figsize=(12, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, color='lightgray')
    ax.add_feature(cfeature.OCEAN, color='lightblue')
    
    # Set extent
    margin = 2
    ax.set_extent([data['longitude'].min() - margin, data['longitude'].max() + margin,
                   data['latitude'].min() - margin, data['latitude'].max() + margin])
    
    # Plot the drifter path
    ax.plot(data['longitude'], data['latitude'], 'k--', linewidth=1, alpha=0.5,
            transform=ccrs.PlateCarree(), label='Drifter path')
    
    # Plot current vectors as arrows
    # Scale arrows based on current speed
    scale = 0.02  # Adjust this to make arrows visible but not overlapping
    
    for i, row in plot_data.iterrows():
        if pd.notna(row['current_speed']):  # Skip NaN values
            # Calculate arrow components
            speed = row['current_speed']
            direction_rad = np.radians(row['current_direction'])
            
            # Arrow length proportional to speed
            dx = speed * np.cos(direction_rad) * scale
            dy = speed * np.sin(direction_rad) * scale
            
            # Color arrow by speed
            if speed < 25:
                color = 'blue'
                alpha = 0.6
            elif speed < 50:
                color = 'orange'
                alpha = 0.7
            else:
                color = 'red'
                alpha = 0.8
            
            ax.arrow(row['longitude'], row['latitude'], dx, dy,
                    head_width=0.3, head_length=0.2, fc=color, ec=color,
                    alpha=alpha, transform=ccrs.PlateCarree())
    
    # Mark start and end
    ax.plot(data['longitude'].iloc[0], data['latitude'].iloc[0], 'go', 
            markersize=10, transform=ccrs.PlateCarree(), label='Start')
    ax.plot(data['longitude'].iloc[-1], data['latitude'].iloc[-1], 'ro', 
            markersize=10, transform=ccrs.PlateCarree(), label='End')
    
    plt.title('Ocean Current Vectors\n(Arrows show current speed and direction)', 
              fontsize=14, fontweight='bold')
    plt.legend(loc='upper right')
    
    # Add gridlines
    ax.gridlines(draw_labels=True, alpha=0.3)
    
    # Add legend for arrow colors
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='>', color='blue', label='Slow (<25 cm/s)', 
                             markersize=10, linestyle='None'),
                      Line2D([0], [0], marker='>', color='orange', label='Moderate (25-50 cm/s)', 
                             markersize=10, linestyle='None'),
                      Line2D([0], [0], marker='>', color='red', label='Fast (>50 cm/s)', 
                             markersize=10, linestyle='None')]
    
    ax.legend(handles=legend_elements, loc='lower left', title='Current Speed')
    
    plt.tight_layout()
    plt.savefig('current_vectors.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("🏹 Each arrow shows the current direction and speed at that location")
    print("🔵 Blue arrows = slow currents")
    print("🟠 Orange arrows = moderate currents") 
    print("🔴 Red arrows = fast currents")

def analyze_current_patterns(data):
    """
    Analyze interesting patterns in the current data.
    
    Teach students to think like oceanographers!
    """
    print("\n🔍 Analyzing ocean current patterns...")
    
    # Pattern 1: Speed distribution
    print("\n📊 Pattern 1: Current Speed Distribution")
    slow_percent = (data['current_speed'] < 25).mean() * 100
    moderate_percent = ((data['current_speed'] >= 25) & (data['current_speed'] < 50)).mean() * 100
    fast_percent = (data['current_speed'] >= 50).mean() * 100
    
    print(f"   • Slow currents (<25 cm/s): {slow_percent:.1f}% of the time")
    print(f"   • Moderate currents (25-50 cm/s): {moderate_percent:.1f}% of the time")
    print(f"   • Fast currents (>50 cm/s): {fast_percent:.1f}% of the time")
    
    # Pattern 2: Dominant direction
    print("\n🧭 Pattern 2: Current Direction Analysis")
    
    # Convert directions to compass quadrants
    def direction_to_quadrant(deg):
        if -45 <= deg < 45:
            return "East"
        elif 45 <= deg < 135:
            return "North"
        elif deg >= 135 or deg < -135:
            return "West"
        else:
            return "South"
    
    data['quadrant'] = data['current_direction'].apply(direction_to_quadrant)
    quadrant_counts = data['quadrant'].value_counts(normalize=True) * 100
    
    for direction, percent in quadrant_counts.items():
        print(f"   • {direction}ward flow: {percent:.1f}% of the time")
    
    dominant_direction = quadrant_counts.index[0]
    print(f"   • The current flows mostly toward the {dominant_direction}")
    
    # Pattern 3: Speed vs Location
    print("\n📍 Pattern 3: Current Speed by Location")
    
    # Divide track into segments
    n_segments = 4
    segment_size = len(data) // n_segments
    
    for i in range(n_segments):
        start_idx = i * segment_size
        end_idx = (i + 1) * segment_size if i < n_segments - 1 else len(data)
        
        segment_data = data.iloc[start_idx:end_idx]
        avg_lat = segment_data['latitude'].mean()
        avg_lon = segment_data['longitude'].mean()
        avg_speed = segment_data['current_speed'].mean()
        
        print(f"   • Segment {i+1} ({avg_lat:.1f}°N, {avg_lon:.1f}°W): {avg_speed:.1f} cm/s average")
    
    # Pattern 4: Correlation with temperature
    if 'sst' in data.columns:
        print("\n🌡️ Pattern 4: Current Speed vs Temperature")
        correlation = data['current_speed'].corr(data['sst'])
        
        if correlation > 0.3:
            print(f"   • Strong positive correlation ({correlation:.2f})")
            print("   • Faster currents tend to occur in warmer water!")
        elif correlation < -0.3:
            print(f"   • Strong negative correlation ({correlation:.2f})")
            print("   • Faster currents tend to occur in cooler water!")
        else:
            print(f"   • Weak correlation ({correlation:.2f})")
            print("   • Current speed and temperature are not strongly related")

def plot_current_analysis_dashboard(data):
    """
    Create a comprehensive dashboard showing all aspects of current analysis.
    """
    print("\n📋 Creating current analysis dashboard...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Ocean Current Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # Plot 1: Speed time series
    axes[0,0].plot(data['time'], data['current_speed'], 'b-', alpha=0.7, linewidth=0.8)
    smooth = data['current_speed'].rolling(window=48, center=True).mean()
    axes[0,0].plot(data['time'], smooth, 'r-', linewidth=2)
    axes[0,0].set_title('Current Speed Over Time')
    axes[0,0].set_ylabel('Speed (cm/s)')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Speed histogram
    axes[0,1].hist(data['current_speed'], bins=20, edgecolor='black', alpha=0.7, color='skyblue')
    axes[0,1].axvline(data['current_speed'].mean(), color='red', linestyle='--', linewidth=2)
    axes[0,1].set_title('Current Speed Distribution')
    axes[0,1].set_xlabel('Speed (cm/s)')
    axes[0,1].set_ylabel('Count')
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Direction rose plot (simplified)
    direction_bins = np.arange(0, 360, 30)
    direction_counts, _ = np.histogram(data['current_direction'] % 360, bins=direction_bins)
    theta = np.radians(direction_bins[:-1] + 15)  # Center of bins
    
    ax_polar = plt.subplot(2, 3, 3, projection='polar')
    ax_polar.bar(theta, direction_counts, width=np.radians(30), alpha=0.7)
    ax_polar.set_title('Current Direction Distribution')
    ax_polar.set_theta_zero_location('E')  # East = 0°
    ax_polar.set_theta_direction(1)  # Counter-clockwise
    
    # Plot 4: Speed vs latitude
    axes[1,0].scatter(data['latitude'], data['current_speed'], alpha=0.5, s=10, c='blue')
    z = np.polyfit(data['latitude'], data['current_speed'], 1)
    p = np.poly1d(z)
    axes[1,0].plot(data['latitude'], p(data['latitude']), "r--", alpha=0.8, linewidth=2)
    axes[1,0].set_title('Current Speed vs Latitude')
    axes[1,0].set_xlabel('Latitude (°N)')
    axes[1,0].set_ylabel('Speed (cm/s)')
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot 5: U vs V velocity scatter
    axes[1,1].scatter(data['u_velocity'], data['v_velocity'], 
                     c=data['current_speed'], s=20, cmap='viridis', alpha=0.7)
    axes[1,1].axhline(0, color='k', linestyle='-', alpha=0.3)
    axes[1,1].axvline(0, color='k', linestyle='-', alpha=0.3)
    axes[1,1].set_title('Current Components (U vs V)')
    axes[1,1].set_xlabel('Eastward Velocity (cm/s)')
    axes[1,1].set_ylabel('Northward Velocity (cm/s)')
    axes[1,1].grid(True, alpha=0.3)
    
    # Plot 6: Speed vs temperature (if available)
    if 'sst' in data.columns:
        axes[1,2].scatter(data['sst'], data['current_speed'], alpha=0.5, s=10, c='green')
        z = np.polyfit(data['sst'], data['current_speed'], 1)
        p = np.poly1d(z)
        axes[1,2].plot(data['sst'], p(data['sst']), "r--", alpha=0.8, linewidth=2)
        axes[1,2].set_title('Current Speed vs Temperature')
        axes[1,2].set_xlabel('Temperature (°C)')
        axes[1,2].set_ylabel('Speed (cm/s)')
    else:
        axes[1,2].text(0.5, 0.5, 'Temperature data\nnot available', 
                      transform=axes[1,2].transAxes, ha='center', va='center')
        axes[1,2].set_title('Current Speed vs Temperature')
    
    axes[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('current_analysis_dashboard.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    """
    Main function that runs the complete ocean current analysis.
    
    This teaches students how ocean currents work and how to analyze them!
    """
    print("🌊 LESSON 5: OCEAN CURRENT ANALYSIS")
    print("=" * 50)
    print()
    print("Welcome to the invisible rivers of the sea! 🏊‍♀️")
    print("Ocean currents are streams of water flowing through the ocean,")
    print("carrying heat, nutrients, marine life, and even floating trash")
    print("around our planet. Let's discover how to measure and visualize them!")
    print()
    
    # Step 1: Load data with realistic current patterns
    print("STEP 1: Loading drifter data...")
    drifter_data = load_drifter_data_with_currents()
    
    # Step 2: Calculate velocities from position changes
    print("\nSTEP 2: Calculating current velocities...")
    current_data = calculate_velocity_from_positions(drifter_data)
    
    # Step 3: Plot current speed over time
    plot_current_speed_timeseries(current_data)
    
    # Step 4: Create current vector map
    plot_current_vectors(current_data)
    
    # Step 5: Analyze current patterns
    analyze_current_patterns(current_data)
    
    # Step 6: Create comprehensive dashboard
    plot_current_analysis_dashboard(current_data)
    
    print("\n" + "=" * 50)
    print("🎉 LESSON 5 COMPLETE!")
    print("\nWhat you've learned:")
    print("• How to calculate ocean current velocity from position data")
    print("• Ocean currents vary in speed and direction")
    print("• How to visualize currents using vector arrows")
    print("• Current patterns reveal ocean circulation systems")
    print("• How currents transport everything in the ocean")
    print("\n🌊 Ocean currents are like invisible highways in the sea!")
    print("They transport:")
    print("  • Heat from the equator to the poles (climate regulation)")
    print("  • Nutrients that feed marine ecosystems")
    print("  • Marine larvae and adult animals")
    print("  • Unfortunately, plastic pollution too")
    print("\n🚀 Ready for Lesson 6: Data Quality and Validation!")
    print("=" * 50)

# Educational extensions
"""
🎓 EDUCATIONAL EXTENSIONS:

OCEAN SCIENCE CONNECTIONS:
1. Major Current Systems:
   - Gulf Stream (warm water north along US East Coast)
   - Kuroshio Current (Pacific equivalent of Gulf Stream)
   - Antarctic Circumpolar Current (world's largest current)
   - California Current (cold water south along US West Coast)

2. Why Currents Matter:
   - Climate regulation (Gulf Stream keeps Europe warm)
   - Marine ecosystems (upwelling brings nutrients)
   - Navigation (ships use currents to save fuel)
   - Pollution transport (garbage patches, oil spills)

3. What Drives Currents:
   - Wind (surface currents)
   - Density differences (deep currents)
   - Earth's rotation (Coriolis effect)
   - Coastline shapes (boundary currents)

REAL-WORLD APPLICATIONS:
- Search and rescue operations
- Oil spill response
- Fisheries management
- Climate modeling
- Renewable energy (current turbines)

ADVANCED ACTIVITIES:
- Compare current data from different seasons
- Look up major current systems on ocean current maps
- Research how climate change affects ocean currents
- Investigate the Great Pacific Garbage Patch
- Study how marine animals use currents for migration

PROGRAMMING CONCEPTS LEARNED:
- Vector calculations (speed and direction)
- Coordinate system transformations
- Time series differentiation
- Statistical correlation analysis
- Data visualization with arrows/vectors
"""

if __name__ == "__main__":
    main()
