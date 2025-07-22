#!/usr/bin/env python3
"""
LESSON 4: SEA SURFACE TEMPERATURE ANALYSIS
===========================================

Learning Goals:
- Understand what sea surface temperature (SST) tells us about the ocean
- Learn to work with temperature data from drifters
- Create temperature plots and maps
- Discover patterns in ocean temperature

What you'll learn about the ocean:
- Ocean temperature varies by location and season
- Warm water is typically found near the equator
- Cold water is found near the poles
- Ocean currents transport warm and cold water around the globe

Prerequisites: Complete Lessons 1-3 first!
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def load_drifter_data_with_temperature():
    """
    Load drifter data that includes temperature measurements.
    
    For this educational example, we'll create realistic sample data.
    In a real application, you'd load this from the GDP dataset.
    
    Returns:
        pandas.DataFrame: Drifter data with lat, lon, time, and temperature
    """
    print("📊 Loading drifter data with temperature measurements...")
    
    # Create sample data that represents realistic ocean drifter measurements
    # In real life, this would come from: clouddrift.datasets.gdp1h()
    
    # Generate sample track across different ocean temperatures
    days = 90  # 3 months of data
    times = pd.date_range('2023-01-01', periods=days*24, freq='H')
    
    # Create a track that moves from warm to cold water
    # Start in warm tropical Atlantic, move north to cooler waters
    start_lat, start_lon = 10.0, -40.0  # Tropical Atlantic
    end_lat, end_lon = 45.0, -30.0      # North Atlantic
    
    lats = np.linspace(start_lat, end_lat, len(times))
    lons = np.linspace(start_lon, end_lon, len(times))
    
    # Add some realistic random movement
    lats += np.random.normal(0, 0.5, len(times))
    lons += np.random.normal(0, 0.5, len(times))
    
    # Calculate realistic temperatures based on latitude
    # Warmer near equator, cooler towards poles
    base_temp = 28 - 0.4 * np.abs(lats)  # Basic temperature gradient
    seasonal_variation = 2 * np.sin(2 * np.pi * np.arange(len(times)) / (365*24))
    daily_variation = 0.5 * np.sin(2 * np.pi * np.arange(len(times)) / 24)
    random_noise = np.random.normal(0, 0.3, len(times))
    
    temperatures = base_temp + seasonal_variation + daily_variation + random_noise
    
    # Create DataFrame
    data = pd.DataFrame({
        'time': times,
        'latitude': lats,
        'longitude': lons,
        'sst': temperatures,  # Sea Surface Temperature in Celsius
        'drifter_id': 12345   # Sample drifter ID
    })
    
    print(f"✅ Loaded {len(data)} temperature measurements")
    print(f"📍 Temperature range: {data['sst'].min():.1f}°C to {data['sst'].max():.1f}°C")
    
    return data

def plot_temperature_timeseries(data):
    """
    Create a time series plot of sea surface temperature.
    
    This helps students see how temperature changes over time.
    """
    print("\n📈 Creating temperature time series plot...")
    
    plt.figure(figsize=(12, 6))
    
    # Plot temperature over time
    plt.plot(data['time'], data['sst'], 'b-', linewidth=1, alpha=0.7)
    
    # Add a smooth trend line to show overall pattern
    # Rolling mean over 7 days (168 hours)
    smooth_temp = data['sst'].rolling(window=168, center=True).mean()
    plt.plot(data['time'], smooth_temp, 'r-', linewidth=2, label='7-day average')
    
    plt.title('Sea Surface Temperature Over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Temperature (°C)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add some annotations to help students understand
    plt.text(0.02, 0.98, 'Notice how temperature changes as the drifter moves!', 
             transform=plt.gca().transAxes, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
             verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig('temperature_timeseries.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print some interesting statistics
    print(f"🌡️  Average temperature: {data['sst'].mean():.1f}°C")
    print(f"🥵 Warmest temperature: {data['sst'].max():.1f}°C")
    print(f"🥶 Coolest temperature: {data['sst'].min():.1f}°C")

def plot_temperature_histogram(data):
    """
    Create a histogram showing the distribution of temperatures.
    
    This helps students understand how often different temperatures occur.
    """
    print("\n📊 Creating temperature distribution histogram...")
    
    plt.figure(figsize=(10, 6))
    
    # Create histogram
    n, bins, patches = plt.hist(data['sst'], bins=20, edgecolor='black', alpha=0.7)
    
    # Color the bars based on temperature (blue for cold, red for warm)
    for i, (patch, temp) in enumerate(zip(patches, bins)):
        if temp < 15:
            patch.set_facecolor('blue')
        elif temp > 25:
            patch.set_facecolor('red')
        else:
            patch.set_facecolor('orange')
    
    plt.title('Distribution of Sea Surface Temperatures', fontsize=16, fontweight='bold')
    plt.xlabel('Temperature (°C)', fontsize=12)
    plt.ylabel('Number of Observations', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add vertical line for average temperature
    avg_temp = data['sst'].mean()
    plt.axvline(avg_temp, color='black', linestyle='--', linewidth=2, 
                label=f'Average: {avg_temp:.1f}°C')
    plt.legend()
    
    # Add explanation text
    plt.text(0.02, 0.98, 'Blue = Cold water\nOrange = Moderate\nRed = Warm water', 
             transform=plt.gca().transAxes, fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
             verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig('temperature_histogram.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_temperature_map(data):
    """
    Create a map showing the drifter path colored by temperature.
    
    This helps students see the relationship between location and temperature.
    """
    print("\n🗺️  Creating temperature map...")
    
    # Create map with ocean-focused projection
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)
    ax.add_feature(cfeature.LAND, color='lightgray')
    ax.add_feature(cfeature.OCEAN, color='lightblue')
    
    # Set map extent to focus on our data
    margin = 5  # degrees
    ax.set_extent([data['longitude'].min() - margin, data['longitude'].max() + margin,
                   data['latitude'].min() - margin, data['latitude'].max() + margin])
    
    # Plot the drifter track colored by temperature
    scatter = ax.scatter(data['longitude'], data['latitude'], 
                        c=data['sst'], s=20, cmap='coolwarm',
                        transform=ccrs.PlateCarree(), alpha=0.8)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal', 
                       pad=0.05, aspect=50, shrink=0.8)
    cbar.set_label('Sea Surface Temperature (°C)', fontsize=12)
    
    # Mark start and end points
    ax.plot(data['longitude'].iloc[0], data['latitude'].iloc[0], 
            'go', markersize=10, transform=ccrs.PlateCarree(), 
            label='Start (warm water)')
    ax.plot(data['longitude'].iloc[-1], data['latitude'].iloc[-1], 
            'rs', markersize=10, transform=ccrs.PlateCarree(), 
            label='End (cool water)')
    
    plt.title('Drifter Path Colored by Sea Surface Temperature', 
              fontsize=16, fontweight='bold')
    plt.legend(loc='upper right')
    
    # Add gridlines
    ax.gridlines(draw_labels=True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('temperature_map.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("🌊 Notice how temperature changes as the drifter moves north!")
    print("🔥 Red/warm colors = tropical/warm water")
    print("❄️  Blue/cool colors = northern/cool water")

def analyze_temperature_patterns(data):
    """
    Analyze interesting patterns in the temperature data.
    
    This teaches students to think like data scientists!
    """
    print("\n🔍 Analyzing temperature patterns...")
    
    # Add some derived columns for analysis
    data['month'] = data['time'].dt.month
    data['hour'] = data['time'].dt.hour
    data['latitude_rounded'] = data['latitude'].round()
    
    # Pattern 1: Temperature by latitude
    print("\n📍 Pattern 1: Temperature vs. Latitude")
    lat_temp = data.groupby('latitude_rounded')['sst'].mean()
    print(f"   • At {lat_temp.index.min()}°N: {lat_temp.iloc[0]:.1f}°C (southern/warmer)")
    print(f"   • At {lat_temp.index.max()}°N: {lat_temp.iloc[-1]:.1f}°C (northern/cooler)")
    print(f"   • Temperature drops about {(lat_temp.iloc[0] - lat_temp.iloc[-1])/len(lat_temp):.2f}°C per degree latitude")
    
    # Pattern 2: Daily temperature variation
    print("\n🕐 Pattern 2: Daily Temperature Cycle")
    hourly_temp = data.groupby('hour')['sst'].mean()
    warmest_hour = hourly_temp.idxmax()
    coolest_hour = hourly_temp.idxmin()
    print(f"   • Warmest time: {warmest_hour:02d}:00 hours ({hourly_temp.max():.1f}°C)")
    print(f"   • Coolest time: {coolest_hour:02d}:00 hours ({hourly_temp.min():.1f}°C)")
    print(f"   • Daily temperature range: {hourly_temp.max() - hourly_temp.min():.1f}°C")
    
    # Pattern 3: Temperature trend over the journey
    print("\n📈 Pattern 3: Temperature Trend Over Journey")
    start_temp = data['sst'].iloc[:100].mean()  # First 100 observations
    end_temp = data['sst'].iloc[-100:].mean()   # Last 100 observations
    temp_change = end_temp - start_temp
    
    if temp_change > 0:
        print(f"   • Temperature INCREASED by {temp_change:.1f}°C during the journey")
        print("   • The drifter moved toward warmer water! 🌡️⬆️")
    else:
        print(f"   • Temperature DECREASED by {abs(temp_change):.1f}°C during the journey")
        print("   • The drifter moved toward cooler water! 🌡️⬇️")

def create_temperature_summary_plot(data):
    """
    Create a comprehensive summary plot with multiple temperature views.
    
    This brings together everything students have learned!
    """
    print("\n📋 Creating comprehensive temperature summary...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Sea Surface Temperature Analysis Summary', fontsize=16, fontweight='bold')
    
    # Plot 1: Time series
    axes[0,0].plot(data['time'], data['sst'], 'b-', alpha=0.6, linewidth=0.8)
    smooth_temp = data['sst'].rolling(window=168, center=True).mean()
    axes[0,0].plot(data['time'], smooth_temp, 'r-', linewidth=2)
    axes[0,0].set_title('Temperature Over Time')
    axes[0,0].set_ylabel('Temperature (°C)')
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Histogram
    axes[0,1].hist(data['sst'], bins=15, edgecolor='black', alpha=0.7, color='skyblue')
    axes[0,1].axvline(data['sst'].mean(), color='red', linestyle='--', linewidth=2)
    axes[0,1].set_title('Temperature Distribution')
    axes[0,1].set_xlabel('Temperature (°C)')
    axes[0,1].set_ylabel('Count')
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Temperature vs Latitude
    axes[1,0].scatter(data['latitude'], data['sst'], alpha=0.5, s=10)
    z = np.polyfit(data['latitude'], data['sst'], 1)
    p = np.poly1d(z)
    axes[1,0].plot(data['latitude'], p(data['latitude']), "r--", alpha=0.8, linewidth=2)
    axes[1,0].set_title('Temperature vs Latitude')
    axes[1,0].set_xlabel('Latitude (°N)')
    axes[1,0].set_ylabel('Temperature (°C)')
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot 4: Daily temperature cycle
    hourly_avg = data.groupby(data['time'].dt.hour)['sst'].mean()
    axes[1,1].plot(hourly_avg.index, hourly_avg.values, 'o-', linewidth=2, markersize=4)
    axes[1,1].set_title('Average Daily Temperature Cycle')
    axes[1,1].set_xlabel('Hour of Day')
    axes[1,1].set_ylabel('Temperature (°C)')
    axes[1,1].set_xlim(0, 23)
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('temperature_summary.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    """
    Main function that runs the complete sea surface temperature analysis.
    
    This is like a recipe that puts all the ingredients together!
    """
    print("🌊 LESSON 4: SEA SURFACE TEMPERATURE ANALYSIS")
    print("=" * 50)
    print()
    print("Welcome to ocean temperature analysis! 🌡️")
    print("You're about to discover fascinating patterns in sea surface temperature.")
    print()
    
    # Step 1: Load the data
    drifter_data = load_drifter_data_with_temperature()
    
    # Step 2: Basic temperature time series
    plot_temperature_timeseries(drifter_data)
    
    # Step 3: Temperature distribution
    plot_temperature_histogram(drifter_data)
    
    # Step 4: Temperature map
    plot_temperature_map(drifter_data)
    
    # Step 5: Analyze patterns
    analyze_temperature_patterns(drifter_data)
    
    # Step 6: Summary plot
    create_temperature_summary_plot(drifter_data)
    
    print("\n" + "=" * 50)
    print("🎉 LESSON 4 COMPLETE!")
    print("\nWhat you've learned:")
    print("• How to work with temperature data from ocean drifters")
    print("• Ocean temperature varies with location (latitude) and time")
    print("• How to create different types of temperature visualizations")
    print("• How to identify patterns in oceanographic data")
    print("\n🚀 Ready for Lesson 5: Ocean Current Analysis!")
    print("=" * 50)

# Educational notes and extensions for teachers/students
"""
🎓 EDUCATIONAL EXTENSIONS:

For students who want to learn more:

1. TEMPERATURE SCIENCE:
   - Why is water warmer near the equator? (Solar heating)
   - What causes daily temperature variations? (Sun heating during day)
   - How do ocean currents transport heat? (Gulf Stream, etc.)

2. DATA SCIENCE SKILLS:
   - Rolling averages smooth noisy data
   - Histograms show data distributions
   - Scatter plots reveal relationships
   - Color maps show spatial patterns

3. REAL-WORLD CONNECTIONS:
   - Climate change and ocean warming
   - Marine ecosystems and temperature
   - Weather patterns and sea surface temperature
   - Fishing industry and water temperature

4. ADVANCED ACTIVITIES:
   - Compare temperature data from different seasons
   - Analyze temperature data from different ocean basins
   - Look up real drifter data and compare to our examples
   - Research how marine animals respond to temperature changes

5. PROGRAMMING SKILLS LEARNED:
   - Working with time series data
   - Creating multiple plot types
   - Using color maps and colorbars
   - Statistical analysis (mean, min, max, trends)
   - Data grouping and aggregation
"""

if __name__ == "__main__":
    main()
