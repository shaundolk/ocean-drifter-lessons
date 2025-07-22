 
"""
Script 2: Simple Trajectory Plot - Mapping Ocean Adventures!
===========================================================

What you'll learn:
Now that we know how to load drifter data, let's visualize it! This script 
teaches you to create your first map showing where ocean drifters traveled.

Think of it like drawing the path you took on a family road trip, but instead
of roads, we're showing ocean currents!

What this script does:
- Creates a simple map of the ocean
- Draws lines showing where drifters traveled
- Colors different drifters with different colors
- Adds labels so we know what we're looking at

Learning goals:
- Create your first scientific plot
- Understand how to map ocean data
- Learn to customize colors and labels
- Practice interpreting drifter paths
"""

# Import our toolboxes
import matplotlib.pyplot as plt  # This is our plotting/drawing tool
import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime, timedelta

# Import our data loading function from the previous script
# (In a real setup, you might need to adjust this import)
def load_sample_drifter_data():
    """Create sample drifter data for learning"""
    
    print("🌊 Creating sample drifter trajectories...")
    
    # Create more realistic drifter paths
    n_drifters = 4
    n_points_per_drifter = 150  # About 1 week of hourly data per drifter
    
    # Starting positions for our drifters (different parts of the Atlantic)
    start_positions = [
        {"lat": 35.0, "lon": -65.0, "name": "Bermuda Drifter"},      # Near Bermuda
        {"lat": 42.0, "lon": -55.0, "name": "Labrador Drifter"},    # Labrador Sea
        {"lat": 25.0, "lon": -80.0, "name": "Florida Drifter"},     # Off Florida
        {"lat": 40.0, "lon": -40.0, "name": "Mid-Atlantic Drifter"} # Mid-Atlantic
    ]
    
    # Create realistic drifter trajectories
    all_data = []
    
    for i, start in enumerate(start_positions):
        # Create a curved path that simulates ocean current movement
        t = np.linspace(0, 2*np.pi, n_points_per_drifter)
        
        # Add some realistic drift patterns
        lat_drift = start["lat"] + np.cumsum(np.random.normal(0, 0.01, n_points_per_drifter))
        lon_drift = start["lon"] + np.cumsum(np.random.normal(0.02, 0.015, n_points_per_drifter))
        
        # Add some wave-like movement (currents aren't straight lines!)
        lat_wave = 0.5 * np.sin(t * 0.3) * np.exp(-t/10)  # Decreasing wave motion
        lon_wave = 0.3 * np.cos(t * 0.4) * np.exp(-t/8)
        
        final_lat = lat_drift + lat_wave
        final_lon = lon_drift + lon_wave
        
        # Create temperature data (warmer in south, cooler in north)
        base_temp = 25 - (final_lat - 20) * 0.5  # Temperature decreases with latitude
        temperatures = base_temp + np.random.normal(0, 1.5, n_points_per_drifter)
        
        # Create time stamps
        start_time = datetime(2023, 6, 1) + timedelta(days=i)  # Stagger start times
        times = [start_time + timedelta(hours=h) for h in range(n_points_per_drifter)]
        
        # Store data for this drifter
        drifter_data = {
            'id': [i] * n_points_per_drifter,
            'latitude': final_lat,
            'longitude': final_lon,
            'sst': temperatures,
            'time': times,
            'name': [start["name"]] * n_points_per_drifter
        }
        
        all_data.append(pd.DataFrame(drifter_data))
    
    # Combine all drifters
    df = pd.concat(all_data, ignore_index=True)
    
    # Convert to xarray dataset
    ds = xr.Dataset({
        'latitude': (['obs'], df['latitude']),
        'longitude': (['obs'], df['longitude']),
        'sst': (['obs'], df['sst']),
        'id': (['obs'], df['id']),
    }, coords={'time': (['obs'], df['time'])})
    
    print(f"✅ Created {n_drifters} drifter trajectories with {len(df)} total observations")
    return ds

def create_simple_trajectory_plot(ds):
    """
    Create a simple plot showing drifter trajectories
    
    This is like creating a treasure map showing the paths our ocean 
    explorers took!
    """
    
    print("\n🗺️  CREATING YOUR FIRST DRIFTER MAP")
    print("=" * 50)
    
    # Set up our drawing canvas
    plt.figure(figsize=(12, 8))  # Make a nice big plot
    
    # Define colors for different drifters (like giving each one a unique crayon)
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    
    # Get unique drifter IDs
    unique_ids = np.unique(ds.id.values)
    
    print(f"📍 Plotting {len(unique_ids)} drifter trajectories...")
    
    # Plot each drifter's path
    for i, drifter_id in enumerate(unique_ids):
        # Select data for this specific drifter
        drifter_data = ds.where(ds.id == drifter_id, drop=True)
        
        # Extract latitude and longitude
        lats = drifter_data.latitude.values
        lons = drifter_data.longitude.values
        
        # Remove any NaN values
        valid_points = ~(np.isnan(lats) | np.isnan(lons))
        lats = lats[valid_points]
        lons = lons[valid_points]
        
        if len(lats) > 0:  # Only plot if we have valid data
            # Plot the trajectory line
            plt.plot(lons, lats, 
                    color=colors[i % len(colors)], 
                    linewidth=2, 
                    alpha=0.7,
                    label=f'Drifter {drifter_id}')
            
            # Mark the starting point
            plt.plot(lons[0], lats[0], 
                    color=colors[i % len(colors)], 
                    marker='o', 
                    markersize=8, 
                    markeredgecolor='black',
                    markeredgewidth=1)
            
            # Mark the ending point  
            plt.plot(lons[-1], lats[-1], 
                    color=colors[i % len(colors)], 
                    marker='s', 
                    markersize=8, 
                    markeredgecolor='black',
                    markeredgewidth=1)
            
            print(f"   ✓ Drifter {drifter_id}: {len(lats)} points plotted")

    # Make our map look professional
    plt.xlabel('Longitude (degrees East)', fontsize=12)
    plt.ylabel('Latitude (degrees North)', fontsize=12)
    plt.title('Ocean Drifter Trajectories\n(○ = Start, □ = End)', fontsize=14, fontweight='bold')
    
    # Add a grid to make it easier to read coordinates
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Add a legend
    plt.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    
    # Make the plot look neat
    plt.tight_layout()
    
    # Show the plot
    plt.show()
    
    print("✅ Map created successfully!")

def analyze_trajectories(ds):
    """
    Analyze the drifter trajectories to learn interesting facts
    
    Let's be data detectives and discover cool facts about our drifters!
    """
    
    print("\n🔍 ANALYZING DRIFTER JOURNEYS")
    print("=" * 50)
    
    unique_ids = np.unique(ds.id.values)
    
    for drifter_id in unique_ids:
        # Get data for this drifter
        drifter_data = ds.where(ds.id == drifter_id, drop=True)
        
        lats = drifter_data.latitude.values
        lons = drifter_data.longitude.values
        
        # Remove NaN values
        valid_mask = ~(np.isnan(lats) | np.isnan(lons))
        lats = lats[valid_mask]
        lons = lons[valid_mask]
        
        if len(lats) > 1:
            # Calculate how far the drifter traveled
            # This is approximate - we're treating Earth as flat locally
            lat_distance = np.abs(lats[-1] - lats[0]) * 111  # ~111 km per degree latitude
            lon_distance = np.abs(lons[-1] - lons[0]) * 111 * np.cos(np.radians(np.mean(lats)))
            total_distance = np.sqrt(lat_distance**2 + lon_distance**2)
            
            # Direction of travel
            if lats[-1] > lats[0]:
                lat_direction = "North"
            else:
                lat_direction = "South"
                
            if lons[-1] > lons[0]:
                lon_direction = "East"
            else:
                lon_direction = "West"
            
            print(f"\n🎯 Drifter {drifter_id} Journey Summary:")
            print(f"   Starting position: {lats[0]:.2f}°N, {lons[0]:.2f}°E")
            print(f"   Ending position:   {lats[-1]:.2f}°N, {lons[-1]:.2f}°E")
            print(f"   Overall direction: {lat_direction}-{lon_direction}")
            print(f"   Straight-line distance: {total_distance:.1f} km")
            print(f"   Number of observations: {len(lats)}")

def create_detailed_trajectory_plot(ds):
    """
    Create a more detailed trajectory plot with temperature information
    
    Now let's add color to show temperature - warmer colors for warmer water!
    """
    
    print("\n🌡️  CREATING TEMPERATURE-COLORED TRAJECTORY MAP")
    print("=" * 50)
    
    plt.figure(figsize=(14, 10))
    
    unique_ids = np.unique(ds.id.values)
    
    # Create a colormap for temperature (blue=cold, red=hot)
    from matplotlib import cm
    
    for i, drifter_id in enumerate(unique_ids):
        drifter_data = ds.where(ds.id == drifter_id, drop=True)
        
        lats = drifter_data.latitude.values
        lons = drifter_data.longitude.values
        temps = drifter_data.sst.values if 'sst' in drifter_data else None
        
        # Remove NaN values
        if temps is not None:
            valid_mask = ~(np.isnan(lats) | np.isnan(lons) | np.isnan(temps))
        else:
            valid_mask = ~(np.isnan(lats) | np.isnan(lons))
            
        lats = lats[valid_mask]
        lons = lons[valid_mask]
        
        if temps is not None:
            temps = temps[valid_mask]
            
            # Create a scatter plot colored by temperature
            scatter = plt.scatter(lons, lats, 
                               c=temps, 
                               cmap='coolwarm',  # Blue to red colormap
                               s=30,  # Size of points
                               alpha=0.7,
                               label=f'Drifter {drifter_id}')
            
            # Connect the points with a line
            plt.plot(lons, lats, 
                    color='gray', 
                    linewidth=1, 
                    alpha=0.5)
        else:
            # If no temperature data, just plot the trajectory
            plt.plot(lons, lats, 
                    linewidth=2, 
                    alpha=0.7,
                    label=f'Drifter {drifter_id}')
    
    # Add a colorbar to show what the colors mean
    if 'sst' in ds:
        cbar = plt.colorbar(scatter, shrink=0.8)
        cbar.set_label('Sea Surface Temperature (°C)', fontsize=12)
    
    plt.xlabel('Longitude (degrees East)', fontsize=12)
    plt.ylabel('Latitude (degrees North)', fontsize=12)
    plt.title('Ocean Drifter Trajectories Colored by Temperature\n(Warmer colors = warmer water)', 
              fontsize=14, fontweight='bold')
    
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    
    print("✅ Temperature map created successfully!")

def main():
    """
    Run our trajectory plotting analysis
    """
    
    print("🚀 WELCOME TO DRIFTER TRAJECTORY MAPPING!")
    print("=" * 60)
    
    # Load our data
    dataset = load_sample_drifter_data()
    
    if dataset is not None:
        # Create simple trajectory plot
        create_simple_trajectory_plot(dataset)
        
        # Analyze the trajectories
        analyze_trajectories(dataset)
        
        # Create detailed temperature plot
        create_detailed_trajectory_plot(dataset)
        
        print("\n🎉 MISSION ACCOMPLISHED!")
        print("You've created your first ocean trajectory maps!")
        print("\nWhat you learned:")
        print("- How to plot drifter paths on a map")
        print("- How to use colors to show different drifters")
        print("- How to add temperature information to maps")
        print("- How to analyze drifter movement patterns")
        
        print("\n🔮 Next time:")
        print("- Add real coastlines to your maps")
        print("- Create animations showing drifter movement over time")
        print("- Compare drifter speeds in different regions")

if __name__ == "__main__":
    main()

"""
🤔 REFLECTION QUESTIONS FOR STUDENTS:
====================================

1. Looking at the trajectories, can you guess the direction of ocean 
   currents in this region?

2. Which drifter traveled the farthest? Which stayed in one area?

3. Do you notice any relationship between latitude (north-south position) 
   and temperature?

4. If you were a marine biologist, how might these drifter paths help you 
   understand where fish might travel?

5. What do you think causes some drifters to move in curved paths instead 
   of straight lines?

💡 COOL OCEAN FACTS:
===================
- The Gulf Stream current can move drifters 100+ km per day!
- Ocean currents are like underwater rivers
- Temperature differences help drive ocean circulation
- Some drifters have crossed entire ocean basins
- Drifter data helps track pollution and search for lost ships

🧪 EXPERIMENT IDEAS:
===================
1. Try changing the colors in the script - what looks best?
2. Modify the plot size - what happens with very small or large plots?
3. Add your own annotations to mark interesting features
4. Calculate which drifter moved fastest (distance ÷ time)

🔗 WHAT'S NEXT:
===============
In Script 3, we'll add real coastlines and make our maps look like 
professional oceanographic charts!
"""
