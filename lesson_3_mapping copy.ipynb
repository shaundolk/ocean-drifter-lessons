"""
Script 3: Adding Coastlines - Making Professional Ocean Maps!
============================================================

What you'll learn:
Real oceanographers don't just plot data in empty space - they add coastlines,
country borders, and geographic features to make maps that tell a story!

This script teaches you to create maps that look like they belong in a 
scientific journal or textbook.

What this script does:
- Adds realistic coastlines to your drifter maps
- Shows country borders and major geographic features
- Demonstrates different map projections (ways to show Earth on flat paper)
- Creates publication-quality figures

Learning goals:
- Use Cartopy library for professional map-making
- Understand map projections and coordinate systems
- Add geographic context to oceanographic data
- Create maps worthy of scientific presentations

Note: This script requires the 'cartopy' library. If you get import errors,
you can install it with: pip install cartopy
"""

# Import our mapping tools
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime, timedelta

# Professional mapping tools
try:
    import cartopy.crs as ccrs           # Coordinate reference systems
    import cartopy.feature as cfeature   # Geographic features (coastlines, etc.)
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    CARTOPY_AVAILABLE = True
    print("✅ Cartopy mapping library loaded successfully!")
except ImportError:
    CARTOPY_AVAILABLE = False
    print("⚠️  Cartopy not available - we'll use basic matplotlib instead")
    print("   To get professional maps, install cartopy: pip install cartopy")

def load_atlantic_drifter_data():
    """
    Create realistic Atlantic Ocean drifter data for demonstration
    
    We'll simulate drifters in recognizable parts of the Atlantic where
    coastlines will provide meaningful context.
    """
    
    print("🌊 Creating Atlantic Ocean drifter data...")
    
    # Define realistic drifter scenarios in the Atlantic
    drifter_scenarios = [
        {
            "name": "Gulf Stream Explorer",
            "start_lat": 35.0, "start_lon": -75.0,  # Off North Carolina
            "drift_lat": 0.05, "drift_lon": 0.08,   # Northeast drift
            "temp_base": 22, "temp_var": 3
        },
        {
            "name": "Caribbean Current",
            "start_lat": 18.0, "start_lon": -65.0,  # Near Puerto Rico
            "drift_lat": 0.02, "drift_lon": 0.06,   # Northwest drift
            "temp_base": 27, "temp_var": 2
        },
        {
            "name": "Sargasso Sea Wanderer",
            "start_lat": 32.0, "start_lon": -60.0,  # Central Sargasso Sea
            "drift_lat": 0.01, "drift_lon": 0.02,   # Slow circular motion
            "temp_base": 24, "temp_var": 2.5
        },
        {
            "name": "Labrador Current",
            "start_lat": 50.0, "start_lon": -50.0,  # Labrador Sea
            "drift_lat": -0.03, "drift_lon": 0.04,  # Southeast drift
            "temp_base": 8, "temp_var": 4
        },
        {
            "name": "Canary Current",
            "start_lat": 28.0, "start_lon": -20.0,  # Off West Africa
            "drift_lat": -0.02, "drift_lon": -0.03, # Southwest drift
            "temp_base": 20, "temp_var": 3
        }
    ]
    
    n_points_per_drifter = 200  # About 8 days of hourly data
    all_data = []
    
    for i, scenario in enumerate(drifter_scenarios):
        # Create realistic trajectory
        t = np.linspace(0, 2*np.pi, n_points_per_drifter)
        
        # Base drift
        lat_drift = np.cumsum(np.full(n_points_per_drifter, scenario["drift_lat"]))
        lon_drift = np.cumsum(np.full(n_points_per_drifter, scenario["drift_lon"]))
        
        # Add some randomness and wave-like motion
        lat_noise = np.cumsum(np.random.normal(0, 0.005, n_points_per_drifter))
        lon_noise = np.cumsum(np.random.normal(0, 0.005, n_points_per_drifter))
        
        # Wave motion (ocean eddies and meanders)
        lat_wave = 0.3 * np.sin(t * 0.7 + i) + 0.2 * np.cos(t * 1.2)
        lon_wave = 0.4 * np.cos(t * 0.8 + i) + 0.15 * np.sin(t * 1.5)
        
        # Final positions
        final_lat = scenario["start_lat"] + lat_drift + lat_noise + lat_wave
        final_lon = scenario["start_lon"] + lon_drift + lon_noise + lon_wave
        
        # Temperature based on latitude and scenario
        base_temp = scenario["temp_base"] - (final_lat - scenario["start_lat"]) * 0.3
        temperatures = base_temp + np.random.normal(0, scenario["temp_var"], n_points_per_drifter)
        
        # Create time series
        start_time = datetime(2023, 7, 1) + timedelta(days=i*2)
        times = [start_time + timedelta(hours=h) for h in range(n_points_per_drifter)]
        
        # Store data
        drifter_data = {
            'id': [i] * n_points_per_drifter,
            'latitude': final_lat,
            'longitude': final_lon,
            'sst': temperatures,
            'time': times,
            'name': [scenario["name"]] * n_points_per_drifter
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
    
    print(f"✅ Created {len(drifter_scenarios)} realistic Atlantic drifter trajectories")
    return ds

def create_basic_coastline_map(ds):
    """
    Create a map with coastlines using basic matplotlib
    (for when Cartopy is not available)
    """
    
    print("\n🗺️  CREATING BASIC MAP WITH COASTLINE APPROXIMATION")
    print("=" * 55)
    
    plt.figure(figsize=(14, 10))
    
    # Plot drifter trajectories
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    unique_ids = np.unique(ds.id.values)
    
    for i, drifter_id in enumerate(unique_ids):
        drifter_data = ds.where(ds.id == drifter_id, drop=True)
        
        lats = drifter_data.latitude.values
        lons = drifter_data.longitude.values
        
        valid_mask = ~(np.isnan(lats) | np.isnan(lons))
        lats = lats[valid_mask]
        lons = lons[valid_mask]
        
        if len(lats) > 0:
            plt.plot(lons, lats, 
                    color=colors[i % len(colors)], 
                    linewidth=2.5, 
                    alpha=0.8,
                    label=f'Drifter {drifter_id}')
            
            # Mark start and end
            plt.plot(lons[0], lats[0], 'o', 
                    color=colors[i % len(colors)], 
                    markersize=8, 
                    markeredgecolor='black')
            plt.plot(lons[-1], lats[-1], 's', 
                    color=colors[i % len(colors)], 
                    markersize=8, 
                    markeredgecolor='black')
    
    # Add simple coastline approximations (major features)
    # East Coast of North America
    us_coast_lat = [25, 30, 35, 40, 45, 50]
    us_coast_lon = [-80, -81, -75, -70, -65, -60]
    plt.plot(us_coast_lon, us_coast_lat, 'k-', linewidth=2, label='Coastlines (approx.)')
    
    # Caribbean islands (simplified)
    caribbean_lat = [18, 20, 22, 24]
    caribbean_lon = [-65, -70, -75, -78]
    plt.plot(caribbean_lon, caribbean_lat, 'k-', linewidth=2)
    
    # West Africa coast (simplified)
    africa_coast_lat = [10, 20, 30, 35]
    africa_coast_lon = [-15, -17, -10, -6]
    plt.plot(africa_coast_lon, africa_coast_lat, 'k-', linewidth=2)
    
    plt.xlabel('Longitude (degrees East)', fontsize=12)
    plt.ylabel('Latitude (degrees North)', fontsize=12)
    plt.title('Atlantic Ocean Drifter Trajectories\n(Basic coastline approximation)', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    
    print("✅ Basic coastline map created!")
    print("💡 For professional maps with accurate coastlines, install Cartopy")

def create_professional_map(ds):
    """
    Create a professional map with accurate coastlines using Cartopy
    """
    
    if not CARTOPY_AVAILABLE:
        print("⚠️  Cartopy not available - skipping professional map")
        return
    
    print("\n🌍 CREATING PROFESSIONAL CARTOGRAPHIC MAP")
    print("=" * 50)
    
    # Set up the map projection
    # PlateCarree is like a flat world map - good for showing large ocean areas
    fig = plt.figure(figsize=(16, 12))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Define the map extent (boundaries)
    # This focuses on the North Atlantic where our drifters are
    ax.set_global()  # Start with global view
    
    # Get data extent to focus the map
    all_lats = ds.latitude.values[~np.isnan(ds.latitude.values)]
    all_lons = ds.longitude.values[~np.isnan(ds.longitude.values)]
    
    if len(all_lats) > 0 and len(all_lons) > 0:
        lat_margin = (max(all_lats) - min(all_lats)) * 0.2
        lon_margin = (max(all_lons) - min(all_lons)) * 0.2
        
        ax.set_extent([min(all_lons) - lon_margin, max(all_lons) + lon_margin,
                      min(all_lats) - lat_margin, max(all_lats) + lat_margin],
                     crs=ccrs.PlateCarree())
    
    # Add geographic features
    print("   📍 Adding coastlines...")
    ax.add_feature(cfeature.COASTLINE, linewidth=1.5, color='black')
    
    print("   🏛️  Adding country borders...")
    ax.add_feature(cfeature.BORDERS, linewidth=1, color='gray', alpha=0.7)
    
    print("   🏔️  Adding land areas...")
    ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.5)
    
    print("   🌊 Adding ocean areas...")
    ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
    
    print("   🏞️  Adding lakes and rivers...")
    ax.add_feature(cfeature.LAKES, color='lightblue', alpha=0.5)
    ax.add_feature(cfeature.RIVERS, color='blue', alpha=0.7)
    
    # Plot drifter trajectories
    colors = ['red', 'darkblue', 'green', 'orange', 'purple']
    unique_ids = np.unique(ds.id.values)
    
    print("   🎯 Plotting drifter trajectories...")
    
    for i, drifter_id in enumerate(unique_ids):
        drifter_data = ds.where(ds.id == drifter_id, drop=True)
        
        lats = drifter_data.latitude.values
        lons = drifter_data.longitude.values
        temps = drifter_data.sst.values
        
        valid_mask = ~(np.isnan(lats) | np.isnan(lons))
        lats = lats[valid_mask]
        lons = lons[valid_mask]
        temps = temps[valid_mask] if not np.all(np.isnan(temps)) else None
        
        if len(lats) > 1:
            # Plot trajectory line
            ax.plot(lons, lats, 
                   color=colors[i % len(colors)], 
                   linewidth=3, 
                   alpha=0.8,
                   label=f'Drifter {drifter_id}',
                   transform=ccrs.PlateCarree())
            
            # Mark start point (circle)
            ax.plot(lons[0], lats[0], 'o', 
                   color=colors[i % len(colors)], 
                   markersize=10, 
                   markeredgecolor='black',
                   markeredgewidth=2,
                   transform=ccrs.PlateCarree())
            
            # Mark end point (square)
            ax.plot(lons[-1], lats[-1], 's', 
                   color=colors[i % len(colors)], 
                   markersize=10, 
                   markeredgecolor='black',
                   markeredgewidth=2,
                   transform=ccrs.PlateCarree())
    
    # Add gridlines with labels
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}
    
    # Add title and legend
    ax.set_title('Atlantic Ocean Drifter Trajectories\nwith Geographic Context', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Create custom legend
    legend = ax.legend(loc='upper left', frameon=True, fancybox=True, 
                      shadow=True, fontsize=10)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Professional map created successfully!")

def create_temperature_context_map(ds):
    """
    Create a map showing temperature with geographic context
    """
    
    if not CARTOPY_AVAILABLE:
        print("⚠️  Skipping temperature context map (requires Cartopy)")
        return
    
    print("\n🌡️  CREATING TEMPERATURE MAP WITH GEOGRAPHIC CONTEXT")
    print("=" * 55)
    
    fig = plt.figure(figsize=(16, 12))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Set extent based on data
    all_lats = ds.latitude.values[~np.isnan(ds.latitude.values)]
    all_lons = ds.longitude.values[~np.isnan(ds.longitude.values)]
    
    if len(all_lats) > 0:
        lat_margin = (max(all_lats) - min(all_lats)) * 0.15
        lon_margin = (max(all_lons) - min(all_lons)) * 0.15
        
        ax.set_extent([min(all_lons) - lon_margin, max(all_lons) + lon_margin,
                      min(all_lats) - lat_margin, max(all_lats) + lat_margin],
                     crs=ccrs.PlateCarree())
    
    # Add geographic features
    ax.add_feature(cfeature.COASTLINE, linewidth=1.5, color='black')
    ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.7)
    ax.add_feature(cfeature.OCEAN, color='white', alpha=0.3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.8, color='gray', alpha=0.6)
    
    # Plot temperature data
    unique_ids = np.unique(ds.id.values)
    
    # Collect all temperature data for colorbar scaling
    all_temps = []
    
    for drifter_id in unique_ids:
        drifter_data = ds.where(ds.id == drifter_id, drop=True)
        temps = drifter_data.sst.values
        valid_temps = temps[~np.isnan(temps)]
        all_temps.extend(valid_temps)
    
    if len(all_temps) > 0:
        temp_min, temp_max = min(all_temps), max(all_temps)
        
        for i, drifter_id in enumerate(unique_ids):
            drifter_data = ds.where(ds.id == drifter_id, drop=True)
            
            lats = drifter_data.latitude.values
            lons = drifter_data.longitude.values
            temps = drifter_data.sst.values
            
            valid_mask = ~(np.isnan(lats) | np.isnan(lons) | np.isnan(temps))
            lats = lats[valid_mask]
            lons = lons[valid_mask]
            temps = temps[valid_mask]
            
            if len(lats) > 0:
                # Create scatter plot colored by temperature
                scatter = ax.scatter(lons, lats, 
                                   c=temps, 
                                   cmap='RdYlBu_r',  # Red-Yellow-Blue (reversed)
                                   s=60, 
                                   alpha=0.8,
                                   vmin=temp_min, vmax=temp_max,
                                   edgecolors='black',
                                   linewidth=0.5,
                                   transform=ccrs.PlateCarree())
                
                # Connect points with thin lines
                ax.plot(lons, lats, 
                       color='gray', 
                       linewidth=1, 
                       alpha=0.4,
                       transform=ccrs.PlateCarree())
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.05)
        cbar.set_label('Sea Surface Temperature (°C)', fontsize=12, fontweight='bold')
        cbar.ax.tick_params(labelsize=10)
    
    # Add gridlines
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.8, color='gray', alpha=0.5, linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    
    ax.set_title('Sea Surface Temperature from Ocean Drifters\nwith Geographic Context', 
                fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Temperature context map created!")

def analyze_geographic_context(ds):
    """
    Analyze drifter data in geographic context
    """
    
    print("\n🔍 ANALYZING DRIFTERS IN GEOGRAPHIC CONTEXT")
    print("=" * 50)
    
    unique_ids = np.unique(ds.id.values)
    
    # Define major ocean regions (simplified)
    regions = {
        "Gulf Stream Region": {"lat_range": (35, 42), "lon_range": (-80, -65)},
        "Sargasso Sea": {"lat_range": (25, 35), "lon_range": (-70, -40)},
        "Caribbean": {"lat_range": (10, 25), "lon_range": (-85, -60)},
        "Labrador Sea": {"lat_range": (45, 60), "lon_range": (-65, -45)},
        "Eastern Atlantic": {"lat_range": (20, 40), "lon_range": (-30, -10)}
    }
    
    for drifter_id in unique_ids:
        drifter_data = ds.where(ds.id == drifter_id, drop=True)
        
        lats = drifter_data.latitude.values
        lons = drifter_data.longitude.values
        temps = drifter_data.sst.values
        
        valid_mask = ~(np.isnan(lats) | np.isnan(lons))
        lats = lats[valid_mask]
        lons = lons[valid_mask]
        temps = temps[valid_mask] if not np.all(np.isnan(temps)) else None
        
        if len(lats) > 0:
            print(f"\n🎯 Drifter {drifter_id} Geographic Analysis:")
            
            # Check which regions the drifter visited
            regions_visited = []
            for region_name, bounds in regions.items():
                lat_in_range = np.any((lats >= bounds["lat_range"][0]) & 
                                     (lats <= bounds["lat_range"][1]))
                lon_in_range = np.any((lons >= bounds["lon_range"][0]) & 
                                     (lons <= bounds["lon_range"][1]))
                
                if lat_in_range and lon_in_range:
                    regions_visited.append(region_name)
            
            print(f"   Regions visited: {', '.join(regions_visited) if regions_visited else 'Open ocean'}")
            print(f"   Latitude range: {min(lats):.2f}° to {max(lats):.2f}°N")
            print(f"   Longitude range: {min(lons):.2f}° to {max(lons):.2f}°E")
            
            if temps is not None and len(temps) > 0:
                print(f"   Temperature range: {min(temps):.1f}° to {max(temps):.1f}°C")
                print(f"   Average temperature: {np.mean(temps):.1f}°C")

def main():
    """
    Run the coastline mapping demonstration
    """
    
    print("🌍 WELCOME TO PROFESSIONAL OCEAN MAPPING!")
    print("=" * 60)
    
    # Load Atlantic drifter data
    dataset = load_atlantic_drifter_data()
    
    if dataset is not None:
        # Create basic map (always works)
        create_basic_coastline_map(dataset)
        
        # Create professional map (if Cartopy available)
        create_professional_map(dataset)
        
        # Create temperature context map (if Cartopy available)
        create_temperature_context_map(dataset)
        
        # Analyze geographic context
        analyze_geographic_context(dataset)
        
        print("\n🎉 MAPPING MISSION COMPLETE!")
        print("\nWhat you accomplished:")
        print("✓ Created maps with geographic context")
        print("✓ Added realistic coastlines and borders")
        print("✓ Visualized temperature in geographic context")
        print("✓ Analyzed drifter movements by region")
        
        if CARTOPY_AVAILABLE:
            print("\n🌟 Professional cartographic features used!")
        else:
            print("\n💡 Install Cartopy for even better maps: pip install cartopy")
        
        print("\n🔮 Coming up next:")
        print("- Animate drifter movements over time")
        print("- Calculate and visualize current speeds")
        print("- Compare seasonal temperature patterns")

if __name__ == "__main__":
    main()

"""
🤔 REFLECTION QUESTIONS FOR STUDENTS:
====================================

1. How does adding coastlines change your interpretation of the drifter paths?

2. Can you identify any drifters that might be following known ocean currents 
   like the Gulf Stream?

3. What do you notice about temperature patterns near coastlines versus 
   open ocean?

4. If you were planning a sailing trip, how might drifter data help you 
   choose your route?

5. Why might some drifters stay close to coastlines while others venture 
   into open ocean?

💡 MAPPING SCIENCE FACTS:
========================
- Map projections are different ways to show Earth's curved surface on flat paper
- The Mercator projection (common in web maps) distorts area near the poles
- PlateCarree projection preserves angles and is good for global ocean data
- Professional oceanographers use multiple projections depending on their region
- Satellite data and drifter data work together to map ocean currents

🔧 TECHNICAL SKILLS LEARNED:
===========================
- Using Cartopy for professional map projections
- Adding geographic features (coastlines, borders, land/ocean)
- Coordinate reference systems and transformations  
- Creating publication-quality scientific figures
- Combining point data with geographic context

🧪 EXPERIMENT IDEAS:
===================
1. Try different map projections (Mollweide, Robinson, etc.)
2. Focus on smaller regions for more detailed analysis
3. Add city labels or other geographic markers
4. Compare your drifter paths to known ocean current maps
5. Create maps showing only specific time periods

🔗 WHAT'S NEXT:
===============
In Script 4, we'll bring our maps to life with animations showing how 
drifters move through time - like creating a movie of ocean currents!
"""
