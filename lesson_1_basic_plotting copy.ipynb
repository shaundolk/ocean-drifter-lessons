 
"""
Script 1: Basic Data Loading - Your First Look at Ocean Drifter Data!
==================================================================

What are ocean drifters?
Ocean drifters are floating instruments that drift with ocean currents, 
measuring temperature, location, and speed. They're like GPS trackers for 
the ocean! Scientists release thousands of these into the ocean to 
understand how water moves around our planet.

What this script does:
- Shows you how to load drifter data from the internet
- Teaches you to peek at the data to see what's inside
- Explains the different measurements drifters collect

Learning goals:
- Understand what ocean drifter data looks like
- Learn to load data from online sources
- Practice looking at data structure and contents
"""

# Import the tools we need (these are like importing different toolboxes)
import xarray as xr  # This helps us work with ocean/climate data
import pandas as pd  # This helps us work with tables of data
import numpy as np   # This helps us work with numbers and calculations

def load_sample_drifter_data():
    """
    Load a small sample of drifter data to explore
    
    Think of this like opening a treasure chest of ocean data!
    We're loading data that shows where drifters went and what they measured.
    """
    
    print("🌊 Welcome to Ocean Drifter Data Science! 🌊")
    print("=" * 50)
    
    # This is the web address where NOAA stores drifter data
    # It's like a library catalog number for ocean data
    data_url = "https://www.aoml.noaa.gov/ftp/pub/phod/buoydata/hourly_product/ncei/current/gdp_hourly”.nc
    
    print("📍 Loading drifter data from NOAA...")
    print(f"Data source: {data_url}")
    
    try:
        # Load the data (this might take a moment - we're downloading from the internet!)
        ds = xr.open_zarr(data_url, consolidated=True)
        
        print("✅ Data loaded successfully!")
        print("\n" + "=" * 50)
        
        return ds
        
    except Exception as e:
        print(f"❌ Oops! Couldn't load the data. Error: {e}")
        print("\n🔧 This might happen if:")
        print("   - Your internet connection is slow")
        print("   - The NOAA servers are busy")
        print("   - The data location changed")
        
        # For demonstration, create some sample data
        print("\n📝 Creating sample data for learning...")
        return create_sample_data()

def create_sample_data():
    """
    Create sample drifter data for learning when we can't access real data
    
    This creates fake but realistic drifter data so you can still learn!
    """
    
    # Create sample data that looks like real drifter data
    n_obs = 1000  # Number of observations
    n_ids = 5     # Number of different drifters
    
    # Create time series (hourly data for about 8 days per drifter)
    times = pd.date_range('2023-01-01', periods=n_obs//n_ids, freq='H')
    
    # Sample data structure
    data = {
        'id': np.repeat(range(n_ids), n_obs//n_ids),
        'lon': np.random.normal(-60, 10, n_obs),  # Longitude (Atlantic Ocean)
        'lat': np.random.normal(30, 5, n_obs),    # Latitude (North Atlantic)
        'sst': np.random.normal(20, 3, n_obs),    # Sea Surface Temperature (°C)
        'time': np.tile(times, n_ids)
    }
    
    # Convert to xarray dataset (the format real drifter data uses)
    ds = xr.Dataset({
        'longitude': (['obs'], data['lon']),
        'latitude': (['obs'], data['lat']),
        'sst': (['obs'], data['sst']),
        'id': (['obs'], data['id']),
    }, coords={'time': (['obs'], data['time'])})
    
    print("✅ Sample data created!")
    return ds

def explore_data_structure(ds):
    """
    Look at what's inside our drifter data
    
    This is like opening a box and seeing what's inside - we want to understand
    what information the drifters collected for us!
    """
    
    print("\n🔍 EXPLORING THE DATA STRUCTURE")
    print("=" * 50)
    
    print("📊 Overall data information:")
    print(f"   - Data type: {type(ds)}")
    print(f"   - Number of observations: {ds.dims.get('obs', 'Unknown')}")
    
    print("\n📈 What measurements do we have?")
    print("   Variables (things the drifters measured):")
    for var_name in ds.data_vars:
        var = ds[var_name]
        print(f"   - {var_name}: {var.attrs.get('long_name', 'No description')}")
        print(f"     Units: {var.attrs.get('units', 'No units specified')}")
    
    print("\n📍 Coordinate information:")
    print("   Coordinates (how the data is organized):")
    for coord_name in ds.coords:
        coord = ds[coord_name]
        print(f"   - {coord_name}: {coord.attrs.get('long_name', coord_name)}")

def peek_at_data_values(ds):
    """
    Look at actual numbers in our data
    
    Now let's see the actual measurements! This is like reading the first 
    few pages of an exciting book about the ocean.
    """
    
    print("\n👀 LOOKING AT ACTUAL DATA VALUES")
    print("=" * 50)
    
    # Show first few data points
    print("📋 First 10 observations:")
    
    try:
        # Try to show real data structure
        if 'longitude' in ds:
            sample_data = ds.isel(obs=slice(0, 10))  # Get first 10 observations
            
            print("Position and Temperature Data:")
            print(f"{'Index':<8} {'Longitude':<12} {'Latitude':<12} {'Temperature':<12} {'Drifter ID':<12}")
            print("-" * 60)
            
            for i in range(min(10, len(sample_data.obs))):
                lon = float(sample_data.longitude.isel(obs=i).values)
                lat = float(sample_data.latitude.isel(obs=i).values)
                temp = float(sample_data.sst.isel(obs=i).values) if 'sst' in sample_data else 'N/A'
                drifter_id = int(sample_data.id.isel(obs=i).values) if 'id' in sample_data else 'N/A'
                
                print(f"{i:<8} {lon:<12.3f} {lat:<12.3f} {temp:<12.1f} {drifter_id:<12}")
                
    except Exception as e:
        print(f"Couldn't display sample data: {e}")
        print("The data structure might be different than expected.")

def data_summary_stats(ds):
    """
    Calculate basic statistics about our data
    
    This gives us the "big picture" - like asking "what's the warmest 
    temperature recorded?" or "how far north did the drifters go?"
    """
    
    print("\n📊 DATA SUMMARY STATISTICS")
    print("=" * 50)
    
    try:
        if 'longitude' in ds:
            print("🌐 Geographic Range (where the drifters traveled):")
            print(f"   Longitude: {float(ds.longitude.min().values):.2f}° to {float(ds.longitude.max().values):.2f}°")
            print(f"   Latitude:  {float(ds.latitude.min().values):.2f}° to {float(ds.latitude.max().values):.2f}°")
        
        if 'sst' in ds:
            print("\n🌡️  Temperature Information:")
            print(f"   Coldest water: {float(ds.sst.min().values):.1f}°C")
            print(f"   Warmest water: {float(ds.sst.max().values):.1f}°C")
            print(f"   Average temperature: {float(ds.sst.mean().values):.1f}°C")
        
        if 'id' in ds:
            unique_drifters = len(np.unique(ds.id.values))
            print(f"\n🎯 Number of different drifters: {unique_drifters}")
            
    except Exception as e:
        print(f"Couldn't calculate statistics: {e}")

def main():
    """
    Main function that runs our data exploration
    
    This is like the table of contents - it runs all our functions in order
    to give you a complete introduction to drifter data!
    """
    
    # Step 1: Load the data
    dataset = load_sample_drifter_data()
    
    if dataset is not None:
        # Step 2: Explore what's in the data
        explore_data_structure(dataset)
        
        # Step 3: Look at actual values
        peek_at_data_values(dataset)
        
        # Step 4: Calculate summary statistics
        data_summary_stats(dataset)
        
        print("\n🎉 CONGRATULATIONS!")
        print("You've completed your first look at ocean drifter data!")
        print("\nNext steps:")
        print("- Try running the next script to make your first plot")
        print("- Think about what questions you'd like to ask about the ocean")
        print("- Imagine where these drifters might have traveled!")
        
    else:
        print("❌ Couldn't load data for exploration.")

# This runs our main function when the script is executed
if __name__ == "__main__":
    main()

"""
🤔 REFLECTION QUESTIONS FOR STUDENTS:
====================================

1. What surprised you most about the drifter data?

2. Looking at the temperature range, what season do you think this data 
   was collected? Why?

3. If you were a drifter floating in the ocean, what would you want to 
   measure besides temperature and location?

4. How do you think scientists use this data to understand ocean currents?

5. What questions would you like to ask about the ocean that this data 
   might help answer?

💡 FUN FACTS:
=============
- Ocean drifters can travel thousands of miles!
- Some drifters last for years, sending data every hour
- The Global Drifter Program maintains over 1,300 active drifters
- Drifter data helps improve weather forecasting
- Ocean currents transport heat around the planet, affecting climate

🔗 WHAT'S NEXT:
===============
In the next script, we'll create our first map showing where these 
drifters traveled. Get ready to see the ocean come alive with data!
"""
