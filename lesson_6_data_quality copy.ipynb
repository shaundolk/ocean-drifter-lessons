 
#!/usr/bin/env python3
"""
LESSON 6: DATA QUALITY AND VALIDATION
====================================

Learning Goals:
- Understand why data quality matters in science
- Learn to identify problems in real data
- Practice cleaning and validating oceanographic data
- Develop critical thinking about data reliability

What you'll learn about real-world data:
- All real data has errors and missing values
- Sensors can malfunction or drift out of calibration
- Environmental conditions can affect data quality
- Scientists must validate data before drawing conclusions

This is one of the most important skills in data science!
Bad data leads to bad conclusions, which can affect policy,
research, and even lives. Let's learn to be data detectives! 🕵️

Prerequisites: Complete Lessons 1-5 first!
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def create_realistic_messy_data():
    """
    Create a dataset with realistic problems that occur in real oceanographic data.
    
    Real data is messy! This simulates common issues scientists face.
    """
    print("🔧 Creating realistic messy oceanographic dataset...")
    print("(In real life, this messiness comes from sensor failures, harsh ocean conditions, etc.)")
    
    # Start with clean data
    hours = 30 * 24  # 30 days of hourly data
    times = pd.date_range('2023-06-01', periods=hours, freq='H')
    
    # Create a realistic drifter track
    lats = 35.0 + 0.02 * np.arange(hours) + 2 * np.sin(2 * np.pi * np.arange(hours) / (7 * 24))
    lons = -70.0 + 0.03 * np.arange(hours) + 1.5 * np.cos(2 * np.pi * np.arange(hours) / (5 * 24))
    
    # Create realistic temperature data
    base_temp = 22 + 3 * np.sin(2 * np.pi * np.arange(hours) / (365 * 24))  # Seasonal
    daily_temp = 1.5 * np.sin(2 * np.pi * np.arange(hours) / 24)  # Daily cycle
    noise = np.random.normal(0, 0.3, hours)
    clean_temps = base_temp + daily_temp + noise
    
    # Start with clean data
    data = pd.DataFrame({
        'time': times,
        'latitude': lats,
        'longitude': lons,
        'sst': clean_temps,
        'drifter_id': 34567
    })
    
    # Now add realistic problems that occur in real data!
    print("\n🚨 Adding realistic data problems:")
    
    # Problem 1: Missing data (very common!)
    print("   • Adding missing data gaps (sensor power failures)")
    missing_indices = np.random.choice(len(data), size=int(0.05 * len(data)), replace=False)
    data.loc[missing_indices, 'sst'] = np.nan
    
    # Create a longer gap (like when a drifter goes underwater)
    gap_start = len(data) // 3
    gap_end = gap_start + 48  # 2-day gap
    data.loc[gap_start:gap_end, ['latitude', 'longitude']] = np.nan
    
    # Problem 2: Outliers (sensor spikes)
    print("   • Adding outlier values (sensor malfunctions)")
    outlier_indices = np.random.choice(len(data), size=8, replace=False)
    for idx in outlier_indices:
        if np.random.random() > 0.5:
            data.loc[idx, 'sst'] = data.loc[idx, 'sst'] + np.random.uniform(8, 15)  # Hot spike
        else:
            data.loc[idx, 'sst'] = data.loc[idx, 'sst'] - np.random.uniform(5, 10)  # Cold spike
    
    # Problem 3: Sensor drift (gradual calibration error)
    print("   • Adding sensor drift (gradual calibration error)")
    drift_start = len(data) // 2
    drift_amount = np.linspace(0, 2.5, len(data) - drift_start)  # Gradual warming bias
    data.loc[drift_start:, 'sst'] += drift_amount
    
    # Problem 4: Impossible values
    print("   • Adding impossible values (data transmission errors)")
    impossible_indices = np.random.choice(len(data), size=3, replace=False)
    data.loc[impossible_indices, 'sst'] = [-999, 99.9, 150]  # Common error codes and impossible values
    
    # Problem 5: Stuck sensor (repeated values)
    print("   • Adding stuck sensor values (frozen readings)")
    stuck_start = int(0.7 * len(data))
    stuck_end = stuck_start + 12  # 12 hours stuck
    stuck_value = data.loc[stuck_start, 'sst']
    data.loc[stuck_start:stuck_end, 'sst'] = stuck_value
    
    # Problem 6: Date/time errors
    print("   • Adding timestamp errors (GPS/clock issues)")
    # Duplicate some timestamps
    duplicate_idx = len(data) // 4
    data.loc[duplicate_idx, 'time'] = data.loc[duplicate_idx - 1, 'time']
    
    print(f"\n✅ Created messy dataset with {len(data)} records")
    print(f"📊 Now has realistic problems that scientists encounter daily!")
    
    return data

def identify_data_problems(data):
    """
    Systematically identify problems in the dataset.
    
    This teaches students to be data detectives!
    """
    print("\n🕵️ DETECTIVE MODE: Let's find all the data problems!")
    print("=" * 50)
    
    problems_found = []
    
    # Check 1: Missing data
    print("\n🔍 CHECK 1: Looking for missing data...")
    missing_counts = data.isnull().sum()
    total_records = len(data)
    
    for column in missing_counts.index:
        if missing_counts[column] > 0:
            percent_missing = (missing_counts[column] / total_records) * 100
            print(f"   ❌ {column}: {missing_counts[column]} missing values ({percent_missing:.1f}%)")
            problems_found.append(f"Missing data in {column}")
        else:
            print(f"   ✅ {column}: No missing values")
    
    # Check 2: Impossible values
    print("\n🔍 CHECK 2: Looking for impossible values...")
    
    # Temperature should be between -2°C and 35°C for surface ocean
    temp_problems = data[(data['sst'] < -2) | (data['sst'] > 35) | (np.abs(data['sst']) > 100)]
    if len(temp_problems) > 0:
        print(f"   ❌ Found {len(temp_problems)} impossible temperature values:")
        for idx, row in temp_problems.iterrows():
            print(f"      • {row['sst']:.1f}°C at {row['time']}")
        problems_found.append("Impossible temperature values")
    else:
        print("   ✅ All temperature values are physically reasonable")
    
    # Latitude should be between -90 and 90
    lat_problems = data[(data['latitude'] < -90) | (data['latitude'] > 90)]
    if len(lat_problems) > 0:
        print(f"   ❌ Found {len(lat_problems)} impossible latitude values")
        problems_found.append("Impossible latitude values")
    else:
        print("   ✅ All latitude values are valid")
    
    # Longitude should be between -180 and 180
    lon_problems = data[(data['longitude'] < -180) | (data['longitude'] > 180)]
    if len(lon_problems) > 0:
        print(f"   ❌ Found {len(lon_problems)} impossible longitude values")
        problems_found.append("Impossible longitude values")
    else:
        print("   ✅ All longitude values are valid")
    
    # Check 3: Outliers (using statistical methods)
    print("\n🔍 CHECK 3: Looking for statistical outliers...")
    
    # Use interquartile range (IQR) method for temperature
    Q1 = data['sst'].quantile(0.25)
    Q3 = data['sst'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    temp_outliers = data[(data['sst'] < lower_bound) | (data['sst'] > upper_bound)]
    temp_outliers = temp_outliers.dropna(subset=['sst'])  # Remove NaN values
    
    if len(temp_outliers) > 0:
        print(f"   ❌ Found {len(temp_outliers)} temperature outliers:")
        print(f"      Normal range: {lower_bound:.1f}°C to {upper_bound:.1f}°C")
        for idx, row in temp_outliers.head(3).iterrows():  # Show first 3
            print(f"      • {row['sst']:.1f}°C at {row['time']}")
        if len(temp_outliers) > 3:
            print(f"      ... and {len(temp_outliers) - 3} more")
        problems_found.append("Statistical outliers in temperature")
    else:
        print("   ✅ No statistical outliers found in temperature")
    
    # Check 4: Duplicate timestamps
    print("\n🔍 CHECK 4: Looking for duplicate timestamps...")
    duplicate_times = data[data['time'].duplicated()]
    if len(duplicate_times) > 0:
        print(f"   ❌ Found {len(duplicate_times)} duplicate timestamps")
        for idx, row in duplicate_times.head(3).iterrows():
            print(f"      • Duplicate time: {row['time']}")
        problems_found.append("Duplicate timestamps")
    else:
        print("   ✅ No duplicate timestamps found")
    
    # Check 5: Stuck sensor (repeated values)
    print("\n🔍 CHECK 5: Looking for stuck sensors (repeated values)...")
    
    # Look for sequences of identical values
    temp_diff = data['sst'].diff().abs()
    stuck_threshold = 0.001  # Values closer than this are considered "stuck"
    consecutive_stuck = 0
    max_stuck = 0
    stuck_sequences = []
    
    for i, diff in enumerate(temp_diff):
        if pd.notna(diff) and diff < stuck_threshold:
            consecutive_stuck += 1
        else:
            if consecutive_stuck > max_stuck:
                max_stuck = consecutive_stuck
            if consecutive_stuck >= 5:  # 5+ consecutive identical values
                stuck_sequences.append((i - consecutive_stuck, i, consecutive_stuck))
            consecutive_stuck = 0
    
    if stuck_sequences:
        print(f"   ❌ Found {len(stuck_sequences)} stuck sensor sequences:")
        for start_idx, end_idx, length in stuck_sequences[:3]:
            print(f"      • {length} identical values starting at {data.loc[start_idx, 'time']}")
        problems_found.append("Stuck sensor readings")
    else:
        print("   ✅ No stuck sensor readings detected")
    
    # Check 6: Unrealistic rates of change
    print("\n🔍 CHECK 6: Looking for unrealistic rates of change...")
    
    # Temperature shouldn't change more than 5°C per hour in surface ocean
    temp_change = data['sst'].diff().abs()
    large_changes = data[temp_change > 5]
    large_changes = large_changes.dropna(subset=['sst'])
    
    if len(large_changes) > 0:
        print(f"   ❌ Found {len(large_changes)} unrealistic temperature changes (>5°C/hour):")
        for idx, row in large_changes.head(3).iterrows():
            prev_temp = data.loc[idx-1, 'sst'] if idx > 0 else np.nan
            change = row['sst'] - prev_temp if pd.notna(prev_temp) else np.nan
            print(f"      • {change:.1f}°C change at {row['time']}")
        problems_found.append("Unrealistic temperature changes")
    else:
        print("   ✅ All temperature changes are realistic")
    
    # Summary
    print("\n" + "=" * 50)
    print(f"🎯 DETECTION SUMMARY: Found {len(problems_found)} types of problems")
    for i, problem in enumerate(problems_found, 1):
        print(f"   {i}. {problem}")
    
    if not problems_found:
        print("   🎉 Congratulations! This dataset is clean!")
    else:
        print(f"\n🔧 Time for data cleaning! Let's fix these problems.")
    
    return problems_found

def clean_data_step_by_step(data):
    """
    Clean the data step by step, explaining each decision.
    
    This teaches students the decision-making process in data cleaning.
    """
    print("\n🧹 STEP-BY-STEP DATA CLEANING")
    print("=" * 50)
    print("Real scientists must make careful decisions about how to handle each problem.")
    print("Let's work through this systematically!\n")
    
    # Make a copy so we don't modify the original messy data
    clean_data = data.copy()
    cleaning_log = []
    
    print("CLEANING STEP 1: Remove impossible values")
    print("Decision rule: Remove temperatures outside physical limits (-2°C to 35°C)")
    
    # Remove impossible temperatures
    impossible_temp = (clean_data['sst'] < -2) | (clean_data['sst'] > 35) | (np.abs(clean_data['sst']) > 100)
    n_impossible = impossible_temp.sum()
    
    if n_impossible > 0:
        print(f"   🗑️  Removing {n_impossible} impossible temperature values")
        clean_data.loc[impossible_temp, 'sst'] = np.nan
        cleaning_log.append(f"Removed {n_impossible} impossible temperature values")
    else:
        print("   ✅ No impossible values to remove")
    
    print("\nCLEANING STEP 2: Handle statistical outliers")
    print("Decision rule: Flag extreme outliers but keep moderate ones (they might be real!)")
    
    # Use a more conservative outlier detection (3 standard deviations)
    temp_mean = clean_data['sst'].mean()
    temp_std = clean_data['sst'].std()
    extreme_outliers = (np.abs(clean_data['sst'] - temp_mean) > 3 * temp_std)
    n_extreme = extreme_outliers.sum()
    
    if n_extreme > 0:
        print(f"   🚨 Flagging {n_extreme} extreme outliers (>3σ from mean)")
        clean_data.loc[extreme_outliers, 'sst'] = np.nan
        cleaning_log.append(f"Removed {n_extreme} extreme outliers")
    else:
        print("   ✅ No extreme outliers to flag")
    
    print("\nCLEANING STEP 3: Fix stuck sensor readings")
    print("Decision rule: Replace sequences of 5+ identical values with NaN")
    
    # Find and replace stuck values
    temp_values = clean_data['sst'].values
    stuck_count = 0
    
    for i in range(1, len(temp_values) - 4):
        if (pd.notna(temp_values[i]) and 
            np.all(np.abs(temp_values[i:i+5] - temp_values[i]) < 0.001)):
            # Found 5 consecutive identical values
            temp_values[i:i+5] = np.nan
            stuck_count += 5
    
    clean_data['sst'] = temp_values
    
    if stuck_count > 0:
        print(f"   🔧 Removed {stuck_count} stuck sensor readings")
        cleaning_log.append(f"Removed {stuck_count} stuck sensor readings")
    else:
        print("   ✅ No stuck sensor readings found")
    
    print("\nCLEANING STEP 4: Handle duplicate timestamps")
    print("Decision rule: Keep first occurrence, remove duplicates")
    
    n_duplicates = clean_data['time'].duplicated().sum()
    if n_duplicates > 0:
        print(f"   🗑️  Removing {n_duplicates} duplicate timestamps")
        clean_data = clean_data.drop_duplicates(subset=['time'], keep='first')
        cleaning_log.append(f"Removed {n_duplicates} duplicate timestamps")
    else:
        print("   ✅ No duplicate timestamps to remove")
    
    print("\nCLEANING STEP 5: Interpolate small gaps")
    print("Decision rule: Interpolate gaps ≤ 6 hours, leave larger gaps as missing")
    
    # Count missing values before interpolation
    missing_before = clean_data['sst'].isnull().sum()
    
    # Simple linear interpolation for small gaps
    clean_data['sst'] = clean_data['sst'].interpolate(method='linear', limit=6)
    
    missing_after = clean_data['sst'].isnull().sum()
    interpolated = missing_before - missing_after
    
    if interpolated > 0:
        print(f"   🔧 Interpolated {interpolated} missing values in small gaps")
        cleaning_log.append(f"Interpolated {interpolated} missing values")
    else:
        print("   ✅ No small gaps to interpolate")
    
    # Summary of cleaning process
    print("\n" + "=" * 50)
    print("🎉 DATA CLEANING COMPLETE!")
    print(f"📊 Original dataset: {len(data)} records")
    print(f"📊 Cleaned dataset: {len(clean_data)} records")
    
    print(f"\n📝 Cleaning log:")
    for i, action in enumerate(cleaning_log, 1):
        print(f"   {i}. {action}")
    
    # Calculate data quality metrics
    original_missing = data.isnull().sum().sum()
    final_missing = clean_data.isnull().sum().sum()
    
    print(f"\n📈 Data quality improvement:")
    print(f"   • Missing values: {original_missing} → {final_missing}")
    print(f"   • Temperature coverage: {(~clean_data['sst'].isnull()).mean()*100:.1f}%")
    
    return clean_data, cleaning_log

def visualize_data_quality(original_data, clean_data):
    """
    Create visualizations comparing original messy data to cleaned data.
    
    This shows students the impact of data cleaning!
    """
    print("\n📊 Creating before/after data quality visualizations...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Data Quality: Before and After Cleaning', fontsize=16, fontweight='bold')
    
    # Plot 1: Time series comparison
    axes[0,0].plot(original_data['time'], original_data['sst'], 'r-', alpha=0.7, 
                   linewidth=0.8, label='Original (messy)')
    axes[0,0].plot(clean_data['time'], clean_data['sst'], 'b-', alpha=0.8, 
                   linewidth=1.2, label='Cleaned')
    axes[0,0].set_title('Temperature Time Series: Before vs After')
    axes[0,0].set_ylabel('Temperature (°C)')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Histogram comparison
    axes[0,1].hist(original_data['sst'].dropna(), bins=30, alpha=0.7, 
                   color='red', label='Original', density=True)
    axes[0,1].hist(clean_data['sst'].dropna(), bins=30, alpha=0.7, 
                   color='blue', label='Cleaned', density=True)
    axes[0,1].set_title('Temperature Distribution: Before vs After')
    axes[0,1].set_xlabel('Temperature (°C)')
    axes[0,1].set_ylabel('Density')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Missing data pattern
    missing_original = original_data.isnull()
    missing_clean = clean_data.isnull()
    
    # Create missing data heatmap
    columns = ['time', 'latitude', 'longitude', 'sst']
    missing_counts_orig = [missing_original[col].sum() for col in columns if col in missing_original.columns]
    missing_counts_clean = [missing_clean[col].sum() for col in columns if col in missing_clean.columns]
    
    x = np.arange(len(columns))
    width = 0.35
    
    axes[0,2].bar(x - width/2, missing_counts_orig, width, label='Original', color='red', alpha=0.7)
    axes[0,2].bar(x + width/2, missing_counts_clean, width, label='Cleaned', color='blue', alpha=0.7)
    axes[0,2].set_title('Missing Data Count: Before vs After')
    axes[0,2].set_ylabel('Number of Missing Values')
    axes[0,2].set_xticks(x)
    axes[0,2].set_xticklabels(columns, rotation=45)
    axes[0,2].legend()
    axes[0,2].grid(True, alpha=0.3)
    
    # Plot 4: Data quality over time (original)
    window_size = 24 * 3  # 3-day windows
    time_windows = []
    quality_orig = []
    quality_clean = []
    
    for i in range(0, len(original_data) - window_size, window_size):
        window_data_orig = original_data.iloc[i:i+window_size]
        window_data_clean = clean_data.iloc[i:i+window_size]
        
        # Calculate quality as percentage of non-missing temperature data
        quality_orig.append((~window_data_orig['sst'].isnull()).mean() * 100)
        quality_clean.append((~window_data_clean['sst'].isnull()).mean() * 100)
        time_windows.append(window_data_orig['time'].iloc[0])
    
    axes[1,0].plot(time_windows, quality_orig, 'r-o', label='Original', markersize=4)
    axes[1,0].plot(time_windows, quality_clean, 'b-o', label='Cleaned', markersize=4)
    axes[1,0].set_title('Data Quality Over Time')
    axes[1,0].set_ylabel('Data Coverage (%)')
    axes[1,0].set_ylim(0, 105)
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Plot 5: Temperature change rates
    temp_change_orig = np.abs(original_data['sst'].diff())
    temp_change_clean = np.abs(clean_data['sst'].diff())
    
    axes[1,1].hist(temp_change_orig.dropna(), bins=30, alpha=0.7, color='red', 
                   label='Original', density=True, range=(0, 5))
    axes[1,1].hist(temp_change_clean.dropna(), bins=30, alpha=0.7, color='blue', 
                   label='Cleaned', density=True, range=(0, 5))
    axes[1,1].set_title('Temperature Change Rates')
    axes[1,1].set_xlabel('|ΔTemperature| (°C/hour)')
    axes[1,1].set_ylabel('Density')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    # Plot 6: Quality metrics summary
    metrics = ['Valid Temp', 'Valid Lat', 'Valid Lon', 'Realistic Changes']
    orig_scores = [
        (~original_data['sst'].isnull()).mean() * 100,
        (~original_data['latitude'].isnull()).mean() * 100,
        (~original_data['longitude'].isnull()).mean() * 100,
        (np.abs(original_data['sst'].diff()) <= 5).mean() * 100
    ]
    clean_scores = [
        (~clean_data['sst'].isnull()).mean() * 100,
        (~clean_data['latitude'].isnull()).mean() * 100,
        (~clean_data['longitude'].isnull()).mean() * 100,
        (np.abs(clean_data['sst'].diff()) <= 5).mean() * 100
    ]
    
    x = np.arange(len(metrics))
    axes[1,2].bar(x - width/2, orig_scores, width, label='Original', color='red', alpha=0.7)
    axes[1,2].bar(x + width/2, clean_scores, width, label='Cleaned', color='blue', alpha=0.7)
    axes[1,2].set_title('Data Quality Metrics')
    axes[1,2].set_ylabel('Quality Score (%)')
    axes[1,2].set_xticks(x)
    axes[1,2].set_xticklabels(metrics, rotation=45)
    axes[1,2].set_ylim(0, 105)
    axes[1,2].legend()
    axes[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data_quality_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("📈 Key improvements from cleaning:")
    print(f"   • Removed {((~original_data['sst'].isnull()).mean() - (~clean_data['sst'].isnull()).mean()) * 100:.1f}% of bad data")
    print(f"   • Improved temperature coverage: {(~clean_data['sst'].isnull()).mean()*100:.1f}%")
    print(f"   • Eliminated extreme outliers and impossible values")
    print(f"   • Fixed sensor malfunctions and data transmission errors")

def create_data_quality_checklist():
    """
    Create a checklist for students to use on their own data.
    
    This gives them a practical tool for future data analysis!
    """
    checklist = """
🔍 DATA QUALITY CHECKLIST FOR OCEANOGRAPHIC DATA
===============================================

Before analyzing any dataset, check these items:

📊 COMPLETENESS CHECKS:
□ Count missing values in each column
□ Identify patterns in missing data (random vs systematic)
□ Check if gaps are too large for your analysis

🎯 ACCURACY CHECKS:
□ Temperature: -2°C to 35°C for surface ocean
□ Latitude: -90° to 90°
□ Longitude: -180° to 180°
□ Check for impossible negative depths
□ Verify date/time formats and ranges

📈 CONSISTENCY CHECKS:
□ Look for duplicate records (same time/location)
□ Check for unrealistic rates of change
□ Identify stuck sensors (repeated identical values)
□ Verify units are consistent throughout dataset

🚨 OUTLIER DETECTION:
□ Use box plots to visualize outliers
□ Calculate z-scores (values >3σ from mean)
□ Use domain knowledge (what's physically possible?)
□ Don't automatically remove all outliers - some are real!

🔧 CLEANING DECISIONS:
□ Document all changes made to the data
□ Keep original data file unchanged
□ Use conservative cleaning (when in doubt, flag don't delete)
□ Test sensitivity of results to cleaning choices

✅ VALIDATION STEPS:
□ Compare cleaned data to original visually
□ Check that cleaning didn't introduce new problems
□ Validate against known patterns (seasonal cycles, etc.)
□ Get second opinion from colleagues when possible

💡 REMEMBER:
- Perfect data doesn't exist in the real world
- Some uncertainty is always present
- Document your quality control process
- When reporting results, mention data limitations
"""
    
    print(checklist)
    
    # Save checklist to file
    with open('data_quality_checklist.txt', 'w') as f:
        f.write(checklist)
    
    print("\n💾 Checklist saved as 'data_quality_checklist.txt'")
    print("📋 Use this checklist for all your future data analysis projects!")

def main():
    """
    Main function that runs the complete data quality lesson.
    
    This teaches students one of the most important skills in data science!
    """
    print("🔍 LESSON 6: DATA QUALITY AND VALIDATION")
    print("=" * 50)
    print()
    print("Welcome to the most important lesson in data science! 🎯")
    print("Real data is messy, and learning to clean it properly")
    print("is what separates good scientists from great ones.")
    print()
    print("Today you'll learn to:")
    print("• Identify problems in real datasets")
    print("• Make smart decisions about data cleaning")
    print("• Validate your cleaning process")
    print("• Create professional quality control workflows")
    print()
    
    # Step 1: Create messy, realistic data
    print("STEP 1: Creating realistic messy dataset...")
    messy_data = create_realistic_messy_data()
    
    # Step 2: Systematically identify problems
    problems = identify_data_problems(messy_data)
    
    # Step 3: Clean the data step by step
    clean_data, cleaning_log = clean_data_step_by_step(messy_data)
    
    # Step 4: Visualize the impact of cleaning
    visualize_data_quality(messy_data, clean_data)
    
    # Step 5: Provide tools for future use
    create_data_quality_checklist()
    
    print("\n" + "=" * 50)
    print("🎉 LESSON 6 COMPLETE!")
    print("\nWhat you've learned:")
    print("• How to systematically identify data problems")
    print("• Decision-making process for data cleaning")
    print("• The importance of documenting your cleaning process")
    print("• How to validate that cleaning improved data quality")
    print("• Tools and checklists for future data analysis")
    print("\n🌟 Key Takeaways:")
    print("• ALL real data has problems - expect and plan for this")
    print("• Conservative cleaning is better than aggressive cleaning")
    print("• Always document what you changed and why")
    print("• Visualize before/after to validate your cleaning")
    print("• When in doubt, ask for help from experienced colleagues")
    print("\n🚀 Ready for Lesson 7: Creating Scientific Reports!")
    print("=" * 50)

# Educational extensions
"""
🎓 EDUCATIONAL EXTENSIONS:

REAL-WORLD DATA PROBLEMS:
1. Sensor Issues:
   - Battery failures → gaps in data
   - Biofouling → drift in calibration
   - Mechanical damage → spurious readings
   - Temperature effects on electronics

2. Environmental Challenges:
   - Storms → lost instruments
   - Marine animals → damaged sensors
   - Ice formation → stuck moving parts
   - Saltwater corrosion → gradual failure

3. Data Transmission Problems:
   - Satellite communication failures
   - Data compression errors
   - Network timeouts → duplicate records
   - File corruption during transfer

PROFESSIONAL PRACTICES:
- Always keep original data files
- Use version control for cleaning scripts
- Document all assumptions and decisions
- Get multiple people to review cleaning
- Test sensitivity of results to cleaning choices
- Report data limitations in publications

ADVANCED TOPICS:
- Statistical tests for outlier detection
- Machine learning for anomaly detection
- Uncertainty quantification
- Data fusion from multiple sources
- Real-time quality control systems

CAREER CONNECTIONS:
- Data quality analyst positions
- Scientific data management roles
- Research data repositories
- Environmental monitoring agencies
- Quality assurance in research labs
"""

if __name__ == "__main__":
    main()
