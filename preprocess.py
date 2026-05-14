import pandas as pd
import numpy as np
import os

def calculate_deg_rate(group):
    """
    Calculates the degradation rate (slope of LapTime_s vs TyreLife) for a given stint.
    Requires at least 2 data points with varying TyreLife to calculate a valid slope.
    """
    # Check if we have enough points and if TyreLife actually varies to avoid polyfit errors
    if len(group) > 1 and group['TyreLife'].nunique() > 1:
        # np.polyfit returns [slope, intercept] for a degree 1 fit
        slope, _ = np.polyfit(group['TyreLife'], group['LapTime_s'], 1)
        return slope
    return 0.0

def main():
    # ---------------------------------------------------------
    # 1. Setup Paths & Load Data
    # ---------------------------------------------------------
    input_path = 'data/raw/master_laps.csv'
    output_dir = 'data/processed'
    output_path = os.path.join(output_dir, 'master_processed.csv')

    print(f"Loading data from {input_path}...")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: Could not find {input_path}. Please ensure the file exists.")
        return

    print(f"Initial shape: {df.shape}")

    # ---------------------------------------------------------
    # 2. Handle Missing Values
    # ---------------------------------------------------------
    # Drop rows where critical columns are null
    df = df.dropna(subset=['Compound', 'TyreLife', 'LapTime_s'])
    
    # ---------------------------------------------------------
    # 3. Remove Outliers
    # ---------------------------------------------------------
    # Keep lap times between 70s and 130s (filters out safety cars, VSC, out/in laps)
    # Remove Lap 1 (standing start, heavily influenced by traffic and launch)
    df = df[(df['LapTime_s'] >= 70) & (df['LapTime_s'] <= 130)]
    df = df[df['LapNumber'] > 1]

    # ---------------------------------------------------------
    # 4. Feature Engineering
    # ---------------------------------------------------------
    # Compound Encoding
    compound_map = {'SOFT': 0, 'MEDIUM': 1, 'HARD': 2, 'INTERMEDIATE': 3, 'WET': 4}
    df['compound_encoded'] = df['Compound'].str.upper().map(compound_map)

    # Convert FreshTyre to integer (1 if True, 0 if False)
    # Handles both boolean and string representations of True/False
    df['is_fresh_tyre'] = df['FreshTyre'].astype(str).str.lower().map({'true': 1, '1': 1, '1.0': 1}).fillna(0).astype(int)

    # Tyre Life Squared (captures non-linear degradation, or the "cliff")
    df['tyre_life_squared'] = df['TyreLife'] ** 2

    # Define the columns that uniquely identify a single stint for a driver
    stint_grouping = ['Year', 'GrandPrix', 'SessionType', 'Driver', 'Stint']

    # Stint Progress: Current TyreLife / Max TyreLife in that specific stint
    max_tyre_life = df.groupby(stint_grouping)['TyreLife'].transform('max')
    # Prevent division by zero just in case
    df['stint_progress'] = np.where(max_tyre_life > 0, df['TyreLife'] / max_tyre_life, 0)

    # Degradation Rate (Slope of LapTime vs TyreLife)
    print("Calculating degradation rates per stint (this may take a moment)...")
    deg_rates = df.groupby(stint_grouping).apply(calculate_deg_rate, include_groups=False).reset_index(name='deg_rate')
    
    # Merge the calculated degradation rates back into the main dataframe
    df = pd.merge(df, deg_rates, on=stint_grouping, how='left')

    # ---------------------------------------------------------
    # 5. Save Processed Data
    # ---------------------------------------------------------
    # Create the processed directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"\nSuccessfully saved cleaned data to {output_path}")

    # ---------------------------------------------------------
    # 6. Final Outputs
    # ---------------------------------------------------------
    print(f"Final shape: {df.shape}")
    print("\nPreview of engineered columns:")
    preview_cols = ['Driver', 'LapNumber', 'Compound', 'compound_encoded', 
                    'TyreLife', 'tyre_life_squared', 'is_fresh_tyre', 
                    'stint_progress', 'deg_rate', 'LapTime_s']
    print(df[preview_cols].head())

if __name__ == "__main__":
    main()