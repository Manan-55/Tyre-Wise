import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import os
import json

def train_compound_model(df: pd.DataFrame, compound_name: str, features: list, target: str, output_dir: str):
    """
    Trains a specialized XGBoost model for a specific tire compound.
    """
    print(f"\n[{compound_name}] Initializing training pipeline...")
    
    # Filter data for the specific compound
    compound_data = df[df['Compound'] == compound_name].copy()
    
    if len(compound_data) < 100:
        print(f"[{compound_name}] Warning: Not enough data to train. Skipping.")
        return None

    X = compound_data[features]
    y = compound_data[target]

    # Chronological split or random split? 
    # For F1, random split is okay for lap-level degradation, but a time-series split (by Year/Race) is better. 
    # Sticking to random for this iteration to ensure diverse track coverage in both sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Configure XGBoost for GPU hardware acceleration
    params = {
    'n_estimators': 500,
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'tree_method': 'hist',
    'device': 'cuda' if xgb.build_info().get('USE_CUDA') else 'cpu',
    'objective': 'reg:squarederror',
    'early_stopping_rounds': 50,  # ← moved here
}
    model = xgb.XGBRegressor(**params)
    
    print(f"[{compound_name}] Fitting model on {len(X_train)} laps...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # Evaluation
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"[{compound_name}] Test MAE: {mae:.3f} seconds per lap")

    # Save the model artifact
    model_path = os.path.join(output_dir, f"xgb_{compound_name.lower()}.json")
    model.save_model(model_path)
    print(f"[{compound_name}] Model saved to {model_path}")

    return mae

def main():
    input_path = 'data/processed/master_processed.csv'
    models_dir = 'backend/models' # Note the new directory structure targeting the API
    
    os.makedirs(models_dir, exist_ok=True)

    print(f"Loading preprocessed data from {input_path}...")
    df = pd.read_csv(input_path)

    # 1. Define Features & Target
    # Notice we removed 'compound_encoded' because we are separating the models entirely
    features = [
        'LapNumber', 
        'TyreLife', 
        'tyre_life_squared', # The polynomial feature modeling "the cliff"
        'is_fresh_tyre', 
        'stint_progress', 
        'deg_rate'
    ]
    target = 'LapTime_s'

    # 2. Train independent models for the primary race compounds
    compounds = ['SOFT', 'MEDIUM', 'HARD']
    performance_metrics = {}

    for compound in compounds:
        mae = train_compound_model(df, compound, features, target, models_dir)
        if mae:
            performance_metrics[compound] = mae

    # 3. Save feature schema for the FastAPI backend to use during inference
    schema_path = os.path.join(models_dir, 'feature_schema.json')
    with open(schema_path, 'w') as f:
        json.dump({"features": features}, f)
    
    print("\n--- Training Complete ---")
    print("Metrics summary (Seconds off actual lap time):")
    for comp, mae in performance_metrics.items():
        print(f" - {comp}: {mae:.3f}s")

if __name__ == "__main__":
    main()