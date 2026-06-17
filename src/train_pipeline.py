from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import data_prep
import evaluation
import joblib
import os
import plots as plots
from paths import MODEL_PATH, PROJECT_ROOT
    

def run_pipeline():
    print("=======================================================")
    print("   LAUNCHING LIVE ECMWF FORECAST BIAS CORRECTION MAP  ")
    print("=======================================================\n")
    
    # Step 1: Stream data from cloud storage
    # Set sample_fraction=1.0 to train on all 5+ million rows if your machine can handle it
    time_raw, soil_raw, error_raw = data_prep.load_live_ecmwf_data(sample_fraction=0.50)
    
    print("\nStep 2: Engineering features and structural arrays...")
    X = data_prep.combine_features(time_raw, soil_raw)
    y = error_raw
    
    # Step 3: Train/Test Split (80% training data, 20% validation)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"   Training Size: {X_train.shape[0]:,} samples")
    print(f"   Testing Size:  {X_test.shape[0]:,} samples")
    
    # Step 4: Establish the baseline score
    print("\nStep 5: Establishing zero-correction baseline metrics...")
    baseline_scores = evaluation.get_baseline_scores(y_test)
    
    # Step 5: Fit Random Forest
    print("\nStep 6: Fitting Random Forest Regressor to operational data...")
    print("   (Using parallel processing cores)...")
    rf_model = RandomForestRegressor(
        n_estimators=20, 
        max_depth=20, 
        min_samples_split=10, 
        random_state=42, 
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf_model, MODEL_PATH)
    print(f"\n   Model saved to {MODEL_PATH}")
    
    # Step 6: Prediction & Validation
    print("\nStep 7: Evaluating correction predictions against true observations...")
    predictions = rf_model.predict(X_test)
    model_scores = evaluation.calculate_metrics(y_test, predictions)
    
    # Step 7: Print beautiful markdown results
    print("\n================== REGRESSION MODEL SUMMARY ==================")
    evaluation.print_comparison_table(baseline_scores, model_scores, model_name="Random Forest")
    print("==============================================================")

    print("\nStep 7: Generating diagnostic asset visualizations...")
    plots_dir = PROJECT_ROOT / "plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Extract a slice of the test features for 1D plotting compatibility
    # X_test[:, 0] is time_of_day, X_test[:, 1] is soil_temperature
    plots.plot_diurnal_bias(
        time_of_day=X_test[:, 0], 
        forecast_error=y_test, 
        predictions=predictions, 
        save_path=str(plots_dir / "diurnal_bias_correction.png"),
    )
    
    plots.plot_2d_interaction_density(
        time_of_day=X_test[:, 0], 
        soil_temperature=X_test[:, 1], 
        forecast_error=y_test, 
        save_path=str(plots_dir / "interaction_heatmap.png"),
    )
    print("\n================== PIPELINE COMPLETELY EXECUTE ==================")

if __name__ == "__main__":
    run_pipeline()