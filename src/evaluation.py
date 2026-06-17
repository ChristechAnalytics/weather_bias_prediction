import numpy as np
from sklearn import metrics

def calculate_metrics(y_true, y_pred):
    """
    Calculates and returns MAE and RMSE for given true and predicted values.
    """
    mae = metrics.mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_true, y_pred))
    return {"MAE": mae, "RMSE": rmse}

def get_baseline_scores(forecast_error_test):
    """
    Establishes the operational baseline by comparing actual forecast errors
    against a vector of zeroes (representing a perfect correction).
    """
    # Create the vector of zeroes using broadcasting
    zero_baseline = 0.0 * forecast_error_test
    return calculate_metrics(zero_baseline, forecast_error_test)

def print_comparison_table(baseline, model_results, model_name="Model"):
    """
    Prints a clean performance comparison table.
    """
    print(f"\n{'Metric':<10} | {'Baseline':<12} | {model_name:<12} | {'Improvement':<12}")
    print("-" * 55)
    for metric in ["MAE", "RMSE"]:
        b_val = baseline[metric]
        m_val = model_results[metric]
        imp = ((b_val - m_val) / b_val) * 100
        print(f"{metric:<10} | {b_val:<12.4f} | {m_val:<12.4f} | {imp:.2f}%")