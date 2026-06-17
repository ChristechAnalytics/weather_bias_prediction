import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d

def plot_diurnal_bias(time_of_day, forecast_error, predictions=None, save_path=None):
    """
    Plots the raw forecast error and optional model predictions against the time of day
    to visualize the diurnal wave cycle bias.
    """
    plt.figure(figsize=(10, 6))
    
    # Calculate binned or running means to see the trend through the noise
    bins = np.linspace(0, 24, 49) # Half-hour bins
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    
    # Calculate mean raw error per bin
    bin_indices = np.digitize(time_of_day, bins) - 1
    raw_means = [forecast_error[bin_indices == i].mean() if np.any(bin_indices == i) else np.nan for i in range(48)]
    
    plt.plot(bin_centers, raw_means, label='Observed Bias (Raw Forecast Error)', color='darkblue', linewidth=2)
    
    if predictions is not None:
        pred_means = [predictions[bin_indices == i].mean() if np.any(bin_indices == i) else np.nan for i in range(48)]
        plt.plot(bin_centers, pred_means, label='Random Forest Correction Profile', color='crimson', linestyle='--', linewidth=2)
        
    plt.axhline(0, color='gray', linestyle=':', alpha=0.7)
    plt.xlabel('Local Time of Day (Decimal Hours)')
    plt.ylabel('Forecast Error (°C)')
    plt.title('Diurnal Cycle Model Bias & Correction Profiles')
    plt.xlim(0, 24)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   [PLOT] Saved diurnal bias visualization to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_2d_interaction_density(time_of_day, soil_temperature, forecast_error, save_path=None):
    """
    Generates a 2D color-coded density grid showing how the average forecast error 
    varies across combinations of soil temperature and time of day.
    """
    plt.figure(figsize=(11, 7))
    
    # Create a 2D histogram grid to calculate average errors across features
    # Adjust bins to match your full dataset limits smoothly
    time_bins = np.linspace(0, 24, 50)
    soil_bins = np.linspace(-20, 50, 50)
    
    # Use profile/binned statistics to find the average error in each pixel grid
    statistic, x_edges, y_edges, _ = binned_statistic_2d(
        soil_temperature, time_of_day, forecast_error,
        statistic='mean', bins=[soil_bins, time_bins]
    )
    
    # Plot using an elegant diverging colormap ('RdYlBu_r' or 'coolwarm') 
    # Center the colorbar at 0 so Red = Warm Bias, Blue = Cold Bias
    mesh = plt.pcolormesh(x_edges, y_edges, statistic.T, cmap='coolwarm', vmin=-3, vmax=3)
    
    cbar = plt.colorbar(mesh)
    cbar.set_label('Average Forecast Error (°C)', rotation=270, labelpad=15)
    
    plt.xlabel('Model Soil Temperature (°C)')
    plt.ylabel('Local Time of Day (Hours)')
    plt.title('NWP Forecast Error Regimes: Soil Temperature vs. Time of Day')
    plt.grid(True, alpha=0.2, linestyle='--')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   [PLOT] Saved 2D interaction heatmap to {save_path}")
    else:
        plt.show()
    plt.close()