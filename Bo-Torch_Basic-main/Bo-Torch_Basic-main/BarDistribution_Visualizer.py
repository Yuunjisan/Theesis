"""
TabPFN BarDistribution Histogram Visualizer
==========================================

This script visualizes the actual histogram distributions that TabPFN creates
for predictions at specific points. This shows what's "under the hood" of
TabPFN's uncertainty quantification.
"""

import numpy as np
import matplotlib.pyplot as plt
from tabpfn import TabPFNRegressor
import torch
from pyDOE import lhs

# Set random seed for reproducibility
np.random.seed(42)

def target_function(x):
    """Target function: x² * sin(x) + 5"""
    return x**2 * np.sin(x) + 5

def downsample_bardistribution(criterion, logits, n_vis_bins=150):
    """
    Downsample the high-resolution BarDistribution to fewer bins for visualization
    
    Args:
        criterion: The BarDistribution object
        logits: The logits tensor [1, n_bins]
        n_vis_bins: Number of bins for visualization (default: 150)
        
    Returns:
        bin_centers: Array of bin centers for visualization
        probabilities: Array of probabilities for each visualization bin
    """
    
    # Get the effective range using quantiles
    q001 = criterion.icdf(logits, torch.tensor(0.001, device=logits.device))[0].detach().cpu().numpy()
    q999 = criterion.icdf(logits, torch.tensor(0.999, device=logits.device))[0].detach().cpu().numpy()
    
    # Ensure scalar values
    q001 = float(q001.item()) if isinstance(q001, np.ndarray) and q001.size == 1 else float(q001)
    q999 = float(q999.item()) if isinstance(q999, np.ndarray) and q999.size == 1 else float(q999)
    
    # Create visualization bins spanning the effective range
    bin_edges = np.linspace(q001, q999, n_vis_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Calculate probabilities for each visualization bin using the CDF
    vis_probabilities = np.zeros(n_vis_bins)
    
    for i in range(n_vis_bins):
        # Create tensors with correct batch dimension to match logits [1, n_bins]
        lower_edge = torch.tensor([[bin_edges[i]]], device=logits.device)  # Shape: [1, 1]
        upper_edge = torch.tensor([[bin_edges[i+1]]], device=logits.device)  # Shape: [1, 1]
        
        # Get CDF values at bin edges
        p_lower = float(criterion.cdf(logits, lower_edge)[0].detach().cpu().numpy())
        p_upper = float(criterion.cdf(logits, upper_edge)[0].detach().cpu().numpy())
        
        # Probability in this bin is the CDF difference
        vis_probabilities[i] = p_upper - p_lower
    
    return bin_centers, vis_probabilities

def visualize_bardistribution_at_points(model, prediction_points, title_suffix="", n_vis_bins=150):
    """
    Visualize the BarDistribution histograms at specific prediction points
    
    Args:
        model: Fitted TabPFNRegressor
        prediction_points: List of x-values where to visualize distributions
        title_suffix: Additional text for plot titles
        n_vis_bins: Number of bins for visualization (default: 150)
    """
    
    n_points = len(prediction_points)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    # Ensure we don't have more points than subplots
    n_points = min(n_points, 6)
    
    for i, x_point in enumerate(prediction_points[:n_points]):
        ax = axes[i]
        
        # Get prediction at single point
        X_single = np.array([[x_point]])
        
        # Get BarDistribution output
        output = model.predict(X_single, output_type="full")
        logits = output["logits"]  # Shape: [1, n_bins] (5000 bins)
        criterion = output["criterion"]
        
        # Downsample to visualization resolution
        bin_centers, probabilities = downsample_bardistribution(criterion, logits, n_vis_bins)
        
        # Calculate bin width for visualization
        if len(bin_centers) > 1:
            bin_width = (bin_centers[-1] - bin_centers[0]) / (len(bin_centers) - 1)
            bar_width = bin_width * 0.9  # 90% of bin width for nice appearance
        else:
            bar_width = 1.0
        
        # Plot the downsampled histogram
        ax.bar(bin_centers, probabilities, width=bar_width, 
               alpha=0.7, color='purple', edgecolor='black', linewidth=0.3)
        
        # Calculate and plot statistics from original high-resolution distribution
        mean_val = criterion.mean(logits.unsqueeze(0))[0].detach().cpu().numpy()
        median_val = criterion.median(logits.unsqueeze(0))[0].detach().cpu().numpy()
        
        # Ensure scalar values
        mean_val = float(mean_val.item()) if isinstance(mean_val, np.ndarray) and mean_val.size == 1 else float(mean_val)
        median_val = float(median_val.item()) if isinstance(median_val, np.ndarray) and median_val.size == 1 else float(median_val)
        
        # Plot mean and median with thinner lines
        ax.axvline(mean_val, color='darkorange', linewidth=1.5, linestyle='-', 
                  label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='green', linewidth=1.5, linestyle='--', 
                  label=f'Median: {median_val:.2f}')
        
        # Calculate quantiles for both display and axis scaling
        q01 = criterion.icdf(logits.unsqueeze(0), torch.tensor(0.01, device=logits.device))[0].detach().cpu().numpy()
        q05 = criterion.icdf(logits.unsqueeze(0), torch.tensor(0.05, device=logits.device))[0].detach().cpu().numpy()
        q10 = criterion.icdf(logits.unsqueeze(0), torch.tensor(0.1, device=logits.device))[0].detach().cpu().numpy()
        q25 = criterion.icdf(logits.unsqueeze(0), torch.tensor(0.25, device=logits.device))[0].detach().cpu().numpy()
        q75 = criterion.icdf(logits.unsqueeze(0), torch.tensor(0.75, device=logits.device))[0].detach().cpu().numpy()
        q90 = criterion.icdf(logits.unsqueeze(0), torch.tensor(0.9, device=logits.device))[0].detach().cpu().numpy()
        q95 = criterion.icdf(logits.unsqueeze(0), torch.tensor(0.95, device=logits.device))[0].detach().cpu().numpy()
        q99 = criterion.icdf(logits.unsqueeze(0), torch.tensor(0.99, device=logits.device))[0].detach().cpu().numpy()
        
        # Ensure scalar values for quantiles
        q01 = float(q01.item()) if isinstance(q01, np.ndarray) and q01.size == 1 else float(q01)
        q05 = float(q05.item()) if isinstance(q05, np.ndarray) and q05.size == 1 else float(q05)
        q10 = float(q10.item()) if isinstance(q10, np.ndarray) and q10.size == 1 else float(q10)
        q25 = float(q25.item()) if isinstance(q25, np.ndarray) and q25.size == 1 else float(q25)
        q75 = float(q75.item()) if isinstance(q75, np.ndarray) and q75.size == 1 else float(q75)
        q90 = float(q90.item()) if isinstance(q90, np.ndarray) and q90.size == 1 else float(q90)
        q95 = float(q95.item()) if isinstance(q95, np.ndarray) and q95.size == 1 else float(q95)
        q99 = float(q99.item()) if isinstance(q99, np.ndarray) and q99.size == 1 else float(q99)
        
        # Plot quantiles with thinner lines
        ax.axvline(q10, color='lightblue', linewidth=1, linestyle=':', alpha=0.7, 
                  label=f'10%: {q10:.2f}')
        ax.axvline(q90, color='lightblue', linewidth=1, linestyle=':', alpha=0.7, 
                  label=f'90%: {q90:.2f}')
        
        # True function value for comparison with thinner line
        true_val = target_function(x_point)
        ax.axvline(true_val, color='red', linewidth=1.5, linestyle='-', alpha=0.8,
                  label=f'True: {true_val:.2f}')
        
        # Scale x-axis based on prediction distribution
        # Use 1st to 99th percentile with padding for good visualization
        q01 = criterion.icdf(logits.unsqueeze(0), torch.tensor(0.01, device=logits.device))[0].detach().cpu().numpy()
        q99 = criterion.icdf(logits.unsqueeze(0), torch.tensor(0.99, device=logits.device))[0].detach().cpu().numpy()
        
        # Ensure scalar values
        q01 = float(q01.item()) if isinstance(q01, np.ndarray) and q01.size == 1 else float(q01)
        q99 = float(q99.item()) if isinstance(q99, np.ndarray) and q99.size == 1 else float(q99)
        
        # Add padding (10% of range on each side)
        range_width = q99 - q01
        padding = range_width * 0.1 if range_width > 0 else 1.0
        
        x_min = q01 - padding
        x_max = q99 + padding
        
        # Include true value in the range if it's outside
        if true_val < x_min:
            x_min = true_val - padding
        if true_val > x_max:
            x_max = true_val + padding
        
        ax.set_xlim(x_min, x_max)
        
        ax.set_title(f'BarDistribution at x = {x_point:.2f}', fontweight='bold')
        ax.set_xlabel('Predicted Value')
        ax.set_ylabel('Probability')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Add minimal range info
        ax.text(0.98, 0.02, f'Range: [{x_min:.1f}, {x_max:.1f}]', 
                transform=ax.transAxes, horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7), fontsize=8)
    
    # Hide unused subplots
    for i in range(n_points, 6):
        axes[i].set_visible(False)
    
    plt.suptitle(f'TabPFN BarDistribution Histograms {title_suffix}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def create_training_data(n_samples=20, x_range=(-6, 6), noise_std=0.05):
    """Create training data using LHS"""
    np.random.seed(42)
    samples_normalized = lhs(1, samples=n_samples, criterion="maximin")
    x_train = x_range[0] + (x_range[1] - x_range[0]) * samples_normalized
    y_train = target_function(x_train.ravel()) + np.random.normal(0, noise_std, n_samples)
    return x_train, y_train

def compare_distributions_at_different_regions():
    """Compare BarDistributions in different regions (near data, interpolation, extrapolation)"""
    
    # Create training data
    x_train, y_train = create_training_data(n_samples=20)
    
    print("Training data points:")
    for i, (x, y) in enumerate(zip(x_train.ravel(), y_train)):
        print(f"  Point {i+1}: x = {x:.2f}, y = {y:.2f}")
    
    # Fit TabPFN model
    model = TabPFNRegressor(n_estimators=8, softmax_temperature=0.9, device="auto")
    model.fit(x_train, y_train)
    
    # Select points in different regions
    prediction_points = [
        x_train.ravel()[0],  # Near training point
        (x_train.ravel()[0] + x_train.ravel()[1]) / 2,  # Interpolation
        -5.5,  # Extrapolation (left)
        0.0,   # Center region
        2.0,   # Another interpolation
        5.5    # Extrapolation (right)
    ]
    
    # Visualize distributions with custom bin resolution
    print(f"\nVisualizing with 500 bins (downsampled from 5000)")
    fig = visualize_bardistribution_at_points(model, prediction_points, 
                                            "(Different Regions)", n_vis_bins=500)
    plt.savefig('bardistribution_regions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return model, x_train, y_train

def analyze_distribution_properties(model, x_points):
    """Analyze statistical properties of distributions at different points"""
    
    print("\n" + "="*80)
    print("BARDISTRIBUTION ANALYSIS")
    print("="*80)
    
    for x_point in x_points:
        X_single = np.array([[x_point]])
        
        # Get BarDistribution
        output = model.predict(X_single, output_type="full")
        logits = output["logits"]
        criterion = output["criterion"]
        
        # Calculate statistics with proper scalar conversion
        mean_val = criterion.mean(logits)[0].detach().cpu().numpy()
        median_val = criterion.median(logits)[0].detach().cpu().numpy()
        variance_val = criterion.variance(logits)[0].detach().cpu().numpy()
        
        # Ensure scalar values
        mean_val = float(mean_val.item()) if isinstance(mean_val, np.ndarray) and mean_val.size == 1 else float(mean_val)
        median_val = float(median_val.item()) if isinstance(median_val, np.ndarray) and median_val.size == 1 else float(median_val)
        variance_val = float(variance_val.item()) if isinstance(variance_val, np.ndarray) and variance_val.size == 1 else float(variance_val)
        
        # Calculate additional quantiles
        q05 = criterion.icdf(logits, torch.tensor(0.05, device=logits.device))[0].detach().cpu().numpy()
        q25 = criterion.icdf(logits, torch.tensor(0.25, device=logits.device))[0].detach().cpu().numpy()
        q75 = criterion.icdf(logits, torch.tensor(0.75, device=logits.device))[0].detach().cpu().numpy()
        q95 = criterion.icdf(logits, torch.tensor(0.95, device=logits.device))[0].detach().cpu().numpy()
        
        # Ensure scalar values for quantiles
        q05 = float(q05.item()) if isinstance(q05, np.ndarray) and q05.size == 1 else float(q05)
        q25 = float(q25.item()) if isinstance(q25, np.ndarray) and q25.size == 1 else float(q25)
        q75 = float(q75.item()) if isinstance(q75, np.ndarray) and q75.size == 1 else float(q75)
        q95 = float(q95.item()) if isinstance(q95, np.ndarray) and q95.size == 1 else float(q95)
        
        # True value
        true_val = target_function(x_point)
        
        print(f"\nAt x = {x_point:.2f}:")
        print(f"  True value:    {true_val:.3f}")
        print(f"  Mean:          {mean_val:.3f}")
        print(f"  Median:        {median_val:.3f}")
        print(f"  Variance:      {variance_val:.3f}")
        print(f"  Std dev:       {np.sqrt(variance_val):.3f}")
        print(f"  5%-95% range:  [{q05:.3f}, {q95:.3f}] (width: {q95-q05:.3f})")
        print(f"  25%-75% range: [{q25:.3f}, {q75:.3f}] (width: {q75-q25:.3f})")
        print(f"  Mean-Median:   {mean_val - median_val:.3f} (skewness indicator)")

def main():
    """Main demonstration function"""
    
    print("🔍 TabPFN BarDistribution Histogram Visualizer")
    print("=" * 60)
    
    # Create and analyze distributions
    model, x_train, y_train = compare_distributions_at_different_regions()
    
    # Analyze properties
    analysis_points = [-5.0, -2.0, 0.0, 2.0, 5.0]
    analyze_distribution_properties(model, analysis_points)
    
    print("\n🎉 BarDistribution visualization completed!")
    print("📁 Plot saved as: bardistribution_regions.png")
    print("\nKey insights:")
    print("• Each histogram shows TabPFN's full predictive distribution")
    print("• Narrow histograms = high confidence")
    print("• Wide histograms = high uncertainty") 
    print("• Skewed histograms = asymmetric uncertainty")
    print("• Multiple peaks = multimodal predictions")

if __name__ == "__main__":
    main() 