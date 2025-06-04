import numpy as np
from matplotlib import pyplot as plt
from tabpfn import TabPFNRegressor
from pyDOE import lhs
import os
import torch
import matplotlib.gridspec as gridspec

# Set random seed for reproducibility
np.random.seed(42)

def f(X:np.ndarray):
    x = X.copy()
    return x**2 * np.sin(x) + 5

def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def downsample_bardistribution(criterion, logits, n_vis_bins=150):
    """
    Downsample the high-resolution BarDistribution to fewer bins for visualization
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
        # Create tensors with correct dimension to match logits [1, n_bins]
        lower_edge = torch.tensor([bin_edges[i]], device=logits.device)  # Shape: [1]
        upper_edge = torch.tensor([bin_edges[i+1]], device=logits.device)  # Shape: [1]
        
        # Get CDF values at bin edges
        p_lower = criterion.cdf(logits, lower_edge)[0].detach().cpu().numpy()
        p_upper = criterion.cdf(logits, upper_edge)[0].detach().cpu().numpy()
        
        # Ensure scalar values and convert to float
        p_lower = float(p_lower.item()) if isinstance(p_lower, np.ndarray) and p_lower.size == 1 else float(p_lower)
        p_upper = float(p_upper.item()) if isinstance(p_upper, np.ndarray) and p_upper.size == 1 else float(p_upper)
        
        # Probability in this bin is the CDF difference
        vis_probabilities[i] = p_upper - p_lower
    
    return bin_centers, vis_probabilities

def generate_tabpfn_plot(n_samples, softmax_temp, n_estimators, save_dir, xi=None, fi=None, y_min=None, y_max=None):
    """Generate and save a TabPFN plot with specified samples, softmax temperature, and number of estimators"""
    
    # Define bounds
    lb = -6.0
    ub = 6.0

    # Use provided samples or generate new ones
    if xi is None or fi is None:
        # Sample points using Latin Hypercube Sampling
        random_state = np.random.RandomState(42)  # Use fixed seed 42
        # Use the random_state parameter in lhs function (requires pyDOE2)
        # If using original pyDOE which doesn't have random_state param:
        np.random.seed(42)  # Set seed right before LHS call
        samples_normalized = lhs(1, samples=n_samples, criterion="center")
        xi = lb + (ub - lb) * samples_normalized
        fi = f(xi).ravel()

    # Create and fit the TabPFN model
    model = TabPFNRegressor(n_estimators=n_estimators, softmax_temperature=softmax_temp, device="auto")
    model.fit(xi, fi)

    # Define points for predictions (fine grid for smooth visualization)
    X_pred = np.atleast_2d(np.linspace(lb, ub, 100)).T

    # Get predictions using BarDistribution directly
    output = model.predict(X_pred, output_type="full")
    logits = output["logits"]
    criterion = output["criterion"]  # This is the BarDistribution
    
    # Extract predictions directly from BarDistribution
    mean = criterion.mean(logits).detach().cpu().numpy()
    median = criterion.median(logits).detach().cpu().numpy()
    
    # Get quantiles directly from BarDistribution
    quantile_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    quantiles = []
    for q in quantile_values:
        q_tensor = torch.tensor(q, device=logits.device)
        q_pred = criterion.icdf(logits, q_tensor).detach().cpu().numpy()
        quantiles.append(q_pred)

    # ============================================================================
    # CREATE SINGLE FIGURE WITH PROPER GRID LAYOUT (VERTICAL)
    # ============================================================================
    
    # Determine optimal grid for BarDistribution plots
    bardist_cols = 3  # Always use 3 columns for consistent layout
    bardist_rows = (min(n_samples, 12) + bardist_cols - 1) // bardist_cols  # Calculate rows needed
    
    # Calculate figure height based on layout
    sausage_height = 4  # Height for sausage plot
    bardist_height = 2.5 * bardist_rows  # Height per row of BarDistribution plots
    total_height = sausage_height + bardist_height + 1  # Extra space for spacing
    
    # Calculate figure width based on number of columns to prevent stretching
    # Always 3 columns, so use consistent width
    fig_width = 11  # Optimal width for 3-column layout
    
    # Create figure with proper proportions
    fig = plt.figure(figsize=(fig_width, total_height))
    
    # Create GridSpec: top row for sausage plot, bottom rows for BarDistribution
    # Total rows = 2 (1 for sausage, 1+ for BarDistribution grid)
    gs = gridspec.GridSpec(1 + bardist_rows, bardist_cols, 
                          height_ratios=[2] + [1] * bardist_rows,
                          hspace=0.4, wspace=0.3)
    
    # ============================================================================
    # TOP SUBPLOT: SAUSAGE PLOT (spans all columns in top row)
    # ============================================================================
    ax_sausage = fig.add_subplot(gs[0, :])  # Top row, all columns
    
    # Plot the true function
    X_true = np.atleast_2d(np.linspace(lb, ub, 500)).T
    ax_sausage.plot(X_true, f(X_true), 'r--', label='True function', linewidth=2)

    # Plot multiple confidence intervals with decreasing opacity
    alphas = [0.05, 0.1, 0.15, 0.2]
    # quantile_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # Indices:           [0,   1,   2,   3,   4,   5,   6,   7,   8]
    # So: (0,8)=10-90%, (1,7)=20-80%, (2,6)=30-70%, (3,5)=40-60%
    quantile_pairs = [(0, 8), (1, 7), (2, 6), (3, 5)]  # 10-90%, 20-80%, 30-70%, 40-60%
    
    for (lower_idx, upper_idx), alpha_val in zip(quantile_pairs, alphas):
        ax_sausage.fill_between(X_pred.ravel(), 
                        quantiles[lower_idx].ravel(), 
                        quantiles[upper_idx].ravel(), 
                        alpha=alpha_val, color='purple', 
                        label=f'Confidence region' if lower_idx == 0 else None)

    # Plot the predicted mean
    ax_sausage.plot(X_pred, mean, color='darkorange', label='TabPFN mean', linewidth=2)
    
    # Plot the predicted median
    ax_sausage.plot(X_pred, median, color='green', linestyle='--', label='TabPFN median', linewidth=2)

    # Plot training data
    ax_sausage.scatter(xi, fi, color='navy', s=50, alpha=0.8, 
                      label='Training points')

    ax_sausage.set_xlabel('x', fontsize=12)
    ax_sausage.set_ylabel('f(x)', fontsize=12)
    ax_sausage.set_title(f'TabPFN Uncertainty Quantification (n={n_samples}, temp={softmax_temp}, estimators={n_estimators})', 
                        fontsize=14, fontweight='bold')
    
    ax_sausage.legend(fontsize=9)
    ax_sausage.grid(True, alpha=0.3)
    ax_sausage.tick_params(labelsize=10)
    
    # Add confidence band explanation
    ax_sausage.text(0.02, 0.98, 'Purple bands:\n90-10% (lightest)\n80-20%\n70-30%\n60-40% (darkest)', 
                   transform=ax_sausage.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9), 
                   fontsize=8, color='black')
    
    # Set y-axis limits if provided
    if y_min is not None and y_max is not None:
        ax_sausage.set_ylim(y_min, y_max)
    
    # ============================================================================
    # BOTTOM SUBPLOTS: BARDISTRIBUTION PLOTS (grid below sausage plot)
    # ============================================================================
    
    # Use actual training sample points for BarDistribution analysis
    prediction_points = xi.ravel()[:min(len(xi), 12)]  # Limit to 12 max
    
    for i, x_point in enumerate(prediction_points):
        row = i // bardist_cols
        col = i % bardist_cols
        
        # Create proper subplot in grid (starting from row 1, since row 0 is sausage plot)
        ax_bardist = fig.add_subplot(gs[row + 1, col])
        
        # Get prediction at single point
        X_single = np.array([[x_point]])
        
        # Get BarDistribution output
        output = model.predict(X_single, output_type="full")
        logits = output["logits"]
        criterion = output["criterion"]
        
        # Downsample to visualization resolution
        bin_centers, probabilities = downsample_bardistribution(criterion, logits, 150)
        
        # Calculate bin width for visualization
        if len(bin_centers) > 1:
            bin_width = (bin_centers[-1] - bin_centers[0]) / (len(bin_centers) - 1)
            bar_width = bin_width * 0.9
        else:
            bar_width = 1.0
        
        # Plot the histogram with confidence band color coding
        # First, calculate the confidence band boundaries at this x-point
        q10 = criterion.icdf(logits, torch.tensor(0.1, device=logits.device))[0].detach().cpu().numpy()
        q20 = criterion.icdf(logits, torch.tensor(0.2, device=logits.device))[0].detach().cpu().numpy()
        q30 = criterion.icdf(logits, torch.tensor(0.3, device=logits.device))[0].detach().cpu().numpy()
        q40 = criterion.icdf(logits, torch.tensor(0.4, device=logits.device))[0].detach().cpu().numpy()
        q60 = criterion.icdf(logits, torch.tensor(0.6, device=logits.device))[0].detach().cpu().numpy()
        q70 = criterion.icdf(logits, torch.tensor(0.7, device=logits.device))[0].detach().cpu().numpy()
        q80 = criterion.icdf(logits, torch.tensor(0.8, device=logits.device))[0].detach().cpu().numpy()
        q90 = criterion.icdf(logits, torch.tensor(0.9, device=logits.device))[0].detach().cpu().numpy()
        
        # Ensure scalar values
        q10 = float(q10.item()) if isinstance(q10, np.ndarray) and q10.size == 1 else float(q10)
        q20 = float(q20.item()) if isinstance(q20, np.ndarray) and q20.size == 1 else float(q20)
        q30 = float(q30.item()) if isinstance(q30, np.ndarray) and q30.size == 1 else float(q30)
        q40 = float(q40.item()) if isinstance(q40, np.ndarray) and q40.size == 1 else float(q40)
        q60 = float(q60.item()) if isinstance(q60, np.ndarray) and q60.size == 1 else float(q60)
        q70 = float(q70.item()) if isinstance(q70, np.ndarray) and q70.size == 1 else float(q70)
        q80 = float(q80.item()) if isinstance(q80, np.ndarray) and q80.size == 1 else float(q80)
        q90 = float(q90.item()) if isinstance(q90, np.ndarray) and q90.size == 1 else float(q90)
        
        # Color-code bars based on confidence bands - matching sausage plot purples but lighter
        bar_colors = []
        
        for center in bin_centers:
            if q40 <= center <= q60:  # 40-60% band (darkest - matches alpha=0.2)
                bar_colors.append(('#6A5ACD', 0.8))  # Slate blue (lighter dark purple)
            elif q30 <= center <= q70:  # 30-70% band (matches alpha=0.15)
                bar_colors.append(('#8A2BE2', 0.7))  # Blue violet
            elif q20 <= center <= q80:  # 20-80% band (matches alpha=0.1)
                bar_colors.append(('#9370DB', 0.6))  # Medium slate blue
            elif q10 <= center <= q90:  # 10-90% band (lightest - matches alpha=0.05)
                bar_colors.append(('#B19CD9', 0.5))  # Light purple
            else:  # Outside all bands
                bar_colors.append(('#D3D3D3', 0.4))  # Light gray
        
        # Plot bars with lighter purple gradient matching sausage plot
        for j, (center, prob) in enumerate(zip(bin_centers, probabilities)):
            color, alpha = bar_colors[j]
            ax_bardist.bar(center, prob, width=bar_width, 
                   alpha=alpha, color=color, edgecolor='black', linewidth=0.1)
        
        # # Add boundary indicators with matching lighter purple tones
        # boundary_colors = ['#6A5ACD', '#6A5ACD', '#8A2BE2', '#8A2BE2', '#9370DB', '#9370DB']
        # boundaries = [q40, q60, q30, q70, q20, q80]
        
        # for boundary, color in zip(boundaries, boundary_colors):
        #     ax_bardist.axvline(boundary, color=color, alpha=0.6, linewidth=1.2, linestyle='--')
        
        # Calculate statistics
        mean_val = criterion.mean(logits)[0].detach().cpu().numpy()
        median_val = criterion.median(logits)[0].detach().cpu().numpy()
        mean_val = float(mean_val.item()) if isinstance(mean_val, np.ndarray) and mean_val.size == 1 else float(mean_val)
        median_val = float(median_val.item()) if isinstance(median_val, np.ndarray) and median_val.size == 1 else float(median_val)
        
        # True function value
        true_val = f(X_single)[0]  # Use same input format as model prediction
        true_val = float(true_val.item()) if isinstance(true_val, np.ndarray) and true_val.size == 1 else float(true_val)
        
        # Plot statistics as super thin dashed lines
        ax_bardist.axvline(mean_val, color='darkorange', linewidth=0.8, linestyle='--', 
                          alpha=0.8, label=f'Mean: {mean_val:.1f}')
        ax_bardist.axvline(median_val, color='green', linewidth=0.8, linestyle='--', 
                          alpha=0.8, label=f'Med: {median_val:.1f}')
        ax_bardist.axvline(true_val, color='red', linewidth=0.8, linestyle='--', 
                          alpha=0.8, label=f'True: {true_val:.1f}')
        
        # Calculate quantiles for axis scaling
        q01 = criterion.icdf(logits, torch.tensor(0.01, device=logits.device))[0].detach().cpu().numpy()
        q99 = criterion.icdf(logits, torch.tensor(0.99, device=logits.device))[0].detach().cpu().numpy()
        q01 = float(q01.item()) if isinstance(q01, np.ndarray) and q01.size == 1 else float(q01)
        q99 = float(q99.item()) if isinstance(q99, np.ndarray) and q99.size == 1 else float(q99)
        
        # Set axis limits with padding
        range_width = q99 - q01
        padding = range_width * 0.1 if range_width > 0 else 1.0
        x_min = q01 - padding
        x_max = q99 + padding
        
        if true_val < x_min:
            x_min = true_val - padding
        if true_val > x_max:
            x_max = true_val + padding
        
        ax_bardist.set_xlim(x_min, x_max)
        
        # Labels and formatting
        ax_bardist.set_title(f'x = {x_point:.2f}', fontsize=11, fontweight='bold')
        
        # Only bottom row gets x-labels
        if row == bardist_rows - 1:
            ax_bardist.set_xlabel('Predicted Value', fontsize=10)
        
        # Only left column gets y-labels  
        if col == 0:
            ax_bardist.set_ylabel('Probability', fontsize=10)
            
        # Add compact legend for first subplot only
        if i == 0:
            ax_bardist.legend(fontsize=6, loc='upper right')
        else:
            ax_bardist.legend(fontsize=6, loc='upper right')
            
        ax_bardist.grid(True, alpha=0.3)
        ax_bardist.tick_params(labelsize=8)
    
    # ============================================================================
    # SAVE COMBINED FIGURE
    # ============================================================================
    plt.tight_layout()
    
    # Save the combined plot
    base_filename = f"tabpfn_samples_{n_samples}_temp_{softmax_temp:.2f}_estimators_{n_estimators}"
    combined_filename = f"{base_filename}_combined.png"
    plt.savefig(os.path.join(save_dir, combined_filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    return model

# Create plots directory
plots_dir = ensure_dir("tabpfn_plots")

# Define sample sizes and temperatures to test
sample_sizes = [5,6,7,8,9,10,15,20]
temperatures = [0.9]
n_estimators = [8]

# Calculate y-axis limits based on the true function with a 20% margin
lb = -6.0
ub = 6.0
X_true = np.atleast_2d(np.linspace(lb, ub, 500)).T
y_true = f(X_true)
# Calculate min and max with 20% margin
y_min_val = y_true.min()
y_max_val = y_true.max()
y_range = y_max_val - y_min_val
margin = 0.2 * y_range
y_min = y_min_val - margin
y_max = y_max_val + margin

# Generate plots for each combination
for n_samples in sample_sizes:
    # Generate samples once for each sample size
    np.random.seed(42)
    samples_normalized = lhs(1, samples=n_samples, criterion="center")
    xi = lb + (ub - lb) * samples_normalized
    fi = f(xi).ravel()
    
    for temp in temperatures:
        for n_est in n_estimators:
            generate_tabpfn_plot(n_samples, temp, n_est, plots_dir, xi=xi, fi=fi, y_min=y_min, y_max=y_max)

print(f"All plots saved to '{plots_dir}' directory.")
