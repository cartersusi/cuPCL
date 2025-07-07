import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Data extracted from benchmark results
algorithms = ['PassThrough', 'VoxelGrid', 'ICP', 'Approx Nearest\nSearch', 'Radius Search', 'Segmentation']
cuda_times = [0.741079, 6.05894, 87.9349, 11.2881, 0.13354, 20.7557]  # ms
pcl_times = [1.4146, 5.49534, 426.742, 7.50806, 0.985471, 32.0006]    # ms

# Calculate speedup ratios (PCL/CUDA)
speedup_ratios = [pcl_times[i] / cuda_times[i] for i in range(len(cuda_times))]

# Create figure with subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Point Cloud Processing: CUDA vs PCL Performance Comparison\n(NVIDIA Orin, 119,978 points)', 
             fontsize=16, fontweight='bold')

# 1. Side-by-side bar chart of execution times
x = np.arange(len(algorithms))
width = 0.35

bars1 = ax1.bar(x - width/2, cuda_times, width, label='CUDA (GPU)', color='#2E8B57', alpha=0.8)
bars2 = ax1.bar(x + width/2, pcl_times, width, label='PCL (CPU)', color='#CD853F', alpha=0.8)

ax1.set_xlabel('Algorithm')
ax1.set_ylabel('Execution Time (ms)')
ax1.set_title('Execution Time Comparison')
ax1.set_xticks(x)
ax1.set_xticklabels(algorithms, rotation=45, ha='right')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}', ha='center', va='bottom', fontsize=8)

for bar in bars2:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}', ha='center', va='bottom', fontsize=8)

# 2. Log scale comparison for better visualization of large differences
ax2.bar(x - width/2, cuda_times, width, label='CUDA (GPU)', color='#2E8B57', alpha=0.8)
ax2.bar(x + width/2, pcl_times, width, label='PCL (CPU)', color='#CD853F', alpha=0.8)
ax2.set_xlabel('Algorithm')
ax2.set_ylabel('Execution Time (ms) - Log Scale')
ax2.set_title('Execution Time Comparison (Log Scale)')
ax2.set_xticks(x)
ax2.set_xticklabels(algorithms, rotation=45, ha='right')
ax2.set_yscale('log')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Speedup ratios
colors = ['green' if ratio > 1 else 'red' for ratio in speedup_ratios]
bars3 = ax3.bar(algorithms, speedup_ratios, color=colors, alpha=0.7)
ax3.set_xlabel('Algorithm')
ax3.set_ylabel('Speedup Ratio (PCL Time / CUDA Time)')
ax3.set_title('CUDA Speedup Over PCL\n(>1 = CUDA faster, <1 = PCL faster)')
ax3.tick_params(axis='x', rotation=45)
ax3.grid(True, alpha=0.3)
ax3.axhline(y=1, color='black', linestyle='--', alpha=0.5)

# Add value labels on speedup bars
for bar, ratio in zip(bars3, speedup_ratios):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{ratio:.1f}x', ha='center', va='bottom', fontsize=9, fontweight='bold')

# 4. Performance summary table
performance_data = {
    'Algorithm': algorithms,
    'CUDA (ms)': [f'{t:.2f}' for t in cuda_times],
    'PCL (ms)': [f'{t:.2f}' for t in pcl_times],
    'Speedup': [f'{r:.1f}x' for r in speedup_ratios]
}

# Create table
ax4.axis('tight')
ax4.axis('off')
table = ax4.table(cellText=[[performance_data[col][i] for col in performance_data.keys()] 
                           for i in range(len(algorithms))],
                 colLabels=list(performance_data.keys()),
                 cellLoc='center',
                 loc='center')

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Color code the speedup column
for i in range(len(algorithms)):
    if speedup_ratios[i] > 1:
        table[(i+1, 3)].set_facecolor('#90EE90')  # Light green for CUDA faster
    else:
        table[(i+1, 3)].set_facecolor('#FFB6C1')  # Light red for PCL faster

ax4.set_title('Performance Summary', fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('cuda_vs_pcl_performance_comparison.png', dpi=300)

# Print summary statistics
print("\n=== PERFORMANCE SUMMARY ===")
print(f"Total algorithms tested: {len(algorithms)}")
print(f"CUDA wins: {sum(1 for r in speedup_ratios if r > 1)}")
print(f"PCL wins: {sum(1 for r in speedup_ratios if r < 1)}")
print(f"Best CUDA speedup: {max(speedup_ratios):.1f}x (Radius Search)")
print(f"Worst CUDA performance: {min(speedup_ratios):.1f}x (Approx Nearest Search)")
print(f"Average CUDA speedup: {np.mean(speedup_ratios):.1f}x")

print("\n=== ALGORITHM ANALYSIS ===")
for i, alg in enumerate(algorithms):
    status = "CUDA faster" if speedup_ratios[i] > 1 else "PCL faster"
    print(f"{alg}: {speedup_ratios[i]:.1f}x speedup ({status})")