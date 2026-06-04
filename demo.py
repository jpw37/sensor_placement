"""
Python port of demo.m — demonstrates the Poisson solver and interior pinning.

Run:  python demo.py
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from poisson_solver import ndgrid_gradient, poisson_solver

# --- grid ---
ds = 0.01
xmin, xmax = -1.0, 1.0
ymin, ymax = -1.0, 1.0
x = np.arange(xmin, xmax + ds / 2, ds)
y = np.arange(ymin, ymax + ds / 2, ds)
X, Y = np.meshgrid(x, y, indexing='ij')  # ndgrid ordering: axis 0 = x

# --- analytic field ---
L0 = 1.0
x0, y0 = -0.5, 0.0
r = np.sqrt((X - x0)**2 + (Y - y0)**2)
p_true = -np.exp(-(r**2) / L0**2)

gx_true, gy_true = ndgrid_gradient(p_true, ds, ds)

# --- add noise to the gradient ---
rng = np.random.default_rng(42)
var_x, var_y = 0.001, 0.001
gx_noisy = gx_true + np.sqrt(var_x) * rng.standard_normal(X.shape)
gy_noisy = gy_true + np.sqrt(var_y) * rng.standard_normal(X.shape)

domain_mask = np.ones(X.shape, dtype=bool)

# --- solve without explicit pinning (constant fixed at corner automatically) ---
print("Solving without explicit pinning …")
t0 = time.perf_counter()
P_unpinned, _, _ = poisson_solver(ds, gx_noisy, gy_noisy, domain_mask)
print(f"  elapsed: {time.perf_counter() - t0:.3f} s")

# shift to match true value at corner (0, 0) for comparison
P_unpinned -= P_unpinned[0, 0] - p_true[0, 0]

# --- solve with three pinned interior points ---
pin_ij = [(25, 50), (100, 100), (150, 75), (20, 10), (10,15), (20,5), (5, 150), (10,180), (20, 170), (180, 15), (185, 30), (190, 20), (100,50), (120, 80), (150, 150), (50,100), (60, 120)]
pinned = {ij: float(p_true[ij]) for ij in pin_ij}

print(f"Solving with {len(pinned)} pinned interior points …")
t0 = time.perf_counter()
P_pinned, _, _ = poisson_solver(ds, gx_noisy, gy_noisy, domain_mask,
                                pinned_points=pinned)
print(f"  elapsed: {time.perf_counter() - t0:.3f} s")

# verify pinned values
for ij, v in pinned.items():
    actual = P_pinned[ij]
    print(f"  pin {ij}: expected {v:.6f}, got {actual:.6f}, err {abs(actual - v):.2e}")

# --- metrics ---
err_unpinned = np.abs(P_unpinned - p_true)
err_pinned   = np.abs(P_pinned   - p_true)
print(f"\nMAE (unpinned): {err_unpinned.mean():.4f}")
print(f"MAE (pinned):   {err_pinned.mean():.4f}")

# --- plot ---
vmin = p_true.min()
vmax = p_true.max()

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

im0 = axes[0].pcolormesh(X, Y, p_true, shading='auto', vmin=vmin, vmax=vmax)
axes[0].set_title('True p')
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].pcolormesh(X, Y, P_unpinned, shading='auto', vmin=vmin, vmax=vmax)
axes[1].set_title('Reconstructed (no pins)')
plt.colorbar(im1, ax=axes[1])

im2 = axes[2].pcolormesh(X, Y, P_pinned, shading='auto', vmin=vmin, vmax=vmax)
axes[2].set_title(f'Reconstructed ({len(pinned)} pins)')
for ij in pin_ij:
    axes[2].plot(x[ij[0]], y[ij[1]], 'r+', markersize=10, markeredgewidth=2)
plt.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.savefig('demo_output.png', dpi=120)
print("\nPlot saved to demo_output.png")
plt.show()
