import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd

# Parameters
n = 30  # Dimension of the matrix
r = 2  # Rank of the ground truth matrix
kappa_list = [10,100,1000]  # Condition number(s) for the ground truth matrix
m = 3 * n * r  # Number of measurements

# Optimization settings
T = 10000  # Maximum number of iterations
eta = 0.5  # Learning rate
thresh_up = 1e10  # Upper threshold for error (divergence)
thresh_low = 1e-14  # Lower threshold for error (convergence)

# Initialize error storage
errors_GD = np.zeros((len(kappa_list), T))
errors_AGN = np.zeros((len(kappa_list), T))

# Generate ground truth matrix X_star
U_seed = np.sign(np.random.rand(n, r) - 0.5)
U_star, _, _ = svd(U_seed, full_matrices=False)
U_star = U_star[:, :r]  # Orthogonalize U_star

V_seed = np.sign(np.random.rand(n, r) - 0.5)
V_star, _, _ = svd(V_seed, full_matrices=False)
V_star = V_star[:, :r]  # Orthogonalize V_star

# Generate measurement matrices {A_k}
As = [np.random.randn(n, n) / np.sqrt(m) for _ in range(m)]

# Main loop over condition numbers
for i_kappa, kappa in enumerate(kappa_list):
    sigma_star = np.linspace(1, 1 / kappa, r)  # Singular values of X_star
    L_star = U_star @ np.diag(np.sqrt(sigma_star))  # Left factor of X_star
    R_star = V_star @ np.diag(np.sqrt(sigma_star))  # Right factor of X_star
    X_star = L_star @ R_star.T  # Ground truth matrix

    # Generate measurements y = A_k(X_star)
    y = np.zeros(m)
    for k in range(m):
        y[k] = np.sum(As[k] * X_star)

    # Spectral initialization
    Y = np.zeros((n, n))
    for k in range(m):
        Y += y[k] * As[k]
    d = 2 * r  # Overparameterization factor

    # Gradient Descent (GD)
    L = np.random.randn(n, d) / 10  # Initialize L
    R = np.random.randn(n, d) / 10  # Initialize R

    for t in range(T):
        X = L @ R.T  # Current estimate
        error = np.linalg.norm(X - X_star, 'fro')  # Frobenius norm error
        errors_GD[i_kappa, t] = error

        # Check for convergence or divergence
        if not np.isfinite(error) or error > thresh_up or error < thresh_low:
            break

        # Compute gradient update
        Z = np.zeros((n, n))
        for k in range(m):
            Z += (np.sum(As[k] * X) - y[k]) * As[k]
        L_plus = L - eta * Z @ R  # Update L
        R_plus = R - eta * Z.T @ L  # Update R
        L, R = L_plus, R_plus

    # Approximate Gauss-Newton (AGN)
    L = np.random.randn(n, d) / 10  # Reinitialize L
    R = np.random.randn(n, d) / 10  # Reinitialize R

    for t in range(T):
        X = L @ R.T  # Current estimate
        error = np.linalg.norm(X - X_star, 'fro')  # Frobenius norm error
        errors_AGN[i_kappa, t] = error

        # Check for convergence or divergence
        if not np.isfinite(error) or error > thresh_up or error < thresh_low:
            break

        # Compute AGN update for L
        Z = np.zeros((n, n))
        for k in range(m):
            Z += (np.sum(As[k] * X) - y[k]) * As[k]
        Delta_L = np.linalg.lstsq(R, Z.T, rcond=None)[0]  # Solve for Delta_L
        L = L - eta * Delta_L.T  # Update L

        # Compute AGN update for R
        X = L @ R.T
        Z = np.zeros((n, n))
        for k in range(m):
            Z += (np.sum(As[k] * X) - y[k]) * As[k]
        Delta_R = np.linalg.lstsq(L, Z, rcond=None)[0]  # Solve for Delta_R
        R = R - eta * Delta_R  # Update R

# Plot results
clrs = [(0.5, 0, 0.5), (1, 0.5, 0), (1, 0, 0), (0, 0.5, 0), (0, 0, 1)]  # Colors for plotting
mks = ['o', 'x', 'p', 's', 'd']  # Markers for plotting
plt.figure(figsize=(10, 8))
lgd = []

# Plot GD errors
for i_kappa, kappa in enumerate(kappa_list):
    errors = errors_GD[i_kappa, :]
    errors = errors[errors > thresh_low]  # Filter out converged/diverged points
    t_subs = np.arange(1, len(errors) + 1)
    plt.semilogy(t_subs - 1, errors, color=clrs[0], marker=mks[i_kappa], markersize=9)
    lgd.append(f'GD κ={kappa}')

# Plot AGN errors
for i_kappa, kappa in enumerate(kappa_list):
    errors = errors_AGN[i_kappa, :]
    errors = errors[errors > thresh_low]  # Filter out converged/diverged points
    t_subs = np.arange(1, len(errors) + 1)
    plt.semilogy(t_subs - 1, errors, color=clrs[1], marker=mks[i_kappa], markersize=9)
    lgd.append(f'AGN κ={kappa}')

# Add labels and legend
plt.xlabel('Iteration count')
plt.ylabel('Relative error')
plt.legend(lgd, loc='upper right', fontsize=12)
plt.grid(True)

# Save figure
fig_name = f'MS_n={n}_r={r}_m={m}.png'
plt.savefig(fig_name)
plt.show()
