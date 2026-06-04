"""
Python port of Connor's MATLAB Poisson solver, extended with interior Dirichlet pinning.

Solves: ∇²p = source  (default source=0)
with Neumann BCs on the domain boundary: ∂p/∂n = (gx, gy)·n
and optional Dirichlet pinning at selected interior points.

Uses ndgrid ordering throughout (axis 0 = x, axis 1 = y).
"""

import warnings
import numpy as np
import scipy.sparse
import scipy.sparse.linalg


def ndgrid_gradient(h, dx, dy):
    """
    Compute gradient of h in ndgrid ordering (axis 0 = x, axis 1 = y).

    numpy.gradient uses meshgrid ordering internally, so we transpose,
    differentiate, then transpose back — matching ndgrid_gradient.m.

    Parameters
    ----------
    h : ndarray of shape (nx, ny)
    dx, dy : float  grid spacings

    Returns
    -------
    gx, gy : ndarray of shape (nx, ny)
    """
    h_meshgrid = h.T
    gy_mesh, gx_mesh = np.gradient(h_meshgrid, dy, dx)
    return gx_mesh.T, gy_mesh.T


def poisson_solver(ds, gx, gy, domain_mask, pinned_points=None, source=None):
    """
    Reconstruct scalar field p from its gradient (gx, gy) on a 2D grid.

    Solves ∇²p = div(g) + source with:
      - Neumann BCs on the domain boundary (∂p/∂n = g·n, trapezoidal rule)
      - Optional Dirichlet pinning at interior points

    Parameters
    ----------
    ds : float
        Uniform grid spacing (dx = dy = ds).
    gx : ndarray of shape (nx, ny)
        x-component of the gradient field (ndgrid: axis 0 = x).
    gy : ndarray of shape (nx, ny)
        y-component of the gradient field.
    domain_mask : ndarray of bool, shape (nx, ny)
        True for active (participating) grid points.
    pinned_points : dict {(i, j): value} or None
        Grid indices (ndgrid, 0-based) to fix at specified values.
        If None or empty, one arbitrary active point is pinned to 0
        to resolve the constant-of-integration ambiguity.
    source : ndarray of shape (nx, ny) or None
        Explicit forcing term on the RHS. Added as ds² * source[active].
        Defaults to zero.

    Returns
    -------
    p : ndarray of shape (nx, ny)
        Solution field; NaN outside domain_mask.
    A : scipy.sparse.csr_matrix
        System matrix (before pinning rows have been applied).
    b : ndarray of shape (n,)
        RHS vector (after pinning has been applied).
    """
    nx, ny = gx.shape
    domain_flat = domain_mask.ravel(order='F')  # Fortran (column-major) to match MATLAB

    # --- index map: grid → compact integer (1-based to match MATLAB; 0 = absent) ---
    n = int(domain_mask.sum())
    index = np.zeros((nx, ny), dtype=np.int32)
    index[domain_mask] = np.arange(1, n + 1, dtype=np.int32)

    # --- neighbor index arrays (0 = no neighbor / outside domain) ---
    # N: y+1, E: x+1, S: y-1, W: x-1  (ndgrid: x=axis0, y=axis1)
    N = np.zeros((nx, ny), dtype=np.int32)
    E = np.zeros((nx, ny), dtype=np.int32)
    S = np.zeros((nx, ny), dtype=np.int32)
    W = np.zeros((nx, ny), dtype=np.int32)

    N[:, :-1] = index[:, 1:]   # neighbor at y+1
    E[:-1, :] = index[1:, :]   # neighbor at x+1
    S[:, 1:]  = index[:, :-1]  # neighbor at y-1
    W[1:, :]  = index[:-1, :]  # neighbor at x-1

    N_f = N[domain_mask]
    E_f = E[domain_mask]
    S_f = S[domain_mask]
    W_f = W[domain_mask]

    has_N = N_f != 0
    has_E = E_f != 0
    has_S = S_f != 0
    has_W = W_f != 0

    linear_index = np.arange(1, n + 1, dtype=np.int32)

    # --- sparse matrix in COO format (1-based; convert to 0-based for scipy) ---
    A_row = np.concatenate([
        linear_index,
        linear_index[has_N],
        linear_index[has_E],
        linear_index[has_S],
        linear_index[has_W],
    ]) - 1

    A_col = np.concatenate([
        linear_index,
        N_f[has_N],
        E_f[has_E],
        S_f[has_S],
        W_f[has_W],
    ]) - 1

    diag_vals = (has_N.astype(np.float64) + has_E.astype(np.float64)
                 + has_S.astype(np.float64) + has_W.astype(np.float64))

    A_val = np.concatenate([
        diag_vals,
        -np.ones(has_N.sum()),
        -np.ones(has_E.sum()),
        -np.ones(has_S.sum()),
        -np.ones(has_W.sum()),
    ])

    A = scipy.sparse.coo_matrix((A_val, (A_row, A_col)), shape=(n, n)).tocsr()

    # --- RHS: Neumann flux contributions at boundary faces ---
    # Flat indices into the full (nx*ny) array for active points
    array_index = np.flatnonzero(domain_flat)  # 0-based, Fortran order

    def _flat_idx(mask_into_active):
        return array_index[mask_into_active]

    def _neighbor_flat(neighbor_arr, mask_into_active):
        # neighbor_arr holds 1-based compact indices; convert to flat grid index
        nbr_compact = neighbor_arr[mask_into_active] - 1  # 0-based compact
        # We need the flat (Fortran) grid indices of these neighbor compact points
        # Rebuild the reverse map: compact index -> flat grid index
        return array_index[nbr_compact]

    # Build reverse map once: compact index (0-based) → flat grid index (Fortran)
    # array_index[k] is the flat grid index of compact point k
    gx_flat = gx.ravel(order='F')
    gy_flat = gy.ravel(order='F')

    b = np.zeros(n)

    # N face: y+1 neighbor exists → this point is NOT on the north boundary
    # Neumann contribution: -0.5*ds*(gy_center + gy_north)
    nbr_flat = array_index[N_f[has_N] - 1]
    b[has_N] -= 0.5 * ds * (gy_flat[_flat_idx(has_N)] + gy_flat[nbr_flat])

    # E face: x+1 neighbor exists
    nbr_flat = array_index[E_f[has_E] - 1]
    b[has_E] -= 0.5 * ds * (gx_flat[_flat_idx(has_E)] + gx_flat[nbr_flat])

    # S face: y-1 neighbor exists
    nbr_flat = array_index[S_f[has_S] - 1]
    b[has_S] += 0.5 * ds * (gy_flat[_flat_idx(has_S)] + gy_flat[nbr_flat])

    # W face: x-1 neighbor exists
    nbr_flat = array_index[W_f[has_W] - 1]
    b[has_W] += 0.5 * ds * (gx_flat[_flat_idx(has_W)] + gx_flat[nbr_flat])

    # --- optional source term ---
    if source is not None:
        source_flat = source.ravel(order='F')
        b += ds**2 * source_flat[array_index]

    # --- Dirichlet pinning ---
    if not pinned_points:
        # Fix constant of integration at the first active point
        first_active = tuple(int(v) for v in np.argwhere(domain_mask)[0])
        warnings.warn(
            f"No pinned_points supplied. Pinning point {first_active} to 0 "
            "to fix the constant of integration.",
            stacklevel=2,
        )
        pinned_points = {first_active: 0.0}

    A_lil = A.tolil()
    for (i, j), val in pinned_points.items():
        if not domain_mask[i, j]:
            raise ValueError(f"Pinned point ({i}, {j}) is outside domain_mask.")
        row = int(index[i, j]) - 1  # 0-based compact index
        A_lil.rows[row] = [row]
        A_lil.data[row] = [1.0]
        b[row] = float(val)

    A = A_lil.tocsr()

    # --- solve ---
    sol = scipy.sparse.linalg.spsolve(A, b)

    p = np.full((nx, ny), np.nan)
    p[domain_mask] = sol[index[domain_mask] - 1]

    return p, A, b
