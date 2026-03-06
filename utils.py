"""
Structural Analysis Utility Functions
Solver: Classical Slope-Deflection Method (SDM)

Theory Reference:
  Hibbeler, R.C. - Structural Analysis (10th Ed.) Ch. 11
  Mosley, Bungey & Hulse - Reinforced Concrete Design to BS 8110

Sign Convention (throughout this module):
  Moments  : SAGGING = POSITIVE (tension at bottom fiber)
             HOGGING = NEGATIVE (tension at top fiber)
  Shear    : Upward on left face of cut = POSITIVE
  Axial    : Tension = POSITIVE, Compression = NEGATIVE
  Chord    : Chord rotation psi = (delta_j_perp - delta_i_perp) / L

Slope-Deflection Equations (the heart of SDM):
  M_ij = (2EI/L)(2*theta_i + theta_j - 3*psi) + FEM_ij
  M_ji = (2EI/L)(2*theta_j + theta_i - 3*psi) + FEM_ji

  where:
    theta_i, theta_j = joint rotations (unknown — solved by joint equilibrium)
    psi              = chord rotation  (known from geometry / sway)
    FEM_ij, FEM_ji   = fixed-end moments (known from load tables)
"""

import numpy as np
import pandas as pd
from scipy.linalg import solve


# ===========================================================================
#  SECTION 1 — FIXED-END MOMENTS AND SHEARS
#  These are standard textbook formulas (Hibbeler Table 11-1, Mosley App. B)
# ===========================================================================

def _fem_for_load(load, L):
    """
    Fixed-end moments and shears for a single load on a fixed-fixed beam.

    Returns: FEM_ij, FEM_ji, FES_ij, FES_ji
      FEM_ij  = moment at near-end i  (negative = hogging, per SDM convention)
      FEM_ji  = moment at far-end  j  (positive = hogging at j, per SDM)
      FES_ij  = vertical reaction at i (positive = upward)
      FES_ji  = vertical reaction at j (positive = upward)
    """
    t = load['Type']

    if t == "UDL":
        w = load['W1']
        return (-w*L**2/12,  w*L**2/12,  w*L/2,  w*L/2)

    elif t == "VDL":
        w1, w2 = load['W1'], load['W2']
        return (
            -(7*w1 + 3*w2)*L**2/60,
             (3*w1 + 7*w2)*L**2/60,
             (7*w1 + 3*w2)*L/20,
             (3*w1 + 7*w2)*L/20
        )

    elif t == "Point Load":
        P = load['P']
        a = load['a']
        b = load['b'] if load['b'] > 0 else (L - a)
        return (
            -P*a*b**2/L**2,
             P*a**2*b/L**2,
             P*b**2*(3*a + b)/L**3,
             P*a**2*(a + 3*b)/L**3
        )

    elif t == "Moment":
        M = load['M']
        a = load['a']
        return (
            -M*(1 - a/L),
             M*(a/L),
             M/L,
            -M/L
        )

    return (0.0, 0.0, 0.0, 0.0)


def get_member_fem(member_id, loads_df, L):
    """
    Superimpose all loads on a member to get total FEM and FES.
    Returns: (FEM_i, FEM_j, FES_i, FES_j)
    """
    fi = fj = vi = vj = 0.0
    for _, load in loads_df[loads_df['Member'] == member_id].iterrows():
        dfi, dfj, dvi, dvj = _fem_for_load(load, L)
        fi += dfi;  fj += dfj
        vi += dvi;  vj += dvj
    return fi, fj, vi, vj


def _ss_shear_at_i(loads_df, member_id, L):
    """
    Simply-supported reaction at end i due to applied loads.
    (Used in SDM shear recovery — NOT the fixed-end shear.)
    """
    V = 0.0
    for _, load in loads_df[loads_df['Member'] == member_id].iterrows():
        t = load['Type']
        if t == "UDL":
            V += load['W1'] * L / 2
        elif t == "VDL":
            V += (2*load['W1'] + load['W2']) * L / 6
        elif t == "Point Load":
            b = load['b'] if load['b'] > 0 else (L - load['a'])
            V += load['P'] * b / L
        elif t == "Moment":
            V += -load['M'] / L
    return V


def _total_transverse_load(loads_df, member_id, L):
    """Total downward force on member (for vertical equilibrium check)."""
    W = 0.0
    for _, load in loads_df[loads_df['Member'] == member_id].iterrows():
        t = load['Type']
        if   t == "UDL":        W += load['W1'] * L
        elif t == "VDL":        W += (load['W1'] + load['W2']) * L / 2
        elif t == "Point Load": W += load['P']
    return W


# ===========================================================================
#  SECTION 2 — BUILD DOF MAP AND GEOMETRY HELPERS
# ===========================================================================

def _build_dof_map(nodes_df):
    """Map each node to its 3 global DOF indices: dx, dy, rot."""
    dof_map = {}
    for _, row in nodes_df.iterrows():
        n = int(row['Node'])
        dof_map[n] = {
            'dx' : (n-1)*3,
            'dy' : (n-1)*3 + 1,
            'rot': (n-1)*3 + 2
        }
    return dof_map


def _geom(member, nodes_df):
    """Return (xi,yi,xj,yj, L, cos_theta, sin_theta) for a member."""
    ni = int(member['Node_I']);  nj = int(member['Node_J'])
    xi = nodes_df[nodes_df['Node']==ni]['X'].values[0]
    yi = nodes_df[nodes_df['Node']==ni]['Y'].values[0]
    xj = nodes_df[nodes_df['Node']==nj]['X'].values[0]
    yj = nodes_df[nodes_df['Node']==nj]['Y'].values[0]
    L  = np.sqrt((xj-xi)**2 + (yj-yi)**2)
    return xi, yi, xj, yj, L, (xj-xi)/L, (yj-yi)/L


# ===========================================================================
#  SECTION 3 — GLOBAL STIFFNESS MATRIX
#  Assembled from SDM element stiffness matrices.
#
#  The 6x6 local stiffness matrix is DERIVED directly from SDM equations:
#
#    SDM:  M_ij = (4EI/L)*theta_i + (2EI/L)*theta_j + (6EI/L^2)*delta + FEM_ij
#
#  Recognising:
#    4EI/L = 2k  (near-end bending stiffness)
#    2EI/L = k   (far-end  bending stiffness, where k = 2EI/L)
#    6EI/L^2     (chord / sway coupling)
#    12EI/L^3    (transverse shear stiffness = 2 * 6EI/L^2 / L)
#
#  The assembled matrix is identical to the Euler-Bernoulli beam element —
#  the SDM is simply the physical interpretation of these same coefficients.
# ===========================================================================

def assemble_global_stiffness(nodes_df, members_df):
    """
    Assemble global stiffness matrix using SDM-derived element matrices.

    Returns: K_global (3n x 3n),  dof_map
    """
    dof_map = _build_dof_map(nodes_df)
    n_dof   = 3 * len(nodes_df)
    K       = np.zeros((n_dof, n_dof))

    for _, member in members_df.iterrows():
        ni = int(member['Node_I']);  nj = int(member['Node_J'])
        E  = member['E'];  I = member['I'];  A = member['A']
        _, _, _, _, L, c, s = _geom(member, nodes_df)

        # SDM stiffness coefficients
        k    = 2.0*E*I / L          # 2EI/L  (SDM near-end factor / 2)
        s3L  = 6.0*E*I / L**2       # 6EI/L^2  (sway-rotation coupling)
        s12  = 12.0*E*I / L**3      # 12EI/L^3 (transverse shear)
        ea   = E*A / L              # axial

        # Local stiffness [u_i, v_i, th_i, u_j, v_j, th_j]
        # Diagonal bending terms: 4EI/L = 2k (near), 2EI/L = k (far)
        kL = np.array([
            [ ea,   0,    0,   -ea,   0,    0  ],
            [  0,  s12,  s3L,   0,  -s12,  s3L ],
            [  0,  s3L,  2*k,   0,  -s3L,   k  ],
            [-ea,   0,    0,    ea,   0,    0  ],
            [  0, -s12, -s3L,   0,   s12, -s3L ],
            [  0,  s3L,   k,   0,  -s3L,  2*k  ]
        ])

        # Rotation transformation  (local -> global)
        T = np.zeros((6,6))
        T[0:2,0:2] = [[ c, s],[-s, c]]
        T[2,2] = 1.0
        T[3:5,3:5] = [[ c, s],[-s, c]]
        T[5,5] = 1.0

        kG   = T.T @ kL @ T
        dofs = ([dof_map[ni]['dx'], dof_map[ni]['dy'], dof_map[ni]['rot']] +
                [dof_map[nj]['dx'], dof_map[nj]['dy'], dof_map[nj]['rot']])
        for r in range(6):
            for cc in range(6):
                K[dofs[r], dofs[cc]] += kG[r, cc]

    return K, dof_map


# ===========================================================================
#  SECTION 4 — EQUIVALENT NODAL LOAD VECTOR  (SDM release step)
#
#  SDM procedure:
#    1. Lock all joints (fix every rotation and sway).
#    2. Compute FEM / FES in each member due to applied loads.
#    3. The imaginary clamps at each joint carry the unbalanced FEM.
#    4. Release clamps: apply NEGATIVE of clamp forces as external loads.
#       => This is the equivalent nodal load vector F_fixed.
# ===========================================================================

def calculate_fixed_end_actions(members_df, loads_df, dof_map, nodes_df):
    """
    Build the equivalent nodal load vector from SDM fixed-end actions.
    Returns: F_fixed  (n_dof,)
    """
    n_dof   = len(dof_map) * 3
    F_fixed = np.zeros(n_dof)
    if len(loads_df) == 0:
        return F_fixed

    for _, load in loads_df.iterrows():
        mid    = int(load['Member'])
        member = members_df[members_df['Member'] == mid].iloc[0]
        ni = int(member['Node_I']);  nj = int(member['Node_J'])
        _, _, _, _, L, c, s = _geom(member, nodes_df)

        FEM_i, FEM_j, FES_i, FES_j = _fem_for_load(load, L)

        # Transform upward shear reactions to global Cartesian
        # Local-y direction (perpendicular to member, upward) in global:
        #   x_global = -sin(theta),  y_global = cos(theta)
        Fi_x = FES_i * (-s);  Fi_y = FES_i * c
        Fj_x = FES_j * (-s);  Fj_y = FES_j * c

        F_fixed[dof_map[ni]['dx']]  += Fi_x
        F_fixed[dof_map[ni]['dy']]  += Fi_y
        F_fixed[dof_map[ni]["rot"]] += -FEM_i
        F_fixed[dof_map[nj]['dx']]  += Fj_x
        F_fixed[dof_map[nj]['dy']]  += Fj_y
        F_fixed[dof_map[nj]["rot"]] += -FEM_j

    return F_fixed


# ===========================================================================
#  SECTION 5 — BOUNDARY CONDITIONS  (SDM: restrain fixed joints)
# ===========================================================================

def apply_boundary_conditions(K_global, F_fixed, supports_df, nodes_df, dof_map):
    """
    Apply support restraints and prescribed settlements.

    SDM treatment of support types:
      Fixed    : theta=0, Dx=0, Dy=delta_y  (all 3 DOFs restrained)
      Pinned   : Dx=0, Dy=delta_y           (2 DOFs restrained, theta FREE)
      Roller   : Dy=delta_y                 (1 DOF  restrained)
      Cantilever: same as Fixed

    Prescribed settlement handled by partition:
      F_free -= K_free,restrained * delta_restrained

    Returns: K_reduced, F_reduced, fixed_dofs
    """
    n_dof      = K_global.shape[0]
    fixed_dofs = []
    prescribed = {}

    for _, sup in supports_df.iterrows():
        node  = int(sup['Node'])
        stype = sup['Type']
        d_dx  = dof_map[node]['dx']
        d_dy  = dof_map[node]['dy']
        d_rot = dof_map[node]['rot']

        if stype in ("Fixed", "Cantilever"):
            fixed_dofs += [d_dx, d_dy, d_rot]
            prescribed[d_dx]  = sup['Dx']
            prescribed[d_dy]  = sup['Dy']
            prescribed[d_rot] = sup['Rotation']

        elif stype == "Pinned":
            fixed_dofs += [d_dx, d_dy]
            prescribed[d_dx] = sup['Dx']
            prescribed[d_dy] = sup['Dy']

        elif stype == "Roller":
            fixed_dofs.append(d_dy)
            prescribed[d_dy] = sup['Dy']

    fixed_dofs = list(set(fixed_dofs))
    free_dofs  = [i for i in range(n_dof) if i not in fixed_dofs]

    # Settlement correction: move prescribed-displacement columns to RHS
    F_settle = np.zeros(n_dof)
    for dof, delta in prescribed.items():
        if delta != 0.0:
            F_settle -= K_global[:, dof] * delta

    K_reduced = K_global[np.ix_(free_dofs, free_dofs)]
    F_reduced = (F_fixed + F_settle)[free_dofs]
    return K_reduced, F_reduced, fixed_dofs


# ===========================================================================
#  SECTION 6 — SOLVE FOR JOINT DISPLACEMENTS / ROTATIONS
#
#  In pure SDM this is solving the simultaneous equations:
#    ΣM = 0  at each free joint    → θ unknowns
#    ΣH = 0  for each free storey  → Δ (sway) unknowns
#
#  In matrix form:  K_reduced * delta_free = F_reduced
# ===========================================================================

def solve_displacements(K_reduced, F_reduced, fixed_dofs, n_nodes):
    """
    Solve SDM simultaneous equations for unknown rotations and sway.
    Returns: full displacement vector (zeros at restrained DOFs)
    """
    n_dof         = 3 * n_nodes
    displacements = np.zeros(n_dof)
    free_dofs     = [i for i in range(n_dof) if i not in fixed_dofs]
    if len(free_dofs) > 0:
        displacements[free_dofs] = solve(K_reduced, F_reduced)
    return displacements


# ===========================================================================
#  SECTION 7 — SDM BACK-SUBSTITUTION: MEMBER END FORCES
#
#  Once theta and psi are known, substitute into the slope-deflection
#  equations to recover member end moments.  Then use moment equilibrium
#  to find shear forces.
#
#  The SDM equations:
#    M_ij = (2EI/L)(2*theta_i + theta_j - 3*psi) + FEM_ij   ...(near end)
#    M_ji = (2EI/L)(2*theta_j + theta_i - 3*psi) + FEM_ji   ...(far  end)
#
#  Chord rotation:
#    psi = (delta_j_perp - delta_i_perp) / L
#    where delta_perp = perpendicular (local-y) component of global displacement
#
#  Shear recovery (from free-body diagram of member):
#    Take moments about j:
#      V_i * L  =  M_ij + M_ji  +  Sigma(load moments about j)
#    => V_i  =  (M_ij + M_ji)/L  +  V_ss_i
#
#    where V_ss_i is the simply-supported reaction at i due to loads alone.
#
#    Vertical equilibrium:
#      V_j  =  W_total - V_i   (both upward positive)
#
#  Sign Convention reported to user:
#    M_i (+) = SAGGING at end i  (tension at bottom)
#    M_j (+) = SAGGING at end j
#    V_i (+) = upward on left face (at i)
#    V_j (+) = upward on right face (at j)  [same direction as V_i for no-load case]
#    N   (+) = TENSION
# ===========================================================================

def calculate_member_forces(members_df, nodes_df, displacements, loads_df, dof_map):
    """
    SDM Back-Substitution to recover member end forces.
    See module docstring for full derivation.
    """
    results = []

    for _, member in members_df.iterrows():
        mid = int(member['Member'])
        ni  = int(member['Node_I'])
        nj  = int(member['Node_J'])
        E   = member['E'];  I = member['I'];  A = member['A']
        _, _, _, _, L, c, s = _geom(member, nodes_df)

        # --- Joint displacements (global) ---
        dx_i = displacements[dof_map[ni]['dx']]
        dy_i = displacements[dof_map[ni]['dy']]
        th_i = displacements[dof_map[ni]['rot']]
        dx_j = displacements[dof_map[nj]['dx']]
        dy_j = displacements[dof_map[nj]['dy']]
        th_j = displacements[dof_map[nj]['rot']]

        # --- Chord rotation psi (SDM) ---
        # Local-y (perpendicular) displacements: v = -sin*dx + cos*dy
        vi_perp = -s*dx_i + c*dy_i
        vj_perp = -s*dx_j + c*dy_j
        psi     = (vj_perp - vi_perp) / L

        # --- SDM slope-deflection equations ---
        k      = 2.0 * E * I / L          # SDM factor (2EI/L)
        FEM_i, FEM_j, _, _ = get_member_fem(mid, loads_df, L)

        # End moments (clockwise positive at each end, per SDM convention)
        M_ij = k*(2*th_i + th_j - 3*psi) + FEM_i
        M_ji = k*(2*th_j + th_i - 3*psi) + FEM_j

        # --- Convert to sagging-positive convention ---
        # M_ij: clockwise at end i.  For a beam with downward loads,
        #   hogging at i => M_ij < 0 when FEM_i < 0 (UDL case: FEM_i=-wL^2/12)
        #   sagging in span => positive values mid-span.
        # The SDM CW convention at end i already matches "hogging = negative"
        # when looking at the LEFT end of the member in the standard orientation.
        # At end j the CW convention gives the SAME physical direction,
        # so we negate M_ji to switch to "sagging positive at right end".
        M_i =  M_ij     # sagging positive (hogging is negative)
        M_j = -M_ji     # negate: CW at j = hogging at j from span perspective

        # --- Shear back-substitution (FBD of member) ---
        # Taking ΣM about j = 0 (upward forces +ve, CCW moments +ve at cut):
        #   V_i * L  +  M_i  -  M_j  -  (load moments about j)  =  0
        #   V_i = (M_j - M_i) / L  +  V_ss_i
        # where V_ss_i = simply-supported reaction at i from loads alone.
        V_ss_i  = _ss_shear_at_i(loads_df, mid, L)
        W_total = _total_transverse_load(loads_df, mid, L)

        V_i = (M_j - M_i) / L + V_ss_i
        V_j =  W_total - V_i     # vertical equilibrium: V_i + V_j = total downward load

        # --- Axial force (from local axial deformation) ---
        u_i = c*dx_i + s*dy_i    # local-x displacement at i
        u_j = c*dx_j + s*dy_j    # local-x displacement at j
        N_i =  E*A/L * (u_j - u_i)   # tension positive
        N_j = -N_i

        results.append({
            'Member'   : mid,
            'N_i (kN)' : round(N_i, 4),
            'V_i (kN)' : round(V_i, 4),
            'M_i (kNm)': round(M_i, 4),
            'N_j (kN)' : round(N_j, 4),
            'V_j (kN)' : round(V_j, 4),
            'M_j (kNm)': round(M_j, 4),
        })

    df = pd.DataFrame(results)
    # Short-name aliases for backward compatibility with diagrams.py
    df['N_i'] = df['N_i (kN)']
    df['V_i'] = df['V_i (kN)']
    df['M_i'] = df['M_i (kNm)']
    df['N_j'] = df['N_j (kN)']
    df['V_j'] = df['V_j (kN)']
    df['M_j'] = df['M_j (kNm)']
    return df


# ===========================================================================
#  SECTION 8 — SUPPORT REACTIONS
# ===========================================================================

def calculate_reactions(K_global, displacements, F_fixed, supports_df, nodes_df, dof_map):
    """
    Calculate support reactions from global equilibrium:
        R = K * delta - F_fixed

    Sign convention:
        Rx (+) = rightward
        Ry (+) = upward
        Mz (+) = anticlockwise
    """
    F_total = F_fixed - K_global @ displacements
    rows = []
    for _, sup in supports_df.iterrows():
        node = int(sup['Node'])
        rows.append({
            'Node'        : node,
            'Support Type': sup['Type'],
            'Rx (kN)'     : round( F_total[dof_map[node]['dx']],  4),
            'Ry (kN)'     : round( F_total[dof_map[node]['dy']],  4),
            'Mz (kNm)'    : round(-F_total[dof_map[node]['rot']], 4),  # negate: report reaction convention
        })
    return pd.DataFrame(rows)
