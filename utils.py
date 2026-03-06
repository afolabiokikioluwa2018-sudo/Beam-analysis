"""
Structural Analysis — Slope-Deflection Method (SDM)
====================================================

SIGN CONVENTIONS (consistent throughout):
  Moment  : Sagging = POSITIVE (tension at bottom)
            Hogging = NEGATIVE (tension at top)
  Shear   : Upward on LEFT face of section = POSITIVE
  Axial   : Tension = POSITIVE

SLOPE-DEFLECTION EQUATIONS:
  For member i→j with span L, EI constant:

    M_ij = (2EI/L)(2θ_i + θ_j − 3ψ) + FEM_ij   [near-end]
    M_ji = (2EI/L)(2θ_j + θ_i − 3ψ) + FEM_ji   [far-end]

  where:
    θ_i, θ_j  = joint rotations    (unknown — solved from ΣM=0)
    ψ          = chord rotation     = (δ_j⊥ − δ_i⊥) / L
    FEM        = fixed-end moments  (standard tables, hogging = −ve at near-end)

MEMBER FORCE RECOVERY (free-body diagram):
  Taking ΣM about j of the free body [0, L]:

    V_i · L + M_i − M_j − ΣW_moments_about_j = 0

    V_i = (M_j − M_i)/L + V_ss_i

  where V_ss_i = simply-supported reaction at i from loads alone.

  Vertical equilibrium:
    V_j = W_total − V_i

SFD FORMULA along member (at position x from i):
  V(x) = V_i − Σ [w·x  +  P·H(x−a)]       H = Heaviside step
        = V_i − (UDL × x) − P·(x>a)

BMD FORMULA along member (at position x from i):
  M(x) = M_i + V_i·x − Σ [w·x²/2  +  P·(x−a)·H(x−a)]
       = M_i + V_i·x − (UDL·x²/2) − P·max(x−a, 0)

  Sagging M(x) > 0 → tension at bottom  ← plot BELOW baseline
  Hogging  M(x) < 0 → tension at top    ← plot ABOVE baseline

References:
  Hibbeler R.C. — Structural Analysis 10th Ed., Ch.11
  Mosley, Bungey & Hulse — RC Design to BS 8110, App. B
"""

import numpy as np
import pandas as pd
from scipy.linalg import solve


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 1 — FIXED-END MOMENTS AND SHEARS
#  Standard formulas (Hibbeler Table 11-1).
#  Sign: FEM_ij negative for downward loads (hogging at near-end).
# ═══════════════════════════════════════════════════════════════════════════

def _fem_for_load(load, L):
    """
    Returns (FEM_ij, FEM_ji, FES_ij, FES_ji) for a single load.
    FEM: hogging = negative (SDM/sagging-positive convention)
    FES: upward  = positive
    """
    t = load['Type']

    if t == 'UDL':
        w = float(load['W1'])
        return (-w*L**2/12,   +w*L**2/12,   w*L/2,     w*L/2)

    elif t == 'VDL':
        w1, w2 = float(load['W1']), float(load['W2'])
        return (
            -(7*w1 + 3*w2)*L**2 / 60,
            +(3*w1 + 7*w2)*L**2 / 60,
             (7*w1 + 3*w2)*L   / 20,
             (3*w1 + 7*w2)*L   / 20,
        )

    elif t == 'Point Load':
        P = float(load['P'])
        a = float(load['a'])
        b = float(load['b']) if float(load['b']) > 0 else (L - a)
        return (
            -P*a*b**2 / L**2,
            +P*a**2*b / L**2,
             P*b**2*(3*a + b) / L**3,
             P*a**2*(a + 3*b) / L**3,
        )

    elif t == 'Moment':
        M = float(load['M'])
        a = float(load['a'])
        return (
            -M*(1 - a/L),
            +M*(a/L),
            +M/L,
            -M/L,
        )

    return (0.0, 0.0, 0.0, 0.0)


def _get_member_fem(member_id, loads_df, L):
    """Superimpose all load FEMs/FESs for one member."""
    fi = fj = vi = vj = 0.0
    for _, ld in loads_df[loads_df['Member'] == member_id].iterrows():
        dfi, dfj, dvi, dvj = _fem_for_load(ld, L)
        fi += dfi; fj += dfj; vi += dvi; vj += dvj
    return fi, fj, vi, vj


def _ss_shear_i(member_id, loads_df, L):
    """
    Simply-supported reaction at end i from applied loads.
    Used in shear recovery: V_i = (M_j−M_i)/L + V_ss_i
    """
    V = 0.0
    for _, ld in loads_df[loads_df['Member'] == member_id].iterrows():
        t = ld['Type']
        if t == 'UDL':
            V += float(ld['W1']) * L / 2
        elif t == 'VDL':
            V += (2*float(ld['W1']) + float(ld['W2'])) * L / 6
        elif t == 'Point Load':
            b = float(ld['b']) if float(ld['b']) > 0 else (L - float(ld['a']))
            V += float(ld['P']) * b / L
        elif t == 'Moment':
            V += -float(ld['M']) / L
    return V


def _total_load(member_id, loads_df, L):
    """Total downward transverse force on member."""
    W = 0.0
    for _, ld in loads_df[loads_df['Member'] == member_id].iterrows():
        t = ld['Type']
        if   t == 'UDL':         W += float(ld['W1']) * L
        elif t == 'VDL':         W += (float(ld['W1']) + float(ld['W2'])) * L / 2
        elif t == 'Point Load':  W += float(ld['P'])
    return W


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 2 — DOF MAP AND GEOMETRY
# ═══════════════════════════════════════════════════════════════════════════

def _build_dof_map(nodes_df):
    dof_map = {}
    for _, row in nodes_df.iterrows():
        n = int(row['Node'])
        dof_map[n] = {'dx': (n-1)*3, 'dy': (n-1)*3+1, 'rot': (n-1)*3+2}
    return dof_map


def _geom(member, nodes_df):
    ni = int(member['Node_I']); nj = int(member['Node_J'])
    xi = float(nodes_df[nodes_df['Node']==ni]['X'].values[0])
    yi = float(nodes_df[nodes_df['Node']==ni]['Y'].values[0])
    xj = float(nodes_df[nodes_df['Node']==nj]['X'].values[0])
    yj = float(nodes_df[nodes_df['Node']==nj]['Y'].values[0])
    L  = float(np.sqrt((xj-xi)**2 + (yj-yi)**2))
    c  = (xj-xi)/L;  s = (yj-yi)/L
    return xi, yi, xj, yj, L, c, s


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 3 — GLOBAL STIFFNESS MATRIX
#  SDM stiffness coefficients for Euler-Bernoulli beam element:
#    4EI/L = near-end bending stiffness
#    2EI/L = far-end  bending stiffness (carry-over)
#    6EI/L² = rotation-shear coupling
#    12EI/L³ = transverse shear stiffness
# ═══════════════════════════════════════════════════════════════════════════

def assemble_global_stiffness(nodes_df, members_df):
    dof_map = _build_dof_map(nodes_df)
    n_dof   = 3 * len(nodes_df)
    K       = np.zeros((n_dof, n_dof))

    for _, mb in members_df.iterrows():
        ni = int(mb['Node_I']); nj = int(mb['Node_J'])
        E  = float(mb['E']); I = float(mb['I']); A = float(mb['A'])
        _, _, _, _, L, c, s = _geom(mb, nodes_df)

        k4  = 4.0*E*I / L      # 4EI/L  — near-end rotational stiffness
        k2  = 2.0*E*I / L      # 2EI/L  — far-end  rotational stiffness
        k6  = 6.0*E*I / L**2   # 6EI/L² — rotation-sway coupling
        k12 = 12.0*E*I / L**3  # 12EI/L³— transverse shear stiffness
        ea  = E*A / L

        # Local stiffness [u_i, v_i, θ_i, u_j, v_j, θ_j]
        kL = np.array([
            [ ea,    0,    0,   -ea,    0,    0  ],
            [  0,  k12,   k6,    0,  -k12,   k6  ],
            [  0,   k6,   k4,    0,   -k6,   k2  ],
            [-ea,    0,    0,    ea,    0,    0  ],
            [  0, -k12,  -k6,    0,   k12,  -k6  ],
            [  0,   k6,   k2,    0,   -k6,   k4  ],
        ])

        # Transformation matrix (local ↔ global)
        T = np.zeros((6, 6))
        T[0:2, 0:2] = [[ c,  s], [-s,  c]]
        T[2,   2  ] = 1.0
        T[3:5, 3:5] = [[ c,  s], [-s,  c]]
        T[5,   5  ] = 1.0

        kG   = T.T @ kL @ T
        dofs = ([dof_map[ni]['dx'], dof_map[ni]['dy'], dof_map[ni]['rot']] +
                [dof_map[nj]['dx'], dof_map[nj]['dy'], dof_map[nj]['rot']])
        for r in range(6):
            for cc in range(6):
                K[dofs[r], dofs[cc]] += kG[r, cc]

    return K, dof_map


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 4 — EQUIVALENT NODAL LOAD VECTOR
#  SDM step 1: lock all joints → compute FEM in each member.
#  SDM step 2: release clamps → apply −FEM as external loads at joints.
# ═══════════════════════════════════════════════════════════════════════════

def calculate_fixed_end_actions(members_df, loads_df, dof_map, nodes_df):
    n_dof   = len(dof_map) * 3
    F_fixed = np.zeros(n_dof)
    if len(loads_df) == 0:
        return F_fixed

    for _, ld in loads_df.iterrows():
        mid = int(ld['Member'])
        mb  = members_df[members_df['Member']==mid].iloc[0]
        ni  = int(mb['Node_I']); nj = int(mb['Node_J'])
        _, _, _, _, L, c, s = _geom(mb, nodes_df)

        FEM_i, FEM_j, FES_i, FES_j = _fem_for_load(ld, L)

        # Upward FES in global coordinates: local-y → global (−sinθ, cosθ)
        F_fixed[dof_map[ni]['dx']]  += FES_i * (-s)
        F_fixed[dof_map[ni]['dy']]  += FES_i * c
        F_fixed[dof_map[ni]['rot']] += -FEM_i   # release step: −FEM
        F_fixed[dof_map[nj]['dx']]  += FES_j * (-s)
        F_fixed[dof_map[nj]['dy']]  += FES_j * c
        F_fixed[dof_map[nj]['rot']] += -FEM_j   # release step: −FEM

    return F_fixed


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 5 — BOUNDARY CONDITIONS
# ═══════════════════════════════════════════════════════════════════════════

def apply_boundary_conditions(K_global, F_fixed, supports_df, nodes_df, dof_map):
    n_dof      = K_global.shape[0]
    fixed_dofs = []
    prescribed = {}

    for _, sup in supports_df.iterrows():
        node  = int(sup['Node'])
        stype = sup['Type']
        d_dx  = dof_map[node]['dx']
        d_dy  = dof_map[node]['dy']
        d_rot = dof_map[node]['rot']

        if stype in ('Fixed', 'Cantilever'):
            fixed_dofs += [d_dx, d_dy, d_rot]
            prescribed[d_dx]  = float(sup['Dx'])
            prescribed[d_dy]  = float(sup['Dy'])
            prescribed[d_rot] = float(sup['Rotation'])
        elif stype == 'Pinned':
            fixed_dofs += [d_dx, d_dy]
            prescribed[d_dx] = float(sup['Dx'])
            prescribed[d_dy] = float(sup['Dy'])
        elif stype == 'Roller':
            fixed_dofs.append(d_dy)
            prescribed[d_dy] = float(sup['Dy'])

    fixed_dofs = list(set(fixed_dofs))
    free_dofs  = [i for i in range(n_dof) if i not in fixed_dofs]

    # Prescribed settlement correction
    F_settle = np.zeros(n_dof)
    for dof, delta in prescribed.items():
        if delta != 0.0:
            F_settle -= K_global[:, dof] * delta

    K_reduced = K_global[np.ix_(free_dofs, free_dofs)]
    F_reduced = (F_fixed + F_settle)[free_dofs]
    return K_reduced, F_reduced, fixed_dofs


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 6 — SOLVE JOINT DISPLACEMENTS (SDM: ΣM=0 at each free joint)
# ═══════════════════════════════════════════════════════════════════════════

def solve_displacements(K_reduced, F_reduced, fixed_dofs, n_nodes):
    n_dof         = 3 * n_nodes
    displacements = np.zeros(n_dof)
    free_dofs     = [i for i in range(n_dof) if i not in fixed_dofs]
    if len(free_dofs) > 0:
        displacements[free_dofs] = solve(K_reduced, F_reduced)
    return displacements


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 7 — MEMBER FORCE RECOVERY (SDM back-substitution)
#
#  STEP 1: Chord rotation
#    ψ = (δ_j⊥ − δ_i⊥) / L         [δ⊥ = local-y displacement]
#
#  STEP 2: Slope-deflection equations
#    M_ij = (2EI/L)(2θ_i + θ_j − 3ψ) + FEM_ij
#    M_ji = (2EI/L)(2θ_j + θ_i − 3ψ) + FEM_ji
#
#  STEP 3: Convert to sagging-positive convention
#    M_i =  M_ij   (CW at left end = hogging → already negative when hogging)
#    M_j = −M_ji   (negate: CW at right end = hogging from span perspective)
#
#  STEP 4: Shear from FBD (ΣM about j = 0)
#    V_i · L = M_j − M_i + ΣW_moment_about_j
#           (M_j − M_i accounts for end-moment contribution)
#    V_i = (M_j − M_i)/L + V_ss_i
#    V_j = W_total − V_i
#
#  REPORTED SIGN:
#    V_i > 0 → upward on LEFT  face  (positive shear convention)
#    V_j > 0 → upward on RIGHT face  (equilibrium reaction at j)
#    M_i > 0 → sagging at end i
#    M_j > 0 → sagging at end j
# ═══════════════════════════════════════════════════════════════════════════

def calculate_member_forces(members_df, nodes_df, displacements, loads_df, dof_map):
    results = []

    for _, mb in members_df.iterrows():
        mid = int(mb['Member'])
        ni  = int(mb['Node_I']); nj = int(mb['Node_J'])
        E   = float(mb['E']); I = float(mb['I']); A = float(mb['A'])
        _, _, _, _, L, c, s = _geom(mb, nodes_df)

        # Joint displacements
        dx_i = displacements[dof_map[ni]['dx']]
        dy_i = displacements[dof_map[ni]['dy']]
        th_i = displacements[dof_map[ni]['rot']]
        dx_j = displacements[dof_map[nj]['dx']]
        dy_j = displacements[dof_map[nj]['dy']]
        th_j = displacements[dof_map[nj]['rot']]

        # STEP 1: Chord rotation ψ = (δ_j⊥ − δ_i⊥) / L
        vi_perp = -s*dx_i + c*dy_i
        vj_perp = -s*dx_j + c*dy_j
        psi     = (vj_perp - vi_perp) / L

        # STEP 2: SDM slope-deflection equations
        k      = 2.0*E*I / L
        FEM_i, FEM_j, _, _ = _get_member_fem(mid, loads_df, L)

        M_ij = k*(2*th_i + th_j - 3*psi) + FEM_i
        M_ji = k*(2*th_j + th_i - 3*psi) + FEM_j

        # STEP 3: Convert to sagging-positive member convention
        M_i =  M_ij        # CW at i = hogging → negative  ✓
        M_j = -M_ji        # negate CW at j → hogging = negative  ✓

        # STEP 4: Shear recovery from FBD
        #   ΣM_j = 0:  V_i·L + M_i − M_j − ΣW_moments_j = 0
        #   V_i = (M_j − M_i)/L + V_ss_i
        V_ss_i  = _ss_shear_i(mid, loads_df, L)
        W_total = _total_load(mid, loads_df, L)

        V_i = (M_j - M_i) / L + V_ss_i
        V_j = W_total - V_i

        # Axial (tension positive)
        u_i  = c*dx_i + s*dy_i
        u_j  = c*dx_j + s*dy_j
        N_i  =  E*A/L * (u_j - u_i)
        N_j  = -N_i

        results.append({
            'Member'   : mid,
            'N_i (kN)' : round(N_i,  6),
            'V_i (kN)' : round(V_i,  6),
            'M_i (kNm)': round(M_i,  6),
            'N_j (kN)' : round(N_j,  6),
            'V_j (kN)' : round(V_j,  6),
            'M_j (kNm)': round(M_j,  6),
        })

    df = pd.DataFrame(results)
    # Short aliases (used by diagrams.py and main.py)
    df['N_i'] = df['N_i (kN)']
    df['V_i'] = df['V_i (kN)']
    df['M_i'] = df['M_i (kNm)']
    df['N_j'] = df['N_j (kN)']
    df['V_j'] = df['V_j (kN)']
    df['M_j'] = df['M_j (kNm)']
    return df


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 8 — SUPPORT REACTIONS
# ═══════════════════════════════════════════════════════════════════════════

def calculate_reactions(K_global, displacements, F_fixed, supports_df, nodes_df, dof_map):
    """
    R = F_fixed − K·δ
    Ry (+) = upward,  Rx (+) = rightward,  Mz (+) = anticlockwise
    """
    F_total = F_fixed - K_global @ displacements
    rows = []
    for _, sup in supports_df.iterrows():
        node = int(sup['Node'])
        rows.append({
            'Node'        : node,
            'Support Type': sup['Type'],
            'Rx (kN)'     : round( F_total[dof_map[node]['dx']],  6),
            'Ry (kN)'     : round( F_total[dof_map[node]['dy']],  6),
            'Mz (kNm)'    : round(-F_total[dof_map[node]['rot']], 6),
        })
    return pd.DataFrame(rows)
