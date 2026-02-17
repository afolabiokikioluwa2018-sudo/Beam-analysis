"""
Utility functions for structural analysis
Includes stiffness matrix assembly, FEM calculations, and solvers
"""

import numpy as np
import pandas as pd
from scipy.linalg import solve

def assemble_global_stiffness(nodes_df, members_df):
    """
    Assemble the global stiffness matrix using Direct Stiffness Method
    
    Returns:
        K_global: Global stiffness matrix
        dof_map: Dictionary mapping node numbers to DOF indices
    """
    n_nodes = len(nodes_df)
    n_dof = 3 * n_nodes  # 3 DOF per node (Dx, Dy, Rotation)
    K_global = np.zeros((n_dof, n_dof))
    
    # Create DOF mapping
    dof_map = {}
    for idx, row in nodes_df.iterrows():
        node_num = int(row['Node'])
        dof_map[node_num] = {
            'dx': (node_num - 1) * 3,
            'dy': (node_num - 1) * 3 + 1,
            'rot': (node_num - 1) * 3 + 2
        }
    
    # Assemble member stiffness matrices
    for idx, member in members_df.iterrows():
        node_i = int(member['Node_I'])
        node_j = int(member['Node_J'])
        E = member['E']
        I = member['I']
        A = member['A']
        
        # Get node coordinates
        xi = nodes_df[nodes_df['Node'] == node_i]['X'].values[0]
        yi = nodes_df[nodes_df['Node'] == node_i]['Y'].values[0]
        xj = nodes_df[nodes_df['Node'] == node_j]['X'].values[0]
        yj = nodes_df[nodes_df['Node'] == node_j]['Y'].values[0]
        
        # Member length and orientation
        L = np.sqrt((xj - xi)**2 + (yj - yi)**2)
        cos_theta = (xj - xi) / L
        sin_theta = (yj - yi) / L
        
        # Local stiffness matrix for beam element
        k_local = np.array([
            [E*A/L, 0, 0, -E*A/L, 0, 0],
            [0, 12*E*I/L**3, 6*E*I/L**2, 0, -12*E*I/L**3, 6*E*I/L**2],
            [0, 6*E*I/L**2, 4*E*I/L, 0, -6*E*I/L**2, 2*E*I/L],
            [-E*A/L, 0, 0, E*A/L, 0, 0],
            [0, -12*E*I/L**3, -6*E*I/L**2, 0, 12*E*I/L**3, -6*E*I/L**2],
            [0, 6*E*I/L**2, 2*E*I/L, 0, -6*E*I/L**2, 4*E*I/L]
        ])
        
        # Transformation matrix
        T = np.zeros((6, 6))
        T[0:2, 0:2] = [[cos_theta, sin_theta], [-sin_theta, cos_theta]]
        T[2, 2] = 1
        T[3:5, 3:5] = [[cos_theta, sin_theta], [-sin_theta, cos_theta]]
        T[5, 5] = 1
        
        # Global stiffness matrix for member
        k_global_member = T.T @ k_local @ T
        
        # Assembly into global matrix
        dofs_i = [dof_map[node_i]['dx'], dof_map[node_i]['dy'], dof_map[node_i]['rot']]
        dofs_j = [dof_map[node_j]['dx'], dof_map[node_j]['dy'], dof_map[node_j]['rot']]
        dofs = dofs_i + dofs_j
        
        for m in range(6):
            for n in range(6):
                K_global[dofs[m], dofs[n]] += k_global_member[m, n]
    
    return K_global, dof_map


def calculate_fixed_end_actions(members_df, loads_df, dof_map, nodes_df):
    """
    Calculate fixed end moments and forces for all load cases
    
    Returns:
        F_fixed: Fixed end force vector
    """
    n_dof = len(dof_map) * 3
    F_fixed = np.zeros(n_dof)
    
    if len(loads_df) == 0:
        return F_fixed
    
    for idx, load in loads_df.iterrows():
        member_id = int(load['Member'])
        load_type = load['Type']
        
        # Find member info
        member = members_df[members_df['Member'] == member_id].iloc[0]
        node_i = int(member['Node_I'])
        node_j = int(member['Node_J'])
        
        # Get member length and orientation
        xi = nodes_df[nodes_df['Node'] == node_i]['X'].values[0]
        yi = nodes_df[nodes_df['Node'] == node_i]['Y'].values[0]
        xj = nodes_df[nodes_df['Node'] == node_j]['X'].values[0]
        yj = nodes_df[nodes_df['Node'] == node_j]['Y'].values[0]
        L = np.sqrt((xj - xi)**2 + (yj - yi)**2)
        cos_theta = (xj - xi) / L
        sin_theta = (yj - yi) / L
        
        # Calculate FEM based on load type
        if load_type == "UDL":
            w = load['W1']  # kN/m
            # Fixed end moments for UDL
            FEM_i = -w * L**2 / 12
            FEM_j = w * L**2 / 12
            # Fixed end shears
            FES_i = -w * L / 2
            FES_j = -w * L / 2
            
        elif load_type == "VDL":
            w1 = load['W1']
            w2 = load['W2']
            # Trapezoidal load FEM
            FEM_i = -(w1 * L**2 / 20) * (7 - w2/w1) if w1 != 0 else 0
            FEM_j = (w2 * L**2 / 20) * (7 - w1/w2) if w2 != 0 else 0
            FES_i = -(w1 + w2) * L / 2
            FES_j = -(w1 + w2) * L / 2
            
        elif load_type == "Point Load":
            P = load['P']
            a = load['a']
            b = load['b'] if load['b'] > 0 else L - a
            # Point load FEM
            FEM_i = -P * a * b**2 / L**2
            FEM_j = P * a**2 * b / L**2
            FES_i = -P * b**2 * (3*a + b) / L**3
            FES_j = -P * a**2 * (a + 3*b) / L**3
            
        elif load_type == "Moment":
            M = load['M']
            a = load['a']
            # Applied moment FEM
            FEM_i = -M * (1 - a/L)
            FEM_j = M * a / L
            FES_i = 0
            FES_j = 0
        
        else:
            continue
        
        # Transform to global coordinates
        # Vertical forces (perpendicular to member axis)
        Fi_x = -FES_i * sin_theta
        Fi_y = FES_i * cos_theta
        Fj_x = -FES_j * sin_theta
        Fj_y = FES_j * cos_theta
        
        # Add to global force vector
        F_fixed[dof_map[node_i]['dx']] += Fi_x
        F_fixed[dof_map[node_i]['dy']] += Fi_y
        F_fixed[dof_map[node_i]['rot']] += FEM_i
        F_fixed[dof_map[node_j]['dx']] += Fj_x
        F_fixed[dof_map[node_j]['dy']] += Fj_y
        F_fixed[dof_map[node_j]['rot']] += FEM_j
    
    return F_fixed


def apply_boundary_conditions(K_global, F_fixed, supports_df, nodes_df, dof_map):
    """
    Apply boundary conditions and support settlements
    
    Returns:
        K_reduced: Reduced stiffness matrix
        F_reduced: Reduced force vector
        fixed_dofs: List of fixed DOF indices
    """
    n_dof = K_global.shape[0]
    fixed_dofs = []
    prescribed_displacements = {}
    
    # Process supports
    for idx, support in supports_df.iterrows():
        node = int(support['Node'])
        supp_type = support['Type']
        dx = support['Dx']
        dy = support['Dy']
        rot = support['Rotation']
        
        dof_dx = dof_map[node]['dx']
        dof_dy = dof_map[node]['dy']
        dof_rot = dof_map[node]['rot']
        
        if supp_type == "Fixed":
            fixed_dofs.extend([dof_dx, dof_dy, dof_rot])
            prescribed_displacements[dof_dx] = dx
            prescribed_displacements[dof_dy] = dy
            prescribed_displacements[dof_rot] = rot
            
        elif supp_type == "Pinned":
            fixed_dofs.extend([dof_dx, dof_dy])
            prescribed_displacements[dof_dx] = dx
            prescribed_displacements[dof_dy] = dy
            
        elif supp_type == "Roller":
            fixed_dofs.append(dof_dy)
            prescribed_displacements[dof_dy] = dy
    
    # Remove duplicates
    fixed_dofs = list(set(fixed_dofs))
    free_dofs = [i for i in range(n_dof) if i not in fixed_dofs]
    
    # Handle support settlements
    F_settlement = np.zeros(n_dof)
    for dof, disp in prescribed_displacements.items():
        if disp != 0:
            F_settlement -= K_global[:, dof] * disp
    
    # Partition matrices
    K_reduced = K_global[np.ix_(free_dofs, free_dofs)]
    F_reduced = (F_fixed + F_settlement)[free_dofs]
    
    return K_reduced, F_reduced, fixed_dofs


def solve_displacements(K_reduced, F_reduced, fixed_dofs, n_nodes):
    """
    Solve for unknown displacements
    
    Returns:
        displacements: Complete displacement vector
    """
    n_dof = 3 * n_nodes
    displacements = np.zeros(n_dof)
    
    # Solve for free DOFs
    free_dofs = [i for i in range(n_dof) if i not in fixed_dofs]
    
    if len(free_dofs) > 0:
        displacements_free = solve(K_reduced, F_reduced)
        displacements[free_dofs] = displacements_free
    
    return displacements


def calculate_member_forces(members_df, nodes_df, displacements, loads_df, dof_map):
    """
    Calculate member end forces from displacements
    
    Returns:
        DataFrame with member forces
    """
    results = []
    
    for idx, member in members_df.iterrows():
        member_id = int(member['Member'])
        node_i = int(member['Node_I'])
        node_j = int(member['Node_J'])
        E = member['E']
        I = member['I']
        A = member['A']
        
        # Get coordinates
        xi = nodes_df[nodes_df['Node'] == node_i]['X'].values[0]
        yi = nodes_df[nodes_df['Node'] == node_i]['Y'].values[0]
        xj = nodes_df[nodes_df['Node'] == node_j]['X'].values[0]
        yj = nodes_df[nodes_df['Node'] == node_j]['Y'].values[0]
        
        L = np.sqrt((xj - xi)**2 + (yj - yi)**2)
        cos_theta = (xj - xi) / L
        sin_theta = (yj - yi) / L
        
        # Get member displacements
        u = np.array([
            displacements[dof_map[node_i]['dx']],
            displacements[dof_map[node_i]['dy']],
            displacements[dof_map[node_i]['rot']],
            displacements[dof_map[node_j]['dx']],
            displacements[dof_map[node_j]['dy']],
            displacements[dof_map[node_j]['rot']]
        ])
        
        # Transformation matrix
        T = np.zeros((6, 6))
        T[0:2, 0:2] = [[cos_theta, sin_theta], [-sin_theta, cos_theta]]
        T[2, 2] = 1
        T[3:5, 3:5] = [[cos_theta, sin_theta], [-sin_theta, cos_theta]]
        T[5, 5] = 1
        
        # Local displacements
        u_local = T @ u
        
        # Local stiffness
        k_local = np.array([
            [E*A/L, 0, 0, -E*A/L, 0, 0],
            [0, 12*E*I/L**3, 6*E*I/L**2, 0, -12*E*I/L**3, 6*E*I/L**2],
            [0, 6*E*I/L**2, 4*E*I/L, 0, -6*E*I/L**2, 2*E*I/L],
            [-E*A/L, 0, 0, E*A/L, 0, 0],
            [0, -12*E*I/L**3, -6*E*I/L**2, 0, 12*E*I/L**3, -6*E*I/L**2],
            [0, 6*E*I/L**2, 2*E*I/L, 0, -6*E*I/L**2, 4*E*I/L]
        ])
        
        # Local forces
        f_local = k_local @ u_local
        
        # Get fixed end actions for this member
        FEM_i, FEM_j, FES_i, FES_j = get_member_fem(member_id, loads_df, L)
        
        # Total member forces
        N_i = f_local[0]
        V_i = f_local[1] - FES_i
        M_i = f_local[2] - FEM_i
        N_j = f_local[3]
        V_j = f_local[4] - FES_j
        M_j = f_local[5] - FEM_j
        
        results.append({
            'Member': member_id,
            'N_i': N_i,
            'V_i': V_i,
            'M_i': M_i,
            'N_j': N_j,
            'V_j': V_j,
            'M_j': M_j
        })
    
    return pd.DataFrame(results)


def get_member_fem(member_id, loads_df, L):
    """Helper function to get FEM for a specific member"""
    FEM_i_total = 0
    FEM_j_total = 0
    FES_i_total = 0
    FES_j_total = 0
    
    member_loads = loads_df[loads_df['Member'] == member_id]
    
    for idx, load in member_loads.iterrows():
        load_type = load['Type']
        
        if load_type == "UDL":
            w = load['W1']
            FEM_i_total += -w * L**2 / 12
            FEM_j_total += w * L**2 / 12
            FES_i_total += -w * L / 2
            FES_j_total += -w * L / 2
            
        elif load_type == "VDL":
            w1 = load['W1']
            w2 = load['W2']
            FEM_i_total += -(w1 * L**2 / 20) * (7 - w2/w1) if w1 != 0 else 0
            FEM_j_total += (w2 * L**2 / 20) * (7 - w1/w2) if w2 != 0 else 0
            FES_i_total += -(w1 + w2) * L / 2
            FES_j_total += -(w1 + w2) * L / 2
            
        elif load_type == "Point Load":
            P = load['P']
            a = load['a']
            b = load['b'] if load['b'] > 0 else L - a
            FEM_i_total += -P * a * b**2 / L**2
            FEM_j_total += P * a**2 * b / L**2
            FES_i_total += -P * b**2 * (3*a + b) / L**3
            FES_j_total += -P * a**2 * (a + 3*b) / L**3
            
        elif load_type == "Moment":
            M = load['M']
            a = load['a']
            FEM_i_total += -M * (1 - a/L)
            FEM_j_total += M * a / L
    
    return FEM_i_total, FEM_j_total, FES_i_total, FES_j_total


def calculate_reactions(K_global, displacements, F_fixed, supports_df, nodes_df, dof_map):
    """
    Calculate support reactions
    
    Returns:
        DataFrame with reactions
    """
    reactions_list = []
    
    # Calculate total forces
    F_total = K_global @ displacements - F_fixed
    
    for idx, support in supports_df.iterrows():
        node = int(support['Node'])
        
        Rx = F_total[dof_map[node]['dx']]
        Ry = F_total[dof_map[node]['dy']]
        Mz = F_total[dof_map[node]['rot']]
        
        reactions_list.append({
            'Node': node,
            'Support Type': support['Type'],
            'Rx (kN)': Rx,
            'Ry (kN)': Ry,
            'Mz (kNm)': Mz
        })
    
    return pd.DataFrame(reactions_list)
