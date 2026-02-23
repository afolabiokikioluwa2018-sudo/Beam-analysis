"""
CEG 410 - Beam and Frame Analysis Web Application
Main Streamlit Application File
Author: Structural Engineering Analysis Tool
"""

import streamlit as st
import numpy as np
import pandas as pd
from utils import (
    assemble_global_stiffness,
    calculate_fixed_end_actions,
    apply_boundary_conditions,
    solve_displacements,
    calculate_member_forces,
    calculate_reactions
)
from diagrams import plot_structure, plot_sfd, plot_bmd, plot_deflection
from examples import load_example
import json

# Page configuration
st.set_page_config(
    page_title="Beam & Frame Analysis",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'nodes' not in st.session_state:
    st.session_state.nodes = pd.DataFrame(columns=['Node', 'X', 'Y'])
if 'members' not in st.session_state:
    st.session_state.members = pd.DataFrame(columns=['Member', 'Node_I', 'Node_J', 'E', 'I', 'A'])
if 'supports' not in st.session_state:
    st.session_state.supports = pd.DataFrame(columns=['Node', 'Type', 'Dx', 'Dy', 'Rotation'])
if 'loads' not in st.session_state:
    st.session_state.loads = pd.DataFrame(columns=['Member', 'Type', 'W1', 'W2', 'P', 'M', 'a', 'b'])
if 'solved' not in st.session_state:
    st.session_state.solved = False

# Header
st.markdown('<div class="main-header">🏗️ Beam & Frame Analysis System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">CEG 410 Group 8 - Unlimited Spans | Multi-Storey Frames | Direct Stiffness Method</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📋 Project Setup")
    
    # Example presets
    st.subheader("Quick Start Examples")
    example_choice = st.selectbox(
        "Load Example:",
        ["None", "Simply Supported Beam", "Continuous Beam - Multi Span", 
         "Portal Frame", "Multi-Storey Frame", "Frame with Settlement"],
        key="example_selector"
    )
    
    # Info about capabilities
    st.info("""
    **📐 Capabilities:**
    - ✅ Unlimited spans (beams)
    - ✅ Multi-storey frames (2D)
    - ✅ Any support combination
    - ✅ Multiple loads per member
    - ✅ Support settlements
    """)
    
    if st.button("Load Example"):
        if example_choice != "None":
            nodes, members, supports, loads = load_example(example_choice)
            st.session_state.nodes = nodes
            st.session_state.members = members
            st.session_state.supports = supports
            st.session_state.loads = loads
            st.session_state.solved = False
            st.success(f"✅ Loaded: {example_choice}")
    
    st.divider()
    
    # Material properties
    st.subheader("Default Material Properties")
    default_E = st.number_input("Young's Modulus E (kN/m²)", value=200e6, format="%.2e")
    default_I = st.number_input("Moment of Inertia I (m⁴)", value=0.001, format="%.6f")
    default_A = st.number_input("Cross-sectional Area A (m²)", value=0.01, format="%.4f")
    
    st.divider()
    
    # Reinforcement design
    st.subheader("🔧 Reinforcement Design (BS 8110)")
    st.session_state.fck = st.number_input("Concrete Grade fcu (N/mm²)", value=30.0, min_value=20.0, 
                                           help="BS 8110: Typical grades 25, 30, 35, 40")
    st.session_state.fy = st.number_input("Steel Grade fy (N/mm²)", value=460.0, min_value=250.0,
                                          help="BS 8110: Typical grades 250, 460")
    st.session_state.beam_width = st.number_input("Beam Width b (mm)", value=300.0, min_value=150.0)
    st.session_state.beam_depth = st.number_input("Overall Depth h (mm)", value=500.0, min_value=200.0)
    st.session_state.cover = st.number_input("Cover to reinforcement (mm)", value=25.0, min_value=20.0,
                                             help="BS 8110: Minimum 25mm for mild exposure")

# Main content tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📐 Structure Definition", "🔧 Analysis", "📊 Results", 
    "📈 Diagrams", "🏗️ Reinforcement", "ℹ️ How to Use"
])

with tab1:
    st.header("Structure Definition")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1️⃣ Define Nodes")
        st.write("Add nodes (joints) with X, Y coordinates")
        
        with st.form("node_form"):
            ncol1, ncol2, ncol3 = st.columns(3)
            node_id = ncol1.number_input("Node ID", min_value=1, value=1, step=1)
            node_x = ncol2.number_input("X (m)", value=0.0, step=0.5)
            node_y = ncol3.number_input("Y (m)", value=0.0, step=0.5)
            
            if st.form_submit_button("➕ Add Node"):
                new_node = pd.DataFrame({
                    'Node': [int(node_id)],
                    'X': [node_x],
                    'Y': [node_y]
                })
                st.session_state.nodes = pd.concat([st.session_state.nodes, new_node], ignore_index=True)
                st.session_state.solved = False
                st.success(f"Node {int(node_id)} added!")
        
        st.dataframe(st.session_state.nodes, use_container_width=True, height=200)
        
        if st.button("🗑️ Clear All Nodes"):
            st.session_state.nodes = pd.DataFrame(columns=['Node', 'X', 'Y'])
            st.session_state.solved = False
    
    with col2:
        st.subheader("2️⃣ Define Members")
        st.write("Connect nodes to create members (beams/columns)")
        
        with st.form("member_form"):
            mcol1, mcol2, mcol3 = st.columns(3)
            member_id = mcol1.number_input("Member ID", min_value=1, value=1, step=1)
            node_i = mcol2.number_input("Start Node", min_value=1, value=1, step=1)
            node_j = mcol3.number_input("End Node", min_value=1, value=2, step=1)
            
            mcol4, mcol5, mcol6 = st.columns(3)
            mem_E = mcol4.number_input("E (kN/m²)", value=default_E, format="%.2e", key="mem_e")
            mem_I = mcol5.number_input("I (m⁴)", value=default_I, format="%.6f", key="mem_i")
            mem_A = mcol6.number_input("A (m²)", value=default_A, format="%.4f", key="mem_a")
            
            if st.form_submit_button("➕ Add Member"):
                new_member = pd.DataFrame({
                    'Member': [int(member_id)],
                    'Node_I': [int(node_i)],
                    'Node_J': [int(node_j)],
                    'E': [mem_E],
                    'I': [mem_I],
                    'A': [mem_A]
                })
                st.session_state.members = pd.concat([st.session_state.members, new_member], ignore_index=True)
                st.session_state.solved = False
                st.success(f"Member {int(member_id)} added!")
        
        st.dataframe(st.session_state.members, use_container_width=True, height=200)
        
        if st.button("🗑️ Clear All Members"):
            st.session_state.members = pd.DataFrame(columns=['Member', 'Node_I', 'Node_J', 'E', 'I', 'A'])
            st.session_state.solved = False
    
    st.divider()
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("3️⃣ Define Supports")
        st.write("Specify boundary conditions and settlements")
        
        with st.form("support_form"):
            scol1, scol2 = st.columns(2)
            supp_node = scol1.number_input("Node", min_value=1, value=1, step=1, key="supp_node")
            supp_type = scol2.selectbox("Type", ["Fixed", "Pinned", "Roller"], key="supp_type")
            
            scol3, scol4, scol5 = st.columns(3)
            supp_dx = scol3.number_input("Settlement Dx (m)", value=0.0, format="%.6f", key="supp_dx")
            supp_dy = scol4.number_input("Settlement Dy (m)", value=0.0, format="%.6f", key="supp_dy")
            supp_rot = scol5.number_input("Rotation (rad)", value=0.0, format="%.6f", key="supp_rot")
            
            if st.form_submit_button("➕ Add Support"):
                new_support = pd.DataFrame({
                    'Node': [int(supp_node)],
                    'Type': [supp_type],
                    'Dx': [supp_dx],
                    'Dy': [supp_dy],
                    'Rotation': [supp_rot]
                })
                st.session_state.supports = pd.concat([st.session_state.supports, new_support], ignore_index=True)
                st.session_state.solved = False
                st.success(f"Support at Node {int(supp_node)} added!")
        
        st.dataframe(st.session_state.supports, use_container_width=True, height=200)
        
        if st.button("🗑️ Clear All Supports"):
            st.session_state.supports = pd.DataFrame(columns=['Node', 'Type', 'Dx', 'Dy', 'Rotation'])
            st.session_state.solved = False
    
    with col4:
        st.subheader("4️⃣ Define Loads")
        st.write("Apply loads to members")
        
        with st.form("load_form"):
            lcol1, lcol2 = st.columns(2)
            load_member = lcol1.number_input("Member", min_value=1, value=1, step=1, key="load_mem")
            load_type = lcol2.selectbox("Type", ["UDL", "VDL", "Point Load", "Moment"], key="load_type")
            
            lcol3, lcol4, lcol5 = st.columns(3)
            w1 = lcol3.number_input("W1 (kN/m)", value=0.0, key="w1", help="For UDL/VDL")
            w2 = lcol4.number_input("W2 (kN/m)", value=0.0, key="w2", help="For VDL")
            P = lcol5.number_input("P (kN)", value=0.0, key="P", help="Point load")
            
            lcol6, lcol7, lcol8 = st.columns(3)
            M = lcol6.number_input("M (kNm)", value=0.0, key="M", help="Moment")
            a = lcol7.number_input("a (m)", value=0.0, key="a", help="Distance from start")
            b = lcol8.number_input("b (m)", value=0.0, key="b", help="Distance from end")
            
            if st.form_submit_button("➕ Add Load"):
                new_load = pd.DataFrame({
                    'Member': [int(load_member)],
                    'Type': [load_type],
                    'W1': [w1],
                    'W2': [w2],
                    'P': [P],
                    'M': [M],
                    'a': [a],
                    'b': [b]
                })
                st.session_state.loads = pd.concat([st.session_state.loads, new_load], ignore_index=True)
                st.session_state.solved = False
                st.success(f"Load on Member {int(load_member)} added!")
        
        st.dataframe(st.session_state.loads, use_container_width=True, height=200)
        
        if st.button("🗑️ Clear All Loads"):
            st.session_state.loads = pd.DataFrame(columns=['Member', 'Type', 'W1', 'W2', 'P', 'M', 'a', 'b'])
            st.session_state.solved = False

with tab2:
    st.header("🔧 Analysis")
    
    if len(st.session_state.nodes) > 0 and len(st.session_state.members) > 0:
        # Display structure statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📍 Nodes", len(st.session_state.nodes))
        
        with col2:
            st.metric("🔗 Members", len(st.session_state.members))
        
        with col3:
            st.metric("⚓ Supports", len(st.session_state.supports))
        
        with col4:
            if st.session_state.solved:
                n_reactions = len(st.session_state.reactions)
                st.metric("⚡ Reactions", n_reactions)
            else:
                st.metric("⚡ Reactions", "—")
        
        st.divider()
        
        # Support details table
        if len(st.session_state.supports) > 0:
            st.subheader("⚓ Support Configuration")
            support_summary = st.session_state.supports[['Node', 'Type']].copy()
            
            # Count DOFs restrained per support
            def count_restraints(support_type):
                if support_type == 'Fixed':
                    return '3 DOF (Dx, Dy, Rz)'
                elif support_type == 'Pinned':
                    return '2 DOF (Dx, Dy)'
                else:  # Roller
                    return '1 DOF (Dy)'
            
            support_summary['Restraints'] = st.session_state.supports['Type'].apply(count_restraints)
            
            # Check for settlements
            has_settlement = (
                (st.session_state.supports['Dx'] != 0) | 
                (st.session_state.supports['Dy'] != 0) | 
                (st.session_state.supports['Rotation'] != 0)
            ).any()
            
            if has_settlement:
                support_summary['Settlement'] = st.session_state.supports.apply(
                    lambda row: '✓' if (row['Dx'] != 0 or row['Dy'] != 0 or row['Rotation'] != 0) else '—',
                    axis=1
                )
            
            st.dataframe(support_summary, use_container_width=True, hide_index=True)
        
        st.divider()
        
        st.subheader("Structure Preview")
        try:
            fig = plot_structure(st.session_state.nodes, st.session_state.members, 
                                st.session_state.supports, st.session_state.loads)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error plotting structure: {e}")
        
        st.divider()
        
        if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
            try:
                with st.spinner("Assembling global stiffness matrix..."):
                    K_global, dof_map = assemble_global_stiffness(
                        st.session_state.nodes, st.session_state.members
                    )
                
                with st.spinner("Calculating fixed end actions..."):
                    F_fixed = calculate_fixed_end_actions(
                        st.session_state.members, st.session_state.loads, dof_map, st.session_state.nodes
                    )
                
                with st.spinner("Applying boundary conditions..."):
                    K_reduced, F_reduced, fixed_dofs = apply_boundary_conditions(
                        K_global, F_fixed, st.session_state.supports, 
                        st.session_state.nodes, dof_map
                    )
                
                with st.spinner("Solving for displacements..."):
                    displacements = solve_displacements(
                        K_reduced, F_reduced, fixed_dofs, len(st.session_state.nodes)
                    )
                
                with st.spinner("Calculating member forces..."):
                    member_forces = calculate_member_forces(
                        st.session_state.members, st.session_state.nodes,
                        displacements, st.session_state.loads, dof_map
                    )
                
                with st.spinner("Calculating reactions..."):
                    reactions = calculate_reactions(
                        K_global, displacements, F_fixed, st.session_state.supports,
                        st.session_state.nodes, dof_map
                    )
                
                # Store results
                st.session_state.displacements = displacements
                st.session_state.member_forces = member_forces
                st.session_state.reactions = reactions
                st.session_state.solved = True
                
                # Display reaction summary immediately
                st.success("✅ Analysis completed successfully!")
                
                # Show reaction summary
                st.subheader("⚡ Reaction Summary")
                total_rx = reactions['Rx (kN)'].sum()
                total_ry = reactions['Ry (kN)'].sum()
                total_mz = reactions['Mz (kNm)'].sum()
                
                col1, col2, col3 = st.columns(3)
                col1.metric("ΣRx (kN)", f"{total_rx:.2f}")
                col2.metric("ΣRy (kN)", f"{total_ry:.2f}")
                col3.metric("ΣMz (kNm)", f"{total_mz:.2f}")
                
                st.balloons()
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {e}")
                st.exception(e)
    else:
        st.warning("⚠️ Please define nodes and members first!")

with tab3:
    st.header("📊 Analysis Results")
    
    if st.session_state.solved:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔽 Reactions")
            st.dataframe(st.session_state.reactions, use_container_width=True)
        
        with col2:
            st.subheader("📏 Displacements")
            disp_df = pd.DataFrame({
                'Node': range(1, len(st.session_state.displacements)//3 + 1),
                'Dx (m)': [st.session_state.displacements[i*3] for i in range(len(st.session_state.displacements)//3)],
                'Dy (m)': [st.session_state.displacements[i*3+1] for i in range(len(st.session_state.displacements)//3)],
                'Rotation (rad)': [st.session_state.displacements[i*3+2] for i in range(len(st.session_state.displacements)//3)]
            })
            st.dataframe(disp_df, use_container_width=True)
        
        st.divider()
        
        st.subheader("🔩 Member End Forces")
        st.dataframe(st.session_state.member_forces, use_container_width=True)
        
        st.divider()
        
        st.subheader("📊 Detailed Shear Force & Bending Moment Analysis")
        
        # Store detailed values in session state for diagram plotting
        st.session_state.detailed_sf_bm = []
        
        # Tabs for different views
        detail_tab1, detail_tab2, detail_tab3 = st.tabs([
            "📍 Span End Values", "📈 Detailed Values (Every 10%)", "📊 Summary Table"
        ])
        
        with detail_tab1:
            st.write("**Shear Force and Bending Moment at 0% and 100% of each span**")
            
            # Calculate SF and BM at span ends
            span_forces = []
            
            for idx, member in st.session_state.members.iterrows():
                member_id = int(member['Member'])
                node_i = int(member['Node_I'])
                node_j = int(member['Node_J'])
                
                # Get node coordinates
                xi = st.session_state.nodes[st.session_state.nodes['Node'] == node_i]['X'].values[0]
                yi = st.session_state.nodes[st.session_state.nodes['Node'] == node_i]['Y'].values[0]
                xj = st.session_state.nodes[st.session_state.nodes['Node'] == node_j]['X'].values[0]
                yj = st.session_state.nodes[st.session_state.nodes['Node'] == node_j]['Y'].values[0]
                L = np.sqrt((xj - xi)**2 + (yj - yi)**2)
                
                # Get member forces
                forces = st.session_state.member_forces[st.session_state.member_forces['Member'] == member_id].iloc[0]
                V_i = forces['V_i']
                M_i = forces['M_i']
                
                # Get loads on this member
                member_loads = st.session_state.loads[st.session_state.loads['Member'] == member_id]
                
                # Calculate SF and BM at start
                SF_start = V_i
                BM_start = M_i
                
                # Calculate SF and BM at end
                SF_end = V_i
                BM_end = M_i + V_i * L
                
                for _, load in member_loads.iterrows():
                    if load['Type'] == 'UDL':
                        w = load['W1']
                        SF_end -= w * L
                        BM_end -= w * L**2 / 2
                        
                    elif load['Type'] == 'VDL':
                        w1 = load['W1']
                        w2 = load['W2']
                        SF_end -= (w1 + w2) * L / 2
                        BM_end -= (w1 + w2) * L**2 / 6
                        
                    elif load['Type'] == 'Point Load':
                        P = load['P']
                        a = load['a']
                        SF_end -= P
                        BM_end -= P * (L - a)
                        
                    elif load['Type'] == 'Moment':
                        M = load['M']
                        BM_end += M
                
                span_forces.append({
                    'Span': f'Member {member_id}',
                    'From': f'Node {node_i}',
                    'To': f'Node {node_j}',
                    'Length (m)': f'{L:.3f}',
                    'SF @ 0% (kN)': f'{SF_start:.3f}',
                    'BM @ 0% (kNm)': f'{BM_start:.3f}',
                    'SF @ 100% (kN)': f'{SF_end:.3f}',
                    'BM @ 100% (kNm)': f'{BM_end:.3f}'
                })
            
            span_forces_df = pd.DataFrame(span_forces)
            st.dataframe(span_forces_df, use_container_width=True, hide_index=True)
        
        with detail_tab2:
            st.write("**Shear Force and Bending Moment at 10% intervals along each span**")
            
            for idx, member in st.session_state.members.iterrows():
                member_id = int(member['Member'])
                node_i = int(member['Node_I'])
                node_j = int(member['Node_J'])
                
                # Get node coordinates
                xi = st.session_state.nodes[st.session_state.nodes['Node'] == node_i]['X'].values[0]
                yi = st.session_state.nodes[st.session_state.nodes['Node'] == node_i]['Y'].values[0]
                xj = st.session_state.nodes[st.session_state.nodes['Node'] == node_j]['X'].values[0]
                yj = st.session_state.nodes[st.session_state.nodes['Node'] == node_j]['Y'].values[0]
                L = np.sqrt((xj - xi)**2 + (yj - yi)**2)
                
                # Get member forces
                forces = st.session_state.member_forces[st.session_state.member_forces['Member'] == member_id].iloc[0]
                V_i = forces['V_i']
                M_i = forces['M_i']
                
                # Get loads on this member
                member_loads = st.session_state.loads[st.session_state.loads['Member'] == member_id]
                
                st.markdown(f"#### Member {member_id} (Node {node_i} → Node {node_j}, L = {L:.3f} m)")
                
                detailed_values = []
                member_detailed_data = {'member': member_id, 'positions': [], 'sf': [], 'bm': []}
                
                # Calculate at 11 points (0%, 10%, 20%, ..., 100%)
                for percent in range(0, 101, 10):
                    x = (percent / 100.0) * L
                    
                    # Calculate SF at position x
                    SF_x = V_i
                    BM_x = M_i + V_i * x
                    
                    # Apply load effects
                    for _, load in member_loads.iterrows():
                        if load['Type'] == 'UDL':
                            w = load['W1']
                            SF_x -= w * x
                            BM_x -= w * x**2 / 2
                            
                        elif load['Type'] == 'VDL':
                            w1 = load['W1']
                            w2 = load['W2']
                            # Linear variation of load
                            w_x = w1 + (w2 - w1) * x / L
                            # Integration for SF and BM
                            SF_x -= (w1 * x + (w2 - w1) * x**2 / (2 * L))
                            BM_x -= (w1 * x**2 / 2 + (w2 - w1) * x**3 / (6 * L))
                            
                        elif load['Type'] == 'Point Load':
                            P = load['P']
                            a = load['a']
                            if x >= a:
                                SF_x -= P
                                BM_x -= P * (x - a)
                                
                        elif load['Type'] == 'Moment':
                            M = load['M']
                            a = load['a']
                            if x >= a:
                                BM_x += M
                    
                    member_detailed_data['positions'].append(x)
                    member_detailed_data['sf'].append(SF_x)
                    member_detailed_data['bm'].append(BM_x)
                    
                    detailed_values.append({
                        'Position': f'{percent}%',
                        'Distance (m)': f'{x:.3f}',
                        'Shear Force (kN)': f'{SF_x:.3f}',
                        'Bending Moment (kNm)': f'{BM_x:.3f}'
                    })
                
                st.session_state.detailed_sf_bm.append(member_detailed_data)
                
                detailed_df = pd.DataFrame(detailed_values)
                st.dataframe(detailed_df, use_container_width=True, hide_index=True)
                st.divider()
        
        with detail_tab3:
            st.write("**Summary: Maximum and Minimum Values**")
            
            summary_data = []
            
            for member_data in st.session_state.detailed_sf_bm:
                member_id = member_data['member']
                sf_values = member_data['sf']
                bm_values = member_data['bm']
                positions = member_data['positions']
                
                # Find max/min
                max_sf = max(sf_values)
                min_sf = min(sf_values)
                max_bm = max(bm_values)
                min_bm = min(bm_values)
                
                max_sf_pos = positions[sf_values.index(max_sf)]
                min_sf_pos = positions[sf_values.index(min_sf)]
                max_bm_pos = positions[bm_values.index(max_bm)]
                min_bm_pos = positions[bm_values.index(min_bm)]
                
                summary_data.append({
                    'Member': member_id,
                    'Max SF (kN)': f'{max_sf:.3f}',
                    'at (m)': f'{max_sf_pos:.3f}',
                    'Min SF (kN)': f'{min_sf:.3f}',
                    'at (m) ': f'{min_sf_pos:.3f}',
                    'Max BM (kNm)': f'{max_bm:.3f}',
                    'at (m)  ': f'{max_bm_pos:.3f}',
                    'Min BM (kNm)': f'{min_bm:.3f}',
                    'at (m)   ': f'{min_bm_pos:.3f}'
                })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        # Add explanation
        st.info("""
        📌 **Sign Convention:** 
        - **Positive SF (+)** = Upward force on left face of section
        - **Negative SF (−)** = Downward force on left face of section
        - **Positive BM (+)** = Sagging moment (tension at bottom fiber)
        - **Negative BM (−)** = Hogging moment (tension at top fiber)
        
        💡 **Tip:** Use the tabs above to view:
        - **Span End Values**: Quick view of forces at supports
        - **Detailed Values**: Complete analysis at 10% intervals
        - **Summary Table**: Maximum and minimum values with locations
        """)
        
    else:
        st.info("👈 Run analysis first to see results!")

with tab4:
    st.header("📈 Force and Moment Diagrams")
    
    if st.session_state.solved:
        st.info("💡 **Tip:** Hover over diagrams to see exact values. Zoom and pan using plot controls.")
        
        try:
            st.subheader("Shear Force Diagram")
            st.write("**Sign Convention:** Positive (+) = Upward force on left face | Negative (−) = Downward force")
            sfd_fig = plot_sfd(st.session_state.members, st.session_state.nodes,
                              st.session_state.member_forces, st.session_state.loads)
            st.plotly_chart(sfd_fig, use_container_width=True)
            
            st.divider()
            
            st.subheader("Bending Moment Diagram")
            st.write("**Sign Convention:** Positive (+) = Sagging (tension at bottom) | Negative (−) = Hogging (tension at top)")
            bmd_fig = plot_bmd(st.session_state.members, st.session_state.nodes,
                              st.session_state.member_forces, st.session_state.loads)
            st.plotly_chart(bmd_fig, use_container_width=True)
            
            st.divider()
            
            st.subheader("Deflection Diagram")
            st.write("**Exaggerated scale for visualization**")
            defl_fig = plot_deflection(st.session_state.members, st.session_state.nodes,
                                      st.session_state.displacements)
            st.plotly_chart(defl_fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error plotting diagrams: {e}")
            st.exception(e)
    else:
        st.info("👈 Run analysis first to see diagrams!")

with tab5:
    st.header("🏗️ Reinforcement Design (BS 8110:1997)")
    
    if st.session_state.solved:
        st.info("📋 Design based on BS 8110:1997 - Structural Use of Concrete")
        
        # Get max moment for design
        max_moment_positive = 0
        max_moment_negative = 0
        max_moment_member = 1
        
        for idx, row in st.session_state.member_forces.iterrows():
            M_i = row['M_i']
            M_j = row['M_j']
            if M_i > max_moment_positive:
                max_moment_positive = M_i
                max_moment_member = row['Member']
            if M_j > max_moment_positive:
                max_moment_positive = M_j
            if M_i < max_moment_negative:
                max_moment_negative = M_i
            if M_j < max_moment_negative:
                max_moment_negative = M_j
        
        max_moment_design = max(abs(max_moment_positive), abs(max_moment_negative))
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Maximum Sagging Moment", f"{max_moment_positive:.2f} kNm")
        with col2:
            st.metric("Maximum Hogging Moment", f"{abs(max_moment_negative):.2f} kNm")
        
        st.write(f"**Designing for critical moment:** {max_moment_design:.2f} kNm")
        
        # Design parameters
        b = st.session_state.beam_width
        h = st.session_state.beam_depth
        cover = st.session_state.cover
        fcu = st.session_state.fck  # Using fcu for BS 8110
        fy = st.session_state.fy
        
        # Assume 20mm bars for calculation of d
        bar_dia = 20
        d = h - cover - bar_dia/2 - 8  # 8mm links
        
        st.write(f"**Section Properties:**")
        st.write(f"- Width (b) = {b:.0f} mm")
        st.write(f"- Overall depth (h) = {h:.0f} mm")
        st.write(f"- Effective depth (d) = {d:.0f} mm")
        st.write(f"- Concrete grade (fcu) = {fcu:.0f} N/mm²")
        st.write(f"- Steel grade (fy) = {fy:.0f} N/mm²")
        
        st.divider()
        
        # Convert moment to N-mm
        M = max_moment_design * 1e6  # kNm to Nmm
        
        # BS 8110 design constants
        # γm for concrete = 1.5 (Table 2.2)
        # γm for steel = 1.05 (Table 2.2)
        gamma_m_concrete = 1.5
        gamma_m_steel = 1.05
        
        # Design stresses
        # For fcu ≤ 40 N/mm², stress block factor = 0.67fcu/γm
        design_concrete_stress = 0.67 * fcu / gamma_m_concrete
        design_steel_stress = fy / gamma_m_steel
        
        # K = M/(bd²fcu)
        K = M / (b * d**2 * fcu)
        
        # K' (balanced condition) - BS 8110 Clause 3.4.4.4
        # For fy = 460, K' = 0.156
        # For fy = 250, K' = 0.207
        if fy == 460:
            K_prime = 0.156
        elif fy == 250:
            K_prime = 0.207
        else:
            # Linear interpolation
            K_prime = 0.156 + (fy - 460) * (0.207 - 0.156) / (250 - 460)
        
        st.subheader("📐 Design Calculations")
        st.write(f"**K = M/(bd²fcu) = {K:.4f}**")
        st.write(f"**K' (balanced) = {K_prime:.3f}**")
        
        if K <= K_prime:
            st.success("✅ **Section is SINGLY REINFORCED** (Tension steel only)")
            
            # Calculate lever arm (z)
            # z = d[0.5 + √(0.25 - K/0.9)]
            z = d * (0.5 + np.sqrt(0.25 - K/0.9))
            
            # But z should not be greater than 0.95d
            if z > 0.95 * d:
                z = 0.95 * d
            
            st.write(f"**Lever arm (z) = {z:.0f} mm** (≤ 0.95d = {0.95*d:.0f} mm)")
            
            # Required tension steel
            # As = M / (0.87 * fy * z)
            As_req = M / (0.87 * fy * z)
            
            st.write(f"**Required tension steel (As) = {As_req:.0f} mm²**")
            
            # Minimum steel (BS 8110 Clause 3.12.5.3)
            # As,min = 0.13% of bh for fy = 460
            # As,min = 0.24% of bh for fy = 250
            if fy == 460:
                As_min = 0.0013 * b * h
            elif fy == 250:
                As_min = 0.0024 * b * h
            else:
                As_min = 0.0013 * b * h  # Conservative
            
            st.write(f"**Minimum steel required (BS 8110) = {As_min:.0f} mm²**")
            
            As_provide = max(As_req, As_min)
            
            st.divider()
            st.subheader("🔩 Bar Arrangement")
            
            # Suggest bar sizes
            bar_sizes = [12, 16, 20, 25, 32]
            bar_areas = [113, 201, 314, 491, 804]  # mm² for each bar
            
            st.write("**Suggested arrangements:**")
            
            for bar_dia, bar_area in zip(bar_sizes, bar_areas):
                n_bars = np.ceil(As_provide / bar_area)
                As_provided = n_bars * bar_area
                
                if n_bars <= 6 and n_bars >= 2:  # Practical range
                    utilization = (As_provide / As_provided) * 100
                    st.write(f"- **{int(n_bars)} bars of {bar_dia}mm ϕ** → Provided: {As_provided:.0f} mm² ({utilization:.1f}% utilized)")
            
            # Check maximum steel (BS 8110 Clause 3.12.6.1)
            # Maximum = 4% of gross cross-sectional area
            As_max = 0.04 * b * h
            
            st.info(f"💡 **Maximum steel allowed:** {As_max:.0f} mm² (4% of bh)")
            
            if As_provide > As_max:
                st.error("⚠️ Steel area exceeds maximum! Consider increasing section size.")
            
        else:
            st.warning("⚠️ **Section requires DOUBLY REINFORCED design** (Compression steel needed)")
            st.write(f"K = {K:.4f} > K' = {K_prime:.3f}")
            
            st.write("**Design procedure for doubly reinforced section:**")
            
            # Moment resisted by concrete (Mc)
            Mc = K_prime * fcu * b * d**2
            
            st.write(f"1. Moment capacity as singly reinforced = {Mc/1e6:.2f} kNm")
            st.write(f"2. Additional moment requiring compression steel = {(M - Mc)/1e6:.2f} kNm")
            
            # Lever arm for singly reinforced part
            z = 0.775 * d  # Approximate for K = K'
            
            # Tension steel for Mc
            As1 = Mc / (0.87 * fy * z)
            
            # Additional tension steel for (M - Mc)
            d_prime = cover + 8 + bar_dia/2  # Depth to compression steel
            As2 = (M - Mc) / (0.87 * fy * (d - d_prime))
            
            # Total tension steel
            As_total = As1 + As2
            
            # Compression steel
            # Assuming fsc = 0.87fy (steel yields)
            Asc = (M - Mc) / (0.87 * fy * (d - d_prime))
            
            st.write(f"3. **Required tension steel (As) = {As_total:.0f} mm²**")
            st.write(f"4. **Required compression steel (A'sc) = {Asc:.0f} mm²**")
            
            st.info("💡 Note: Detailed doubly reinforced design requires checking compression steel stress. Consider increasing beam depth if possible.")
        
        st.divider()
        
        # Shear design
        st.subheader("✂️ Shear Design Check")
        
        # Get maximum shear
        max_shear = max(abs(st.session_state.member_forces['V_i'].max()), 
                       abs(st.session_state.member_forces['V_j'].max()))
        
        st.write(f"**Maximum shear force (V) = {max_shear:.2f} kN**")
        
        # Shear stress
        v = (max_shear * 1000) / (b * d)  # N/mm²
        
        st.write(f"**Shear stress (v) = V/(bd) = {v:.3f} N/mm²**")
        
        # Concrete shear stress (vc) - BS 8110 Table 3.8
        # Simplified: vc depends on (100As/bd) and fcu
        As_bd_percent = (As_provide / (b * d)) * 100
        
        # Simplified vc calculation (conservative)
        if fcu == 25:
            vc = 0.4  # N/mm² (conservative)
        elif fcu == 30:
            vc = 0.45
        elif fcu == 40:
            vc = 0.5
        else:
            vc = 0.4  # Conservative
        
        st.write(f"**Concrete shear strength (vc) ≈ {vc:.2f} N/mm²** (BS 8110 Table 3.8)")
        
        if v < vc:
            st.success(f"✅ **No shear reinforcement required** (v < vc)")
            st.write("Provide nominal links as per BS 8110 Clause 3.12.7")
        elif v < 0.8 * np.sqrt(fcu):
            st.warning(f"⚠️ **Shear links required** (v > vc)")
            
            # Required shear reinforcement
            # Asv/sv = bv(v - vc)/(0.87fyv)
            # Assume fyv = fy and 8mm links (Asv = 2 x 50 = 100mm² for 2 legs)
            fyv = fy
            Asv = 2 * 50  # Two legs of 8mm (50mm² each)
            
            sv_required = (Asv * 0.87 * fyv) / (b * (v - vc))
            
            st.write(f"**Required link spacing (sv) = {sv_required:.0f} mm**")
            st.write(f"**Provide 8mm links @ {min(int(sv_required), 300):.0f}mm centers**")
            st.info("💡 Maximum spacing = 0.75d or 300mm (BS 8110 Clause 3.12.7)")
            
        else:
            st.error(f"⚠️ **Section inadequate for shear!** v > 0.8√fcu")
            st.write("Consider increasing beam depth or width")
        st.divider()
        
        # Deflection check
        st.subheader("📏 Deflection Criteria (BS 8110 Clause 3.4.6)")
        
        st.write("**Allowable Span/Effective Depth Ratios (BS 8110 Table 3.9)**")
        
        # Get maximum span
        max_span = 0
        max_span_member = 1
        
        for idx, member in st.session_state.members.iterrows():
            member_id = int(member['Member'])
            node_i = int(member['Node_I'])
            node_j = int(member['Node_J'])
            
            xi = st.session_state.nodes[st.session_state.nodes['Node'] == node_i]['X'].values[0]
            yi = st.session_state.nodes[st.session_state.nodes['Node'] == node_i]['Y'].values[0]
            xj = st.session_state.nodes[st.session_state.nodes['Node'] == node_j]['X'].values[0]
            yj = st.session_state.nodes[st.session_state.nodes['Node'] == node_j]['Y'].values[0]
            L = np.sqrt((xj - xi)**2 + (yj - yi)**2)
            
            if L > max_span:
                max_span = L
                max_span_member = member_id
        
        # Convert to mm
        span_mm = max_span * 1000
        
        st.write(f"**Critical span (Member {max_span_member}):** {span_mm:.0f} mm = {max_span:.2f} m")
        st.write(f"**Effective depth (d):** {d:.0f} mm")
        
        # Actual span/depth ratio
        actual_ratio = span_mm / d
        st.write(f"**Actual span/effective depth ratio = {actual_ratio:.2f}**")
        
        # Basic span/depth ratios from BS 8110 Table 3.9
        # For rectangular sections:
        support_conditions = st.session_state.supports['Type'].value_counts()
        
        if 'Fixed' in support_conditions and support_conditions.get('Fixed', 0) >= 2:
            basic_ratio = 26  # Cantilever or continuous
            condition = "Continuous beam"
        elif 'Pinned' in support_conditions or 'Roller' in support_conditions:
            if len(st.session_state.supports) == 2:
                basic_ratio = 20  # Simply supported
                condition = "Simply supported"
            else:
                basic_ratio = 26  # Continuous
                condition = "Continuous beam"
        else:
            basic_ratio = 20  # Default to simply supported
            condition = "Simply supported"
        
        st.write(f"**Basic span/depth ratio (BS 8110 Table 3.9):** {basic_ratio} ({condition})")
        
        # Modification factor for tension reinforcement
        # MF = 0.55 + (477 - fs)/(120(0.9 + M/bd²))
        # where fs = service stress in tension steel
        # fs can be approximated as (5/8)fy × (As,req/As,prov) for fy = 460
        
        if As_provide > 0:
            # Service stress approximation
            fs = (5/8) * fy * (As_req / As_provide)
            
            # M/bd² in N/mm²
            M_bd2 = M / (b * d**2)
            
            # Modification factor
            MF = 0.55 + (477 - fs) / (120 * (0.9 + M_bd2))
            
            # Limit MF
            if MF > 2.0:
                MF = 2.0
            elif MF < 0.55:
                MF = 0.55
            
            st.write(f"**Modification factor for tension steel:** {MF:.3f}")
            
            # Allowable span/depth ratio
            allowable_ratio = basic_ratio * MF
            
            st.write(f"**Allowable span/depth ratio = {basic_ratio} × {MF:.3f} = {allowable_ratio:.2f}**")
            
            # Check
            if actual_ratio <= allowable_ratio:
                percentage = (actual_ratio / allowable_ratio) * 100
                st.success(f"✅ **DEFLECTION CRITERIA SATISFIED**")
                st.write(f"Actual ratio ({actual_ratio:.2f}) ≤ Allowable ratio ({allowable_ratio:.2f})")
                st.write(f"Utilization: {percentage:.1f}%")
            else:
                excess = ((actual_ratio - allowable_ratio) / allowable_ratio) * 100
                st.error(f"❌ **DEFLECTION CRITERIA NOT SATISFIED**")
                st.write(f"Actual ratio ({actual_ratio:.2f}) > Allowable ratio ({allowable_ratio:.2f})")
                st.write(f"Exceeds by: {excess:.1f}%")
                st.warning("⚠️ **Recommendations:**")
                st.write("- Increase beam depth")
                st.write("- Reduce span length")
                st.write("- Provide compression reinforcement")
                st.write("- Use higher strength concrete")
        
        # Actual deflection from analysis
        st.divider()
        st.write("**Actual Deflections from Analysis:**")
        
        # Find maximum vertical deflection
        max_deflection = 0
        max_defl_node = 0
        
        for i in range(len(st.session_state.displacements) // 3):
            dy = abs(st.session_state.displacements[i * 3 + 1])  # Vertical displacement
            if dy > max_deflection:
                max_deflection = dy
                max_defl_node = i + 1
        
        max_deflection_mm = max_deflection * 1000  # Convert to mm
        
        st.write(f"**Maximum vertical deflection:** {max_deflection_mm:.3f} mm at Node {max_defl_node}")
        
        # Deflection limits (BS 8110 Table 3.1)
        # Span/250 for general members
        limit_general = span_mm / 250
        # Span/350 for members supporting brittle finishes
        limit_brittle = span_mm / 350
        
        st.write(f"**Deflection limits (BS 8110):**")
        st.write(f"- General case: Span/250 = {limit_general:.2f} mm")
        st.write(f"- Brittle finishes: Span/350 = {limit_brittle:.2f} mm")
        
        if max_deflection_mm <= limit_brittle:
            st.success(f"✅ **Deflection satisfies both criteria** (suitable for brittle finishes)")
        elif max_deflection_mm <= limit_general:
            st.success(f"✅ **Deflection satisfies general criteria** (not suitable for brittle finishes)")
            st.info("💡 May crack plaster or other brittle finishes")
        else:
            excess_percent = ((max_deflection_mm - limit_general) / limit_general) * 100
            st.error(f"❌ **Excessive deflection!** Exceeds limit by {excess_percent:.1f}%")
            st.warning("⚠️ Increase beam stiffness (depth or I value)")
        
    else:
        st.info("👈 Run analysis first to design reinforcement!")

with tab6:
    st.header("ℹ️ How to Use This Application")
    
    st.markdown("""
    ### 📚 Quick Start Guide
    
    #### 🏗️ What Can This App Analyze?
    
    **Beams:**
    - ✅ Simply supported beams
    - ✅ Cantilever beams
    - ✅ **Continuous beams (UNLIMITED spans!)**
    - ✅ Beams with overhangs
    - ✅ Composite loading on any span
    
    **Frames:**
    - ✅ Portal frames (single/multi-bay)
    - ✅ **Multi-storey frames (any height)**
    - ✅ Braced frames
    - ✅ Frames with inclined members
    - ✅ 2D frames of any configuration
    
    **Special Cases:**
    - ✅ Support settlements (Dx, Dy, Rotation)
    - ✅ Temperature effects
    - ✅ Multiple loads per member
    - ✅ Any combination of supports
    
    ---
    
    #### 1️⃣ Structure Definition
    
    **For Multi-Span Beams (e.g., 5 spans):**
    - **Nodes**: 6 nodes (one at each support)
    - **Members**: 5 members (one per span)
    - **Supports**: 6 supports (typically: 1 Pinned + 5 Rollers)
    - **Loads**: Apply separately to each member
    
    **For Frames (e.g., 2-storey portal):**
    - **Nodes**: Define at all joints (corners, intersections)
    - **Members**: Columns and beams connecting nodes
    - **Supports**: Fixed or Pinned at base
    - **Loads**: On beams (gravity) and columns (lateral)
    
    **Key Rule:** 
    - Number of **nodes** = Number of **joints/supports**
    - Number of **members** = Number of **spans/columns/beams**
    - Multiple loads? Add multiple entries with same member number!
    
    ---
    
    #### 2️⃣ Analysis Method
    This application uses the **Direct Stiffness Method** with:
    - Local to global transformation matrices
    - Fixed end moments for all load types
    - Support settlement effects
    - Slope-deflection equations
    - Matrix assembly and solution using NumPy/SciPy
    
    ---
    
    #### 3️⃣ Sign Conventions
    - **Moments**: Sagging (positive), Hogging (negative)
    - **Shear**: Positive upward on left face
    - **Displacements**: Positive in positive X, Y directions
    - **Axial Forces**: Tension positive, Compression negative
    
    ---
    
    #### 4️⃣ Load Types
    - **UDL**: Uniformly Distributed Load (W1 = intensity in kN/m)
    - **VDL**: Varying Distributed Load (trapezoidal, W1 at start, W2 at end)
    - **Point Load**: Concentrated force at distance 'a' from start (kN)
    - **Moment**: Concentrated moment at distance 'a' from start (kNm)
    
    **Multiple loads on same member?** Just add multiple load entries with the same member number!
    
    Example: Member 2 with UDL + 2 Point Loads:
    ```
    Member | Type       | Values
    -------|------------|--------
    2      | UDL        | W1=10
    2      | Point Load | P=50, a=2
    2      | Point Load | P=30, a=4
    ```
    
    ---
    
    #### 5️⃣ Support Types
    
    | Type   | Restraints         | Use For |
    |--------|-------------------|---------|
    | Fixed  | Dx, Dy, Rotation  | Cantilever ends, frame bases |
    | Pinned | Dx, Dy            | Simple supports, frame bases |
    | Roller | Dy only           | Interior supports, allows horizontal movement |
    
    ---
    
    #### 6️⃣ Support Settlements
    - Enter Dx, Dy, or Rotation values for any support
    - System automatically calculates equivalent loads
    - Useful for foundation settlement analysis
    
    ---
    
    #### 7️⃣ Reinforcement Design
    - Based on **BS 8110:1997** (British Standard)
    - Limit state design method
    - Automatically checks for singly/doubly reinforced sections
    - Suggests bar arrangements (12mm to 32mm bars)
    - Includes shear design check
    - Verifies minimum and maximum steel requirements
    - **Deflection criteria checks (BS 8110 Clause 3.4.6 & Table 3.9)**
    - Compares actual vs allowable span/depth ratios
    - Checks actual deflection against limits (Span/250, Span/350)
    
    ---
    
    ### 🎯 Common Problems & Solutions
    
    **Problem Type 1: 5-Span Continuous Beam**
    ```
    Nodes: 6 (at X = 0, 5, 10, 15, 20, 25)
    Members: 5 (connecting consecutive nodes)
    Supports: 6 (1 Pinned, 5 Rollers)
    Loads: Apply to each member as needed
    ```
    
    **Problem Type 2: Portal Frame**
    ```
    Nodes: 4 (corners: base-left, top-left, top-right, base-right)
    Members: 3 (left column, beam, right column)
    Supports: 2 Fixed (at bases)
    Loads: UDL on beam + lateral load on beam/column
    ```
    
    **Problem Type 3: 3-Storey Frame**
    ```
    Nodes: Define at each floor level and column base
    Members: All columns and beams
    Supports: Fixed at ground level
    Loads: Floor loads on beams, wind loads laterally
    ```
    
    ---
    
    ### 💡 Pro Tips
    
    ✅ Use example presets to understand structure setup  
    ✅ Always check structure preview before analysis  
    ✅ For frames: Draw on paper first, number nodes systematically  
    ✅ Start simple, then add complexity  
    ✅ Export results using browser's print function (Ctrl+P)  
    ✅ For large structures (10+ members): Be patient, analysis may take 5-10 seconds  
    
    ---
    
    ### 🐛 Troubleshooting
    
    **"Analysis failed"**
    - Ensure all nodes are numbered correctly (start from 1)
    - Check that members reference existing nodes
    - Verify at least one support is defined
    - Make sure structure is stable (not a mechanism)
    
    **"Singular matrix"**
    - Structure is unstable (insufficient supports)
    - Check support configuration
    - Ensure proper boundary conditions
    
    **Results seem wrong**
    - Verify load values and units (kN, kNm, m)
    - Check member orientations (Node_I to Node_J)
    - Review support types (Fixed vs Pinned vs Roller)
    
    ---
    
    ### 📖 References
    - Hibbeler, R.C. "Structural Analysis"
    - **BS 8110:1997** - Structural Use of Concrete (British Standard)
    - Matrix Structural Analysis - McGuire, Gallagher, Ziemian
    - Ghali, A., Neville, A.M. "Structural Analysis: A Unified Classical and Matrix Approach"
    - Mosley, Bungey & Hulse "Reinforced Concrete Design to BS 8110"
    
    ---
    
    ### 🎓 For CEG 410 Students
    
    This application demonstrates:
    - ✅ Direct Stiffness Method (matrix approach)
    - ✅ Slope-Deflection equations
    - ✅ Fixed End Moments for all load cases
    - ✅ Support settlement analysis
    - ✅ Complete force analysis (reactions, moments, shears)
    - ✅ **BS 8110:1997 reinforcement design**
    - ✅ Shear design and link spacing calculations
    
    **Perfect for:** Assignments, projects, and verification of hand calculations!
    
    ---
    
    ### 📚 Design Code: BS 8110:1997
    
    **Reinforcement Design Standards:**
    - Concrete grades: fcu = 25, 30, 35, 40 N/mm²
    - Steel grades: fy = 250, 460 N/mm²
    - Partial safety factors: γm = 1.5 (concrete), γm = 1.05 (steel)
    - Minimum cover: 25mm (mild exposure), 35mm (moderate), 40mm+ (severe)
    - K' values: 0.156 (fy=460), 0.207 (fy=250)
    - Min steel: 0.13% bh (fy=460), 0.24% bh (fy=250)
    - Max steel: 4% of gross area
    - Shear design: BS 8110 Table 3.8 and Clause 3.12.7
    
    **Deflection Limits (BS 8110):**
    - Span/depth ratios: Table 3.9 (Simply supported = 20, Continuous = 26)
    - Modification factors for steel stress applied
    - Maximum deflection: Span/250 (general), Span/350 (brittle finishes)
    - Actual deflections calculated from analysis
    """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><b>CEG 410 - Structural Analysis Project</b></p>
    <p>Direct Stiffness Method | Slope-Deflection Analysis | BS 8110:1997 Design</p>
     <p> Developed by Okik's</p
    <p>Made with ❤️ using Streamlit & NumPy</p>
</div>
""", unsafe_allow_html=True)

