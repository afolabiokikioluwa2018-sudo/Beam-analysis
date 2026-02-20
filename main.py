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
st.markdown('<div class="sub-header">CEG 410 - Structural Analysis using Direct Stiffness Method & Slope Deflection</div>', unsafe_allow_html=True)

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
    st.subheader("🔧 Reinforcement Design")
    st.session_state.fck = st.number_input("Concrete Grade fck (N/mm²)", value=25.0, min_value=15.0)
    st.session_state.fy = st.number_input("Steel Grade fy (N/mm²)", value=415.0, min_value=250.0)
    st.session_state.beam_width = st.number_input("Beam Width b (mm)", value=300.0, min_value=150.0)
    st.session_state.beam_depth = st.number_input("Beam Depth d (mm)", value=500.0, min_value=200.0)
    st.session_state.cover = st.number_input("Clear Cover (mm)", value=25.0, min_value=20.0)

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
        
        st.subheader("📊 Shear Force & Bending Moment at Span Ends")
        st.write("Values at 0% (start) and 100% (end) of each span")
        
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
            V_j = -forces['V_j']
            M_j = forces['M_j']
            
            # Calculate SF and BM at 0% (start of span)
            SF_start = V_i
            BM_start = M_i
            
            # Calculate SF and BM at 100% (end of span)
            # Get loads on this member
            member_loads = st.session_state.loads[st.session_state.loads['Member'] == member_id]
            
            # Calculate shear and moment at end considering all loads
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
                'From Node': node_i,
                'To Node': node_j,
                'Length (m)': f'{L:.3f}',
                'SF @ 0% (kN)': f'{SF_start:.3f}',
                'BM @ 0% (kNm)': f'{BM_start:.3f}',
                'SF @ 100% (kN)': f'{SF_end:.3f}',
                'BM @ 100% (kNm)': f'{BM_end:.3f}'
            })
        
        span_forces_df = pd.DataFrame(span_forces)
        st.dataframe(span_forces_df, use_container_width=True, hide_index=True)
        
        # Add explanation
        st.info("""
        📌 **Note:** 
        - **0%** = Start of span (at Node I)
        - **100%** = End of span (at Node J)
        - **Positive SF** = Upward on left face
        - **Positive BM** = Sagging (tension at bottom)
        """)
        
    else:
        st.info("👈 Run analysis first to see results!")

with tab4:
    st.header("📈 Force and Moment Diagrams")
    
    if st.session_state.solved:
        try:
            st.subheader("Shear Force Diagram")
            sfd_fig = plot_sfd(st.session_state.members, st.session_state.nodes,
                              st.session_state.member_forces, st.session_state.loads)
            st.plotly_chart(sfd_fig, use_container_width=True)
            
            st.divider()
            
            st.subheader("Bending Moment Diagram")
            bmd_fig = plot_bmd(st.session_state.members, st.session_state.nodes,
                              st.session_state.member_forces, st.session_state.loads)
            st.plotly_chart(bmd_fig, use_container_width=True)
            
            st.divider()
            
            st.subheader("Deflection Diagram")
            defl_fig = plot_deflection(st.session_state.members, st.session_state.nodes,
                                      st.session_state.displacements)
            st.plotly_chart(defl_fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error plotting diagrams: {e}")
    else:
        st.info("👈 Run analysis first to see diagrams!")

with tab5:
    st.header("🏗️ Reinforcement Design")
    
    if st.session_state.solved:
        st.info("📋 Design based on IS 456:2000 and limit state method")
        
        # Get max moment for design
        max_moment = st.session_state.member_forces['M_i'].abs().max()
        max_moment = max(max_moment, st.session_state.member_forces['M_j'].abs().max())
        
        st.metric("Maximum Bending Moment", f"{max_moment:.2f} kNm")
        
        # Design calculations
        b = st.session_state.beam_width
        d = st.session_state.beam_depth - st.session_state.cover - 10  # Effective depth
        fck = st.session_state.fck
        fy = st.session_state.fy
        
        # Convert moment to N-mm
        Mu = max_moment * 1e6
        
        # Calculate required steel
        Mu_lim = 0.138 * fck * b * d * d
        
        if Mu <= Mu_lim:
            st.success("✅ Section is singly reinforced")
            # Calculate Ast
            k = Mu / (fck * b * d * d)
            j = 1 - k / 3
            Ast = Mu / (0.87 * fy * j * d)
            
            st.write(f"**Required tension steel:** {Ast:.0f} mm²")
            
            # Suggest bar arrangement
            bar_sizes = [12, 16, 20, 25, 32]
            for bar_dia in bar_sizes:
                bar_area = np.pi * bar_dia**2 / 4
                n_bars = np.ceil(Ast / bar_area)
                if n_bars <= 6:
                    st.write(f"- Use {int(n_bars)} bars of {bar_dia}mm ϕ (Provided: {n_bars*bar_area:.0f} mm²)")
                    break
        else:
            st.warning("⚠️ Section requires doubly reinforced design")
            st.write("Moment exceeds limiting moment - compression steel required")
        
        # Minimum steel check
        Ast_min = 0.85 * b * d / fy
        st.write(f"**Minimum steel required:** {Ast_min:.0f} mm²")
        
    else:
        st.info("👈 Run analysis first to design reinforcement!")

with tab6:
    st.header("ℹ️ How to Use This Application")
    
    st.markdown("""
    ### 📚 Quick Start Guide
    
    #### 1️⃣ Structure Definition
    - **Nodes**: Define joint locations with X, Y coordinates (in meters)
    - **Members**: Connect nodes to create beams/columns. Specify E, I, and A
    - **Supports**: Add Fixed, Pinned, or Roller supports at nodes
    - **Loads**: Apply UDL, VDL, Point loads, or Moments on members
    
    #### 2️⃣ Analysis Method
    This application uses the **Direct Stiffness Method** with:
    - Local to global transformation matrices
    - Fixed end moments for all load types
    - Support settlement effects
    - Slope-deflection equations
    
    #### 3️⃣ Sign Conventions
    - **Moments**: Sagging (positive), Hogging (negative)
    - **Shear**: Positive upward on left face
    - **Displacements**: Positive in positive X, Y directions
    
    #### 4️⃣ Load Types
    - **UDL**: Uniformly Distributed Load (W1 = W2)
    - **VDL**: Varying Distributed Load (trapezoidal, W1 ≠ W2)
    - **Point Load**: Concentrated force at distance 'a' from start
    - **Moment**: Concentrated moment at distance 'a' from start
    
    #### 5️⃣ Support Settlements
    - Enter Dx, Dy, or Rotation values for any support
    - System automatically calculates equivalent loads
    
    #### 6️⃣ Reinforcement Design
    - Based on IS 456:2000 code
    - Limit state design method
    - Automatically checks for singly/doubly reinforced sections
    
    ### 🎯 Tips
    - Use example presets to get started quickly
    - Always check structure preview before analysis
    - Results include reactions, forces, and diagrams
    - Export results using browser's print function
    
    ### 🐛 Troubleshooting
    - Ensure all nodes are numbered sequentially
    - Check that members reference existing nodes
    - Verify at least one support is defined
    - Make sure load values are in correct units
    
    ### 📖 References
    - Hibbeler, R.C. "Structural Analysis"
    - IS 456:2000 - Indian Standard for RCC Design
    - Matrix Structural Analysis - McGuire, Gallagher, Ziemian
    """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><b>CEG 410 - Structural Analysis Project</b></p>
    <p>Direct Stiffness Method | Slope-Deflection Analysis | IS 456:2000 Design</p>
    <p>Made with ❤️ using Streamlit & NumPy</p>
</div>
""", unsafe_allow_html=True)
