"""
CEG 410 - Beam and Frame Analysis Web Application
Main Streamlit Application File

Features:
- Unlimited beam spans
- Multi-storey 2D frames
- Any support combination
- Support settlements
- SFD, BMD, Deflection
- RCC Reinforcement Design (IS 456:2000)

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

# -------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------
st.set_page_config(
    page_title="Beam & Frame Analysis",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# SESSION STATE INITIALIZATION
# -------------------------------------------------
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

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.markdown('<div class="main-header">🏗️ Beam & Frame Analysis System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">CEG 410 – Unlimited Spans | Multi-Storey Frames | Direct Stiffness Method</div>',
    unsafe_allow_html=True
)

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
with st.sidebar:
    st.header("📋 Project Setup")

    st.subheader("Quick Start Examples")
    example_choice = st.selectbox(
        "Load Example:",
        [
            "None",
            "Simply Supported Beam",
            "Continuous Beam (Multi Span)",
            "Portal Frame",
            "Multi-Storey Frame",
            "Frame with Settlement"
        ]
    )

    st.info("""
    **Capabilities**
    - Unlimited beam spans
    - Multi-storey frames
    - Support settlements
    - Multiple loads per member
    """)

    if st.button("Load Example"):
        if example_choice != "None":
            nodes, members, supports, loads = load_example(example_choice)
            st.session_state.nodes = nodes
            st.session_state.members = members
            st.session_state.supports = supports
            st.session_state.loads = loads
            st.session_state.solved = False
            st.success(f"Loaded: {example_choice}")

    st.divider()

    st.subheader("Default Material Properties")
    default_E = st.number_input("E (kN/m²)", value=200e6, format="%.2e")
    default_I = st.number_input("I (m⁴)", value=0.001, format="%.6f")
    default_A = st.number_input("A (m²)", value=0.01, format="%.4f")

    st.divider()

    st.subheader("🔧 RCC Design Parameters")
    st.session_state.fck = st.number_input("Concrete Grade fck (N/mm²)", value=25.0)
    st.session_state.fy = st.number_input("Steel Grade fy (N/mm²)", value=415.0)
    st.session_state.beam_width = st.number_input("Beam Width b (mm)", value=300.0)
    st.session_state.beam_depth = st.number_input("Beam Depth d (mm)", value=500.0)
    st.session_state.cover = st.number_input("Clear Cover (mm)", value=25.0)

# -------------------------------------------------
# MAIN TABS
# -------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📐 Structure Definition",
    "🔧 Analysis",
    "📊 Results",
    "📈 Diagrams",
    "🏗️ Reinforcement",
    "ℹ️ How to Use"
])

# -------------------------------------------------
# TAB 1 – STRUCTURE DEFINITION
# -------------------------------------------------
with tab1:
    st.header("Structure Definition")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Nodes")
        with st.form("node_form"):
            nid = st.number_input("Node ID", min_value=1, step=1)
            x = st.number_input("X (m)")
            y = st.number_input("Y (m)")
            if st.form_submit_button("Add Node"):
                st.session_state.nodes = pd.concat([
                    st.session_state.nodes,
                    pd.DataFrame({'Node': [nid], 'X': [x], 'Y': [y]})
                ], ignore_index=True)
                st.session_state.solved = False

        st.dataframe(st.session_state.nodes, use_container_width=True)

    with col2:
        st.subheader("Members")
        with st.form("member_form"):
            mid = st.number_input("Member ID", min_value=1, step=1)
            ni = st.number_input("Start Node", min_value=1, step=1)
            nj = st.number_input("End Node", min_value=1, step=1)
            E = st.number_input("E", value=default_E, format="%.2e")
            I = st.number_input("I", value=default_I, format="%.6f")
            A = st.number_input("A", value=default_A, format="%.4f")

            if st.form_submit_button("Add Member"):
                st.session_state.members = pd.concat([
                    st.session_state.members,
                    pd.DataFrame({
                        'Member': [mid],
                        'Node_I': [ni],
                        'Node_J': [nj],
                        'E': [E],
                        'I': [I],
                        'A': [A]
                    })
                ], ignore_index=True)
                st.session_state.solved = False

        st.dataframe(st.session_state.members, use_container_width=True)

# -------------------------------------------------
# TAB 2 – ANALYSIS
# -------------------------------------------------
with tab2:
    st.header("Run Structural Analysis")

    if st.button("🚀 Run Analysis", use_container_width=True):
        try:
            K, dof_map = assemble_global_stiffness(
                st.session_state.nodes,
                st.session_state.members
            )

            F_fixed = calculate_fixed_end_actions(
                st.session_state.members,
                st.session_state.loads,
                dof_map,
                st.session_state.nodes
            )

            K_r, F_r, fixed_dofs = apply_boundary_conditions(
                K, F_fixed,
                st.session_state.supports,
                st.session_state.nodes,
                dof_map
            )

            displacements = solve_displacements(
                K_r, F_r, fixed_dofs,
                len(st.session_state.nodes)
            )

            member_forces = calculate_member_forces(
                st.session_state.members,
                st.session_state.nodes,
                displacements,
                st.session_state.loads,
                dof_map
            )

            reactions = calculate_reactions(
                K,
                displacements,
                F_fixed,
                st.session_state.supports,
                st.session_state.nodes,
                dof_map
            )

            st.session_state.displacements = displacements
            st.session_state.member_forces = member_forces
            st.session_state.reactions = reactions
            st.session_state.solved = True

            st.success("Analysis completed successfully")

        except Exception as e:
            st.error(str(e))

# -------------------------------------------------
# TAB 3 – RESULTS (INCLUDES SPAN-END FORCES)
# -------------------------------------------------
with tab3:
    if st.session_state.solved:
        st.subheader("Support Reactions")
        st.dataframe(st.session_state.reactions)

        st.subheader("Member End Forces")
        st.dataframe(st.session_state.member_forces)

# -------------------------------------------------
# TAB 4 – DIAGRAMS
# -------------------------------------------------
with tab4:
    if st.session_state.solved:
        st.plotly_chart(
            plot_sfd(
                import streamlit as st
from diagrams import plot_sfd

# ---- SESSION STATE SAFETY ----
if "members" not in st.session_state:
    st.session_state.members = []

if "nodes" not in st.session_state:
    st.session_state.nodes = []

if "loads" not in st.session_state:
    st.session_state.loads = []


st.title("Beam & Frame Analysis")

# ---- AFTER USER INPUTS (IMPORTANT) ----
fig = plot_sfd(
    st.session_state.members,
    st.session_state.nodes,
    st.session_state.loads
)

if fig is None:
    st.warning("Add at least one structural member to view the SFD.")
else:
    st.plotly_chart(fig, use_container_width=True)
        )

        st.plotly_chart(
            plot_bmd(
                st.session_state.members,
                st.session_state.nodes,
                st.session_state.member_forces,
                st.session_state.loads
            ),
            use_container_width=True
        )

        st.plotly_chart(
            plot_deflection(
                st.session_state.members,
                st.session_state.nodes,
                st.session_state.displacements
            ),
            use_container_width=True
        )

# -------------------------------------------------
# TAB 5 – RCC DESIGN
# -------------------------------------------------
with tab5:
    if st.session_state.solved:
        max_m = max(
            st.session_state.member_forces['M_i'].abs().max(),
            st.session_state.member_forces['M_j'].abs().max()
        )

        b = st.session_state.beam_width
        d = st.session_state.beam_depth - st.session_state.cover - 10
        fy = st.session_state.fy
        fck = st.session_state.fck

        Mu = max_m * 1e6
        Ast = Mu / (0.87 * fy * d)

        st.metric("Max Moment (kNm)", f"{max_m:.2f}")
        st.metric("Required Steel Area Ast (mm²)", f"{Ast:.0f}")

# -------------------------------------------------
# TAB 6 – HOW TO USE
# -------------------------------------------------
with tab6:
    st.markdown("""
    ### How to Use
    1. Define Nodes and Members
    2. Assign Supports and Loads
    3. Run Analysis
    4. View Results & Diagrams
    5. Perform RCC Design
    """)

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.divider()
st.markdown(
    "<p style='text-align:center;color:#666'>CEG 410 – Structural Analysis Project</p>",
    unsafe_allow_html=True
)

