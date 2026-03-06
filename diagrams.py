"""
Plotting functions for structure visualization and diagrams
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

def plot_structure(nodes_df, members_df, supports_df, loads_df):
    """
    Plot the structural configuration with supports and loads (improved visualization)
    """
    fig = go.Figure()
    
    # Plot members
    for idx, member in members_df.iterrows():
        node_i = int(member['Node_I'])
        node_j = int(member['Node_J'])
        
        xi = nodes_df[nodes_df['Node'] == node_i]['X'].values[0]
        yi = nodes_df[nodes_df['Node'] == node_i]['Y'].values[0]
        xj = nodes_df[nodes_df['Node'] == node_j]['X'].values[0]
        yj = nodes_df[nodes_df['Node'] == node_j]['Y'].values[0]
        
        fig.add_trace(go.Scatter(
            x=[xi, xj],
            y=[yi, yj],
            mode='lines+text',
            line=dict(color='blue', width=4),
            text=[f'  M{int(member["Member"])}', ''],
            textposition='top center',
            name=f'Member {int(member["Member"])}',
            showlegend=False
        ))
    
    # Plot nodes
    fig.add_trace(go.Scatter(
        x=nodes_df['X'],
        y=nodes_df['Y'],
        mode='markers+text',
        marker=dict(size=12, color='red', symbol='circle'),
        text=[f'  N{int(n)}' for n in nodes_df['Node']],
        textposition='top right',
        name='Nodes',
        showlegend=False
    ))
    
    # Plot supports with improved symbols
    for idx, support in supports_df.iterrows():
        node = int(support['Node'])
        x = nodes_df[nodes_df['Node'] == node]['X'].values[0]
        y = nodes_df[nodes_df['Node'] == node]['Y'].values[0]
        
        if support['Type'] == 'Fixed':
            # Fixed support - filled square
            fig.add_trace(go.Scatter(
                x=[x], y=[y-0.15],
                mode='markers',
                marker=dict(size=20, color='black', symbol='square'),
                name='Fixed',
                showlegend=False
            ))
            # Add hatching lines
            for i in range(5):
                x_offset = -0.15 + i * 0.075
                fig.add_trace(go.Scatter(
                    x=[x + x_offset, x + x_offset],
                    y=[y-0.15, y-0.3],
                    mode='lines',
                    line=dict(color='black', width=1),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
        elif support['Type'] == 'Pinned':
            # Pinned support - triangle
            fig.add_trace(go.Scatter(
                x=[x-0.15, x, x+0.15, x-0.15],
                y=[y-0.3, y, y-0.3, y-0.3],
                fill='toself',
                fillcolor='green',
                mode='lines',
                line=dict(color='green', width=2),
                name='Pinned',
                showlegend=False
            ))
            
        elif support['Type'] == 'Roller':
            # Roller support - triangle on circles
            fig.add_trace(go.Scatter(
                x=[x-0.15, x, x+0.15, x-0.15],
                y=[y-0.25, y, y-0.25, y-0.25],
                fill='toself',
                fillcolor='orange',
                mode='lines',
                line=dict(color='orange', width=2),
                name='Roller',
                showlegend=False
            ))
            # Add rollers (circles)
            fig.add_trace(go.Scatter(
                x=[x-0.1, x+0.1],
                y=[y-0.35, y-0.35],
                mode='markers',
                marker=dict(size=8, color='orange', symbol='circle'),
                showlegend=False
            ))
            
        elif support['Type'] == 'Cantilever':
            # Cantilever - like fixed but indicated differently
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode='markers',
                marker=dict(size=25, color='purple', symbol='diamond'),
                name='Cantilever',
                showlegend=False
            ))
    
    # Plot loads with IMPROVED visualization
    for idx, load in loads_df.iterrows():
        member_id = int(load['Member'])
        member = members_df[members_df['Member'] == member_id].iloc[0]
        node_i = int(member['Node_I'])
        node_j = int(member['Node_J'])
        
        xi = nodes_df[nodes_df['Node'] == node_i]['X'].values[0]
        yi = nodes_df[nodes_df['Node'] == node_i]['Y'].values[0]
        xj = nodes_df[nodes_df['Node'] == node_j]['X'].values[0]
        yj = nodes_df[nodes_df['Node'] == node_j]['Y'].values[0]
        L = np.sqrt((xj - xi)**2 + (yj - yi)**2)
        
        if load['Type'] == 'UDL':
            # Show UDL as multiple arrows with proper spacing
            w = load['W1']
            n_arrows = int(L * 3)  # More arrows for longer spans
            if n_arrows < 5:
                n_arrows = 5
            if n_arrows > 15:
                n_arrows = 15
                
            arrow_length = 0.5 if w > 0 else -0.5
            
            for i in range(n_arrows):
                t = i / (n_arrows - 1)
                x_arrow = xi + t * (xj - xi)
                y_arrow = yi + t * (yj - yi)
                
                fig.add_annotation(
                    x=x_arrow, y=y_arrow,
                    ax=x_arrow, ay=y_arrow + arrow_length,
                    xref='x', yref='y',
                    axref='x', ayref='y',
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1.2,
                    arrowwidth=2,
                    arrowcolor='purple'
                )
            
            # Add UDL label
            x_mid = (xi + xj) / 2
            y_mid = (yi + yj) / 2
            fig.add_annotation(
                x=x_mid, y=y_mid + 0.8,
                text=f'UDL: {w:.1f} kN/m',
                showarrow=False,
                font=dict(size=11, color='purple'),
                bgcolor='white',
                bordercolor='purple',
                borderwidth=1
            )
            
        elif load['Type'] == 'Point Load':
            # Point load as single large arrow
            P = load['P']
            a = load['a']
            t = a / L if L > 0 else 0
            x_load = xi + t * (xj - xi)
            y_load = yi + t * (yj - yi)
            
            arrow_length = 0.8 if P > 0 else -0.8
            
            fig.add_annotation(
                x=x_load, y=y_load,
                ax=x_load, ay=y_load + arrow_length,
                xref='x', yref='y',
                axref='x', ayref='y',
                text=f'{P:.0f} kN',
                showarrow=True,
                arrowhead=2,
                arrowsize=1.5,
                arrowwidth=4,
                arrowcolor='red',
                font=dict(size=11, color='red'),
                bgcolor='white',
                bordercolor='red',
                borderwidth=1
            )
            
        elif load['Type'] == 'VDL':
            # Varying distributed load - trapezoidal
            w1 = load['W1']
            w2 = load['W2']
            n_arrows = 7
            
            for i in range(n_arrows):
                t = i / (n_arrows - 1)
                x_arrow = xi + t * (xj - xi)
                y_arrow = yi + t * (yj - yi)
                
                # Vary arrow length
                w_t = w1 + (w2 - w1) * t
                arrow_length = 0.3 + 0.4 * (w_t / max(w1, w2, 1))
                
                fig.add_annotation(
                    x=x_arrow, y=y_arrow,
                    ax=x_arrow, ay=y_arrow + arrow_length,
                    xref='x', yref='y',
                    axref='x', ayref='y',
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor='purple'
                )
            
            # Add VDL label
            x_mid = (xi + xj) / 2
            y_mid = (yi + yj) / 2
            fig.add_annotation(
                x=x_mid, y=y_mid + 0.8,
                text=f'VDL: {w1:.1f}→{w2:.1f} kN/m',
                showarrow=False,
                font=dict(size=10, color='purple'),
                bgcolor='white',
                bordercolor='purple',
                borderwidth=1
            )
    
    fig.update_layout(
        title='Structural Configuration with Load Diagram',
        xaxis_title='X (m)',
        yaxis_title='Y (m)',
        showlegend=False,
        hovermode='closest',
        height=550,
        yaxis=dict(scaleanchor="x", scaleratio=1),
        plot_bgcolor='#f8f9fa'
    )
    
    return fig


def plot_sfd(members_df, nodes_df, member_forces_df, loads_df):
    """
    Plot Shear Force Diagram for all members
    """
    fig = go.Figure()
    
    if len(members_df) == 0:
        fig.add_annotation(text="No members to plot", showarrow=False)
        return fig
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for idx, member in members_df.iterrows():
        member_id = int(member['Member'])
        node_i = int(member['Node_I'])
        node_j = int(member['Node_J'])
        
        xi = nodes_df[nodes_df['Node'] == node_i]['X'].values[0]
        yi = nodes_df[nodes_df['Node'] == node_i]['Y'].values[0]
        xj = nodes_df[nodes_df['Node'] == node_j]['X'].values[0]
        yj = nodes_df[nodes_df['Node'] == node_j]['Y'].values[0]
        
        L = np.sqrt((xj - xi)**2 + (yj - yi)**2)
        
        # Get member forces
        forces = member_forces_df[member_forces_df['Member'] == member_id].iloc[0]
        V_i = forces['V_i']
        
        # Get loads
        member_loads = loads_df[loads_df['Member'] == member_id]
        
        # Generate shear values along the member
        # Sign convention: V_i is the upward shear force at the left end (node i)
        # As we move from i to j, downward loads reduce the shear
        n_points = 200
        x_local = np.linspace(0, L, n_points)
        V = np.zeros(n_points)
        
        for i, x in enumerate(x_local):
            V_x = V_i  # Start with end shear (upward positive at left face)
            
            for _, load in member_loads.iterrows():
                if load['Type'] == 'UDL':
                    w = load['W1']
                    V_x -= w * x          # downward load reduces V
                elif load['Type'] == 'VDL':
                    w1 = load['W1']
                    w2 = load['W2']
                    V_x -= (w1 * x + (w2 - w1) * x**2 / (2 * L))
                elif load['Type'] == 'Point Load':
                    P = load['P']
                    a = load['a']
                    if x >= a:
                        V_x -= P          # downward point load reduces V
            
            V[i] = V_x
        
        # Plot coordinates
        if abs(xj - xi) > abs(yj - yi):
            x_plot = xi + x_local * (xj - xi) / L
        else:
            x_plot = np.full_like(x_local, (xi + xj) / 2)
        
        color = colors[idx % len(colors)]
        
        fig.add_trace(go.Scatter(
            x=x_plot, y=V,
            mode='lines',
            line=dict(width=3, color=color),
            name=f'Member {member_id}',
            fill='tozeroy',
            fillcolor=color,
            opacity=0.3
        ))
        
        fig.add_trace(go.Scatter(
            x=x_plot, y=np.zeros_like(x_plot),
            mode='lines',
            line=dict(color='black', width=1, dash='dash'),
            showlegend=False,
            hoverinfo='skip'
        ))

        # Annotate max and min shear values
        max_idx = np.argmax(V)
        min_idx = np.argmin(V)
        for ann_idx in set([0, len(V)-1, max_idx, min_idx]):
            if abs(V[ann_idx]) > 0.01:
                fig.add_annotation(
                    x=x_plot[ann_idx], y=V[ann_idx],
                    text=f'{V[ann_idx]:.2f} kN',
                    showarrow=True, arrowhead=2,
                    bgcolor='white', bordercolor=color,
                    font=dict(size=9)
                )
    
    fig.update_layout(
        title='Shear Force Diagram (Positive = Upward on Left Face)',
        xaxis_title='Position (m)',
        yaxis_title='Shear Force (kN)',
        hovermode='closest',
        height=450,
        showlegend=True,
        yaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='black')
    )
    
    return fig


def plot_bmd(members_df, nodes_df, member_forces_df, loads_df):
    """
    Plot Bending Moment Diagram
    """
    fig = go.Figure()
    
    if len(members_df) == 0:
        fig.add_annotation(text="No members to plot", showarrow=False)
        return fig
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for idx, member in members_df.iterrows():
        member_id = int(member['Member'])
        node_i = int(member['Node_I'])
        node_j = int(member['Node_J'])
        
        xi = nodes_df[nodes_df['Node'] == node_i]['X'].values[0]
        yi = nodes_df[nodes_df['Node'] == node_i]['Y'].values[0]
        xj = nodes_df[nodes_df['Node'] == node_j]['X'].values[0]
        yj = nodes_df[nodes_df['Node'] == node_j]['Y'].values[0]
        
        L = np.sqrt((xj - xi)**2 + (yj - yi)**2)
        
        # Get member forces
        forces = member_forces_df[member_forces_df['Member'] == member_id].iloc[0]
        M_i = forces['M_i']
        V_i = forces['V_i']
        
        # Get loads
        member_loads = loads_df[loads_df['Member'] == member_id]
        
        # Generate moment values along the member
        # M(x) = M_i + V_i*x - load contributions
        # Positive moment = sagging (tension at bottom)
        n_points = 200
        x_local = np.linspace(0, L, n_points)
        M = np.zeros(n_points)
        
        for i, x in enumerate(x_local):
            M_x = M_i + V_i * x   # moment from end conditions

            for _, load in member_loads.iterrows():
                if load['Type'] == 'UDL':
                    w = load['W1']
                    M_x -= w * x**2 / 2          # downward UDL sagging
                elif load['Type'] == 'VDL':
                    w1 = load['W1']
                    w2 = load['W2']
                    M_x -= (w1 * x**2 / 2 + (w2 - w1) * x**3 / (6 * L))
                elif load['Type'] == 'Point Load':
                    P = load['P']
                    a = load['a']
                    if x >= a:
                        M_x -= P * (x - a)
                elif load['Type'] == 'Moment':
                    M_load = load['M']
                    a = load['a']
                    if x >= a:
                        M_x += M_load
            
            M[i] = M_x
        
        # Plot coordinates
        if abs(xj - xi) > abs(yj - yi):
            x_plot = xi + x_local * (xj - xi) / L
        else:
            x_plot = np.full_like(x_local, (xi + xj) / 2)
        
        color = colors[idx % len(colors)]
        
        fig.add_trace(go.Scatter(
            x=x_plot, y=M,
            mode='lines',
            line=dict(width=3, color=color),
            name=f'Member {member_id}',
            fill='tozeroy',
            fillcolor=color,
            opacity=0.3
        ))
        
        fig.add_trace(go.Scatter(
            x=x_plot, y=np.zeros_like(x_plot),
            mode='lines',
            line=dict(color='black', width=1, dash='dash'),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Mark max moment
        max_idx = np.argmax(np.abs(M))
        if abs(M[max_idx]) > 0.1:
            fig.add_annotation(
                x=x_plot[max_idx], y=M[max_idx],
                text=f'{M[max_idx]:.2f} kNm',
                showarrow=True,
                arrowhead=2,
                bgcolor='white',
                bordercolor=color
            )
    
    fig.update_layout(
        title='Bending Moment Diagram (Sagging +ve, plotted on tension face)',
        xaxis_title='Position (m)',
        yaxis_title='Bending Moment (kNm)',
        hovermode='closest',
        height=450,
        showlegend=True,
        yaxis=dict(
            autorange='reversed',   # sagging (positive) plotted downward — tension face convention
            zeroline=True, zerolinewidth=2, zerolinecolor='black'
        )
    )
    
    return fig


def plot_deflection(members_df, nodes_df, displacements):
    """
    Plot deflected shape
    """
    fig = go.Figure()
    
    scale_factor = 100
    
    # Original structure
    for idx, member in members_df.iterrows():
        node_i = int(member['Node_I'])
        node_j = int(member['Node_J'])
        
        xi = nodes_df[nodes_df['Node'] == node_i]['X'].values[0]
        yi = nodes_df[nodes_df['Node'] == node_i]['Y'].values[0]
        xj = nodes_df[nodes_df['Node'] == node_j]['X'].values[0]
        yj = nodes_df[nodes_df['Node'] == node_j]['Y'].values[0]
        
        fig.add_trace(go.Scatter(
            x=[xi, xj], y=[yi, yj],
            mode='lines',
            line=dict(color='lightgray', width=2, dash='dash'),
            name='Original' if idx == 0 else '',
            showlegend=(idx == 0)
        ))
    
    # Deflected structure
    for idx, member in members_df.iterrows():
        node_i = int(member['Node_I'])
        node_j = int(member['Node_J'])
        
        xi = nodes_df[nodes_df['Node'] == node_i]['X'].values[0]
        yi = nodes_df[nodes_df['Node'] == node_i]['Y'].values[0]
        xj = nodes_df[nodes_df['Node'] == node_j]['X'].values[0]
        yj = nodes_df[nodes_df['Node'] == node_j]['Y'].values[0]
        
        dxi = displacements[(node_i - 1) * 3] * scale_factor
        dyi = displacements[(node_i - 1) * 3 + 1] * scale_factor
        dxj = displacements[(node_j - 1) * 3] * scale_factor
        dyj = displacements[(node_j - 1) * 3 + 1] * scale_factor
        
        fig.add_trace(go.Scatter(
            x=[xi + dxi, xj + dxj],
            y=[yi + dyi, yj + dyj],
            mode='lines',
            line=dict(color='red', width=3),
            name='Deflected' if idx == 0 else '',
            showlegend=(idx == 0)
        ))
    
    fig.update_layout(
        title=f'Deflected Shape (Scale: {scale_factor}x)',
        xaxis_title='X (m)',
        yaxis_title='Y (m)',
        hovermode='closest',
        height=500,
        yaxis=dict(scaleanchor="x", scaleratio=1)
    )
    
    return fig
