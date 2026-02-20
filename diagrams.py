"""
Plotting functions for structure visualization and diagrams
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

def plot_structure(nodes_df, members_df, supports_df, loads_df):
    """
    Plot the structural configuration with supports and loads
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
            line=dict(color='blue', width=3),
            text=[f'  M{int(member["Member"])}', ''],
            textposition='top center',
            name=f'Member {int(member["Member"])}'
        ))
    
    # Plot nodes
    fig.add_trace(go.Scatter(
        x=nodes_df['X'],
        y=nodes_df['Y'],
        mode='markers+text',
        marker=dict(size=12, color='red', symbol='circle'),
        text=[f'  N{int(n)}' for n in nodes_df['Node']],
        textposition='top right',
        name='Nodes'
    ))
    
    # Plot supports
    for idx, support in supports_df.iterrows():
        node = int(support['Node'])
        x = nodes_df[nodes_df['Node'] == node]['X'].values[0]
        y = nodes_df[nodes_df['Node'] == node]['Y'].values[0]
        
        if support['Type'] == 'Fixed':
            marker_symbol = 'square'
            marker_color = 'black'
        elif support['Type'] == 'Pinned':
            marker_symbol = 'triangle-up'
            marker_color = 'green'
        else:  # Roller
            marker_symbol = 'circle'
            marker_color = 'orange'
        
        fig.add_trace(go.Scatter(
            x=[x],
            y=[y],
            mode='markers',
            marker=dict(size=15, color=marker_color, symbol=marker_symbol),
            name=f'{support["Type"]} Support'
        ))
    
    # Plot loads
    for idx, load in loads_df.iterrows():
        member_id = int(load['Member'])
        member = members_df[members_df['Member'] == member_id].iloc[0]
        node_i = int(member['Node_I'])
        node_j = int(member['Node_J'])
        
        xi = nodes_df[nodes_df['Node'] == node_i]['X'].values[0]
        yi = nodes_df[nodes_df['Node'] == node_i]['Y'].values[0]
        xj = nodes_df[nodes_df['Node'] == node_j]['X'].values[0]
        yj = nodes_df[nodes_df['Node'] == node_j]['Y'].values[0]
        
        if load['Type'] == 'UDL':
            n_arrows = 5
            for i in range(n_arrows):
                t = i / (n_arrows - 1)
                x_arrow = xi + t * (xj - xi)
                y_arrow = yi + t * (yj - yi)
                fig.add_annotation(
                    x=x_arrow, y=y_arrow,
                    ax=x_arrow, ay=y_arrow - 0.3,
                    xref='x', yref='y',
                    axref='x', ayref='y',
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor='purple'
                )
        elif load['Type'] == 'Point Load':
            a = load['a']
            L = np.sqrt((xj - xi)**2 + (yj - yi)**2)
            t = a / L if L > 0 else 0
            x_load = xi + t * (xj - xi)
            y_load = yi + t * (yj - yi)
            
            fig.add_annotation(
                x=x_load, y=y_load,
                ax=x_load, ay=y_load - 0.5,
                xref='x', yref='y',
                axref='x', ayref='y',
                text=f'{load["P"]} kN',
                showarrow=True,
                arrowhead=2,
                arrowsize=1.5,
                arrowwidth=3,
                arrowcolor='red'
            )
    
    fig.update_layout(
        title='Structural Configuration',
        xaxis_title='X (m)',
        yaxis_title='Y (m)',
        showlegend=True,
        hovermode='closest',
        height=500,
        yaxis=dict(scaleanchor="x", scaleratio=1)
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
        
        # Generate shear values
        n_points = 200
        x_local = np.linspace(0, L, n_points)
        V = np.zeros(n_points)
        
        for i, x in enumerate(x_local):
            V_x = V_i
            
            for _, load in member_loads.iterrows():
                if load['Type'] == 'UDL':
                    w = load['W1']
                    V_x -= w * x
                elif load['Type'] == 'VDL':
                    w1 = load['W1']
                    w2 = load['W2']
                    V_x -= (w1 * x + (w2 - w1) * x**2 / (2 * L))
                elif load['Type'] == 'Point Load':
                    P = load['P']
                    a = load['a']
                    if x >= a:
                        V_x -= P
            
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
    
    fig.update_layout(
        title='Shear Force Diagram',
        xaxis_title='Position (m)',
        yaxis_title='Shear Force (kN)',
        hovermode='closest',
        height=450,
        showlegend=True
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
        
        # Generate moment values
        n_points = 200
        x_local = np.linspace(0, L, n_points)
        M = np.zeros(n_points)
        
        for i, x in enumerate(x_local):
            M_x = M_i + V_i * x
            
            for _, load in member_loads.iterrows():
                if load['Type'] == 'UDL':
                    w = load['W1']
                    M_x -= w * x**2 / 2
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
        title='Bending Moment Diagram (Sagging Positive)',
        xaxis_title='Position (m)',
        yaxis_title='Bending Moment (kNm)',
        hovermode='closest',
        height=450,
        showlegend=True
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
