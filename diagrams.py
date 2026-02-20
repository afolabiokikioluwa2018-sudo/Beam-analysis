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
    
    # Plot loads (simplified representation)
    for idx, load in loads_df.iterrows():
        member_id = int(load['Member'])
        member = members_df[members_df['Member'] == member_id].iloc[0]
        node_i = int(member['Node_I'])
        node_j = int(member['Node_J'])
        
        xi = nodes_df[nodes_df['Node'] == node_i]['X'].values[0]
        yi = nodes_df[nodes_df['Node'] == node_i]['Y'].values[0]
        xj = nodes_df[nodes_df['Node'] == node_j]['X'].values[0]
        yj = nodes_df[nodes_df['Node'] == node_j]['Y'].values[0]
        
        # Mid-point of member
        xm = (xi + xj) / 2
        ym = (yi + yj) / 2
        
        if load['Type'] == 'UDL':
            # Show as distributed arrows
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
            t = a / L
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


def calculate_shear_values(member_id, members_df, nodes_df, member_forces_df, loads_df, n_points=100):
    """
    Calculate detailed shear force values along a member
    Returns: x_positions, shear_values, critical_points
    """
    member = members_df[members_df['Member'] == member_id].iloc[0]
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
    
    # Get loads on this member
    member_loads = loads_df[loads_df['Member'] == member_id]
    
    # Generate positions along member
    x_local = np.linspace(0, L, n_points)
    V = np.zeros(n_points)
    
    # Critical points for exact values
    critical_points = [(0, V_i)]  # Start point
    
    # Calculate shear at each point
    for i, x in enumerate(x_local):
        V[i] = V_i
        
        # Apply loads
        for _, load in member_loads.iterrows():
            if load['Type'] == 'UDL':
                w = load['W1']
                V[i] -= w * x
                
            elif load['Type'] == 'VDL':
                w1 = load['W1']
                w2 = load['W2']
                # Trapezoidal load effect on shear
                w_avg = w1 + (w2 - w1) * x / L
                V[i] -= w_avg * x / 2
                
            elif load['Type'] == 'Point Load':
                P = load['P']
                a = load['a']
                if x >= a:
                    V[i] -= P
                    if abs(x - a) < L/n_points:  # Near point load
                        critical_points.append((a, V_i - P))
    
    # Add end point
    critical_points.append((L, V[-1]))
    
    return x_local, V, critical_points


def calculate_moment_values(member_id, members_df, nodes_df, member_forces_df, loads_df, n_points=100):
    """
    Calculate detailed bending moment values along a member
    Returns: x_positions, moment_values, critical_points
    """
    member = members_df[members_df['Member'] == member_id].iloc[0]
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
    
    # Get loads on this member
    member_loads = loads_df[loads_df['Member'] == member_id]
    
    # Generate positions
    x_local = np.linspace(0, L, n_points)
    M = np.zeros(n_points)
    
    # Critical points
    critical_points = [(0, M_i)]
    
    # Calculate moment at each point
    for i, x in enumerate(x_local):
        M[i] = M_i + V_i * x
        
        # Apply load effects
        for _, load in member_loads.iterrows():
            if load['Type'] == 'UDL':
                w = load['W1']
                M[i] -= w * x**2 / 2
                
            elif load['Type'] == 'VDL':
                w1 = load['W1']
                w2 = load['W2']
                # Trapezoidal load moment effect
                M[i] -= (w1 * x**2 / 2) + ((w2 - w1) * x**3) / (6 * L)
                
            elif load['Type'] == 'Point Load':
                P = load['P']
                a = load['a']
                if x >= a:
                    M[i] -= P * (x - a)
                if abs(x - a) < L/n_points:
                    critical_points.append((a, M[i]))
                    
            elif load['Type'] == 'Moment':
                M_load = load['M']
                a = load['a']
                if x >= a:
                    M[i] += M_load
    
    # Find maximum moment
    max_idx = np.argmax(np.abs(M))
    if max_idx > 0 and max_idx < len(M) - 1:
        critical_points.append((x_local[max_idx], M[max_idx]))
    
    # Add end point
    critical_points.append((L, M[-1]))
    
    return x_local, M, critical_points


def plot_sfd(members_df, nodes_df, member_forces_df, loads_df):
    """
    Plot Shear Force Diagram with detailed values
    """
    fig = make_subplots(
        rows=len(members_df), cols=1,
        subplot_titles=[f'Member {int(m["Member"])}' for _, m in members_df.iterrows()],
        vertical_spacing=0.1
    )
    
    all_shear_data = []
    
    for idx, member in members_df.iterrows():
        member_id = int(member['Member'])
        node_i = int(member['Node_I'])
        node_j = int(member['Node_J'])
        
        xi = nodes_df[nodes_df['Node'] == node_i]['X'].values[0]
        xj = nodes_df[nodes_df['Node'] == node_j]['X'].values[0]
        
        L = abs(xj - xi)
        
        # Calculate detailed shear values
        x_local, V, critical_pts = calculate_shear_values(
            member_id, members_df, nodes_df, member_forces_df, loads_df
        )
        
        # Store data for table
        for pos, val in critical_pts:
            all_shear_data.append({
                'Member': member_id,
                'Position (m)': f'{pos:.3f}',
                'Shear Force (kN)': f'{val:.3f}'
            })
        
        # Global x-coordinates for plotting
        x_global = xi + x_local
        
        # Plot shear force
        fig.add_trace(
            go.Scatter(
                x=x_global,
                y=V,
                mode='lines',
                line=dict(width=3, color='blue'),
                fill='tozeroy',
                fillcolor='rgba(0, 100, 255, 0.2)',
                name=f'M{member_id}',
                hovertemplate='Position: %{x:.2f}m<br>Shear: %{y:.2f} kN<extra></extra>'
            ),
            row=idx+1, col=1
        )
        
        # Add zero line
        fig.add_trace(
            go.Scatter(
                x=[xi, xj],
                y=[0, 0],
                mode='lines',
                line=dict(color='black', width=1, dash='dash'),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=idx+1, col=1
        )
        
        # Add critical point annotations
        for pos, val in critical_pts:
            if abs(val) > 0.01:  # Only annotate significant values
                fig.add_annotation(
                    x=xi + pos,
                    y=val,
                    text=f'{val:.2f}',
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=1,
                    arrowcolor='red',
                    ax=0,
                    ay=-40 if val > 0 else 40,
                    font=dict(size=10, color='red'),
                    row=idx+1, col=1
                )
        
        # Update axes
        fig.update_xaxes(title_text='Position (m)', row=idx+1, col=1)
        fig.update_yaxes(title_text='Shear (kN)', row=idx+1, col=1)
    
    fig.update_layout(
        title_text='Shear Force Diagram (SFD)',
        height=300 * len(members_df),
        showlegend=True
    )
    
    return fig, pd.DataFrame(all_shear_data)


def plot_bmd(members_df, nodes_df, member_forces_df, loads_df):
    """
    Plot Bending Moment Diagram with detailed values
    Sign convention: Sagging positive
    """
    fig = make_subplots(
        rows=len(members_df), cols=1,
        subplot_titles=[f'Member {int(m["Member"])}' for _, m in members_df.iterrows()],
        vertical_spacing=0.1
    )
    
    all_moment_data = []
    
    for idx, member in members_df.iterrows():
        member_id = int(member['Member'])
        node_i = int(member['Node_I'])
        node_j = int(member['Node_J'])
        
        xi = nodes_df[nodes_df['Node'] == node_i]['X'].values[0]
        xj = nodes_df[nodes_df['Node'] == node_j]['X'].values[0]
        
        # Calculate detailed moment values
        x_local, M, critical_pts = calculate_moment_values(
            member_id, members_df, nodes_df, member_forces_df, loads_df
        )
        
        # Store data for table
        for pos, val in critical_pts:
            all_moment_data.append({
                'Member': member_id,
                'Position (m)': f'{pos:.3f}',
                'Moment (kNm)': f'{val:.3f}'
            })
        
        # Global x-coordinates
        x_global = xi + x_local
        
        # Plot moment
        fig.add_trace(
            go.Scatter(
                x=x_global,
                y=M,
                mode='lines',
                line=dict(width=3, color='green'),
                fill='tozeroy',
                fillcolor='rgba(0, 200, 100, 0.2)',
                name=f'M{member_id}',
                hovertemplate='Position: %{x:.2f}m<br>Moment: %{y:.2f} kNm<extra></extra>'
            ),
            row=idx+1, col=1
        )
        
        # Add zero line
        fig.add_trace(
            go.Scatter(
                x=[xi, xj],
                y=[0, 0],
                mode='lines',
                line=dict(color='black', width=1, dash='dash'),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=idx+1, col=1
        )
        
        # Add critical point annotations
        for pos, val in critical_pts:
            if abs(val) > 0.01:
                fig.add_annotation(
                    x=xi + pos,
                    y=val,
                    text=f'{val:.2f}',
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=1,
                    arrowcolor='darkgreen',
                    ax=0,
                    ay=-40 if val > 0 else 40,
                    font=dict(size=10, color='darkgreen'),
                    row=idx+1, col=1
                )
        
        # Update axes
        fig.update_xaxes(title_text='Position (m)', row=idx+1, col=1)
        fig.update_yaxes(title_text='Moment (kNm)', row=idx+1, col=1)
    
    fig.update_layout(
        title_text='Bending Moment Diagram (BMD) - Sagging Positive',
        height=300 * len(members_df),
        showlegend=True
    )
    
    return fig, pd.DataFrame(all_moment_data)


def plot_deflection(members_df, nodes_df, displacements):
    """
    Plot deflected shape of structure
    """
    fig = go.Figure()
    
    scale_factor = 100  # Exaggerate deflections for visibility
    
    # Original structure
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
        
        # Add displacements
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
