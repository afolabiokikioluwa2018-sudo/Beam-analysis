"""
Plotting functions for structure visualization and diagrams.

SFD convention : Positive = upward force on LEFT face of section
                 Diagram plotted with positive VALUES going UPWARD  (normal orientation)

BMD convention : Positive = SAGGING (tension at bottom fiber)
                 Diagram plotted with sagging going DOWNWARD (tension-face convention)
                 Implementation: plot -M(x) on a normal y-axis so that
                   sagging (M>0) → -M<0 → below baseline  ✓
                   hogging (M<0) → -M>0 → above baseline  ✓
"""

import plotly.graph_objects as go
import numpy as np
import pandas as pd

_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
           '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

def _color(idx):
    return _COLORS[idx % len(_COLORS)]


# =========================================================================
#  STRUCTURE PLOT
# =========================================================================

def plot_structure(nodes_df, members_df, supports_df, loads_df):
    fig = go.Figure()
    x_range = float(nodes_df['X'].max() - nodes_df['X'].min()) if len(nodes_df) > 1 else 1.0
    sym = max(0.05 * x_range, 0.15)
    arrow_h = max(0.04 * x_range, 0.5)

    for _, member in members_df.iterrows():
        ni = int(member['Node_I']); nj = int(member['Node_J'])
        xi = nodes_df[nodes_df['Node']==ni]['X'].values[0]
        yi = nodes_df[nodes_df['Node']==ni]['Y'].values[0]
        xj = nodes_df[nodes_df['Node']==nj]['X'].values[0]
        yj = nodes_df[nodes_df['Node']==nj]['Y'].values[0]
        fig.add_trace(go.Scatter(x=[xi,xj], y=[yi,yj], mode='lines+text',
            line=dict(color='#2196F3', width=5),
            text=[f'  M{int(member["Member"])}',''], textposition='top center',
            showlegend=False))

    fig.add_trace(go.Scatter(x=nodes_df['X'], y=nodes_df['Y'],
        mode='markers+text',
        marker=dict(size=12, color='#F44336', symbol='circle'),
        text=[f'  N{int(n)}' for n in nodes_df['Node']],
        textposition='top right', showlegend=False))

    for _, support in supports_df.iterrows():
        node = int(support['Node'])
        sx = nodes_df[nodes_df['Node']==node]['X'].values[0]
        sy = nodes_df[nodes_df['Node']==node]['Y'].values[0]
        stype = support['Type']
        if stype in ('Fixed','Cantilever'):
            fig.add_trace(go.Scatter(x=[sx], y=[sy-sym*0.6], mode='markers',
                marker=dict(size=22, color='#212121', symbol='square'), showlegend=False))
        elif stype == 'Pinned':
            fig.add_trace(go.Scatter(
                x=[sx-sym, sx, sx+sym, sx-sym],
                y=[sy-sym*1.2, sy, sy-sym*1.2, sy-sym*1.2],
                fill='toself', fillcolor='#4CAF50', mode='lines',
                line=dict(color='#4CAF50',width=2), showlegend=False))
        elif stype == 'Roller':
            fig.add_trace(go.Scatter(
                x=[sx-sym, sx, sx+sym, sx-sym],
                y=[sy-sym, sy, sy-sym, sy-sym],
                fill='toself', fillcolor='#FF9800', mode='lines',
                line=dict(color='#FF9800',width=2), showlegend=False))
            fig.add_trace(go.Scatter(x=[sx-sym*0.5, sx+sym*0.5],
                y=[sy-sym*1.4, sy-sym*1.4], mode='markers',
                marker=dict(size=8, color='#FF9800', symbol='circle'), showlegend=False))

    for _, load in loads_df.iterrows():
        mid = int(load['Member'])
        member = members_df[members_df['Member']==mid].iloc[0]
        ni = int(member['Node_I']); nj = int(member['Node_J'])
        xi = nodes_df[nodes_df['Node']==ni]['X'].values[0]
        yi = nodes_df[nodes_df['Node']==ni]['Y'].values[0]
        xj = nodes_df[nodes_df['Node']==nj]['X'].values[0]
        yj = nodes_df[nodes_df['Node']==nj]['Y'].values[0]
        L = np.sqrt((xj-xi)**2+(yj-yi)**2)

        if load['Type'] == 'UDL':
            w = load['W1']
            n_arr = max(5, min(15, int(L*3)))
            al = -arrow_h if w > 0 else arrow_h
            for k in range(n_arr):
                t = k/(n_arr-1)
                xa = xi+t*(xj-xi); ya = yi+t*(yj-yi)
                fig.add_annotation(x=xa, y=ya, ax=xa, ay=ya-al,
                    xref='x', yref='y', axref='x', ayref='y',
                    showarrow=True, arrowhead=2, arrowsize=1.2,
                    arrowwidth=2, arrowcolor='#9C27B0')
            fig.add_annotation(x=(xi+xj)/2, y=(yi+yj)/2+arrow_h*1.5,
                text=f'UDL: {w:.1f} kN/m', showarrow=False,
                font=dict(size=11, color='#9C27B0'),
                bgcolor='white', bordercolor='#9C27B0', borderwidth=1)

        elif load['Type'] == 'Point Load':
            P = load['P']; a = load['a']
            t = a/L if L>0 else 0
            xa = xi+t*(xj-xi); ya = yi+t*(yj-yi)
            al = -arrow_h*1.5 if P>0 else arrow_h*1.5
            fig.add_annotation(x=xa, y=ya, ax=xa, ay=ya-al,
                xref='x', yref='y', axref='x', ayref='y',
                text=f'{P:.0f} kN', showarrow=True,
                arrowhead=2, arrowsize=1.5, arrowwidth=4, arrowcolor='#F44336',
                font=dict(size=11, color='#F44336'),
                bgcolor='white', bordercolor='#F44336', borderwidth=1)

    fig.update_layout(title='Structural Configuration',
        xaxis_title='X (m)', yaxis_title='Y (m)',
        showlegend=False, hovermode='closest', height=500,
        yaxis=dict(scaleanchor='x', scaleratio=1), plot_bgcolor='#f8f9fa')
    return fig


# =========================================================================
#  SHEAR FORCE DIAGRAM
#  V(x) = V_i - cumulative load from 0 to x
#  Positive shear → plots ABOVE baseline (upward)   NO axis flip needed
# =========================================================================

def plot_sfd(members_df, nodes_df, member_forces_df, loads_df):
    fig = go.Figure()
    if len(members_df) == 0:
        fig.add_annotation(text='No members to plot', showarrow=False)
        return fig

    for idx, member in members_df.iterrows():
        mid = int(member['Member'])
        ni  = int(member['Node_I']); nj = int(member['Node_J'])
        xi = nodes_df[nodes_df['Node']==ni]['X'].values[0]
        yi = nodes_df[nodes_df['Node']==ni]['Y'].values[0]
        xj = nodes_df[nodes_df['Node']==nj]['X'].values[0]
        yj = nodes_df[nodes_df['Node']==nj]['Y'].values[0]
        L  = np.sqrt((xj-xi)**2+(yj-yi)**2)

        forces = member_forces_df[member_forces_df['Member']==mid].iloc[0]
        V_i    = float(forces['V_i'])
        member_loads = loads_df[loads_df['Member']==mid]

        n_pts   = 300
        x_local = np.linspace(0, L, n_pts)
        V       = np.zeros(n_pts)

        for k, x in enumerate(x_local):
            Vx = V_i
            for _, ld in member_loads.iterrows():
                if ld['Type'] == 'UDL':
                    Vx -= ld['W1'] * x
                elif ld['Type'] == 'VDL':
                    w1=ld['W1']; w2=ld['W2']
                    Vx -= (w1*x + (w2-w1)*x**2/(2*L))
                elif ld['Type'] == 'Point Load':
                    if x >= ld['a']:
                        Vx -= ld['P']
            V[k] = Vx

        if abs(xj-xi) >= abs(yj-yi):
            x_plot = xi + x_local*(xj-xi)/L
        else:
            x_plot = np.full(n_pts, (xi+xj)/2)

        color = _color(idx)
        fig.add_trace(go.Scatter(x=x_plot, y=V, mode='lines',
            line=dict(width=2.5, color=color),
            name=f'Member {mid}', fill='tozeroy',
            fillcolor=color, opacity=0.25,
            customdata=V,
            hovertemplate='x=%{x:.2f} m<br>V=%{customdata:.3f} kN<extra>M' + str(mid) + '</extra>'))

        fig.add_trace(go.Scatter(x=x_plot, y=np.zeros(n_pts), mode='lines',
            line=dict(color='black', width=1.5, dash='dash'),
            showlegend=False, hoverinfo='skip'))

        annotated = {}
        for ax_, av_ in [(x_plot[0],V[0]),(x_plot[-1],V[-1]),
                         (x_plot[np.argmax(V)],V.max()),
                         (x_plot[np.argmin(V)],V.min())]:
            key = round(float(ax_), 2)
            if key not in annotated and abs(av_) > 1e-3:
                annotated[key] = True
                fig.add_annotation(x=ax_, y=av_,
                    text=f'{av_:.2f} kN',
                    showarrow=True, arrowhead=2, arrowsize=0.8,
                    bgcolor='white', bordercolor=color,
                    font=dict(size=9, color=color), ax=0, ay=-25)

    fig.update_layout(
        title='Shear Force Diagram',
        xaxis_title='Position (m)',
        yaxis_title='Shear Force (kN)',
        hovermode='closest', height=450, showlegend=True,
        yaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='black'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    return fig


# =========================================================================
#  BENDING MOMENT DIAGRAM
#
#  KEY: We plot  −M(x)  on a NORMAL (non-reversed) y-axis.
#       Sagging  M > 0  →  −M < 0  →  below baseline  ✓  (tension face)
#       Hogging  M < 0  →  −M > 0  →  above baseline  ✓
#       Hover tooltip shows the REAL M value (not negated).
# =========================================================================

def plot_bmd(members_df, nodes_df, member_forces_df, loads_df):
    fig = go.Figure()
    if len(members_df) == 0:
        fig.add_annotation(text='No members to plot', showarrow=False)
        return fig

    for idx, member in members_df.iterrows():
        mid = int(member['Member'])
        ni  = int(member['Node_I']); nj = int(member['Node_J'])
        xi = nodes_df[nodes_df['Node']==ni]['X'].values[0]
        yi = nodes_df[nodes_df['Node']==ni]['Y'].values[0]
        xj = nodes_df[nodes_df['Node']==nj]['X'].values[0]
        yj = nodes_df[nodes_df['Node']==nj]['Y'].values[0]
        L  = np.sqrt((xj-xi)**2+(yj-yi)**2)

        forces = member_forces_df[member_forces_df['Member']==mid].iloc[0]
        M_i    = float(forces['M_i'])
        V_i    = float(forces['V_i'])
        member_loads = loads_df[loads_df['Member']==mid]

        n_pts   = 300
        x_local = np.linspace(0, L, n_pts)
        M       = np.zeros(n_pts)

        for k, x in enumerate(x_local):
            Mx = M_i + V_i*x
            for _, ld in member_loads.iterrows():
                if ld['Type'] == 'UDL':
                    Mx -= ld['W1']*x**2/2
                elif ld['Type'] == 'VDL':
                    w1=ld['W1']; w2=ld['W2']
                    Mx -= (w1*x**2/2 + (w2-w1)*x**3/(6*L))
                elif ld['Type'] == 'Point Load':
                    if x >= ld['a']:
                        Mx -= ld['P']*(x-ld['a'])
                elif ld['Type'] == 'Moment':
                    if x >= ld['a']:
                        Mx += ld['M']
            M[k] = Mx

        # ── NEGATE so sagging plots BELOW baseline ────────────────────────
        M_plot = -M

        if abs(xj-xi) >= abs(yj-yi):
            x_plot = xi + x_local*(xj-xi)/L
        else:
            x_plot = np.full(n_pts, (xi+xj)/2)

        color = _color(idx)
        fig.add_trace(go.Scatter(x=x_plot, y=M_plot, mode='lines',
            line=dict(width=2.5, color=color),
            name=f'Member {mid}', fill='tozeroy',
            fillcolor=color, opacity=0.25,
            customdata=M,
            hovertemplate='x=%{x:.2f} m<br>M=%{customdata:.3f} kNm<extra>M' + str(mid) + '</extra>'))

        fig.add_trace(go.Scatter(x=x_plot, y=np.zeros(n_pts), mode='lines',
            line=dict(color='black', width=1.5, dash='dash'),
            showlegend=False, hoverinfo='skip'))

        # Peak annotation (show real M value)
        peak_idx = int(np.argmax(np.abs(M)))
        if abs(M[peak_idx]) > 0.1:
            fig.add_annotation(x=x_plot[peak_idx], y=M_plot[peak_idx],
                text=f'{M[peak_idx]:.2f} kNm',
                showarrow=True, arrowhead=2,
                bgcolor='white', bordercolor=color,
                font=dict(size=9, color=color), ax=0, ay=-30)

        # End moment annotations
        for ax_, am_, amp_ in [(x_plot[0],M[0],M_plot[0]),(x_plot[-1],M[-1],M_plot[-1])]:
            if abs(am_) > 0.1:
                fig.add_annotation(x=ax_, y=amp_,
                    text=f'{am_:.2f} kNm',
                    showarrow=True, arrowhead=2,
                    bgcolor='white', bordercolor=color,
                    font=dict(size=9, color=color), ax=0, ay=25)

    fig.update_layout(
        title='Bending Moment Diagram  (Sagging +ve — plotted on tension face)',
        xaxis_title='Position (m)',
        yaxis_title='Bending Moment (kNm)  [↓ Sagging | ↑ Hogging]',
        hovermode='closest', height=450, showlegend=True,
        yaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='black'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    return fig


# =========================================================================
#  DEFLECTION DIAGRAM
# =========================================================================

def plot_deflection(members_df, nodes_df, displacements):
    fig = go.Figure()
    max_disp = float(np.max(np.abs(displacements))) if np.max(np.abs(displacements)) > 0 else 1e-6
    x_range  = float(nodes_df['X'].max()-nodes_df['X'].min()) if len(nodes_df)>1 else 1.0
    scale    = (x_range*0.05)/max_disp if max_disp>0 else 100
    scale    = max(10, min(500, round(scale/10)*10))

    for idx, member in members_df.iterrows():
        ni=int(member['Node_I']); nj=int(member['Node_J'])
        xi=nodes_df[nodes_df['Node']==ni]['X'].values[0]
        yi=nodes_df[nodes_df['Node']==ni]['Y'].values[0]
        xj=nodes_df[nodes_df['Node']==nj]['X'].values[0]
        yj=nodes_df[nodes_df['Node']==nj]['Y'].values[0]
        fig.add_trace(go.Scatter(x=[xi,xj], y=[yi,yj], mode='lines',
            line=dict(color='lightgray',width=2,dash='dash'),
            name='Original' if idx==0 else '', showlegend=(idx==0)))

    for idx, member in members_df.iterrows():
        ni=int(member['Node_I']); nj=int(member['Node_J'])
        xi=nodes_df[nodes_df['Node']==ni]['X'].values[0]
        yi=nodes_df[nodes_df['Node']==ni]['Y'].values[0]
        xj=nodes_df[nodes_df['Node']==nj]['X'].values[0]
        yj=nodes_df[nodes_df['Node']==nj]['Y'].values[0]
        dxi=displacements[(ni-1)*3]*scale; dyi=displacements[(ni-1)*3+1]*scale
        dxj=displacements[(nj-1)*3]*scale; dyj=displacements[(nj-1)*3+1]*scale
        fig.add_trace(go.Scatter(x=[xi+dxi,xj+dxj], y=[yi+dyi,yj+dyj], mode='lines',
            line=dict(color='#F44336',width=3),
            name='Deflected' if idx==0 else '', showlegend=(idx==0)))

    fig.update_layout(title=f'Deflected Shape  (Scale: {scale}×)',
        xaxis_title='X (m)', yaxis_title='Y (m)',
        hovermode='closest', height=450,
        yaxis=dict(scaleanchor='x', scaleratio=1))
    return fig
