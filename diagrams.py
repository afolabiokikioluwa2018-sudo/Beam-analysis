"""
diagrams.py — Structural Analysis Visualization
================================================
Standard UK structural engineering conventions:

  BMD : Sagging (M > 0, tension at bottom) → plotted BELOW the beam axis
        Hogging  (M < 0, tension at top)   → plotted ABOVE the beam axis
        Implementation: plot −M on a normal y-axis
          sagging  M > 0  →  −M < 0  →  below zero  ✓
          hogging  M < 0  →  −M > 0  →  above zero  ✓

  SFD : Positive shear (upward on left face) → ABOVE baseline

  Point loads produce KINKS (slope change) in BMD — sharp, not smooth.
  UDL produces PARABOLIC segments.
"""

import plotly.graph_objects as go
import numpy as np
import pandas as pd

_MEMBER_COLORS = [
    ('#1565C0', 'rgba(21,101,192,0.20)'),
    ('#C62828', 'rgba(198,40,40,0.20)'),
    ('#2E7D32', 'rgba(46,125,50,0.20)'),
    ('#6A1B9A', 'rgba(106,27,154,0.20)'),
    ('#E65100', 'rgba(230,81,0,0.20)'),
    ('#00695C', 'rgba(0,105,92,0.20)'),
    ('#AD1457', 'rgba(173,20,87,0.20)'),
]

def _line_color(idx): return _MEMBER_COLORS[idx % len(_MEMBER_COLORS)][0]
def _fill_color(idx): return _MEMBER_COLORS[idx % len(_MEMBER_COLORS)][1]


# ── helpers ───────────────────────────────────────────────────────────────

def _geom(member, nodes_df):
    ni = int(member['Node_I']); nj = int(member['Node_J'])
    xi = float(nodes_df[nodes_df['Node']==ni]['X'].values[0])
    yi = float(nodes_df[nodes_df['Node']==ni]['Y'].values[0])
    xj = float(nodes_df[nodes_df['Node']==nj]['X'].values[0])
    yj = float(nodes_df[nodes_df['Node']==nj]['Y'].values[0])
    L  = float(np.sqrt((xj-xi)**2 + (yj-yi)**2))
    return xi, yi, xj, yj, L


def _sample_x(L, member_loads, n=500):
    """Dense x-array with triple points at every point-load position for sharp kinks."""
    xs = list(np.linspace(0.0, L, n))
    eps = L * 1e-6
    for _, ld in member_loads.iterrows():
        if ld['Type'] in ('Point Load', 'Moment'):
            a = float(ld['a'])
            if eps < a < L - eps:
                xs += [a - eps, a, a + eps]
    return np.array(sorted(set(np.clip(xs, 0.0, L))))


def _shear(x_arr, V_i, member_loads, L):
    """V(x) — positive = upward on left face. Step at each point load."""
    V = np.full(len(x_arr), float(V_i))
    for _, ld in member_loads.iterrows():
        t = ld['Type']
        if t == 'UDL':
            V -= float(ld['W1']) * x_arr
        elif t == 'VDL':
            w1, w2 = float(ld['W1']), float(ld['W2'])
            V -= w1 * x_arr + (w2 - w1) * x_arr**2 / (2.0 * L)
        elif t == 'Point Load':
            V -= float(ld['P']) * (x_arr > float(ld['a'])).astype(float)
    return V


def _moment(x_arr, M_i, V_i, member_loads, L):
    """M(x) — sagging = positive. Ramp after each point load creates kink."""
    M = float(M_i) + float(V_i) * x_arr
    for _, ld in member_loads.iterrows():
        t = ld['Type']
        if t == 'UDL':
            M -= float(ld['W1']) * x_arr**2 / 2.0
        elif t == 'VDL':
            w1, w2 = float(ld['W1']), float(ld['W2'])
            M -= w1 * x_arr**2 / 2.0 + (w2 - w1) * x_arr**3 / (6.0 * L)
        elif t == 'Point Load':
            a = float(ld['a'])
            mask = x_arr > a
            M[mask] -= float(ld['P']) * (x_arr[mask] - a)
        elif t == 'Moment':
            M[x_arr >= float(ld['a'])] += float(ld['M'])
    return M


def _to_global_x(xi, xj, yi, yj, L, x_local):
    if abs(xj - xi) >= abs(yj - yi):
        return xi + x_local * (xj - xi) / L
    return np.full(len(x_local), (xi + xj) / 2.0)


# ══════════════════════════════════════════════════════════════════════════
#  STRUCTURE DIAGRAM
# ══════════════════════════════════════════════════════════════════════════

def plot_structure(nodes_df, members_df, supports_df, loads_df):
    try:
        fig = go.Figure()
        xspan = max(float(nodes_df['X'].max() - nodes_df['X'].min()), 1.0)
        sym = max(0.035 * xspan, 0.12)
        arh = max(0.055 * xspan, 0.40)

        for _, mb in members_df.iterrows():
            ni = int(mb['Node_I']); nj = int(mb['Node_J'])
            xi = float(nodes_df[nodes_df['Node']==ni]['X'].values[0])
            yi = float(nodes_df[nodes_df['Node']==ni]['Y'].values[0])
            xj = float(nodes_df[nodes_df['Node']==nj]['X'].values[0])
            yj = float(nodes_df[nodes_df['Node']==nj]['Y'].values[0])
            fig.add_trace(go.Scatter(
                x=[xi, xj], y=[yi, yj], mode='lines+text',
                line=dict(color='#1565C0', width=6),
                text=[f'  M{int(mb["Member"])}', ''],
                textposition='top center',
                textfont=dict(size=12, color='#1565C0'),
                showlegend=False, hoverinfo='skip'))

        fig.add_trace(go.Scatter(
            x=nodes_df['X'], y=nodes_df['Y'],
            mode='markers+text',
            marker=dict(size=12, color='#C62828', symbol='circle',
                        line=dict(color='white', width=2)),
            text=[f'  N{int(n)}' for n in nodes_df['Node']],
            textposition='top right',
            textfont=dict(size=11, color='#C62828'),
            showlegend=False))

        for _, sup in supports_df.iterrows():
            node = int(sup['Node']); stype = sup['Type']
            sx = float(nodes_df[nodes_df['Node']==node]['X'].values[0])
            sy = float(nodes_df[nodes_df['Node']==node]['Y'].values[0])
            if stype in ('Fixed', 'Cantilever'):
                for k in range(6):
                    xo = sx - sym * 1.2 + k * sym * 0.48
                    fig.add_trace(go.Scatter(
                        x=[xo, xo - sym*0.3], y=[sy, sy - sym*0.8],
                        mode='lines', line=dict(color='#424242', width=2),
                        showlegend=False, hoverinfo='skip'))
                fig.add_trace(go.Scatter(
                    x=[sx - sym*1.2, sx + sym*1.2], y=[sy, sy],
                    mode='lines', line=dict(color='#424242', width=4),
                    showlegend=False, hoverinfo='skip'))
            elif stype == 'Pinned':
                fig.add_trace(go.Scatter(
                    x=[sx-sym, sx, sx+sym, sx-sym],
                    y=[sy-sym*1.5, sy, sy-sym*1.5, sy-sym*1.5],
                    fill='toself', fillcolor='#43A047',
                    mode='lines', line=dict(color='#2E7D32', width=2),
                    showlegend=False, hoverinfo='skip'))
                fig.add_trace(go.Scatter(
                    x=[sx-sym*1.2, sx+sym*1.2], y=[sy-sym*1.5, sy-sym*1.5],
                    mode='lines', line=dict(color='#2E7D32', width=3),
                    showlegend=False, hoverinfo='skip'))
            elif stype == 'Roller':
                fig.add_trace(go.Scatter(
                    x=[sx-sym, sx, sx+sym, sx-sym],
                    y=[sy-sym, sy, sy-sym, sy-sym],
                    fill='toself', fillcolor='#FB8C00',
                    mode='lines', line=dict(color='#E65100', width=2),
                    showlegend=False, hoverinfo='skip'))
                for xo in [sx-sym*0.5, sx, sx+sym*0.5]:
                    fig.add_trace(go.Scatter(
                        x=[xo], y=[sy-sym*1.35], mode='markers',
                        marker=dict(size=7, color='#E65100', symbol='circle'),
                        showlegend=False, hoverinfo='skip'))
                fig.add_trace(go.Scatter(
                    x=[sx-sym*1.2, sx+sym*1.2], y=[sy-sym*1.7, sy-sym*1.7],
                    mode='lines', line=dict(color='#E65100', width=3),
                    showlegend=False, hoverinfo='skip'))

        for _, ld in loads_df.iterrows():
            mid = int(ld['Member'])
            mb_row = members_df[members_df['Member']==mid].iloc[0]
            ni = int(mb_row['Node_I']); nj = int(mb_row['Node_J'])
            xi = float(nodes_df[nodes_df['Node']==ni]['X'].values[0])
            yi = float(nodes_df[nodes_df['Node']==ni]['Y'].values[0])
            xj = float(nodes_df[nodes_df['Node']==nj]['X'].values[0])
            yj = float(nodes_df[nodes_df['Node']==nj]['Y'].values[0])
            L  = float(np.sqrt((xj-xi)**2 + (yj-yi)**2))

            if ld['Type'] == 'UDL':
                w = float(ld['W1'])
                n_arr = max(5, min(15, int(L * 2.5)))
                for k in range(n_arr):
                    t = k / (n_arr - 1)
                    xa = xi + t*(xj-xi); ya = yi + t*(yj-yi)
                    fig.add_annotation(
                        x=xa, y=ya, ax=xa, ay=ya + arh,
                        xref='x', yref='y', axref='x', ayref='y',
                        showarrow=True, arrowhead=2, arrowsize=1.2,
                        arrowwidth=2, arrowcolor='#7B1FA2')
                fig.add_annotation(
                    x=(xi+xj)/2, y=(yi+yj)/2 + arh*1.8,
                    text=f'<b>w = {w:.1f} kN/m</b>', showarrow=False,
                    font=dict(size=11, color='#7B1FA2'),
                    bgcolor='rgba(255,255,255,0.9)',
                    bordercolor='#7B1FA2', borderwidth=1.5)

            elif ld['Type'] == 'Point Load':
                P = float(ld['P']); a = float(ld['a'])
                t = a/L if L > 0 else 0
                xa = xi + t*(xj-xi); ya = yi + t*(yj-yi)
                fig.add_annotation(
                    x=xa, y=ya, ax=xa, ay=ya + arh*2.0,
                    xref='x', yref='y', axref='x', ayref='y',
                    text=f'<b>{P:.0f} kN</b>',
                    showarrow=True, arrowhead=3, arrowsize=1.5,
                    arrowwidth=3, arrowcolor='#C62828',
                    font=dict(size=11, color='#C62828'),
                    bgcolor='rgba(255,255,255,0.9)',
                    bordercolor='#C62828', borderwidth=1.5,
                    yanchor='bottom')

            elif ld['Type'] == 'VDL':
                w1 = float(ld['W1']); w2 = float(ld['W2'])
                for k in range(8):
                    t  = k / 7
                    xa = xi + t*(xj-xi); ya = yi + t*(yj-yi)
                    wt = w1 + (w2-w1)*t
                    al = arh * max(0.3, abs(wt)/max(abs(w1), abs(w2), 1e-6))
                    fig.add_annotation(
                        x=xa, y=ya, ax=xa, ay=ya + al,
                        xref='x', yref='y', axref='x', ayref='y',
                        showarrow=True, arrowhead=2, arrowsize=1.0,
                        arrowwidth=2, arrowcolor='#7B1FA2')
                fig.add_annotation(
                    x=(xi+xj)/2, y=(yi+yj)/2 + arh*1.8,
                    text=f'<b>VDL {w1:.1f}→{w2:.1f} kN/m</b>', showarrow=False,
                    font=dict(size=11, color='#7B1FA2'),
                    bgcolor='rgba(255,255,255,0.9)',
                    bordercolor='#7B1FA2', borderwidth=1.5)

        fig.update_layout(
            title=dict(text='Structural Configuration', font=dict(size=16)),
            xaxis_title='X (m)', yaxis_title='Y (m)',
            showlegend=False, hovermode='closest', height=480,
            yaxis=dict(scaleanchor='x', scaleratio=1),
            plot_bgcolor='#FAFAFA', paper_bgcolor='white',
            margin=dict(l=60, r=40, t=60, b=60))
        return fig

    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f'Error in structure plot: {str(e)}',
                           showarrow=False, xref='paper', yref='paper', x=0.5, y=0.5)
        return fig


# ══════════════════════════════════════════════════════════════════════════
#  SHEAR FORCE DIAGRAM
# ══════════════════════════════════════════════════════════════════════════

def plot_sfd(members_df, nodes_df, member_forces_df, loads_df):
    """
    SFD — positive shear (upward on left face) plots ABOVE baseline.
    Step discontinuities at point-load positions via boolean mask.
    """
    try:
        fig = go.Figure()

        if len(members_df) == 0:
            fig.add_annotation(text='No members to plot', showarrow=False)
            return fig

        for idx, member in members_df.iterrows():
            mid = int(member['Member'])
            xi, yi, xj, yj, L = _geom(member, nodes_df)
            forces = member_forces_df[member_forces_df['Member']==mid].iloc[0]
            V_i    = float(forces['V_i'])
            member_loads = loads_df[loads_df['Member']==mid]

            x_local = _sample_x(L, member_loads)
            V       = _shear(x_local, V_i, member_loads, L)
            x_plot  = _to_global_x(xi, xj, yi, yj, L, x_local)

            lc = _line_color(idx)
            fc = _fill_color(idx)

            fig.add_trace(go.Scatter(
                x=x_plot, y=V, mode='lines',
                line=dict(width=2.5, color=lc),
                fill='tozeroy', fillcolor=fc,
                name=f'Member {mid}',
                customdata=np.column_stack([x_local, V]),
                hovertemplate=(
                    'x = %{customdata[0]:.3f} m<br>'
                    'V = %{customdata[1]:.3f} kN'
                    '<extra>M' + str(mid) + '</extra>')))

            fig.add_trace(go.Scatter(
                x=[x_plot[0], x_plot[-1]], y=[0, 0],
                mode='lines', line=dict(color='black', width=1.5, dash='dot'),
                showlegend=False, hoverinfo='skip'))

            # Key value annotations
            seen = {}
            for ax_, av_ in [
                (x_plot[0],  V[0]),
                (x_plot[-1], V[-1]),
                (x_plot[int(np.argmax(V))],  float(V.max())),
                (x_plot[int(np.argmin(V))],  float(V.min())),
            ]:
                k = round(float(ax_), 2)
                if k not in seen and abs(av_) > 0.05:
                    seen[k] = True
                    fig.add_annotation(
                        x=ax_, y=av_,
                        text=f'<b>{av_:.2f} kN</b>',
                        showarrow=True, arrowhead=2,
                        ax=0, ay=(-25 if av_ >= 0 else 25),
                        bgcolor='rgba(255,255,255,0.85)',
                        bordercolor=lc, borderwidth=1.5,
                        font=dict(size=9, color=lc))

        fig.add_hline(y=0, line_width=2, line_color='black')
        fig.update_layout(
            title=dict(text='Shear Force Diagram', font=dict(size=16)),
            xaxis_title='Position (m)',
            yaxis_title='Shear Force (kN)',
            hovermode='closest', height=420, showlegend=True,
            plot_bgcolor='#FAFAFA', paper_bgcolor='white',
            yaxis=dict(zeroline=False),
            legend=dict(orientation='h', yanchor='bottom', y=1.02,
                        xanchor='right', x=1),
            margin=dict(l=70, r=40, t=70, b=60))
        return fig

    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f'Error creating SFD: {str(e)}',
                           showarrow=False, xref='paper', yref='paper', x=0.5, y=0.5)
        return fig


# ══════════════════════════════════════════════════════════════════════════
#  BENDING MOMENT DIAGRAM
#
#  Convention (BS 8110 / Hibbeler):
#    Sagging (M > 0) → drawn BELOW beam axis  (tension face = bottom)
#    Hogging  (M < 0) → drawn ABOVE beam axis  (tension face = top)
#
#  Method: plot y = −M(x) on a NORMAL y-axis
#    sagging  M > 0  →  y = −M < 0  →  below zero  ✓
#    hogging  M < 0  →  y = −M > 0  →  above zero  ✓
#
#  Point loads: kinks captured by triple-sampling around each load position.
#  UDL: smooth parabola from dense uniform sampling.
#  Hover shows REAL moment value (not negated).
# ══════════════════════════════════════════════════════════════════════════

def plot_bmd(members_df, nodes_df, member_forces_df, loads_df):
    try:
        fig = go.Figure()

        if len(members_df) == 0:
            fig.add_annotation(text='No members to plot', showarrow=False)
            return fig

        for idx, member in members_df.iterrows():
            mid = int(member['Member'])
            xi, yi, xj, yj, L = _geom(member, nodes_df)
            forces = member_forces_df[member_forces_df['Member']==mid].iloc[0]
            M_i    = float(forces['M_i'])
            V_i    = float(forces['V_i'])
            member_loads = loads_df[loads_df['Member']==mid]

            # Build M(x) with sharp kinks at point-load positions
            x_local = _sample_x(L, member_loads, n=500)
            M_real  = _moment(x_local, M_i, V_i, member_loads, L)

            # Negate: sagging → below baseline, hogging → above baseline
            M_plot  = -M_real

            x_plot  = _to_global_x(xi, xj, yi, yj, L, x_local)
            lc = _line_color(idx)
            fc = _fill_color(idx)

            # Main BMD trace (fill='tozeroy' handles mixed sign automatically)
            fig.add_trace(go.Scatter(
                x=x_plot, y=M_plot,
                mode='lines',
                line=dict(width=2.5, color=lc),
                fill='tozeroy',
                fillcolor=fc,
                name=f'Member {mid}',
                customdata=np.column_stack([x_local, M_real]),
                hovertemplate=(
                    'x = %{customdata[0]:.3f} m<br>'
                    'M = %{customdata[1]:.3f} kNm'
                    '<extra>M' + str(mid) + '</extra>')))

            # Beam axis (zero line for this member)
            fig.add_trace(go.Scatter(
                x=[x_plot[0], x_plot[-1]], y=[0.0, 0.0],
                mode='lines',
                line=dict(color='black', width=2),
                showlegend=False, hoverinfo='skip'))

            # ── Annotations ───────────────────────────────────────────────
            annotations = {}

            # Peak sagging (most positive real M → most negative plotted y)
            if M_real.max() > 0.1:
                i_s = int(np.argmax(M_real))
                annotations[round(float(x_plot[i_s]), 3)] = (
                    x_plot[i_s], M_plot[i_s], M_real[i_s])

            # Peak hogging (most negative real M → most positive plotted y)
            if M_real.min() < -0.1:
                i_h = int(np.argmin(M_real))
                annotations[round(float(x_plot[i_h]), 3)] = (
                    x_plot[i_h], M_plot[i_h], M_real[i_h])

            # Moment value at each point-load kink position
            for _, ld in member_loads.iterrows():
                if ld['Type'] == 'Point Load':
                    a_pos = float(ld['a'])
                    i_k = int(np.argmin(np.abs(x_local - a_pos)))
                    k   = round(float(x_plot[i_k]), 3)
                    if k not in annotations and abs(M_real[i_k]) > 0.1:
                        annotations[k] = (x_plot[i_k], M_plot[i_k], M_real[i_k])

            # End values
            for i_e in [0, -1]:
                if abs(M_real[i_e]) > 0.1:
                    k = round(float(x_plot[i_e]), 3)
                    if k not in annotations:
                        annotations[k] = (x_plot[i_e], M_plot[i_e], M_real[i_e])

            for _, (ax_, ay_p, am_r) in annotations.items():
                ay_off = -28 if ay_p <= 0 else 28
                fig.add_annotation(
                    x=ax_, y=ay_p,
                    text=f'<b>{am_r:.2f} kNm</b>',
                    showarrow=True, arrowhead=2, arrowsize=0.8, arrowwidth=1.5,
                    ax=0, ay=ay_off,
                    bgcolor='rgba(255,255,255,0.90)',
                    bordercolor=lc, borderwidth=1.5,
                    font=dict(size=9, color=lc))

        # Global zero line (beam reference axis)
        fig.add_hline(y=0, line_width=2.5, line_color='black')

        # Direction labels on y-axis
        fig.add_annotation(
            xref='paper', yref='paper', x=0.01, y=0.99,
            text='↑ Hogging (−ve)', showarrow=False,
            xanchor='left', yanchor='top',
            font=dict(size=10, color='#1565C0'),
            bgcolor='rgba(255,255,255,0.8)')
        fig.add_annotation(
            xref='paper', yref='paper', x=0.01, y=0.01,
            text='↓ Sagging (+ve)', showarrow=False,
            xanchor='left', yanchor='bottom',
            font=dict(size=10, color='#C62828'),
            bgcolor='rgba(255,255,255,0.8)')

        fig.update_layout(
            title=dict(
                text=('Bending Moment Diagram  '
                      '<span style="color:#C62828">↓ Sagging (+ve)</span>  '
                      '|  '
                      '<span style="color:#1565C0">↑ Hogging (−ve)</span>'),
                font=dict(size=15)),
            xaxis_title='Position (m)',
            yaxis_title='Bending Moment (kNm)',
            hovermode='closest', height=480, showlegend=True,
            plot_bgcolor='#FAFAFA', paper_bgcolor='white',
            yaxis=dict(zeroline=False, autorange=True),
            legend=dict(orientation='h', yanchor='bottom', y=1.02,
                        xanchor='right', x=1),
            margin=dict(l=70, r=40, t=80, b=60))
        return fig

    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f'Error creating BMD: {str(e)}',
                           showarrow=False, xref='paper', yref='paper', x=0.5, y=0.5)
        return fig


# ══════════════════════════════════════════════════════════════════════════
#  DEFLECTION DIAGRAM
# ══════════════════════════════════════════════════════════════════════════

def plot_deflection(members_df, nodes_df, displacements):
    try:
        fig = go.Figure()
        max_disp = float(np.max(np.abs(displacements)))
        if max_disp < 1e-12:
            max_disp = 1e-12
        xspan = max(float(nodes_df['X'].max() - nodes_df['X'].min()), 1.0)
        scale = max(10, min(1000, round((xspan * 0.05) / max_disp / 10) * 10))

        shown_orig = False
        for _, mb in members_df.iterrows():
            ni = int(mb['Node_I']); nj = int(mb['Node_J'])
            xi = float(nodes_df[nodes_df['Node']==ni]['X'].values[0])
            yi = float(nodes_df[nodes_df['Node']==ni]['Y'].values[0])
            xj = float(nodes_df[nodes_df['Node']==nj]['X'].values[0])
            yj = float(nodes_df[nodes_df['Node']==nj]['Y'].values[0])
            fig.add_trace(go.Scatter(
                x=[xi, xj], y=[yi, yj], mode='lines',
                line=dict(color='#9E9E9E', width=2, dash='dash'),
                name='Original' if not shown_orig else '',
                showlegend=not shown_orig))
            shown_orig = True

        shown_defl = False
        for _, mb in members_df.iterrows():
            ni = int(mb['Node_I']); nj = int(mb['Node_J'])
            xi = float(nodes_df[nodes_df['Node']==ni]['X'].values[0])
            yi = float(nodes_df[nodes_df['Node']==ni]['Y'].values[0])
            xj = float(nodes_df[nodes_df['Node']==nj]['X'].values[0])
            yj = float(nodes_df[nodes_df['Node']==nj]['Y'].values[0])
            dxi = displacements[(ni-1)*3]   * scale
            dyi = displacements[(ni-1)*3+1] * scale
            dxj = displacements[(nj-1)*3]   * scale
            dyj = displacements[(nj-1)*3+1] * scale
            fig.add_trace(go.Scatter(
                x=[xi+dxi, xj+dxj], y=[yi+dyi, yj+dyj],
                mode='lines', line=dict(color='#C62828', width=3),
                name='Deflected' if not shown_defl else '',
                showlegend=not shown_defl))
            shown_defl = True

        fig.update_layout(
            title=dict(text=f'Deflected Shape  (Scale: {scale}×)',
                       font=dict(size=16)),
            xaxis_title='X (m)', yaxis_title='Y (m)',
            hovermode='closest', height=420,
            plot_bgcolor='#FAFAFA', paper_bgcolor='white',
            yaxis=dict(scaleanchor='x', scaleratio=1),
            legend=dict(orientation='h', yanchor='bottom', y=1.02,
                        xanchor='right', x=1),
            margin=dict(l=70, r=40, t=70, b=60))
        return fig

    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f'Error in deflection plot: {str(e)}',
                           showarrow=False, xref='paper', yref='paper', x=0.5, y=0.5)
        return fig
