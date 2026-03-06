"""
diagrams.py — Structural Analysis Visualization
================================================
Standard UK/Eurocode structural engineering conventions:

  BMD : Moments plotted on the TENSION FACE
        Sagging (M > 0, tension at bottom) → plotted BELOW the beam axis
        Hogging  (M < 0, tension at top)   → plotted ABOVE the beam axis
        Implementation: plot (−M) on a normal y-axis
          sagging  M > 0  →  −M < 0  →  below zero  ✓
          hogging  M < 0  →  −M > 0  →  above zero  ✓

  SFD : Positive shear (upward on left face) → plotted ABOVE baseline
        No axis reversal.

  Point loads create KINKS in the BMD (slope change, moment is continuous).
  UDL creates PARABOLIC segments.
  Kinks are captured by dense sampling around every point-load position.
"""

import plotly.graph_objects as go
import numpy as np
import pandas as pd


# ── colour palette (distinct, colourblind-friendly) ───────────────────────
_MEMBER_COLORS = [
    ('#1565C0', 'rgba(21,101,192,0.20)'),   # blue
    ('#C62828', 'rgba(198,40,40,0.20)'),    # red
    ('#2E7D32', 'rgba(46,125,50,0.20)'),    # green
    ('#6A1B9A', 'rgba(106,27,154,0.20)'),   # purple
    ('#E65100', 'rgba(230,81,0,0.20)'),     # orange
    ('#00695C', 'rgba(0,105,92,0.20)'),     # teal
    ('#AD1457', 'rgba(173,20,87,0.20)'),    # pink
]

def _line_color(idx):  return _MEMBER_COLORS[idx % len(_MEMBER_COLORS)][0]
def _fill_color(idx):  return _MEMBER_COLORS[idx % len(_MEMBER_COLORS)][1]


# ══════════════════════════════════════════════════════════════════════════
#  CORE CALCULATION HELPERS
# ══════════════════════════════════════════════════════════════════════════

def _sample_x(L, member_loads, n=400):
    """
    Build an x-array [0, L] with:
    • n uniformly spaced points
    • triple points around every point-load / moment position
      (x-ε, x, x+ε) so that slope-change kinks are rendered sharply.
    """
    xs = list(np.linspace(0.0, L, n))
    eps = L * 1e-6
    for _, ld in member_loads.iterrows():
        if ld['Type'] in ('Point Load', 'Moment'):
            a = float(ld['a'])
            if eps < a < L - eps:
                xs += [a - eps, a, a + eps]
    xs = np.array(sorted(set(np.clip(xs, 0.0, L))))
    return xs


def _shear(x_arr, V_i, member_loads, L):
    """V(x) — positive = upward on left face."""
    V = np.full(len(x_arr), V_i)
    for _, ld in member_loads.iterrows():
        t = ld['Type']
        if t == 'UDL':
            w = float(ld['W1'])
            V -= w * x_arr
        elif t == 'VDL':
            w1, w2 = float(ld['W1']), float(ld['W2'])
            V -= w1 * x_arr + (w2 - w1) * x_arr**2 / (2.0 * L)
        elif t == 'Point Load':
            a = float(ld['a'])
            P = float(ld['P'])
            # step function: strictly after load position
            V -= P * (x_arr > a).astype(float)
    return V


def _moment(x_arr, M_i, V_i, member_loads, L):
    """M(x) — sagging = positive."""
    M = M_i + V_i * x_arr
    for _, ld in member_loads.iterrows():
        t = ld['Type']
        if t == 'UDL':
            w = float(ld['W1'])
            M -= w * x_arr**2 / 2.0
        elif t == 'VDL':
            w1, w2 = float(ld['W1']), float(ld['W2'])
            M -= w1 * x_arr**2 / 2.0 + (w2 - w1) * x_arr**3 / (6.0 * L)
        elif t == 'Point Load':
            a = float(ld['a'])
            P = float(ld['P'])
            # ramp after load position — creates kink (slope change)
            mask = x_arr > a
            M[mask] -= P * (x_arr[mask] - a)
        elif t == 'Moment':
            a = float(ld['a'])
            M_val = float(ld['M'])
            M[x_arr >= a] += M_val
    return M


def _geom(member, nodes_df):
    ni = int(member['Node_I']); nj = int(member['Node_J'])
    xi = float(nodes_df[nodes_df['Node']==ni]['X'].values[0])
    yi = float(nodes_df[nodes_df['Node']==ni]['Y'].values[0])
    xj = float(nodes_df[nodes_df['Node']==nj]['X'].values[0])
    yj = float(nodes_df[nodes_df['Node']==nj]['Y'].values[0])
    L  = float(np.sqrt((xj-xi)**2 + (yj-yi)**2))
    return xi, yi, xj, yj, L


def _to_global_x(xi, xj, yi, yj, L, x_local):
    """Map local x-coordinate to global X for horizontal/near-horizontal members."""
    if abs(xj - xi) >= abs(yj - yi):
        return xi + x_local * (xj - xi) / L
    else:
        return np.full(len(x_local), (xi + xj) / 2.0)


# ══════════════════════════════════════════════════════════════════════════
#  STRUCTURE DIAGRAM
# ══════════════════════════════════════════════════════════════════════════

def plot_structure(nodes_df, members_df, supports_df, loads_df):
    fig = go.Figure()

    xspan = float(nodes_df['X'].max() - nodes_df['X'].min()) if len(nodes_df) > 1 else 1.0
    xspan = max(xspan, 1.0)
    sym   = max(0.035 * xspan, 0.12)
    arh   = max(0.055 * xspan, 0.40)

    # ── members ──────────────────────────────────────────────────────────
    for _, mb in members_df.iterrows():
        ni = int(mb['Node_I']); nj = int(mb['Node_J'])
        xi = float(nodes_df[nodes_df['Node']==ni]['X'].values[0])
        yi = float(nodes_df[nodes_df['Node']==ni]['Y'].values[0])
        xj = float(nodes_df[nodes_df['Node']==nj]['X'].values[0])
        yj = float(nodes_df[nodes_df['Node']==nj]['Y'].values[0])
        fig.add_trace(go.Scatter(x=[xi,xj], y=[yi,yj], mode='lines+text',
            line=dict(color='#1565C0', width=6),
            text=[f'  M{int(mb["Member"])}', ''],
            textposition='top center',
            textfont=dict(size=12, color='#1565C0'),
            showlegend=False, hoverinfo='skip'))

    # ── nodes ─────────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=nodes_df['X'], y=nodes_df['Y'],
        mode='markers+text',
        marker=dict(size=12, color='#C62828', symbol='circle',
                    line=dict(color='white', width=2)),
        text=[f'  N{int(n)}' for n in nodes_df['Node']],
        textposition='top right',
        textfont=dict(size=11, color='#C62828'),
        showlegend=False, name='Nodes'))

    # ── supports ──────────────────────────────────────────────────────────
    for _, sup in supports_df.iterrows():
        node = int(sup['Node']); stype = sup['Type']
        sx = float(nodes_df[nodes_df['Node']==node]['X'].values[0])
        sy = float(nodes_df[nodes_df['Node']==node]['Y'].values[0])

        if stype in ('Fixed', 'Cantilever'):
            for k in range(6):
                xo = sx - sym*1.2 + k * sym*0.48
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
                x=[sx - sym, sx, sx + sym, sx - sym],
                y=[sy - sym*1.5, sy, sy - sym*1.5, sy - sym*1.5],
                fill='toself', fillcolor='#43A047',
                mode='lines', line=dict(color='#2E7D32', width=2),
                showlegend=False, hoverinfo='skip'))
            # ground line
            fig.add_trace(go.Scatter(
                x=[sx - sym*1.2, sx + sym*1.2],
                y=[sy - sym*1.5, sy - sym*1.5],
                mode='lines', line=dict(color='#2E7D32', width=3),
                showlegend=False, hoverinfo='skip'))

        elif stype == 'Roller':
            fig.add_trace(go.Scatter(
                x=[sx - sym, sx, sx + sym, sx - sym],
                y=[sy - sym, sy, sy - sym, sy - sym],
                fill='toself', fillcolor='#FB8C00',
                mode='lines', line=dict(color='#E65100', width=2),
                showlegend=False, hoverinfo='skip'))
            # rollers
            for xo in [sx - sym*0.5, sx, sx + sym*0.5]:
                fig.add_trace(go.Scatter(x=[xo], y=[sy - sym*1.35],
                    mode='markers',
                    marker=dict(size=7, color='#E65100',
                                symbol='circle',
                                line=dict(color='#BF360C', width=1)),
                    showlegend=False, hoverinfo='skip'))
            fig.add_trace(go.Scatter(
                x=[sx - sym*1.2, sx + sym*1.2],
                y=[sy - sym*1.7, sy - sym*1.7],
                mode='lines', line=dict(color='#E65100', width=3),
                showlegend=False, hoverinfo='skip'))

    # ── loads ─────────────────────────────────────────────────────────────
    for _, ld in loads_df.iterrows():
        mid    = int(ld['Member'])
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
                fig.add_annotation(x=xa, y=ya, ax=xa, ay=ya + arh,
                    xref='x', yref='y', axref='x', ayref='y',
                    showarrow=True, arrowhead=2, arrowsize=1.2,
                    arrowwidth=2, arrowcolor='#7B1FA2')
            xm = (xi+xj)/2; ym = (yi+yj)/2
            fig.add_annotation(x=xm, y=ym + arh*1.8,
                text=f'<b>w = {w:.1f} kN/m</b>', showarrow=False,
                font=dict(size=11, color='#7B1FA2'),
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='#7B1FA2', borderwidth=1.5)

        elif ld['Type'] == 'Point Load':
            P = float(ld['P']); a = float(ld['a'])
            t = a/L if L > 0 else 0
            xa = xi + t*(xj-xi); ya = yi + t*(yj-yi)
            fig.add_annotation(x=xa, y=ya, ax=xa, ay=ya + arh*2.0,
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
                al = arh * max(0.3, abs(wt)/max(abs(w1),abs(w2),1e-6))
                fig.add_annotation(x=xa, y=ya, ax=xa, ay=ya + al,
                    xref='x', yref='y', axref='x', ayref='y',
                    showarrow=True, arrowhead=2, arrowsize=1.0,
                    arrowwidth=2, arrowcolor='#7B1FA2')
            fig.add_annotation(x=(xi+xj)/2, y=(yi+yj)/2 + arh*1.8,
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


# ══════════════════════════════════════════════════════════════════════════
#  SHEAR FORCE DIAGRAM
# ══════════════════════════════════════════════════════════════════════════

def plot_sfd(members_df, nodes_df, member_forces_df, loads_df):
    """
    SFD — standard orientation:
      Positive shear (upward on left face) → ABOVE baseline
      Negative shear                        → BELOW baseline
    """
    fig = go.Figure()
    if len(members_df) == 0:
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

        # Filled area
        fig.add_trace(go.Scatter(
            x=x_plot, y=V,
            mode='lines',
            line=dict(width=2.5, color=lc),
            fill='tozeroy',
            fillcolor=fc,
            name=f'Member {mid}',
            customdata=np.column_stack([x_local, V]),
            hovertemplate='x = %{customdata[0]:.3f} m<br>'
                          'V = %{customdata[1]:.3f} kN<extra>M' + str(mid) + '</extra>'))

        # Zero baseline
        fig.add_trace(go.Scatter(
            x=[x_plot[0], x_plot[-1]], y=[0, 0],
            mode='lines', line=dict(color='black', width=1.5, dash='dot'),
            showlegend=False, hoverinfo='skip'))

        # Annotate start, end, max, min
        key_pts = {
            round(float(x_plot[0]),  4): (x_plot[0],  V[0]),
            round(float(x_plot[-1]), 4): (x_plot[-1], V[-1]),
            round(float(x_plot[int(np.argmax(V))]), 4): (x_plot[int(np.argmax(V))], V.max()),
            round(float(x_plot[int(np.argmin(V))]), 4): (x_plot[int(np.argmin(V))], V.min()),
        }
        for _, (ax_, av_) in key_pts.items():
            if abs(av_) > 0.05:
                fig.add_annotation(
                    x=ax_, y=av_,
                    text=f'<b>{av_:.2f} kN</b>',
                    showarrow=True, arrowhead=2, arrowsize=0.8, arrowwidth=1.5,
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


# ══════════════════════════════════════════════════════════════════════════
#  BENDING MOMENT DIAGRAM
#
#  Engineering convention (BS 8110 / Eurocode / Hibbeler):
#    Sagging (M > 0, tension bottom) → drawn BELOW the beam axis
#    Hogging  (M < 0, tension top)   → drawn ABOVE the beam axis
#
#  Implementation:
#    Plot  y = −M(x)  on a NORMAL (non-reversed) y-axis.
#      Sagging  M > 0  →  y = −M < 0  →  below zero  ✓
#      Hogging  M < 0  →  y = −M > 0  →  above zero  ✓
#
#  Point loads cause a KINK (slope change) in the BMD — captured by
#  inserting triple sample points around every point-load position.
#  UDL produces a PARABOLA — captured by dense uniform sampling.
# ══════════════════════════════════════════════════════════════════════════

def plot_bmd(members_df, nodes_df, member_forces_df, loads_df):
    fig = go.Figure()
    if len(members_df) == 0:
        return fig

    for idx, member in members_df.iterrows():
        mid = int(member['Member'])
        xi, yi, xj, yj, L = _geom(member, nodes_df)

        forces = member_forces_df[member_forces_df['Member']==mid].iloc[0]
        M_i    = float(forces['M_i'])
        V_i    = float(forces['V_i'])
        member_loads = loads_df[loads_df['Member']==mid]

        # ── Build M(x) array with kinks at point-load positions ──────────
        x_local = _sample_x(L, member_loads, n=500)
        M_real  = _moment(x_local, M_i, V_i, member_loads, L)

        # ── Negate for plotting (sagging below, hogging above) ────────────
        M_plot  = -M_real

        x_plot  = _to_global_x(xi, xj, yi, yj, L, x_local)

        lc = _line_color(idx)
        fc = _fill_color(idx)

        # ── Separate positive and negative regions for correct fill ───────
        # Split into segments where M_plot > 0 (hogging, above) and < 0 (sagging, below)
        # Use a single trace with fill='tozeroy' — Plotly handles the split automatically
        fig.add_trace(go.Scatter(
            x=x_plot,
            y=M_plot,
            mode='lines',
            line=dict(width=2.5, color=lc),
            fill='tozeroy',
            fillcolor=fc,
            name=f'Member {mid}',
            customdata=np.column_stack([x_local, M_real]),
            hovertemplate='x = %{customdata[0]:.3f} m<br>'
                          'M = %{customdata[1]:.3f} kNm<extra>M' + str(mid) + '</extra>'))

        # ── Zero baseline (beam axis) ─────────────────────────────────────
        fig.add_trace(go.Scatter(
            x=[x_plot[0], x_plot[-1]], y=[0.0, 0.0],
            mode='lines',
            line=dict(color='black', width=2, dash='solid'),
            showlegend=False, hoverinfo='skip'))

        # ── Annotate key moments ──────────────────────────────────────────
        # Find: max sagging (most positive M), max hogging (most negative M),
        # and kink values at each point-load location
        annotations = {}

        # Peak sagging
        if M_real.max() > 0.1:
            i_peak = int(np.argmax(M_real))
            annotations[round(float(x_plot[i_peak]),3)] = (
                x_plot[i_peak], M_plot[i_peak], M_real[i_peak])

        # Peak hogging
        if M_real.min() < -0.1:
            i_peak = int(np.argmin(M_real))
            annotations[round(float(x_plot[i_peak]),3)] = (
                x_plot[i_peak], M_plot[i_peak], M_real[i_peak])

        # Values at point-load kink positions
        for _, ld in member_loads.iterrows():
            if ld['Type'] == 'Point Load':
                a_pos = float(ld['a'])
                # find index closest to a_pos
                i_k = int(np.argmin(np.abs(x_local - a_pos)))
                key = round(float(x_plot[i_k]), 3)
                if key not in annotations and abs(M_real[i_k]) > 0.1:
                    annotations[key] = (x_plot[i_k], M_plot[i_k], M_real[i_k])

        # End values
        for i_e, ay_off in [(0, 20), (-1, 20)]:
            if abs(M_real[i_e]) > 0.1:
                key = round(float(x_plot[i_e]), 3)
                if key not in annotations:
                    annotations[key] = (x_plot[i_e], M_plot[i_e], M_real[i_e])

        for _, (ax_, ay_plot, am_real) in annotations.items():
            ay_offset = -28 if ay_plot <= 0 else 28
            fig.add_annotation(
                x=ax_, y=ay_plot,
                text=f'<b>{am_real:.2f} kNm</b>',
                showarrow=True, arrowhead=2, arrowsize=0.8, arrowwidth=1.5,
                ax=0, ay=ay_offset,
                bgcolor='rgba(255,255,255,0.90)',
                bordercolor=lc, borderwidth=1.5,
                font=dict(size=9, color=lc))

    # ── Axis zero line (beam reference) ───────────────────────────────────
    fig.add_hline(y=0, line_width=2.5, line_color='black')

    # ── Y-axis labels to explain convention ───────────────────────────────
    fig.update_layout(
        title=dict(
            text='Bending Moment Diagram  '
                 '<span style="color:#C62828">(↓ Sagging +ve)</span>  '
                 '<span style="color:#1565C0">(↑ Hogging −ve)</span>',
            font=dict(size=15)),
        xaxis_title='Position (m)',
        yaxis_title='Bending Moment (kNm)',
        hovermode='closest', height=480, showlegend=True,
        plot_bgcolor='#FAFAFA', paper_bgcolor='white',
        # NO axis reversal — we plot -M so normal axis gives correct convention
        yaxis=dict(
            zeroline=False,
            # Annotate which direction is sagging/hogging via ticksuffix trick:
            autorange=True,
        ),
        legend=dict(orientation='h', yanchor='bottom', y=1.02,
                    xanchor='right', x=1),
        margin=dict(l=70, r=40, t=80, b=60))

    # Add sagging/hogging direction labels on y-axis
    fig.add_annotation(
        x=0, y=0, xref='paper', yref='paper',
        text='← Hogging (−ve)',
        showarrow=False, xanchor='left', yanchor='top',
        font=dict(size=10, color='#1565C0'),
        bgcolor='rgba(255,255,255,0.7)')
    fig.add_annotation(
        x=0, y=0, xref='paper', yref='paper',
        text='← Sagging (+ve)',
        showarrow=False, xanchor='left', yanchor='bottom',
        font=dict(size=10, color='#C62828'),
        bgcolor='rgba(255,255,255,0.7)')

    return fig


# ══════════════════════════════════════════════════════════════════════════
#  DEFLECTION DIAGRAM
# ══════════════════════════════════════════════════════════════════════════

def plot_deflection(members_df, nodes_df, displacements):
    fig = go.Figure()

    max_disp = float(np.max(np.abs(displacements)))
    if max_disp < 1e-12:
        max_disp = 1e-12
    xspan = float(nodes_df['X'].max() - nodes_df['X'].min()) if len(nodes_df) > 1 else 1.0
    scale = max(10, min(1000, round((xspan * 0.05) / max_disp / 10) * 10))

    # Original (grey dashed)
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

    # Deflected (red)
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
            mode='lines',
            line=dict(color='#C62828', width=3),
            name='Deflected' if not shown_defl else '',
            showlegend=not shown_defl))
        shown_defl = True

    fig.update_layout(
        title=dict(text=f'Deflected Shape  (Scale factor: {scale}×)',
                   font=dict(size=16)),
        xaxis_title='X (m)', yaxis_title='Y (m)',
        hovermode='closest', height=420,
        plot_bgcolor='#FAFAFA', paper_bgcolor='white',
        yaxis=dict(scaleanchor='x', scaleratio=1),
        legend=dict(orientation='h', yanchor='bottom', y=1.02,
                    xanchor='right', x=1),
        margin=dict(l=70, r=40, t=70, b=60))
    return fig
