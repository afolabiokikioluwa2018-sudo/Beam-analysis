"""
diagrams.py — Force and Moment Diagram Visualization
=====================================================

SFD FORMULA  (derived from FBD, left portion [0, x]):
  ΣFy = 0:
    V(x) = V_i − w·x                      (UDL)
          − P · H(x − a)                   (Point load, H = unit step)
          − (w₁·x + (w₂−w₁)·x²/2L)        (VDL)

  Sign: V(x) > 0 means upward shear on left face → plotted ABOVE baseline

BMD FORMULA  (derived from FBD, left portion [0, x]):
  ΣM_cut = 0  (CCW positive):
    M(x) = M_i + V_i·x − w·x²/2            (UDL)
                        − P·(x−a)·H(x−a)   (Point load — ramp, creates KINK)
                        − (w₁x²/2 + (w₂−w₁)x³/6L)  (VDL)

  Sign: M(x) > 0 → sagging (tension bottom) → plotted ABOVE baseline
        M(x) < 0 → hogging (tension top)    → plotted BELOW baseline

  Plotting convention (standard structural engineering):
    Plot y = M(x) directly on a NORMAL y-axis:
      Sagging  M > 0  →  y > 0  →  ABOVE baseline  ✓
      Hogging  M < 0  →  y < 0  →  BELOW baseline  ✓

KINKS vs SMOOTH CURVES:
  Point load → V has a STEP DISCONTINUITY at x=a (shear jumps by P)
  Point load → M has a SLOPE CHANGE (kink) at x=a (moment is continuous)
  UDL        → V is LINEAR,  M is PARABOLIC  (smooth)
  Both captured by inserting extra x-points at every load position.
"""

import plotly.graph_objects as go
import numpy as np
import pandas as pd


# ── colour palette ─────────────────────────────────────────────────────────
_COLORS = [
    ('#1565C0', 'rgba(21,101,192,0.18)'),    # blue
    ('#C62828', 'rgba(198,40,40,0.18)'),     # red
    ('#2E7D32', 'rgba(46,125,50,0.18)'),     # green
    ('#6A1B9A', 'rgba(106,27,154,0.18)'),    # purple
    ('#E65100', 'rgba(230,81,0,0.18)'),      # orange
    ('#00695C', 'rgba(0,105,92,0.18)'),      # teal
    ('#AD1457', 'rgba(173,20,87,0.18)'),     # pink
    ('#F57F17', 'rgba(245,127,23,0.18)'),    # amber
]
def _lc(i): return _COLORS[i % len(_COLORS)][0]
def _fc(i): return _COLORS[i % len(_COLORS)][1]


# ── geometry helper ────────────────────────────────────────────────────────
def _geom(member, nodes_df):
    ni = int(member['Node_I']); nj = int(member['Node_J'])
    xi = float(nodes_df[nodes_df['Node']==ni]['X'].values[0])
    yi = float(nodes_df[nodes_df['Node']==ni]['Y'].values[0])
    xj = float(nodes_df[nodes_df['Node']==nj]['X'].values[0])
    yj = float(nodes_df[nodes_df['Node']==nj]['Y'].values[0])
    L  = float(np.sqrt((xj-xi)**2 + (yj-yi)**2))
    return xi, yi, xj, yj, L


# ═══════════════════════════════════════════════════════════════════════════
#  CORE FORMULA: x-array with critical points inserted
# ═══════════════════════════════════════════════════════════════════════════

def _x_array(L, member_loads, n=600):
    """
    Build x ∈ [0, L] with:
      • n uniform points (smooth parabolas for UDL)
      • For each point load at position a:
          insert x = a−ε, a, a+ε  so the kink/step is pixel-sharp
    """
    xs  = list(np.linspace(0.0, L, n))
    eps = L * 1e-5
    for _, ld in member_loads.iterrows():
        if ld['Type'] in ('Point Load', 'Moment'):
            a = float(ld['a'])
            if eps < a < L - eps:
                xs += [a - eps, a, a + eps]
    return np.array(sorted(set(np.clip(xs, 0.0, L))))


# ═══════════════════════════════════════════════════════════════════════════
#  CORE FORMULA: V(x) — Shear Force
# ═══════════════════════════════════════════════════════════════════════════

def _V(x, V_i, member_loads, L):
    """Vectorised shear V(x) along member. x is a numpy array."""
    V = np.full_like(x, float(V_i), dtype=float)
    for _, ld in member_loads.iterrows():
        t = ld['Type']
        if t == 'UDL':
            V -= float(ld['W1']) * x
        elif t == 'VDL':
            w1, w2 = float(ld['W1']), float(ld['W2'])
            V -= w1*x + (w2 - w1)*x**2 / (2.0*L)
        elif t == 'Point Load':
            V -= float(ld['P']) * (x > float(ld['a'])).astype(float)
    return V


# ═══════════════════════════════════════════════════════════════════════════
#  CORE FORMULA: M(x) — Bending Moment
# ═══════════════════════════════════════════════════════════════════════════

def _M(x, M_i, V_i, member_loads, L):
    """Vectorised moment M(x) along member. x is a numpy array."""
    M = float(M_i) + float(V_i)*x
    for _, ld in member_loads.iterrows():
        t = ld['Type']
        if t == 'UDL':
            M -= float(ld['W1']) * x**2 / 2.0
        elif t == 'VDL':
            w1, w2 = float(ld['W1']), float(ld['W2'])
            M -= w1*x**2/2.0 + (w2 - w1)*x**3/(6.0*L)
        elif t == 'Point Load':
            diff = x - float(ld['a'])
            M   -= float(ld['P']) * np.maximum(diff, 0.0)
        elif t == 'Moment':
            M[x >= float(ld['a'])] += float(ld['M'])
    return M


def _to_global(xi, xj, yi, yj, L, x_local):
    if abs(xj - xi) >= abs(yj - yi):
        return xi + x_local * (xj - xi) / L
    return np.full(len(x_local), (xi + xj) / 2.0)


# ═══════════════════════════════════════════════════════════════════════════
#  STRUCTURE DIAGRAM
# ═══════════════════════════════════════════════════════════════════════════

def plot_structure(nodes_df, members_df, supports_df, loads_df):
    try:
        fig   = go.Figure()
        xspan = max(float(nodes_df['X'].max() - nodes_df['X'].min()), 1.0)
        sym   = max(0.035*xspan, 0.12)
        arh   = max(0.055*xspan, 0.40)

        # members
        for _, mb in members_df.iterrows():
            xi, yi, xj, yj, L = _geom(mb, nodes_df)
            fig.add_trace(go.Scatter(x=[xi,xj], y=[yi,yj],
                mode='lines+text', line=dict(color='#1565C0', width=6),
                text=[f'  M{int(mb["Member"])}',''], textposition='top center',
                textfont=dict(size=12, color='#1565C0'),
                showlegend=False, hoverinfo='skip'))

        # nodes
        fig.add_trace(go.Scatter(x=nodes_df['X'], y=nodes_df['Y'],
            mode='markers+text',
            marker=dict(size=12, color='#C62828', symbol='circle',
                        line=dict(color='white', width=2)),
            text=[f'  N{int(n)}' for n in nodes_df['Node']],
            textposition='top right',
            textfont=dict(size=11, color='#C62828'), showlegend=False))

        # supports
        for _, sup in supports_df.iterrows():
            node  = int(sup['Node']); stype = sup['Type']
            sx    = float(nodes_df[nodes_df['Node']==node]['X'].values[0])
            sy    = float(nodes_df[nodes_df['Node']==node]['Y'].values[0])
            if stype in ('Fixed','Cantilever'):
                for k in range(6):
                    xo = sx - sym*1.2 + k*sym*0.48
                    fig.add_trace(go.Scatter(x=[xo, xo-sym*0.3], y=[sy, sy-sym*0.8],
                        mode='lines', line=dict(color='#424242',width=2),
                        showlegend=False, hoverinfo='skip'))
                fig.add_trace(go.Scatter(x=[sx-sym*1.2,sx+sym*1.2],y=[sy,sy],
                    mode='lines', line=dict(color='#424242',width=4),
                    showlegend=False, hoverinfo='skip'))
            elif stype == 'Pinned':
                fig.add_trace(go.Scatter(
                    x=[sx-sym,sx,sx+sym,sx-sym], y=[sy-sym*1.5,sy,sy-sym*1.5,sy-sym*1.5],
                    fill='toself', fillcolor='#43A047',
                    mode='lines', line=dict(color='#2E7D32',width=2),
                    showlegend=False, hoverinfo='skip'))
                fig.add_trace(go.Scatter(x=[sx-sym*1.2,sx+sym*1.2],y=[sy-sym*1.5,sy-sym*1.5],
                    mode='lines', line=dict(color='#2E7D32',width=3),
                    showlegend=False, hoverinfo='skip'))
            elif stype == 'Roller':
                fig.add_trace(go.Scatter(
                    x=[sx-sym,sx,sx+sym,sx-sym], y=[sy-sym,sy,sy-sym,sy-sym],
                    fill='toself', fillcolor='#FB8C00',
                    mode='lines', line=dict(color='#E65100',width=2),
                    showlegend=False, hoverinfo='skip'))
                for xo in [sx-sym*0.5, sx, sx+sym*0.5]:
                    fig.add_trace(go.Scatter(x=[xo], y=[sy-sym*1.35], mode='markers',
                        marker=dict(size=7, color='#E65100'),
                        showlegend=False, hoverinfo='skip'))
                fig.add_trace(go.Scatter(x=[sx-sym*1.2,sx+sym*1.2],y=[sy-sym*1.7,sy-sym*1.7],
                    mode='lines', line=dict(color='#E65100',width=3),
                    showlegend=False, hoverinfo='skip'))

        # loads
        for _, ld in loads_df.iterrows():
            mid    = int(ld['Member'])
            mb_row = members_df[members_df['Member']==mid].iloc[0]
            xi, yi, xj, yj, L = _geom(mb_row, nodes_df)
            if ld['Type'] == 'UDL':
                w = float(ld['W1']); n_a = max(5, min(15, int(L*2.5)))
                for k in range(n_a):
                    t = k/(n_a-1); xa=xi+t*(xj-xi); ya=yi+t*(yj-yi)
                    fig.add_annotation(x=xa,y=ya,ax=xa,ay=ya+arh,
                        xref='x',yref='y',axref='x',ayref='y',
                        showarrow=True,arrowhead=2,arrowsize=1.2,
                        arrowwidth=2,arrowcolor='#7B1FA2')
                fig.add_annotation(x=(xi+xj)/2,y=(yi+yj)/2+arh*1.8,
                    text=f'<b>w={w:.1f} kN/m</b>',showarrow=False,
                    font=dict(size=11,color='#7B1FA2'),
                    bgcolor='rgba(255,255,255,0.9)',
                    bordercolor='#7B1FA2',borderwidth=1.5)
            elif ld['Type'] == 'Point Load':
                P=float(ld['P']); a=float(ld['a']); t=a/L
                xa=xi+t*(xj-xi); ya=yi+t*(yj-yi)
                fig.add_annotation(x=xa,y=ya,ax=xa,ay=ya+arh*2,
                    xref='x',yref='y',axref='x',ayref='y',
                    text=f'<b>{P:.0f} kN</b>',
                    showarrow=True,arrowhead=3,arrowsize=1.5,
                    arrowwidth=3,arrowcolor='#C62828',
                    font=dict(size=11,color='#C62828'),
                    bgcolor='rgba(255,255,255,0.9)',
                    bordercolor='#C62828',borderwidth=1.5,yanchor='bottom')
            elif ld['Type'] == 'VDL':
                w1=float(ld['W1']); w2=float(ld['W2'])
                for k in range(8):
                    t=k/7; xa=xi+t*(xj-xi); ya=yi+t*(yj-yi)
                    wt=w1+(w2-w1)*t
                    al=arh*max(0.3,abs(wt)/max(abs(w1),abs(w2),1e-6))
                    fig.add_annotation(x=xa,y=ya,ax=xa,ay=ya+al,
                        xref='x',yref='y',axref='x',ayref='y',
                        showarrow=True,arrowhead=2,arrowsize=1.0,
                        arrowwidth=2,arrowcolor='#7B1FA2')
                fig.add_annotation(x=(xi+xj)/2,y=(yi+yj)/2+arh*1.8,
                    text=f'<b>VDL {w1:.1f}→{w2:.1f} kN/m</b>',showarrow=False,
                    font=dict(size=11,color='#7B1FA2'),
                    bgcolor='rgba(255,255,255,0.9)',
                    bordercolor='#7B1FA2',borderwidth=1.5)

        fig.update_layout(title=dict(text='Structural Configuration',font=dict(size=16)),
            xaxis_title='X (m)', yaxis_title='Y (m)',
            showlegend=False, hovermode='closest', height=480,
            yaxis=dict(scaleanchor='x',scaleratio=1),
            plot_bgcolor='#FAFAFA', paper_bgcolor='white',
            margin=dict(l=60,r=40,t=60,b=60))
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f'Structure plot error: {e}',showarrow=False,
            xref='paper',yref='paper',x=0.5,y=0.5)
        return fig


# ═══════════════════════════════════════════════════════════════════════════
#  SHEAR FORCE DIAGRAM
# ═══════════════════════════════════════════════════════════════════════════

def plot_sfd(members_df, nodes_df, member_forces_df, loads_df):
    try:
        fig = go.Figure()
        if len(members_df) == 0:
            return fig

        for idx, member in members_df.iterrows():
            mid = int(member['Member'])
            xi, yi, xj, yj, L = _geom(member, nodes_df)
            forces = member_forces_df[member_forces_df['Member']==mid].iloc[0]
            V_i    = float(forces['V_i'])
            ml     = loads_df[loads_df['Member']==mid]

            x  = _x_array(L, ml)
            V  = _V(x, V_i, ml, L)
            xg = _to_global(xi, xj, yi, yj, L, x)

            lc = _lc(idx); fc = _fc(idx)

            fig.add_trace(go.Scatter(
                x=xg, y=V, mode='lines',
                line=dict(width=2.5, color=lc),
                fill='tozeroy', fillcolor=fc,
                name=f'Member {mid}',
                customdata=np.stack([x, V], axis=1),
                hovertemplate=(
                    'x = %{customdata[0]:.3f} m<br>'
                    'V = %{customdata[1]:.3f} kN'
                    '<extra>M'+str(mid)+'</extra>')))

            # beam axis line
            fig.add_trace(go.Scatter(
                x=[xg[0], xg[-1]], y=[0,0], mode='lines',
                line=dict(color='black',width=1.5,dash='dot'),
                showlegend=False, hoverinfo='skip'))

            # annotate start, end, max, min
            seen = {}
            for ax_, av_ in [(xg[0],V[0]),(xg[-1],V[-1]),
                             (xg[np.argmax(V)],V.max()),
                             (xg[np.argmin(V)],V.min())]:
                k = round(float(ax_),2)
                if k not in seen and abs(av_) > 0.05:
                    seen[k] = True
                    fig.add_annotation(x=ax_,y=av_,
                        text=f'<b>{av_:.2f} kN</b>',
                        showarrow=True,arrowhead=2,ax=0,
                        ay=(-25 if av_>=0 else 25),
                        bgcolor='rgba(255,255,255,0.85)',
                        bordercolor=lc,borderwidth=1.5,
                        font=dict(size=9,color=lc))

        fig.add_hline(y=0, line_width=2, line_color='black')
        fig.update_layout(
            title=dict(text='Shear Force Diagram',font=dict(size=16)),
            xaxis_title='Position (m)',
            yaxis_title='Shear Force (kN)',
            hovermode='closest', height=420, showlegend=True,
            plot_bgcolor='#FAFAFA', paper_bgcolor='white',
            yaxis=dict(zeroline=False),
            legend=dict(orientation='h',yanchor='bottom',y=1.02,xanchor='right',x=1),
            margin=dict(l=70,r=40,t=70,b=60))
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f'SFD error: {e}',showarrow=False,
            xref='paper',yref='paper',x=0.5,y=0.5)
        return fig


# ═══════════════════════════════════════════════════════════════════════════
#  BENDING MOMENT DIAGRAM
#
#  CORRECTED PLOTTING CONVENTION (standard structural engineering):
#
#  Plot y = M(x) DIRECTLY on normal y-axis:
#    Sagging  M > 0  →  y > 0  →  ABOVE baseline  ✓  (tension at bottom)
#    Hogging  M < 0  →  y < 0  →  BELOW baseline  ✓  (tension at top)
#
#  This matches what every structural engineering textbook shows:
#  a simply-supported beam under gravity load has its sagging peak
#  appearing as a hump ABOVE the beam centreline.
#
#  Hover shows the true M value (sagging = positive, hogging = negative).
# ═══════════════════════════════════════════════════════════════════════════

def plot_bmd(members_df, nodes_df, member_forces_df, loads_df):
    try:
        fig = go.Figure()
        if len(members_df) == 0:
            return fig

        for idx, member in members_df.iterrows():
            mid = int(member['Member'])
            xi, yi, xj, yj, L = _geom(member, nodes_df)
            forces = member_forces_df[member_forces_df['Member']==mid].iloc[0]
            M_i    = float(forces['M_i'])
            V_i    = float(forces['V_i'])
            ml     = loads_df[loads_df['Member']==mid]

            x      = _x_array(L, ml, n=600)
            M_real = _M(x, M_i, V_i, ml, L)

            # ── KEY FIX ──────────────────────────────────────────────────
            # Plot M directly (no negation).
            #   Sagging M > 0  →  y > 0  →  above baseline  ✓
            #   Hogging M < 0  →  y < 0  →  below baseline  ✓
            # ─────────────────────────────────────────────────────────────
            M_plot = M_real   # was: -M_real  (WRONG — now corrected)

            xg  = _to_global(xi, xj, yi, yj, L, x)
            lc  = _lc(idx); fc = _fc(idx)

            fig.add_trace(go.Scatter(
                x=xg, y=M_plot, mode='lines',
                line=dict(width=2.5, color=lc),
                fill='tozeroy', fillcolor=fc,
                name=f'Member {mid}',
                customdata=np.stack([x, M_real], axis=1),
                hovertemplate=(
                    'x = %{customdata[0]:.3f} m<br>'
                    'M = %{customdata[1]:.3f} kNm'
                    '<extra>M'+str(mid)+'</extra>')))

            # beam axis
            fig.add_trace(go.Scatter(
                x=[xg[0],xg[-1]], y=[0,0], mode='lines',
                line=dict(color='black',width=2),
                showlegend=False, hoverinfo='skip'))

            # ── Annotations ────────────────────────────────────────────
            ann = {}

            # Peak sagging (largest positive M) — appears above baseline
            if M_real.max() > 0.1:
                i = int(np.argmax(M_real))
                ann[round(float(xg[i]),3)] = (xg[i], M_plot[i], M_real[i])

            # Peak hogging (most negative M) — appears below baseline
            if M_real.min() < -0.1:
                i = int(np.argmin(M_real))
                ann[round(float(xg[i]),3)] = (xg[i], M_plot[i], M_real[i])

            # Moment at each point-load position (kink location)
            for _, ld in ml.iterrows():
                if ld['Type'] == 'Point Load':
                    a = float(ld['a'])
                    i = int(np.argmin(np.abs(x - a)))
                    k = round(float(xg[i]),3)
                    if k not in ann and abs(M_real[i]) > 0.1:
                        ann[k] = (xg[i], M_plot[i], M_real[i])

            # End values
            for ie in [0, -1]:
                if abs(M_real[ie]) > 0.1:
                    k = round(float(xg[ie]),3)
                    if k not in ann:
                        ann[k] = (xg[ie], M_plot[ie], M_real[ie])

            for _, (ax_, ay_, am_) in ann.items():
                # Label above the point if sagging (positive), below if hogging (negative)
                label_offset = 28 if ay_ >= 0 else -28
                fig.add_annotation(x=ax_, y=ay_,
                    text=f'<b>{am_:.2f} kNm</b>',
                    showarrow=True, arrowhead=2, arrowsize=0.8, arrowwidth=1.5,
                    ax=0, ay=label_offset,
                    bgcolor='rgba(255,255,255,0.90)',
                    bordercolor=lc, borderwidth=1.5,
                    font=dict(size=9, color=lc))

        # global zero line
        fig.add_hline(y=0, line_width=2.5, line_color='black')

        # direction labels — corrected to match new convention
        fig.add_annotation(xref='paper', yref='paper', x=0.01, y=0.99,
            text='↑ Sagging (+ve)',
            showarrow=False, xanchor='left', yanchor='top',
            font=dict(size=10, color='#C62828'),
            bgcolor='rgba(255,255,255,0.8)')
        fig.add_annotation(xref='paper', yref='paper', x=0.01, y=0.01,
            text='↓ Hogging (−ve)',
            showarrow=False, xanchor='left', yanchor='bottom',
            font=dict(size=10, color='#1565C0'),
            bgcolor='rgba(255,255,255,0.8)')

        fig.update_layout(
            title=dict(
                text=('Bending Moment Diagram  '
                      '<span style="color:#C62828">↑ Sagging (+ve)</span>  '
                      '|  '
                      '<span style="color:#1565C0">↓ Hogging (−ve)</span>'),
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
        fig.add_annotation(text=f'BMD error: {e}', showarrow=False,
            xref='paper', yref='paper', x=0.5, y=0.5)
        return fig


# ═══════════════════════════════════════════════════════════════════════════
#  DEFLECTION DIAGRAM
# ═══════════════════════════════════════════════════════════════════════════

def plot_deflection(members_df, nodes_df, displacements):
    try:
        fig     = go.Figure()
        max_d   = max(float(np.max(np.abs(displacements))), 1e-12)
        xspan   = max(float(nodes_df['X'].max()-nodes_df['X'].min()), 1.0)
        scale   = max(10, min(1000, round((xspan*0.05)/max_d/10)*10))

        orig_shown = defl_shown = False
        for _, mb in members_df.iterrows():
            ni=int(mb['Node_I']); nj=int(mb['Node_J'])
            xi,yi,xj,yj,_ = _geom(mb, nodes_df)
            fig.add_trace(go.Scatter(x=[xi,xj],y=[yi,yj],mode='lines',
                line=dict(color='#9E9E9E',width=2,dash='dash'),
                name='Original' if not orig_shown else '',
                showlegend=not orig_shown))
            orig_shown = True

        for _, mb in members_df.iterrows():
            ni=int(mb['Node_I']); nj=int(mb['Node_J'])
            xi,yi,xj,yj,_ = _geom(mb, nodes_df)
            dxi=displacements[(ni-1)*3  ]*scale; dyi=displacements[(ni-1)*3+1]*scale
            dxj=displacements[(nj-1)*3  ]*scale; dyj=displacements[(nj-1)*3+1]*scale
            fig.add_trace(go.Scatter(
                x=[xi+dxi,xj+dxj],y=[yi+dyi,yj+dyj],mode='lines',
                line=dict(color='#C62828',width=3),
                name='Deflected' if not defl_shown else '',
                showlegend=not defl_shown))
            defl_shown = True

        fig.update_layout(
            title=dict(text=f'Deflected Shape  (Scale: {scale}×)',font=dict(size=16)),
            xaxis_title='X (m)',yaxis_title='Y (m)',
            hovermode='closest',height=420,
            plot_bgcolor='#FAFAFA',paper_bgcolor='white',
            yaxis=dict(scaleanchor='x',scaleratio=1),
            legend=dict(orientation='h',yanchor='bottom',y=1.02,xanchor='right',x=1),
            margin=dict(l=70,r=40,t=70,b=60))
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f'Deflection error: {e}',showarrow=False,
            xref='paper',yref='paper',x=0.5,y=0.5)
        return fig
