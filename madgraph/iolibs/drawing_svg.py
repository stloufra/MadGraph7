################################################################################
#
# Copyright (c) 2009 The MadGraph5_aMC@NLO Development team and Contributors
#
# This file is a part of the MadGraph5_aMC@NLO project, an application which
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph5_aMC@NLO license which should accompany this
# distribution.
#
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
#
################################################################################

"""Classes for writing SVG files containing Feynman diagrams.

DrawDiagramSVG  - writes one diagram to a single SVG file.
MultiSVGDiagramDrawer - writes a list of diagrams to a folder of SVG files,
                        one file per diagram.

The line-drawing routines re-implement in Python the PostScript macros found in
drawing_eps_header.inc (Fgluon, Fphoton, Ffermion, Fhiggs, Fghost, …) so that
the resulting SVG output looks as close as possible to the EPS output.

Coordinate convention
---------------------
All positions from the drawing engine are in [0, 1]^2 with y increasing
upward.  rescale() maps them to SVG canvas pixels where y increases downward.
All SVG helper functions work in SVG canvas pixels.
"""

from __future__ import division
from __future__ import absolute_import

import os
import math

import madgraph.core.drawing as draw
import madgraph.core.base_objects as base_objects
import madgraph.loop.loop_base_objects as loop_objects
import logging

logger = logging.getLogger('madgraph.drawing_svg')

# ---------------------------------------------------------------------------
# SVG drawing primitives
# These replicate the PostScript macros in drawing_eps_header.inc.
# All coordinates are in SVG canvas pixels (y increases downward).
# ---------------------------------------------------------------------------

_Fr = 5.0        # base size parameter (scaled up vs EPS Fr=2.5 for screen visibility)
_Fnopoints = 10  # points per half-period for gluon / full period for photon


def _basis(x1, y1, x2, y2):
    """Return (dist, xl, yl, xt, yt) – longitudinal and transverse unit
    vectors scaled by _Fr.  SVG y-down convention."""
    dx, dy = x2 - x1, y2 - y1
    dist = math.sqrt(dx * dx + dy * dy)
    if dist < 1e-10:
        return 1e-10, 0.0, 0.0, 0.0, 0.0
    xl = dx / dist * _Fr
    yl = dy / dist * _Fr
    # perpendicular (90° CCW in math = 90° CW in SVG y-down)
    xt = -dy / dist * _Fr
    yt =  dx / dist * _Fr
    return dist, xl, yl, xt, yt


def _arrow_svg(mx, my, xl, yl, xt, yt):
    """Return SVG markup for a filled arrowhead at (mx, my) pointing toward
    (xl, yl) direction.  Mirrors the Farrow PostScript macro."""
    p0 = (mx,           my)
    p1 = (mx + xt - xl, my + yt - yl)
    p2 = (mx + xl,      my + yl)
    p3 = (mx - xl - xt, my - yl - yt)
    pts = ' '.join(f'{p[0]:.2f},{p[1]:.2f}' for p in [p0, p1, p2, p3])
    return f'<polygon points="{pts}" fill="black" stroke="none"/>\n'


def _polyline_svg(pts, stroke='black', width=1.0, dash='', extra=''):
    """Return an SVG <path> element for a sequence of (x, y) points."""
    if not pts:
        return ''
    d = 'M ' + ' L '.join(f'{x:.2f} {y:.2f}' for x, y in pts)
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ''
    return (f'<path d="{d}" stroke="{stroke}" stroke-width="{width:.2f}"'
            f' fill="none"{dash_attr}{extra}/>\n')


# --- straight fermion line --------------------------------------------------

def _svg_fermion(x1, y1, x2, y2):
    """Straight line with arrowhead at mid-point."""
    dist, xl, yl, xt, yt = _basis(x1, y1, x2, y2)
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    out  = f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}"'
    out += ' stroke="black" stroke-width="1"/>\n'
    out += _arrow_svg(mx, my, xl, yl, xt, yt)
    return out


def _svg_scalar(x1, y1, x2, y2):
    """Plain straight line, no arrow (for 'scalar' particles)."""
    return (f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}"'
            ' stroke="black" stroke-width="1"/>\n')


# --- dashed line (Higgs) ---------------------------------------------------

def _svg_higgs(x1, y1, x2, y2):
    """Dashed line.  Dash length scales with line length (as in Fhiggs)."""
    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    n_dashes = max(1, round(dist / (2 * _Fr)))
    dash = dist / (2 * n_dashes)
    return (f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}"'
            f' stroke="black" stroke-width="1"'
            f' stroke-dasharray="{dash:.2f},{dash:.2f}"/>\n')


# --- dotted line (ghost) ---------------------------------------------------

def _svg_ghost(x1, y1, x2, y2):
    """Row of small filled circles plus arrowhead at mid-point (Fghost)."""
    dist, xl, yl, xt, yt = _basis(x1, y1, x2, y2)
    n = max(2, round(dist / _Fr))
    r_dot = _Fr / 5.0
    out = ''
    for i in range(n + 1):
        t = i / n
        cx = x1 + (x2 - x1) * t
        cy = y1 + (y2 - y1) * t
        out += f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r_dot:.2f}" fill="black"/>\n'
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    out += _arrow_svg(mx, my, xl, yl, xt, yt)
    return out


# --- sinusoidal wave (photon) ----------------------------------------------

def _svg_photon(x1, y1, x2, y2):
    """Sinusoidal wave (Fphoton)."""
    dist, xl, yl, xt, yt = _basis(x1, y1, x2, y2)
    Fn = max(1, round(dist / (2 * _Fr)))
    N = _Fnopoints * Fn   # Fn full waves, Fnopoints steps per wave
    pts = []
    for i in range(N * 2 + 1):     # iterate over 2*Fn half-periods
        t = i / (N * 2)
        px = x1 + (x2 - x1) * t
        py = y1 + (y2 - y1) * t
        wave = math.sin(i * math.pi / _Fnopoints)
        px += xt * wave / 2
        py += yt * wave / 2
        pts.append((px, py))
    return _polyline_svg(pts)


def _svg_photon_half(x1, y1, x2, y2):
    """Half-frequency photon (Fphotonr – used for wino/neutralino)."""
    dist, xl, yl, xt, yt = _basis(x1, y1, x2, y2)
    Fn = max(1, round(dist / _Fr))  # twice as many half-periods
    N = _Fnopoints * Fn
    pts = []
    for i in range(N + 1):
        t = i / N
        px = x1 + (x2 - x1) * t
        py = y1 + (y2 - y1) * t
        wave = math.sin(i * math.pi / _Fnopoints)
        px += xt * wave
        py += yt * wave
        pts.append((px, py))
    return _polyline_svg(pts)


# --- curly line (gluon) ----------------------------------------------------

def _svg_gluon(x1, y1, x2, y2):
    """Loops above the connecting line (Fgluon).
    Each loop is one half-period: (1-cos) perpendicular + sin longitudinal."""
    dist, xl, yl, xt, yt = _basis(x1, y1, x2, y2)
    # Fgluon type=-2: Fn = round(dist/Fr) (half-periods for gluon vs photon)
    Fn = max(2, 2 * round(dist / (2 * _Fr)))   # even number of half-periods
    N = _Fnopoints * Fn
    pts = []
    for i in range(N + 1):
        t = i / N
        px = x1 + (x2 - x1) * t
        py = y1 + (y2 - y1) * t
        angle = i * math.pi / _Fnopoints   # π per half-period
        px += xt * (1 - math.cos(angle))
        py += yt * (1 - math.cos(angle))
        px += xl * math.sin(angle)
        py += yl * math.sin(angle)
        pts.append((px, py))
    return _polyline_svg(pts)


# --- arc utilities (for curved/circled loop lines) -------------------------

def _arc_params(x1, y1, x2, y2, e):
    """Compute arc center, radius, and start/end angles (SVG coords, y-down).
    e is the curvature/eccentricity parameter from the EPS macros.
    Negating e compensates for the y-axis flip vs EPS."""
    lam = (1 - e * e) / (2 * e)
    xc = (x1 + x2 + lam * (y2 - y1)) / 2
    yc = (y1 + y2 + lam * (x1 - x2)) / 2
    r = math.sqrt((x1 - xc) ** 2 + (y1 - yc) ** 2)
    th1 = math.atan2(y1 - yc, x1 - xc)
    th2 = math.atan2(y2 - yc, x2 - xc)
    return xc, yc, r, th1, th2


def _arc_pts_list(x1, y1, x2, y2, e, n=60):
    """Return list of (x,y) along the arc from P1 to P2 with eccentricity e."""
    xc, yc, r, th1, th2 = _arc_params(x1, y1, x2, y2, e)
    # Choose sweep direction mirroring EPS: e>0 → arcn (decreasing angle)
    if e > 0:
        if th2 > th1:
            th2 -= 2 * math.pi
    else:
        if th2 < th1:
            th2 += 2 * math.pi
    # For tadpoles (tiny chord, large |e|): angular span may be tiny but
    # should actually be ~2π.  Force the long-way-round in that case.
    if abs(th2 - th1) < 0.1:
        if e > 0:
            th2 -= 2 * math.pi
        else:
            th2 += 2 * math.pi
    pts = []
    for i in range(n + 1):
        th = th1 + (th2 - th1) * i / n
        pts.append((xc + r * math.cos(th), yc + r * math.sin(th)))
    return pts, xc, yc, r, th1, th2


# --- curved fermion (Ffermionl) --------------------------------------------

def _svg_fermion_arc(x1, y1, x2, y2, e):
    """Arc with arrowhead at mid-angle."""
    pts, xc, yc, r, th1, th2 = _arc_pts_list(x1, y1, x2, y2, e)
    thc = (th1 + th2) / 2
    mx = xc + r * math.cos(thc)
    my = yc + r * math.sin(thc)
    # Tangent at midpoint (direction of arrow)
    # tangent perpendicular to radius, direction depends on sweep
    dth = th2 - th1
    tx = -math.sin(thc) * (1 if dth >= 0 else -1)
    ty =  math.cos(thc) * (1 if dth >= 0 else -1)
    xl, yl = tx * _Fr, ty * _Fr
    xt, yt = -ty * _Fr, tx * _Fr
    out = _polyline_svg(pts)
    out += _arrow_svg(mx, my, xl, yl, xt, yt)
    return out


# --- curved dashed (Fhiggsl) -----------------------------------------------

def _svg_higgs_arc(x1, y1, x2, y2, e):
    """Dashed arc."""
    pts, xc, yc, r, th1, th2 = _arc_pts_list(x1, y1, x2, y2, e)
    arc_len = r * abs(th2 - th1)
    n_dashes = max(1, round(arc_len / (2 * _Fr)))
    dash = arc_len / (2 * n_dashes)
    return _polyline_svg(pts, dash=f'{dash:.2f},{dash:.2f}')


# --- curved ghost (Fghostl) ------------------------------------------------

def _svg_ghost_arc(x1, y1, x2, y2, e):
    """Dotted arc with arrowhead."""
    pts, xc, yc, r, th1, th2 = _arc_pts_list(x1, y1, x2, y2, e)
    arc_len = r * abs(th2 - th1)
    n_dots = max(2, round(arc_len / _Fr))
    r_dot = _Fr / 5.0
    out = ''
    for i in range(n_dots + 1):
        th = th1 + (th2 - th1) * i / n_dots
        cx = xc + r * math.cos(th)
        cy = yc + r * math.sin(th)
        out += f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r_dot:.2f}" fill="black"/>\n'
    # arrowhead at midpoint
    thc = (th1 + th2) / 2
    mx = xc + r * math.cos(thc)
    my = yc + r * math.sin(thc)
    dth = th2 - th1
    tx = -math.sin(thc) * (1 if dth >= 0 else -1)
    ty =  math.cos(thc) * (1 if dth >= 0 else -1)
    xl, yl = tx * _Fr, ty * _Fr
    xt, yt = -ty * _Fr, tx * _Fr
    out += _arrow_svg(mx, my, xl, yl, xt, yt)
    return out


# --- curved photon (Fphotonl) ----------------------------------------------

def _svg_photon_arc(x1, y1, x2, y2, e):
    """Sinusoidal wave along an arc."""
    _, xc, yc, r, th1, th2 = _arc_pts_list(x1, y1, x2, y2, e, n=1)
    arc_len = r * abs(th2 - th1)
    Fn = max(1, round(arc_len / (2 * _Fr)))
    N = _Fnopoints * Fn
    pts = []
    dth = th2 - th1
    for i in range(2 * N + 1):
        t = i / (2 * N)
        th = th1 + dth * t
        wave = math.sin(i * math.pi / _Fnopoints) / 2
        # radial displacement
        rr = r + wave * _Fr
        pts.append((xc + rr * math.cos(th), yc + rr * math.sin(th)))
    return _polyline_svg(pts)


# --- curved gluon (Fgluonl) ------------------------------------------------

def _svg_gluon_arc(x1, y1, x2, y2, e):
    """Curly loops along an arc."""
    _, xc, yc, r, th1, th2 = _arc_pts_list(x1, y1, x2, y2, e, n=1)
    arc_len = r * abs(th2 - th1)
    Fn = max(2, 2 * round(arc_len / (2 * _Fr)))
    N = _Fnopoints * Fn
    pts = []
    dth = th2 - th1
    for i in range(N + 1):
        t = i / N
        th = th1 + dth * t
        angle = i * math.pi / _Fnopoints
        loop_r  = (1 - math.cos(angle)) * _Fr   # radial (outward)
        loop_th = math.sin(angle) * _Fr / r      # tangential (in angle units)
        rr = r + loop_r
        pts.append((xc + rr * math.cos(th + loop_th),
                    yc + rr * math.sin(th + loop_th)))
    return _polyline_svg(pts)


# --- blob (Fblob) ----------------------------------------------------------

def _svg_blob(cx, cy, size=1.5):
    """Filled circle for non-QED/QCD vertex blobs."""
    r = size * _Fr
    return (f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}"'
            ' fill="gray" stroke="black" stroke-width="0.5"/>\n')


# ===========================================================================
# SvgDiagramDrawer  –  one diagram → one SVG file
# ===========================================================================

class SvgDiagramDrawer(draw.DiagramDrawer):
    """Write a single Feynman diagram as an SVG file.

    API mirrors EpsDiagramDrawer so callers can swap one for the other.
    """

    # SVG canvas size in pixels (extra height for diagram label below drawing area)
    canvas_width  = 360
    canvas_height = 400

    # Drawing area (in canvas pixels) – maps [0,1]^2 onto this rectangle.
    # y is flipped because SVG y-axis points down.
    draw_x_min = 30.0
    draw_x_max = 330.0
    draw_y_min = 30.0    # corresponds to diagram y=1
    draw_y_max = 330.0   # corresponds to diagram y=0

    blob_size = 1.5
    font_size = 11

    def rescale(self, x, y):
        """Map [0,1]^2 → SVG canvas pixels (y flipped)."""
        sx = self.draw_x_min + (self.draw_x_max - self.draw_x_min) * x
        sy = self.draw_y_max - (self.draw_y_max - self.draw_y_min) * y
        return sx, sy

    # ------------------------------------------------------------------
    # Framework hooks
    # ------------------------------------------------------------------

    def initialize(self):
        """Open file and write SVG header."""
        super(SvgDiagramDrawer, self).initialize()
        w, h = self.canvas_width, self.canvas_height
        header = (
            f'<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<svg xmlns="http://www.w3.org/2000/svg"'
            f' viewBox="0 0 {w} {h}" width="{w}" height="{h}">\n'
            f'<g font-family="serif" font-size="{self.font_size}"'
            f' font-style="italic">\n'
        )
        self.file.writelines(header)

    def conclude(self):
        """Close SVG and write file."""
        self.text += '</g>\n</svg>\n'
        super(SvgDiagramDrawer, self).conclude()

    # ------------------------------------------------------------------
    # Line-drawing helpers
    # ------------------------------------------------------------------

    def _svg(self, x1, y1, x2, y2, fn, *args):
        """Rescale endpoints and call fn(x1_svg, y1_svg, x2_svg, y2_svg, ...)."""
        sx1, sy1 = self.rescale(x1, y1)
        sx2, sy2 = self.rescale(x2, y2)
        self.text += fn(sx1, sy1, sx2, sy2, *args)

    def _svg_arc(self, x1, y1, x2, y2, fn, e):
        sx1, sy1 = self.rescale(x1, y1)
        sx2, sy2 = self.rescale(x2, y2)
        # y-flip reverses the curvature sign
        self.text += fn(sx1, sy1, sx2, sy2, -e)

    # ------------------------------------------------------------------
    # Vertex blob
    # ------------------------------------------------------------------

    def draw_vertex(self, vertex, bypass=None):
        """Draw blob for non-QED/QCD vertices."""
        if bypass is None:
            bypass = ['QED', 'QCD']
        if not self.model:
            return
        interaction = self.model.get_interaction(vertex.id)
        if interaction:
            order = interaction.get('orders')
            order = [k for k, v in order.items() if v and k not in bypass]
            if order:
                sx, sy = self.rescale(vertex.pos_x, vertex.pos_y)
                self.text += _svg_blob(sx, sy, self.blob_size)

    # ------------------------------------------------------------------
    # Straight lines
    # ------------------------------------------------------------------

    def draw_straight(self, line):
        self._svg(line.begin.pos_x, line.begin.pos_y,
                  line.end.pos_x, line.end.pos_y, _svg_fermion)

    def draw_dashed(self, line):
        self._svg(line.begin.pos_x, line.begin.pos_y,
                  line.end.pos_x, line.end.pos_y, _svg_higgs)

    def draw_dotted(self, line):
        self._svg(line.begin.pos_x, line.begin.pos_y,
                  line.end.pos_x, line.end.pos_y, _svg_ghost)

    def draw_wavy(self, line, opt=0, type=''):
        fn = _svg_photon_half if type == 'r' else _svg_photon
        self._svg(line.begin.pos_x, line.begin.pos_y,
                  line.end.pos_x, line.end.pos_y, fn)

    def draw_curly(self, line, type=''):
        # EPS forces left-to-right ordering for the gluon; we mirror that.
        if (line.begin.pos_x < line.end.pos_x) or \
                (line.begin.pos_x == line.end.pos_x and
                 line.begin.pos_y > line.end.pos_y):
            self._svg(line.begin.pos_x, line.begin.pos_y,
                      line.end.pos_x, line.end.pos_y, _svg_gluon)
        else:
            self._svg(line.end.pos_x, line.end.pos_y,
                      line.begin.pos_x, line.begin.pos_y, _svg_gluon)

    def draw_scurly(self, line):
        """Gluino: gluon + straight."""
        self.draw_curly(line)
        self.draw_straight(line)

    def draw_swavy(self, line):
        """Neutralino/wino: photon + straight."""
        self.draw_wavy(line, type='r')
        self.draw_straight(line)

    def draw_double(self, line, type='r'):
        """Double-line (e.g. gravitino)."""
        length = math.sqrt((line.end.pos_y - line.begin.pos_y) ** 2 +
                            (line.end.pos_x - line.begin.pos_x) ** 2)
        if length < 1e-10:
            return
        c1 = (line.end.pos_x - line.begin.pos_x) / length
        c2 = (line.end.pos_y - line.begin.pos_y) / length
        gap = 0.013
        # Two offset lines (EPS uses two Fphoton calls with 0 opt)
        self._svg(line.begin.pos_x, line.begin.pos_y,
                  line.end.pos_x - gap * c1, line.end.pos_y - gap * c2,
                  _svg_photon)
        self._svg(line.begin.pos_x + gap * c1, line.begin.pos_y + gap * c2,
                  line.end.pos_x, line.end.pos_y, _svg_photon)

    # ------------------------------------------------------------------
    # Curved lines (for loop diagrams, two particles in the loop)
    # ------------------------------------------------------------------

    def _curvature(self, line, cercle):
        c = 1 if cercle else 0.4
        if (line.begin.pos_x, line.begin.pos_y) == self.curved_part_start:
            c = -c
        return c

    def draw_curved_straight(self, line, cercle):
        e = self._curvature(line, cercle)
        self._svg_arc(line.begin.pos_x, line.begin.pos_y,
                      line.end.pos_x, line.end.pos_y, _svg_fermion_arc, e)

    def draw_curved_dashed(self, line, cercle):
        e = self._curvature(line, cercle)
        self._svg_arc(line.begin.pos_x, line.begin.pos_y,
                      line.end.pos_x, line.end.pos_y, _svg_higgs_arc, e)

    def draw_curved_dotted(self, line, cercle):
        e = self._curvature(line, cercle)
        self._svg_arc(line.begin.pos_x, line.begin.pos_y,
                      line.end.pos_x, line.end.pos_y, _svg_ghost_arc, e)

    def draw_curved_wavy(self, line, cercle, opt=0, type=''):
        e = self._curvature(line, cercle)
        self._svg_arc(line.begin.pos_x, line.begin.pos_y,
                      line.end.pos_x, line.end.pos_y, _svg_photon_arc, e)

    def draw_curved_curly(self, line, cercle, type=''):
        dist = math.sqrt((line.begin.pos_x - line.end.pos_x) ** 2 +
                         (line.begin.pos_y - line.end.pos_y) ** 2)
        e = 1 if (cercle and dist <= 0.3) else 0.4
        if (line.begin.pos_x, line.begin.pos_y) == self.curved_part_start:
            e = -e
        # EPS calls with reversed endpoints and negated curvature
        self._svg_arc(line.end.pos_x, line.end.pos_y,
                      line.begin.pos_x, line.begin.pos_y, _svg_gluon_arc, e)

    def draw_curved_scurly(self, line, cercle, type=''):
        self.draw_curved_curly(line, cercle, type)
        self.draw_curved_straight(line, cercle)

    # ------------------------------------------------------------------
    # Circled lines (tadpole / single-particle loop)
    # ------------------------------------------------------------------

    def _tadpole_direction(self, line):
        """Return unit direction vector from the vertex (for tadpoles)."""
        direction = None
        for l in line.begin.lines:
            ndx = l.end.pos_x - l.begin.pos_x
            ndy = l.end.pos_y - l.begin.pos_y
            if ndx == 0 and ndy == 0:
                continue
            norm = math.sqrt(ndx ** 2 + ndy ** 2)
            nd = (ndx / norm, ndy / norm)
            if direction is None:
                direction = nd
        return direction or (1.0, 0.0)

    def _circled_svg(self, line, cercle, arc_fn):
        """Draw a circle (tadpole) or loop line."""
        base_e = 5 if cercle else 4
        is_tadpole = (line.begin.pos_x == line.end.pos_x and
                      line.begin.pos_y == line.end.pos_y)
        if is_tadpole:
            # EPS multiplies curvature by 7 for tadpole to force a full circle.
            e = base_e * 7
            d = self._tadpole_direction(line)
            x2 = line.end.pos_x + 0.01 * d[0]
            y2 = line.end.pos_y + 0.01 * d[1]
        else:
            e = base_e
            x2 = line.end.pos_x + 0.01
            y2 = line.end.pos_y + 0.01
        self._svg_arc(line.begin.pos_x, line.begin.pos_y, x2, y2, arc_fn, e)

    def draw_circled_straight(self, line, cercle):
        self._circled_svg(line, cercle, _svg_fermion_arc)

    def draw_circled_dashed(self, line, cercle):
        self._circled_svg(line, cercle, _svg_higgs_arc)

    def draw_circled_dotted(self, line, cercle):
        self._circled_svg(line, cercle, _svg_ghost_arc)

    def draw_circled_wavy(self, line, cercle, opt=0, type=''):
        self._circled_svg(line, cercle, _svg_photon_arc)

    def draw_circled_curly(self, line, cercle, type=''):
        self._circled_svg(line, cercle, _svg_gluon_arc)

    # ------------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------------

    def put_diagram_number(self, number=0):
        """Write 'diagram N' label below the diagram."""
        x, y = self.rescale(0.5, -0.12)
        if hasattr(self, 'diagram_type') and self.diagram_type:
            label = f'{self.diagram_type} diagram {number + 1}'
        else:
            label = f'diagram {number + 1}'
        self.text += (f'<text x="{x:.1f}" y="{y:.1f}"'
                      f' text-anchor="middle" font-style="normal"'
                      f' font-size="{self.font_size - 1}">{label}</text>\n')

        # coupling orders
        if hasattr(self, 'diagram') and self.diagram and \
                hasattr(self.diagram, 'diagram'):
            try:
                orders = self.diagram.diagram.get('orders')
            except Exception:
                orders = {}
            mystr = ', '.join(f'{k}={v}' for k, v in sorted(orders.items())
                              if k != 'WEIGHTED')
            if mystr:
                ox, oy = self.rescale(0.5, -0.20)
                self.text += (f'<text x="{ox:.1f}" y="{oy:.1f}"'
                              f' text-anchor="middle" font-style="normal"'
                              f' font-size="{self.font_size - 2}">'
                              f'({mystr})</text>\n')

    def associate_number(self, line, number):
        """Write leg number near the external vertex."""
        if line.begin.is_external():
            vertex = line.begin
        else:
            vertex = line.end
        x, y = vertex.pos_x, vertex.pos_y
        if x == 0:
            x = -0.04
        else:
            x += 0.04
            y = line._has_ordinate(x)
        sx, sy = self.rescale(x, y)
        self.text += (f'<text x="{sx:.1f}" y="{sy:.1f}"'
                      f' font-style="normal">{number}</text>\n')

    def associate_name(self, line, name, loop=False, reverse=False):
        """Write particle name near the centre of the line."""
        is_tadpole = (line.begin.pos_x == line.end.pos_x and
                      line.begin.pos_y == line.end.pos_y)
        if is_tadpole:
            direction = self._tadpole_direction(line)
            orth = (-direction[1], direction[0])
            scale = 0.08
            dx, dy = scale * orth[0], scale * orth[1]
        else:
            x1, y1 = line.begin.pos_x, line.begin.pos_y
            x2, y2 = line.end.pos_x, line.end.pos_y
            if abs(x1 - x2) < 1e-3:
                dx, dy = 0.015, -0.01
            elif abs(y1 - y2) < 1e-3:
                dx, dy = -0.01, 0.025
            elif (x1 < x2) == (y1 < y2):
                dx = -0.03 * len(name)
                dy =  0.02 * len(name)
            else:
                dx, dy = 0.01, 0.02

        if loop:
            dx *= 1.5
            x1, y1 = line.begin.pos_x, line.begin.pos_y
            x2, y2 = line.end.pos_x, line.end.pos_y
            if x1 == x2:
                if y1 < y2:
                    dx, dy = -dx, -dy
            elif y1 == y2:
                if x1 > x2:
                    dx, dy = -dx, -dy
            elif x1 < x2:
                dx, dy = -dx, -dy
        if reverse:
            dx, dy = -dx, -dy

        x1, y1 = line.begin.pos_x, line.begin.pos_y
        x2, y2 = line.end.pos_x, line.end.pos_y
        x_pos = (x1 + x2) / 2 + dx
        y_pos = (y1 + y2) / 2 + dy
        sx, sy = self.rescale(x_pos, y_pos)
        self.text += f'<text x="{sx:.1f}" y="{sy:.1f}">{name}</text>\n'


# ===========================================================================
# MultiSVGDiagramDrawer  –  list of diagrams → folder of SVG files
# ===========================================================================

class MultiSVGDiagramDrawer(SvgDiagramDrawer):
    """Write a list of diagrams as individual SVG files inside a folder.

    The output *filename* is treated as a directory path.  Each diagram is
    written to ``<folder>/diagram_NNN.svg``.
    """

    def __init__(self, diagramlist=None, filename='diagrams', model=None,
                 amplitude=None, legend='', diagram_type=''):
        super(MultiSVGDiagramDrawer, self).__init__(
            None, filename, model, amplitude)
        self.legend = legend
        self.diagram_type = diagram_type
        self._block_nb = 0

        diagramlist = [d for d in diagramlist
                       if not (isinstance(d, loop_objects.LoopUVCTDiagram) or
                               (isinstance(d, loop_objects.LoopDiagram) and
                                d.get('type') < 0))]
        self.diagramlist = base_objects.DiagramList(diagramlist) \
                           if diagramlist else None

    # ------------------------------------------------------------------

    def initialize(self):
        """Create output directory (do not call super – no file to open)."""
        os.makedirs(self.filename, exist_ok=True)

    def conclude(self):
        """Nothing to do – each diagram file is already closed."""
        pass

    # ------------------------------------------------------------------

    def draw(self, diagramlist='', opt=None):
        """Write each diagram in diagramlist to its own SVG file."""
        if diagramlist == '':
            diagramlist = self.diagramlist
        if diagramlist is None:
            return

        self.initialize()

        for diagram in diagramlist:
            diagram = self.convert_diagram(diagram, self.model, self.amplitude,
                                           opt)
            if diagram is None:
                continue
            self._draw_one(diagram)

    def _draw_one(self, diagram):
        """Write a single diagram to its own SVG file."""
        n = self._block_nb
        svg_path = os.path.join(self.filename, f'diagram_{n + 1:04d}.svg')

        # Temporarily redirect output to the individual file
        old_filename = self.filename
        old_file     = self.file
        old_text     = self.text

        self.filename = svg_path
        self.text = ''
        self.file = True          # signal to base class that we write a file

        # Write SVG header
        super(MultiSVGDiagramDrawer, self).initialize()
        # Draw the diagram
        self.draw_diagram(diagram, n)
        # Write footer & close file
        super(MultiSVGDiagramDrawer, self).conclude()

        # Restore state
        self.filename = old_filename
        self.file     = old_file
        self.text     = old_text
        self._block_nb += 1
