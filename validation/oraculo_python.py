"""
Oráculo Python de vórtice-lineal para V-Genfoil (sin numpy).

Replica bloque a bloque la estructura de panel_method.cpp, parametrizada para
que el test principal pueda comparar panel-a-panel contra el binario C++.

Estructura:
    PARSEC -> puntos (herradura) -> build_linear_system(A, b) -> gamma
    -> v_tangent(panel i) -> Cp(i) -> Cl

Convención geométrica: idéntica al C++.
    - Puntos en orden herradura: TE_sup -> LE -> TE_inf.
    - Normal exterior = (dy, -dx)/length  (gira tangente 90° en sentido horario).
    - Tangente = (dx, dy)/length.
    - V_inf = (cos α, sin α) * v_inf.
    - Cp = 1 - (v_tangent / v_inf)².
    - Kutta por igualdad de velocidades tangenciales inhomogénea:
        A[N, j] = I^t(panel 0, j) - I^t(panel N-1, j)
        b[N]    = V∞·t_{N-1} - V∞·t_0
"""

from __future__ import annotations
import math
from dataclasses import dataclass


# =============================================================================
# Álgebra lineal mínima (sin numpy): listas de listas + Gauss-Jordan.
# Suficiente para una matriz 81×81.
# =============================================================================
def matvec(M, v):
    n = len(M)
    return [sum(M[i][j] * v[j] for j in range(n)) for i in range(n)]


def solve_linear(A, b):
    """Resuelve A·x = b por eliminación gaussiana con pivoteo parcial.
    A es lista de listas (n×n); b es lista (n)."""
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        # Pivoteo: fila con mayor |M[i][col]| en i >= col
        best = max(range(col, n), key=lambda i: abs(M[i][col]))
        M[col], M[best] = M[best], M[col]
        piv = M[col][col]
        if abs(piv) < 1e-300:
            raise ValueError("Sistema singular")
        # Normalizar fila pivote
        for j in range(col, n + 1):
            M[col][j] /= piv
        # Eliminar en las demás filas
        for i in range(n):
            if i == col:
                continue
            factor = M[i][col]
            if factor == 0.0:
                continue
            for j in range(col, n + 1):
                M[i][j] -= factor * M[col][j]
    return [M[i][n] for i in range(n)]


# =============================================================================
# 1. PARSEC: réplica del cálculo de coeficientes de parsec.cpp.
# =============================================================================
@dataclass
class ParsecParameters:
    r_le: float
    x_up: float; y_up: float; d2y_up: float
    x_lo: float; y_lo: float; d2y_lo: float
    y_te: float; dy_te: float
    alpha_te: float; beta_te: float


def _split_trailing_edge(p: ParsecParameters):
    return {
        "y_upper": p.y_te + p.dy_te / 2.0,
        "th_upper": p.alpha_te - p.beta_te / 2.0,
        "y_lower": p.y_te - p.dy_te / 2.0,
        "th_lower": p.alpha_te + p.beta_te / 2.0,
    }


def _parsec_coeffs(r_le, x_m, y_m, d2y_m, y_te_s, th_te_s, is_upper):
    """Calcula los 6 coeficientes polinómicos [a1..a6] para una superficie."""
    assert 0.0 < x_m < 1.0
    a1 = (1.0 if is_upper else -1.0) * math.sqrt(2.0 * r_le)
    s05 = math.sqrt(x_m)
    s15 = s05 * x_m
    s25 = s15 * x_m
    s35 = s25 * x_m
    s45 = s35 * x_m

    A = [
        [s15,        s25,        s35,        s45,        s45 * x_m],
        [1.0,        1.0,        1.0,        1.0,        1.0],
        [1.5*s05,    2.5*s15,    3.5*s25,    4.5*s35,    5.5*s45],
        [1.5,        2.5,        3.5,        4.5,        5.5],
        [0.75/s05,   3.75*s05,   8.75*s15,   15.75*s25,  24.75*s35],
    ]
    b = [
        y_m      - a1 * s05,
        y_te_s   - a1,
        -0.5 * a1 / s05,
        math.tan(th_te_s) - 0.5 * a1,
        d2y_m    + 0.25 * a1 / (s05 * x_m),   # s05 * x_m = s15
    ]
    c2c6 = solve_linear(A, b)
    return [a1] + c2c6


class Parsec:
    def __init__(self, p: ParsecParameters):
        bc = _split_trailing_edge(p)
        self.a_upper = _parsec_coeffs(p.r_le, p.x_up, p.y_up, p.d2y_up,
                                      bc["y_upper"], bc["th_upper"], True)
        self.a_lower = _parsec_coeffs(p.r_le, p.x_lo, p.y_lo, p.d2y_lo,
                                      bc["y_lower"], bc["th_lower"], False)

    def get_y(self, x: float, is_upper: bool) -> float:
        if x == 0.0:
            return 0.0
        a = self.a_upper if is_upper else self.a_lower
        sx = math.sqrt(x)
        xp = sx
        z = 0.0
        for ai in a:
            z += ai * xp
            xp *= x
        return z


# =============================================================================
# 2. Discretización en herradura con cluster coseno (TE -> LE -> TE).
# =============================================================================
def build_herradura(parsec: Parsec, n_up: int, n_lo: int):
    """Devuelve lista de tuplas (x, y) en orden TE_sup -> LE -> TE_inf."""
    n_up_s = n_up + 1
    n_lo_s = n_lo + 1

    def x_cluster(i, n):
        t = i / (n - 1)
        return 0.5 * (1.0 - math.cos(math.pi * t))

    pts = []
    # Extradós: TE_sup (x=1) -> LE (x=0)
    for i in range(n_up_s):
        x = 1.0 - x_cluster(i, n_up_s)
        pts.append((x, parsec.get_y(x, True)))
    # Intradós: LE (x=0) -> TE_inf (x=1), saltamos i=0
    for i in range(1, n_lo_s):
        x = x_cluster(i, n_lo_s)
        pts.append((x, parsec.get_y(x, False)))
    return pts


# =============================================================================
# 3. Vórtice-lineal Katz & Plotkin §3.4 — espejo de panel_method.cpp.
# =============================================================================
def influence_coefficient_normal(panel_points, control_i, vortex_j):
    """Influencia NORMAL (proyectada contra n del panel receptor) del
    vórtice del nodo `vortex_j` sobre el centro del panel `control_i`.
    Excluye autoinducción cuando panel_idx == control_i."""
    N = len(panel_points) - 1

    def panel_influence(panel_idx, is_second_node):
        if control_i == panel_idx:
            return 0.0
        p1, p2 = panel_points[panel_idx], panel_points[panel_idx + 1]
        dx = p2[0] - p1[0]; dy = p2[1] - p1[1]
        length = math.hypot(dx, dy)
        cos_t = dx / length; sin_t = dy / length

        pi1, pi2 = panel_points[control_i], panel_points[control_i + 1]
        control = ((pi1[0] + pi2[0]) / 2.0, (pi1[1] + pi2[1]) / 2.0)
        dxi = pi2[0] - pi1[0]; dyi = pi2[1] - pi1[1]
        li = math.hypot(dxi, dyi)
        n_ix = dyi / li; n_iy = -dxi / li

        xt = control[0] - p1[0]; yt = control[1] - p1[1]
        x_star = xt * cos_t + yt * sin_t
        y_star = -xt * sin_t + yt * cos_t

        r1sq = x_star**2 + y_star**2
        r2sq = (x_star - length)**2 + y_star**2
        term1 = 0.5 * math.log(r1sq / r2sq)
        term2 = (math.atan2(y_star, x_star - length)
                 - math.atan2(y_star, x_star))

        if not is_second_node:
            u_star = (1.0 - x_star / length) * term2 + (y_star / length) * term1
            w_star = -(1.0 - x_star / length) * term1 + (y_star / length) * term2 - 1.0
        else:
            u_star = (x_star / length) * term2 - (y_star / length) * term1
            w_star = -(x_star / length) * term1 - (y_star / length) * term2 + 1.0
        u_star /= (2.0 * math.pi)
        w_star /= (2.0 * math.pi)

        u_g = u_star * cos_t - w_star * sin_t
        w_g = u_star * sin_t + w_star * cos_t
        return u_g * n_ix + w_g * n_iy

    out = 0.0
    if vortex_j > 0:
        out += panel_influence(vortex_j - 1, True)
    if vortex_j < N:
        out += panel_influence(vortex_j, False)
    return out


def tangential_influence(panel_points, control_i, vortex_j):
    """Influencia TANGENCIAL (proyectada contra t del panel receptor)."""
    N = len(panel_points) - 1

    def panel_influence(panel_idx, is_second_node):
        if control_i == panel_idx:
            return 0.0
        p1, p2 = panel_points[panel_idx], panel_points[panel_idx + 1]
        dx = p2[0] - p1[0]; dy = p2[1] - p1[1]
        length = math.hypot(dx, dy)
        cos_t = dx / length; sin_t = dy / length

        pi1, pi2 = panel_points[control_i], panel_points[control_i + 1]
        control = ((pi1[0] + pi2[0]) / 2.0, (pi1[1] + pi2[1]) / 2.0)
        dxi = pi2[0] - pi1[0]; dyi = pi2[1] - pi1[1]
        li = math.hypot(dxi, dyi)
        t_ix = dxi / li; t_iy = dyi / li

        xt = control[0] - p1[0]; yt = control[1] - p1[1]
        x_star = xt * cos_t + yt * sin_t
        y_star = -xt * sin_t + yt * cos_t

        r1sq = x_star**2 + y_star**2
        r2sq = (x_star - length)**2 + y_star**2
        term1 = 0.5 * math.log(r1sq / r2sq)
        term2 = (math.atan2(y_star, x_star - length)
                 - math.atan2(y_star, x_star))

        if not is_second_node:
            u_star = (1.0 - x_star / length) * term2 + (y_star / length) * term1
            w_star = -(1.0 - x_star / length) * term1 + (y_star / length) * term2 - 1.0
        else:
            u_star = (x_star / length) * term2 - (y_star / length) * term1
            w_star = -(x_star / length) * term1 - (y_star / length) * term2 + 1.0
        u_star /= (2.0 * math.pi)
        w_star /= (2.0 * math.pi)

        u_g = u_star * cos_t - w_star * sin_t
        w_g = u_star * sin_t + w_star * cos_t
        return u_g * t_ix + w_g * t_iy

    out = 0.0
    if vortex_j > 0:
        out += panel_influence(vortex_j - 1, True)
    if vortex_j < N:
        out += panel_influence(vortex_j, False)
    return out


def build_linear_system(panel_points, v_inf, alpha_deg, kutta="homogenea"):
    """Monta A·γ = b.  Kutta:
        - 'homogenea':   γ_0 + γ_N = 0       (la de tu C++ original)
        - 'tangencial':  v_tang(panel 0) == v_tang(panel N-1), inhomogénea.
    """
    N = len(panel_points) - 1
    alpha = math.radians(alpha_deg)
    V_ix = math.cos(alpha) * v_inf
    V_iy = math.sin(alpha) * v_inf

    A = [[0.0] * (N + 1) for _ in range(N + 1)]
    b = [0.0] * (N + 1)

    # Filas 0..N-1: V∞·n + Σ γ_j I^n_{ij} = 0
    for i in range(N):
        p1, p2 = panel_points[i], panel_points[i + 1]
        dx = p2[0] - p1[0]; dy = p2[1] - p1[1]
        length = math.hypot(dx, dy)
        n_x = dy / length; n_y = -dx / length
        b[i] = -(V_ix * n_x + V_iy * n_y)
        for j in range(N + 1):
            A[i][j] = influence_coefficient_normal(panel_points, i, j)

    # Fila N: Kutta
    if kutta == "homogenea":
        A[N][0] = 1.0
        A[N][N] = 1.0
        b[N] = 0.0
    elif kutta == "tangencial":
        for j in range(N + 1):
            A[N][j] = (tangential_influence(panel_points, 0, j)
                       - tangential_influence(panel_points, N - 1, j))
        def tangent(panel_idx):
            p1, p2 = panel_points[panel_idx], panel_points[panel_idx + 1]
            dx = p2[0] - p1[0]; dy = p2[1] - p1[1]
            length = math.hypot(dx, dy)
            return (dx / length, dy / length)
        t0 = tangent(0)
        tNm1 = tangent(N - 1)
        b[N] = V_ix * (tNm1[0] - t0[0]) + V_iy * (tNm1[1] - t0[1])
    else:
        raise ValueError(f"Kutta desconocida: {kutta}")

    return A, b


def v_tangent_at(panel_points, gamma, v_inf, alpha_deg, panel_i,
                 include_self_05=True):
    """Replica exacta del cálculo de v_tangent en solve() del C++."""
    alpha = math.radians(alpha_deg)
    V_ix = math.cos(alpha) * v_inf
    V_iy = math.sin(alpha) * v_inf
    p1, p2 = panel_points[panel_i], panel_points[panel_i + 1]
    dx = p2[0] - p1[0]; dy = p2[1] - p1[1]
    length = math.hypot(dx, dy)
    t_x = dx / length; t_y = dy / length

    v = V_ix * t_x + V_iy * t_y
    if include_self_05:
        v += 0.5 * (gamma[panel_i] + gamma[panel_i + 1])
    N = len(panel_points) - 1
    for j in range(N + 1):
        v += tangential_influence(panel_points, panel_i, j) * gamma[j]
    return v


def solve_full(panel_points, v_inf, alpha_deg, kutta="homogenea",
               include_self_05=True):
    """Pipeline completo: A, b, gamma, Cp, Cl, Cm."""
    A, b = build_linear_system(panel_points, v_inf, alpha_deg, kutta=kutta)
    gamma = solve_linear(A, b)

    N = len(panel_points) - 1
    alpha = math.radians(alpha_deg)
    V_ix = math.cos(alpha) * v_inf
    V_iy = math.sin(alpha) * v_inf

    cp = [0.0] * N
    v_t = [0.0] * N
    for i in range(N):
        vt = v_tangent_at(panel_points, gamma, v_inf, alpha_deg, i,
                          include_self_05=include_self_05)
        v_t[i] = vt
        cp[i] = 1.0 - (vt / v_inf) ** 2

    # Cl por integración de Cp·n proyectado contra lift
    cos_a, sin_a = math.cos(alpha), math.sin(alpha)
    Cl = 0.0
    for i in range(N):
        p1, p2 = panel_points[i], panel_points[i + 1]
        dx = p2[0] - p1[0]; dy = p2[1] - p1[1]
        length = math.hypot(dx, dy)
        n_x = dy / length; n_y = -dx / length
        cfx = -cp[i] * length * n_x
        cfy = -cp[i] * length * n_y
        Cl += cfy * cos_a - cfx * sin_a

    # Cm respecto al 25% de cuerda
    Cm = 0.0
    for i in range(N):
        p1, p2 = panel_points[i], panel_points[i + 1]
        mid = ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)
        dx = p2[0] - p1[0]; dy = p2[1] - p1[1]
        length = math.hypot(dx, dy)
        n_x = dy / length; n_y = -dx / length
        cfx = -cp[i] * length * n_x
        cfy = -cp[i] * length * n_y
        rx = mid[0] - 0.25; ry = mid[1] - 0.0
        Cm += rx * cfy - ry * cfx

    return {
        "A": A, "b": b, "gamma": gamma, "cp": cp, "v_tangent": v_t,
        "Cl": Cl, "Cm": Cm,
    }


# =============================================================================
# 4. Main: test con los mismos parámetros PARSEC que el test C++.
# =============================================================================
if __name__ == "__main__":
    p = ParsecParameters(
        r_le=0.0158,
        x_up=0.30, y_up=0.06, d2y_up=-0.6,
        x_lo=0.30, y_lo=-0.06, d2y_lo=0.6,
        y_te=0.0, dy_te=0.0,
        alpha_te=0.0, beta_te=0.0,
    )
    parsec = Parsec(p)
    pts = build_herradura(parsec, n_up=40, n_lo=40)
    print(f"Puntos: {len(pts)}  (esperado 81)")
    print(f"TE_sup = {pts[0]}")
    print(f"LE     = {pts[40]}")
    print(f"TE_inf = {pts[-1]}")

    alpha_deg = 5.0
    v_inf = 1.0

    print("\n========== KUTTA HOMOGÉNEA (γ_0+γ_N=0) — tu C++ original ==========")
    r_h = solve_full(pts, v_inf, alpha_deg, kutta="homogenea",
                     include_self_05=True)
    print(f"  gamma[0]        = {r_h['gamma'][0]: .6f}")
    print(f"  gamma[N]        = {r_h['gamma'][-1]: .6f}")
    print(f"  gamma[0]+gamma[N]= {r_h['gamma'][0]+r_h['gamma'][-1]: .6e}")
    print(f"  v_tang(panel 0) = {r_h['v_tangent'][0]: .6f}")
    print(f"  v_tang(panel N-1)= {r_h['v_tangent'][-1]: .6f}")
    print(f"  Cl              = {r_h['Cl']: .6f}")
    print(f"  Cm              = {r_h['Cm']: .6f}")
    print(f"  Cp min          = {min(r_h['cp']):.6f}  (panel {r_h['cp'].index(min(r_h['cp']))})")
    print(f"  Cp max          = {max(r_h['cp']):.6f}  (panel {r_h['cp'].index(max(r_h['cp']))})")

    print("\n========== KUTTA TANGENCIAL INHOMOGÉNEA ==========")
    r_t = solve_full(pts, v_inf, alpha_deg, kutta="tangencial",
                     include_self_05=True)
    print(f"  gamma[0]        = {r_t['gamma'][0]: .6f}")
    print(f"  gamma[N]        = {r_t['gamma'][-1]: .6f}")
    print(f"  gamma[0]+gamma[N]= {r_t['gamma'][0]+r_t['gamma'][-1]: .6e}")
    print(f"  v_tang(panel 0) = {r_t['v_tangent'][0]: .6f}")
    print(f"  v_tang(panel N-1)= {r_t['v_tangent'][-1]: .6f}")
    print(f"  Cl              = {r_t['Cl']: .6f}")
    print(f"  Cm              = {r_t['Cm']: .6f}")
    print(f"  Cp min          = {min(r_t['cp']):.6f}  (panel {r_t['cp'].index(min(r_t['cp']))})")
    print(f"  Cp max          = {max(r_t['cp']):.6f}  (panel {r_t['cp'].index(max(r_t['cp']))})")

    # Diferencia entre ambas Kutta
    print("\n========== DELTAS (tangencial - homogénea) ==========")
    dgamma = [r_t['gamma'][i] - r_h['gamma'][i] for i in range(len(r_h['gamma']))]
    print(f"  max |Δgamma| = {max(abs(d) for d in dgamma):.4e}")
    print(f"  ΔCl         = {r_t['Cl'] - r_h['Cl']:+.6f}")

    # Cp por panel en formato CSV-like para graficar
    print("\n========== Cp por panel (idx, x_medio, cp_homogenea, cp_tangencial) ==========")
    for i in range(len(pts) - 1):
        mid_x = (pts[i][0] + pts[i + 1][0]) / 2.0
        print(f"{i:3d}  {mid_x:.6f}  {r_h['cp'][i]:+.6f}  {r_t['cp'][i]:+.6f}")