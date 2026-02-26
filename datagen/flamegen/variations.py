"""Fractal flame variation functions (V0–V13+).

Each variation takes (x, y) and returns (x', y').
Based on the Draves & Reckase fractal flame algorithm paper.
All functions are Numba JIT-compiled for performance.
"""

import math

import numba as nb


@nb.njit(cache=True)
def v_linear(x: float, y: float) -> tuple[float, float]:
    return x, y


@nb.njit(cache=True)
def v_sinusoidal(x: float, y: float) -> tuple[float, float]:
    return math.sin(x), math.sin(y)


@nb.njit(cache=True)
def v_spherical(x: float, y: float) -> tuple[float, float]:
    r2 = x * x + y * y + 1e-10
    return x / r2, y / r2


@nb.njit(cache=True)
def v_swirl(x: float, y: float) -> tuple[float, float]:
    r2 = x * x + y * y
    s = math.sin(r2)
    c = math.cos(r2)
    return x * s - y * c, x * c + y * s


@nb.njit(cache=True)
def v_horseshoe(x: float, y: float) -> tuple[float, float]:
    r = math.sqrt(x * x + y * y) + 1e-10
    return (x - y) * (x + y) / r, 2.0 * x * y / r


@nb.njit(cache=True)
def v_polar(x: float, y: float) -> tuple[float, float]:
    r = math.sqrt(x * x + y * y)
    theta = math.atan2(y, x)
    return theta / math.pi, r - 1.0


@nb.njit(cache=True)
def v_handkerchief(x: float, y: float) -> tuple[float, float]:
    r = math.sqrt(x * x + y * y)
    theta = math.atan2(y, x)
    return r * math.sin(theta + r), r * math.cos(theta - r)


@nb.njit(cache=True)
def v_heart(x: float, y: float) -> tuple[float, float]:
    r = math.sqrt(x * x + y * y)
    theta = math.atan2(y, x)
    return r * math.sin(theta * r), -r * math.cos(theta * r)


@nb.njit(cache=True)
def v_disc(x: float, y: float) -> tuple[float, float]:
    r = math.sqrt(x * x + y * y)
    theta = math.atan2(y, x)
    t = theta / math.pi
    return t * math.sin(math.pi * r), t * math.cos(math.pi * r)


@nb.njit(cache=True)
def v_spiral(x: float, y: float) -> tuple[float, float]:
    r = math.sqrt(x * x + y * y) + 1e-10
    theta = math.atan2(y, x)
    return (math.cos(theta) + math.sin(r)) / r, (math.sin(theta) - math.cos(r)) / r


@nb.njit(cache=True)
def v_hyperbolic(x: float, y: float) -> tuple[float, float]:
    r = math.sqrt(x * x + y * y) + 1e-10
    theta = math.atan2(y, x)
    return math.sin(theta) / r, r * math.cos(theta)


@nb.njit(cache=True)
def v_diamond(x: float, y: float) -> tuple[float, float]:
    r = math.sqrt(x * x + y * y)
    theta = math.atan2(y, x)
    return math.sin(theta) * math.cos(r), math.cos(theta) * math.sin(r)


@nb.njit(cache=True)
def v_ex(x: float, y: float) -> tuple[float, float]:
    r = math.sqrt(x * x + y * y)
    theta = math.atan2(y, x)
    p0 = math.sin(theta + r)
    p1 = math.cos(theta - r)
    return r * (p0 * p0 * p0 + p1 * p1 * p1), r * (p0 * p0 * p0 - p1 * p1 * p1)


@nb.njit(cache=True)
def v_julia(x: float, y: float, rng_val: float) -> tuple[float, float]:
    r = math.sqrt(math.sqrt(x * x + y * y))
    theta = math.atan2(y, x) / 2.0
    if rng_val > 0.5:
        theta += math.pi
    return r * math.cos(theta), r * math.sin(theta)


@nb.njit(cache=True)
def v_fisheye(x: float, y: float) -> tuple[float, float]:
    r = math.sqrt(x * x + y * y)
    s = 2.0 / (r + 1.0)
    return s * y, s * x


@nb.njit(cache=True)
def v_eyefish(x: float, y: float) -> tuple[float, float]:
    r = math.sqrt(x * x + y * y)
    s = 2.0 / (r + 1.0)
    return s * x, s * y


@nb.njit(cache=True)
def v_bubble(x: float, y: float) -> tuple[float, float]:
    r2 = x * x + y * y
    s = 4.0 / (r2 + 4.0)
    return s * x, s * y


@nb.njit(cache=True)
def v_cylinder(x: float, y: float) -> tuple[float, float]:
    return math.sin(x), y


@nb.njit(cache=True)
def v_tangent(x: float, y: float) -> tuple[float, float]:
    cy = math.cos(y)
    if abs(cy) < 1e-10:
        cy = 1e-10
    return math.sin(x) / cy, math.tan(y)


# Total number of variations (excluding julia which needs extra rng param)
NUM_VARIATIONS = 18

# Indices for referencing variations
V_LINEAR = 0
V_SINUSOIDAL = 1
V_SPHERICAL = 2
V_SWIRL = 3
V_HORSESHOE = 4
V_POLAR = 5
V_HANDKERCHIEF = 6
V_HEART = 7
V_DISC = 8
V_SPIRAL = 9
V_HYPERBOLIC = 10
V_DIAMOND = 11
V_EX = 12
V_JULIA = 13
V_FISHEYE = 14
V_EYEFISH = 15
V_BUBBLE = 16
V_CYLINDER = 17


@nb.njit(cache=True)
def apply_variation(var_id: int, x: float, y: float, rng_val: float) -> tuple[float, float]:
    """Dispatch to the appropriate variation function by index."""
    if var_id == 0:
        return v_linear(x, y)
    elif var_id == 1:
        return v_sinusoidal(x, y)
    elif var_id == 2:
        return v_spherical(x, y)
    elif var_id == 3:
        return v_swirl(x, y)
    elif var_id == 4:
        return v_horseshoe(x, y)
    elif var_id == 5:
        return v_polar(x, y)
    elif var_id == 6:
        return v_handkerchief(x, y)
    elif var_id == 7:
        return v_heart(x, y)
    elif var_id == 8:
        return v_disc(x, y)
    elif var_id == 9:
        return v_spiral(x, y)
    elif var_id == 10:
        return v_hyperbolic(x, y)
    elif var_id == 11:
        return v_diamond(x, y)
    elif var_id == 12:
        return v_ex(x, y)
    elif var_id == 13:
        return v_julia(x, y, rng_val)
    elif var_id == 14:
        return v_fisheye(x, y)
    elif var_id == 15:
        return v_eyefish(x, y)
    elif var_id == 16:
        return v_bubble(x, y)
    elif var_id == 17:
        return v_cylinder(x, y)
    else:
        return x, y
