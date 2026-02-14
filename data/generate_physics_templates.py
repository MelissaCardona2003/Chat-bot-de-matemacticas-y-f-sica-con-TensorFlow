"""
generate_physics_templates.py — Genera problemas de física con templates paramétricos.

Crea problemas variados de cinemática, dinámica, energía, electricidad,
termodinámica, ondas, óptica y gravitación usando plantillas con valores
aleatorios que producen respuestas calculadas automáticamente.

Cada problema sigue el esquema unificado (data/schema.py):
  - problem, solution (Step 1: ... Answer: ...), domain, source, topic, split

Uso:
    python data/generate_physics_templates.py [--count 3000]
"""

import json
import math
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Callable

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

SEED = 42


# ── Templates de Cinemática ───────────────────────────────────────

def kinematics_distance(rng: random.Random) -> Dict:
    """Distancia = velocidad × tiempo."""
    v = rng.choice([10, 15, 20, 25, 30, 40, 50, 60, 80, 100])
    t = rng.choice([2, 3, 4, 5, 6, 8, 10])
    unit_v = rng.choice(["km/h", "m/s", "mph"])
    d = v * t
    unit_d = {"km/h": "km", "m/s": "m", "mph": "miles"}[unit_v]
    unit_t = "hours" if unit_v in ("km/h", "mph") else "seconds"
    obj = rng.choice(["car", "train", "bicycle", "boat", "runner", "airplane"])
    return {
        "problem": f"A {obj} travels at {v} {unit_v} for {t} {unit_t}. What distance does it cover?",
        "solution": (f"Step 1: Use the formula distance = speed × time.\n"
                     f"Step 2: distance = {v} × {t} = {d} {unit_d}.\n"
                     f"Answer: {d} {unit_d}"),
        "topic": "kinematics",
    }


def kinematics_avg_speed(rng: random.Random) -> Dict:
    """Velocidad promedio = distancia total / tiempo total."""
    d1 = rng.choice([50, 60, 80, 100, 120, 150, 200])
    d2 = rng.choice([30, 40, 50, 60, 80, 100])
    t1 = rng.choice([1, 2, 3, 4, 5])
    t2 = rng.choice([1, 2, 3, 4])
    total_d = d1 + d2
    total_t = t1 + t2
    avg = round(total_d / total_t, 2)
    return {
        "problem": (f"A vehicle travels {d1} km in {t1} hours and then {d2} km "
                    f"in {t2} hours. What is its average speed?"),
        "solution": (f"Step 1: Total distance = {d1} + {d2} = {total_d} km.\n"
                     f"Step 2: Total time = {t1} + {t2} = {total_t} hours.\n"
                     f"Step 3: Average speed = {total_d} / {total_t} = {avg} km/h.\n"
                     f"Answer: {avg} km/h"),
        "topic": "kinematics",
    }


def kinematics_acceleration(rng: random.Random) -> Dict:
    """a = (v_f - v_i) / t."""
    vi = rng.choice([0, 5, 10, 15, 20])
    vf = vi + rng.choice([10, 15, 20, 25, 30, 40])
    t = rng.choice([2, 4, 5, 8, 10])
    a = round((vf - vi) / t, 2)
    return {
        "problem": (f"An object accelerates from {vi} m/s to {vf} m/s in {t} seconds. "
                    f"What is the acceleration?"),
        "solution": (f"Step 1: Use the formula a = (v_final - v_initial) / time.\n"
                     f"Step 2: a = ({vf} - {vi}) / {t} = {vf - vi} / {t} = {a} m/s^2.\n"
                     f"Answer: {a} m/s^2"),
        "topic": "kinematics",
    }


def kinematics_free_fall(rng: random.Random) -> Dict:
    """t = sqrt(2h/g), g=9.8 o 10."""
    g = rng.choice([10, 9.8])
    h = rng.choice([5, 10, 20, 30, 45, 50, 80, 100, 125])
    t = round(math.sqrt(2 * h / g), 2)
    return {
        "problem": (f"An object falls freely from a height of {h} m. "
                    f"How long does it take to reach the ground? (g = {g} m/s^2)"),
        "solution": (f"Step 1: Use the formula h = (1/2) × g × t^2, so t = sqrt(2h/g).\n"
                     f"Step 2: t = sqrt(2 × {h} / {g}) = sqrt({round(2*h/g, 4)}).\n"
                     f"Step 3: t = {t} seconds.\n"
                     f"Answer: {t} seconds"),
        "topic": "kinematics",
    }


def kinematics_final_velocity(rng: random.Random) -> Dict:
    """v_f = v_i + a*t."""
    vi = rng.choice([0, 5, 10, 15, 20])
    a = rng.choice([2, 3, 4, 5, 6, 8, 10])
    t = rng.choice([3, 4, 5, 6, 8, 10])
    vf = vi + a * t
    return {
        "problem": (f"A car starts at {vi} m/s and accelerates at {a} m/s^2 for "
                    f"{t} seconds. What is the final velocity?"),
        "solution": (f"Step 1: Use the formula v_final = v_initial + a × t.\n"
                     f"Step 2: v_final = {vi} + {a} × {t} = {vi} + {a * t} = {vf} m/s.\n"
                     f"Answer: {vf} m/s"),
        "topic": "kinematics",
    }


# ── Templates de Dinámica (Newton) ────────────────────────────────

def dynamics_force(rng: random.Random) -> Dict:
    """F = m × a."""
    m = rng.choice([2, 3, 4, 5, 8, 10, 12, 15, 20, 25, 50])
    a = rng.choice([2, 3, 4, 5, 6, 8, 10])
    f = m * a
    return {
        "problem": f"What is the net force on a {m} kg object with acceleration {a} m/s^2?",
        "solution": (f"Step 1: Use Newton's second law: F = m × a.\n"
                     f"Step 2: F = {m} × {a} = {f} N.\n"
                     f"Answer: {f} N"),
        "topic": "dynamics",
    }


def dynamics_mass_from_force(rng: random.Random) -> Dict:
    """m = F / a."""
    f = rng.choice([10, 20, 30, 40, 50, 60, 80, 100, 150, 200])
    a = rng.choice([2, 4, 5, 8, 10])
    m = round(f / a, 2)
    return {
        "problem": (f"A force of {f} N acts on an object, producing an acceleration "
                    f"of {a} m/s^2. What is the mass of the object?"),
        "solution": (f"Step 1: Use Newton's second law rearranged: m = F / a.\n"
                     f"Step 2: m = {f} / {a} = {m} kg.\n"
                     f"Answer: {m} kg"),
        "topic": "dynamics",
    }


def dynamics_weight(rng: random.Random) -> Dict:
    """W = m × g."""
    m = rng.choice([5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 100])
    g = rng.choice([9.8, 10])
    w = round(m * g, 1)
    return {
        "problem": f"What is the weight of a {m} kg object on Earth? (g = {g} m/s^2)",
        "solution": (f"Step 1: Weight = mass × gravitational acceleration.\n"
                     f"Step 2: W = {m} × {g} = {w} N.\n"
                     f"Answer: {w} N"),
        "topic": "dynamics",
    }


def dynamics_friction(rng: random.Random) -> Dict:
    """F_friction = mu × N."""
    m = rng.choice([5, 10, 15, 20, 25, 30, 50])
    mu = rng.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8])
    g = 10
    n = m * g
    f_fric = round(mu * n, 1)
    return {
        "problem": (f"A {m} kg box slides on a surface with friction coefficient "
                    f"{mu}. What is the friction force? (g = {g} m/s^2)"),
        "solution": (f"Step 1: Normal force N = m × g = {m} × {g} = {n} N.\n"
                     f"Step 2: Friction force = μ × N = {mu} × {n} = {f_fric} N.\n"
                     f"Answer: {f_fric} N"),
        "topic": "dynamics",
    }


def dynamics_momentum(rng: random.Random) -> Dict:
    """p = m × v."""
    m = rng.choice([2, 3, 5, 8, 10, 15, 20, 50, 100, 1000])
    v = rng.choice([2, 3, 5, 8, 10, 15, 20, 30, 50])
    p = m * v
    unit = "kg⋅m/s"
    obj = rng.choice(["ball", "car", "truck", "bullet", "person"])
    return {
        "problem": f"A {m} kg {obj} moves at {v} m/s. What is its momentum?",
        "solution": (f"Step 1: Momentum p = mass × velocity.\n"
                     f"Step 2: p = {m} × {v} = {p} {unit}.\n"
                     f"Answer: {p} {unit}"),
        "topic": "dynamics",
    }


# ── Templates de Energía ──────────────────────────────────────────

def energy_kinetic(rng: random.Random) -> Dict:
    """KE = 0.5 × m × v^2."""
    m = rng.choice([2, 3, 4, 5, 8, 10, 15, 20, 50, 100])
    v = rng.choice([2, 3, 4, 5, 6, 8, 10, 15, 20])
    ke = round(0.5 * m * v**2, 1)
    obj = rng.choice(["object", "ball", "car", "bicycle", "runner"])
    return {
        "problem": f"What is the kinetic energy of a {m} kg {obj} moving at {v} m/s?",
        "solution": (f"Step 1: Use the formula KE = (1/2) × m × v^2.\n"
                     f"Step 2: KE = 0.5 × {m} × {v}^2 = 0.5 × {m} × {v**2} = {ke} J.\n"
                     f"Answer: {ke} J"),
        "topic": "energy",
    }


def energy_potential(rng: random.Random) -> Dict:
    """PE = m × g × h."""
    m = rng.choice([2, 3, 5, 8, 10, 15, 20, 50])
    h = rng.choice([2, 3, 5, 8, 10, 15, 20, 30, 50, 100])
    g = rng.choice([9.8, 10])
    pe = round(m * g * h, 1)
    return {
        "problem": (f"A {m} kg object is at a height of {h} m above the ground. "
                    f"What is its gravitational potential energy? (g = {g} m/s^2)"),
        "solution": (f"Step 1: Use the formula PE = m × g × h.\n"
                     f"Step 2: PE = {m} × {g} × {h} = {pe} J.\n"
                     f"Answer: {pe} J"),
        "topic": "energy",
    }


def energy_work(rng: random.Random) -> Dict:
    """W = F × d × cos(θ)."""
    f = rng.choice([10, 20, 30, 40, 50, 60, 80, 100])
    d = rng.choice([2, 3, 5, 8, 10, 15, 20])
    angle = rng.choice([0, 30, 45, 60])
    cos_val = round(math.cos(math.radians(angle)), 4)
    w = round(f * d * cos_val, 1)
    if angle == 0:
        return {
            "problem": f"A force of {f} N pushes an object {d} m. How much work is done?",
            "solution": (f"Step 1: Work = Force × distance (when force is parallel to motion).\n"
                         f"Step 2: W = {f} × {d} = {w} J.\n"
                         f"Answer: {w} J"),
            "topic": "energy",
        }
    else:
        return {
            "problem": (f"A force of {f} N is applied at {angle}° to the horizontal "
                        f"over a distance of {d} m. How much work is done?"),
            "solution": (f"Step 1: Work = F × d × cos(θ).\n"
                         f"Step 2: W = {f} × {d} × cos({angle}°) = {f} × {d} × {cos_val}.\n"
                         f"Step 3: W = {w} J.\n"
                         f"Answer: {w} J"),
            "topic": "energy",
        }


def energy_power(rng: random.Random) -> Dict:
    """P = W / t."""
    w = rng.choice([100, 200, 300, 500, 600, 800, 1000, 1500, 2000, 5000])
    t = rng.choice([2, 4, 5, 8, 10, 20, 50, 100])
    p = round(w / t, 2)
    return {
        "problem": f"If {w} J of work is done in {t} seconds, what is the power?",
        "solution": (f"Step 1: Power = Work / time.\n"
                     f"Step 2: P = {w} / {t} = {p} W.\n"
                     f"Answer: {p} W"),
        "topic": "energy",
    }


# ── Templates de Electricidad ─────────────────────────────────────

def electricity_ohm(rng: random.Random) -> Dict:
    """V = I × R (Ohm's law)."""
    mode = rng.choice(["V", "I", "R"])
    if mode == "V":
        i = rng.choice([0.5, 1, 2, 3, 4, 5, 10])
        r = rng.choice([2, 4, 5, 8, 10, 20, 50, 100])
        v = round(i * r, 1)
        return {
            "problem": (f"A current of {i} A flows through a {r} Ω resistor. "
                        f"What is the voltage across it?"),
            "solution": (f"Step 1: Use Ohm's law: V = I × R.\n"
                         f"Step 2: V = {i} × {r} = {v} V.\n"
                         f"Answer: {v} V"),
            "topic": "electricity",
        }
    elif mode == "I":
        v = rng.choice([6, 9, 12, 24, 48, 120, 220])
        r = rng.choice([2, 3, 4, 6, 8, 10, 12, 20, 40, 100])
        i = round(v / r, 2)
        return {
            "problem": f"A {v} V battery is connected to a {r} Ω resistor. What is the current?",
            "solution": (f"Step 1: Use Ohm's law rearranged: I = V / R.\n"
                         f"Step 2: I = {v} / {r} = {i} A.\n"
                         f"Answer: {i} A"),
            "topic": "electricity",
        }
    else:
        v = rng.choice([6, 12, 24, 48, 120, 220])
        i = rng.choice([0.5, 1, 2, 3, 4, 5])
        r = round(v / i, 2)
        return {
            "problem": (f"A {v} V source drives a current of {i} A. "
                        f"What is the resistance?"),
            "solution": (f"Step 1: Use Ohm's law rearranged: R = V / I.\n"
                         f"Step 2: R = {v} / {i} = {r} Ω.\n"
                         f"Answer: {r} Ω"),
            "topic": "electricity",
        }


def electricity_power(rng: random.Random) -> Dict:
    """P = V × I or P = I^2 × R."""
    v = rng.choice([6, 12, 24, 48, 120, 220])
    i = rng.choice([0.5, 1, 2, 3, 5, 10])
    p = round(v * i, 1)
    return {
        "problem": (f"An electrical device operates at {v} V with a current of "
                    f"{i} A. What is the power consumed?"),
        "solution": (f"Step 1: Electrical power P = V × I.\n"
                     f"Step 2: P = {v} × {i} = {p} W.\n"
                     f"Answer: {p} W"),
        "topic": "electricity",
    }


def electricity_series_resistance(rng: random.Random) -> Dict:
    """R_total = R1 + R2 + R3."""
    n = rng.choice([2, 3, 4])
    resistors = [rng.choice([2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 50]) for _ in range(n)]
    total = sum(resistors)
    r_str = " + ".join(str(r) for r in resistors)
    r_list = ", ".join(f"{r} Ω" for r in resistors)
    return {
        "problem": (f"Resistors of {r_list} are connected in series. "
                    f"What is the total resistance?"),
        "solution": (f"Step 1: In series, total resistance = R1 + R2 + ... = {r_str}.\n"
                     f"Step 2: R_total = {total} Ω.\n"
                     f"Answer: {total} Ω"),
        "topic": "electricity",
    }


def electricity_energy(rng: random.Random) -> Dict:
    """E = P × t."""
    p = rng.choice([40, 60, 75, 100, 150, 200, 500, 1000, 2000])
    t = rng.choice([1, 2, 3, 4, 5, 8, 10])
    e_j = p * t * 3600
    e_kwh = round(p * t / 1000, 2)
    device = rng.choice(["light bulb", "heater", "motor", "appliance", "fan"])
    return {
        "problem": (f"A {p} W {device} runs for {t} hours. "
                    f"How much energy does it consume in kWh?"),
        "solution": (f"Step 1: Energy = Power × time.\n"
                     f"Step 2: E = {p} W × {t} h = {p * t} Wh.\n"
                     f"Step 3: Convert to kWh: {p * t} / 1000 = {e_kwh} kWh.\n"
                     f"Answer: {e_kwh} kWh"),
        "topic": "electricity",
    }


# ── Templates de Termodinámica ─────────────────────────────────────

def thermo_heat(rng: random.Random) -> Dict:
    """Q = m × c × ΔT."""
    m = rng.choice([0.5, 1, 2, 3, 5, 10])
    substance = rng.choice([
        ("water", 4186), ("iron", 450), ("aluminum", 900),
        ("copper", 385), ("oil", 2000),
    ])
    name, c = substance
    dt = rng.choice([5, 10, 15, 20, 25, 30, 40, 50])
    q = round(m * c * dt, 1)
    return {
        "problem": (f"How much heat is needed to raise the temperature of {m} kg of "
                    f"{name} by {dt}°C? (specific heat = {c} J/(kg·°C))"),
        "solution": (f"Step 1: Use Q = m × c × ΔT.\n"
                     f"Step 2: Q = {m} × {c} × {dt} = {q} J.\n"
                     f"Answer: {q} J"),
        "topic": "thermodynamics",
    }


def thermo_celsius_fahrenheit(rng: random.Random) -> Dict:
    """F = (9/5)×C + 32 or C = (5/9)×(F - 32)."""
    mode = rng.choice(["C_to_F", "F_to_C"])
    if mode == "C_to_F":
        c = rng.choice([0, 10, 20, 25, 30, 37, 50, 100])
        f = round(c * 9 / 5 + 32, 1)
        return {
            "problem": f"Convert {c}°C to Fahrenheit.",
            "solution": (f"Step 1: Use the formula F = (9/5) × C + 32.\n"
                         f"Step 2: F = (9/5) × {c} + 32 = {round(c * 9/5, 1)} + 32 = {f}°F.\n"
                         f"Answer: {f}°F"),
            "topic": "thermodynamics",
        }
    else:
        f = rng.choice([32, 50, 68, 86, 100, 212, 98.6])
        c = round((f - 32) * 5 / 9, 1)
        return {
            "problem": f"Convert {f}°F to Celsius.",
            "solution": (f"Step 1: Use the formula C = (5/9) × (F - 32).\n"
                         f"Step 2: C = (5/9) × ({f} - 32) = (5/9) × {f - 32} = {c}°C.\n"
                         f"Answer: {c}°C"),
            "topic": "thermodynamics",
        }


def thermo_ideal_gas(rng: random.Random) -> Dict:
    """PV = nRT, find one variable."""
    n = rng.choice([1, 2, 3, 5])
    r = 8.314  # J/(mol·K)
    t_k = rng.choice([273, 300, 350, 400, 500])
    v = rng.choice([0.01, 0.02, 0.05, 0.1, 0.5, 1])
    p = round(n * r * t_k / v, 1)
    return {
        "problem": (f"Find the pressure of {n} mol of ideal gas at {t_k} K "
                    f"in a {v} m^3 container. (R = 8.314 J/(mol·K))"),
        "solution": (f"Step 1: Use the ideal gas law: PV = nRT, so P = nRT/V.\n"
                     f"Step 2: P = {n} × 8.314 × {t_k} / {v}.\n"
                     f"Step 3: P = {round(n * r * t_k, 1)} / {v} = {p} Pa.\n"
                     f"Answer: {p} Pa"),
        "topic": "thermodynamics",
    }


# ── Templates de Ondas ────────────────────────────────────────────

def waves_speed(rng: random.Random) -> Dict:
    """v = f × λ."""
    f = rng.choice([50, 100, 200, 440, 500, 1000, 2000, 5000])
    lam = rng.choice([0.1, 0.2, 0.5, 0.68, 1, 1.5, 2, 3, 5])
    v = round(f * lam, 1)
    return {
        "problem": (f"A wave has frequency {f} Hz and wavelength {lam} m. "
                    f"What is the wave speed?"),
        "solution": (f"Step 1: Use the wave equation: v = f × λ.\n"
                     f"Step 2: v = {f} × {lam} = {v} m/s.\n"
                     f"Answer: {v} m/s"),
        "topic": "waves",
    }


def waves_period(rng: random.Random) -> Dict:
    """T = 1/f."""
    f = rng.choice([2, 4, 5, 10, 20, 50, 100, 200, 500, 1000])
    t = round(1 / f, 6)
    return {
        "problem": f"A wave has a frequency of {f} Hz. What is its period?",
        "solution": (f"Step 1: Period T = 1 / frequency.\n"
                     f"Step 2: T = 1 / {f} = {t} seconds.\n"
                     f"Answer: {t} seconds"),
        "topic": "waves",
    }


def waves_pendulum(rng: random.Random) -> Dict:
    """T = 2π × sqrt(L/g)."""
    l = rng.choice([0.25, 0.5, 1, 1.5, 2, 3, 4, 5])
    g = 9.8
    t = round(2 * math.pi * math.sqrt(l / g), 2)
    return {
        "problem": (f"A simple pendulum has a length of {l} m. "
                    f"What is its period? (g = {g} m/s^2)"),
        "solution": (f"Step 1: Use T = 2π × sqrt(L/g).\n"
                     f"Step 2: T = 2π × sqrt({l} / {g}) = 2π × {round(math.sqrt(l/g), 4)}.\n"
                     f"Step 3: T = {t} seconds.\n"
                     f"Answer: {t} seconds"),
        "topic": "waves",
    }


# ── Templates de Óptica ───────────────────────────────────────────

def optics_snell(rng: random.Random) -> Dict:
    """n1 × sin(θ1) = n2 × sin(θ2)."""
    n1 = 1.0  # Air
    n2 = rng.choice([1.33, 1.5, 1.52, 2.42])
    medium = {1.33: "water", 1.5: "glass", 1.52: "crown glass", 2.42: "diamond"}[n2]
    theta1 = rng.choice([15, 20, 30, 45, 60])
    sin_theta2 = round(n1 * math.sin(math.radians(theta1)) / n2, 4)
    if sin_theta2 > 1:
        sin_theta2 = 0.9  # clamp
    theta2 = round(math.degrees(math.asin(sin_theta2)), 1)
    return {
        "problem": (f"Light passes from air (n={n1}) into {medium} (n={n2}) "
                    f"at an angle of {theta1}°. What is the angle of refraction?"),
        "solution": (f"Step 1: Use Snell's law: n1 × sin(θ1) = n2 × sin(θ2).\n"
                     f"Step 2: sin(θ2) = n1 × sin({theta1}°) / n2 = "
                     f"{round(n1 * math.sin(math.radians(theta1)), 4)} / {n2} = {sin_theta2}.\n"
                     f"Step 3: θ2 = arcsin({sin_theta2}) = {theta2}°.\n"
                     f"Answer: {theta2} degrees"),
        "topic": "optics",
    }


def optics_mirror(rng: random.Random) -> Dict:
    """1/f = 1/do + 1/di."""
    f = rng.choice([5, 8, 10, 12, 15, 20, 25, 30])
    do = rng.choice([v for v in [10, 15, 20, 25, 30, 40, 50, 60] if v != f])
    if do == f:
        do = f + 5
    di = round(1 / (1/f - 1/do), 1) if (1/f - 1/do) != 0 else float('inf')
    if abs(di) > 1000:
        do = f + 10
        di = round(1 / (1/f - 1/do), 1)
    mirror_type = rng.choice(["concave mirror", "converging lens"])
    return {
        "problem": (f"A {mirror_type} has focal length {f} cm. An object is placed "
                    f"{do} cm away. Where is the image formed?"),
        "solution": (f"Step 1: Use the mirror/lens equation: 1/f = 1/do + 1/di.\n"
                     f"Step 2: 1/di = 1/f - 1/do = 1/{f} - 1/{do}.\n"
                     f"Step 3: 1/di = {round(1/f, 6)} - {round(1/do, 6)} = {round(1/f - 1/do, 6)}.\n"
                     f"Step 4: di = {di} cm.\n"
                     f"Answer: {di} cm"),
        "topic": "optics",
    }


# ── Templates de Gravitación ──────────────────────────────────────

def gravitation_force(rng: random.Random) -> Dict:
    """F = G × m1 × m2 / r^2."""
    m1 = rng.choice([5.97e24, 7.35e22, 1.99e30, 1000, 5000])
    m2 = rng.choice([1, 10, 50, 100, 500, 1000, 7.35e22])
    r = rng.choice([6.37e6, 3.84e8, 1.5e11, 100, 1000])
    g_const = 6.674e-11
    f = g_const * m1 * m2 / r**2
    f_str = f"{f:.2e}"
    return {
        "problem": (f"Two objects of mass {m1:.2e} kg and {m2:.2e} kg are "
                    f"separated by {r:.2e} m. What is the gravitational force? "
                    f"(G = 6.674e-11 N⋅m^2/kg^2)"),
        "solution": (f"Step 1: Use Newton's law of gravitation: F = G × m1 × m2 / r^2.\n"
                     f"Step 2: F = 6.674e-11 × {m1:.2e} × {m2:.2e} / ({r:.2e})^2.\n"
                     f"Step 3: F = {f_str} N.\n"
                     f"Answer: {f_str} N"),
        "topic": "gravitation",
    }


def gravitation_orbital_speed(rng: random.Random) -> Dict:
    """v = sqrt(GM/r)."""
    body = rng.choice([
        ("Earth", 5.97e24, 6.37e6),
        ("Moon", 7.35e22, 1.74e6),
    ])
    name, m_body, r_surface = body
    h = rng.choice([200e3, 400e3, 500e3, 1000e3, 2000e3])
    r = r_surface + h
    g_const = 6.674e-11
    v = round(math.sqrt(g_const * m_body / r), 1)
    return {
        "problem": (f"A satellite orbits {name} at {h/1000:.0f} km above the surface. "
                    f"What is its orbital speed? (M_{name} = {m_body:.2e} kg, "
                    f"R_{name} = {r_surface/1e6:.2f}e6 m)"),
        "solution": (f"Step 1: Orbital radius r = R + h = {r_surface:.2e} + {h:.2e} = {r:.2e} m.\n"
                     f"Step 2: v = sqrt(G × M / r) = sqrt(6.674e-11 × {m_body:.2e} / {r:.2e}).\n"
                     f"Step 3: v = {v} m/s.\n"
                     f"Answer: {v} m/s"),
        "topic": "gravitation",
    }


# ── Templates de Fluidos ──────────────────────────────────────────

def fluids_pressure(rng: random.Random) -> Dict:
    """P = ρgh."""
    rho = rng.choice([1000, 1025, 900, 13600])
    fluid = {1000: "water", 1025: "seawater", 900: "oil", 13600: "mercury"}[rho]
    h = rng.choice([1, 2, 3, 5, 10, 15, 20, 50])
    g = 9.8
    p = round(rho * g * h, 1)
    return {
        "problem": (f"What is the pressure at the bottom of a {h} m deep "
                    f"column of {fluid}? (density = {rho} kg/m^3, g = {g} m/s^2)"),
        "solution": (f"Step 1: Use P = ρ × g × h.\n"
                     f"Step 2: P = {rho} × {g} × {h} = {p} Pa.\n"
                     f"Answer: {p} Pa"),
        "topic": "fluids",
    }


def fluids_buoyancy(rng: random.Random) -> Dict:
    """F_b = ρ_fluid × V × g."""
    rho = rng.choice([1000, 1025])
    v = rng.choice([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 1])
    g = 9.8
    fb = round(rho * v * g, 1)
    return {
        "problem": (f"An object with volume {v} m^3 is submerged in water "
                    f"(density = {rho} kg/m^3). What is the buoyant force? (g = {g} m/s^2)"),
        "solution": (f"Step 1: Buoyant force = ρ_fluid × V × g (Archimedes' principle).\n"
                     f"Step 2: F_b = {rho} × {v} × {g} = {fb} N.\n"
                     f"Answer: {fb} N"),
        "topic": "fluids",
    }


# ── Registro de todos los templates ───────────────────────────────

ALL_TEMPLATES: List[Callable] = [
    # Cinemática (5)
    kinematics_distance,
    kinematics_avg_speed,
    kinematics_acceleration,
    kinematics_free_fall,
    kinematics_final_velocity,
    # Dinámica (5)
    dynamics_force,
    dynamics_mass_from_force,
    dynamics_weight,
    dynamics_friction,
    dynamics_momentum,
    # Energía (4)
    energy_kinetic,
    energy_potential,
    energy_work,
    energy_power,
    # Electricidad (4)
    electricity_ohm,
    electricity_power,
    electricity_series_resistance,
    electricity_energy,
    # Termodinámica (3)
    thermo_heat,
    thermo_celsius_fahrenheit,
    thermo_ideal_gas,
    # Ondas (3)
    waves_speed,
    waves_period,
    waves_pendulum,
    # Óptica (2)
    optics_snell,
    optics_mirror,
    # Gravitación (2)
    gravitation_force,
    gravitation_orbital_speed,
    # Fluidos (2)
    fluids_pressure,
    fluids_buoyancy,
]


def generate_physics_problems(count: int = 3000, seed: int = SEED) -> List[Dict]:
    """
    Genera problemas de física usando todos los templates.

    Distribuye uniformemente entre templates y añade variación
    con semillas aleatorias diferentes.

    Args:
        count: Número total de problemas a generar.
        seed: Semilla para reproducibilidad.

    Returns:
        Lista de problemas con esquema unificado.
    """
    rng = random.Random(seed)
    problems = []
    seen_problems = set()  # Para evitar duplicados exactos

    per_template = max(1, count // len(ALL_TEMPLATES))
    extra = count - per_template * len(ALL_TEMPLATES)

    for template_fn in ALL_TEMPLATES:
        n = per_template + (1 if extra > 0 else 0)
        if extra > 0:
            extra -= 1

        generated = 0
        attempts = 0
        while generated < n and attempts < n * 10:
            attempts += 1
            try:
                entry = template_fn(rng)
            except (ZeroDivisionError, ValueError, OverflowError):
                continue

            # Deduplicar
            key = entry["problem"]
            if key in seen_problems:
                continue
            seen_problems.add(key)

            # Añadir campos del esquema
            entry["domain"] = "physics"
            entry["source"] = "PhysicsTemplates"
            # split se asigna después
            problems.append(entry)
            generated += 1

    # Barajar
    rng.shuffle(problems)

    print(f"  Generados: {len(problems)} problemas de física")
    return problems


def assign_splits(problems: List[Dict], seed: int = SEED) -> List[Dict]:
    """
    Asigna splits: 85% train, 10% val, 5% test.
    """
    rng = random.Random(seed)
    indices = list(range(len(problems)))
    rng.shuffle(indices)

    n = len(problems)
    n_test = max(1, int(n * 0.05))
    n_val = max(1, int(n * 0.10))

    for i in indices[:n_test]:
        problems[i]["split"] = "test"
    for i in indices[n_test:n_test + n_val]:
        problems[i]["split"] = "val"
    for i in indices[n_test + n_val:]:
        problems[i]["split"] = "train"

    return problems


def main():
    parser = argparse.ArgumentParser(description="Genera problemas de física con templates")
    parser.add_argument("--count", type=int, default=3000,
                        help="Número de problemas a generar (default: 3000)")
    args = parser.parse_args()

    print("=" * 60)
    print("  GENERACIÓN DE PROBLEMAS DE FÍSICA (TEMPLATES)")
    print("=" * 60)

    # Generar
    problems = generate_physics_problems(count=args.count)

    if not problems:
        print("No se generaron problemas.")
        return

    # Asignar splits
    problems = assign_splits(problems)

    # Estadísticas de temas
    from collections import Counter
    topics = Counter(p["topic"] for p in problems)
    splits = Counter(p["split"] for p in problems)

    print(f"\n  Total: {len(problems)} problemas")
    print(f"\n  Por tema:")
    for t, c in sorted(topics.items(), key=lambda x: -x[1]):
        print(f"    {t}: {c}")
    print(f"\n  Por split:")
    for s, c in sorted(splits.items()):
        print(f"    {s}: {c}")

    # Combinar con physics_problems.json manual si existe
    manual_path = DATA_DIR / "physics_problems.json"
    if manual_path.exists():
        with open(manual_path, "r", encoding="utf-8") as f:
            manual = json.load(f)

        from transformer_math_physics_tutor.data.schema import normalize_solution
        for p in manual:
            sol = p.get("solution", "")
            if "Step 1:" not in sol:
                sol = normalize_solution(sol)
            problems.append({
                "problem": p["problem"],
                "solution": sol,
                "domain": "physics",
                "source": "ManualPhysics",
                "topic": p.get("topic", "general_physics"),
                "split": "train",
            })
        print(f"\n  + {len(manual)} problemas manuales de physics_problems.json")

    # Guardar
    output_path = DATA_DIR / "physics_combined.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(problems, f, ensure_ascii=False, indent=2)
    print(f"\n  Guardado: {output_path} ({len(problems)} problemas)")

    # Ejemplo
    if problems:
        ex = problems[0]
        print(f"\n  Ejemplo:")
        print(f"    problem: {ex['problem']}")
        print(f"    solution: {ex['solution']}")
        print(f"    topic: {ex['topic']}, split: {ex['split']}")

    print(f"\n{'=' * 60}")
    print("  GENERACIÓN COMPLETADA")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
