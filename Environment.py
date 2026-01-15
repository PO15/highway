import time
import types
import random
import gymnasium as gym
import highway_env  # registriert env-ids

from aggressivness import NormalIDMVehicle, AggressiveIDMVehicle, DefensiveIDMVehicle


# -----------------------------
# TTC Hilfsfunktionen (Ego)
# -----------------------------
def get_speed(vehicle):
    return float(getattr(vehicle, "speed", 0.0))

def get_length(vehicle, default=5.0):
    return float(getattr(vehicle, "LENGTH", getattr(vehicle, "length", default)))

def get_bumper_gap(rear_vehicle, front_vehicle):
    # bumper-to-bumper Abstand in x-Richtung (Fahrtrichtung)
    dx = float(front_vehicle.position[0] - rear_vehicle.position[0])
    return dx - 0.5 * (get_length(rear_vehicle) + get_length(front_vehicle))

def compute_ttc(gap, closing_speed):
    # gap <= 0 -> Kontakt/Überlappung
    if gap <= 0:
        return 0.0
    # closing_speed <= 0 -> keine Annäherung
    if closing_speed <= 0:
        return float("inf")
    return gap / closing_speed

def ego_ttc_step(env, prev_lane, beta=6.0, rho=0.1, tau=0.3):
    """
    Ego-TTC pro Step:
    - ttc_follow: Ego -> nächster Vordermann auf gleicher Spur
    - cutin_best_phys: kleinste TTC_phys bei Cut-in in Ego-Spur (sonst inf)
    - cutin_best_dvo: passende DVO-Schwelle (sonst inf)
    - cutin_critical: TTC_phys < TTC_DVO ?
    """
    uw = env.unwrapped
    road = uw.road
    ego = uw.vehicle
    ego_lane = getattr(ego, "lane_index", None)

    vehicles = list(road.vehicles)

    # prev_lane initialisieren (damit wir "vorher vs jetzt" vergleichen können)
    for v in vehicles:
        if v not in prev_lane:
            prev_lane[v] = getattr(v, "lane_index", None)

    # -------- 1) Car-following TTC (Ego -> nächster Vordermann) --------
    front = None
    best_dx = float("inf")

    for v in vehicles:
        if v is ego:
            continue
        if getattr(v, "lane_index", None) != ego_lane:
            continue
        dx = float(v.position[0] - ego.position[0])
        if dx > 0 and dx < best_dx:
            best_dx = dx
            front = v

    if front is None:
        ttc_follow = float("inf")
    else:
        gap = get_bumper_gap(ego, front)
        v_rel = get_speed(ego) - get_speed(front)  # >0: Ego nähert sich
        ttc_follow = compute_ttc(gap, v_rel)

    # -------- 2) Cut-in TTC (jemand wechselt in Ego-Spur) --------
    cutin_best_phys = float("inf")
    cutin_best_dvo = float("inf")
    cutin_critical = False

    for v in vehicles:
        if v is ego:
            continue

        prev = prev_lane.get(v, None)
        cur = getattr(v, "lane_index", None)

        # Cut-in Event: vorher andere Spur, jetzt Ego-Spur
        if cur == ego_lane and prev is not None and prev != ego_lane:
            dx = float(v.position[0] - ego.position[0])
            if dx <= 0:
                continue  # nur Cut-in vor Ego

            gap = get_bumper_gap(ego, v)
            v_rel = get_speed(ego) - get_speed(v)
            ttc_phys = compute_ttc(gap, v_rel)

            # DVO Schwelle nur sinnvoll bei v_rel > 0
            if v_rel > 0:
                ttc_dvo = (v_rel / (2.0 * beta)) + rho + (tau / 2.0)
            else:
                ttc_dvo = float("inf")

            # schlimmsten (kleinsten) Cut-in dieses Steps merken
            if ttc_phys < cutin_best_phys:
                cutin_best_phys = ttc_phys
                cutin_best_dvo = ttc_dvo
                cutin_critical = (ttc_phys < ttc_dvo)

    # prev_lane updaten für nächsten Step
    for v in vehicles:
        prev_lane[v] = getattr(v, "lane_index", None)

    return ttc_follow, cutin_best_phys, cutin_best_dvo, cutin_critical


# -----------------------------
# Ego zu IDM + Ignore Action
# -----------------------------
def convert_ego_to_idm(env, target_speed_kmh=None):
    """
    Ersetzt das kontrollierte Ego-Fahrzeug durch ein NormalIDMVehicle (IDM+MOBIL),
    und sorgt dafür, dass env.step(action) die Action ignoriert.
    """
    uw = env.unwrapped
    old_ego = uw.vehicle

    idm_ego = NormalIDMVehicle.create_from(old_ego)

    if target_speed_kmh is not None and hasattr(idm_ego, "target_speed"):
        idm_ego.target_speed = target_speed_kmh / 3.6

    for i, v in enumerate(uw.road.vehicles):
        if v is old_ego:
            uw.road.vehicles[i] = idm_ego
            break

    uw.vehicle = idm_ego
    if hasattr(uw, "controlled_vehicles"):
        if uw.controlled_vehicles:
            uw.controlled_vehicles[0] = idm_ego
        else:
            uw.controlled_vehicles = [idm_ego]

    try:
        uw.action_type.controlled_vehicle = idm_ego
    except Exception:
        pass

    def _act_ignore(self, action):
        self.controlled_vehicle.act(None)

    uw.action_type.act = types.MethodType(_act_ignore, uw.action_type)

    return idm_ego


def apply_global_target_speed(env, target_speed_kmh):
    """Setzt target_speed (falls vorhanden) für ALLE Fahrzeuge auf der Road."""
    uw = env.unwrapped
    v_target = target_speed_kmh / 3.6
    for v in getattr(uw.road, "vehicles", []):
        if hasattr(v, "target_speed"):
            v.target_speed = v_target


def _set_style_color(vehicle):
    """Aggressiv=rot, Defensiv=gelb, Normal=Standardfarbe."""
    if isinstance(vehicle, AggressiveIDMVehicle):
        vehicle.color = (255, 0, 0)
    elif isinstance(vehicle, DefensiveIDMVehicle):
        vehicle.color = (255, 255, 0)
    else:
        if hasattr(vehicle, "color"):
            delattr(vehicle, "color")


def apply_traffic_style_mix(env, p_normal=0.70, p_def=0.15, p_agg=0.15):
    """
    Setzt für alle NICHT-Ego Fahrzeuge einen Mix:
    70% normal, 15% defensiv, 15% aggressiv.
    """
    uw = env.unwrapped
    ego = uw.vehicle

    traffic_idx = [i for i, v in enumerate(uw.road.vehicles) if v is not ego]
    n = len(traffic_idx)
    if n == 0:
        return

    n_def = int(round(p_def * n))
    n_agg = int(round(p_agg * n))
    n_norm = n - n_def - n_agg

    styles = (
        [NormalIDMVehicle] * n_norm
        + [DefensiveIDMVehicle] * n_def
        + [AggressiveIDMVehicle] * n_agg
    )
    random.shuffle(styles)

    for idx, cls in zip(traffic_idx, styles):
        old_v = uw.road.vehicles[idx]
        if isinstance(old_v, cls):
            _set_style_color(old_v)
            continue
        new_v = cls.create_from(old_v)
        _set_style_color(new_v)
        uw.road.vehicles[idx] = new_v


# -----------------------------
# Main
# -----------------------------
def main():
    scenario_config = {
        "vehicles_count": 60,
        "duration": 15,
        "policy_frequency": 10,
        "simulation_frequency": 15,
        "lanes_count": 3,
        "ego_spacing": 1,
        "vehicles_density": 2.0,
        "target_speed_kmh": 150,
        "screen_width": 1400,
        "screen_height": 350,
        "scaling": 1,
        "centering_position": [0.35, 0.5],
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 15,
            "features": ["x", "y", "vx", "vy"],
            "absolute": False,
        },
        "action": {"type": "DiscreteMetaAction"},
    }

    # DVO Parameter (typisch PKW: tau eher 0.3)
    BETA, RHO, TAU = 6.0, 0.1, 0.3

    env = gym.make("highway-v0", render_mode="human")
    obs, info = env.reset(options={"config": scenario_config})

    ts = scenario_config["target_speed_kmh"]

    # Setup nach Reset
    convert_ego_to_idm(env, target_speed_kmh=ts)
    apply_traffic_style_mix(env, p_normal=0.70, p_def=0.15, p_agg=0.15)
    apply_global_target_speed(env, target_speed_kmh=ts)

    # Wichtig: nach dem Mix wurden Vehicles ggf. ersetzt -> prev_lane neu starten
    prev_lane = {}

    # optional: Episode-Minima
    min_follow = float("inf")
    min_cutin = float("inf")
    min_any = float("inf")

    print("Starte Simulation (Ego = NormalIDMVehicle). Beenden: Ctrl+C")
    print("config vehicles_count:", env.unwrapped.config["vehicles_count"])
    print("vehicles in road:", len(env.unwrapped.road.vehicles))
    print("target_speed_kmh:", ts)

    try:
        for step in range(2000):
            action = env.action_space.sample()  # wird ignoriert
            obs, reward, terminated, truncated, info = env.step(action)

            ttc_follow, ttc_cutin_phys, ttc_cutin_dvo, cutin_crit = ego_ttc_step(
                env, prev_lane, beta=BETA, rho=RHO, tau=TAU
            )

            # Minima updaten
            min_follow = min(min_follow, ttc_follow)
            min_cutin = min(min_cutin, ttc_cutin_phys)
            min_any = min(min_any, ttc_follow, ttc_cutin_phys)

            # Ausgabe nur wenn "relevant" (sonst spam)
            if ttc_follow < 2.0:
                print(f"[FOLLOW] TTC={ttc_follow:.2f}s")

            if ttc_cutin_phys < 3.0:  # nur nahe Cut-ins loggen
                print(f"[CUT-IN] TTC_phys={ttc_cutin_phys:.2f}s | TTC_DVO={ttc_cutin_dvo:.2f}s | critical={cutin_crit}")

            time.sleep(0.03)

            # optional: target_speed gelegentlich nachziehen
            if step % 20 == 0:
                apply_global_target_speed(env, target_speed_kmh=ts)

            if terminated or truncated:
                print(f"Episode Ende | min_follow={min_follow:.2f}s | min_cutin={min_cutin:.2f}s | min_any={min_any:.2f}s")
                obs, info = env.reset(options={"config": scenario_config})

                convert_ego_to_idm(env, target_speed_kmh=ts)
                apply_traffic_style_mix(env, p_normal=0.70, p_def=0.15, p_agg=0.15)
                apply_global_target_speed(env, target_speed_kmh=ts)

                prev_lane = {}
                min_follow = float("inf")
                min_cutin = float("inf")
                min_any = float("inf")

    except KeyboardInterrupt:
        print("\nAbbruch per Ctrl+C.")
    finally:
        env.close()
        print("Env geschlossen ✅")


if __name__ == "__main__":
    main()
