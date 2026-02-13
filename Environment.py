import time
import types
import random
import gymnasium as gym
import highway_env  # registriert env-ids

from aggressivness import NormalIDMVehicle, AggressiveIDMVehicle, DefensiveIDMVehicle
import copy

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
    idm_ego.color = (0, 255, 0)  # Ego immer grün

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


def apply_traffic_style_mix(env, p_normal=0.1982, p_def=0.6246, p_agg=0.1772):
    """
    Setzt für alle NICHT-Ego Fahrzeuge einen Mix:
    19.82% normal, 0.6246% defensiv, 17.72% aggressiv.
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


# ---------------------------------------------------------
# 2) Eine Episode laufen lassen und Kennzahlen zurückgeben
# ---------------------------------------------------------
def run_one_episode(env, cfg, episode_seed,
                    BETA=6.0, RHO=0.1, TAU=0.3,
                    ttc_thr_crit=1.5, ttc_thr_warn=3.0):
    """
    Führt genau eine Episode aus.
    Gibt min_follow, min_cutin und min_any zurück.
    """

    # Env resetten: seed + config setzen
    env.reset(seed=episode_seed, options={"config": cfg})

    # Zielgeschwindigkeit (aus cfg)
    ts = cfg["target_speed_kmh"]

    # Setup: Ego zu IDM, Mix anwenden, target speed setzen
    convert_ego_to_idm(env, target_speed_kmh=ts)

    # Ego grün färben (RGB)
    env.unwrapped.vehicle.color = (0, 255, 0)

    # Wichtig: Traffic-Mix reproduzierbar machen
    random.seed(episode_seed)
    apply_traffic_style_mix(env, p_normal=0.1982, p_def=0.6246, p_agg=0.1772)

    apply_global_target_speed(env, target_speed_kmh=ts)

    # -------------------------
    # WARM-UP: erste Sekunden NICHT werten, da sonst TTC häufig ganz am anfang nur durch zufälliges platzieren der Autos
    # -------------------------
    warmup_seconds = 2.0
    warmup_steps = int(warmup_seconds * cfg["policy_frequency"])

    w = 0
    while w < warmup_steps:
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)

        ego_crashed = getattr(env.unwrapped.vehicle, "crashed", False)

        if terminated or truncated or ego_crashed:
            # Warm-up Crash => Spawn-Artefakt => Episode verwerfen
            return None

        # optional: speed nachziehen
        if w % 20 == 0:
            apply_global_target_speed(env, target_speed_kmh=ts)

        w += 1

    # Cut-in-Erkennung erst nach Warm-up starten
    prev_lane = {}

    # Episode-Minima
    min_follow = float("inf")
    min_cutin = float("inf")
    min_any = float("inf")

    # NEU: DVO-Grenze passend zum schlimmsten Cut-in speichern
    min_cutin_dvo = float("inf")

    # Optional: ob jemals TTC_phys < TTC_DVO war
    episode_cutin_critical = 0

    # Wie viele Steps maximal? (Dauer * policy_frequency)
    steps_max = int(cfg["duration"] * cfg["policy_frequency"]) + 5

    # -------- Step-Schleife (ausgeschrieben) --------
    step = 0
    while step < steps_max:
        # Action ist egal (Ego ist IDM und ignoriert Action)
        action = env.action_space.sample()

        # Einen Step simulieren
        _, _, terminated, truncated, _ = env.step(action)

        # TTC berechnen
        ttc_follow, ttc_cutin_phys, ttc_cutin_dvo, cutin_crit = ego_ttc_step(
            env, prev_lane, beta=BETA, rho=RHO, tau=TAU
        )

        # Minima aktualisieren
        if ttc_follow < min_follow:
            min_follow = ttc_follow

        # Wenn wir einen neuen schlimmsten Cut-in finden, merken wir uns auch TTC_DVO dazu
        if ttc_cutin_phys < min_cutin:
            min_cutin = ttc_cutin_phys
            min_cutin_dvo = ttc_cutin_dvo

        # min_any ist Minimum aus beiden
        if ttc_follow < min_any:
            min_any = ttc_follow
        if ttc_cutin_phys < min_any:
            min_any = ttc_cutin_phys

        # Optional: Flag ob DVO jemals verletzt wurde
        if cutin_crit:
            episode_cutin_critical = 1

        # Optional: target_speed regelmäßig nachziehen
        if step % 20 == 0:
            apply_global_target_speed(env, target_speed_kmh=ts)

        # Episode fertig?
        if terminated or truncated:
            break

        step += 1

    # Episode-Flags für Schwellen (0/1)
    episode_below_1p5 = 0
    if min_any < ttc_thr_crit:
        episode_below_1p5 = 1

    episode_below_3p0 = 0
    if min_any < ttc_thr_warn:
        episode_below_3p0 = 1

    # Ergebnis-Dict zurückgeben (damit man später analysieren kann, welche cfg kritisch war)
    result = {
        "episode_seed": episode_seed,
        "lanes_count": cfg["lanes_count"],
        "vehicles_count": cfg["vehicles_count"],
        "vehicles_density": cfg["vehicles_density"],
        "target_speed_kmh": cfg["target_speed_kmh"],
        "duration": cfg["duration"],

        "min_follow": min_follow,
        "min_cutin": min_cutin,
        "min_cutin_dvo": min_cutin_dvo,                  
        "episode_cutin_critical": episode_cutin_critical, 

        "min_any": min_any,
        "episode_below_1p5": episode_below_1p5,
        "episode_below_3p0": episode_below_3p0,
    }
    return result



def replay_case(base_config, case, BETA=6.0, RHO=0.1, TAU=0.3,
                print_thr=3.0, sleep_s=0.03, stop_at_first_critical=False):
    """
    Rendert genau 1 Fall (case) und loggt TTCs, wenn sie klein werden.
    - print_thr: ab welcher TTC-Schwelle du Prints sehen willst (z.B. 3.0)
    - stop_at_first_critical: True -> stoppt beim ersten TTC < print_thr
    """
    cfg = build_cfg_from_case(base_config, case)
    episode_seed = case["episode_seed"]

    env = gym.make("highway-v0", render_mode="human")

    try:
        env.reset(seed=episode_seed, options={"config": cfg})

        ts = cfg["target_speed_kmh"]
        convert_ego_to_idm(env, target_speed_kmh=ts)
        apply_traffic_style_mix(env, p_normal=0.1982, p_def=0.6246, p_agg=0.1772)
        apply_global_target_speed(env, target_speed_kmh=ts)

        prev_lane = {}

        steps_max = int(cfg["duration"] * cfg["policy_frequency"]) + 5

        print("\n=== REPLAY START ===")
        print("episode_seed:", episode_seed)
        print("cfg:", {k: cfg[k] for k in ["lanes_count", "vehicles_count", "vehicles_density", "target_speed_kmh", "duration"]})
        print("min_any (aus MC):", case["min_any"])

        step = 0
        while step < steps_max:
            action = env.action_space.sample()  # wird ignoriert
            obs, reward, terminated, truncated, info = env.step(action)

            ttc_follow, ttc_cutin_phys, ttc_cutin_dvo, cutin_crit = ego_ttc_step(
                env, prev_lane, beta=BETA, rho=RHO, tau=TAU
            )

            # Wenn TTC klein wird, loggen
            if ttc_follow < print_thr:
                print("[step", step, "] FOLLOW TTC =", round(ttc_follow, 2), "s")

            if ttc_cutin_phys < print_thr:
                print("[step", step, "] CUTIN TTC_phys =", round(ttc_cutin_phys, 2),
                      "s | TTC_DVO =", round(ttc_cutin_dvo, 2),
                      "s | crit =", cutin_crit)

            # Optional: beim ersten kritischen Moment stoppen
            if stop_at_first_critical:
                if (ttc_follow < print_thr) or (ttc_cutin_phys < print_thr):
                    print(">>> STOP: Kritischer Moment erreicht (TTC < print_thr).")
                    input("Enter drücken zum Weiterlaufen...")
                    stop_at_first_critical = False  # danach normal weiter

            if step % 20 == 0:
                apply_global_target_speed(env, target_speed_kmh=ts)

            time.sleep(sleep_s)

            if terminated or truncated:
                print("=== REPLAY END (terminated/truncated) ===")
                break

            step += 1

    finally:
        env.close()


def replay_critical_cases(base_config, results, thr=1.5, max_cases=3):
    """
    Nimmt die MC-results, filtert kritische Fälle und rendert die schlimmsten zuerst.
    """
    # kritische Fälle sammeln
    critical = []
    i = 0
    while i < len(results):
        if results[i]["min_any"] < thr:
            critical.append(results[i])
        i += 1

    if len(critical) == 0:
        print("Keine kritischen Fälle gefunden für thr =", thr)
        return

    # nach min_any sortieren (schlimmste zuerst)
    critical_sorted = sorted(critical, key=lambda r: r["min_any"])

    print("\nKritische Fälle gefunden:", len(critical_sorted), "| zeige max:", max_cases)

    # rendern
    c = 0
    while c < max_cases and c < len(critical_sorted):
        case = critical_sorted[c]
        print("\n--- Render Case", c + 1, "von", max_cases, "---")
        replay_case(base_config, case, print_thr=3.0, sleep_s=0.03, stop_at_first_critical=True)

        # zwischen den Fällen Pause
        input("Nächster Fall? Enter drücken...")
        c += 1

def load_results(filename="mc_results.json"):
    import json
    with open(filename, "r") as f:
        return json.load(f)

def find_case_by_seed(results, episode_seed):
    i = 0
    while i < len(results):
        if results[i]["episode_seed"] == episode_seed:
            return results[i]
        i += 1
    return None



def build_cfg_from_case(base_config, case):
    """
    Nimmt deine base_config und überschreibt nur die Parameter,
    die in 'case' gespeichert sind.
    """
    import copy

def build_cfg_from_case(base_config, case):
    cfg = copy.deepcopy(base_config)

    # akzeptiert beide Formate:
    # 1) case["cfg"]["lanes_count"]  (dein IS-JSON)
    # 2) case["lanes_count"]         (altes/MC-Format)
    src = case.get("cfg", case)

    cfg["lanes_count"] = src["lanes_count"]
    cfg["vehicles_count"] = src["vehicles_count"]
    cfg["vehicles_density"] = src["vehicles_density"]
    cfg["target_speed_kmh"] = src["target_speed_kmh"]
    cfg["duration"] = src.get("duration", cfg.get("duration", 17))

    return cfg


def replay_case(base_config, case, BETA=6.0, RHO=0.1, TAU=0.3,
                print_thr=3.0, sleep_s=0.03, stop_at_first_critical=False):
    """
    Rendert genau 1 Fall (case) und loggt TTCs, wenn sie klein werden.
    - print_thr: ab welcher TTC-Schwelle du Prints sehen willst (z.B. 3.0)
    - stop_at_first_critical: True -> stoppt beim ersten TTC < print_thr
    """
    cfg = build_cfg_from_case(base_config, case)
    episode_seed = case["episode_seed"]

    env = gym.make("highway-v0", render_mode="human")

    try:
        env.reset(seed=episode_seed, options={"config": cfg})

        ts = cfg["target_speed_kmh"]
        convert_ego_to_idm(env, target_speed_kmh=ts)
        random.seed(episode_seed)
        apply_traffic_style_mix(env, p_normal=0.1982, p_def=0.6246, p_agg=0.1772)
        apply_global_target_speed(env, target_speed_kmh=ts)

        prev_lane = {}

        steps_max = int(cfg["duration"] * cfg["policy_frequency"]) + 5

        print("\n=== REPLAY START ===")
        print("episode_seed:", episode_seed)
        print("cfg:", {k: cfg[k] for k in ["lanes_count", "vehicles_count", "vehicles_density", "target_speed_kmh", "duration"]})
        #print("min_any (aus MC):", case["min_any"])
        # kompatibel: IS-Format (case["res"]["min_any"]) und altes MC-Format (case["min_any"])
        min_any_saved = None
        if "res" in case and isinstance(case["res"], dict):
            min_any_saved = case["res"].get("min_any", None)
        else:
            min_any_saved = case.get("min_any", None)

        print("min_any (gespeichert):", min_any_saved)


        step = 0
        while step < steps_max:
            action = env.action_space.sample()  # wird ignoriert
            obs, reward, terminated, truncated, info = env.step(action)

            ttc_follow, ttc_cutin_phys, ttc_cutin_dvo, cutin_crit = ego_ttc_step(
                env, prev_lane, beta=BETA, rho=RHO, tau=TAU
            )

            # Wenn TTC klein wird, loggen
            if ttc_follow < print_thr:
                print("[step", step, "] FOLLOW TTC =", round(ttc_follow, 2), "s")

            if ttc_cutin_phys < print_thr:
                print("[step", step, "] CUTIN TTC_phys =", round(ttc_cutin_phys, 2),
                      "s | TTC_DVO =", round(ttc_cutin_dvo, 2),
                      "s | crit =", cutin_crit)

            # Optional: beim ersten kritischen Moment stoppen
            if stop_at_first_critical:
                if (ttc_follow < print_thr) or (ttc_cutin_phys < print_thr):
                    print(">>> STOP: Kritischer Moment erreicht (TTC < print_thr).")
                    input("Enter drücken zum Weiterlaufen...")
                    stop_at_first_critical = False  # danach normal weiter

            if step % 20 == 0:
                apply_global_target_speed(env, target_speed_kmh=ts)

            time.sleep(sleep_s)

            if terminated or truncated:
                print("=== REPLAY END (terminated/truncated) ===")
                break

            step += 1

    finally:
        env.close()


def replay_critical_cases(base_config, results, thr=1.5, max_cases=3):
    """
    Nimmt die MC-results, filtert kritische Fälle und rendert die schlimmsten zuerst.
    """
    # kritische Fälle sammeln
    critical = []
    i = 0
    while i < len(results):
        if results[i]["min_any"] < thr:
            critical.append(results[i])
        i += 1

    if len(critical) == 0:
        print("Keine kritischen Fälle gefunden für thr =", thr)
        return

    # nach min_any sortieren (schlimmste zuerst)
    critical_sorted = sorted(critical, key=lambda r: r["min_any"])

    print("\nKritische Fälle gefunden:", len(critical_sorted), "| zeige max:", max_cases)

    # rendern
    c = 0
    while c < max_cases and c < len(critical_sorted):
        case = critical_sorted[c]
        print("\n--- Render Case", c + 1, "von", max_cases, "---")
        replay_case(base_config, case, print_thr=3.0, sleep_s=0.03, stop_at_first_critical=True)

        # zwischen den Fällen Pause
        input("Nächster Fall? Enter drücken...")
        c += 1

def load_results(filename="mc_results.json"):
    import json
    with open(filename, "r") as f:
        return json.load(f)

def find_case_by_seed(results, episode_seed):
    i = 0
    while i < len(results):
        if results[i]["episode_seed"] == episode_seed:
            return results[i]
        i += 1
    return None






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
    apply_traffic_style_mix(env, p_normal=0.1982, p_def=0.6246, p_agg=0.1772)
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
                apply_traffic_style_mix(env, p_normal=0.1982, p_def=0.6246, p_agg=0.1772)
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
