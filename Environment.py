import time
import random
import copy
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
    if gap <= 0:
        return 0.0
    if closing_speed <= 0:
        return float("inf")
    return gap / closing_speed


# -----------------------------
# Ego TTC: follow + cut-in
# -----------------------------
def ego_ttc_step(env, prev_lane, beta=6.0, rho=0.1, tau=0.3):
    """
    Liefert:
      - ttc_follow
      - ttc_cutin_phys
      - ttc_cutin_dvo (tau Verzögerung)
      - cutin_critical Flag
    """
    uw = env.unwrapped
    ego = uw.vehicle
    ego_lane = getattr(ego, "lane_index", None)
    ego_speed = get_speed(ego)

    ttc_follow = float("inf")
    ttc_cutin_phys = float("inf")
    ttc_cutin_dvo = float("inf")
    cutin_critical = False

    # FOLLOW: nächstes Fahrzeug vorne in gleicher Spur
    front = None
    min_dx = float("inf")
    for v in uw.road.vehicles:
        if v is ego:
            continue
        if getattr(v, "lane_index", None) != ego_lane:
            continue
        dx = float(v.position[0] - ego.position[0])
        if dx > 0 and dx < min_dx:
            min_dx = dx
            front = v

    if front is not None:
        gap = get_bumper_gap(ego, front)
        closing_speed = ego_speed - get_speed(front)
        ttc_follow = compute_ttc(gap, closing_speed)

    # CUT-IN: Spurwechsel von anderer Spur -> ego_spur
    for v in uw.road.vehicles:
        if v is ego:
            continue

        v_id = id(v)
        lane_now = getattr(v, "lane_index", None)
        lane_prev = prev_lane.get(v_id, lane_now)
        prev_lane[v_id] = lane_now

        if lane_prev != ego_lane and lane_now == ego_lane:
            dx = float(v.position[0] - ego.position[0])

            # Kandidat nach dem Wechsel nicht deutlich hinter Ego
            if dx > -5.0:
                gap = get_bumper_gap(ego, v)
                closing_speed = ego_speed - get_speed(v)

                ttc_phys = compute_ttc(gap, closing_speed)
                if ttc_phys < ttc_cutin_phys:
                    ttc_cutin_phys = ttc_phys

                # DVO Näherung: ego fährt tau Sekunden "blind"
                gap_tau = gap - closing_speed * tau
                ttc_dvo = compute_ttc(gap_tau, closing_speed)
                if ttc_dvo < ttc_cutin_dvo:
                    ttc_cutin_dvo = ttc_dvo

                if ttc_dvo < beta * (1.0 - rho):
                    cutin_critical = True

    return ttc_follow, ttc_cutin_phys, ttc_cutin_dvo, cutin_critical


# ---------------------------------------------------------
# Ego -> NormalIDMVehicle (mit create_from, versionsrobust)
# ---------------------------------------------------------
def convert_ego_to_idm(env):
    uw = env.unwrapped
    ego = uw.vehicle
    if isinstance(ego, NormalIDMVehicle):
        return

    new_ego = NormalIDMVehicle.create_from(ego)

    # sicherheitshalber target_speed setzen, falls nicht vorhanden
    if not hasattr(new_ego, "target_speed"):
        new_ego.target_speed = getattr(new_ego, "speed", 0.0)

    new_ego.color = getattr(ego, "color", (0, 255, 0))
    new_ego.crashed = getattr(ego, "crashed", False)

    uw.road.vehicles = [new_ego if v is ego else v for v in uw.road.vehicles]
    uw.vehicle = new_ego


# ---------------------------------------------------------
# Desired speed (target_speed) pro Style setzen
# ---------------------------------------------------------
def apply_target_speed_with_style(env, target_speed_kmh, clip_min_kmh=60, clip_max_kmh=200):
    uw = env.unwrapped
    for v in getattr(uw.road, "vehicles", []):
        if not hasattr(v, "target_speed"):
            continue
        offset = getattr(v.__class__, "SPEED_OFFSET_KMH", 0)
        v_des_kmh = float(target_speed_kmh) + float(offset)
        v_des_kmh = max(clip_min_kmh, min(clip_max_kmh, v_des_kmh))
        v.target_speed = v_des_kmh / 3.6


# ---------------------------------------------------------
# Startgeschwindigkeit pro Style setzen
# ---------------------------------------------------------
def apply_init_speed_with_style(env, init_speed_kmh, clip_min_kmh=0, clip_max_kmh=200):
    """
    v0_kmh = init_speed_kmh + VehicleClass.SPEED_OFFSET_KMH
    Muss NACH apply_traffic_style_mix() kommen.
    """
    uw = env.unwrapped
    for v in getattr(uw.road, "vehicles", []):
        if not hasattr(v, "speed"):
            continue
        offset = getattr(v.__class__, "SPEED_OFFSET_KMH", 0)
        v0_kmh = float(init_speed_kmh) + float(offset)
        v0_kmh = max(clip_min_kmh, min(clip_max_kmh, v0_kmh))
        v.speed = v0_kmh / 3.6


# ---------------------------------------------------------
# Traffic-Mix (normal/def/agg) setzen (mit create_from)
# ---------------------------------------------------------
def apply_traffic_style_mix(env, p_normal=0.1982, p_def=0.6246, p_agg=0.1772):
    uw = env.unwrapped
    ego = uw.vehicle

    traffic_idx = [i for i, v in enumerate(uw.road.vehicles) if v is not ego]
    n = len(traffic_idx)
    if n == 0:
        return

    n_def = int(round(p_def * n))
    n_agg = int(round(p_agg * n))
    n_norm = n - n_def - n_agg

    styles = (["def"] * n_def) + (["agg"] * n_agg) + (["norm"] * n_norm)
    random.shuffle(styles)

    for idx, style in zip(traffic_idx, styles):
        v = uw.road.vehicles[idx]

        if style == "def":
            nv = DefensiveIDMVehicle.create_from(v)
            nv.color = (80, 160, 255)     # blau
        elif style == "agg":
            nv = AggressiveIDMVehicle.create_from(v)
            nv.color = (255, 90, 90)      # rot
        else:
            nv = NormalIDMVehicle.create_from(v)
            nv.color = (220, 220, 220)    # hellgrau/weiß

        # Farben/Crash übernehmen
       # nv.color = getattr(v, "color", (255, 255, 255))
        nv.crashed = getattr(v, "crashed", False)

        # target_speed falls nötig
        if hasattr(nv, "target_speed"):
            nv.target_speed = getattr(v, "target_speed", getattr(v, "speed", 0.0))

        uw.road.vehicles[idx] = nv


# ---------------------------------------------------------
# Eine Episode laufen lassen
# ---------------------------------------------------------
def run_one_episode(env, cfg, episode_seed,
                    BETA=6.0, RHO=0.1, TAU=0.3,
                    ttc_thr_crit=1.5, ttc_thr_warn=3.0):

    env.reset(seed=episode_seed, options={"config": cfg})

    target_speed_kmh = cfg.get("target_speed_kmh", cfg.get("speed_limit_kmh", 130))
    init_speed_kmh = cfg.get("init_speed_kmh", target_speed_kmh)

    # Ego IDM + Traffic Mix
    convert_ego_to_idm(env)
    env.unwrapped.vehicle.color = (0, 255, 0)

    random.seed(episode_seed)
    apply_traffic_style_mix(env, p_normal=0.1982, p_def=0.6246, p_agg=0.1772)

    # erst init_speed (Startzustand), dann target_speed (desired speed)
    apply_init_speed_with_style(env, init_speed_kmh=init_speed_kmh)
    apply_target_speed_with_style(env, target_speed_kmh=target_speed_kmh)

    # Warm-up (Crash => verwerfen)
    warmup_seconds = 2.0
    warmup_steps = int(warmup_seconds * cfg["policy_frequency"])

    for w in range(warmup_steps):
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)

        ego_crashed = getattr(env.unwrapped.vehicle, "crashed", False)
        if terminated or truncated or ego_crashed:
            return None

        if w % 20 == 0:
            apply_target_speed_with_style(env, target_speed_kmh=target_speed_kmh)

    prev_lane = {}

    min_follow = float("inf")
    min_cutin = float("inf")
    min_cutin_dvo = float("inf")
    min_any = float("inf")

    episode_cutin_critical = 0
    episode_below_1p5 = 0
    episode_below_3p0 = 0

    steps_max = int(cfg["duration"] * cfg["policy_frequency"]) + 5

    for step in range(steps_max):
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)

        ttc_follow, ttc_cutin_phys, ttc_cutin_dvo, cutin_crit = ego_ttc_step(
            env, prev_lane, beta=BETA, rho=RHO, tau=TAU
        )

        min_follow = min(min_follow, ttc_follow)
        min_cutin = min(min_cutin, ttc_cutin_phys)
        min_cutin_dvo = min(min_cutin_dvo, ttc_cutin_dvo)

        ttc_any = min(ttc_follow, ttc_cutin_phys)
        min_any = min(min_any, ttc_any)

        if cutin_crit:
            episode_cutin_critical = 1

        if ttc_any < ttc_thr_crit:
            episode_below_1p5 = 1
        if ttc_any < ttc_thr_warn:
            episode_below_3p0 = 1

        if step % 20 == 0:
            apply_target_speed_with_style(env, target_speed_kmh=target_speed_kmh)

        ego_crashed = getattr(env.unwrapped.vehicle, "crashed", False)
        if terminated or truncated or ego_crashed:
            break

    return {
        "episode_seed": episode_seed,
        "lanes_count": cfg["lanes_count"],
        "vehicles_count": cfg["vehicles_count"],
        "vehicles_density": cfg["vehicles_density"],
        "init_speed_kmh": init_speed_kmh,
        "target_speed_kmh": target_speed_kmh,
        "duration": cfg["duration"],

        "min_follow": min_follow,
        "min_cutin": min_cutin,
        "min_cutin_dvo": min_cutin_dvo,
        "episode_cutin_critical": episode_cutin_critical,

        "min_any": min_any,
        "episode_below_1p5": episode_below_1p5,
        "episode_below_3p0": episode_below_3p0,
    }


# ---------------------------------------------------------
# Replay / Helpers
# ---------------------------------------------------------
def build_cfg_from_case(base_config, case):
    cfg = copy.deepcopy(base_config)
    src = case.get("cfg", case)

    cfg["lanes_count"] = src["lanes_count"]
    cfg["vehicles_count"] = src["vehicles_count"]
    cfg["vehicles_density"] = src["vehicles_density"]

    if "target_speed_kmh" in src:
        cfg["target_speed_kmh"] = src["target_speed_kmh"]
    elif "speed_limit_kmh" in src:
        cfg["target_speed_kmh"] = src["speed_limit_kmh"]

    if "init_speed_kmh" in src:
        cfg["init_speed_kmh"] = src["init_speed_kmh"]
    else:
        cfg["init_speed_kmh"] = cfg.get("target_speed_kmh", 130)

    cfg["duration"] = src.get("duration", cfg.get("duration", 17))
    return cfg


def replay_case(base_config, case, BETA=6.0, RHO=0.1, TAU=0.3,
                print_thr=3.0, sleep_s=0.03, stop_at_first_critical=False):

    cfg = build_cfg_from_case(base_config, case)
    episode_seed = case["episode_seed"]

    env = gym.make("highway-v0", render_mode="human")
    try:
        env.reset(seed=episode_seed, options={"config": cfg})

        target_speed_kmh = cfg.get("target_speed_kmh", cfg.get("speed_limit_kmh", 130))
        init_speed_kmh = cfg.get("init_speed_kmh", target_speed_kmh)

        convert_ego_to_idm(env)
        random.seed(episode_seed)
        apply_traffic_style_mix(env, p_normal=0.1982, p_def=0.6246, p_agg=0.1772)
        apply_init_speed_with_style(env, init_speed_kmh=init_speed_kmh)
        apply_target_speed_with_style(env, target_speed_kmh=target_speed_kmh)

        prev_lane = {}
        steps_max = int(cfg["duration"] * cfg["policy_frequency"]) + 5

        print("\n=== REPLAY START ===")
        print("episode_seed:", episode_seed)
        print("cfg:", {k: cfg.get(k) for k in ["lanes_count", "vehicles_count", "vehicles_density",
                                             "init_speed_kmh", "target_speed_kmh", "duration"]})

        min_any_saved = case.get("min_any", None)
        if "res" in case and isinstance(case["res"], dict):
            min_any_saved = case["res"].get("min_any", min_any_saved)
        print("min_any (gespeichert):", min_any_saved)

        for step in range(steps_max):
            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)

            ttc_follow, ttc_cutin_phys, ttc_cutin_dvo, cutin_crit = ego_ttc_step(
                env, prev_lane, beta=BETA, rho=RHO, tau=TAU
            )

            if ttc_follow < print_thr:
                print(f"[step {step}] FOLLOW TTC = {ttc_follow:.2f} s")

            if ttc_cutin_phys < print_thr:
                print(f"[step {step}] CUTIN TTC_phys = {ttc_cutin_phys:.2f} s | TTC_DVO = {ttc_cutin_dvo:.2f} s | crit = {cutin_crit}")

            if stop_at_first_critical and ((ttc_follow < print_thr) or (ttc_cutin_phys < print_thr)):
                print(">>> STOP: Kritischer Moment erreicht (TTC < print_thr).")
                input("Enter drücken zum Weiterlaufen...")
                stop_at_first_critical = False

            if step % 20 == 0:
                apply_target_speed_with_style(env, target_speed_kmh=target_speed_kmh)

            time.sleep(sleep_s)

            if terminated or truncated:
                print("=== REPLAY END (terminated/truncated) ===")
                break

    finally:
        env.close()
