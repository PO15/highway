# Voraussetzung: Du hast eine Environment.py im gleichen Ordner mit:
# - convert_ego_to_idm(env, target_speed_kmh=...)
# - apply_traffic_style_mix(env, ...)
# - apply_global_target_speed(env, target_speed_kmh=...)
# - ego_ttc_step(env, prev_lane, beta=..., rho=..., tau=...)

import copy
import random
import statistics
import gymnasium as gym
import highway_env  # registriert env-ids

import Environment as E  # <- DEINE Environment.py
import time
import copy
import gymnasium as gym
import highway_env
import Environment as E

# ---------------------------------------------------------
# 1) Zufällige Szenario-Config für EINE Episode erzeugen
# ---------------------------------------------------------
def sample_scenario_config(base_config, rng):
    """
    Monte Carlo Sampling:
    Wir nehmen die base_config und überschreiben ein paar Werte zufällig.
    """
    cfg = copy.deepcopy(base_config)

    cfg["lanes_count"] = rng.choice([2, 3, 4])
    cfg["vehicles_count"] = rng.randint(30, 80)
    cfg["vehicles_density"] = rng.uniform(1.0, 3.0)
    cfg["target_speed_kmh"] = rng.choice([90, 110, 120, 130, 150, 160])
    cfg["duration"] = 17

    return cfg


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
    E.convert_ego_to_idm(env, target_speed_kmh=ts)
    # Ego grün färben (RGB)
    env.unwrapped.vehicle.color = (0, 255, 0)

    E.apply_traffic_style_mix(env, p_normal=0.70, p_def=0.15, p_agg=0.15)
    E.apply_global_target_speed(env, target_speed_kmh=ts)

    # -------------------------
    # WARM-UP: erste Sekunden NICHT werten, da sonst TTC häufig ganz am anfang nur durch zufälliges platzieren der Autos
    # -------------------------
    warmup_seconds = 2.0
    warmup_steps = int(warmup_seconds * cfg["policy_frequency"])

    w = 0
    while w < warmup_steps:
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)

        # optional: speed nachziehen
        if w % 20 == 0:
            E.apply_global_target_speed(env, target_speed_kmh=ts)

        if terminated or truncated:
            break

        w += 1

    # Cut-in-Erkennung erst nach Warm-up starten
    prev_lane = {}


    # Damit wir Cut-ins erkennen (vorherige Spur speichern)
    prev_lane = {}

    # Episode-Minima
    min_follow = float("inf")
    min_cutin = float("inf")
    min_any = float("inf")

    # Wie viele Steps maximal? (Dauer * policy_frequency)
    steps_max = int(cfg["duration"] * cfg["policy_frequency"]) + 5

    # -------- Step-Schleife (ausgeschrieben) --------
    step = 0
    while step < steps_max:
        # Action ist egal (Ego ist IDM und ignoriert Action)
        action = env.action_space.sample()

        # Einen Step simulieren
        obs, reward, terminated, truncated, info = env.step(action)

        # TTC berechnen
        ttc_follow, ttc_cutin_phys, ttc_cutin_dvo, cutin_crit = E.ego_ttc_step(
            env, prev_lane, beta=BETA, rho=RHO, tau=TAU
        )

        # Minima aktualisieren (ausgeschrieben, ohne min(...) Tricks)
        if ttc_follow < min_follow:
            min_follow = ttc_follow

        if ttc_cutin_phys < min_cutin:
            min_cutin = ttc_cutin_phys

        # min_any ist Minimum aus beiden
        if ttc_follow < min_any:
            min_any = ttc_follow
        if ttc_cutin_phys < min_any:
            min_any = ttc_cutin_phys

        # Optional: target_speed regelmäßig nachziehen
        if step % 20 == 0:
            E.apply_global_target_speed(env, target_speed_kmh=ts)

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
        "min_any": min_any,
        "episode_below_1p5": episode_below_1p5,
        "episode_below_3p0": episode_below_3p0,
    }
    return result


# ---------------------------------------------------------
# 3) Monte Carlo: viele Episoden laufen lassen + Summary
# ---------------------------------------------------------

def monte_carlo(base_config, N=10, seed=42, render=False,
                ttc_thr_crit=1.5, ttc_thr_warn=3.0, verbose_every=1):
    """
    N Episoden:
    - cfg samplen
    - Episode laufen lassen
    - Ergebnisse sammeln
    - Rates/Statistiken berechnen
    """

    rng = random.Random(seed)

    # Env erstellen (render nur zum Debuggen)
    if render:
        env = gym.make("highway-v0", render_mode="human")
    else:
        env = gym.make("highway-v0", render_mode=None)

    results = []

    try:
        # -------- Episoden-Schleife (ausgeschrieben) --------
        i = 0
        while i < N:
            # 1) cfg ziehen
            cfg = sample_scenario_config(base_config, rng)

            # 2) eigener Episode-Seed
            #    -> jede Episode anders, aber reproduzierbar
            episode_seed = seed * 100000 + i

            # 3) Episode laufen lassen
            res = run_one_episode(
                env, cfg, episode_seed,
                BETA=6.0, RHO=0.1, TAU=0.3,
                ttc_thr_crit=ttc_thr_crit,
                ttc_thr_warn=ttc_thr_warn
            )

            # 4) Ergebnis speichern
            results.append(res)

            # Optional: Fortschritt ausgeben
            if verbose_every != 0:
                if i % verbose_every == 0:
                    print("[Episode", i, "von", N, "] min_any =", round(res["min_any"], 2), "s")

            i += 1

    finally:
        env.close()

    # -------- Summary berechnen (ohne Comprehensions) --------
    min_any_vals = []
    inf_count = 0

    j = 0
    while j < len(results):
        v = results[j]["min_any"]
        min_any_vals.append(v)
        if v == float("inf"):
            inf_count += 1
        j += 1

    # Rate < 1.5
    count_1p5 = 0
    k = 0
    while k < len(results):
        if results[k]["episode_below_1p5"] == 1:
            count_1p5 += 1
        k += 1
    rate_1p5 = count_1p5 / len(results)

    # Rate < 3.0
    count_3p0 = 0
    k = 0
    while k < len(results):
        if results[k]["episode_below_3p0"] == 1:
            count_3p0 += 1
        k += 1
    rate_3p0 = count_3p0 / len(results)

    # Liste ohne inf
    finite = []
    m = 0
    while m < len(min_any_vals):
        if min_any_vals[m] != float("inf"):
            finite.append(min_any_vals[m])
        m += 1

    print("\n=== MONTE CARLO SUMMARY ===")
    print("N:", len(results))
    print("Rate (min_any <", ttc_thr_crit, "s):", round(rate_1p5 * 100, 1), "%")
    print("Rate (min_any <", ttc_thr_warn, "s):", round(rate_3p0 * 100, 1), "%")
    print("Episodes mit min_any=inf:", inf_count, "(", round(inf_count / len(results) * 100, 1), "% )")

    if len(finite) > 0:
        print("min_any (finite) mean:", round(statistics.mean(finite), 2),
              "| median:", round(statistics.median(finite), 2),
              "| min:", round(min(finite), 2))
    else:
        print("Alle Episoden min_any=inf (keine relevanten Interaktionen).")

    # Worst 5 Cases (einfach: sortieren nach min_any)
    worst = sorted(results, key=lambda r: r["min_any"])
    print("\nWorst 5 cases (by min_any):")
    t = 0
    while t < 5 and t < len(worst):
        print(worst[t])
        t += 1

    return results




def build_cfg_from_case(base_config, case):
    """
    Nimmt deine base_config und überschreibt nur die Parameter,
    die in 'case' gespeichert sind.
    """
    cfg = copy.deepcopy(base_config)

    cfg["lanes_count"] = case["lanes_count"]
    cfg["vehicles_count"] = case["vehicles_count"]
    cfg["vehicles_density"] = case["vehicles_density"]
    cfg["target_speed_kmh"] = case["target_speed_kmh"]
    cfg["duration"] = case["duration"]

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
        E.convert_ego_to_idm(env, target_speed_kmh=ts)
        E.apply_traffic_style_mix(env, p_normal=0.70, p_def=0.15, p_agg=0.15)
        E.apply_global_target_speed(env, target_speed_kmh=ts)

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

            ttc_follow, ttc_cutin_phys, ttc_cutin_dvo, cutin_crit = E.ego_ttc_step(
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
                E.apply_global_target_speed(env, target_speed_kmh=ts)

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



# ---------------------------------------------------------
# 4) Main
# ---------------------------------------------------------
def main():
    base_config = {
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

    results = monte_carlo(
        base_config=base_config,
        N=100,               # Anzahl Episoden
        seed=42,            # reproduzierbar
        render=False,       # True nur zum Anschauen (langsam)
        ttc_thr_crit=1.5,   # Schwellwert 1
        ttc_thr_warn=3.0,   # Schwellwert 2
        verbose_every=0     # z.B. 1 oder 5, wenn du Fortschritt sehen willst
    )

    # Render die schlimmsten kritischen Fälle (z.B. TTC < 1.5s)
    replay_critical_cases(base_config, results, thr=3, max_cases=3)



if __name__ == "__main__":
    main()
