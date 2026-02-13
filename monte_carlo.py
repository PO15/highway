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

import json
import pandas as pd

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

def save_to_excel(results, filename="mc_results.xlsx"):
    # Erstellt aus der Liste von Dicts einen DataFrame (Tabelle)
    df = pd.DataFrame(results)
    
    # Speichert den DataFrame als Excel-Datei
    df.to_excel(filename, index=False)
    print(f"Ergebnisse erfolgreich in {filename} gespeichert!")



# ---------------------------------------------------------
# 3) Monte Carlo: viele Episoden laufen lassen + Summary
# ---------------------------------------------------------

def monte_carlo(base_config, N=10, seed=1, render=False,
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
        discarded = 0   # Anzahl verworfener Episoden (Crash im Warm-up)
        i = 0           # Zähler für "Versuche" (jede Simulation bekommt einen neuen seed)

        # Wir laufen so lange, bis wir wirklich N gültige Episoden gesammelt haben
        while len(results) < N:
            # 1) cfg ziehen
            cfg = sample_scenario_config(base_config, rng)

            # 2) eigener Episode-Seed
            #    -> jede Episode anders, aber reproduzierbar
            episode_seed = seed + i

            # 3) Episode laufen lassen
            #    -> falls Crash im Warm-up: run_one_episode gibt None zurück
            res = E.run_one_episode(
                env, cfg, episode_seed,
                BETA=6.0, RHO=0.1, TAU=0.3,
                ttc_thr_crit=ttc_thr_crit,
                ttc_thr_warn=ttc_thr_warn
            )

            # Wenn None: Episode verwerfen und NICHT zählen
            if res is None:
                discarded += 1
                i += 1
                continue

            # 4) Ergebnis speichern (nur gültige Episoden)
            results.append(res)

            # Optional: Fortschritt ausgeben
            if verbose_every != 0:
                # Anzeige über die Anzahl gültiger Episoden (len(results))
                if (len(results) - 1) % verbose_every == 0:
                    print("[Episode", len(results), "von", N, "] min_any =", round(res["min_any"], 2), "s")

            i += 1

    finally:
        env.close()

    # Hinweis ausgeben, wie viele Episoden verworfen wurden
    print("Verworfene Episoden (Crash im Warm-up):", discarded)

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





# ---------------------------------------------------------
# 4) Main
# ---------------------------------------------------------
def main():
    base_config = {
        "vehicles_count": 60,
        "duration": 17,
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
        N=10000,               # Anzahl Episoden
        seed=1,            # reproduzierbar
        render=False,       # True nur zum Anschauen (langsam)
        ttc_thr_crit=1.5,   # Schwellwert 1
        ttc_thr_warn=3.0,   # Schwellwert 2
        verbose_every=1
             # z.B. 1 oder 5, wenn du Fortschritt sehen willst
    )


    # Ergebnisse speichern
    with open("mc_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Gespeichert: mc_results.json")

    save_to_excel(results)

if __name__ == "__main__":
    main()
