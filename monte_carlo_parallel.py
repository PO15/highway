"""
monte_carlo_parallel.py

Parallelisierte Monte-Carlo-Auswertung für highway-env (macOS geeignet).

Voraussetzung:
- Environment.py liegt im gleichen Ordner und stellt E.run_one_episode(env, cfg, seed, ...) bereit
- highway_env ist installiert (pip install highway-env)
"""

import os
import copy
import json
import random
import statistics
import multiprocessing as mp

import gymnasium as gym
import highway_env  # registriert env-ids

import Environment as E  # <- DEINE Environment.py

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


def save_to_excel(results, filename="mc_results_parallel.xlsx"):
    df = pd.DataFrame(results)
    df.to_excel(filename, index=False)
    print(f"Ergebnisse erfolgreich in {filename} gespeichert!")


# ---------------------------------------------------------
# 2) Worker: läuft in separatem Prozess (Env wird hier erzeugt)
# ---------------------------------------------------------
def _worker_run_one(job):
    """
    job = (cfg, episode_seed, ttc_thr_crit, ttc_thr_warn, BETA, RHO, TAU)
    """
    cfg, episode_seed, ttc_thr_crit, ttc_thr_warn, BETA, RHO, TAU = job

    env = gym.make("highway-v0", render_mode=None)
    try:
        res = E.run_one_episode(
            env, cfg, episode_seed,
            BETA=BETA, RHO=RHO, TAU=TAU,
            ttc_thr_crit=ttc_thr_crit,
            ttc_thr_warn=ttc_thr_warn
        )
        return res
    finally:
        env.close()


# ---------------------------------------------------------
# 3) Parallel Monte Carlo: viele Episoden + Summary
# ---------------------------------------------------------
def monte_carlo_parallel(base_config, N=10, seed=1,
                         ttc_thr_crit=1.5, ttc_thr_warn=3.0,
                         BETA=6.0, RHO=0.1, TAU=0.3,
                         n_workers=None, batch_size=None,
                         verbose_every=0):
    """
    Parallelisierte Version der monte_carlo() Funktion.

    Wichtige Eigenschaften:
    - Jeder Versuch i bekommt episode_seed = seed + i  (keine doppelten Seeds)
    - Sampling (cfg) passiert NUR im Main-Prozess => reproduzierbar
    - Wir sammeln die ersten N gültigen Episoden (res != None), wie bei dir seriell.
      (Im letzten Batch laufen evtl. ein paar extra Versuche; die werden ignoriert.)
    """

    # macOS: spawn + __main__ ist Pflicht
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    if n_workers is None:
        n_workers = min(os.cpu_count() or 4, 8)

    if batch_size is None:
        # sinnvolle Default-Größe: 2 Jobs pro Worker
        batch_size = max(n_workers * 2, 2)

    rng = random.Random(seed)

    results = []
    discarded = 0
    i = 0  # Versuchszähler => episode_seed = seed + i

    with mp.Pool(processes=n_workers) as pool:
        while len(results) < N:
            jobs = []

            # Batch vorbereiten (Seeds eindeutig + fortlaufend)
            b = 0
            while b < batch_size:
                cfg = sample_scenario_config(base_config, rng)
                episode_seed = seed + i
                i += 1

                jobs.append((cfg, episode_seed, ttc_thr_crit, ttc_thr_warn, BETA, RHO, TAU))
                b += 1

            batch_res = pool.map(_worker_run_one, jobs)

            # In Reihenfolge auswerten (wie seriell)
            for res in batch_res:
                if res is None:
                    discarded += 1
                else:
                    results.append(res)

                    if verbose_every and (len(results) % verbose_every == 0):
                        print("[Episode", len(results), "von", N, "] min_any =",
                              round(res["min_any"], 2), "s")

                if len(results) >= N:
                    break

    print("Verworfene Episoden (Crash im Warm-up):", discarded)

    # -------- Summary (ähnlich wie in deinem Code) --------
    min_any_vals = []
    inf_count = 0

    for r in results:
        v = r["min_any"]
        min_any_vals.append(v)
        if v == float("inf"):
            inf_count += 1

    # Rate < 1.5
    count_crit = 0
    for r in results:
        if r.get("episode_below_1p5", 0) == 1:
            count_crit += 1
    rate_crit = count_crit / len(results)

    # Rate < 3.0
    count_warn = 0
    for r in results:
        if r.get("episode_below_3p0", 0) == 1:
            count_warn += 1
    rate_warn = count_warn / len(results)

    finite = []
    for v in min_any_vals:
        if v != float("inf"):
            finite.append(v)

    print("\n=== MONTE CARLO SUMMARY (PARALLEL) ===")
    print("N:", len(results))
    print("Rate (min_any <", ttc_thr_crit, "s):", round(rate_crit * 100, 1), "%")
    print("Rate (min_any <", ttc_thr_warn, "s):", round(rate_warn * 100, 1), "%")
    print("Episodes mit min_any=inf:", inf_count, "(", round(inf_count / len(results) * 100, 1), "% )")

    if len(finite) > 0:
        print("min_any (finite) mean:", round(statistics.mean(finite), 2),
              "| median:", round(statistics.median(finite), 2),
              "| min:", round(min(finite), 2))
    else:
        print("Alle Episoden min_any=inf (keine relevanten Interaktionen).")

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
# 4) Main (optional): direkt ausführbar
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



    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("outputs", f"run_{run_tag}")
    os.makedirs(out_dir, exist_ok=True)

    params_path = os.path.join(out_dir, "run_params.json")

    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(
            {"base_config": base_config, "ce_params": ce_params, "is_params": is_params},
            f, indent=2, ensure_ascii=False
        )
    print("Gespeichert:", params_path)


    results = monte_carlo_parallel(
        base_config=base_config,
        N=20000,
        seed=1,
        ttc_thr_crit=1.5,
        ttc_thr_warn=3.0,
        n_workers=None,
        batch_size=None,
        verbose_every=50
    )

    with open("mc_results_parallel.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Gespeichert: mc_results_parallel.json")

    save_to_excel(results, filename="mc_results_parallel.xlsx")


if __name__ == "__main__":
    main()
