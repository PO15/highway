import os
import copy
import json
import random
import multiprocessing as mp
from datetime import datetime

import gymnasium as gym
import highway_env  # registriert env-ids

import pandas as pd
import Environment as E


def sample_scenario_config(base_config, rng):
    cfg = copy.deepcopy(base_config)

    lanes_values = [2, 3, 4]
    lanes_weights = [57, 42, 1]
    cfg["lanes_count"] = rng.choices(lanes_values, weights=lanes_weights, k=1)[0]

    cfg["vehicles_count"] = rng.randint(30, 80)
    cfg["vehicles_density"] = rng.uniform(1.0, 3.0)

    speed_means = [60.0, 85.0, 99.7, 112.0, 117.3, 124.1]
    weights     = [0.8, 12.0, 7.1, 11.6, 4.3, 64.3]

    cfg["target_speed_kmh"] = rng.choices(speed_means, weights=weights, k=1)[0]
    cfg["init_speed_kmh"]   = rng.choices(speed_means, weights=weights, k=1)[0]

    cfg["duration"] = 17
    return cfg


def save_to_excel(results, filename="mc_results_parallel.xlsx"):
    df = pd.DataFrame(results)
    df.to_excel(filename, index=False)
    print(f"Ergebnisse erfolgreich in {filename} gespeichert!")


def _worker_run_one(job):
    cfg, episode_seed, ttc_thr_crit, ttc_thr_warn, BETA, RHO, TAU = job
    env = gym.make("highway-v0", render_mode=None)
    try:
        return E.run_one_episode(
            env, cfg, episode_seed,
            BETA=BETA, RHO=RHO, TAU=TAU,
            ttc_thr_crit=ttc_thr_crit,
            ttc_thr_warn=ttc_thr_warn
        )
    finally:
        env.close()


def monte_carlo_parallel(base_config, N=10000, seed=1,
                         ttc_thr_crit=1.5, ttc_thr_warn=3.0,
                         BETA=6.0, RHO=0.1, TAU=0.3,
                         n_workers=None, batch_size=None,
                         verbose_every=50):

    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
    if batch_size is None:
        batch_size = 100 * n_workers

    print("Workers:", n_workers, "| batch_size:", batch_size)

    rng = random.Random(seed)
    results = []
    i = 0

    with mp.Pool(processes=n_workers) as pool:
        while len(results) < N:
            jobs = []
            b = 0
            while b < batch_size and len(results) + len(jobs) < N:
                cfg = sample_scenario_config(base_config, rng)
                episode_seed = seed + i
                i += 1
                jobs.append((cfg, episode_seed, ttc_thr_crit, ttc_thr_warn, BETA, RHO, TAU))
                b += 1

            batch_out = pool.map(_worker_run_one, jobs)

            for res in batch_out:
                if res is None:
                    continue
                results.append(res)
                if verbose_every and (len(results) % verbose_every == 0):
                    print(f"[Episode {len(results)} von {N}] min_any = {res['min_any']:.2f} s")

    return results


if __name__ == "__main__":
    base_config = {
        "vehicles_count": 60,
        "duration": 17,
        "policy_frequency": 10,
        "simulation_frequency": 15,
        "lanes_count": 3,
        "ego_spacing": 1,
        "vehicles_density": 2.0,

        "target_speed_kmh": 124.1,
        "init_speed_kmh": 85.0,

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
        json.dump({"base_config": base_config}, f, indent=2, ensure_ascii=False)
    print("Gespeichert:", params_path)

    results = monte_carlo_parallel(
        base_config=base_config,
        N=20,
        seed=1,
        ttc_thr_crit=1.5,
        ttc_thr_warn=3.0,
        n_workers=None,
        batch_size=None,
        verbose_every=10
    )

    json_path = os.path.join(out_dir, "mc_results_parallel.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("Gespeichert:", json_path)

    xlsx_path = os.path.join(out_dir, "mc_results_parallel.xlsx")
    save_to_excel(results, filename=xlsx_path)
