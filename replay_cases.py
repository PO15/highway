# replay_cases.py
# Lädt Monte-Carlo-Ergebnisse aus mc_results.json und rendert ausgewählte Cases

import json
import copy

import gymnasium as gym
import highway_env  # registriert env-ids

import Environment as E  # enthält replay_case(...)


def load_results(filename="mc_results.json"):
    with open(filename, "r") as f:
        return json.load(f)


def find_case_by_seed(results, episode_seed):
    i = 0
    while i < len(results):
        if results[i]["episode_seed"] == episode_seed:
            return results[i]
        i += 1
    return None


def pick_worst_case(results):
    # schlimmster = kleinster min_any
    worst = min(results, key=lambda r: r["min_any"])
    return worst


def build_base_config():
    # Muss zu deiner ursprünglichen base_config passen
    return {
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


def main():
    base_config = build_base_config()

    results = load_results("/Users/philippotto/Desktop/highway/outputs/is_run_20260204_011027/is_episodes.json")
    print("Geladene Ergebnisse:", len(results))

    # -------- Option A: per episode_seed auswählen --------
    seed_to_watch = 1078  # <- hier ändern
    case = find_case_by_seed(results, seed_to_watch)

    

    if case is None:
        print("Case nicht gefunden!")
        return

    print("\nCase zum Rendern:")
    print(case)

    # Rendern
    # print_thr=3.0: loggt TTC < 3.0
    # stop_at_first_critical=False: nicht anhalten
    E.replay_case(
        base_config,
        case,
        print_thr=3.0,
        sleep_s=0.03,
        stop_at_first_critical=False
    )


if __name__ == "__main__":
    main()
