# =========================================================
# Latin Hypercube Sampling (LHS) für deine Scenario-Configs
# - LHS für kontinuierliche Parameter (vehicles_count, vehicles_density)
# - Balancierte Zuordnung für diskrete/kategoriale Parameter (lanes_count, target_speed_kmh)
# - gleiche Lauf-/Discard-Logik wie bei Monte Carlo
# =========================================================

import copy
import random
import statistics
import gymnasium as gym
import highway_env
import Environment as E
import json
import pandas as pd

# ---------------------------------------------------------
# LHS: Unit-Samples in [0,1] erzeugen (Random-LHS oder Centered/Median-LHS)
# ---------------------------------------------------------
def lhs_unit(n, d, rng, centered=False):
    """
    Erzeugt eine LHS-Stichprobe X der Größe n x d im Einheitswürfel [0,1]^d.
    centered=False: Random-LHS (zufälliger Punkt je Stratum)
    centered=True : Centered/Median-LHS (Mitte je Stratum)
    """
    # X initialisieren
    X = []
    i = 0
    while i < n:
        row = []
        j = 0
        while j < d:
            row.append(0.0)
            j += 1
        X.append(row)
        i += 1

    # pro Dimension Strata erzeugen und permutieren
    j = 0
    while j < d:
        strata = []
        i = 0
        while i < n:
            if centered:
                u = (i + 0.5) / n
            else:
                u = (i + rng.random()) / n
            strata.append(u)
            i += 1

        rng.shuffle(strata)

        i = 0
        while i < n:
            X[i][j] = strata[i]
            i += 1

        j += 1

    return X


def balanced_levels(n, levels, rng):
    """
    Erstellt eine balancierte Liste der Länge n aus den gegebenen Levels.
    Beispiel: levels=[2,3,4], n=10 -> ungefähr gleich viele 2/3/4, dann shuffle.
    """
    out = []
    k = len(levels)
    base = n // k
    rem = n - base * k

    idx = 0
    while idx < k:
        c = 0
        while c < base:
            out.append(levels[idx])
            c += 1
        idx += 1

    idx = 0
    while idx < rem:
        out.append(levels[idx % k])
        idx += 1

    rng.shuffle(out)
    return out


def generate_lhs_cases(
    n,
    design_seed,
    lanes_levels=(2, 3, 4),
    speed_levels=(90, 110, 120, 130, 150, 160),
    count_min=30,
    count_max=80,
    density_min=1.0,
    density_max=3.0,
    duration=17,
    centered=False
):
    """
    Generiert 'n' Fälle (case dicts) über LHS + balancierte Kategorien.
    """
    rng = random.Random(design_seed)

    # LHS nur für kontinuierliche / quasi-kontinuierliche Variablen
    # Dim 0: vehicles_count (wird nachher auf int gemappt)
    # Dim 1: vehicles_density (float)
    U = lhs_unit(n, 2, rng, centered=centered)

    lanes_list = balanced_levels(n, list(lanes_levels), rng)
    speed_list = balanced_levels(n, list(speed_levels), rng)

    cases = []
    i = 0
    span_count = count_max - count_min + 1
    span_density = density_max - density_min

    while i < n:
        case = {}

        case["lanes_count"] = lanes_list[i]
        case["target_speed_kmh"] = speed_list[i]
        case["duration"] = duration

        # vehicles_count: LHS u -> int im Bereich [count_min, count_max]
        u_count = U[i][0]
        v_count = count_min + int(u_count * span_count)
        if v_count > count_max:
            v_count = count_max
        case["vehicles_count"] = v_count

        # vehicles_density: LHS u -> float im Bereich [density_min, density_max]
        u_den = U[i][1]
        case["vehicles_density"] = density_min + u_den * span_density

        cases.append(case)
        i += 1

    return cases


def build_cfg_from_case(base_config, case):
    cfg = copy.deepcopy(base_config)

    cfg["lanes_count"] = case["lanes_count"]
    cfg["vehicles_count"] = case["vehicles_count"]
    cfg["vehicles_density"] = case["vehicles_density"]
    cfg["target_speed_kmh"] = case["target_speed_kmh"]
    cfg["duration"] = case["duration"]

    return cfg


def save_to_excel(results, filename="lhs_results.xlsx"):
    df = pd.DataFrame(results)
    df.to_excel(filename, index=False)
    print(f"Ergebnisse erfolgreich in {filename} gespeichert!")


# ---------------------------------------------------------
# LHS Sampling: Episoden laufen lassen + identische Summary wie MC
# ---------------------------------------------------------
def lhs_sampling(
    base_config,
    N=10,
    seed=1,
    render=False,
    ttc_thr_crit=1.5,
    ttc_thr_warn=3.0,
    verbose_every=1,
    centered=False,
    oversample_factor=1.2,
    batch_size=2000,
    lanes_levels=(2, 3, 4),
    speed_levels=(90, 110, 120, 130, 150, 160),
    count_min=30,
    count_max=80,
    density_min=1.0,
    density_max=3.0
):
    """
    Space-filling Sampling via LHS:
    - Generiert Cases in Batches (um Discards abzufangen)
    - Läuft solange bis N gültige Episoden gesammelt sind
    """

    # Env erstellen
    if render:
        env = gym.make("highway-v0", render_mode="human")
    else:
        env = gym.make("highway-v0", render_mode=None)

    results = []

    discarded = 0
    i = 0  # Episode-Zähler (Seeds)
    design_seed = seed + 100000  # separater Seed-Stream für das LHS-Design

    try:
        while len(results) < N:
            remaining = N - len(results)

            # Batchgröße wählen (mit Oversampling, um Crashes zu kompensieren)
            n_batch = int(remaining * oversample_factor)
            if n_batch < batch_size:
                n_batch = batch_size

            cases = generate_lhs_cases(
                n=n_batch,
                design_seed=design_seed,
                lanes_levels=lanes_levels,
                speed_levels=speed_levels,
                count_min=count_min,
                count_max=count_max,
                density_min=density_min,
                density_max=density_max,
                duration=17,
                centered=centered
            )
            design_seed += 1

            idx = 0
            while idx < len(cases) and len(results) < N:
                cfg = build_cfg_from_case(base_config, cases[idx])

                episode_seed = seed + i

                res = E.run_one_episode(
                    env, cfg, episode_seed,
                    BETA=6.0, RHO=0.1, TAU=0.3,
                    ttc_thr_crit=ttc_thr_crit,
                    ttc_thr_warn=ttc_thr_warn
                )

                if res is None:
                    discarded += 1
                    i += 1
                    idx += 1
                    continue

                # Falls run_one_episode die cfg-Parameter nicht im res ablegt, hier absichern:
                if "lanes_count" not in res:
                    res["lanes_count"] = cfg["lanes_count"]
                if "vehicles_count" not in res:
                    res["vehicles_count"] = cfg["vehicles_count"]
                if "vehicles_density" not in res:
                    res["vehicles_density"] = cfg["vehicles_density"]
                if "target_speed_kmh" not in res:
                    res["target_speed_kmh"] = cfg["target_speed_kmh"]
                if "duration" not in res:
                    res["duration"] = cfg["duration"]

                res["sampling_method"] = "LHS_centered" if centered else "LHS_random"

                results.append(res)

                if verbose_every != 0:
                    if (len(results) - 1) % verbose_every == 0:
                        print("[Episode", len(results), "von", N, "] min_any =", round(res["min_any"], 2), "s")

                i += 1
                idx += 1

    finally:
        env.close()

    print("Verworfene Episoden (Crash im Warm-up):", discarded)

    # -------- Summary (identisch zu deinem MC) --------
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

    print("\n=== LHS SUMMARY ===")
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

    worst = sorted(results, key=lambda r: r["min_any"])
    print("\nWorst 5 cases (by min_any):")
    t = 0
    while t < 5 and t < len(worst):
        print(worst[t])
        t += 1

    return results


# ---------------------------------------------------------
# Main (Beispiel)
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

    results = lhs_sampling(
        base_config=base_config,
        N=10000,
        seed=1,
        render=False,
        ttc_thr_crit=1.5,
        ttc_thr_warn=3.0,
        verbose_every=1,
        centered=False,          # True = Median/Centered-LHS
        oversample_factor=1.2,
        batch_size=10000
    )

    with open("lhs_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Gespeichert: lhs_results.json")

    save_to_excel(results, filename="lhs_results.xlsx")


if __name__ == "__main__":
    main()
