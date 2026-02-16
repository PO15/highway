import json
import numpy as np
import gymnasium as gym
import highway_env  # noqa: F401


# ---------------------------------------------------------
# Dichte-Messung: fixes Fenster (veh/km/lane)
# ---------------------------------------------------------
def measure_density_window(env, window_m=1000.0, include_ego=False):
    uw = env.unwrapped
    ego = uw.vehicle
    x0 = float(ego.position[0])
    half = float(window_m) / 2.0

    lanes_count = int(uw.config.get("lanes_count", 3))

    count_total = 0
    for v in uw.road.vehicles:
        if (not include_ego) and (v is ego):
            continue
        x = float(v.position[0])
        if (x0 - half) <= x <= (x0 + half):
            count_total += 1

    L_km = float(window_m) / 1000.0
    rho_avg = count_total / (L_km * lanes_count) if L_km > 0 else 0.0

    return float(rho_avg)


# ---------------------------------------------------------
# LOS (optional, HCM grob umgerechnet auf veh/km/ln)
# ---------------------------------------------------------
def los_from_density_km_lane(rho):
    # 11/18/26/35/45 pc/mi/ln -> /1.609 ≈ 6.8/11.2/16.2/21.7/28.0 veh/km/ln
    if rho < 6.8:  return "A"
    if rho < 11.2: return "B"
    if rho < 16.2: return "C"
    if rho < 21.7: return "D"
    if rho < 28.0: return "E"
    return "F"


# ---------------------------------------------------------
# Kalibrierung für EIN lanes_count
# ---------------------------------------------------------
def calibrate_for_lanes(
    lanes_count,
    test_values,
    vehicles_count=50,
    n_samples=20,
    window_m=1000.0,
    include_ego=False,
):
    env = gym.make("highway-v0", render_mode=None)

    results = {}
    for vd in test_values:
        config = {
            "vehicles_count": int(vehicles_count),
            "lanes_count": int(lanes_count),
            "vehicles_density": float(vd),
            "duration": 1,
            "policy_frequency": 1,
            "simulation_frequency": 15,
        }

        rhos = []
        for k in range(int(n_samples)):
            # Reproduzierbar: seed hängt von lanes, vd, k ab
            seed = 100_000 + 10_000 * int(lanes_count) + 100 * int(round(vd * 100)) + k
            env.reset(seed=seed, options={"config": config})
            rhos.append(measure_density_window(env, window_m=window_m, include_ego=include_ego))

        mean_rho = float(np.mean(rhos)) if rhos else 0.0
        std_rho = float(np.std(rhos)) if rhos else 0.0

        results[float(vd)] = {
            "mean": mean_rho,
            "std": std_rho,
            "los": los_from_density_km_lane(mean_rho),
            "samples": [float(x) for x in rhos],
        }

    env.close()
    return results


# ---------------------------------------------------------
# Kalibrierung für mehrere lanes_count (2/3/4)
# ---------------------------------------------------------
def calibrate_all_lanes(
    lanes_list=(2, 3, 4),
    test_values=(0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5),
    vehicles_count=50,
    n_samples=20,
    window_m=1000.0,
    include_ego=False,
    save_path="density_calibration_by_lanes.json",
):
    print("\n" + "=" * 78)
    print("KALIBRIERUNG: vehicles_density → rho_eff [veh/km/lane], getrennt nach lanes_count")
    print("=" * 78)
    print(f"vehicles_count={vehicles_count}, window={window_m:.0f} m, n_samples={n_samples}, include_ego={include_ego}\n")

    all_results = {}

    for lanes in lanes_list:
        print(f"\n--- lanes_count = {lanes} ---")
        res = calibrate_for_lanes(
            lanes_count=lanes,
            test_values=test_values,
            vehicles_count=vehicles_count,
            n_samples=n_samples,
            window_m=window_m,
            include_ego=include_ego,
        )
        all_results[int(lanes)] = {
            "meta": {
                "lanes_count": int(lanes),
                "vehicles_count": int(vehicles_count),
                "window_m": float(window_m),
                "n_samples": int(n_samples),
                "include_ego": bool(include_ego),
            },
            "map": {str(vd): res[vd] for vd in sorted(res.keys())},
        }

        # Pretty print summary
        print("| vehicles_density | rho_eff_mean [veh/km/lane] | rho_eff_std | LOS |")
        print("|------------------|---------------------------|-------------|-----|")
        for vd in sorted(res.keys()):
            print(f"| {vd:16.2f} | {res[vd]['mean']:25.1f} | {res[vd]['std']:11.1f} | {res[vd]['los']:3s} |")

    # Save JSON
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Gespeichert: {save_path}")
    return all_results


# ---------------------------------------------------------
# Invertierung: Ziel-Dichte -> vehicles_density (Interpolation)
# ---------------------------------------------------------
def vd_for_target_density(calib_map_for_one_lane, rho_target):
    """
    calib_map_for_one_lane: dict wie all_results[lanes]["map"]
                            keys sind strings "0.3", "0.5", ...
    rho_target: gewünschte Dichte in veh/km/lane
    """
    vds = np.array(sorted([float(k) for k in calib_map_for_one_lane.keys()]), dtype=float)
    rhos = np.array([calib_map_for_one_lane[str(vd)]["mean"] for vd in vds], dtype=float)

    # monotone Annahme (typisch): rho steigt mit vd
    order = np.argsort(rhos)
    rhos = rhos[order]
    vds = vds[order]

    # clamp auf Randbereich
    rho_target = float(rho_target)
    if rho_target <= rhos[0]:
        return float(vds[0])
    if rho_target >= rhos[-1]:
        return float(vds[-1])

    # lineare Interpolation: rho -> vd
    return float(np.interp(rho_target, rhos, vds))


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    all_results = calibrate_all_lanes(
        lanes_list=(2, 3, 4),
        test_values=(0.5, 1.0, 1.5, 2.0, 2.5, 3),
        vehicles_count=80,
        n_samples=100,
        window_m=1000.0,
        include_ego=False,
        save_path="density_calibration_by_lanes.json",
    )

    # Beispiel: gleiche Ziel-Dichte für alle Lanes wählen
    rho_target = 18.0  # veh/km/lane, Beispiel
    print("\n" + "=" * 78)
    print(f"Beispiel Invertierung: Ziel-Dichte rho_target = {rho_target:.1f} veh/km/lane")
    print("=" * 78)
    for lanes in (2, 3, 4):
        calib_map = all_results[lanes]["map"]
        vd = vd_for_target_density(calib_map, rho_target)
        print(f"lanes_count={lanes}: vehicles_density ≈ {vd:.3f}")
