"""
ce_is_to_excel_varA.py  —  Cross-Entropy (CE) + Importance Sampling (IS) + Excel Export
Variante A für episode_seed: global fortlaufend wie bei Monte Carlo.

Voraussetzung:
- Environment.py enthält:
    run_one_episode(env, cfg, episode_seed, BETA, RHO, TAU, ttc_thr_crit, ttc_thr_warn)
  und gibt entweder ein dict (mit min_any etc.) zurück oder None (wenn Episode verworfen wird).

Excel-Output:
- CE_Episodes   : alle CE-Episoden (inkl. iter, is_elite, gamma)
- CE_Summary    : pro Iteration eine Zeile (gamma, rates, q-Parameter als JSON)
- IS_Episodes   : alle IS-Episoden (inkl. Gewichte + 2 Events <1.5 und <3.0)
- IS_Summary    : 2 Zeilen (event_thr=1.5 und event_thr=3.0)
- q_star_long   : q* im Long-Format (gut zum Plotten)
- Run_Config    : base_config + Parameter (Reproduzierbarkeit)

WICHTIG:
- log_probability_under_nominal_p(cfg) muss zu deinem echten nominalen Sampling passen.
  Hier ist es als uniform über Supports implementiert (wie in deinem Beispiel).
"""

import copy
import math
import random
import statistics
from datetime import datetime
import json, os
from datetime import datetime

import gymnasium as gym
import highway_env  # noqa: F401  (registriert env-ids)
import pandas as pd

import Environment as E  # deine Datei


# =========================================================
# 1) Diskrete Verteilungen: normieren / glätten / sampeln
# =========================================================

def normalize_probs(d: dict) -> dict:
    """Normiert {key: weight} auf Summe=1. Falls Summe<=0 -> uniform."""
    s = sum(d.values())
    if s <= 0:
        n = len(d)
        for k in d:
            d[k] = 1.0 / n
        return d
    for k in d:
        d[k] /= s
    return d


def smooth_update(old_probs: dict, new_probs: dict, alpha: float = 0.7) -> dict:
    """
    Glättung:
      p_new = (1-alpha)*p_old + alpha*p_elite
    Verhindert, dass q zu aggressiv kollabiert.
    """
    out = {}
    for k in old_probs:
        out[k] = (1 - alpha) * old_probs[k] + alpha * new_probs.get(k, 0.0)
    return normalize_probs(out)


def freq_probs(values: list, support_keys: list) -> dict:
    """Empirische Verteilung aus Values auf einem festen Support."""
    counts = {k: 0 for k in support_keys}
    for v in values:
        counts[v] += 1
    total = len(values)
    if total == 0:
        return normalize_probs({k: 1.0 for k in support_keys})
    return normalize_probs({k: counts[k] / total for k in support_keys})


def sample_from_categorical(rng: random.Random, probs_dict: dict):
    """Zieht 1 Wert aus einer diskreten Verteilung {wert: p}."""
    keys = list(probs_dict.keys())
    weights = [probs_dict[k] for k in keys]
    return rng.choices(keys, weights=weights, k=1)[0]


# =========================================================
# 2) Beta-Verteilung: Momentenmethode (Mean/Var -> alpha/beta)
# =========================================================

def beta_moments_to_ab(m: float, v: float, eps: float = 1e-9):
    """
    Momentenmethode für Beta(alpha,beta) auf [0,1].

    E[X]=m, Var[X]=v  ->  t = m(1-m)/v - 1
    alpha = m*t, beta = (1-m)*t
    """
    m = min(max(m, eps), 1 - eps)
    v = max(v, eps)

    t = m * (1 - m) / v - 1.0
    if t <= 0:
        return 1.0, 1.0  # fallback: uniform

    a = max(m * t, 0.1)
    b = max((1 - m) * t, 0.1)
    return a, b


# =========================================================
# 3) CE: q initialisieren, cfg sampeln, q updaten
# =========================================================

def init_q() -> dict:
    """
    Start-Proposal q (meist uniform).
    Du kannst Supports hier an deine Arbeit anpassen.
    """
    q = {}

    q["lanes_count_support"] = [2, 3, 4]
    q["lanes_count_p"] = normalize_probs({2: 1.0, 3: 1.0, 4: 1.0})

    q["vehicles_count_min"] = 30
    q["vehicles_count_max"] = 80
    q["vehicles_count_support"] = list(range(30, 81))
    q["vehicles_count_p"] = normalize_probs({k: 1.0 for k in q["vehicles_count_support"]})

    q["target_speed_support"] = [90, 110, 120, 130, 150, 160]
    q["target_speed_p"] = normalize_probs({k: 1.0 for k in q["target_speed_support"]})

    q["density_min"] = 1.0
    q["density_max"] = 3.0
    q["density_alpha"] = 1.0  # Beta(1,1) = uniform auf [0,1]
    q["density_beta"] = 1.0

    return q


def sample_cfg_from_q(base_config: dict, q: dict, rng: random.Random) -> dict:
    """
    Erzeugt ein konkretes cfg:
      - diskrete Parameter über Categorical
      - density über Beta auf [0,1] und Skalierung auf [density_min, density_max]
    """
    cfg = copy.deepcopy(base_config)

    cfg["lanes_count"] = sample_from_categorical(rng, q["lanes_count_p"])
    cfg["vehicles_count"] = sample_from_categorical(rng, q["vehicles_count_p"])
    cfg["target_speed_kmh"] = sample_from_categorical(rng, q["target_speed_p"])

    z = rng.betavariate(q["density_alpha"], q["density_beta"])  # z in [0,1]
    cfg["vehicles_density"] = q["density_min"] + z * (q["density_max"] - q["density_min"])

    cfg["duration"] = cfg.get("duration", 17)
    return cfg


def update_q_from_elite(q: dict, elite_cfgs: list, smooth_alpha: float = 0.5) -> dict:
    """Update q anhand Elite-Samples."""
    # lanes_count
    lanes_vals = [c["lanes_count"] for c in elite_cfgs]
    lanes_new = freq_probs(lanes_vals, q["lanes_count_support"])
    q["lanes_count_p"] = smooth_update(q["lanes_count_p"], lanes_new, alpha=smooth_alpha)

    # vehicles_count
    vc_vals = [c["vehicles_count"] for c in elite_cfgs]
    vc_new = freq_probs(vc_vals, q["vehicles_count_support"])
    q["vehicles_count_p"] = smooth_update(q["vehicles_count_p"], vc_new, alpha=smooth_alpha)

    # target_speed
    ts_vals = [c["target_speed_kmh"] for c in elite_cfgs]
    ts_new = freq_probs(ts_vals, q["target_speed_support"])
    q["target_speed_p"] = smooth_update(q["target_speed_p"], ts_new, alpha=smooth_alpha)

    # density (Beta)
    dens = [c["vehicles_density"] for c in elite_cfgs]
    dmin, dmax = q["density_min"], q["density_max"]
    scale = (dmax - dmin)

    z_vals = []
    for x in dens:
        z = (x - dmin) / scale
        z = min(max(z, 1e-9), 1 - 1e-9)
        z_vals.append(z)

    if len(z_vals) >= 2:
        m = statistics.mean(z_vals)
        v = statistics.pvariance(z_vals)
        a_new, b_new = beta_moments_to_ab(m, v)
    elif len(z_vals) == 1:
        a_new, b_new = beta_moments_to_ab(z_vals[0], 0.01)
    else:
        a_new, b_new = 1.0, 1.0

    q["density_alpha"] = max(0.1, (1 - smooth_alpha) * q["density_alpha"] + smooth_alpha * a_new)
    q["density_beta"]  = max(0.1, (1 - smooth_alpha) * q["density_beta"]  + smooth_alpha * b_new)

    return q


def cross_entropy_optimize(
    base_config: dict,
    iters: int = 10,
    samples_per_iter: int = 300,
    elite_frac: float = 0.1,
    smooth_alpha: float = 0.3,
    seed: int = 1,
    render: bool = False,
    ttc_thr_crit: float = 1.5,
    ttc_thr_warn: float = 3.0,
    BETA: float = 6.0,
    RHO: float = 0.1,
    TAU: float = 0.3,
):
    """
    CE-Lernen von q:
      - pro Iteration: samples_per_iter Episoden sampeln & simulieren
      - Elite auswählen: k=ceil(elite_frac*m) kleinste min_any
      - q anhand Elite updaten
      - episode_seed (Variante A): global fortlaufend seed + episode_id

    Returns:
      q_star, ce_episode_rows, ce_summary_rows
    """
    rng = random.Random(seed)
    q = init_q()

    env = gym.make("highway-v0", render_mode="human" if render else None)

    ce_episode_rows = []
    ce_summary_rows = []

    episode_id = 0  # <<< VARIANTE A: globaler Counter über alle Iterationen

    try:
        for t in range(1, iters + 1):
            print(f"\n[CE] Starte Iteration {t}/{iters} ...", flush=True)

            batch_records = []
            discarded = 0

            i = 0
            while i < samples_per_iter:
                cfg = sample_cfg_from_q(base_config, q, rng)

                # <<< VARIANTE A: Seeds laufen fortlaufend wie bei MC
                episode_seed = seed + episode_id
                episode_id += 1

                res = E.run_one_episode(
                    env, cfg, episode_seed,
                    BETA=BETA, RHO=RHO, TAU=TAU,
                    ttc_thr_crit=ttc_thr_crit,
                    ttc_thr_warn=ttc_thr_warn
                )

                if res is None:
                    discarded += 1
                    i += 1
                    continue

                min_any = res.get("min_any", float("inf"))
                score = min_any  # CE optimiert auf min_any (kleiner = kritischer)

                row = {
                    "iter": t,
                    "episode_seed": episode_seed,

                    # cfg (MC-like)
                    "lanes_count": cfg.get("lanes_count"),
                    "vehicles_count": cfg.get("vehicles_count"),
                    "vehicles_density": cfg.get("vehicles_density"),
                    "target_speed_kmh": cfg.get("target_speed_kmh"),
                    "duration": cfg.get("duration"),

                    # res (MC-like)
                    "min_follow": res.get("min_follow", float("inf")),
                    "min_cutin": res.get("min_cutin", float("inf")),
                    "min_cutin_dvo": res.get("min_cutin_dvo", float("inf")),
                    "episode_cutin_critical": res.get("episode_cutin_critical", 0),
                    "min_any": min_any,
                    "episode_below_1p5": res.get("episode_below_1p5", 1 if min_any < 1.5 else 0),
                    "episode_below_3p0": res.get("episode_below_3p0", 1 if min_any < 3.0 else 0),

                    # CE extras
                    "score": score,
                    "is_elite": 0,
                    "gamma": None,
                }

                batch_records.append(row)
                i += 1

            if len(batch_records) == 0:
                ce_summary_rows.append({
                    "iter": t, "valid": 0, "discarded": discarded, "elite_k": 0,
                    "elite_frac": elite_frac, "gamma": None, "rate_min_any_below_crit": None,
                    "q_lanes_count_p_json": json.dumps(q["lanes_count_p"]),
                    "q_target_speed_p_json": json.dumps(q["target_speed_p"]),
                    "q_vehicles_count_p_json": json.dumps(q["vehicles_count_p"]),
                    "q_density_alpha": q["density_alpha"], "q_density_beta": q["density_beta"],
                })
                continue

            batch_sorted = sorted(batch_records, key=lambda r: r["score"])
            m = len(batch_sorted)
            k = max(1, int(math.ceil(elite_frac * m)))

            elite_rows = batch_sorted[:k]
            gamma = elite_rows[-1]["score"]

            elite_seeds = {r["episode_seed"] for r in elite_rows}
            for r in batch_records:
                if r["episode_seed"] in elite_seeds:
                    r["is_elite"] = 1
                r["gamma"] = gamma

            below_crit = sum(1 for r in batch_records if r["min_any"] < ttc_thr_crit)
            rate_crit = below_crit / m

            elite_cfgs = [
                {
                    "lanes_count": r["lanes_count"],
                    "vehicles_count": r["vehicles_count"],
                    "vehicles_density": r["vehicles_density"],
                    "target_speed_kmh": r["target_speed_kmh"],
                }
                for r in elite_rows
            ]
            q = update_q_from_elite(q, elite_cfgs, smooth_alpha=smooth_alpha)

            ce_summary_rows.append({
                "iter": t,
                "valid": m,
                "discarded": discarded,
                "elite_k": k,
                "elite_frac": elite_frac,
                "gamma": gamma,
                "rate_min_any_below_crit": rate_crit,

                "q_lanes_count_p_json": json.dumps(q["lanes_count_p"]),
                "q_target_speed_p_json": json.dumps(q["target_speed_p"]),
                "q_vehicles_count_p_json": json.dumps(q["vehicles_count_p"]),
                "q_density_alpha": q["density_alpha"],
                "q_density_beta": q["density_beta"],
            })

            ce_episode_rows.extend(batch_records)

        return q, ce_episode_rows, ce_summary_rows

    finally:
        env.close()


# =========================================================
# 4) IS: log Dichten/Wahrscheinlichkeiten + Gewichte
# =========================================================

def log_beta_density_0_1(z: float, alpha: float, beta: float) -> float:
    """log Dichte der Beta(alpha,beta) auf [0,1] an Stelle z."""
    if z <= 0.0 or z >= 1.0:
        return float("-inf")
    log_B = math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta)
    return (alpha - 1.0) * math.log(z) + (beta - 1.0) * math.log(1.0 - z) - log_B


def log_probability_under_nominal_p(cfg: dict) -> float:
    """
    log(p(cfg)) unter nominalem Sampling.

    HIER: uniform angenommen
      lanes_count in {2,3,4}        -> 1/3
      vehicles_count in {30..80}    -> 1/51
      target_speed in {6 Werte}     -> 1/6
      density uniform [1,3] (PDF)   -> 1/2

    Passe das an, falls dein echtes MC andere Regeln hat.
    """
    if cfg["lanes_count"] not in (2, 3, 4):
        return float("-inf")
    if not (30 <= cfg["vehicles_count"] <= 80):
        return float("-inf")
    if cfg["target_speed_kmh"] not in (90, 110, 120, 130, 150, 160):
        return float("-inf")
    if not (1.0 <= cfg["vehicles_density"] <= 3.0):
        return float("-inf")

    log_p_lanes = -math.log(3.0)
    log_p_vc    = -math.log(51.0)
    log_p_speed = -math.log(6.0)
    log_p_density = -math.log(3.0 - 1.0)  # 1/2

    return log_p_lanes + log_p_vc + log_p_speed + log_p_density


def log_probability_under_proposal_q(cfg: dict, q: dict) -> float:
    """
    log(q(cfg)) unter Proposal q:
      - diskret: Produkt der PMFs
      - density: Beta-PDF auf z in [0,1] + Skalierung auf [min,max] => -log(scale)
    """
    p_lane = q["lanes_count_p"].get(cfg["lanes_count"], 0.0)
    p_vc   = q["vehicles_count_p"].get(cfg["vehicles_count"], 0.0)
    p_spd  = q["target_speed_p"].get(cfg["target_speed_kmh"], 0.0)
    if p_lane <= 0.0 or p_vc <= 0.0 or p_spd <= 0.0:
        return float("-inf")

    log_q = math.log(p_lane) + math.log(p_vc) + math.log(p_spd)

    dmin, dmax = q["density_min"], q["density_max"]
    x = cfg["vehicles_density"]
    if not (dmin <= x <= dmax):
        return float("-inf")

    scale = (dmax - dmin)
    z = (x - dmin) / scale
    z = min(max(z, 1e-12), 1.0 - 1e-12)

    log_q_density = log_beta_density_0_1(z, q["density_alpha"], q["density_beta"]) - math.log(scale)
    return log_q + log_q_density


def estimate_probability_with_importance_sampling_two_thresholds(
    base_config: dict,
    q_star: dict,
    N: int = 5000,
    seed: int = 1,
    render: bool = False,
    event_thrs=(1.5, 3.0),
    ttc_thr_crit: float = 1.5,
    ttc_thr_warn: float = 3.0,
    BETA: float = 6.0,
    RHO: float = 0.1,
    TAU: float = 0.3
):
    """
    EIN IS-Lauf, aber ZWEI Events gleichzeitig:
      I1 = 1(min_any < event_thrs[0])
      I2 = 1(min_any < event_thrs[1])

    Returns:
      is_episode_rows, is_summary_rows (2 Zeilen)
    """
    rng = random.Random(seed)
    env = gym.make("highway-v0", render_mode="human" if render else None)

    rows = []
    discarded = 0

    thr1, thr2 = float(event_thrs[0]), float(event_thrs[1])

    try:
        i = 0
        while len(rows) < N:

            if len(rows) % 50 == 0:
                print(f"[IS] {len(rows)}/{N} gültige Episoden | discarded={discarded}", flush=True)

            cfg = sample_cfg_from_q(base_config, q_star, rng)
            episode_seed = seed + i
            i += 1

            res = E.run_one_episode(
                env, cfg, episode_seed,
                BETA=BETA, RHO=RHO, TAU=TAU,
                ttc_thr_crit=ttc_thr_crit,
                ttc_thr_warn=ttc_thr_warn
            )

            if res is None:
                discarded += 1
                continue

            min_any = res.get("min_any", float("inf"))

            I1 = 1 if (min_any < thr1) else 0
            I2 = 1 if (min_any < thr2) else 0

            logp = log_probability_under_nominal_p(cfg)
            logq = log_probability_under_proposal_q(cfg, q_star)
            if not (math.isfinite(logp) and math.isfinite(logq)):
                continue

            logw = logp - logq

            rows.append({
                "episode_seed": episode_seed,

                # cfg (MC-like)
                "lanes_count": cfg.get("lanes_count"),
                "vehicles_count": cfg.get("vehicles_count"),
                "vehicles_density": cfg.get("vehicles_density"),
                "target_speed_kmh": cfg.get("target_speed_kmh"),
                "duration": cfg.get("duration"),

                # res (MC-like)
                "min_follow": res.get("min_follow", float("inf")),
                "min_cutin": res.get("min_cutin", float("inf")),
                "min_cutin_dvo": res.get("min_cutin_dvo", float("inf")),
                "episode_cutin_critical": res.get("episode_cutin_critical", 0),
                "min_any": min_any,
                "episode_below_1p5": res.get("episode_below_1p5", 1 if min_any < 1.5 else 0),
                "episode_below_3p0": res.get("episode_below_3p0", 1 if min_any < 3.0 else 0),

                # IS extras
                "logp": logp,
                "logq": logq,
                "logw": logw,
                "w_rescaled": None,
                "w_norm_snis": None,

                "event_thr_1": thr1,
                "event_thr_2": thr2,
                "I_event_1": I1,
                "I_event_2": I2,
            })

    finally:
        env.close()

    # Gewichte rescalen (stabil) und SNIS-normalisieren
    logws = [r["logw"] for r in rows]
    max_logw = max(logws)
    w_rescaled = [math.exp(lw - max_logw) for lw in logws]
    sum_w = sum(w_rescaled)

    for r, w in zip(rows, w_rescaled):
        r["w_rescaled"] = w
        r["w_norm_snis"] = (w / sum_w) if sum_w > 0 else float("nan")

    # ESS Diagnose (gemeinsam)
    sum_w2 = sum(w * w for w in w_rescaled)
    ess = (sum_w * sum_w / sum_w2) if sum_w2 > 0 else 0.0

    Nvalid = len(rows)
    sum_wI1 = sum(r["w_rescaled"] * r["I_event_1"] for r in rows)
    sum_wI2 = sum(r["w_rescaled"] * r["I_event_2"] for r in rows)

    # Unbiased (Rescaling zurückmultiplizieren)
    p_hat_unbiased_1 = math.exp(max_logw) * (sum_wI1 / Nvalid)
    p_hat_unbiased_2 = math.exp(max_logw) * (sum_wI2 / Nvalid)

    # SNIS
    p_hat_snis_1 = (sum_wI1 / sum_w) if sum_w > 0 else float("nan")
    p_hat_snis_2 = (sum_wI2 / sum_w) if sum_w > 0 else float("nan")

    summary_rows = [
        {
            "event_thr": thr1,
            "N_valid": Nvalid,
            "discarded": discarded,
            "p_hat_unbiased": p_hat_unbiased_1,
            "p_hat_snis": p_hat_snis_1,
            "ess": ess,
            "max_logw": max_logw,
        },
        {
            "event_thr": thr2,
            "N_valid": Nvalid,
            "discarded": discarded,
            "p_hat_unbiased": p_hat_unbiased_2,
            "p_hat_snis": p_hat_snis_2,
            "ess": ess,
            "max_logw": max_logw,
        },
    ]

    return rows, summary_rows


# =========================================================
# 5) Excel Export
# =========================================================

MC_COLUMNS = [
    "episode_seed",
    "lanes_count",
    "vehicles_count",
    "vehicles_density",
    "target_speed_kmh",
    "duration",
    "min_follow",
    "min_cutin",
    "min_cutin_dvo",
    "episode_cutin_critical",
    "min_any",
    "episode_below_1p5",
    "episode_below_3p0",
]


def q_to_long_rows(q: dict):
    """q* als Long-Format (param, key, value) für Excel."""
    rows = []
    for k, v in q["lanes_count_p"].items():
        rows.append({"param": "lanes_count_p", "key": int(k), "value": float(v)})
    for k, v in q["vehicles_count_p"].items():
        rows.append({"param": "vehicles_count_p", "key": int(k), "value": float(v)})
    for k, v in q["target_speed_p"].items():
        rows.append({"param": "target_speed_p", "key": int(k), "value": float(v)})

    rows.append({"param": "density_min", "key": None, "value": float(q["density_min"])})
    rows.append({"param": "density_max", "key": None, "value": float(q["density_max"])})
    rows.append({"param": "density_alpha", "key": None, "value": float(q["density_alpha"])})
    rows.append({"param": "density_beta", "key": None, "value": float(q["density_beta"])})
    return rows


def save_all_to_excel(
    filepath: str,
    base_config: dict,
    ce_params: dict,
    is_params: dict,
    q_star: dict,
    ce_episode_rows: list,
    ce_summary_rows: list,
    is_episode_rows: list,
    is_summary_rows: list
):
    # CE Episodes
    ce_cols = ["iter"] + MC_COLUMNS + ["score", "is_elite", "gamma"]
    df_ce = pd.DataFrame(ce_episode_rows)
    for c in ce_cols:
        if c not in df_ce.columns:
            df_ce[c] = None
    df_ce = df_ce[ce_cols]

    # CE Summary
    df_ce_sum = pd.DataFrame(ce_summary_rows)

    # IS Episodes
    is_cols = MC_COLUMNS + [
        "event_thr_1", "I_event_1",
        "event_thr_2", "I_event_2",
        "logp", "logq", "logw",
        "w_rescaled", "w_norm_snis",
    ]
    df_is = pd.DataFrame(is_episode_rows)
    for c in is_cols:
        if c not in df_is.columns:
            df_is[c] = None
    df_is = df_is[is_cols]

    # IS Summary (2 Zeilen)
    df_is_sum = pd.DataFrame(is_summary_rows)

    # q_star long
    df_q = pd.DataFrame(q_to_long_rows(q_star))

    # Run Config
    cfg_rows = [{"section": "meta", "key": "created_at", "value": datetime.now().isoformat()}]
    for k, v in ce_params.items():
        cfg_rows.append({"section": "ce_params", "key": k, "value": json.dumps(v) if isinstance(v, (dict, list, tuple)) else v})
    for k, v in is_params.items():
        cfg_rows.append({"section": "is_params", "key": k, "value": json.dumps(v) if isinstance(v, (dict, list, tuple)) else v})
    for k, v in base_config.items():
        cfg_rows.append({"section": "base_config", "key": k, "value": json.dumps(v) if isinstance(v, (dict, list, tuple)) else v})
    df_cfg = pd.DataFrame(cfg_rows)

    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        df_ce.to_excel(writer, sheet_name="CE_Episodes", index=False)
        df_ce_sum.to_excel(writer, sheet_name="CE_Summary", index=False)
        df_is.to_excel(writer, sheet_name="IS_Episodes", index=False)
        df_is_sum.to_excel(writer, sheet_name="IS_Summary", index=False)
        df_q.to_excel(writer, sheet_name="q_star_long", index=False)
        df_cfg.to_excel(writer, sheet_name="Run_Config", index=False)

def sanitize_for_json(x):
    # macht inf/nan JSON-sicher
    if isinstance(x, float):
        if math.isinf(x):
            return "inf" if x > 0 else "-inf"
        if math.isnan(x):
            return "nan"
        return x
    if isinstance(x, dict):
        return {k: sanitize_for_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [sanitize_for_json(v) for v in x]
    return x


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sanitize_for_json(obj), f, indent=2, ensure_ascii=False)





# =========================================================
# 6) Main
# =========================================================

def main():
    # Passe base_config an dein Setup an
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


    # --- CE Parameter ---
    ce_params = dict(
        iters=6,
        samples_per_iter=300,
        elite_frac=0.25,
        smooth_alpha=0.15,
        seed=1,
        render=False,
        ttc_thr_crit=1.5,
        ttc_thr_warn=3.0,
        BETA=6.0,
        RHO=0.1,
        TAU=0.3,
    )

    # --- CE laufen lassen ---
    q_star, ce_episode_rows, ce_summary_rows = cross_entropy_optimize(
        base_config=base_config,
        **ce_params
    )

    # --- IS Parameter (beide Schwellen in einem Run) ---
    is_params = dict(
        N=10000,
        seed=1,
        render=False,
        event_thrs=(1.5, 3.0),
        ttc_thr_crit=ce_params["ttc_thr_crit"],
        ttc_thr_warn=ce_params["ttc_thr_warn"],
        BETA=ce_params["BETA"],
        RHO=ce_params["RHO"],
        TAU=ce_params["TAU"],
    )

    # --- IS laufen lassen ---
    is_episode_rows, is_summary_rows = estimate_probability_with_importance_sampling_two_thresholds(
        base_config=base_config,
        q_star=q_star,
        **is_params
    )

    # --- Excel schreiben ---
    out_xlsx = "ce_is_results.xlsx"
    save_all_to_excel(
        filepath=out_xlsx,
        base_config=base_config,
        ce_params=ce_params,
        is_params=is_params,
        q_star=q_star,
        ce_episode_rows=ce_episode_rows,
        ce_summary_rows=ce_summary_rows,
        is_episode_rows=is_episode_rows,
        is_summary_rows=is_summary_rows
    )

    print(f"\nFertig. Excel geschrieben: {out_xlsx}")
    print("IS Summary Rows:")
    for r in is_summary_rows:
        print(r)

    # ---------- IS JSON Export (MC-like) ----------
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("outputs", f"is_run_{run_tag}")
    os.makedirs(out_dir, exist_ok=True)

    # 1) IS Parameter + q_star (damit du später weißt, womit gesampelt wurde)
    save_json(os.path.join(out_dir, "is_params.json"), {
        "is_params": is_params,
        "q_star": q_star,           # optional, aber sehr praktisch
        "base_config": base_config  # optional
    })

    # 2) IS Episodes (nur das Wichtigste, damit es übersichtlich bleibt)
    is_episodes_compact = []
    for r in is_episode_rows:
        is_episodes_compact.append({
            "episode_seed": r["episode_seed"],
            "cfg": {
                "lanes_count": r["lanes_count"],
                "vehicles_count": r["vehicles_count"],
                "vehicles_density": r["vehicles_density"],
                "target_speed_kmh": r["target_speed_kmh"],
                "duration": r["duration"],
            },
            "res": {
                "min_any": r["min_any"],
                "min_follow": r["min_follow"],
                "min_cutin": r["min_cutin"],
                "min_cutin_dvo": r["min_cutin_dvo"],
            },
            "events": {
                "I_event_1": r.get("I_event_1"),  # <1.5
                "I_event_2": r.get("I_event_2"),  # <3.0
                "event_thr_1": r.get("event_thr_1"),
                "event_thr_2": r.get("event_thr_2"),
            },
            "weights": {
                "logw": r.get("logw"),
                "w_rescaled": r.get("w_rescaled"),
                "w_norm_snis": r.get("w_norm_snis"),
            }
        })

    save_json(os.path.join(out_dir, "is_episodes.json"), is_episodes_compact)

    # 3) IS Summary (2 Zeilen: 1.5 und 3.0)
    save_json(os.path.join(out_dir, "is_summary.json"), is_summary_rows)

    print("IS JSON gespeichert in:", out_dir)
        


if __name__ == "__main__":
    main()
