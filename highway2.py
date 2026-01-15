import time
import math
import gymnasium as gym
import highway_env  # registriert env-ids


# ---------- TTC Hilfsfunktionen ----------

def _safe_speed(v):
    return float(getattr(v, "speed", 0.0))


def _get_lane(road, lane_index, cache):
    lane = cache.get(lane_index)
    if lane is None:
        lane = road.network.get_lane(lane_index)
        cache[lane_index] = lane
    return lane


def _longitudinal_s(lane, position):
    # lane.local_coordinates liefert (s, lateral)
    s, _ = lane.local_coordinates(position)
    return float(s)


def compute_follow_ttc_min(env):
    """
    Globales Minimum TTC für "Auffahren auf gleicher Spur" (rear -> front).
    Effizient: pro Spur nach s sortieren und nur Nachbarpaare prüfen.
    """
    road = env.unwrapped.road
    lane_cache = {}

    by_lane = {}
    for veh in road.vehicles:
        lane_index = getattr(veh, "lane_index", None)
        if lane_index is None:
            continue
        lane = _get_lane(road, lane_index, lane_cache)
        s = _longitudinal_s(lane, veh.position)
        by_lane.setdefault(lane_index, []).append((s, veh))

    global_min = math.inf
    for lane_index, lst in by_lane.items():
        lst.sort(key=lambda x: x[0])  # nach s

        # Nachbarpaare (rear, front)
        for (s_rear, rear), (s_front, front) in zip(lst, lst[1:]):
            gap = s_front - s_rear
            closing = _safe_speed(rear) - _safe_speed(front)  # rear schneller als front?
            if gap > 0 and closing > 1e-6:
                ttc = gap / closing
                if ttc < global_min:
                    global_min = ttc

    return global_min


def compute_cutin_ttc_for_vehicle(env, veh, target_lane_index):
    """
    Cut-in TTC für ein Fahrzeug bezogen auf die Zielspur:
    TTC_front (zu Vordermann in Zielspur) und TTC_rear (zu Hintermann in Zielspur).
    """
    road = env.unwrapped.road
    lane_cache = {}
    lane = _get_lane(road, target_lane_index, lane_cache)

    s_ego = _longitudinal_s(lane, veh.position)
    v_ego = _safe_speed(veh)

    # Nachbarn auf Zielspur (road projiziert ego auf die lane_index)
    try:
        front, rear = road.neighbour_vehicles(veh, lane_index=target_lane_index)
    except TypeError:
        # falls alte Signatur
        front, rear = road.neighbour_vehicles(veh)

    ttc_front = math.inf
    if front is not None:
        s_f = _longitudinal_s(lane, front.position)
        gap = s_f - s_ego
        closing = v_ego - _safe_speed(front)  # ego schließt auf?
        if gap > 0 and closing > 1e-6:
            ttc_front = gap / closing

    ttc_rear = math.inf
    if rear is not None:
        s_r = _longitudinal_s(lane, rear.position)
        gap = s_ego - s_r
        closing = _safe_speed(rear) - v_ego  # rear schließt auf?
        if gap > 0 and closing > 1e-6:
            ttc_rear = gap / closing

    return min(ttc_front, ttc_rear), ttc_front, ttc_rear


# ---------- Monitor (Episode-Metriken + Lane-Change-Events) ----------

class TTCMonitor:
    def __init__(self, policy_frequency: float, follow_every: int = 1):
        self.policy_frequency = float(policy_frequency)
        self.follow_every = int(max(1, follow_every))
        self.reset()

    def reset(self):
        self.episode_min_follow_ttc = math.inf
        self.episode_min_cutin_ttc = math.inf
        self.crashed = False

        # aktive lane-change events: veh -> event-dict
        self._lc_active = {}
        # abgeschlossene events
        self.lc_events = []

        # fallback: falls manche Fahrzeuge kein target_lane_index haben
        self._prev_lane_index = {}

    def _check_crash(self, env, info):
        if info and "crashed" in info:
            self.crashed = self.crashed or bool(info["crashed"])

        # optional: irgendein Fahrzeug gecrashed?
        try:
            road = env.unwrapped.road
            self.crashed = self.crashed or any(bool(getattr(v, "crashed", False)) for v in road.vehicles)
        except Exception:
            pass

    def step(self, env, step_idx: int, info=None):
        self._check_crash(env, info)

        # 1) Lane-follow TTC (nicht zwingend in jedem Step)
        if step_idx % self.follow_every == 0:
            follow_min = compute_follow_ttc_min(env)
            if follow_min < self.episode_min_follow_ttc:
                self.episode_min_follow_ttc = follow_min

        # 2) Cut-in TTC nur wenn LaneChange aktiv
        road = env.unwrapped.road
        vehicles_now = list(road.vehicles)

        # Fahrzeuge, die verschwunden sind -> Events schließen
        for veh in list(self._lc_active.keys()):
            if veh not in vehicles_now:
                ev = self._lc_active.pop(veh)
                ev["end_step"] = step_idx
                ev["duration_s"] = (ev["end_step"] - ev["start_step"]) / self.policy_frequency
                self.lc_events.append(ev)

        for veh in vehicles_now:
            lane_index = getattr(veh, "lane_index", None)
            target_lane_index = getattr(veh, "target_lane_index", None)

            # --- A) "Echter" LaneChange, wenn target_lane_index existiert ---
            is_lane_change = (lane_index is not None and target_lane_index is not None and target_lane_index != lane_index)

            if is_lane_change:
                ttc_min, ttc_f, ttc_r = compute_cutin_ttc_for_vehicle(env, veh, target_lane_index)

                if ttc_min < self.episode_min_cutin_ttc:
                    self.episode_min_cutin_ttc = ttc_min

                if veh not in self._lc_active:
                    # Event starten
                    self._lc_active[veh] = {
                        "start_step": step_idx,
                        "end_step": None,
                        "duration_s": None,
                        "target_lane_index": target_lane_index,
                        "min_ttc": ttc_min,
                        "min_ttc_front": ttc_f,
                        "min_ttc_rear": ttc_r,
                    }
                else:
                    # Event updaten (Minimum merken)
                    ev = self._lc_active[veh]
                    if ttc_min < ev["min_ttc"]:
                        ev["min_ttc"] = ttc_min
                        ev["min_ttc_front"] = ttc_f
                        ev["min_ttc_rear"] = ttc_r

            else:
                # Event beenden, sobald LaneChange nicht mehr aktiv ist
                if veh in self._lc_active:
                    ev = self._lc_active.pop(veh)
                    ev["end_step"] = step_idx
                    ev["duration_s"] = (ev["end_step"] - ev["start_step"]) / self.policy_frequency
                    self.lc_events.append(ev)

            # --- B) Fallback: Fahrzeuge ohne target_lane_index ---
            # Wenn kein target_lane_index vorhanden ist, erkennen wir zumindest einen Spurwechsel,
            # sobald lane_index sich ändert und loggen ein "Instant-Event" im neuen Lane.
            if lane_index is not None and target_lane_index is None:
                prev = self._prev_lane_index.get(veh)
                if prev is not None and prev != lane_index:
                    # "Instant" Cut-in-Messung nach dem Wechsel (nicht perfekt, aber besser als nichts)
                    try:
                        ttc_min, ttc_f, ttc_r = compute_cutin_ttc_for_vehicle(env, veh, lane_index)
                        self.episode_min_cutin_ttc = min(self.episode_min_cutin_ttc, ttc_min)
                        self.lc_events.append({
                            "start_step": step_idx,
                            "end_step": step_idx,
                            "duration_s": 0.0,
                            "target_lane_index": lane_index,
                            "min_ttc": ttc_min,
                            "min_ttc_front": ttc_f,
                            "min_ttc_rear": ttc_r,
                            "fallback": True
                        })
                    except Exception:
                        pass

            if lane_index is not None:
                self._prev_lane_index[veh] = lane_index

    def finalize(self, last_step_idx: int):
        # offene Events schließen
        for veh, ev in list(self._lc_active.items()):
            ev["end_step"] = last_step_idx
            ev["duration_s"] = (ev["end_step"] - ev["start_step"]) / self.policy_frequency
            self.lc_events.append(ev)
        self._lc_active.clear()


# ---------- Dein main ----------

def main():
    scenario_config = {
        "vehicles_count": 30,
        "duration": 40,
        "policy_frequency": 10,
        "simulation_frequency": 15,
        "lanes_count": 4,
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 15,
            "features": ["x", "y", "vx", "vy"],
            "absolute": False,
        },
        "action": {"type": "DiscreteMetaAction"},
    }

    env = gym.make("highway-v0", render_mode="human")  # für speed tests später render_mode weglassen
    env.unwrapped.configure(scenario_config)

    obs, info = env.reset(seed=None)

    IDLE = 1

    # FOLLOW_EVERY=1 -> jedes Step; 2/5 -> deutlich günstiger
    monitor = TTCMonitor(
        policy_frequency=scenario_config["policy_frequency"],
        follow_every=1
    )

    print("Starte Simulation. Beenden mit Ctrl+C.")

    episode_step = 0
    try:
        for step in range(3000):
            action = IDLE
            obs, reward, terminated, truncated, info = env.step(action)

            monitor.step(env, episode_step, info=info)
            episode_step += 1

            time.sleep(0.03)

            if terminated or truncated:
                monitor.finalize(episode_step)

                print("\n--- Episode fertig ---")
                print(f"crashed: {monitor.crashed}")
                print(f"min TTC follow:  {monitor.episode_min_follow_ttc:.3f} s")
                print(f"min TTC cut-in:  {monitor.episode_min_cutin_ttc:.3f} s")
                print(f"lane-change events: {len(monitor.lc_events)}")

                # optional: kritischste 5 Lane-Changes ausgeben
                worst = sorted(monitor.lc_events, key=lambda e: e.get("min_ttc", math.inf))[:5]
                for i, e in enumerate(worst, 1):
                    print(f"  #{i}: min_ttc={e['min_ttc']:.3f}s  front={e['min_ttc_front']:.3f}s  rear={e['min_ttc_rear']:.3f}s  dur={e['duration_s']:.2f}s")

                # neue Episode
                monitor.reset()
                episode_step = 0
                obs, info = env.reset(seed=None)

    except KeyboardInterrupt:
        print("\nAbbruch per Ctrl+C.")
    finally:
        env.close()
        print("Env geschlossen ✅")


if __name__ == "__main__":
    main()
