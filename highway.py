import time
import types
import gymnasium as gym
import highway_env  # registriert env-ids


def convert_ego_to_idm(env, target_speed_kmh=None):
    """
    Ersetzt das kontrollierte Ego-Fahrzeug durch ein IDMVehicle (IDM + MOBIL),
    und sorgt dafür, dass env.step(action) die Action ignoriert (IDM fährt selbst).
    """
    from highway_env.vehicle.behavior import IDMVehicle  # siehe Doku: highway_env.vehicle.behavior.IDMVehicle

    uw = env.unwrapped

    old_ego = uw.vehicle
    idm_ego = IDMVehicle.create_from(old_ego)  # Doku: create_from() kopiert Dynamik + Target-Dynamik :contentReference[oaicite:2]{index=2}

    # optional: Wunschgeschwindigkeit setzen
    if target_speed_kmh is not None:
        idm_ego.target_speed = target_speed_kmh / 3.6

    # 1) ego in road.vehicles ersetzen
    for i, v in enumerate(uw.road.vehicles):
        if v is old_ego:
            uw.road.vehicles[i] = idm_ego
            break

    # 2) env.vehicle / controlled_vehicles aktualisieren
    uw.vehicle = idm_ego
    if hasattr(uw, "controlled_vehicles"):
        if uw.controlled_vehicles:
            uw.controlled_vehicles[0] = idm_ego
        else:
            uw.controlled_vehicles = [idm_ego]

    # 3) action_type so patchen, dass er die Action ignoriert (IDMVehicle unterstützt keine externen Actions) :contentReference[oaicite:3]{index=3}
    try:
        uw.action_type.controlled_vehicle = idm_ego  # property existiert in der ActionType-API :contentReference[oaicite:4]{index=4}
    except Exception:
        pass

    def _act_ignore(self, action):
        # Action aus Gym ignorieren, IDM/MOBIL entscheidet selbst
        self.controlled_vehicle.act(None)

    uw.action_type.act = types.MethodType(_act_ignore, uw.action_type)

    return idm_ego


def main():
    scenario_config = {
        "vehicles_count": 10,
        "duration": 10,
        "policy_frequency": 10,
        "simulation_frequency": 15,
        "lanes_count": 3,
        "ego_spacing": 1,   # kleiner = dichterer Start (probier 0.7–2.0)
        "vehicles_density": 2.0, # steuert Traffic-Spawn-Dichte (spacing = 1 / vehicles_density)
        "screen_width": 1400,
        "screen_height": 350,
        "scaling": 1,              # kleiner = mehr Strecke sichtbar
        "centering_position": [0.35, 0.5],
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 15,
            "features": ["x", "y", "vx", "vy"],
            "absolute": False,
        },
        "action": {"type": "DiscreteMetaAction"},
    }

    env = gym.make("highway-v0", render_mode="human")

    #obs, info = env.reset(options={"config": scenario_config})
    obs, info = env.reset(seed=0, options={"config": scenario_config}) #seed gesetzt

    convert_ego_to_idm(env, target_speed_kmh=200)  # <-- Ego fährt jetzt IDM+MOBIL selbst

    print("Starte Simulation (Ego = IDMVehicle). Beenden: Ctrl+C")

    print("config vehicles_count:", env.unwrapped.config["vehicles_count"])
    print("vehicles in road:", len(env.unwrapped.road.vehicles))


    try:
        for step in range(2000):
            # Action ist egal, wird ignoriert (IDM fährt selbst)
            action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            time.sleep(0.03)

            if terminated or truncated:
                obs, info = env.reset(seed=0, options={"config": scenario_config}) #seed gesetzt
                convert_ego_to_idm(env, target_speed_kmh=200)

    except KeyboardInterrupt:
        print("\nAbbruch per Ctrl+C.")

    finally:
        env.close()
        print("Env geschlossen ✅")


if __name__ == "__main__":
    main()
