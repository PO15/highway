from highway_env.vehicle.behavior import IDMVehicle

class NormalIDMVehicle(IDMVehicle):
    # ACC_MAX: hartes Beschleunigungs-Limit (m/s²). Die am Ende angewendete Beschleunigung wird nach oben auf ACC_MAX begrenzt.
    ACC_MAX = 6.0

    # COMFORT_ACC_MAX: "komfortable" Maximal-Beschleunigung (m/s²) im IDM (höher = spritzigeres Beschleunigen Richtung target_speed).
    COMFORT_ACC_MAX = 3.0

    # COMFORT_ACC_MIN: "komfortable" Maximal-Verzögerung (m/s², negativ) im IDM (stärker negativ = härteres Bremsen wird zugelassen).
    COMFORT_ACC_MIN = -5.0

    # TIME_WANTED: gewünschte Zeitlücke T (s) zum Vordermann im stabilen Folgen (kleiner = dichter auffahren, größer = defensiver).
    TIME_WANTED = 1.0

    # DISTANCE_WANTED: gewünschter Mindestabstand s0 (m) bei Stillstand/niedriger Geschwindigkeit (größer = mehr Puffer).
    DISTANCE_WANTED = 5.0

    # DELTA: Exponent im IDM-Geschwindigkeitsterm (typisch 4). Beeinflusst, wie "scharf" Richtung Zielgeschwindigkeit beschleunigt wird.
    DELTA = 4.0

    #MOBIL

    # POLITENESS: MOBIL-Rücksichtsfaktor (0..1). 0 = egoistisch, höher = Spurwechsel nur wenn andere weniger benachteiligt werden.
    POLITENESS = 0.0

    # LANE_CHANGE_MIN_ACC_GAIN: Mindest-"Nutzen" eines Spurwechsels als Beschleunigungsgewinn (m/s²). Höher = seltenerer Spurwechsel.
    LANE_CHANGE_MIN_ACC_GAIN = 0.1

    # LANE_CHANGE_MAX_BRAKING_IMPOSED: max. zulässiges "Ausbremsen" anderer durch den Spurwechsel (m/s²). Höher = aggressivere Cut-ins.
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 2.0

    # LANE_CHANGE_DELAY: Intervall/Mindestzeit (s) zwischen Spurwechsel-Entscheidungen. Kleiner = häufiger Spurwechsel-Checks.
    LANE_CHANGE_DELAY = 0.5


class DefensiveIDMVehicle(IDMVehicle):
    ACC_MAX = 6.0
    COMFORT_ACC_MAX = 2.2
    COMFORT_ACC_MIN = -4.0
    TIME_WANTED = 2.0
    DISTANCE_WANTED = 12.0
    DELTA = 4.0

    #MOBIL
    POLITENESS = 0.4
    LANE_CHANGE_MIN_ACC_GAIN = 0.35
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 1.0
    LANE_CHANGE_DELAY = 1.5


class AggressiveIDMVehicle(IDMVehicle):
    ACC_MAX = 6.0
    COMFORT_ACC_MAX = 3.5
    COMFORT_ACC_MIN = -6.0
    TIME_WANTED = 1.1
    DISTANCE_WANTED = 7.0
    DELTA = 4.0
    
    #MOBIL
    POLITENESS = 0.0
    LANE_CHANGE_MIN_ACC_GAIN = 0.10
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 3.0
    LANE_CHANGE_DELAY = 0.7
