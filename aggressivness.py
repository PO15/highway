from highway_env.vehicle.behavior import IDMVehicle


class NormalIDMVehicle(IDMVehicle):
    SPEED_OFFSET_KMH = 0
    ACC_MAX = 2.0
    ACC_MIN = -4.5


class AggressiveIDMVehicle(IDMVehicle):
    SPEED_OFFSET_KMH = 10
    ACC_MAX = 3.0
    ACC_MIN = -5.0


class DefensiveIDMVehicle(IDMVehicle):
    SPEED_OFFSET_KMH = -5
    ACC_MAX = 1.5
    ACC_MIN = -4.0
