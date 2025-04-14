import numpy as np

class Supervisor:
    TAILGATE_THRESHOLD = 0.1
    SPEED_THRESHOLD = 30
    VEHICLE_AHEAD_BRAKING_THRESHOLD = -0.1

    ACTIONS_ALL = {
        0: 'LANE_LEFT',
        1: 'IDLE',
        2: 'LANE_RIGHT',
        3: 'FASTER',
        4: 'SLOWER'
    }

    def __init__(self, verbose=False):
        self.verbose = verbose

    def print_obs(self, obs):
        ego_vehicle = obs[0]
        ego_x, ego_y, ego_v = ego_vehicle[1], ego_vehicle[2], ego_vehicle[3]
        print(f"Ego vehicle: x={ego_x}, y={ego_y}, v={ego_v}")

        for vehicle in obs[1:]:
            if vehicle[0] == 0:  # not present
                continue
            vehicle_x, vehicle_y, vehicle_v = vehicle[1], vehicle[2], vehicle[3]
            print(f"Vehicle: x={vehicle_x}, y={vehicle_y}, v={vehicle_v}")

    def get_vehicle_ahead(self, obs):
        ego_vehicle = obs[0]
        ego_x, ego_y = ego_vehicle[1], ego_vehicle[2]

        min_distance = float("inf")
        vehicle_ahead = None
        count = 1
        for vehicle in obs[1:]:
            vehicle_x, vehicle_y = vehicle[1], vehicle[2]
            # Check if vehicle is in the lane of ego vehicle
            if np.abs(vehicle_y - ego_y) > 0.5:
                continue

            # Check if vehicle is ahead of ego vehicle
            if vehicle_x > 0 and vehicle_x < min_distance:
                min_distance = vehicle_x
                vehicle_ahead = vehicle

            count += 1

        return vehicle_ahead

    def decide_action(self, action, obs, info):
        if isinstance(action, np.ndarray):
            action = int(action.item())

        ego_vehicle = obs[0]
        ego_x, ego_y, ego_vx, ego_vy = ego_vehicle[1], ego_vehicle[2], ego_vehicle[3], ego_vehicle[4]
        vehicle_ahead = self.get_vehicle_ahead(obs)

        if self.ACTIONS_ALL[action] in ['FASTER', 'IDLE']:
            if vehicle_ahead is not None:
                vehicle_ahead_x, vehicle_ahead_y, vehicle_ahead_vx, vehicle_ahead_vy = vehicle_ahead[1], vehicle_ahead[2], vehicle_ahead[3], vehicle_ahead[4]

                # VEHICLE_AHEAD_BRAKING_THRESHOLD
                if vehicle_ahead_vx < self.VEHICLE_AHEAD_BRAKING_THRESHOLD:
                    if self.verbose:
                        print("VEHICLE_AHEAD_BRAKING. Slowing down.")
                    action = 4
                elif vehicle_ahead_x < self.TAILGATE_THRESHOLD:
                    if self.verbose:
                        print(f"TAILGATING. Slowing down. Vehicle ahead x: {vehicle_ahead_x}")
                    action = 4
                elif ego_vx > self.SPEED_THRESHOLD:
                    if self.verbose:
                        print("SPEEDING. Slowing down.")
                    action = 4
                else:
                    if self.verbose:
                        print("No action needed.")

        return action