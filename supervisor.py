import numpy as np

class Supervisor:
    TAILGATE_THRESHOLD = 0.12 # about 3 car lengths
    SPEED_THRESHOLD = 30
    VEHICLE_AHEAD_BRAKING_THRESHOLD = -0.1
    CUT_OFF_THRESHOLD = 0.08 # about 2 car lengths
    DEFENSIVE_TRHESHOLD = 0.05 # about 1 car length

    ACTIONS_ALL = {
        0: 'LANE_LEFT',
        1: 'IDLE',
        2: 'LANE_RIGHT',
        3: 'FASTER',
        4: 'SLOWER'
    }

    def __init__(self, env_config, verbose=False):
        self.env_config = env_config
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
        num_lanes = self.env_config['lanes_count']
        ego_vehicle = obs[0]
        ego_x, ego_y = ego_vehicle[1], ego_vehicle[2]

        min_distance = float("inf")
        vehicle_ahead = None
        count = 1
        for vehicle in obs[1:]:
            vehicle_x, vehicle_y = vehicle[1], vehicle[2]
            # if not in same lane, skip
            if np.abs(vehicle_y) > 1 / (num_lanes*2): # could probably also work by checking if abs of number is close to 0 
                continue

            # Check if vehicle is ahead of ego vehicle
            if vehicle_x > 0 and vehicle_x < min_distance:
                min_distance = vehicle_x
                vehicle_ahead = vehicle

            count += 1

        return vehicle_ahead
    
    def check_out_of_bounds_lane_change(self, action, obs):
        num_lanes = self.env_config['lanes_count']
        if action == 0:  # LANE_LEFT
            if obs[0][2] < (1 / num_lanes):  # ego vehicle is in the outer lane
                return True
        elif action == 2:  # LANE_RIGHT
            if obs[0][2] >= 1 - (1 / num_lanes) :  # ego vehicle is in the outer lane
                return True
        
        return False
    

    def check_lane_change_cutoff(self, action, obs):
        num_lanes = self.env_config['lanes_count']
        ego_vehicle = obs[0]
        ego_x, ego_y = ego_vehicle[1], ego_vehicle[2]

        # Define the lane range of relative y values based on the action
        lane_y_range = float
        if action == 0: # LANE_LEFT
            lane_y_range =  [(-1/num_lanes), (-1/(num_lanes*2))]
        elif action == 2: # LANE_RIGHT
            lane_y_range = [(1/(num_lanes*2)), 1/(num_lanes)] 

        def in_range(value, range_bounds, inclusive=True):
            low, high = min(range_bounds), max(range_bounds)
            return low <= value <= high if inclusive else low < value < high

        for vehicle in obs[1:]:
            vehicle_x, vehicle_y = vehicle[1], vehicle[2]
            if in_range(vehicle_y, lane_y_range):
                # Check if the vehicle is within the cut-off threshold
                if np.abs(vehicle_x) < self.CUT_OFF_THRESHOLD: # cut-off is bidirectional
                    return True
        
        return False
               
    def decide_action(self, action, obs, info):
        violations = 0 
        if isinstance(action, np.ndarray):
            action = int(action.item())

        ego_vehicle = obs[0]
        ego_x, ego_y, ego_vx, ego_vy = ego_vehicle[1], ego_vehicle[2], ego_vehicle[3], ego_vehicle[4]
        vehicle_ahead = self.get_vehicle_ahead(obs)
        
        if self.check_out_of_bounds_lane_change(action, obs): # Not a norm violation, just a bug from non restricted action space
            action = 4  # GO-SLOWER fallback

        if self.verbose:
            print(f"Action: {self.ACTIONS_ALL[action]}")

        if self.ACTIONS_ALL[action] in ['FASTER', 'IDLE']:
            if vehicle_ahead is not None:
                vehicle_ahead_x, vehicle_ahead_y, vehicle_ahead_vx, vehicle_ahead_vy = vehicle_ahead[1], vehicle_ahead[2], vehicle_ahead[3], vehicle_ahead[4]

                # VEHICLE_AHEAD_BRAKING_THRESHOLD
                if vehicle_ahead_vx < self.VEHICLE_AHEAD_BRAKING_THRESHOLD:
                    if self.verbose:
                        print("VEHICLE_AHEAD_BRAKING. Slowing down.") # should consider switching lanes if within certain distance 
                    action = 4 # GO-SLOWER fallback
                    violations += 1
                # TAILGATE_THRESHOLD
                elif vehicle_ahead_x < self.TAILGATE_THRESHOLD:
                    if self.verbose:
                        print(f"TAILGATING. Slowing down. Vehicle ahead x: {vehicle_ahead_x}")
                    action = 4 # GO-SLOWER fallback
                    violations += 1
                # SPEED_THRESHOLD
                elif ego_vx > self.SPEED_THRESHOLD:
                    if self.verbose:
                        print("SPEEDING. Slowing down.")
                    action = 4 # GO-SLOWER fallback
                    violations += 1
        
        elif self.ACTIONS_ALL[action] in ['LANE_LEFT', 'LANE_RIGHT']:
            # LANE_CHANGE_CUT_OFF
            if self.check_lane_change_cutoff(action, obs): 
                if vehicle_ahead is not None and vehicle_ahead[0] != 0 and vehicle_ahead[1] < self.DEFENSIVE_TRHESHOLD and vehicle_ahead[3]:  # vehicle ahead is present and within the defensive threshold.
                    if self.verbose:
                        print(f"DEFENSIVE LANE CHANGE. Vehicle ahead x: {vehicle_ahead_x}") # not perfect because accelerating up to a vehicle and then cutting someone off should be a violation
                else: 
                    if self.verbose:
                        print("LANE_CHANGE_CUT_OFF. Slowing down.")

                    action = 4 # GO-SLOWER fallback
                    violations += 1
                    
        return action, violations
