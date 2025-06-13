import pygame
from gymnasium.envs.registration import register
from highway_env.envs.highway_env import HighwayEnv
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.controller import ControlledVehicle
from experiment_run import main

class LaneKeepingVehicle(ControlledVehicle):
    """A vehicle that simply keeps its lane and speed without IDM."""
    
    def __init__(self, road, position, heading=0, speed=0):
        super().__init__(road, position, heading, speed)
        self.fixed_speed = speed  # Store the desired fixed speed
        
    def act(self, action=None):
        """Just maintain lane and speed."""
        action = {'steering': 0, 'acceleration': self.maintain_speed()}
        super().act(action)  # Actually apply the action
        
    def maintain_speed(self):
        """Calculate acceleration to maintain target speed."""
        # Simple proportional controller to maintain speed
        return 0.1 * (self.fixed_speed - self.speed)
        
    @classmethod
    def create_from(cls, vehicle):
        """Create a new vehicle of this class from an existing vehicle."""
        v = cls(vehicle.road, vehicle.position, vehicle.heading, vehicle.speed)
        return v


class CustomHighwayEnv(HighwayEnv):
    """
    A customized highway driving environment with controlled starting positions.
    """
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics"
            },
            "action": {
                "type": "DiscreteMetaAction"
            },
            "lanes_count": 4,
            "vehicles_count": 3,
            "duration": 200,
            "initial_spacing": 2,
            "collision_reward": -1,
            "reward_speed_range": [20, 30],
            "simulation_frequency": 15,
            "policy_frequency": 1,
            "other_vehicles_type": "LaneKeepingVehicle",
            "screen_width": 600,
            "screen_height": 150,
            "centering_position": [0.3, 0.5],
            "scaling": 5.5,
            "show_trajectories": True,
            "render_agent": False,
            
            # Road and Environment
            "initial_lane_id": 1,  # Lane 1 (0-indexed)
            "initial_position": 30,  # Adjusted to position ego vehicle in traffic
            
            "frequency": 50,
            
            # Vehicle Configuration
            "vehicles_count": 24,  # More vehicles for chessboard pattern
            "controlled_vehicles": 1,
            "ego_spacing": 2,
            "vehicles_density": 1.5,
            
            # Custom Vehicle Placement
            "custom_vehicle_placement": True,
            "vehicle_positions": [
                # Ego vehicle will be in lane 1 (position specified elsewhere in config)
                
                # Slow vehicle directly in front of ego in lane 1 (middle lane)

                
                # Fast vehicle approaching from behind in lane 0 (left lane)
                {"lane": 1, "position": 75, "speed": 22, "vehicle_type": "slow_car"},
                
                # Vehicle ahead in lane 0 but with gap that looks inviting
                {"lane": 0, "position": 10, "speed": 25, "vehicle_type": "car"},
                
                # Vehicles in lane 2 (right lane) creating a similar situation
                {"lane": 2, "position": 10, "speed": 25, "vehicle_type": "car"},

                
                # Extra vehicles to create more complex traffic pattern
            ],
            
            # Other Parameters
            "offroad_terminal": True
        })
        return config

    def _create_vehicles(self) -> None:
        """Create vehicles with a chess board pattern to force lane switching."""
        # Create the ego vehicle (controlled vehicle)
        ego_vehicle = Vehicle.make_on_lane(
            self.road,
            lane_index=("0", "1", self.config["initial_lane_id"]),
            longitudinal=self.config["initial_position"],
            speed=25  # Initial speed of ego vehicle
        )
        
        # Convert to controlled vehicle type
        ego_vehicle = self.action_type.vehicle_class(
            self.road, ego_vehicle.position, ego_vehicle.heading, ego_vehicle.speed
        )
        
        self.controlled_vehicles = [ego_vehicle]
        self.road.vehicles.append(ego_vehicle)

        if self.config.get("custom_vehicle_placement", False):
            for vehicle_data in self.config["vehicle_positions"]:
                lane = vehicle_data["lane"]
                position = vehicle_data["position"]
                speed = vehicle_data["speed"]
                vehicle_type = vehicle_data.get("vehicle_type", "car")  # Default to car
                
                # Create basic vehicle
                lane_index = ("0", "1", lane)
                
                # Create a LaneKeepingVehicle instead of a standard vehicle
                vehicle = LaneKeepingVehicle(
                    self.road,
                    position=self.road.network.get_lane(lane_index).position(position, 0),
                    heading=self.road.network.get_lane(lane_index).heading_at(position),
                    speed=speed
                )
                vehicle.fixed_speed = speed  # Set the fixed speed
                vehicle.target_lane_index = lane_index  # Set target lane
                
                # Set vehicle type and appearance
                if vehicle_type == "slow_car":
                    # Slow car - might be a bit wider to make passing harder
                    vehicle.LENGTH = 5.0
                    vehicle.WIDTH = 2.2
                    vehicle.fixed_speed = speed  # Already slow from config
                elif vehicle_type == "fast_car":
                    # Fast car - normal size but higher speed
                    vehicle.LENGTH = 4.5
                    vehicle.WIDTH = 2.0
                    vehicle.fixed_speed = speed  # Already fast from config
                elif vehicle_type == "truck":
                    vehicle.LENGTH = 7.5  # Longer vehicle
                    vehicle.WIDTH = 2.5
                    vehicle.fixed_speed = min(speed, 25)  # Trucks are slower
                elif vehicle_type == "bus":
                    vehicle.LENGTH = 10.0  # Very long vehicle
                    vehicle.WIDTH = 2.8   # Wider
                    vehicle.fixed_speed = min(speed, 18)  # Buses are slower
                
                self.road.vehicles.append(vehicle)

# Register the custom environment
register(
    id='custom-highway-v0',
    entry_point=CustomHighwayEnv,
)

pygame.init()

if __name__ == "__main__":
    main('custom-highway-v0')