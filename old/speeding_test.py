import pygame
from gymnasium.envs.registration import register
from highway_env.envs.highway_env import HighwayEnv
from highway_env.vehicle.kinematics import Vehicle
from experiment_run import main

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
            "vehicles_count": 5,
            "duration": 50,
            "initial_spacing": 2,
            "collision_reward": -1,
            "reward_speed_range": [20, 30],
            "simulation_frequency": 15,
            "policy_frequency": 1,
            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
            "screen_width": 600,
            "screen_height": 150,
            "centering_position": [0.3, 0.5],
            "scaling": 5.5,
            "show_trajectories": False,
            "render_agent": False,
            
            # Road and Environment
            "initial_lane_id": 1,  # Middle lane (0-indexed)
            "initial_position": 10,  # Adjusted to position ego vehicle in traffic
            # Keeping the same rewards as requested

            "frequency": 50,
            
            # Vehicle Configuration
            "vehicles_count": 5,
            "controlled_vehicles": 1,
            "ego_spacing": 2,
            "vehicles_density": 1.5,
            
            # Custom Vehicle Placement
            "custom_vehicle_placement": True,
            "vehicle_positions": [
                # Slow vehicle directly in front of ego in the middle lane
                {"lane": 1, "position": 60, "speed": 29, "vehicle_type": "car"},
                # Faster vehicles in the left lane (lane 0)
                {"lane": 0, "position": 45, "speed": 30, "vehicle_type": "car"},
                
                {"lane": 0, "position": 240, "speed": 30, "vehicle_type": "car"},
                # Faster vehicles in the right lane (lane 2)
                {"lane": 2, "position": 45, "speed": 30, "vehicle_type": "car"},
                {"lane": 3, "position": 45, "speed": 30, "vehicle_type": "car"},
            ],
            
            # Other Parameters
            "offroad_terminal": True
        })
        return config

    def _create_vehicles(self) -> None:
        """Create vehicles with custom starting positions."""
        # Create the ego vehicle (controlled vehicle)
        ego_vehicle = Vehicle.make_on_lane(
            self.road,
            lane_index=("0", "1", self.config["initial_lane_id"]),
            longitudinal=self.config["initial_position"],
            speed=30  # Initial speed of ego vehicle
        )
        
        # Convert to controlled vehicle type
        ego_vehicle = self.action_type.vehicle_class(
            self.road, ego_vehicle.position, ego_vehicle.heading, ego_vehicle.speed
        )
        
        # Set speed threshold for ego vehicle
        
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
                vehicle =Vehicle.make_on_lane(
                    self.road,
                    lane_index=lane_index,
                    longitudinal=position,
                    speed=speed
                )
                
                # Set vehicle type and appearance
                if vehicle_type == "truck":
                    vehicle.LENGTH = 7.5  # Longer vehicle
                    vehicle.WIDTH = 2.5
                    vehicle.target_speed = min(speed, 25)  # Trucks are slower
                elif vehicle_type == "emergency":
                    # Create an emergency vehicle (ambulance, police, etc.)
                    vehicle.LENGTH = 5.0
                    vehicle.WIDTH = 2.0
                    # In newer versions, you might be able to set special graphics
                    if hasattr(vehicle, "sprite_type"):
                        vehicle.sprite_type = "emergency"
                elif vehicle_type == "motorcycle":
                    vehicle.LENGTH = 2.5  # Smaller vehicle
                    vehicle.WIDTH = 1.0
                    vehicle.target_speed = speed * 1.1  # Motorcycles might be faster
                
                self.road.vehicles.append(vehicle)

# Register the custom environment
register(
    id='custom-highway-v0',
    entry_point=CustomHighwayEnv,
)

pygame.init()

if __name__ == "__main__":
    main('custom-highway-v0')