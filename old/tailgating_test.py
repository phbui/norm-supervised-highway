import pygame
from gymnasium.envs.registration import register
from highway_env.envs.highway_env import HighwayEnv
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.behavior import IDMVehicle
import random
from experiment_run import main

class RandomBrakingVehicle(IDMVehicle):
    """A vehicle that randomly brakes to simulate unpredictable behavior with more abrupt braking."""
    
    def __init__(self, road, position, heading=0, speed=0):
        # Call the parent class constructor first
        super().__init__(road, position, heading, speed)
        
        # Initialize target_speed and other attributes
        self.target_speed = speed if speed > 0 else 20  # Default to 20 if speed is 0
        self.normal_target_speed = self.target_speed  # Now safe to use
        self.braking_probability = 0.02  # Probability of random braking per step
        self.braking_duration = 5  # Shorter duration of braking in steps (was 15)
        self.braking_counter = 0
        self.braking_deceleration = -10  # Stronger deceleration when braking (was -5)
    
    @classmethod
    def create_from(cls, vehicle):
        """
        Create a new vehicle identical to the given vehicle.
        Overridden to handle the target_lane_index parameter.
        """
        v = cls(
            vehicle.road,
            vehicle.position,
            heading=vehicle.heading,
            speed=vehicle.speed
        )
        
        # Copy relevant attributes after initialization
        if hasattr(vehicle, 'target_speed'):
            v.target_speed = vehicle.target_speed
        if hasattr(vehicle, 'normal_target_speed'):
            v.normal_target_speed = vehicle.normal_target_speed
        else:
            v.normal_target_speed = v.target_speed
        
        # Copy other important attributes
        v.braking_counter = getattr(vehicle, 'braking_counter', 0)
        v.braking_probability = getattr(vehicle, 'braking_probability', 0.02)
        v.braking_duration = getattr(vehicle, 'braking_duration', 5)
        v.braking_deceleration = getattr(vehicle, 'braking_deceleration', -10)
        
        return v
    
    def act(self, action=None):
        # If currently braking, continue for the duration
        if self.braking_counter > 0:
            self.braking_counter -= 1
            # Apply more abrupt braking
            self.target_speed = max(3, self.normal_target_speed * 0.2)  # Reduce to 20% of normal speed (was 40%)
        # Otherwise, check if we should start braking
        elif random.random() < self.braking_probability:
            self.braking_counter = self.braking_duration
        else:
            # Normal driving
            self.target_speed = self.normal_target_speed
            
        # Call the parent class act method
        return super().act(action)

class CustomHighwayEnv(HighwayEnv):
    """
    A customized highway driving environment with controlled starting positions and more blocking vehicles.
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
            "vehicles_count": 23,
            "duration": 30,
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
            "vehicles_count": 20,  # Increased from 7 to create more traffic
            "controlled_vehicles": 1,
            "ego_spacing": 2,
            "vehicles_density": 2.0,  # Increased from 1.5
            
            # Custom Vehicle Placement with more blocking vehicles
            "custom_vehicle_placement": True,
            "vehicle_positions": [
                # Braking lead cars in ego's lane (lane 1)
                
                {"lane": 1, "position": 80, "speed": 23, "vehicle_type": "braking_car"},
                {"lane": 1, "position": 120, "speed": 23, "vehicle_type": "braking_car"},
                
                # Blocking vehicles in lane 0 (leftmost lane)
                {"lane": 0, "position": 10, "speed": 22, "vehicle_type": "truck"},
                {"lane": 0, "position": 25, "speed": 22, "vehicle_type": "truck"},
                {"lane": 0, "position": 45, "speed": 22, "vehicle_type": "truck"},
                {"lane": 0, "position": 60, "speed": 22, "vehicle_type": "truck"},
                {"lane": 0, "position": 85, "speed": 22, "vehicle_type": "truck"},
                {"lane": 0, "position": 110, "speed": 22, "vehicle_type": "truck"},
                {"lane": 0, "position": 135, "speed": 22, "vehicle_type": "truck"},
                
                # Blocking vehicles in lane 2 (right of ego)
                {"lane": 2, "position": 10, "speed": 22, "vehicle_type": "truck"},
                {"lane": 2, "position": 25, "speed": 22, "vehicle_type": "truck"},
                {"lane": 2, "position": 45, "speed": 22, "vehicle_type": "truck"},
                {"lane": 2, "position": 60, "speed": 22, "vehicle_type": "truck"},
                {"lane": 2, "position": 85, "speed": 22, "vehicle_type": "truck"},
                {"lane": 2, "position": 110, "speed": 22, "vehicle_type": "truck"},
                {"lane": 2, "position": 135, "speed": 22, "vehicle_type": "truck"},
                
                # Blocking vehicles in lane 3 (rightmost lane)
                {"lane": 3, "position": 10, "speed": 22, "vehicle_type": "truck"},
                {"lane": 3, "position": 25, "speed": 22, "vehicle_type": "bus"},
                {"lane": 3, "position": 45, "speed": 22, "vehicle_type": "bus"},
                {"lane": 3, "position": 60, "speed": 22, "vehicle_type": "bus"},
                {"lane": 3, "position": 85, "speed": 22, "vehicle_type": "bus"},
                {"lane": 3, "position": 110, "speed": 22, "vehicle_type": "bus"},
                {"lane": 3, "position": 135, "speed": 22, "vehicle_type": "bus"},
            
                
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
            speed=15  # Initial speed of ego vehicle
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

                if vehicle_type == "braking_car":
                    vehicle = RandomBrakingVehicle.make_on_lane(
                        self.road,
                        lane_index=lane_index,
                        longitudinal=position,
                        speed=speed
                    )
                    vehicle.target_speed = speed
                    vehicle.normal_target_speed = speed
                else:
                    vehicle = Vehicle.make_on_lane(
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
                elif vehicle_type == "bus":
                    vehicle.LENGTH = 10.0  # Very long vehicle
                    vehicle.WIDTH = 2.8
                    vehicle.target_speed = min(speed, 20)  # Buses are even slower
                else: pass
                self.road.vehicles.append(vehicle)

# Register the custom environment
register(
    id='custom-highway-v0',
    entry_point=CustomHighwayEnv,
)

pygame.init()

if __name__ == "__main__":
    main('custom-highway-v0')