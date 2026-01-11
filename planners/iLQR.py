from planners.planner import Planner
import numpy as np
import yaml

from planners.interpolators.quintic_spline import Quintic_Spline_Interpolator

class iLQR_Planner(Planner):

    def __init__(self, cfg):
        super().__init__(cfg)
        with open("config/planners/iLQR.yaml", "r") as f:
            self.planner_params = yaml.safe_load(f)

        self.voxel_resolution = self.planner_params["grid"]["resolution"]


        self.waypoints =  self.sim_params["world"]["waypoints"]  #List of dictionaries
        self.obstacles =  self.sim_params["obstacles"]["static"]
        self.arm_length = self.sim_params["quadcopter"]["arm_length"]

        self.trajectory_generator = Quintic_Spline_Interpolator(cfg)
        
        self.planner_name = "iLQR"
        self.interpolator_name = "Quintic Spline"

    def calculate_trajectory(self): #-> str:
        """Calculate and return time parameterized Trajectory"""

        trajectory = self.trajectory_generator.interpolate_waypoints(planner_waypoints)

        return trajectory, self.trajectory_generator #To evaluate trajectories, feed to controller
    


