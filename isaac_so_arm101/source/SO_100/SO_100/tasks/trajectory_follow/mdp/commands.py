import torch
from isaaclab.envs import ManagerBasedRLEnv

def circular_trajectory_command(
    env: ManagerBasedRLEnv, radius: float = 0.1, speed: float = 0.05, height: float = 0.2
) -> torch.Tensor:
    # Calculate the time based on the current step count
    # env.step_count_for_env is the step count for each individual environment
    # Scale the step count to simulate a smooth, continuous path
    time = (env.step_count_for_env * speed) * (2.0 * torch.pi)

    # Compute X and Y position for the circle in the robot's base frame
    pos_x = radius * torch.cos(time)
    pos_y = radius * torch.sin(time)

    # Define the fixed Z height and orientation (roll, pitch, yaw)
    pos_z = torch.full_like(pos_x, height)
    roll = torch.zeros_like(pos_x)
    pitch = torch.zeros_like(pos_x)
    yaw = torch.zeros_like(pos_x)

    # Return the 6DOF command: [pos_x, pos_y, pos_z, roll, pitch, yaw]
    # The output size will be (num_envs, 6)
    command = torch.stack([pos_x, pos_y, pos_z, roll, pitch, yaw], dim=1)

    return command
