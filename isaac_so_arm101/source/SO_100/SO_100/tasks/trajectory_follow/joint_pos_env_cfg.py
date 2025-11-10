from isaaclab.utils import configclass
from isaaclab.managers import CommandTermCfg, RewardTermCfg as RewTerm, SceneEntityCfg
import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
# Import base robot config
from SO_100.robots import SO_ARM100_CFG
# Import base environment, command, and reward configs from the reach task
from SO_100.tasks.reach.reach_env_cfg import ReachEnvCfg, CommandsCfg, RewardsCfg

@configclass
class TrajectoryFollowCommandsCfg(CommandsCfg):
    def __post_init__(self):
        # 1. Disable the old static end-effector pose command
        # This prevents the robot from receiving the old random goal commands.
        self.ee_pose = None

    # 2. Define the new dynamic trajectory command
    trajectory_target = CommandTermCfg(
        # The custom function that generates the next point in the circle
        func=mdp.circular_trajectory_command,
        # IMPORTANT: Set to (0.0, 0.0) for a dynamic target that updates every step
        resampling_time_range=(0.0, 0.0),
        # Parameters for the circular trajectory
        params={
            "radius": 0.15,
            "speed": 0.005,
            "height": 0.25
        },
        debug_vis=True,
    )

##
# Trajectory Following Rewards
##

@configclass
class TrajectoryFollowRewardsCfg(RewardsCfg):
    def __post_init__(self):
        # 1. Disable the original reach rewards by setting their weights to 0.0
        self.end_effector_position_tracking.weight = 0.0
        self.end_effector_position_tracking_fine_grained.weight = 0.0
        self.end_effector_orientation_tracking.weight = 0.0

    # 2. Add a new reward for tracking the dynamic trajectory
    trajectory_tracking = RewTerm(
        # Uses the existing utility function from the reach MDP
        func=mdp.position_command_error_tanh,
        weight=1.0,  # High weight to prioritize tracking
        params={
            # Target the End-Effector body
            "asset_cfg": SceneEntityCfg("robot", body_names="Fixed_Gripper"),
            "std": 0.05,
            # Reference the new command name
            "command_name": "trajectory_target"
        },
    )

@configclass
class SoArm100TrajFollowEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set the correct robot
        self.scene.robot = SO_ARM100_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Keep the existing JointPositionActionCfg for arm control
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["Shoulder_Rotation", "Shoulder_Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"],
            scale=0.5,
            use_default_offset=True,
        )
        self.actions.gripper_action = None
        
        # Override the default Commands and Rewards with the custom trajectory logic
        self.commands: TrajectoryFollowCommandsCfg = TrajectoryFollowCommandsCfg()
        self.rewards: TrajectoryFollowRewardsCfg = TrajectoryFollowRewardsCfg()

        # Optional: Increase episode length for trajectory following
        self.episode_length_s = 24.0

@configclass
class SoArm100TrajFollowEnvCfg_PLAY(SoArm100TrajFollowEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
