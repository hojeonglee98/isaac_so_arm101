import gymnasium as gym

from . import agents  # Import agents package so the RSL-RL config is available

gym.register(
    id="SO-ARM100-TrajFollow-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        # Point to your custom environment configuration
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm100TrajFollowEnvCfg",
        # Point to your custom RSL-RL agent configuration
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TrajFollowPPORunnerCfg",
    },
    disable_env_checker=True,
)
