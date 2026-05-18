import gymnasium as gym


gym.register(
    id="HCIS-ShellGame-SingleArm-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.shell_game_env_cfg:ShellGameEnvCfg",
    },
)
