from gym.envs.registration import register

register(
    id="real_robot_challenge_phase_1-v1",
    entry_point="rrc_simulation.gym_wrapper.envs.cube_env:CubeEnv",
)
