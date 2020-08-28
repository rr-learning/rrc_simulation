#!/usr/bin/env python3
"""Example evaluation script to evaluate a policy.

This is an example evaluation script which loads a policy trained with PPO. If
this script were moved into the top rrc_simulation folder (since this is where
we will execute the rrc_evaluate command), it would consistute a valid
submission (naturally, imports below would have to be adjusted accordingly).

This script will be executed in an automated procedure.  For this to work, make
sure you do not change the overall structure of the script!

This script expects the following arguments in the given order:
 - Difficulty level (needed for reward computation)
 - initial pose of the cube (as JSON string)
 - goal pose of the cube (as JSON string)
 - file to which the action log is written

It is then expected to initialize the environment with the given initial pose
and execute exactly one episode with the policy that is to be evaluated.

When finished, the action log, which is created by the TriFingerPlatform class,
is written to the specified file.  This log file is crucial as it is used to
evaluate the actual performance of the policy.
"""
import sys
import os

import gym

from rrc_simulation.gym_wrapper.envs import cube_env
from rrc_simulation.tasks import move_cube



class RandomPolicy:
    """Dummy policy which uses random actions."""

    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, observation):
        return self.action_space.sample()


def main():
    initializer = cube_env.RandomInitializer(difficulty=1)

    # if difficulty == 1 (i.e. pushing), we load the policy we trained for that
    # task. otherwise, we just use the RandomPolicy as placeholder. Naturally,
    # when you submit you would have a policy for each difficulty level.

    env1 = gym.make(
        "rrc_simulation.gym_wrapper:real_robot_challenge_phase_1-v1",
        initializer=initializer,
        action_type=cube_env.ActionType.POSITION,
        visualization=True,
    )

    env2 = gym.make(
        "rrc_simulation.gym_wrapper:real_robot_challenge_phase_1-v1",
        initializer=initializer,
        action_type=cube_env.ActionType.POSITION,
        visualization=False,
    )


    envs = [env1, env2]

    for env in envs:
        env.reset()

    for i in range(10000):
        for env in envs:
            action = env.action_space.sample()
            observation, reward, is_done, info = env.step(action)




if __name__ == "__main__":
    main()
