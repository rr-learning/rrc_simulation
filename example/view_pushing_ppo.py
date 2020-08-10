#!/usr/bin/env python3
from stable_baselines import PPO2
from example_pushing_training_env import ExamplePushingTrainingEnv
from example_pushing_training_env import FlatObservationWrapper

import argparse
import os
import gym
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="model path")
    parser.add_argument("--time_steps", required=True, help="time steps")

    args = vars(parser.parse_args())
    time_steps = int(args["time_steps"])
    model_path = str(args["model_path"])

    policy_path = os.path.join(
        model_path, "model_" + str(time_steps) + "_steps"
    )

    model = PPO2.load(policy_path)

    # define a method for the policy fn of your trained model
    def policy_fn(obs):
        return model.predict(obs, deterministic=True)[0]

    env = ExamplePushingTrainingEnv(frameskip=3, visualization=True)
    env = FlatObservationWrapper(env)

    for _ in range(10):
        obs = env.reset()
        done = False
        while not done:
            obs, rew, done, info = env.step(policy_fn(obs))
