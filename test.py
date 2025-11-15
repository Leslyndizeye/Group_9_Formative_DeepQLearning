# !pip install stable-baselines3[extra]
# !pip install gymnasium[atari]
# !pip install autorom[accept-rom-license]

import os
import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
from experiment_tracker import ExperimentTracker

MEMBER_NAME = "Ndizeye Lesly"

LEARNING_RATE = 1e-4
GAMMA = 0.99
BATCH_SIZE = 32
EPSILON_START = 1.0
EPSILON_END = 0.05
EXPLORATION_FRACTION = 0.1

EXPERIMENT_NUMBER = 1
EXPERIMENT_DESCRIPTION = "Baseline CNN training"


class RewardPlotCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []

    def _on_step(self):
        info_list = self.locals.get("infos", [{}])
        for info in info_list:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
        return True

    def save_plot(self, exp_number, name):
        os.makedirs("plots", exist_ok=True)
        plt.figure(figsize=(10, 5))
        plt.plot(self.episode_rewards)
        plt.xlabel("Episodes")
        plt.ylabel("Reward")
        plt.title(f"Experiment {exp_number} Rewards - {name}")
        plt.grid()
        plt.savefig(f"plots/exp{exp_number}_{name.replace(' ', '_')}.png")
        plt.close()


def create_agent():
    print(f"\nRunning Experiment {EXPERIMENT_NUMBER}: {EXPERIMENT_DESCRIPTION}")
    env = make_atari_env("ALE/Pong-v5", n_envs=1, seed=42)
    env = VecFrameStack(env, n_stack=4)

    model = DQN(
        policy="CnnPolicy",
        env=env,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        batch_size=BATCH_SIZE,
        exploration_fraction=EXPLORATION_FRACTION,
        exploration_initial_eps=EPSILON_START,
        exploration_final_eps=EPSILON_END,
        buffer_size=50000,
        verbose=1,
        device="cpu",
    )
    return env, model


def train(model, env, steps=200000):
    callback = RewardPlotCallback()
    model.learn(total_timesteps=steps, callback=callback, progress_bar=True)
    return model, callback


def save_results(model, env, callback):
    os.makedirs("models", exist_ok=True)
    model_path = f"models/dqn_exp{EXPERIMENT_NUMBER}_{MEMBER_NAME.replace(' ', '_')}"
    model.save(model_path)

    callback.save_plot(EXPERIMENT_NUMBER, MEMBER_NAME)

    tracker = ExperimentTracker(MEMBER_NAME)
    tracker.add_experiment(
        experiment_number=EXPERIMENT_NUMBER,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        batch_size=BATCH_SIZE,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay=0.1,
        observed_behavior="Training completed",
        notes=EXPERIMENT_DESCRIPTION,
    )

    env.close()


if __name__ == "__main__":
    try:
        env, agent = create_agent()
        agent, callback = train(agent, env)
        save_results(agent, env, callback)

        print("\nExperiment finished successfully.\n")

    except Exception as e:
        print(f"Error: {e}")
        raise