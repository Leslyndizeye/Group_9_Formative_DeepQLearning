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
import gc
from experiment_tracker import ExperimentTracker

# MEMBER INFO
MEMBER_NAME = "Nicolas Muhigi"

# HYPERPARAMETERS
TOTAL_EXPERIMENTS = 9
TRAINING_STEPS = 300000

# Experiment hyperparameters (index 0 = Experiment 2)
LEARNING_RATES = [1e-3, 5e-5, 2e-4, 7e-5, 3e-4, 2e-5, 1e-4, 8e-4, 3e-4]
GAMMAS = [0.90, 0.99, 0.995, 0.999, 0.96, 0.98, 0.99, 0.92, 0.995]
BATCH_SIZES = [64, 8, 128, 32, 32, 48, 32, 64, 32]
EPSILON_STARTS = [1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 0.3, 1.0]
EPSILON_ENDS = [0.2, 0.05, 0.01, 0.05, 0.6, 0.02, 0.005, 0.01, 0.02]
EXPLORATION_FRACTIONS = [0.15, 0.6, 0.05, 0.05, 0.8, 0.9, 0.08, 0.05, 0.15]

EXPERIMENT_DESCRIPTIONS = [
    "High learning rate prioritizing short-term rewards; fast but unstable learning",
    "Tiny batch size with long exploration phase; noisy but robust learning",
    "Large batch with rapid epsilon decay; stable gradients but fast convergence",
    "Semi-greedy early behavior; high gamma for long-term reward optimization",
    "Highly exploratory agent; tests wide state-space coverage",
    "Very slow epsilon decay; agent stays random for most of training",
    "Very greedy final policy; strong focus on exploitation performance",
    "High learning rate with low exploration; tests unstable but fast learners",
    "Balanced baseline setup; smooth epsilon decay with high gamma"
]

# -----------------------------------------------------
# Callback to track rewards and save plots
# -----------------------------------------------------

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

# -----------------------------------------------------
# Agent creation
# -----------------------------------------------------

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

# -----------------------------------------------------
# Training
# -----------------------------------------------------

def train(model, env, steps=TRAINING_STEPS):
    callback = RewardPlotCallback()
    model.learn(total_timesteps=steps, callback=callback, progress_bar=True)
    return model, callback

# -----------------------------------------------------
# Save results
# -----------------------------------------------------

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
    del env, model, callback
    gc.collect()

# -----------------------------------------------------
# Main Loop
# -----------------------------------------------------

if __name__ == "__main__":
    for i in range(TOTAL_EXPERIMENTS):
        try:
            global EXPERIMENT_NUMBER
            EXPERIMENT_NUMBER = i + 2  # your numbering: 2–10

            EXPERIMENT_DESCRIPTION = EXPERIMENT_DESCRIPTIONS[i]
            LEARNING_RATE = LEARNING_RATES[i]
            GAMMA = GAMMAS[i]
            BATCH_SIZE = BATCH_SIZES[i]
            EPSILON_START = EPSILON_STARTS[i]
            EPSILON_END = EPSILON_ENDS[i]
            EXPLORATION_FRACTION = EXPLORATION_FRACTIONS[i]

            env, agent = create_agent()
            agent, callback = train(agent, env)
            save_results(agent, env, callback)

            print(f"\nExperiment {EXPERIMENT_NUMBER} finished successfully.\n")

        except Exception as e:
            print(f"Error in Experiment {EXPERIMENT_NUMBER}: {e}")
            raise
