# train.py

# !pip install stable-baselines3[extra]
# !pip install gymnasium[atari]
# !pip install autorom[accept-rom-license]
# !pip install fpdf

import os
import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
from fpdf import FPDF
from experiment_tracker import ExperimentTracker

# MEMBER INFO
MEMBER_NAME = "Deolinda Bio Bogore"

# HYPERPARAMETERS

TOTAL_EXPERIMENTS = 10
TRAINING_STEPS = 200000  # You can increase for better performance
LEARNING_RATES = [1e-4, 5e-4, 1e-3, 2e-4, 1e-4, 5e-5, 3e-4, 1e-3, 2e-3, 5e-4]
GAMMAS = [0.99, 0.95, 0.99, 0.98, 0.97, 0.99, 0.96, 0.95, 0.98, 0.97]
BATCH_SIZES = [32, 64, 32, 64, 32, 64, 32, 64, 32, 64]
EPSILON_STARTS = [1.0]*10
EPSILON_ENDS = [0.05]*10
EPSILON_DECAYS = [0.1, 0.1, 0.2, 0.15, 0.1, 0.05, 0.2, 0.15, 0.1, 0.1]

# CALLBACK
class RewardPlotCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []

    def _on_step(self):
        infos = self.locals.get("infos", [{}])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
        return True

    def save_plot(self, exp_number):
        os.makedirs(f"plots/{MEMBER_NAME.replace(' ', '_')}", exist_ok=True)
        plt.figure(figsize=(10,5))
        plt.plot(self.episode_rewards)
        plt.xlabel("Episodes")
        plt.ylabel("Reward")
        plt.title(f"Experiment {exp_number} Rewards")
        plt.grid()
        plt.savefig(f"plots/{MEMBER_NAME.replace(' ', '_')}/exp{exp_number}_reward.png")
        plt.close()

# TRAINING LOOP
def create_agent(lr, gamma, batch_size, eps_start, eps_end):
    env = make_atari_env("ALE/Pong-v5", n_envs=1, seed=42)
    env = VecFrameStack(env, n_stack=4)
    model = DQN(
        policy="CnnPolicy",
        env=env,
        learning_rate=lr,
        gamma=gamma,
        batch_size=batch_size,
        exploration_fraction=0.1,
        exploration_initial_eps=eps_start,
        exploration_final_eps=eps_end,
        buffer_size=50000,
        verbose=1,
        device="cpu",
    )
    return env, model

def save_model(model, exp_number):
    os.makedirs(f"models/{MEMBER_NAME.replace(' ', '_')}", exist_ok=True)
    model_path = f"models/{MEMBER_NAME.replace(' ', '_')}/dqn_exp{exp_number}.zip"
    model.save(model_path)

def generate_pdf(tracker):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, f"{MEMBER_NAME} - DQN Experiments Summary", 0, 1, 'C')
    pdf.ln(5)
    
    pdf.set_font("Arial", '', 12)
    # Table header
    pdf.cell(15, 8, "Exp#", 1)
    pdf.cell(25, 8, "LR", 1)
    pdf.cell(20, 8, "Gamma", 1)
    pdf.cell(25, 8, "Batch", 1)
    pdf.cell(45, 8, "Epsilon(start→end)", 1)
    pdf.cell(60, 8, "Observed Behavior", 1)
    pdf.ln()
    
    # Table content
    for exp in tracker.experiments:
        hp = exp['hyperparameters']
        pdf.cell(15, 8, str(exp['experiment_number']), 1)
        pdf.cell(25, 8, str(hp['learning_rate']), 1)
        pdf.cell(20, 8, str(hp['gamma']), 1)
        pdf.cell(25, 8, str(hp['batch_size']), 1)
        pdf.cell(45, 8, f"{hp['epsilon_start']}→{hp['epsilon_end']}", 1)
        pdf.cell(60, 8, exp['observed_behavior'], 1)
        pdf.ln()
    
    os.makedirs("pdf", exist_ok=True)
    pdf_file = f"pdf/{MEMBER_NAME.replace(' ', '_')}_experiment_table.pdf"
    pdf.output(pdf_file)
    print(f"PDF saved: {pdf_file}")

# MAIN LOOP
if __name__ == "__main__":
    tracker = ExperimentTracker(MEMBER_NAME)
    
    for exp in range(TOTAL_EXPERIMENTS):
        print(f"\n=== Running Experiment {exp+1}/{TOTAL_EXPERIMENTS} ===")
        
        lr = LEARNING_RATES[exp]
        gamma = GAMMAS[exp]
        batch_size = BATCH_SIZES[exp]
        eps_start = EPSILON_STARTS[exp]
        eps_end = EPSILON_ENDS[exp]
        eps_decay = EPSILON_DECAYS[exp]
        
        EXPERIMENT_DESCRIPTION = f"lr={lr}, gamma={gamma}, batch={batch_size}, eps:{eps_start}->{eps_end}"
        
        env, model = create_agent(lr, gamma, batch_size, eps_start, eps_end)
        callback = RewardPlotCallback()
        model.learn(total_timesteps=TRAINING_STEPS, callback=callback, progress_bar=True)
        
        # Save everything
        save_model(model, exp+1)
        callback.save_plot(exp+1)
        
        tracker.add_experiment(
            experiment_number=exp+1,
            learning_rate=lr,
            gamma=gamma,
            batch_size=batch_size,
            epsilon_start=eps_start,
            epsilon_end=eps_end,
            epsilon_decay=eps_decay,
            observed_behavior="Training completed",
            notes=EXPERIMENT_DESCRIPTION
        )
        
        env.close()
    
    # Generate PDF table at the end
    generate_pdf(tracker)
    print("\nAll experiments completed successfully!")
