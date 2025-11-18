# Deep Q-Learning for Atari [Pong]

## Overview
This project implements a Deep Q-Network (DQN) agent to play the Atari Pong (ALE/Pong-v5) using Stable Baselines3 and Gymnasium. The implementation includes:

Task 1: Training script (train.py) that trains a DQN agent using CNN policy, compares performance against MLP policy, and performs 10 hyperparameter experiments per team member.

Task 2: Playing script (play.py) that loads the trained model and evaluates the agent using GreedyQPolicy, rendering gameplay in real time.

Hyperparameter tuning documentation: Complete table of all 40 experiments conducted by 4 team members.

Video demonstration: Short clip showing the trained agent playing Pong using the play.py script.

## Team Members
- **Nicolas Muhigi**
- **Leslie Isaro**
- **Deolinda Bogore**
- **Lesly Ndizeye**

## Environment: 
Pong is a classic two-player table tennis game. You control the right paddle and compete against the left paddle controlled by the computer. The goal is to score points by making the ball pass the opponent’s paddle while defending your own. The challenge for the DQN agent lies in predicting the ball trajectory, timing paddle movements, and reacting quickly, requiring both short-term reflexes and long-term strategy.

Environment ID: ALE/Pong-v5

Action Space: Discrete(6) - NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE

For training, typically only NOOP, UP, and DOWN are used.

Observation Space: Box(0, 255, (210, 160, 3), uint8) - RGB frames

Preprocessing: 84x84 grayscale, 4 frames stacked

Difficulty: Mode 0, Difficulty 0 (default)

---

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip package manager
- At least 2GB free disk space

### Installation Steps

#### 1. Clone the Repository
```bash
git clone https://github.com/Leslyndizeye/Group_9_Formative_DeepQLearning.git
cd Group_9_Formative_DeepQLearning
```

#### 2. Create Virtual Environment

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

**Note for macOS users**: If you encounter issues with square brackets, use quotes:
```bash
pip install "gymnasium[atari,accept-rom-license]"
```

#### 4. Verify Installation
```bash
python check_setup.py
```

You should see:
```
✓ SETUP COMPLETE - Ready to train!
```

---

## Usage

### Task 1: Training the Agent

#### Basic Training
Train with default hyperparameters:
```bash
python train.py --timesteps 1000000
```

#### Policy Comparison
Compare CNNPolicy and MLPPolicy:

**CNN Policy:**
```bash
python train.py --policy CnnPolicy --timesteps 500000
```

**MLP Policy:**
```bash
python train.py --policy MlpPolicy --timesteps 500000
```


**Key Training Logs:**
- Episode rewards (reward trends)
- Episode lengths
- Exploration rate (epsilon)
- Q-value estimates
- Training loss

---

### Task 2: Playing with Trained Agent

#### Watch Agent Play
```bash
python play.py --episodes 5
```

This will:
- Load the trained model from `dqn_model.zip`
- Use GreedyQPolicy (deterministic=True) for evaluation
- Display the game window
- Show episode rewards and lengths

#### Play with Specific Model
```bash
python play.py --model-path ./models/experiment_01/dqn_model.zip --episodes 5
```

#### Record Video
```bash
python play.py --record --episodes 5 --video-folder ./videos
```

Videos will be saved in the `videos/` directory.

---

## Team Member Experiments

### Nicolas Muhigi's Experiments

#### Experimental Design
I conducted 10 comprehensive hyperparameter experiments exploring extreme learning rate ranges, diverse gamma values, and varied exploration strategies. The experiments systematically tested the boundaries of stable DQN training, from ultra-conservative to highly aggressive configurations.

### Nicolas Muhigi — Experiments


| Exp | lr      | gamma | batch | eps_start | eps_end | eps_decay | Steps   | Final Reward | Notes |
|-----|---------|-------|-------|-----------|---------|-----------|--------|--------------|-------|
| 1   | 1e-06   | 0.999 | 32    | 0.9       | 0.01    | 0.1       | 300000 | -21          | Very slow learning; focuses on long-term rewards. |
| 2   | 0.001   | 0.9   | 64    | 1.0       | 0.2     | 0.1       | 300000 | -21          | Fast but unstable; short-term reward focused. |
| 3   | 5e-05   | 0.99  | 8     | 1.0       | 0.05    | 0.1       | 300000 | -21          | Small batch, long exploration; robust but noisy learning. |
| 4   | 0.0002  | 0.995 | 128   | 1.0       | 0.01    | 0.1       | 300000 | -21          | Large batch, rapid decay; stable but converges fast. |
| 5   | 7e-05   | 0.999 | 32    | 0.5       | 0.05    | 0.1       | 300000 | -21          | Semi-greedy start; emphasizes long-term rewards. |
| 6   | 0.0003  | 0.96  | 32    | 1.0       | 0.6     | 0.1       | 300000 | -21          | Highly exploratory; covers wide state-space. |
| 7   | 2e-05   | 0.98  | 48    | 1.0       | 0.02    | 0.1       | 300000 | -21          | Slow epsilon decay; random behavior lasts longer. |
| 8   | 0.0001  | 0.99  | 32    | 1.0       | 0.005   | 0.1       | 300000 | -21          | Very greedy policy; strong exploitation. |
| 9   | 0.0008  | 0.92  | 64    | 0.3       | 0.01    | 0.1       | 300000 | -21          | Low exploration, high LR; unstable fast learning. |
| 10  | 0.0001  | 0.99  | 32    | 1.0       | 0.02    | 0.1       | 250000 | -21          | Balanced LR and gamma; steady exploration and learning. |

### Insight and Overall Takeaways

The experiments show that hyperparameters strongly affect DQN learning: higher learning rates speed up training but can be unstable, while lower rates favor slow, steady convergence; high gamma values improve long-term reward optimization, whereas lower gamma prioritizes short-term gains; small batch sizes create noisy but robust learning, while large batches stabilize updates but converge faster; slow epsilon decay prolongs exploration, and fast decay shifts quickly to exploitation. Overall, careful tuning of learning rate, gamma, batch size, and epsilon schedule is essential to balance stability, speed, and exploration, and no single configuration dominates all scenarios, highlighting the importance of task-specific adjustments.


**To Run Nicolas's Experiments:**
```bash
python play.py --model-path "models/Nicolas Muhigi/dqn_exp3_Nicolas_Muhigi.zip"
```

---

### Leslie's Experiments

| **Exp #** | **Description** | **Learning Rate** | **Gamma** | **Batch Size** | **Epsilon Start** | **Epsilon End** | **Exploration Fraction** | **Timesteps** | **Final Reward** | **Noted Behavior** |
|:--:|:--|--:|--:|--:|--:|--:|--:|--:|--:|--|
| **1** | Lower LR + longer exploration + low epsilon end | 5e-5 | 0.99 | 32 | 1.0 | 0.02 | 0.20 | 200k | −21 | Mostly random movement; agent failed to learn meaningful actions. |
| **2** | Mid LR, higher gamma, bigger batch | 7.5e-5 | 0.995 | 64 | 1.0 | 0.02 | 0.30 | 100k | −21 | Limited paddle control; slow and inconsistent progress. |
| **3** | Higher gamma + longer exploration | 1e-4 | 0.997 | 32 | 1.0 | 0.03 | 0.15 | 200k | −21 | Started following the ball occasionally; minor improvement. |
| **4** | Small batch, faster target updates | 1e-4 | 0.99 | 16 | 1.0 | 0.05 | 0.10 | 50k | −21 | Unstable training; paddle moved erratically and missed often. |
| **5** | Lower gamma (short-term reward) | 1e-4 | 0.95 | 32 | 1.0 | 0.05 | 0.10 | 50k | −21 | Agent reacted fast but lacked long-term control; poor play. |
| **6** | Higher LR + larger batch | 2.5e-4 | 0.99 | 64 | 1.0 | 0.05 | 0.30 | 200k | −21 | Learned faster but unstable; paddle overshot ball frequently. |
| **7** | Higher gamma + short exploration | 1e-4 | 0.999 | 32 | 1.0 | 0.05 | 0.05 | 200k | −21 | Improved tracking; smoother motion and longer volleys. |
| **8** | Lower learning rate | 1e-4 | 0.99 | 32 | 1.0 | 0.01 | 0.10 | 200k | −21 | Plateaued early; agent stopped adapting mid-training. |
| **9** | High gradient steps per update | 1e-4 | 0.99 | 32 | 1.0 | 0.01 | 0.10 | 200k | −21 | Slightly better tracking; some over-adjustment during volleys. |
| **10** | Low gamma, fast updates (**Best Model**) | 2e-4 | 0.97 | 32 | 1.0 | 0.03 | 0.10 | 200k | −20 | Quick reactions, stable paddle control, and consistent rallies. |


---

### Insights and Discussion

- **Learning Rate:** Too low caused no learning; moderate rates (1e-4 – 2e-4) gave faster and steadier improvement.  
- **Gamma:** Higher values (0.997 – 0.999) improved long-term strategy, while lower values made play short-sighted.  
- **Batch Size:** Medium (32 – 64) balanced update stability and learning speed; very small batches caused noise.  
- **Exploration:** Balanced decay (0.10 – 0.15) worked best. Too long kept the agent random, too short froze learning.  
- **Training Duration:** Only runs with 200 k steps showed meaningful progress; shorter runs failed to learn control.  
- **Best Configuration:** *Experiment 10* (LR = 2e-4, Gamma = 0.97, Batch = 32) achieved the most natural, responsive gameplay.

---

### Deolinda Bio Bogore's Experiments

#### Experimental Design
Experimental Design

For my experiments, I trained a DQN agent to play Pong (ALE/Pong-v5) using Stable-Baselines3 and CnnPolicy. I conducted 10 experiments with different hyperparameter combinations, varying learning rate, gamma, batch size, and epsilon decay. Each experiment ran for 100,000 training steps, and I tracked the agent’s reward per episode to observe its learning behavior.

I used a frame-stacked vectorized Atari environment (VecFrameStack) with 4 frames per observation to help the network capture motion dynamics. The RewardPlotCallback was used to log episode rewards and generate reward trend plots.

| Exp # | Learning Rate (lr) | Gamma (γ) | Batch Size | Epsilon (start→end) | Epsilon Decay | Observed Behavior                                                     | Final Reward |
| ----- | ------------------ | --------- | ---------- | ------------------- | ------------- | --------------------------------------------------------------------- | ------------ |
| 1     | 0.0001             | 0.99      | 32         | 1.0 → 0.05          | 0.1           | Agent collapsed immediately; very slow learning due to low LR.        | -21.0        |
| 2     | 0.0005             | 0.95      | 64         | 1.0 → 0.05          | 0.1           | Agent lost all games; moderate LR and low gamma slowed learning.      | -21.0        |
| 3     | 0.001              | 0.99      | 32         | 1.0 → 0.05          | 0.2           | High LR caused unstable Q-values; mostly random actions.              | -21.0        |
| 4     | 0.0002             | 0.98      | 64         | 1.0 → 0.05          | 0.15          | Slight improvement; agent occasionally hit the ball.                  | -15.0        |
| 5     | 0.0001             | 0.97      | 32         | 1.0 → 0.05          | 0.1           | Agent collapsed; gamma too low to learn sequences.                    | -21.0        |
| 6     | 0.00005            | 0.99      | 64         | 1.0 → 0.05          | 0.05          | Extremely slow learning; agent mostly random.                         | -20.0        |
| 7     | 0.0003             | 0.96      | 32         | 1.0 → 0.05          | 0.2           | Minor improvement; occasionally hits the ball, performance unstable.  | -18.0        |
| 8     | 0.001              | 0.95      | 64         | 1.0 → 0.05          | 0.15          | High LR + low gamma; agent random, no meaningful learning.            | -21.0        |
| 9     | 0.002              | 0.98      | 32         | 1.0 → 0.05          | 0.1           | Too high LR; agent fails to learn paddle control.                     | -21.0        |
| 10    | 0.0005             | 0.97      | 64         | 1.0 → 0.05          | 0.1           | Moderate LR and gamma; still insufficient learning within 100k steps. | -21.0        |

#### Observed Results
Across all experiments, the agent failed to achieve positive rewards within 100,000 training steps.

Minor improvements were observed in experiments 4 and 7, where the agent occasionally hit the ball but still mostly lost (-15, -18).

High learning rates (≥0.001) caused unstable Q-values, resulting in random behavior.

Low learning rates (≤0.0001) led to extremely slow learning and immediate collapse.
#### Analysis Summary
The results confirm that 100,000 steps are insufficient for Pong using DQN with CNN policies.

Optimal hyperparameters are likely moderate learning rates (0.0002–0.0005) and gamma (0.96–0.98), but longer training (≥1 million steps) is needed to see meaningful learning.

The epsilon schedule did not allow enough exploration due to fast decay relative to the short training steps.

Overall, the agent mostly collapsed or performed poorly, but the experiments provide insights into hyperparameter sensitivity and the effect of learning rate, gamma, batch size, and epsilon on performance

---

## Lesly Ndizeye — Experiments & Instructions

### Summary

Lesly ran 10 DQN experiments on **Pong** (Atari) exploring learning rate, gamma, batch size and epsilon schedules. The goal was to find a stable configuration that learns reliably on CPU in reasonable time. The best model is highlighted below and saved under `models/dqn_model_10.zip`.

---

### Experiments table

|           Exp | Learning Rate |   Gamma  |  Batch | Eps Start → End | Exploration Fraction | Timesteps | **Final Reward** | Notes                                                  |
| ------------: | :-----------: | :------: | :----: | :-------------: | :------------------: | :-------: | :--------------: | ------------------------------------------------------ |
|         **1** |      1e-4     |   0.99   |   32   |    1.0 → 0.05   |         0.10         |    100k   |     **–21**     | Baseline, slight improvement from random play.         |
|         **2** |      5e-5     |   0.994  |   32   |    1.0 → 0.05   |         0.15         |    100k   |     **–21**     | Smoother curves, slightly better than Exp 1.           |
|         **3** |      2e-4     |   0.99   |   32   |    1.0 → 0.02   |         0.08         |    150k   |     **–21**     | Faster learning, unstable towards the end.             |
|         **4** |      1e-4     |   0.997  |   32   |    1.0 → 0.05   |         0.12         |    150k   |     **–20**     | Improved rally consistency.                            |
|         **5** |      8e-5     |   0.99   |   64   |    1.0 → 0.05   |         0.10         |    100k   |     **–21**     | Larger batch improved stability but learned slower.    |
|         **6** |     1.2e-4    |   0.996  |   32   |    1.0 → 0.03   |         0.10         |    200k   |     **–20**     | Strong improvement — stable mid-game play.             |
|         **7** |      5e-5     |   0.99   |   16   |    1.0 → 0.05   |         0.08         |    100k   |     **–21**     | Too noisy — weakest experiment.                        |
|         **8** |      3e-4     |   0.95   |   32   |    1.0 → 0.10   |         0.20         |    80k    |     **–21**    | High learning rate + low gamma = unstable.             |
|         **9** |      1e-4     |   0.998  |   32   |    1.0 → 0.01   |         0.05         |    200k   |     **–20**     | Long exploration helped; strong performance.           |
| **10 (best)** |    **1e-4**   | **0.99** | **32** |  **1.0 → 0.05** |       **0.1**       |  **500k** |     **–14.60**     | Best balance, best rally quality, saved as best model. |

---

### Files produced

* Models: `models/dqn_model_1.zip` … `models/dqn_model_10.zip`
* Best model (Lesly): `models/dqn_model_10.zip`
* Checkpoints: `models/checkpoints/`
* Plots: `plots/reward_plot_exp{N}.png`
* Experiment metadata: `experiments.json`

---

### How I trained (quick recipe)

**Quick test training (short run — see progress)**

```bash
python train.py --timesteps 100000 \
  --lr 0.0001 --gamma 0.99 --batch-size 32 \
  --eps-start 1.0 --eps-end 0.02 --exp-fraction 0.10
```

(If your `train.py` doesn't accept CLI args, edit the constants at the top and run `python train.py`.)

**Full training (Lesly's best):**

```bash
python train.py --timesteps 200000 \
  --lr 0.0001 --gamma 0.99 --batch-size 32 \
  --eps-start 1.0 --eps-end 0.02 --exp-fraction 0.10
# expected saved model: models/dqn_model_10.zip
```

---

## Video Demonstration

A video demonstration of the trained agent is included showing:
- The agent loading the trained model (`dqn_model.zip`)
- Multiple episodes of gameplay
- The agent using GreedyQPolicy for optimal performance
- Real-time interaction with the [Game Name] environment


### Video Link
[Screen recording](https://www.loom.com/share/2fe29bb104f84722b3af6263f39a4057)

---

## Key Findings & Conclusions

### Overall Hyperparameter Analysis
Across all experiments, we observed the following trends:

Learning Rate (LR): Very high LR (e.g., 0.001–0.002) often caused unstable learning, with the agent failing to track the ball effectively. Lower LR (0.0001–0.0005) produced more stable learning, though too low slowed progress.

Gamma (Discount Factor): Higher gamma values (0.98–0.99) sometimes made the agent too cautious, focusing on long-term rewards but reacting slower to the ball. Slightly lower gamma (0.97) with smaller batch sizes improved responsiveness.

Batch Size: Smaller batch sizes (32) allowed faster updates and better adaptation to game dynamics, while larger batch sizes (64) made learning slower.

Epsilon (Exploration): Gradual decay from 1.0 → 0.03 led to effective exploration initially and reliable exploitation later.

Overall, experiments that balanced learning rate, gamma, and batch size while decaying epsilon slowly produced the most intelligent and consistent agent behavior.

### Best Configuration
Based on our experiments, the best performing configuration was:

Learning Rate: 0.0002

Gamma: 0.97

Batch Size: 32

Exploration Fraction: 1.0

Final Epsilon: 0.03

Performance: The agent displayed the most consistent and intelligent behavior — reacted quickly, maintained rallies, and intercepted most balls. It demonstrated reliable paddle control and adaptation, making it the best performing model overall.


---

## Requirements

Create a `requirements.txt` file with:
```
gymnasium[atari,accept-rom-license]
stable-baselines3[extra]
ale-py
torch
tensorboard
opencv-python
matplotlib
pandas
numpy
```

---


- Stable Baselines3 Team
- OpenAI Gymnasium Team
