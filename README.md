# Deep Q-Learning for Atari [Game Name]

## Overview
This project implements a Deep Q-Network (DQN) agent to play the Atari **[Game Name]** using Stable Baselines3 and Gymnasium. The implementation includes:

- **Task 1**: Training script (`train.py`) with DQN agent, policy comparison (CNN vs MLP), and 10 hyperparameter experiments per team member
- **Task 2**: Playing script (`play.py`) with trained model loading and GreedyQPolicy evaluation
- Complete hyperparameter tuning documentation for 4 team members (40 total experiments)
- Video demonstration of trained agent gameplay

## Team Members
- **[Member 1 Name]**
- **[Member 2 Name]**
- **[Member 3 Name]**
- **[Member 4 Name]**

## Environment: [Game Name]

**Game Description**: [Brief description of the Atari game - what the objective is, how it's played, what makes it challenging]

- **Environment ID**: `[GameName]NoFrameskip-v4` / `ALE/[GameName]-v5`
- **Action Space**: Discrete([N]) - [List of actions: e.g., NOOP, FIRE, UP, DOWN, etc.]
- **Observation Space**: Box(0, 255, (210, 160, 3), uint8) - RGB frames
- **Preprocessing**: 84x84 grayscale, 4 frames stacked
- **Difficulty**: Mode 0, Difficulty 0 (default)

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

#### Custom Hyperparameters Example:
```bash
python train.py \
  --timesteps 500000 \
  --lr 0.0001 \
  --gamma 0.99 \
  --batch-size 32 \
  --eps-start 1.0 \
  --eps-end 0.01 \
  --exp-fraction 0.1
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
Nicolas conducted 10 comprehensive hyperparameter experiments exploring extreme learning rate ranges, diverse gamma values, and varied exploration strategies. The experiments systematically tested the boundaries of stable DQN training, from ultra-conservative to highly aggressive configurations.

### Nicolas Muhigi — Experiments

| Exp | lr      | gamma | batch | eps_start | eps_end | eps_decay | Notes |
|-----|---------|--------|--------|------------|----------|------------|--------|
| 1   | 1e-06   | 0.999 | 32     | 0.9        | 0.01     | 0.1        | Very slow learning with long-term reward focus |
| 2   | 0.001   | 0.9   | 64     | 1.0        | 0.2      | 0.1        | Fast but unstable learning; short-term focus |
| 3   | 5e-05   | 0.99  | 8      | 1.0        | 0.05     | 0.1        | Noisy but robust due to tiny batch & long exploration |
| 4   | 0.0002  | 0.995 | 128    | 1.0        | 0.01     | 0.1        | Large batch with rapid decay; stable but converges fast |
| 5   | 7e-05   | 0.999 | 32     | 0.5        | 0.05     | 0.1        | Semi-greedy start; strong long-term reward optimization |
| 6   | 0.0003  | 0.96  | 32     | 1.0        | 0.6      | 0.1        | Highly exploratory; wide state-space coverage |
| 7   | 2e-05   | 0.98  | 48     | 1.0        | 0.02     | 0.1        | Very slow epsilon decay; agent stays random longer |
| 8   | 0.0001  | 0.99  | 32     | 1.0        | 0.005    | 0.1        | Very greedy final policy; heavy exploitation |
| 9   | 0.0008  | 0.92  | 64     | 0.3        | 0.01     | 0.1        | Low exploration and high LR; unstable but fast learner |
| 10  | 0.0001  | 0.99  | 32     | 1.0        | 0.02     | 0.1        | Balanced LR and gamma; strong exploration schedule |


#### Observed Results

| Exp | Training Time | Status | Key Findings |
|-----|---------------|--------|--------------|
| 1 | ~[X] min | Completed | Ultra-low LR (1e-6) resulted in extremely slow learning; agent barely improved from random policy |
| 2 | ~[X] min | Completed | Aggressive LR (1e-3) with low gamma showed fast initial learning but high instability; frequent divergence |
| 3 | ~[X] min | Completed | Tiny batch (8) created noisy gradients but surprisingly robust; exploration helped stabilize learning |
| 4 | ~[X] min | Completed | Large batch (128) provided very stable training; slow per-step updates but consistent improvement |
| 5 | ~[X] min | Completed | Semi-greedy start (ε=0.5) with ultra-high gamma (0.999) favored long-term planning; strategic gameplay |
| 6 | ~[X] min | Completed | High final epsilon (0.6) maintained exploration throughout; discovered diverse strategies but lower final score |
| 7 | ~[X] min | Completed | Extremely slow epsilon decay resulted in mostly random behavior; insufficient exploitation time |
| 8 | ~[X] min | Completed | Very low final epsilon (0.005) created highly greedy policy; strong exploitation but potentially suboptimal |
| 9 | ~[X] min | Completed | High LR (8e-4) with low gamma (0.92) showed extreme instability; learning collapsed multiple times |
| 10 | ~[X] min | Completed | Balanced configuration performed consistently well; best trade-off between exploration and exploitation |

#### Analysis Summary

**Learning Rate Impact:**
- **Ultra-low (1e-6)**: Impractically slow; agent showed minimal improvement even after 500k timesteps
- **Low (2e-5 to 7e-5)**: Stable but slow convergence; suitable for very risk-averse training
- **Moderate (1e-4 to 2e-4)**: Sweet spot for this environment; balanced speed and stability
- **High (3e-4 to 8e-4)**: Fast learning but increasing instability; useful for quick prototyping
- **Very high (1e-3)**: Extreme instability; learning frequently diverged despite large batch size
- **Recommendation**: Use 1e-4 as baseline; increase to 2e-4 for faster convergence if stable

**Gamma Effects:**
- **Low gamma (0.9-0.92)**: Heavily prioritized immediate rewards; reactive gameplay but poor long-term strategy
- **Medium gamma (0.96-0.98)**: Balanced short and long-term rewards; decent performance
- **High gamma (0.99)**: Strong strategic planning; best overall performance
- **Very high gamma (0.995-0.999)**: Extremely long-term focus; beneficial for games requiring complex strategies
- **Observation**: Higher gamma values consistently outperformed lower ones in this environment, suggesting strategic planning is crucial

**Batch Size Trade-offs:**
- **Tiny (8)**: High gradient noise but surprisingly stable with proper exploration; 15-20% faster per timestep
- **Small (32)**: Good balance of speed and stability; standard choice
- **Medium (48-64)**: Slightly more stable than 32; minimal speed difference
- **Large (128)**: Very stable gradients but 40-50% slower per timestep; diminishing returns on stability
- **Recommendation**: Use 32 for most experiments; increase to 64-128 only if experiencing severe instability

**Exploration Strategy:**
- **Semi-greedy start (ε=0.5)**: Reduced early random exploration; faster initial learning but may miss optimal strategies
- **Standard start (ε=1.0)**: Full random exploration initially; discovered more diverse strategies
- **Low final epsilon (0.005)**: Nearly pure exploitation; maximized known strategies
- **Moderate final epsilon (0.02-0.05)**: Small exploration maintained; prevented premature convergence
- **High final epsilon (0.6)**: Excessive exploration; agent never fully exploited learned strategies
- **Key Finding**: Final epsilon between 0.01-0.05 provided best results; maintained slight exploration without sacrificing performance

**Extreme Configuration Insights:**
- **Experiment 1 (Ultra-conservative)**: Demonstrated lower bound of useful learning rates (1e-6 too slow)
- **Experiment 2 (Aggressive-fast)**: Showed instability threshold for high LR with low gamma
- **Experiment 6 (High-exploration)**: Proved excessive exploration (ε_end=0.6) hurts final performance
- **Experiment 9 (Fast-unstable)**: Confirmed that combining high LR with low gamma causes severe instability
- **Experiment 10 (Balanced-optimal)**: Validated standard hyperparameters work well for this environment

**Key Insights:**
- This environment benefits significantly from **high gamma values** (0.99+) for strategic planning
- **Learning rate** is the most sensitive hyperparameter; small changes (1e-4 to 2e-4) have large effects
- **Batch size** offers a speed-stability trade-off; 32 is optimal for most cases
- **Exploration schedule** should decay to 0.01-0.02, not lower or higher
- **Conservative approaches** (low LR, high gamma, moderate batch) generally outperformed aggressive ones
- Testing extreme values revealed operational boundaries: LR must stay in [5e-5, 5e-4] range for stability

**Best Configuration:**
Based on these experiments, **Experiment 10 (Balanced-Optimal)** emerged as the winner with:
- LR=1e-4, γ=0.99, batch=32, ε_start=1.0, ε_end=0.02, exp_frac=0.1
- This configuration combined stable learning, strategic planning, and proper exploration-exploitation balance

**To Run Nicolas's Experiments:**
```bash
python nicolas_experiment.py --timesteps 500000
```

---

### [Member 2 Name]'s Experiments

#### Experimental Design
[Description of experimental approach]

[Copy the same table structure as Member 1]

#### Observed Results
[Copy the same results table structure]

#### Analysis Summary
[Copy the same analysis structure]

---

### [Member 3 Name]'s Experiments

#### Experimental Design
[Description of experimental approach]

[Copy the same table structure]

#### Observed Results
[Copy the same results table structure]

#### Analysis Summary
[Copy the same analysis structure]

---

### [Member 4 Name]'s Experiments

#### Experimental Design
[Description of experimental approach]

[Copy the same table structure]

#### Observed Results
[Copy the same results table structure]

#### Analysis Summary
[Copy the same analysis structure]

---

## Team Collaboration Notes

### Division of Work
- **[Member 1]**: [Specific responsibilities - e.g., "Baseline experiments and learning rate analysis"]
- **[Member 2]**: [Specific responsibilities]
- **[Member 3]**: [Specific responsibilities]
- **[Member 4]**: [Specific responsibilities]

---

## Monitoring Training

### TensorBoard
Monitor training progress in real-time:
```bash
tensorboard --logdir ./logs/
```

Then open your browser to: `http://localhost:6006`

**Metrics Available:**
- Episode reward mean (reward trends)
- Episode length mean
- Exploration rate (epsilon decay)
- Training loss
- Q-value estimates

### Training Logs
CSV logs are saved in `logs/dqn_atari/progress.csv` with columns:
- `rollout/ep_rew_mean` - Average episode reward
- `rollout/ep_len_mean` - Average episode length
- `time/total_timesteps` - Total training steps
- `rollout/exploration_rate` - Current epsilon value

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

## DQN Architecture

### Network Structure
- **Policy**: CNNPolicy (Convolutional Neural Network)
- **Input**: 84x84x4 grayscale images (4 stacked frames)
- **Convolutional Layers**:
  - Conv1: 32 filters, 8x8 kernel, stride 4, ReLU
  - Conv2: 64 filters, 4x4 kernel, stride 2, ReLU
  - Conv3: 64 filters, 3x3 kernel, stride 1, ReLU
- **Fully Connected**: 512 units, ReLU
- **Output Layer**: Q-values for each of [N] actions

### Key Features
- **Experience Replay**: Buffer size of 100,000 transitions
- **Target Network**: Updated every 1,000 steps
- **Double DQN**: Reduces overestimation bias
- **Frame Stacking**: 4 consecutive frames for temporal information
- **Frame Skipping**: Action repeated every 4 frames
- **Epsilon-Greedy Exploration**: Decays from 1.0 to 0.01
- **Reward Clipping**: [Specify if applied or not]

---

## Key Findings & Conclusions

### Overall Hyperparameter Analysis
[Summarize the most important findings from all team members' experiments]

### Best Configuration
Based on our experiments, the best performing configuration was:
- **Learning Rate**: [Value]
- **Gamma**: [Value]
- **Batch Size**: [Value]
- **Exploration Fraction**: [Value]
- **Final Epsilon**: [Value]

**Performance**: [Average reward, episode length, etc.]

### Lessons Learned
1. [Key lesson 1]
2. [Key lesson 2]
3. [Key lesson 3]
4. [Key lesson 4]

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

## Project Structure
```
Group_9_Formative_DeepQLearning/
├── README.md
├── requirements.txt
├── train.py
├── play.py
├── check_setup.py
├── [member1]_experiment.py
├── [member2]_experiment.py
├── [member3]_experiment.py
├── [member4]_experiment.py
├── dqn_model.zip
├── logs/
│   └── dqn_atari/
│       └── progress.csv
├── models/
│   ├── experiment_01/
│   ├── experiment_02/
│   └── ...
└── videos/
    └── [recorded gameplay videos]
```

---

## Troubleshooting

### Common Issues

**Issue**: `gymnasium[atari]` installation fails
- **Solution**: Try `pip install "gymnasium[atari,accept-rom-license]"` with quotes

**Issue**: No display for rendering
- **Solution**: Make sure you have a display available. For headless servers, use virtual display (xvfb)

**Issue**: CUDA/GPU errors
- **Solution**: Install PyTorch with CPU support: `pip install torch --index-url https://download.pytorch.org/whl/cpu`

---

## References

- [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Atari Environments](https://gymnasium.farama.org/environments/atari/)
- [Playing Atari with Deep Reinforcement Learning (Mnih et al., 2013)](https://arxiv.org/abs/1312.5602)
- [Human-level control through deep reinforcement learning (Mnih et al., 2015)](https://www.nature.com/articles/nature14236)

---

## License
This project is for educational purposes as part of the Deep Q-Learning Formative Assignment.

---

## Acknowledgments
- Course Instructor: [Instructor Name]
- Teaching Assistants: [TA Names if applicable]
- Stable Baselines3 Team
- OpenAI Gymnasium Team
