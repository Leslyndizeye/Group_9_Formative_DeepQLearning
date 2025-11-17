# ------------------------------------------------------
# Run a trained DQN Atari agent (SB3) with human render.
# ------------------------------------------------------

import os
import sys
import time
import argparse
import numpy as np
import gymnasium as gym
import ale_py  # ensure ALE namespace is registered
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.save_util import load_from_zip_file


def list_zip_models(root_dir: str):
    if not os.path.isdir(root_dir):
        return []
    return sorted(
        [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(".zip")]
    )


def pick_latest_model(root_dir: str):
    zips = list_zip_models(root_dir)
    if not zips:
        return None
    # Latest by modified time
    return max(zips, key=os.path.getmtime)


def build_env(env_id: str, seed: int, human_render: bool = True):
    """
    Build the same preprocessing as training:
      - make_atari_env (handles NoFrameskip-wrappers)
      - VecFrameStack(4)
    """
    env_kwargs = {"render_mode": "human"} if human_render else {}
    env = make_atari_env(env_id, n_envs=1, seed=seed, env_kwargs=env_kwargs)
    env = VecFrameStack(env, n_stack=4)
    return env


def try_patch_numpy_numeric():
    # Some older archives expect numpy._core.numeric in sys.modules
    try:
        import numpy.core.numeric as _numeric
        if "numpy._core.numeric" not in sys.modules:
            sys.modules["numpy._core.numeric"] = _numeric
    except Exception:
        pass


def load_model(model_path: str, env, seed: int, inference_buffer_size: int = 1_000):
    """
    Robust loader:
      1) Try normal DQN.load with smaller buffer
      2) If that fails, fallback to parameter-only load and instantiate a tiny DQN for inference
    """
    model = None

    # First attempt: normal load but override memory-hungry params
    try:
        model = DQN.load(
            model_path,
            env=env,
            custom_objects={
                "buffer_size": inference_buffer_size,
                "learning_starts": 0,
            },
        )
    except ModuleNotFoundError:
        # Patch numpy alias and retry
        try_patch_numpy_numeric()
        try:
            model = DQN.load(
                model_path,
                env=env,
                custom_objects={"buffer_size": inference_buffer_size, "learning_starts": 0},
            )
        except Exception:
            model = None
    except (ValueError, MemoryError):
        model = None
    except Exception:
        model = None

    if model is not None:
        return model

    # Fallback: parameter-only load (skip metadata that can be incompatible)
    try:
        _data, params, _vars = load_from_zip_file(
            model_path, device="cpu", load_data=False, print_system_info=False
        )
        # Create a light DQN just to host the weights (no training planned)
        model = DQN(
            "CnnPolicy",
            env,
            seed=seed,
            verbose=0,
            buffer_size=inference_buffer_size,
            learning_starts=0,
            train_freq=4,
            gradient_steps=1,
            exploration_fraction=0.0,
            exploration_initial_eps=0.0,
            exploration_final_eps=0.0,
        )
        model.set_parameters(params, exact_match=False)
        print(f"[OK] Loaded parameters in fallback mode (buffer_size={inference_buffer_size}).")
        return model
    except MemoryError as me:
        raise MemoryError(
            f"MemoryError while loading {model_path}. "
            f"Try lowering --inference-buffer-size or close other apps."
        ) from me


def run_episodes(env, model: DQN, n_episodes: int, fps: int):
    # Greedy (deterministic) evaluation
    model.exploration_rate = 0.0
    for ep in range(n_episodes):
        obs = env.reset()
        # VecEnv reset returns observations directly (not (obs, info))
        done = False
        ep_return = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            ep_return += float(rewards[0])  # VecEnv returns arrays
            done = bool(dones[0])
            if fps > 0:
                time.sleep(1.0 / fps)
        print(f"Episode {ep + 1} return: {ep_return:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Play a trained SB3 DQN Atari model.")
    parser.add_argument(
        "--env-id",
        default="PongNoFrameskip-v4",  # matches your train.py; ALE/Pong-v5 also works
        help="Gymnasium Atari env id used in training (e.g., PongNoFrameskip-v4 or ALE/Pong-v5).",
    )
    parser.add_argument(
        "--models-dir",
        default="Leslie Isaro",
        help="Directory where your .zip models are saved.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to a specific model .zip. If not set, latest zip in --models-dir is used.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--fps", type=int, default=60, help="Playback FPS pacing.")
    parser.add_argument(
        "--inference-buffer-size",
        type=int,
        default=1000,
        help="Smaller replay buffer size for inference-only loading.",
    )
    args = parser.parse_args()

    model_path = args.model_path
    if model_path is None:
        model_path = pick_latest_model(args.models_dir)

    if model_path is None or not os.path.isfile(model_path):
        print("No model .zip found.")
        print(f"Checked: --model-path={args.model_path} and folder '{args.models_dir}'")
        zips = list_zip_models(args.models_dir)
        if zips:
            print("Available:")
            for z in zips:
                print(" -", z)
        sys.exit(1)

    print(f"[INFO] Using model: {model_path}")
    print(f"[INFO] Building env: {args.env_id} (render_mode=human)")
    env = build_env(args.env_id, seed=args.seed, human_render=True)

    try:
        model = load_model(
            model_path,
            env=env,
            seed=args.seed,
            inference_buffer_size=args.inference_buffer_size,
        )
    except Exception as e:
        env.close()
        raise

    try:
        run_episodes(env, model, n_episodes=args.episodes, fps=args.fps)
    finally:
        env.close()
    print("Done.")


if __name__ == "__main__":
    main()
