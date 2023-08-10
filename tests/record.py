import gymnasium as gym
import gymnasium_hybrid
from gymnasium import wrappers


if __name__ == '__main__':
    env = gym.make('Sliding-v0')
    env = gym.wrappers.Monitor(env, "./video", force=True)
    env.metadata["render.modes"] = ["human", "rgb_array"]
    env.reset()

    done = False
    while not done:
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        done = terminated or truncated

    env.close()
