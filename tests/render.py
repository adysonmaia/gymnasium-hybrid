import gymnasium as gym
import time
import gymnasium_hybrid


if __name__ == '__main__':
    env = gym.make('Sliding-v0', render_mode='human')
    env.reset()

    done = False
    while not done:
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        done = terminated or truncated
        env.render()
        time.sleep(0.1)

    time.sleep(1)
    env.close()
