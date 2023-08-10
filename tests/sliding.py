import gymnasium as gym
import gymnasium_hybrid
import time


if __name__ == '__main__':
    env = gym.make('Sliding-v0')
    env.reset()

    ACTION_SPACE = env.action_space[0].n
    PARAMETERS_SPACE = env.action_space[1].shape[0]
    OBSERVATION_SPACE = env.observation_space.shape[0]

    done = False
    while not done:
        state, reward, terminated, truncated, info = env.step(env.action_space.sample())
        done = terminated or truncated
        print(f'State: {state} Reward: {reward} Terminated: {terminated} Truncated: {truncated}')
        time.sleep(0.1)
