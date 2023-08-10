from gymnasium.envs.registration import register
from gymnasium_hybrid.environments import MovingEnv
from gymnasium_hybrid.environments import SlidingEnv

register(
    id='Moving-v0',
    entry_point='gymnasium_hybrid:MovingEnv',
)
register(
    id='Sliding-v0',
    entry_point='gymnasium_hybrid:SlidingEnv',
)