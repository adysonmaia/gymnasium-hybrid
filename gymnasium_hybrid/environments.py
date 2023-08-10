import numpy as np
from typing import Tuple
from typing import Optional, Any, Union
from collections import namedtuple

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from gymnasium.error import DependencyNotInstalled
gym.logger.set_level(40)  # noqa

from gymnasium_hybrid.agents import BaseAgent
from gymnasium_hybrid.agents import MovingAgent
from gymnasium_hybrid.agents import SlidingAgent


# Action Id
ACCELERATE = 0
TURN = 1
BREAK = 2


Target = namedtuple('Target', ['x', 'y', 'radius'])


class Action:
    """"
    Action class to store and standardize the action for the environment.
    """
    def __init__(self, id_: int, parameters: Union[list, np.ndarray]):
        """"
        Initialization of an action.

        Args:
            id_: The id of the selected action.
            parameters: The parameters of an action.
        """
        self.id = id_
        self.parameters = parameters

    @property
    def parameter(self) -> float:
        """"
        Property method to return the parameter related to the action selected.

        Returns:
            The parameter related to this action_id
        """
        if len(self.parameters) == 2:
            return self.parameters[self.id]
        else:
            return self.parameters[0]


class BaseEnv(gym.Env[np.ndarray, Tuple]):
    """"
    Gym environment parent class.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(
            self,
            max_turn: float = np.pi/2,
            max_acceleration: float = 0.5,
            delta_t: float = 0.005,
            max_step: int = 200,
            penalty: float = 0.001,
            break_value: float = 0.1,
            render_mode: Optional[str] = None
    ):
        """Initialization of the gym environment.

        Args:
            seed (int): Seed used to get reproducible results.
            max_turn (float): Maximum turn during one step (in radian).
            max_acceleration (float): Maximum acceleration during one step.
            delta_t (float): Time duration of one step.
            max_step (int): Maximum number of steps in one episode.
            penalty (float): Score penalty given at the agent every step.
            break_value (float): Break value when performing break action.
        """
        super().__init__()
        # Agent Parameters
        self.max_turn = max_turn
        self.max_acceleration = max_acceleration
        self.break_value = break_value

        # Environment Parameters
        self.delta_t = delta_t
        self.max_step = max_step
        self.field_size = 1.0
        self.target_radius = 0.1
        self.penalty = penalty

        # Render Parameters
        self.render_mode = render_mode
        self.screen_width = 400
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.surf = None
        self.isopen = True

        # Initialization
        self.target = None
        self.viewer = None
        self.current_step = None
        self.agent = BaseAgent(break_value=break_value, delta_t=delta_t)

        parameters_min = np.array([0, -1])
        parameters_max = np.array([1, +1])

        self.action_space = spaces.Tuple((spaces.Discrete(3),
                                          spaces.Box(parameters_min, parameters_max, dtype=np.float32)))
        self.observation_space = spaces.Box(-np.ones(10), np.ones(10))

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.current_step = 0

        limit = self.field_size-self.target_radius
        low = [-limit, -limit, self.target_radius]
        high = [limit, limit, self.target_radius]
        self.target = Target(*self.np_random.uniform(low, high))

        low = [-self.field_size, -self.field_size, 0]
        high = [self.field_size, self.field_size, 2 * np.pi]
        self.agent.reset(*self.np_random.uniform(low, high))

        if self.render_mode == "human":
            self.render()

        state = self.get_state()
        return state, {}

    def step(self, raw_action: Tuple[int, np.ndarray]) -> Tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = Action(*raw_action)
        last_distance = self.distance
        self.current_step += 1

        if action.id == TURN:
            rotation = self.max_turn * max(min(action.parameter, 1), -1)
            self.agent.turn(rotation)
        elif action.id == ACCELERATE:
            acceleration = self.max_acceleration * max(min(action.parameter, 1), 0)
            self.agent.accelerate(acceleration)
        elif action.id == BREAK:
            self.agent.break_()

        if self.distance < self.target_radius and self.agent.speed == 0:
            reward = self.get_reward(last_distance, True)
            terminated = True
            truncated = False
        elif abs(self.agent.x) > self.field_size or abs(self.agent.y) > self.field_size or self.current_step > self.max_step:
            reward = -1
            terminated = True
            truncated = True
        else:
            reward = self.get_reward(last_distance)
            terminated = False
            truncated = False

        return self.get_state(), reward, terminated, truncated, {}

    def get_state(self) -> np.ndarray:
        state = [
            self.agent.x,
            self.agent.y,
            self.agent.speed,
            np.cos(self.agent.theta),
            np.sin(self.agent.theta),
            self.target.x,
            self.target.y,
            self.distance,
            0 if self.distance > self.target_radius else 1,
            self.current_step / self.max_step
        ]
        state = np.asarray(state, dtype=np.float32)
        return state

    def get_reward(self, last_distance: float, goal: bool = False) -> float:
        return last_distance - self.distance - self.penalty + (1 if goal else 0)

    @property
    def distance(self) -> float:
        return self.get_distance(self.agent.x, self.agent.y, self.target.x, self.target.y)

    @staticmethod
    def get_distance(x1: float, y1: float, x2: float, y2: float) -> float:
        return np.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2))

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[classic-control]`"
            ) from e

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        unit_x = self.screen_width / 2
        unit_y = self.screen_height / 2
        agent_radius = 0.05

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        # agent_coord = (int(unit_x * (1 + self.agent.x)), int(unit_y * (1 + self.agent.y)))
        agent_coord = pygame.math.Vector2(
            unit_x * (1 + self.agent.x),
            unit_y * (1 + self.agent.y)
        )
        # Draw agent
        gfxdraw.aacircle(
            self.surf,
            int(agent_coord.x),
            int(agent_coord.y),
            int(unit_x * agent_radius),
            (25, 76, 229)
        )
        gfxdraw.filled_circle(
            self.surf,
            int(agent_coord.x),
            int(agent_coord.y),
            int(unit_x * agent_radius),
            (25, 76, 229)
        )

        # Draw agent's arrow
        t, r, m = 0.1 * unit_x, 0.04 * unit_y, 0.06 * unit_x
        pole_coords = []
        for coord in [(t, 0), (m, r), (m, -r)]:
            coord = pygame.math.Vector2(coord).rotate_rad(self.agent.theta)
            coord += agent_coord
            pole_coords.append(coord)

        gfxdraw.aapolygon(
            self.surf,
            pole_coords,
            (0, 0, 0)
        )
        gfxdraw.filled_polygon(
            self.surf,
            pole_coords,
            (0, 0, 0)
        )

        # Draw target
        target_coord = pygame.math.Vector2(
            unit_x * (1 + self.target.x),
            unit_y * (1 + self.target.y)
        )
        gfxdraw.aacircle(
            self.surf,
            int(target_coord.x),
            int(target_coord.y),
            int(unit_x * self.target_radius),
            (255, 127, 127)
        )
        gfxdraw.filled_circle(
            self.surf,
            int(target_coord.x),
            int(target_coord.y),
            int(unit_x * self.target_radius),
            (255, 127, 127)
        )

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def render_old(self, mode='human'):
        screen_width = 400
        screen_height = 400
        unit_x = screen_width / 2
        unit_y = screen_height / 2
        agent_radius = 0.05

        if self.viewer is None:
            from gymnasium.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            agent = rendering.make_circle(unit_x * agent_radius)
            self.agent_trans = rendering.Transform(translation=(unit_x * (1 + self.agent.x), unit_y * (1 + self.agent.y)))  # noqa
            agent.add_attr(self.agent_trans)
            agent.set_color(0.1, 0.3, 0.9)
            self.viewer.add_geom(agent)

            t, r, m = 0.1 * unit_x, 0.04 * unit_y, 0.06 * unit_x
            arrow = rendering.FilledPolygon([(t, 0), (m, r), (m, -r)])
            self.arrow_trans = rendering.Transform(rotation=self.agent.theta)  # noqa
            arrow.add_attr(self.arrow_trans)
            arrow.add_attr(self.agent_trans)
            arrow.set_color(0, 0, 0)
            self.viewer.add_geom(arrow)

            target = rendering.make_circle(unit_x * self.target_radius)
            target_trans = rendering.Transform(translation=(unit_x * (1 + self.target.x), unit_y * (1 + self.target.y)))
            target.add_attr(target_trans)
            target.set_color(1, 0.5, 0.5)
            self.viewer.add_geom(target)

        self.arrow_trans.set_rotation(self.agent.theta)
        self.agent_trans.set_translation(unit_x * (1 + self.agent.x), unit_y * (1 + self.agent.y))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False


class MovingEnv(BaseEnv):
    def __init__(
            self,
            max_turn: float = np.pi/2,
            max_acceleration: float = 0.5,
            delta_t: float = 0.005,
            max_step: int = 200,
            penalty: float = 0.001,
            break_value: float = 0.1,
            render_mode: Optional[str] = None
    ):

        super(MovingEnv, self).__init__(
            max_turn=max_turn,
            max_acceleration=max_acceleration,
            delta_t=delta_t,
            max_step=max_step,
            penalty=penalty,
            break_value=break_value,
            render_mode=render_mode
        )

        self.agent = MovingAgent(
            break_value=break_value,
            delta_t=delta_t,
        )


class SlidingEnv(BaseEnv):
    def __init__(
            self,
            max_turn: float = np.pi/2,
            max_acceleration: float = 0.5,
            delta_t: float = 0.005,
            max_step: int = 200,
            penalty: float = 0.001,
            break_value: float = 0.1,
            render_mode: Optional[str] = None
    ):

        super(SlidingEnv, self).__init__(
            max_turn=max_turn,
            max_acceleration=max_acceleration,
            delta_t=delta_t,
            max_step=max_step,
            penalty=penalty,
            break_value=break_value,
            render_mode=render_mode
        )

        self.agent = SlidingAgent(
            break_value=break_value,
            delta_t=delta_t
        )
