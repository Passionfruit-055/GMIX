import numpy as np
from gymnasium.utils import EzPickle

from pettingzoo.mpe._mpe_utils.core import Agent, Landmark
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env
from pettingzoo.utils.conversions import parallel_wrapper_fn

from env.entities import DangerZone, DangerWorld, DangerPoint

__all__ = ["env", "parallel_env", "raw_env"]


class raw_env(SimpleEnv, EzPickle):
    def __init__(
            self, local_ratio=0.5, max_cycles=25, continuous_actions=False, render_mode=None
    ):
        EzPickle.__init__(
            self,
            local_ratio=local_ratio,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
        )
        assert (
                0.0 <= local_ratio <= 1.0
        ), "local_ratio is a proportion. Must be between 0 and 1."
        scenario = Scenario()
        world = scenario.make_world()
        SimpleEnv.__init__(
            self,
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            local_ratio=local_ratio,
        )
        self.metadata["name"] = "simple_reference_risk_v1"


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)


class Scenario(BaseScenario):
    def make_world(self):
        world = DangerWorld()
        # set any world properties first
        world.dim_c = 10
        world.collaborative = True  # whether agents share rewards
        # add agents
        world.agents = [Agent() for i in range(2)]
        for i, agent in enumerate(world.agents):
            agent.name = f"agent_{i}"
            agent.collide = False
        # add landmarks
        world.landmarks = [Landmark() for i in range(3)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = False
            landmark.movable = False
        # add danger points
        world.danger_points = [DangerPoint() for _ in range(2)]
        for i, danger_point in enumerate(world.danger_points):
            danger_point.name = "danger_point %d" % i
            danger_point.collide = False
            danger_point.movable = False
        # add danger zones
        world.danger_zones = [DangerZone(dp) for dp in world.danger_points]
        for i, danger_zone in enumerate(world.danger_zones):
            danger_zone.name = "danger_zone %d" % i
            danger_zone.collide = False
            danger_zone.movable = False

        return world

    def reset_world(self, world, np_random):
        # assign goals to agents
        for agent in world.agents:
            agent.goal_a = None
            agent.goal_b = None
        # want other agent to go to the goal landmark
        world.agents[0].goal_a = world.agents[1]
        world.agents[0].goal_b = np_random.choice(world.landmarks)
        world.agents[1].goal_a = world.agents[0]
        world.agents[1].goal_b = np_random.choice(world.landmarks)
        # random properties for agents
        for i, agent in enumerate(world.agents):
            # agent.color = np.array([0.25, 0.25, 0.25])
            agent.color = np.array([139, 38, 113]) / 200
        # random properties for landmarks
        world.landmarks[0].color = np.array([23, 129, 181]) / 200
        world.landmarks[1].color = np.array([34, 148, 83]) / 200
        world.landmarks[2].color = np.array([238, 63, 77]) / 200

        # # special colors for goals
        # world.agents[0].goal_a.color = world.agents[0].goal_b.color
        # world.agents[1].goal_a.color = world.agents[1].goal_b.color

        # set fixed initial states
        starting_point = [np.array((-0.25, 0.25)), np.array((-0.25, -0.25))]
        for agent, s_p in zip(world.agents, starting_point):
            # agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_pos = s_p
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        landmarks = [np.array((0.7, 0.3)), np.array((0.8, -0.5)), np.array((0.95, -0.1))]
        for i, (landmark, lm) in enumerate(zip(world.landmarks, landmarks)):
            # landmark.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            landmark.state.p_pos = lm
            landmark.state.p_vel = np.zeros(world.dim_p)
        danger_points = [np.array((0, 0.90)), np.array((0, -0.90))]
        for i, (danger_point, d_p) in enumerate(zip(world.danger_points, danger_points)):
            # danger_point.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            danger_point.state.p_pos = d_p

        for dp, dz in zip(world.danger_points, world.danger_zones):
            dz.state.p_pos = dp.state.p_pos
            dz.agent_in_zone = []

    def reward(self, agent, world):
        if agent.goal_a is None or agent.goal_b is None:
            agent_reward = 0.0
        else:
            agent_reward = np.sqrt(
                np.sum(np.square(agent.goal_a.state.p_pos - agent.goal_b.state.p_pos))
            )
        return -agent_reward

    def global_reward(self, world):
        all_rewards = sum(self.reward(agent, world) for agent in world.agents)
        return all_rewards / len(world.agents)

    def observation(self, agent, world):
        # goal color
        goal_color = [np.zeros(world.dim_color), np.zeros(world.dim_color)]
        if agent.goal_b is not None:
            goal_color[1] = agent.goal_b.color

        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)
        # check in danger zones
        return np.concatenate([agent.state.p_pos] + entity_pos + [goal_color[1]] + comm)
