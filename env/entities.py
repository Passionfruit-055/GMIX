from pettingzoo.mpe._mpe_utils.core import Entity, EntityState, np, Agent, World


class DangerPoint(Entity):
    # this entity is mainly for rendering
    def __init__(self):
        super().__init__()
        self.name = "DangerPoint"
        self.size = 0.05
        self.collide = False
        self.movable = False
        self.color = np.array([128, 0, 0])  # maroon
        self.state = EntityState()
        self.state.p_pos = np.zeros(2)


class DangerZone(Entity):
    def __init__(self, source_point: DangerPoint):
        super().__init__()
        self.name = "DangerZone"
        self.size = 0.15
        self.collide = False
        self.movable = False
        self.color = np.array([256, 0, 0])  # red
        self.state = EntityState()
        self.source_point = source_point
        self.state.p_pos = source_point.state.p_pos
        self.risk_radius = self.size
        self.agent_in_zone = []

    def inDangerZone(self, agent):
        if not isinstance(agent, Agent):
            raise TypeError("agent must be an instance of Agent")
        agent_pos = agent.state.p_pos
        danger_zone = [(p - self.risk_radius, p + self.risk_radius) for p in self.state.p_pos]
        danger = True
        for ap, dz in zip(agent_pos, danger_zone):
            if ap < dz[0] or ap > dz[1]:
                danger = False
                break
        return danger


class DangerWorld(World):
    def __init__(self):
        super().__init__()
        self.danger_points = []
        self.danger_zones = []
        self.collaborative = True

    @property
    def entities(self):
        return self.agents + self.landmarks + self.danger_points + self.danger_zones

    def danger_zone_determination(self):
        for danger_zone in self.danger_zones:
            danger_zone.agent_in_zone = []
            for agent in self.agents:
                if danger_zone.inDangerZone(agent):
                    # 看具体需要统计什么
                    danger_zone.agent_in_zone.append(agent)

    def step(self):
        super().step()
        self.danger_zone_determination()

    def danger_infos(self):
        # if an agent in coincide of n danger zones, counted n times
        agents_in_danger = [agent for dz in self.danger_zones for agent in dz.agent_in_zone]

        danger_times = [agents_in_danger.count(agent) for agent in self.agents]

        return danger_times


if __name__ == '__main__':
    pass
