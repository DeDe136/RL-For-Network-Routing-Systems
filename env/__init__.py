from gymnasium.envs.registration import register

register(
    id="NetworkRouting-QL-v0",
    entry_point="env.Q_Learning.routing_env:NetworkRoutingEnv",
    max_episode_steps=20,
)

register(
    id="NetworkRouting-DQN-v0",
    entry_point="env.DQN.routing_env:NetworkRoutingEnv",
    max_episode_steps=20,
)

from env.Q_Learning.routing_env import NetworkRoutingEnv as QLearningEnv
from env.DQN.routing_env import NetworkRoutingEnv as DQNEnv
__all__ = ["QLearningEnv", "DQNEnv"]