from gymnasium.envs.registration import register

register(
    id="NetworkRouting-v0",
    entry_point="env.routing_env:NetworkRoutingEnv",
    max_episode_steps=20,
)

from env.routing_env import NetworkRoutingEnv
__all__ = ["NetworkRoutingEnv"]