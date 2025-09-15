from ddpg.agent import Agent
from ddpg.agent_her import AgentHer


def create_agent(agent_type, env=None, **kwargs):
    """
    Factory function to create different types of agents.

    Args:
        agent_type: Type of agent to create ("agent" or "agent_her")
        env: Environment instance (required for agent_her)
        **kwargs: Agent configuration parameters

    Returns:
        Agent instance of the specified type
    """
    if agent_type == "agent":
        return Agent(**kwargs)
    elif agent_type == "agent_her":
        if env is None:
            raise ValueError("Environment 'env' parameter is required for agent_her")
        return AgentHer(env=env, **kwargs)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}. "
                        f"Supported types: 'agent', 'agent_her'")


def get_supported_agent_types():
    """
    Get list of supported agent types.

    Returns:
        List of supported agent type strings
    """
    return ["agent", "agent_her"]