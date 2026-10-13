from openpi.agents.base import BaseAgent
from openpi.agents.er import ERAgent
from openpi.agents.ewc import EWCAgent
from openpi.agents.info import InfoAgent
from openpi.agents.lwf import LwfAgent


def agent(agent_name: str, config):
    registry = {
        "BASE": BaseAgent,
        "ER": ERAgent,
        "EWC": EWCAgent,
        "INFO": InfoAgent,
        "LWF": LwfAgent,
    }
    try:
        return registry[agent_name.upper()](config)
    except KeyError as exc:
        raise ValueError(f"Unknown agent: {agent_name}") from exc
