import json
import os

from base import Agent, Settings, State, hooks
from modules import actors, discriminators, generators, prompters, tools
from templates import default_timeout


async def replay_history(starting_state: State, settings: Settings) -> None:
    dummy_agent = Agent(
        state=State(task_string=starting_state.task_string),
        settings=settings,
        toolkit_dict={},
    )
    dummy_agent.set_toolkit_dict(getattr(tools, dummy_agent.settings.toolkit))
    hooks.log("UI MESSAGE: Replaying actions. Printing actions and outputs.")
    node_is_action_result = False
    action_result = None
    for node_id in range(starting_state.last_node_id + 1):
        if node_is_action_result:
            # The assert should catch instances where the replay led to a different result
            # than in the original run. There are benign cases for this, but let's keep
            # this for now.
            assert action_result == starting_state.nodes[node_id].message
            node_is_action_result = False
            continue
        node = starting_state.nodes[node_id]
        dummy_agent.append(node.message, parent=node.parent)
        action_result = await getattr(actors, dummy_agent.settings.actor)(dummy_agent)
        if action_result is not None:
            dummy_agent.append(action_result)
            node_is_action_result = True
    hooks.log("UI Message: Replay finished")


async def main(*args):
    global agent
    print("STARTING AGENT")
    task = await hooks.getTask()
    task_string = task.instructions.strip()
    hooks.log("Task:", task_string)
    usage_limits = (await hooks.get_usage()).usageLimits.tokens
    state = State(
        task_string=task_string,
        next_step={"module_type": "prompter", "args": {}},
        token_limit=usage_limits,
        timeout=default_timeout,
    )

    with open("/home/agent/settings.json") as f:
        settings = Settings(**json.loads(f.read()))

    if os.environ.get("STARTING_STATE"):
        state = State(**json.loads(os.environ["STARTING_STATE"]))
        await replay_history(state, settings)
    elif os.environ.get("STARTING_STATE_PATH"):
        with open(os.environ["STARTING_STATE_PATH"]) as f:
            state = State(**json.load(f))
            await replay_history(state, settings)

    agent = Agent(
        state=state,
        settings=settings,
        toolkit_dict={},
    )

    while True:
        toolkit_dict = getattr(tools, agent.settings.toolkit)
        if task.scoring.intermediate:
            toolkit_dict = {**toolkit_dict, **tools.scoring_tools}
        agent.set_toolkit_dict(toolkit_dict)
        # Almost always the agent should follow the order below.
        # Usually a prompter will conclude by setting the next_step to be
        # "generator", and so on.
        # The only reason the flow is designed like this is to allow for more
        # flexibility and more powerful state-editing experiments.
        if agent.state.next_step["module_type"] == "prompter":
            await getattr(prompters, agent.settings.prompter)(agent)
        elif agent.state.next_step["module_type"] == "generator":
            await getattr(generators, agent.settings.generator)(agent)
        elif agent.state.next_step["module_type"] == "discriminator":
            await getattr(discriminators, agent.settings.discriminator)(agent)
        elif agent.state.next_step["module_type"] == "actor":
            await getattr(actors, agent.settings.actor)(agent)
        else:
            raise ValueError("Invalid module type as next step")
        usage_info = await hooks.get_usage()
        agent.state.token_usage = usage_info.usage.tokens
        agent.state.token_limit = usage_info.usageLimits.tokens
        hooks.save_state(agent.state.dict())


if __name__ == "__main__":
    hooks.main(main)
