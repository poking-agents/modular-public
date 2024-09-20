import json
import os
import sys
from typing import Any

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


def get_json_size_in_mb(json_obj: Any) -> float:
    json_str = json.dumps(json_obj)
    size_in_bytes = sys.getsizeof(json_str)
    size_in_mb = size_in_bytes / (1024 * 1024)
    return size_in_mb


def trim_state(
    state: dict,
    limit: float = 75.0,
    content_cutoff: int = 4096,
) -> dict:
    """
    Trims the state to stay within the platform limits, for hooks.save_state.

    Parameters:
        state: dict, The state to trim.
        limit: float, The maximum size for the trimmed state. Vivaria supports up to 100MB,
            so we're being conservative and defaulting to 75MB.
        content_cutoff: int, The number of characters to maintain in messages.
    """
    # TODOs:
    # Ensure that the content_cutoff stays above max_tokens
    # Preferentially trim tool outputs before actions
    # Consider making a mutating version of this function, to prevent agent out-of-memory errors

    state_copy = json.loads(
        json.dumps(state)
    )  # Deep copy of state to avoid direct mutation issues
    total_size = get_json_size_in_mb(state_copy)

    if total_size < limit:
        return state_copy

    print(
        f"State size is {total_size:.2f}MB, which is above the limit of {limit:.2f}MB."
    )
    # Trimming content in nodes' messages
    for node in state_copy["nodes"]:
        if (
            "content" in node["message"]
            and len(node["message"]["content"]) > content_cutoff
        ):
            old_content = node["message"]["content"]
            new_content = (
                old_content[:content_cutoff]
                + f"\n[Note: Content trimmed to {content_cutoff} characters]"
            )
            node["message"]["content"] = new_content
            total_size = get_json_size_in_mb(state_copy)
            if total_size < limit:
                break

    # Trimming content within messages in next_step args if necessary
    next_step = state_copy["next_step"]
    if "args" in next_step and "messages" in next_step["args"]:
        for message in next_step["args"]["messages"]:
            if "content" in message and len(message["content"]) > content_cutoff:
                old_content = message["content"]
                new_content = (
                    old_content[:content_cutoff]
                    + f"\n[Note: Content trimmed to {content_cutoff} characters]"
                )
                message["content"] = new_content
                total_size = get_json_size_in_mb(state_copy)
                if total_size < limit:
                    break

    print(f"State size is {total_size}MB after trimming.")
    return state_copy


async def main(*args):
    global agent
    print("STARTING AGENT")
    task = await hooks.getTask()
    task_string = task.instructions.strip()
    hooks.log("Task:", task_string)
    usage_info = await hooks.get_usage()
    usage_limits = usage_info.usageLimits.tokens

    state = State(
        task_string=task_string,
        next_step={"module_type": "prompter", "args": {}},
        token_limit=usage_limits,
        timeout=default_timeout,
    )

    with open("/home/agent/settings.json") as f:
        settings = Settings(**json.loads(f.read()))

    if os.environ.get("STARTING_STATE"):
        state = State.parse_raw(os.environ["STARTING_STATE"])
        await replay_history(state, settings)
    elif os.environ.get("STARTING_STATE_PATH"):
        state = State.parse_file(os.environ["STARTING_STATE_PATH"])
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
        hooks.save_state(trim_state(agent.state.dict()))
        await agent.autosubmit()


if __name__ == "__main__":
    hooks.main(main)
