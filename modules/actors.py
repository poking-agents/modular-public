import json
import math
import os
import random
from typing import Optional

from base import Agent, Message
from templates import prompt_to_search, reject_arguments_prompt, reject_command_prompt


async def get_result_message_simple(agent: Agent) -> Optional[Message]:
    last_node = agent.state.nodes[agent.state.last_node_id]
    if not last_node.message.function_call:
        return None

    tool_name = last_node.message.function_call["name"]
    output = ""
    if tool_name not in agent.toolkit_dict:
        return Message(
            role="function", # OpenAI API expects function after a function call
            content=reject_command_prompt,
            name=tool_name,
            function_call=None,
        )

    tool_info = agent.toolkit_dict[tool_name]
    tool_fn = tool_info["function"]
    tool_params = tool_info["parameters"]
    agent_args = last_node.message.function_call["arguments"]
    try:
        # parse arguments as JSON to be unpacked on function call
        agent_args = json.loads(agent_args)
    except json.JSONDecodeError:
        pass

    if not tool_params:
        if agent_args:
            return Message(
                role="function", # OpenAI API expects function after a function call
                content=reject_arguments_prompt,
                name=tool_name,
                function_call=None,
            )
        return Message(
            role="function",
            name=tool_name,
            content=await tool_fn(agent.state),
            function_call=None,
        )

    required_args = tool_params.get("required", [])
    if isinstance(agent_args, dict):
        try:
            output = await tool_fn(agent.state, **agent_args)
        except TypeError:
            return Message(
                role="function",
                content=reject_arguments_prompt,
                name=tool_name,
                function_call=None,
        )
    elif len(required_args) != 1:
        # tell agent the command failed
        return Message(
            role="function",
            content=reject_arguments_prompt,
            name=tool_name,
            function_call=None,
        )
    else:
        # assume there is only one required argument, and it's the passed string
        agent_args = {required_args[0]: agent_args}
        output = await tool_fn(agent.state, **agent_args)

    return Message(
        role="function",
        name=tool_name,
        content=output,
        function_call=None,
    )


async def _basic(agent: Agent) -> None:
    output = await get_result_message_simple(agent)
    if output is not None:
        agent.append(output)
    agent.state.next_step["module_type"] = "prompter"


async def maybe_prompt_to_search_post_act(output: Message) -> Optional[Message]:
    if len(output.content) > 4500:
        long_output_dir = "/home/agent/long_outputs"
        try:
            os.mkdir(long_output_dir)
        except FileExistsError:
            pass
        filename = f"{long_output_dir}/long_output_{str(1 + math.floor(random.random() * 9007199254740990))}.txt"
        with open(filename, "w") as f:
            f.write(output.content)
        output.content = prompt_to_search.format(
            filename=filename,
            output_start=output.content[:100],
            output_end=output.content[-100:],
        )
        return output


async def _always_save(agent: Agent) -> None:
    output = await get_result_message_simple(agent)
    output_dir = "/home/agent/tool_outputs"
    try:
        os.mkdir(output_dir)
    except FileExistsError:
        pass
    if output is not None:
        filename = f"{output_dir}/tool_output_{str(1 + math.floor(random.random() * 9007199254740990))}.txt"
        with open(filename, "w") as f:
            f.write(output.content)
        output.content += (
            "\n\n" + f"[Note: the above tool output has been saved to {filename}]"
        )
        agent.append(output, metadata={"saved_output_filename": filename})
    agent.state.next_step["module_type"] = "prompter"


async def _prompt_to_search(agent: Agent) -> None:
    output = await get_result_message_simple(agent)
    if output is not None:
        await maybe_prompt_to_search_post_act(output)
        agent.append(output)
    agent.state.next_step["module_type"] = "prompter"
