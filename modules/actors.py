import os
import math
import json
import random
from typing import Dict, Tuple

from templates import reject_arguments_prompt, reject_command_prompt, prompt_to_search
from base import Agent, Message

async def get_result_message_simple(agent: Agent) -> Message:
    last_node = agent.state.nodes[agent.state.last_node_id]
    if last_node.message.function_call:
        tool_name = last_node.message.function_call["name"]
        tool_args = last_node.message.function_call["arguments"]
        output = ""
        if tool_name in agent.toolkit_dict:
            try: # parse arguments as JSON to be unpacked on function call
                json_args = json.loads(tool_args)
                if isinstance(json_args, int):
                    # we probably want to treat this as in the case below: as the single argument
                    raise TypeError
                output = await agent.toolkit_dict[tool_name]["function"](agent.state, **json_args)
            except (json.JSONDecodeError, TypeError):
                try: # assume there is only one required argument, and it's the passed string
                    required_args = agent.toolkit_dict[tool_name]["parameters"]["required"]
                    if len(required_args) != 1:
                        raise TypeError
                    tool_args = {required_args[0]: tool_args}
                    output = await agent.toolkit_dict[tool_name]["function"](agent.state, **tool_args)
                except (TypeError): # tell agent the command failed
                    return Message(
                        role="user",
                        content=reject_arguments_prompt,
                        function_call=None,
                    )
            return Message(
                role="function",
                name=tool_name,
                content=output,
                function_call=None,
            )
        else:
            return Message(
                role="user",
                content=reject_command_prompt,
                function_call=None,
            )


async def _basic(agent: Agent) -> None:
    output = await get_result_message_simple(agent)
    if output is not None:
        agent.append(output)
    agent.state.next_step["module_type"] = "prompter"


async def maybe_prompt_to_search_post_act(output: Message) -> Message:
    if len(output.content) > 4500:
        long_output_dir = "/home/agent/long_outputs"
        try:
            os.mkdir(long_output_dir)
        except FileExistsError:
            pass
        filename = f"{long_output_dir}/long_output_{str(1 + math.floor(random.random() * 9007199254740990))}.txt"
        with open(filename, "w") as f:
            f.write(output.content)
        output.content = (
            prompt_to_search.replace("{{&filename}}", filename)
            .replace("{{&output_start}}", output.content[:100])
            .replace("{{&output_end}}", output.content[-100:])
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
        output.content += "\n\n" + f"[Note: the above tool output has been saved to {filename}]"
        agent.append(output, metadata={"saved_output_filename": filename})
    agent.state.next_step["module_type"] = "prompter"


async def _prompt_to_search(agent: Agent) -> None:
    output = await get_result_message_simple(agent)
    if output is not None:
        await maybe_prompt_to_search_post_act(output)
        agent.append(output)
    agent.state.next_step["module_type"] = "prompter"
