import re
import json
import asyncio
from typing import Dict, List, Tuple
from itertools import product
from functools import partial

from base import Agent, Message, hooks
from templates import (
    claude_basic_system_prompt,
    gpt_basic_system_prompt,
    compare_options_prompt_v1,
    compare_and_regenerate_prompt_v1,
    assess_and_backtrack_prompt
)
from pyhooks.types import (
    MiddlemanSettings,
    MiddlemanResult,
    OpenaiChatMessage,
)

async def _basic(agent: Agent) -> None:
    node_metadata = {
        "g__generation_metadata": agent.state.next_step["args"]["generation_metadata"],
    }
    agent.append(agent.state.next_step["args"]["options"][0], metadata=node_metadata)
    agent.state.next_step["module_type"] = "actor"


async def generate_comparison_claude_legacy(agent: Agent, middleman_settings: MiddlemanSettings, options_prompt_template: str, system_prompt: str=claude_basic_system_prompt) -> MiddlemanResult:
    wrapped_messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": "Your current task is the following: " + agent.state.task_string,
        },
    ]
    # This code is duplicated from generators:_claude_legacy_factory
    for msg in agent.state.next_step["args"]["messages"]:
        role = msg.role
        content = msg.content
        if msg.function_call is not None:
            tool_name = msg.function_call["name"]
            tool_args = msg.function_call["arguments"]
            content += f"<{tool_name}>{tool_args}</{tool_name}>"
        elif msg.role == "function":
            role = "user"
            content = f"<{msg.name}-output>{msg.content}</{msg.name}-output>"
        wrapped_messages.append({
            "role": role,
            "content": content,
        })
    formatted_options = ""
    for i, option in enumerate(agent.state.next_step["args"]["options"]):
        option_string = json.dumps({"content": option.content, "function_call": option.function_call})
        formatted_options += f"\n\nOption {i}:\n{option_string}"
    wrapped_messages.append({
        "role": "user",
        "content": options_prompt_template.replace("{{&options}}", formatted_options),
    })
    generation = await hooks.generate(
        messages=wrapped_messages,
        settings=middleman_settings,
    )
    return generation


async def generate_comparison_gpt(agent: Agent, middleman_settings: MiddlemanSettings, options_prompt_template: str, system_prompt: str=gpt_basic_system_prompt) -> MiddlemanResult:
    messages = agent.state.next_step["args"]["messages"]
    options = agent.state.next_step["args"]["options"]
    wrapped_messages = [
        OpenaiChatMessage(
            role="user",
            content=system_prompt,
        ),
        OpenaiChatMessage(
            role="user",
            content="You are assigned this task: " + agent.state.task_string,
        ),
    ]
    wrapped_messages += [OpenaiChatMessage(**msg.dict()) for msg in messages]
    tools = [
        {
            "name": k,
            "description": v["description"],
            "parameters": v["parameters"],
        }
        for k, v in agent.toolkit_dict.items()
    ]
    formatted_options = ""
    for i, option in enumerate(options):
        option_string = json.dumps({"content": option.content, "function_call": option.function_call})
        formatted_options += f"\n\nOption {i}:\n{option_string}"
    wrapped_messages.append(OpenaiChatMessage(
        role="user",
        content=options_prompt_template.replace("{{&options}}", formatted_options),
    ))
    generation = await hooks.generate(
        messages=wrapped_messages,
        settings=middleman_settings,
        functions=tools,
    )
    return generation


async def _compare_options_factory(agent: Agent, middleman_settings: MiddlemanSettings=None, comparison_generator=None) -> None:
    if middleman_settings is None or comparison_generator is None:
        raise ValueError("Do not call _compare_options_factory directly. Use a partial application of it instead.")

    options = agent.state.next_step["args"]["options"]
    choice_is_good = False
    while not choice_is_good:
        generation = await comparison_generator(agent, middleman_settings, compare_options_prompt_v1)
        choice = re.search(r"<FINAL CHOICE>\s*\[?\s*(\d+)\s*\]?\s*$", generation.outputs[0].completion)
        if choice is not None:
            choice = int(choice.group(1))
            if 0 <= choice < len(options):
                choice_is_good = True
    node_metadata = {
        "d__compare_options__original_options": options,
        "g__generation_metadata": agent.state.next_step["args"]["generation_metadata"],
    }
    agent.append(options[choice], metadata=node_metadata)
    agent.state.next_step["module_type"] = "actor"


async def _compare_and_regenerate_gpt_factory(agent: Agent, n_rounds: int=1, middleman_settings: MiddlemanSettings=None) -> None:
    if middleman_settings is None:
        raise ValueError("Do not call _compare_and_regenerate_factory directly. Use a partial application of it instead.")

    middleman_settings_copy = MiddlemanSettings(
        n=(1 if n_rounds < 2 else len(agent.state.next_step["args"]["options"])),
        model=middleman_settings.model,
        temp=middleman_settings.temp,
        max_tokens=middleman_settings.max_tokens,
        stop=middleman_settings.stop,
    )
    new_options = []
    function_call = None
    while not len(new_options):
        generations = await comparison_generator(agent, middleman_settings_copy, compare_and_regenerate_prompt_v1)
        for generation in generations.outputs:
            action = re.search(r"<FINAL ACTION>[^\{]*(\{.+\})[^\}]*$", generation.completion, re.DOTALL)
            try:
                action = json.loads(action.group(1))
                content = "" if "content" not in action else action["content"]
                function_call = None if "function_call" not in action else action["function_call"]
                assert isinstance(function_call, (None.__class__, dict))
                new_options.append(Message(
                    role="assistant",
                    content=content,
                    function_call=function_call,
                ))
            except (json.JSONDecodeError, AttributeError, AssertionError):
                continue
    if len(new_options) == 1:
        node_metadata = {
            "d__compare_and_regenerate__original_options": agent.state.next_step["args"]["options"],
            "g__generation_metadata": agent.state.next_step["args"]["generation_metadata"],
        }
        agent.append(new_options[0], metadata=node_metadata)
        agent.state.next_step["module_type"] = "actor"
    else:
        agent.state.next_step["args"]["options"] = new_options
        await _compare_and_regenerate_gpt_factory(agent, n_rounds=n_rounds-1, middleman_settings=middleman_settings)

async def _assess_and_backtrack_gpt_factory(agent: Agent, middleman_settings: MiddlemanSettings=None, comparison_generator=None) -> None:
    if middleman_settings is None:
        raise ValueError("Do not call _assess_and_backtrack_gpt_factory directly. Use a partial application of it instead.")

    middleman_settings_copy = MiddlemanSettings(
        n=1,
        model=middleman_settings.model,
        temp=middleman_settings.temp,
        max_tokens=middleman_settings.max_tokens,
        stop=middleman_settings.stop,
    )

    action = agent.state.next_step["args"]["options"][0]
    approved = None
    while approved is None:
        generations = await comparison_generator(agent, middleman_settings_copy, assess_and_backtrack_prompt)
        completion = generations.outputs[0].completion
        print(completion)

        if "APPROVE" in completion:
            approved = True
            node_metadata = {
                "d__compare_and_regenerate__original_options": agent.state.next_step["args"]["options"],
                "g__generation_metadata": agent.state.next_step["args"]["generation_metadata"],
            }
            agent.append(action, metadata=node_metadata)
            agent.state.next_step["module_type"] = "actor"
        elif "REJECT" in completion:
            approved = False
            agent.append(Message(
                role="user",
                content=f"The following action is not on the right track: {action.content}\n\nConsider a different approach.",
                function_call=None,
            ))
            agent.state.next_step["module_type"] = "generator"

models_and_comparison_generators = [
    ("gpt-4-0613", "4", generate_comparison_gpt),
    ("gpt-4-turbo-2024-04-09", "4t", generate_comparison_gpt),
    ("gpt-4o-2024-05-13", "4o", generate_comparison_gpt),
    ("gpt-4o-mini-2024-07-18", "4om", generate_comparison_gpt),
    ("claude-3-opus-20240229", "c3o", generate_comparison_claude_legacy),
    ("claude-3-sonnet-20240229", "c3s", generate_comparison_claude_legacy),
    ("claude-3-haiku-20240229", "c3h", generate_comparison_claude_legacy),
]
for model, desc, comparison_generator in models_and_comparison_generators:
    globals()[f"_compare_options_{desc}"] = partial(_compare_options_factory, middleman_settings=MiddlemanSettings(
        n=1,
        model=model,
        temp=1,
        max_tokens=3600,
        stop=["</FINAL CHOICE>"]
    ), comparison_generator=comparison_generator)

# TODO: Make compatible with claude_legacy format (currently the resulting claude_legacy compatible functions are not intended to work)
for (model, desc, _), n_rounds in product(models_and_comparison_generators, range(1, 6)):
    globals()[f"_compare_and_regenerate_{n_rounds}_rounds_gpt_{desc}"] = partial(_compare_and_regenerate_gpt_factory, n_rounds=n_rounds, middleman_settings=MiddlemanSettings(
        n=1,
        model=model,
        temp=1,
        max_tokens=4096,
        stop=["</FINAL ACTION>"]
    ))

for (model, desc, comparison_generator) in models_and_comparison_generators:
    globals()[f"_assess_and_backtrack_gpt_{desc}"] = partial(_assess_and_backtrack_gpt_factory, middleman_settings=MiddlemanSettings(
        n=1,
        model=model,
        temp=1,
        max_tokens=4096,
        stop=[]
    ), comparison_generator=comparison_generator)
