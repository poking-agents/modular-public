import copy
from functools import partial
from itertools import product
from typing import Optional, cast

from pyhooks.types import MiddlemanSettings, OpenaiChatMessage
import anthropic
from anthropic.types import Message as AnthropicMessage

from base import Agent, Message, hooks
from templates import (
    claude_basic_system_prompt,
    get_tool_descriptions,
    gpt_basic_system_prompt,
)

ANTHROPIC_STOP_SEQUENCE_LIMIT = 4


async def _claude_legacy_factory(
    agent: Agent, middleman_settings: Optional[MiddlemanSettings] = None
) -> None:
    if middleman_settings is None:
        raise ValueError(
            "Do not call _claude_legacy_factory directly. Use a partial application of it instead."
        )

    client = anthropic.AsyncClient()

    # Prepare stop sequences
    stop_sequences = [f"</{tool}" for tool in agent.toolkit_dict][
        :ANTHROPIC_STOP_SEQUENCE_LIMIT
    ]

    # Prepare system and messages
    system_content = claude_basic_system_prompt.format(
        tools="\n".join(get_tool_descriptions(list(agent.toolkit_dict.keys())))
    )

    messages = []
    messages.append(
        AnthropicMessage(
            role="user",
            content="Your current task is the following: " + agent.state.task_string,
        )
    )

    # Convert existing messages
    for msg in agent.state.next_step["args"]["messages"]:
        content = msg.content
        role = "assistant" if msg.role == "assistant" else "user"

        if msg.function_call is not None:
            tool_name = msg.function_call["name"]
            tool_args = msg.function_call["arguments"]
            content += f"<{tool_name}>{tool_args}</{tool_name}>"
        elif msg.role == "function":
            role = "user"
            content = f"<{msg.name}-output>{msg.content}</{msg.name}-output>"

        messages.append(AnthropicMessage(role=role, content=content))

    # Add prompt for function call if needed
    if messages and messages[-1].role == "assistant":
        messages.append(
            AnthropicMessage(
                role="user",
                content="No function call was included in the last message. Please include a function call in the next message using the <[tool_name]> [args] </[tool_name]> syntax.",
            )
        )

    # Make API call
    response = await client.messages.create(
        model=middleman_settings.model,
        messages=messages,
        system=system_content,
        max_tokens=middleman_settings.max_tokens,
        temperature=middleman_settings.temp,
        stop_sequences=stop_sequences,
        n=middleman_settings.n,
    )

    # Process response
    processed_messages = []
    for message in response.content:
        generation = message.text
        last_tool_loc, last_tool = max(
            [(generation.find(f"<{tool}>"), tool) for tool in agent.toolkit_dict]
        )

        content = generation
        function_call = None
        if last_tool_loc != -1:
            content, raw_function_call = generation.rsplit(f"<{last_tool}>", maxsplit=1)
            function_call = {
                "type": "function",
                "name": last_tool,
                "arguments": raw_function_call.removesuffix(f"</{last_tool}>"),
            }

        processed_messages.append(
            Message(
                role="assistant",
                content=content,
                function_call=function_call,
            )
        )

    # Update agent state
    agent.state.next_step["module_type"] = "discriminator"
    agent.state.next_step["args"]["options"] = processed_messages
    agent.state.next_step["args"]["generation_metadata"] = {
        "model": response.model,
        "usage": response.usage,
    }


claude_legacy_compat_models = [
    ("claude-3-opus-20240229", "c3o"),
    ("claude-3-sonnet-20240229", "c3s"),
    ("claude-3-haiku-20240307", "c3h"),
    ("claude-3-5-sonnet-20240620", "c3.5s"),
    ("claude-3-5-sonnet-20241022", "c3.5sv2"),
]
for model_pair, n in product(claude_legacy_compat_models, [1, 2, 4, 8, 16, 32, 64]):
    model, desc = model_pair
    globals()[f"_claude_legacy_{n}x{desc}"] = partial(
        _claude_legacy_factory,
        middleman_settings=MiddlemanSettings(
            n=n,
            model=model,
            temp=1,
            max_tokens=4096,
        ),
    )


async def _gpt_basic_factory(
    agent: Agent, middleman_settings: MiddlemanSettings | None = None
) -> None:
    if middleman_settings is None:
        raise ValueError(
            "Do not call _gpt_basic_factory directly. Use a partial application of it instead."
        )

    messages = agent.state.next_step["args"]["messages"]

    # make a copy so we can decrement n later
    middleman_settings_copy = copy.deepcopy(middleman_settings)
    wrapped_messages = [
        {
            "role": "user",
            "content": gpt_basic_system_prompt,
        },
        {
            "role": "user",
            "content": "You are assigned this task: " + agent.state.task_string,
        },
    ]
    wrapped_messages += [msg.dict() for msg in messages]
    tools = [
        {
            "name": k,
            "description": v["description"],
            "parameters": v["parameters"],
        }
        for k, v in agent.toolkit_dict.items()
    ]
    num_to_generate = middleman_settings.n
    generations = []
    generation_metadata = {}
    while middleman_settings_copy.n > 0:
        generation_n = await hooks.generate(
            messages=[cast(OpenaiChatMessage, msg) for msg in wrapped_messages],
            settings=middleman_settings_copy,
            functions=tools,
        )
        generations += [
            g
            for g in generation_n.outputs or []
            if g.function_call is None or g.function_call["name"] in agent.toolkit_dict
        ]
        middleman_settings_copy.n = num_to_generate - len(generations)
        generation_metadata = {
            k: v for k, v in generation_n.dict().items() if k != "outputs"
        }

    options = [
        Message(
            role="assistant",
            content=g.completion,
            function_call=(g.function_call if g.function_call else None),
        )
        for g in generations
    ]
    agent.state.next_step["module_type"] = "discriminator"
    agent.state.next_step["args"].update(
        generation_metadata=generation_metadata,
        options=options,
    )


gpt_models = [
    ("gpt-4-0613", "4"),
    ("gpt-4-turbo-2024-04-09", "4t"),
    ("gpt-4o-2024-05-13", "4o"),
    ("gpt-4o-mini-2024-07-18", "4om"),
    ("o1-preview-2024-09-12", "o1p"),
    ("o1-mini-2024-09-12", "o1m"),
]
for model_pair, n in product(gpt_models, [1, 2, 4, 8, 16, 32, 64]):
    model, desc = model_pair
    globals()[f"_gpt_basic_{n}x{desc}"] = partial(
        _gpt_basic_factory,
        middleman_settings=MiddlemanSettings(
            n=n, model=model, temp=1, max_tokens=4096, stop=[]
        ),
    )
