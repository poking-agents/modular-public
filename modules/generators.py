import copy
from functools import partial
from itertools import product
from typing import List, Optional, Dict, Any

from pyhooks.types import MiddlemanSettings
import anthropic
from anthropic.types import MessageParam
from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)

from base import Agent, Message
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

    messages: List[MessageParam] = []
    messages.append(
        MessageParam(
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

        messages.append(MessageParam(role=role, content=content))

    # Add prompt for function call if needed
    if messages and messages[-1]["role"] == "assistant":
        messages.append(
            MessageParam(
                role="user",
                content="No function call was included in the last message. Please include a function call in the next message using the <[tool_name]> [args] </[tool_name]> syntax.",
            )
        )

    # Make API call
    response = await client.messages.create(
        model=middleman_settings.model,
        messages=messages,
        system=system_content,
        max_tokens=middleman_settings.max_tokens or 4096,
        temperature=middleman_settings.temp,
        stop_sequences=stop_sequences,
    )

    # Process response
    processed_messages = []
    for message in response.content:
        if message.type != "text":
            raise ValueError(f"Unsupported message type: {message.type}")

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
    client = AsyncOpenAI()

    # make a copy so we can decrement n later
    middleman_settings_copy = copy.deepcopy(middleman_settings)
    wrapped_messages: List[ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": gpt_basic_system_prompt,
        },
        {
            "role": "user",
            "content": "You are assigned this task: " + agent.state.task_string,
        },
    ]

    # Convert existing messages to proper OpenAI format
    for msg in messages:
        msg_dict = msg.dict()
        wrapped_msg: Dict[str, Any] = {
            "role": msg_dict["role"],
            "content": msg_dict["content"],
        }
        if msg_dict.get("function_call"):
            wrapped_msg["tool_calls"] = [
                {
                    "type": "function",
                    "function": {
                        "name": msg_dict["function_call"]["name"],
                        "arguments": msg_dict["function_call"]["arguments"],
                    },
                }
            ]
        wrapped_messages.append(wrapped_msg)  # type: ignore

    tools: List[ChatCompletionToolParam] = [
        {
            "type": "function",
            "function": {
                "name": k,
                "description": v["description"],
                "parameters": v["parameters"],
            },
        }
        for k, v in agent.toolkit_dict.items()
    ]

    num_to_generate = middleman_settings.n
    generations = []
    generation_metadata: Dict[str, Any] = {}

    while middleman_settings_copy.n > 0:
        try:
            response: ChatCompletion = await client.chat.completions.create(
                model=middleman_settings.model,
                messages=wrapped_messages,
                tools=tools,
                temperature=middleman_settings.temp,
                max_tokens=middleman_settings.max_tokens or 4096,
                n=middleman_settings_copy.n,
            )

            for choice in response.choices:
                message = choice.message
                function_call = None
                if message.tool_calls:
                    tool_call = message.tool_calls[0].function
                    if tool_call.name in agent.toolkit_dict:
                        function_call = {
                            "type": "function",
                            "name": tool_call.name,
                            "arguments": tool_call.arguments,
                        }

                generations.append(
                    {
                        "completion": message.content or "",
                        "function_call": function_call,
                    }
                )

            if response.usage:
                generation_metadata = {
                    "model": response.model,
                    "usage": {
                        "completion_tokens": response.usage.completion_tokens,
                        "prompt_tokens": response.usage.prompt_tokens,
                        "total_tokens": response.usage.total_tokens,
                    },
                }

            middleman_settings_copy.n = num_to_generate - len(generations)

        except Exception as e:
            print(f"Error during OpenAI API call: {str(e)}")
            break

    options = [
        Message(
            role="assistant",
            content=g["completion"],
            function_call=g["function_call"],
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
