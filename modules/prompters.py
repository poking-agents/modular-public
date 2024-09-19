import json
from typing import Any, List

import tiktoken

from base import Agent, Message, Node
from templates import (
    notice_retroactively_trimmed_prompt,
    notice_retroactively_using_saved_output,
)


def _format_score_message(message: Message) -> Message:
    # Some actors add extra lines to the score message, but the score output
    # is always the first line.
    score_output, *extra = message.content.splitlines()
    result = json.loads(score_output)

    score_content = []
    score = result.get("score", None)
    if result.get("status", "noScore") == "invalidSubmission":
        score_content.append("Your submission is invalid.")
    elif score is not None:
        score_content.append(f"The score is {score}.")

    content = [*score_content]
    details: dict[str, Any] = result.get("message", {})
    if details:
        content.extend(
            [
                "",
                "Other details about your submission:",
            ]
        )
        for key, value in details.items():
            content.append(f"  - {key}: {value}")

    exec_result: dict[str, Any] = result.get("execResult", {})
    if exec_result:
        content.extend(
            [
                "",
                "Process execution details:",
                f"  - return code: {exec_result.get('exitStatus', '')}",
                f"  - stdout: {exec_result.get('stdout', '')}",
                f"  - stderr: {exec_result.get('stderr', '')}",
            ]
        )

    if len(content) > len(score_content):
        # Repeat score message at the end so it doesn't get buried in
        # verbose output.
        content.extend(["", *score_content])

    return Message(
        **{
            **message.dict(),
            "content": "\n".join((*content, *extra)),
        }
    )


async def _basic(agent: Agent) -> None:
    """
    Retrieve all messages in order of agent.state.get_path() with no modifications.
    (lineage of most recent node)
    """
    agent.state.next_step["module_type"] = "generator"
    agent.state.next_step["args"]["messages"] = [
        _format_score_message(message)
        if (
            (message := agent.state.nodes[node_id].message).role == "function"
            and message.name == "score"
        )
        else message
        for node_id in agent.state.get_path()
    ]


def trim_message_list(messages: List[Message], target_tok_length: int) -> List[Message]:
    """
    Trim messages by removing each message starting with the 5th message, until
    the total token length is less than target_tok_length. Include a message in
    the trimmed portion indicating that the sequence has been trimmed.

    Note that this function always uses tiktoken's cl100k base tokenizer, and
    disregards many details about how to use it for message formats.
    TODO: use the correct tokenizer for any given situation, and use it properly.
    """
    enc = tiktoken.get_encoding("cl100k_base")
    tokens_to_use = target_tok_length - len(
        enc.encode(notice_retroactively_trimmed_prompt)
    )
    for msg in messages[:4]:
        tokens_to_use -= len(enc.encode(msg.content))
        tokens_to_use -= len(enc.encode(json.dumps(msg.function_call)))

    tail_messages_to_use = []
    for msg in messages[4:][::-1]:
        tokens_to_use -= len(enc.encode(msg.content))
        tokens_to_use -= len(enc.encode(json.dumps(msg.function_call)))
        if tokens_to_use < 0:
            break
        tail_messages_to_use.append(msg)
    if tokens_to_use >= 0:
        return messages

    return (
        messages[:4]
        + [
            Message(
                role="user",
                content=notice_retroactively_trimmed_prompt,
                function_call=None,
            )
        ]
        + tail_messages_to_use[::-1]
    )


def _get_trimmed_message(node: Node, token_usage_fraction: float) -> Message:
    message = node.message
    message = (
        _format_score_message(message)
        if message.role == "function" and message.name == "score"
        else message
    )

    needs_trimming = message.role == "function" and (
        len(message.content) > 100_000
        or (len(message.content) > 8_000 and token_usage_fraction > 0.5)
    )
    if not needs_trimming:
        return message

    content = (
        notice_retroactively_using_saved_output.replace(
            "{{&filename}}",
            node.metadata["saved_output_filename"],
        )
        .replace("{{&output_start}}", message.content[:300])
        .replace("{{&output_end}}", message.content[-300:])
    )
    return Message(
        role="function",
        content=content,
        name=message.name,
        function_call=None,
    )


async def _context_and_usage_aware(agent: Agent) -> None:
    node_ids = agent.state.get_path()
    messages = []
    token_usage_fraction = agent.state.token_usage / agent.state.token_limit
    for node_id in node_ids:
        messages.append(
            _get_trimmed_message(agent.state.nodes[node_id], token_usage_fraction)
        )
    token_usage_message = f"So far in this attempt at the task, you have used {agent.state.token_usage} tokens, out of the total limit of {agent.state.token_limit} tokens."
    if token_usage_fraction > (9 / 10):
        token_usage_message += " You should submit a final answer soon."
    elif token_usage_fraction > (3 / 4):
        token_usage_message += " You should attempt to reach a final answer soon."
    messages.append(
        Message(
            role="user",
            content=token_usage_message,
            name=None,
            function_call=None,
        )
    )
    # TODO: be more principled about the target_tok_length setting
    if "claude" in agent.settings.generator:
        target_tok_length = 0.75 * 200_000
    else:
        target_tok_length = 0.75 * 128_000
    messages = trim_message_list(messages, target_tok_length)
    agent.state.next_step["module_type"] = "generator"
    agent.state.next_step["args"]["messages"] = messages
