import asyncio  # noqa: F401
import json
from typing import Any

import pytest
import pytest_mock

import base
import modules.actors as actors
import templates


@pytest.fixture(name="tool_object")
def fixture_tool_object(request: pytest.FixtureRequest):
    tool_info = getattr(request, "param", {})

    tool_fn = tool_info["function"]

    async def async_tool_fn(*args, **kwargs):
        return tool_fn(*args, **kwargs)

    return {**tool_info, "function": async_tool_fn}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ["tool_object", "function_call", "expected_output"],
    [
        (
            TOOL_OBJECT_NO_ARGS := {
                "function": lambda _: json.dumps({"score": 1}),
                "parameters": {},
            },
            {"name": "score", "arguments": ""},
            base.Message(
                role="function",
                name="score",
                content='{"score": 1}',
                function_call=None,
            ),
        ),
        (
            TOOL_OBJECT_NO_ARGS,
            {"name": "score", "arguments": "{}"},
            base.Message(
                role="function",
                name="score",
                content='{"score": 1}',
                function_call=None,
            ),
        ),
        (
            TOOL_OBJECT_NO_ARGS,
            {"name": "score", "arguments": "Extra stuff"},
            RESPONSE_REJECT := base.Message(
                role="user",
                content=templates.reject_arguments_prompt,
                function_call=None,
            ),
        ),
        (
            TOOL_OBJECT_ONE_ARG := {
                "function": lambda _, comment: json.dumps(
                    {"score": 1, "comment": comment}
                ),
                "parameters": {"required": ["comment"]},
            },
            {"name": "score", "arguments": ""},
            base.Message(
                role="function",
                name="score",
                content='{"score": 1, "comment": ""}',
                function_call=None,
            ),
        ),
        (
            TOOL_OBJECT_ONE_ARG,
            {"name": "score", "arguments": "2"},
            base.Message(
                role="function",
                name="score",
                content='{"score": 1, "comment": 2}',
                function_call=None,
            ),
        ),
        (
            TOOL_OBJECT_ONE_ARG,
            {"name": "score", "arguments": "hello"},
            base.Message(
                role="function",
                name="score",
                content='{"score": 1, "comment": "hello"}',
                function_call=None,
            ),
        ),
        (
            TOOL_OBJECT_ONE_ARG,
            {
                "name": "score",
                "arguments": json.dumps({"comment": "hello", "extra": "stuff"}),
            },
            RESPONSE_REJECT,
        ),
        (
            TOOL_OBJECT_TWO_ARG := {
                "function": lambda _, comment, score: json.dumps(
                    {"score": score, "comment": comment}
                ),
                "parameters": {"required": ["comment", "score"]},
            },
            {
                "name": "score",
                "arguments": json.dumps(
                    {
                        "comment": "hello",
                        "score": 1,
                    }
                ),
            },
            base.Message(
                role="function",
                name="score",
                content='{"score": 1, "comment": "hello"}',
            ),
        ),
        (
            TOOL_OBJECT_TWO_ARG,
            {"name": "score", "arguments": "hello"},
            RESPONSE_REJECT,
        ),
        (
            TOOL_OBJECT_TWO_ARG,
            {"name": "score", "arguments": json.dumps({"comment": "hello"})},
            RESPONSE_REJECT,
        ),
    ],
    indirect=["tool_object"],
)
async def test_get_result_message_simple(
    mocker: pytest_mock.MockerFixture,
    tool_object: dict[str, Any],
    function_call: dict,
    expected_output: base.Message,
):
    agent = mocker.Mock(
        state=mocker.Mock(
            last_node_id=0,
            nodes=[
                mocker.Mock(
                    message=base.Message(
                        role="assistant",
                        content="<score>",
                        function_call=function_call,
                    ),
                )
            ],
        ),
        toolkit_dict={"score": tool_object},
    )

    output = await actors.get_result_message_simple(agent)

    assert output == expected_output
