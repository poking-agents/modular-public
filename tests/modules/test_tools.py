from __future__ import annotations

import datetime
import json
import random
from typing import TYPE_CHECKING

import pyhooks
import pytest

import base
import modules.actors as actors
import modules.generators as generators
import modules.prompters as prompters
import modules.tools as tools
import templates

if TYPE_CHECKING:
    from pyfakefs.fake_filesystem import FakeFilesystem
    from pytest_mock import MockerFixture


@pytest.mark.asyncio
async def test_score_fn(mocker: MockerFixture):
    expected_output = pyhooks.types.ScoreResult(
        status="scoringSucceeded",
        score=0.75,
        message={"foo": "bar"},
        execResult=pyhooks.types.ExecResult(
            stdout="stdout",
            stderr="stderr",
            exitStatus=0,
        ),
    )
    mocker.patch("pyhooks.Hooks.score", autospec=True, return_value=expected_output)
    action_mock = mocker.patch("pyhooks.Hooks.action", autospec=True, return_value=mocker.AsyncMock()())

    output = await tools.score_fn(base.State(task_string="test task"))

    assert isinstance(output, str)
    assert json.loads(output) == expected_output.model_dump()
    action_mock.assert_called_once()
    assert isinstance(action_mock.call_args.args[1], dict)
    assert action_mock.call_args.args[1]["type"] == "score"
    assert isinstance(action_mock.call_args.args[1]["args"], dict)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ["actor", "expected_output_file"],
    [
        ("_basic", None),
        (
            "_always_save",
            "/home/agent/tool_outputs/tool_output_7605875871743421.txt",
        ),
    ],
)
async def test_score_feedback(
    fs: FakeFilesystem,
    mocker: MockerFixture,
    actor: str,
    expected_output_file: str | None,
):
    random.seed(0)
    fs.create_dir("/home/agent")

    score_result = pyhooks.types.ScoreResult(
        status="scoringSucceeded",
        score=0.75,
        message={"foo": "bar"},
        execResult=pyhooks.types.ExecResult(
            stdout="stdout",
            stderr="stderr",
            exitStatus=0,
        ),
    )
    score_mock = mocker.patch(
        "pyhooks.Hooks.score",
        autospec=True,
        return_value=score_result,
    )
    mocker.patch("pyhooks.Hooks.log", autospec=True)
    mocker.patch("pyhooks.Hooks.log_with_attributes", autospec=True)
    action_mock = mocker.patch("pyhooks.Hooks.action", autospec=True, return_value=mocker.AsyncMock()())
    generate_mock = mocker.patch(
        "pyhooks.Hooks.generate",
        autospec=True,
        return_value=mocker.Mock(
            outputs=[mocker.Mock(completion="test completion")],
            model_dump=mocker.Mock(return_value={"foo": "bar"}),
        ),
    )

    time_limit = 5
    state = base.State(
        task_string="test task",
        next_step={"module_type": "actor", "args": {}},
        token_limit=1000000,
        timeout=5,
        time_limit=time_limit,
    )
    agent = base.Agent(
        state=state,
        settings=base.Settings(
            actor=actor,
            discriminator="_basic",
            generator="_claude_legacy_1xc3.5s",
            prompter="_basic",
            toolkit="_basic",
            autosubmit=False,
        ),
        toolkit_dict={**tools._basic, **tools.scoring_tools},
    )
    agent.append(
        base.Message(
            role="assistant",
            content="score",
            function_call={
                "name": "score",
                "type": "function",
                "arguments": "",
            },
        )
    )

    await getattr(actors, agent.settings.actor)(agent)
    assert score_mock.call_count == 1, agent.state.nodes[
        agent.state.last_node_id
    ].message
    action_mock.assert_called_once()
    assert isinstance(action_mock.call_args.args[1], dict)
    assert action_mock.call_args.args[1]["type"] == "score"
    assert isinstance(action_mock.call_args.args[1]["args"], dict)
    expected_content = json.dumps(score_result.model_dump())
    if expected_output_file is not None:
        expected_content += f"\n\n[Note: the above tool output has been saved to {expected_output_file}]"
    assert agent.state.nodes[agent.state.last_node_id].message == base.Message(
        role="function",
        content=expected_content,
        name="score",
        function_call=None,
    )
    assert state.next_step["module_type"] == "prompter"

    await getattr(prompters, agent.settings.prompter)(agent)
    assert state.next_step["module_type"] == "generator"

    await getattr(generators, agent.settings.generator)(agent)
    assert state.next_step["module_type"] == "discriminator"

    generate_mock.assert_called_once()
    call_args = generate_mock.call_args
    assert len(call_args.args) == 1
    assert isinstance(call_args.args[0], pyhooks.Hooks)

    assert {*call_args.kwargs} == {"messages", "settings"}
    assert isinstance(call_args.kwargs["settings"], pyhooks.MiddlemanSettings)

    assert isinstance(call_args.kwargs["messages"], list)
    messages: list[pyhooks.OpenaiChatMessage] = call_args.kwargs["messages"]
    assert len(messages) == 4
    assert {type(message) for message in messages} == {pyhooks.OpenaiChatMessage}

    expected_score_message = prompters._format_score_message(
        base.Message(
            role="function",
            name="score",
            content=score_result.json(),
        )
    )
    if expected_output_file is not None:
        expected_score_message.content += f"\n\n[Note: the above tool output has been saved to {expected_output_file}]"

    for idx_message, (expected_role, expected_content) in enumerate(
        [
            (
                "system",
                templates.claude_basic_system_prompt.format(
                    tools="\n".join(
                        templates.get_tool_descriptions(list(agent.toolkit_dict.keys()))
                    )
                ),
            ),
            ("user", "Your current task is the following: test task"),
            ("assistant", "score<score></score>"),
            (
                "user",
                f"<score-output>{expected_score_message.content}</score-output>",
            ),
        ]
    ):
        message = messages[idx_message]
        assert message.role == expected_role
        assert message.content == expected_content

    if expected_output_file is not None:
        assert fs.exists(expected_output_file)


@pytest.mark.asyncio
async def test_score_log_fn(mocker: MockerFixture):
    expected_output = [
        pyhooks.types.ScoreLogEntry(
            score=0.75,
            message={"foo": "bar"},
            scoredAt=datetime.datetime.now().isoformat(),
            elapsedSeconds=1.0,
        )
    ]
    mocker.patch("pyhooks.Hooks.scoreLog", autospec=True, return_value=expected_output)
    action_mock = mocker.patch("pyhooks.Hooks.action", autospec=True, return_value=mocker.AsyncMock()())

    output = await tools.score_log_fn(base.State(task_string="test task"))

    assert isinstance(output, str)
    assert json.loads(output) == [x.model_dump() for x in expected_output]
    action_mock.assert_called_once()
    assert isinstance(action_mock.call_args.args[1], dict)
    assert action_mock.call_args.args[1]["type"] == "score_log"
    assert isinstance(action_mock.call_args.args[1]["args"], dict)


@pytest.mark.asyncio
async def test_run_python(mocker: MockerFixture):
    expected_output = "test output"
    mocker.patch("base.actions.run_python", autospec=True, return_value=expected_output)
    action_mock = mocker.patch("pyhooks.Hooks.action", autospec=True, return_value=mocker.AsyncMock()())

    test_code = "test code"
    output = await tools.run_python(base.State(task_string="test task", timeout=5), test_code)

    assert output == expected_output
    action_mock.assert_called_once()
    assert isinstance(action_mock.call_args.args[1], dict)
    assert action_mock.call_args.args[1]["type"] == "run_python"
    assert isinstance(action_mock.call_args.args[1]["args"], dict)
    assert action_mock.call_args.args[1]["args"]["code"] == test_code


@pytest.mark.asyncio
async def test_run_bash_state(mocker: MockerFixture):
    expected_output = json.dumps({
        "stdout": "test stdout",
        "stderr": "test stderr",
        "status": 0
    })
    mocker.patch("base.actions.run_bash", autospec=True, return_value=expected_output)
    action_mock = mocker.patch("pyhooks.Hooks.action", autospec=True, return_value=mocker.AsyncMock()())

    test_command = "test command"
    output = await tools.run_bash_state(base.State(task_string="test task", timeout=5), test_command)

    assert output == "test stdout\ntest stderr\nExit code: 0"
    action_mock.assert_called_once()
    assert isinstance(action_mock.call_args.args[1], dict)
    assert action_mock.call_args.args[1]["type"] == "run_bash"
    assert isinstance(action_mock.call_args.args[1]["args"], dict)
    assert action_mock.call_args.args[1]["args"]["command"] == test_command


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ["submission", "expected_args"],
    [
        (None, ""),
        ("test submission", "test submission"),
        (["this", "is", "a", "list"], '["this", "is", "a", "list"]'),
        ({"foo": "bar"}, '{"foo": "bar"}'),
        (True, "true"),
        (123, "123"),
        ("", ""),
    ],
)
async def test_return_fn(submission: Any, expected_args: str, mocker: MockerFixture):
    submit_mock = mocker.patch("pyhooks.Hooks.submit", autospec=True)

    await tools.return_fn(base.State(task_string="test task"), submission)
    submit_mock.assert_awaited_once()
    assert submit_mock.call_args.args[1] == expected_args


from typing import Any

@pytest.mark.asyncio
@pytest.mark.parametrize(
    ["submission", "expected_submit_arg", "existing_submissions", "expect_existing_submit"],
    [
        ("test submission", "test submission", [], False),
        ("test submission", "test submission", ["test submission"], True),
        (None, "", [], False),
        ("", "", [], False),
        (123, "123", [], False),
        (123, "123", ['123'], True),
        (123, "123", ['456', '123', '789'], True),
        (123, "123", ['12', '3', '456', '7'], False),
        (["a", "b"], '["a", "b"]', [], False),
        (["a", "b"], '["a", "b"]', ['["a", "b"]'], True),
        (["b", "a"], '["b", "a"]', ['["a", "b"]'], False),
        (True, "true", [], False),
        (True, "true", ['true'], True),
        ({"key": "value"}, '{"key": "value"}', [], False),
        ({"key": "value"}, '{"key": "value"}', ['{"key": "value"}'], True),
        ({"key": "value"}, '{"key": "value"}', ['{"key": "not-value"}','{"key": "value"}'], True),
        ("repeat submission", "repeat submission", ["repeat submission"], True),
        (None, "", [""], True),
    ],
)
@pytest.mark.parametrize(
    ["token_limit", "token_usage", "expect_limit_submit", "expect_more_than_10", "expect_less_than_10"],
    [
        # Ample tokens remaining (>90%)
        (1000, 100, False, True, False),
        # Low tokens remaining (<90%, >5%)
        (1000, 920, False, False, True),
        # Very low tokens remaining (<5%)
        (1000, 970, True, False, False),
    ],
)
async def test_double_return_fn(
    submission: Any,
    expected_submit_arg: str,
    existing_submissions: list[str],
    expect_existing_submit: bool,
    token_limit: int,
    token_usage: int,
    expect_limit_submit: bool,
    expect_more_than_10: bool,
    expect_less_than_10: bool,
    mocker: MockerFixture,
):
    submit_mock = mocker.patch("pyhooks.Hooks.submit", autospec=True)
    state = base.State(
        task_string="test task",
        token_limit=token_limit,
        token_usage=token_usage,
        submissions=existing_submissions.copy(),
    )

    result = await tools.double_return_fn(state, submission)

    if expect_limit_submit or expect_existing_submit:
        submit_mock.assert_awaited_once()
        assert submit_mock.call_args.args[1] == expected_submit_arg
    else:
        submit_mock.assert_not_awaited()
        if result:
            if expect_more_than_10:
                assert "more than 10%" in result
                assert "less than 10%" not in result
            if expect_less_than_10:
                assert "less than 10%" in result
                assert "more than 10%" not in result
