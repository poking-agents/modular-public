import datetime
import json
import random

import pyfakefs
import pyhooks
import pytest
import pytest_mock

import base
import modules.actors as actors
import modules.generators as generators
import modules.prompters as prompters
import modules.tools as tools
import templates


@pytest.mark.asyncio
async def test_score_fn(mocker: pytest_mock.MockerFixture):
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

    output = await tools.score_fn(None)

    assert isinstance(output, str)
    assert json.loads(output) == expected_output


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
    fs: pyfakefs.fake_filesystem.FakeFilesystem,
    mocker: pytest_mock.MockerFixture,
    actor: str,
    expected_output_file: str | None,
):
    random.seed(0)

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
    generate_mock = mocker.patch(
        "pyhooks.Hooks.generate",
        autospec=True,
        return_value=mocker.Mock(
            outputs=[mocker.Mock(completion="test completion")],
            dict=mocker.Mock(return_value={"foo": "bar"}),
        ),
    )

    time_limit = 5
    state = base.State(
        task_string="test task",
        next_step={"module_type": "actor", "args": {}},
        token_limit=1000000,
        timeout=5,
        time_limit=time_limit,
        scores="",
    )
    agent = base.Agent(
        state=state,
        settings=base.Settings(
            actor=actor,
            discriminator="_basic",
            generator="_claude_legacy_1xc3.5s",
            prompter="_basic",
            toolkit="_basic_aird",
            autosubmit=False,
        ),
        toolkit_dict=tools._basic_aird,
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
    expected_content = score_result.json()
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
    messages = call_args.kwargs["messages"]
    assert len(messages) == 4
    assert {type(message) for message in messages} == {dict}

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
            ("system", templates.claude_basic_system_prompt),
            ("user", "Your current task is the following: test task"),
            ("assistant", "score<score></score>"),
            (
                "user",
                f"<score-output>{expected_score_message.content}</score-output>",
            ),
        ]
    ):
        message = messages[idx_message]
        assert message["role"] == expected_role
        assert message["content"] == expected_content

    if expected_output_file is not None:
        assert fs.exists(expected_output_file)


@pytest.mark.asyncio
async def test_score_log_fn(mocker: pytest_mock.MockerFixture):
    expected_output = [
        pyhooks.types.ScoreLogEntry(
            score=0.75,
            message={"foo": "bar"},
            scoredAt=datetime.datetime.now().isoformat(),
            elapsedSeconds=1.0,
        )
    ]
    mocker.patch("pyhooks.Hooks.scoreLog", autospec=True, return_value=expected_output)

    output = await tools.score_log_fn(None)

    assert isinstance(output, str)
    assert json.loads(output) == [x.dict() for x in expected_output]
