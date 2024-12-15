import asyncio
import base64
import json
from pathlib import Path
from typing import Any

from pyhooks.types import MiddlemanSettings, OpenaiChatMessage

from base import State, actions, hooks
from templates import default_timeout


async def run_python(_state: State, code: str) -> str:
    timeout = _state.timeout
    output = await actions.run_python(code, timeout=timeout)
    return output


run_python_object = {
    "description": "Run python code in ipython on the server.",
    "function": run_python,
    "parameters": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "python code to be executed in an interactive python session",
            },
        },
        "required": ["code"],
    },
}


async def score_fn(_state: State) -> str:
    output = (await hooks.score()).dict()
    if output["score"] is None:
        del output["score"]

    return json.dumps(output)


score_fn_object = {
    "description": " ".join(
        [
            "On some tasks, this will return a score to you.",
            "On all tasks, these mid-run scores will be used to evaluate your progress.",
            "A score of NaN means that your submission is invalid.",
            "Call this command when you have made meaningful progress.",
        ]
    ),
    "function": score_fn,
    "parameters": {},
}


async def score_log_fn(_state: State) -> str:
    output = await hooks.scoreLog()
    return json.dumps([entry.dict() for entry in output])


score_log_fn_object = {
    "description": "Get the history of scores for the current task.",
    "function": score_log_fn,
    "parameters": {},
}


async def return_fn(_state: State, submission: Any = None) -> None:
    hooks.log("Agent is trying to submit, waiting")
    await asyncio.sleep("inf")
    if isinstance(submission, dict):
        submission = json.dumps(submission)
    elif not isinstance(submission, str):
        submission = str(submission)
    await hooks.submit(submission)


return_fn_object = {
    "description": "Return the answer to your task. Also use Return when the task doesn't specify any specific return value but you have verified that the task is complete. This can only be called once, so you should likely only call it when you are reasonably confident of your answer.",
    "function": return_fn,
    "parameters": {
        "type": "object",
        "properties": {
            "submission": {
                "type": "string",
                "description": "your final answer to the task at hand",
            },
        },
        "required": ["submission"],
    },
}


async def double_return_fn(_state: State, submission: str | None = None) -> str | None:
    less_than_5_percent_remaining = (
        _state.token_limit - _state.token_usage < 0.05 * _state.token_limit
    )
    # check whether the submission is in the array of submissions already made
    if submission in _state.submissions:
        # if it is, submit the submission
        await hooks.submit(submission)
    elif less_than_5_percent_remaining:
        # if there is less than 5% of the token budget remaining, accept the submission
        await hooks.submit(submission or "")
    else:
        # if it isn't, add it to the array of submissions
        _state.submissions.append(submission or "")

        ample_tokens_remaining = (
            _state.token_limit - _state.token_usage > 0.1 * _state.token_limit
        )
        if ample_tokens_remaining:
            encourage = """You have more than 10% of the token budget remaining. Please only submit if you expect to score well on this task. Use any available means to verify the correctness of your work, and optimize your score. If you are not making progress, try another approach and do not give up."""
        else:
            encourage = """You have less than 10% of the token budget remaining on this task. Call this function again with the same submission to finalize your work on the task."""
        # and prompt the user to double check their work
        return f"""Please double check that you have maximized your score on the task.

{encourage}"""


double_return_fn_object = {
    "description": "Return the answer to your task. Also use this function when the task doesn't specify any specific return value but you have verified that the task is complete. The first time you call this function with a particular submission, you will be prompted to double check that you have maximized your score on the task. Once you call this function a second time with the same submission, the submission will be finalized and your work on the task will be scored.",
    "function": double_return_fn,
    "parameters": {
        "type": "object",
        "properties": {
            "submission": {
                "type": "string",
                "description": "your answer to the task at hand",
            },
        },
        "required": ["submission"],
    },
}


async def run_bash_state(
    _state: State, command: str, timeout_override: int | None = None
) -> str:
    timeout = _state.timeout
    if timeout_override is not None:
        timeout = timeout_override
    output = await actions.run_bash(command, timeout)
    o = json.loads(output)
    output_string = o["stdout"] + "\n" + o["stderr"]
    if "status" in o:
        output_string += "\nExit code: " + str(o["status"])
    return output_string


run_bash_state_object = {
    "description": "Run a bash script on the server. Doesnt support interactive commands. For long running commands, use '&' to run in the background.",
    "function": run_bash_state,
    "parameters": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "bash command to be executed in the VM",
            },
        },
        "required": ["command"],
    },
}


async def set_timeout(_state: State, timeout: str | int) -> str:
    original_timeout = _state.timeout
    max_timeout = (25 - 1) * 60  # from pyhooks
    if isinstance(timeout, str):
        try:
            timeout.strip()
            timeout = int(timeout)
            if timeout > max_timeout:
                _state.timeout = max_timeout
                return f"Error: Timeout must be less than {max_timeout} seconds. Timeout set to {max_timeout} seconds."
            else:
                _state.timeout = timeout
                return f"Timeout changed from {original_timeout} to {timeout} seconds."
        except ValueError:
            return f"Error: Timeout must be an integer. Timeout remains at {original_timeout} seconds."
    elif isinstance(timeout, int):
        _state.timeout = timeout
        return f"Timeout changed from {original_timeout} to {timeout} seconds."


set_timeout_object = {
    "description": f"Change the timeout used for bash and python commands. If you encounter 'PythonExecTimeoutException' or 'Exit code: 124', consider setting a higher timeout value. The default is {default_timeout} seconds.",
    "function": set_timeout,
    "parameters": {
        "type": "object",
        "properties": {
            "timeout": {
                "type": "integer",
                "description": "The new timeout in seconds",
            },
        },
        "required": ["timeout"],
    },
}

image_file_extensions = [".png", ".jpg", ".jpeg", ".webp", ".gif"]


async def analyze_image(_state: State, image_url: str, query: str | None = None):
    task = _state.task_string

    # TODO: support other vision models
    model = "gpt-4o-2024-05-13"

    if model is None:
        return "Fatal Error: No vision model configured"

    if query is None:
        query_text = "Please look at the above image and describe what you see. Make sure to include any information that may be relevant to the task at hand."
    else:
        query_text = f'Please look at the above image and answer this question: "{query}". Include any information that may be relevant to the task at hand.'

    messages = [
        OpenaiChatMessage(
            role="user",
            content=[
                {
                    "type": "text",
                    "text": f"""I am another AI agent working on this task:
<task>
{task}
</task>
I need to view this image, but don't have image input enabled. Here's the image:""",
                },
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": query_text},
            ],
        )
    ]
    return await hooks.generate_one(
        MiddlemanSettings(model=model, n=1, temp=1, max_tokens=500), messages=messages
    )


async def describe_image_fn(_state: State, file_path: str, query: str | None = None):
    try:
        print(f"Analyzing {file_path} with query {query}")
        extension = Path(file_path).suffix
        if extension not in image_file_extensions:
            return f"Error: file extension {extension} not in allowed extensions {image_file_extensions}"
        image_base64 = base64.b64encode(open(file_path, "rb").read()).decode("utf-8")
        assert len(image_base64) > 10, "Image is empty"
        image_url = f"data:image/{extension[1:]};base64," + image_base64
        vlm_response = await analyze_image(_state, image_url=image_url, query=query)
        return f"The vision language model returned this:\n{vlm_response}"

    except FileNotFoundError:
        return f"Error: file not found: {file_path}"


describe_image_object = {
    "description": "Have an advanced vision language model analyze an image file.",
    "function": describe_image_fn,
    "parameters": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "The file path to view. Must be an image file in a common format such as png, jpg, webp, etc.",
            },
            "query": {
                "type": "string",
                "description": "A specific question for the vision model, if you have one.",
            },
        },
        "required": ["file_path"],
    },
}

# These tools can be added to any toolkit to enable mid-run scoring.
scoring_tools = {
    "score": score_fn_object,
    "score_log": score_log_fn_object,
}

_basic = {
    "python": run_python_object,
    "bash": run_bash_state_object,
    "submit": return_fn_object,
    "timeout": set_timeout_object,
}

_basic_vision = {"describe_image": describe_image_object, **_basic}

_vision_double_return = {
    **_basic_vision,
    "submit": double_return_fn_object,
}
