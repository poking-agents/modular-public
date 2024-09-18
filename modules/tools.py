from pathlib import Path
from base import hooks, State, actions
import base64
import json

from pyhooks.types import (
    MiddlemanSettings,
    OpenaiChatMessage,
)


async def run_python(_state: State, code: str = None, timeout: int = 600) -> str:
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


async def return_fn(_state: State, submission: str = None) -> None:
    await hooks.submit(submission)


return_fn_object = {
    "description": "Return the answer to your task. Also use Return when the task doesnt specify any specific return value but you have verified that the task is complete. This can only be called once, so you should likely only call it when you are reasonably confident of your answer.",
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


async def double_return_fn(_state: State, submission: str = None) -> None | str:
    less_than_5_percent_remaining = (
        _state.token_limit - _state.token_usage < 0.05 * _state.token_limit
    )
    # check whether the submission is in the array of submissions already made
    if submission in _state.submissions:
        # if it is, submit the submission
        await hooks.submit(submission)
    elif less_than_5_percent_remaining:
        # if there is less than 5% of the token budget remaining, accept the submission
        await hooks.submit(submission)
    else:
        # if it isn't, add it to the array of submissions
        _state.submissions.append(submission)

        ample_tokens_remaining = (
            _state.token_limit - _state.token_usage > 0.1 * _state.token_limit
        )
        if ample_tokens_remaining:
            encourage = f"""You have more than 10% of the token budget remaining. Please only submit if you expect to score well on this task. Use any available means to verify the correctness of your work, and optimize your score. If you are not making progress, try another approach and do not give up."""
        else:
            encourage = f"""You have less than 10% of the token budget remaining on this task. Call this function again with the same submission to finalize your work on the task."""
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


async def run_bash_state(_state: State, command: str = None, timeout: int = 600) -> str:
    output = await actions.run_bash(command, timeout)
    o = json.loads(output)
    output_string = o["stdout"] + "\n" + o["stderr"]
    if "status" in o:
        output_string += "\nExit code: " + str(o["status"])
    return output_string


run_bash_state_object = {
    "description": "Run a bash script on the server. Doesnt support interactive commands.",
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

image_file_extensions = [".png", ".jpg", ".jpeg", ".webp", ".gif"]


async def analyze_image(_state: State, image_url: str, query: str):
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


async def describe_image_fn(_state: State, file_path: str = None, query: str = None):
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

_basic = {
    "python": run_python_object,
    "bash": run_bash_state_object,
    "submit": return_fn_object,
}

_basic_vision = {"describeImage": describe_image_object, **_basic}

_vision_double_return = {
    "describeImage": describe_image_object,
    "bash": run_bash_state_object,
    "python": run_python_object,
    "submit": double_return_fn_object,
}