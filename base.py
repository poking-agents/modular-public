from __future__ import annotations

import json
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field
from pyhooks import Actions, Hooks

hooks = Hooks()
actions = Actions()


class Message(BaseModel):
    role: Literal["assistant", "function", "system", "tool", "user"]
    content: str
    function_call: Optional[Dict] = None
    name: Optional[str] = None


class Node(BaseModel):
    node_id: int
    parent: int
    children: List[int]
    message: Message
    metadata: Dict = Field(default_factory=dict)

    def get_path(self, nodes: List[Node]) -> List[int]:
        path = [self.node_id]
        while path[-1] != 0:
            path.append(nodes[path[-1]].parent)
        return path[::-1]


def is_dict_exact_match(obj: dict, model: BaseModel) -> bool:
    model_fields = set(model.__fields__.keys())
    obj_keys = set(obj.keys())
    return obj_keys.issubset(model_fields) and all(
        field in obj_keys
        for field in model_fields
        if model.__fields__[field].default is None
    )


def convert_to_custom_type(obj):
    if isinstance(obj, dict):
        if is_dict_exact_match(obj, Node):
            return Node(**obj)
        elif is_dict_exact_match(obj, Message):
            return Message(**obj)
        else:
            return {k: convert_to_custom_type(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_custom_type(item) for item in obj]
    else:
        return obj


class State(BaseModel):
    task_string: str
    nodes: List[Node] = Field(default_factory=list)
    last_node_id: int = -1
    last_rating_options: Optional[List] = None
    next_step: Dict = Field(default_factory=dict)
    token_limit: int = 500000
    token_usage: int = 0
    timeout: int = 600
    submissions: List[str] = Field(default_factory=list)

    class Config:
        json_encoders = {Node: lambda v: v.dict(), Message: lambda v: v.dict()}

    @classmethod
    def parse_obj(cls, obj):
        if "next_step" in obj:
            obj["next_step"] = convert_to_custom_type(obj["next_step"])
        if "extra" in obj:
            obj["extra"] = convert_to_custom_type(obj["extra"])
        if "nodes" in obj:
            obj["nodes"] = convert_to_custom_type(obj["nodes"])
        return super().parse_obj(obj)

    def generate_node(
        self,
        message: Message,
        parent: Optional[int] = None,
        children: List[int] = [],
        metadata: Dict = {},
    ):
        if parent is None:
            parent = self.last_node_id
        self.last_node_id += 1
        new_node = Node(
            node_id=self.last_node_id,
            parent=parent,
            children=children,
            message=message,
            metadata=metadata,
        )
        if parent != -1:
            self.nodes[parent].children.append(self.last_node_id)
        self.nodes.append(new_node)
        return new_node

    def get_path(self, node_id: Optional[int] = None) -> List[int]:
        if not self.nodes:
            return []
        if node_id is None:
            node_id = self.last_node_id
        return self.nodes[node_id].get_path(self.nodes)


class Settings(BaseModel):
    toolkit: str
    prompter: str
    generator: str
    discriminator: str
    actor: str


okabe_ito = {
    "black": "#000000",
    "orange": "#E69F00",
    "light_blue": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "red": "#D55E00",
    "purple": "#CC79A7",
}


def lighten_color(hex_color, amount=0.8):
    hex_color = hex_color.strip("#")
    rgb = [int(hex_color[i : i + 2], 16) for i in (0, 2, 4)]
    new_rgb = [int((255 - val) * amount + val) for val in rgb]
    return "#" + "".join(f"{val:02x}" for val in new_rgb)


def style(step_kind: str | None) -> dict:
    if step_kind == "bash":
        # if action.observation.status is not None and action.observation.status != 0:
        #     color = okabe_ito["red"]
        # else:
        color = okabe_ito["orange"]
    elif step_kind == "python":
        color = okabe_ito["light_blue"]
    elif step_kind == "note":
        color = okabe_ito["green"]
    elif step_kind == "observation":
        color = okabe_ito["yellow"]
    else:
        color = "#ffffff"

    lightened = lighten_color(color, 0.8)
    return {
        "style": {
            "background-color": lightened,
            "border": f"2px solid {color}",
        }
    }


class Agent(BaseModel):
    state: State
    settings: Settings
    toolkit_dict: Dict

    def set_toolkit_dict(self: Agent, toolkit_dict: Dict):
        self.toolkit_dict = toolkit_dict

    def append(
        self,
        message: Message,
        parent: Optional[int] = None,
        children: List[int] = [],
        metadata: Dict = {},
    ):
        if (
            message.function_call is not None
            and "arguments" in message.function_call
            and isinstance(message.function_call["arguments"], (dict, list))
        ):
            # arguments must be stringified, otherwise OAI API will error out
            message.function_call["arguments"] = json.dumps(
                message.function_call["arguments"], indent=2, sort_keys=True
            )
        self.log(message)
        self.state.generate_node(message, parent, children, metadata)

    def log(self, message: Message):
        step_kind = None
        if message.role == "tool":
            hooks.log(f"output:\n```\n{message.content}\n```")
        elif message.role == "assistant":
            message_content = message.content
            if message.function_call is not None:
                step_kind = message.function_call["name"]
                message_content += f"\n\n{message.function_call['name']}:\n{message.function_call['arguments']}"
            hooks.log_with_attributes(style(step_kind), message_content)
        else:
            hooks.log_with_attributes(style("observation"), message.content)
