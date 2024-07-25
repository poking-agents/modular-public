from __future__ import annotations
from typing import List, Dict, Set
import json

import asyncio
from pydantic import BaseModel

from pyhooks import Hooks, Actions
from pyhooks.types import MiddlemanSettings

hooks = Hooks()
actions = Actions()


class Message(BaseModel):
    role: str
    content: str
    function_call: Dict = None
    name: str = None


class Node(BaseModel):
    node_id: int
    parent: int
    children: List[int]
    message: Message
    metadata: Dict = {}

    def get_path(self, nodes: List[Node]) -> List[int]:
        path = [self.node_id]
        while path[-1] != 0:
            path.append(nodes[path[-1]].parent)
        return path[::-1]


class State(BaseModel):
    task_string: str
    nodes: List[Node] = []
    last_node_id: int = -1
    next_step: Dict = {}
    token_limit: int = 500000
    token_usage: int = 0
    submissions: List[str] = []

    def generate_node(
        self,
        message: Message,
        parent: int = None,
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

    def get_path(self, node_id: int = None) -> List[int]:
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


class Agent(BaseModel):
    state: State
    settings: Settings
    toolkit_dict: Dict

    def set_toolkit_dict(self: Agent, toolkit_dict: Dict):
        self.toolkit_dict = toolkit_dict

    def append(
        self,
        message: Message,
        parent: int = None,
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
        if message.role == "tool":
            hooks.log(f"output:\n```\n{message.content}\n```")
        elif message.role == "assistant":
            message_content = message.content
            if message.function_call is not None:
                message_content += f"\n\n{message.function_call['name']}:\n{message.function_call['arguments']}"
            hooks.log(message_content)
        else:
            hooks.log(message.content)
