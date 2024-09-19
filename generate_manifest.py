import json
from itertools import product

TOOLKITS = ["_basic", "_basic_vision", "_basic_scoring", "_vision_double_return"]

PROMPTERS = ["_basic", "_context_and_usage_aware"]

GENERATORS = []
for model, n in product(["c3o", "c3s", "c3h", "c3.5s"], [1, 2, 4, 8, 16, 32, 64]):
    GENERATORS.append(f"_claude_legacy_{n}x{model}")
for gpt, n in product(["4", "4t", "4o", "4om"], [1, 2, 4, 8, 16, 32, 64]):
    GENERATORS.append(f"_gpt_basic_{n}x{gpt}")

DISCRIMINATORS = ["_basic"]
DISCRIMINATORS += [
    f"_compare_options_{model}"
    for model in ["4", "4t", "4o", "4om", "c3o", "c3s", "c3h", "c3.5s"]
]
DISCRIMINATORS += [
    f"_compare_and_regenerate_{n_rounds}_rounds_gpt_{gpt}"
    for gpt, n_rounds in product(["4", "4t", "4o", "4om"], range(1, 6))
]
DISCRIMINATORS += [
    f"_assess_and_backtrack_gpt_{gpt}" for gpt in ["4", "4t", "4o", "4om"]
]

ACTORS = ["_basic", "_prompt_to_search", "_always_save"]

MANIFEST = {
    "settingsSchema": {
        "type": "object",
        "properties": {
            "toolkit": {"type": "string"},
            "prompter": {"type": "string"},
            "generator": {"type": "string"},
            "discriminator": {"type": "string"},
            "actor": {"type": "string"},
        },
        "additionalProperties": False,
        "required": ["toolkit", "prompter", "generator", "discriminator", "actor"],
    },
    "stateSchema": {
        "type": "object",
        "properties": {
            "task_string": {"type": "string"},
            "nodes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "node_id": {"type": "integer"},
                        "parent": {"type": "integer"},
                        "children": {"type": "array", "items": {"type": "integer"}},
                        "message": {
                            "type": "object",
                            "properties": {
                                "name": {"type": ["string", "null"]},
                                "role": {"type": "string"},
                                "content": {"type": "string"},
                                "function_call": {
                                    "type": ["object", "null"],
                                    "properties": {
                                        "name": {"type": "string"},
                                        "arguments": {"type": "string"},
                                    },
                                    "additionalProperties": False,
                                    "required": ["name", "arguments"],
                                },
                            },
                            "additionalProperties": False,
                            "required": ["role", "content"],
                        },
                        "metadata": {"type": "object", "additionalProperties": True},
                    },
                    "additionalProperties": False,
                    "required": ["node_id", "parent", "children", "message"],
                },
            },
            "last_node_id": {"type": "integer"},
            "next_step": {
                "type": "object",
                "properties": {
                    "module_type": {"type": "string"},
                    "args": {"type": "object", "additionalProperties": True},
                },
                "additionalProperties": False,
                "required": ["module_type", "args"],
            },
            "token_limit": {"type": "integer"},
            "token_usage": {"type": "integer"},
            "timeout": {"type": "integer"},
        },
        "additionalProperties": False,
        "required": ["task_string", "nodes", "last_node_id"],
    },
}


def generate_manifest():
    settings_packs = {}
    for toolkit, prompter, generator, discriminator, actor in product(
        TOOLKITS, PROMPTERS, GENERATORS, DISCRIMINATORS, ACTORS
    ):
        settings_pack_name = "".join(
            [
                toolkit.replace("_basic", "") + "t",
                prompter.replace("_basic", "") + "p",
                generator.replace("_basic", "") + "g",
                discriminator.replace("_basic", "") + "d",
                actor.replace("_basic", "") + "a",
            ]
        )
        settings_packs[settings_pack_name] = {
            "toolkit": toolkit,
            "prompter": prompter,
            "generator": generator,
            "discriminator": discriminator,
            "actor": actor,
        }
    MANIFEST["settingsPacks"] = settings_packs
    MANIFEST["defaultSettingsPack"] = "tp_gpt_1x4ogda"
    with open("manifest.json", "w") as f:
        f.write(json.dumps(MANIFEST, indent=4, sort_keys=True))


if __name__ == "__main__":
    generate_manifest()
