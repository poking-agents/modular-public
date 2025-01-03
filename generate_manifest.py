import json
from itertools import product

TOOLKITS = ["_basic", "_basic_vision", "_vision_double_return"]

PROMPTERS = ["_basic", "_context_and_usage_aware"]

GENERATORS = []
for model, n in product(
    ["c3o", "c3s", "c3h", "c3.5s", "c3.5sv2", "4", "4-0613", "4-1106", "4-0125"], [1, 2, 4, 8, 16, 32, 64]
):
    GENERATORS.append(f"_claude_legacy_{n}x{model}")
for gpt, n in product(
    ["4", "4t", "4o", "4om", "o1p", "o1m", "o1"], [1, 2, 4, 8, 16, 32, 64]
):
    GENERATORS.append(f"_gpt_basic_{n}x{gpt}")

DISCRIMINATORS = ["_basic"]
DISCRIMINATORS += [
    f"_compare_options_{model}"
    for model in [
        "4",
        "4t",
        "4o",
        "4om",
        "o1p",
        "o1m",
        "o1",
        "c3o",
        "c3s",
        "c3h",
        "c3.5s",
        "c3.5sv2",
    ]
]
DISCRIMINATORS += [
    f"_fixed_rating_{model}"
    for model in [
        "4",
        "4t",
        "4o",
        "4om",
        "o1p",
        "o1m",
        "o1",
        "c3o",
        "c3s",
        "c3h",
        "c3.5s",
        "c3.5sv2",
    ]
]
DISCRIMINATORS += [
    f"_compare_and_regenerate_{n_rounds}_rounds_gpt_{gpt}"
    for gpt, n_rounds in product(
        ["4", "4t", "4o", "4om", "o1p", "o1m", "o1"], range(1, 6)
    )
]
DISCRIMINATORS += [
    f"_assess_and_backtrack_gpt_{gpt}"
    for gpt in ["4", "4t", "4o", "4om", "o1p", "o1m", "o1"]
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
            "autosubmit": {"type": "boolean"},
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
                                        "type": {"type": "string"},
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
            "submissions": {
                "type": "array",
                "items": {"type": "string"},
            },
            "token_limit": {"type": "integer"},
            "token_usage": {"type": "integer"},
            "time_limit": {"type": "integer"},
            "time_usage": {"type": "integer"},
            "timeout": {"type": "integer"},
            "last_rating_options": {
                "oneOf": [
                    {
                        "type": "array",
                        "items": {"$ref": "#/$defs/ratingOption"},
                        "title": "Rating options",
                    },
                    {
                        "type": "null",
                        "title": "null",
                    },
                ],
                "default": "null",
            },
        },
        "additionalProperties": False,
        "required": ["task_string", "nodes", "last_node_id"],
        "$defs": {
            "ratingOption": {
                "type": "object",
                "title": "Rating option",
                "properties": {
                    "action": {"type": "string"},
                    "description": {
                        "type": ["string", "null"],
                        "default": None,
                    },
                    "fixedRating": {
                        "type": ["number", "null"],
                        "default": None,
                    },
                    "userId": {
                        "type": ["string", "null"],
                        "default": None,
                    },
                    "editOfOption": {
                        "type": ["integer", "null"],
                        "default": None,
                    },
                    "duplicates": {
                        "type": ["integer", "null"],
                        "default": "null",
                    },
                },
                "additionalProperties": False,
                "required": ["action"],
            },
        },
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
    manifest = {
        **MANIFEST,
        "settingsPacks": settings_packs,
        "defaultSettingsPack": "t_context_and_usage_awarep_gpt_1x4ogda",
    }
    with open("manifest.json", "w") as f:
        f.write(json.dumps(manifest, indent=4, sort_keys=True))


if __name__ == "__main__":
    generate_manifest()
