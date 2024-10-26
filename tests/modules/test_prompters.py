import pytest

import base
import modules.prompters as prompters


@pytest.mark.parametrize(
    ("content", "metadata", "expect_trim", "expect_full_output_instructions"),
    [
        ("Hello world!", {}, False, False),
        ("Hello world!", {"saved_output_filename": "test.txt"}, False, False),
        ("Hello world!" * 10000, {}, True, False),
        ("Hello world!" * 10000, {"saved_output_filename": "test.txt"}, True, True),
    ],
)
def test_get_trimmed_message(
    content: str,
    metadata: dict[str, str],
    expect_trim: bool,
    expect_full_output_instructions: bool,
):
    node = base.Node(
        node_id=0,
        parent=-1,
        children=[],
        message=base.Message(role="function", content=content, name="test"),
        metadata=metadata,
    )

    trimmed_message = prompters._get_trimmed_message(node, 0.5)

    assert isinstance(trimmed_message, base.Message)
    if not expect_trim:
        assert trimmed_message.content == node.message.content
    else:
        assert len(trimmed_message.content) < len(node.message.content)

    if not expect_full_output_instructions:
        assert "The full output is saved as" not in trimmed_message.content
    else:
        assert (
            f"The full output is saved as {metadata['saved_output_filename']}."
            in trimmed_message.content
        )
