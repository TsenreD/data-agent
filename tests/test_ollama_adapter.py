from models.ollama_adapter import OllamaAdapter


def test_parse_structured_content_extracts_json_after_reasoning() -> None:
    adapter = OllamaAdapter(model="demo-model", base_url="http://localhost:11434")
    schema = {
        "type": "object",
        "properties": {
            "filter_condition": {"type": ["string", "null"]},
            "row_indices": {"type": "array", "items": {"type": "integer"}},
        },
        "required": ["filter_condition", "row_indices"],
        "additionalProperties": False,
    }

    content = """
    <think>I will inspect the rows first.</think>
    {"filter_condition": "label.isna()", "row_indices": []}
    """.strip()

    parsed = adapter._parse_structured_content(content, schema)

    assert parsed == {"filter_condition": "label.isna()", "row_indices": []}


def test_build_request_includes_max_tokens_for_openai_compatible() -> None:
    adapter = OllamaAdapter(
        model="demo-model",
        base_url="http://localhost:8000/v1/chat/completions",
        max_tokens=8192,
    )

    _, payload = adapter._build_request([{"role": "user", "content": "hi"}], None)

    assert payload["max_tokens"] == 8192


def test_build_request_includes_num_predict_for_ollama_api() -> None:
    adapter = OllamaAdapter(
        model="demo-model",
        base_url="http://localhost:11434",
        max_tokens=8192,
    )

    _, payload = adapter._build_request([{"role": "user", "content": "hi"}], None)

    assert payload["options"]["num_predict"] == 8192
