"""Unit tests for platform sync ``llm_config`` merge and env-gated ``llmConfigurationId``."""

import pytest

from agent_builder.handlers import sync as sync_mod
from agent_builder.utils.constants import (
    ENV_SYNC_POPULATE_LLM_CONFIGURATION_ID,
    LLM_CONFIGURATION_ID,
)


def test_merged_llm_config_fills_defaults_when_no_raw_config():
    task: dict = {"name": "t", "description": "d", "taskPrompt": {"prompt": "p"}}
    m = sync_mod._merged_llm_config_dict_from_platform_task(task)
    assert "model" in m
    assert m.get(LLM_CONFIGURATION_ID) in (None, "")


def test_populate_flag_on_sets_platform_id(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(ENV_SYNC_POPULATE_LLM_CONFIGURATION_ID, "1")
    task: dict = {
        "llmConfigurationId": "platform-cfg-99",
        "llm_config": {"temperature": 0.2},
    }
    m = sync_mod._merged_llm_config_dict_from_platform_task(task)
    assert m[LLM_CONFIGURATION_ID] == "platform-cfg-99"
    assert m["temperature"] == 0.2


def test_populate_flag_off_does_not_set_platform_id(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(ENV_SYNC_POPULATE_LLM_CONFIGURATION_ID, raising=False)
    task: dict = {"llmConfigurationId": "platform-cfg-99", "llm_config": {}}
    m = sync_mod._merged_llm_config_dict_from_platform_task(task)
    assert m.get(LLM_CONFIGURATION_ID) in (None, "")
