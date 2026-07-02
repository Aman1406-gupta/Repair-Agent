from agent_builder.base.configs import LLMConfig, TaskConfig
from agent_builder.prebuilt_tasks.think_act import ThinkActTask

from repair_agent.prompts.failure_analysis import FAILURE_ANALYSIS_PROMPT
from repair_agent.prompts.repair_generation import REPAIR_GENERATION_PROMPT
from repair_agent.prompts.pull_request import PULL_REQUEST_PROMPT


LLM_CONFIGURATION = LLMConfig()


class TaskFactory:

    @staticmethod
    def create_failure_analysis_task(tools):
        return ThinkActTask(
            task_config=TaskConfig(
                name="failure_analysis",
                description="Analyze a failing unit test and identify the root cause.",
                system_template=FAILURE_ANALYSIS_PROMPT,
                llm_config=LLM_CONFIGURATION,
            ),
            tools=tools,
            handoffs=[],
            memory=None,
        )

    @staticmethod
    def create_repair_task(tools):
        return ThinkActTask(
            task_config=TaskConfig(
                name="repair_generation",
                description="Generate a repair patch for the failing code.",
                system_template=REPAIR_GENERATION_PROMPT,
                llm_config=LLM_CONFIGURATION,
            ),
            tools=tools,
            handoffs=[],
            memory=None,
        )

    @staticmethod
    def create_pull_request_task(tools):
        return ThinkActTask(
            task_config=TaskConfig(
                name="pull_request",
                description="Generate a GitHub Pull Request description.",
                system_template=PULL_REQUEST_PROMPT,
                llm_config=LLM_CONFIGURATION,
            ),
            tools=tools,
            handoffs=[],
            memory=None,
        )