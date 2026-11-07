"""MockLLM that reports every call to the per-request budget meter (Task 15, Runtime layer)."""
from mock_llm import MockLLM
from runtime_guard import record_llm_call


def _flatten(messages) -> str:
    if isinstance(messages, str):
        return messages
    return "".join(str(m.get("content", "")) for m in messages)


class MeteredMockLLM(MockLLM):
    def call(self, messages, tools=None, callbacks=None, available_functions=None,
             from_task=None, from_agent=None, response_model=None):
        reply = super().call(messages, tools=tools, callbacks=callbacks, available_functions=available_functions,
                             from_task=from_task, from_agent=from_agent, response_model=response_model)
        record_llm_call(_flatten(messages), reply)
        return reply