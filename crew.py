"""Task 7 (+ Task 15 governance): CrewAI crew with a Retrieval Agent, a Lookup Agent and a
Response Composer. Agents are built through make_agent(), which enforces least autonomy."""
import os

# Must be set BEFORE crewai is imported / kickoff() runs: keeps the run fully offline.
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_TRACING_ENABLED"] = "false"   # no trace-sharing prompt / upload

from crewai import Agent, Crew, Process, Task

from crew_tools import RagTool, StatusTool, TOOL_CALLS
from governance import check_wiring
from metered_llm import MeteredMockLLM
from runtime_guard import current_meter
from schemas import SupportResponse, validate_response


def build_llm():
    if os.environ.get("MOCK_LLM", "true").lower() == "true":
        return MeteredMockLLM(model="mock-llm")
    raise NotImplementedError("Optional: wire a real LLM here when MOCK_LLM=false.")


def make_agent(role: str, goal: str, backstory: str, tools: list, llm) -> Agent:
    """The ONLY way agents are created. Refuses to give an agent a tool it is not entitled to."""
    for tool in tools:
        check_wiring(tool.name, getattr(tool, "bound_role", ""), role)
    return Agent(role=role, goal=goal, backstory=backstory, tools=tools, llm=llm,
                 allow_delegation=False, verbose=False,
                 max_retry_limit=0)               # a budget abort must not be silently retried


def tool_wiring(crew: Crew) -> dict:
    """{agent role: [tool names]} for a built crew."""
    return {a.role: [t.name for t in (a.tools or [])] for a in crew.agents}


def build_crew(question: str) -> Crew:
    llm = build_llm()

    retrieval_agent = make_agent(
        "Retrieval Agent", "Find the hiring-policy passages that answer the user's question.",
        "You search the Naukri.com policy knowledge base and report only what it says.",
        [RagTool(bound_role="Retrieval Agent")], llm)

    lookup_agent = make_agent(
        "Lookup Agent", "Check the status of the job application the user asks about.",
        "You are the only agent allowed to read application records.",
        [StatusTool(bound_role="Lookup Agent")], llm)

    composer_agent = make_agent(
        "Response Composer", "Combine the policy findings and application status into one clear draft answer.",
        "You write concise support answers using only the findings you are given.",
        [], llm)

    retrieve = Task(
        description="Search the policy knowledge base for this question.\nUSER QUESTION: " + question,
        expected_output="One line starting with POLICY: containing the grounded policy answer.",
        agent=retrieval_agent)

    lookup = Task(
        description="Look up the application mentioned in this question, if any.\nUSER QUESTION: " + question,
        expected_output="One line starting with STATUS: describing the application.",
        agent=lookup_agent)

    compose = Task(
        description="Write the final draft answer for the user from the findings in your context.",
        expected_output="A JSON object matching the SupportResponse schema.",
        agent=composer_agent, context=[retrieve, lookup],
        output_pydantic=SupportResponse)

    return Crew(agents=[retrieval_agent, lookup_agent, composer_agent],
                tasks=[retrieve, lookup, compose], process=Process.sequential, verbose=False,
                tracing=False)


def run_crew(question: str) -> dict:
    """Run the crew and VALIDATE its raw output against the SupportResponse schema.
    Raises pydantic.ValidationError if the response does not conform, and BudgetExceeded
    if the per-request token budget is passed while the crew is running."""
    TOOL_CALLS.clear()
    try:
        result = build_crew(question).kickoff()
    except Exception as exc:
        meter = current_meter()
        if meter is not None and meter.exceeded is not None:
            raise meter.exceeded from exc          # whatever CrewAI wrapped it in, surface the budget error
        raise
    response = validate_response(str(result.raw))                  # schema check in code
    called = sorted({c["tool"] for c in TOOL_CALLS})
    if sorted(response.tools_used) != called:                      # consistency check
        raise ValueError("tools_used {} does not match the tools actually called {}".format(
            response.tools_used, called))
    return {"question": question, "draft": response.answer, "response": response.model_dump(),
            "tool_calls": list(TOOL_CALLS)}


if __name__ == "__main__":
    samples = [
        "How many days do I have to submit documents for background verification?",   # RAG only
        "What is the status of APP-0002?",                                            # lookup only
        "Can you check APP-0010 and tell me the notice period policy?",              # both tools
    ]
    for q in samples:
        out = run_crew(q)
        print("=" * 78)
        print("Q:", q)
        print("Tools invoked:", [(c["tool"], c["input"]) for c in out["tool_calls"]])
        print("Draft answer :", out["draft"])