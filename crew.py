"""Task 7: CrewAI crew with a Retrieval Agent, a Lookup Agent and a Response Composer."""
import os

# Must be set BEFORE crewai is imported / kickoff() runs: keeps the run fully offline.
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_TRACING_ENABLED"] = "false"   # no trace-sharing prompt / upload

from crewai import Agent, Crew, Process, Task

from mock_llm import MockLLM
from crew_tools import RagTool, StatusTool, TOOL_CALLS
from schemas import SupportResponse, validate_response


def build_llm():
    if os.environ.get("MOCK_LLM", "true").lower() == "true":
        return MockLLM(model="mock-llm")
    raise NotImplementedError("Optional: wire a real LLM here when MOCK_LLM=false.")


def build_crew(question: str) -> Crew:
    llm = build_llm()

    retrieval_agent = Agent(
        role="Retrieval Agent",
        goal="Find the hiring-policy passages that answer the user's question.",
        backstory="You search the Naukri.com policy knowledge base and report only what it says.",
        tools=[RagTool()], llm=llm, allow_delegation=False, verbose=False)

    lookup_agent = Agent(
        role="Lookup Agent",
        goal="Check the status of the job application the user asks about.",
        backstory="You are the only agent allowed to read application records.",
        tools=[StatusTool()], llm=llm, allow_delegation=False, verbose=False)

    composer_agent = Agent(
        role="Response Composer",
        goal="Combine the policy findings and application status into one clear draft answer.",
        backstory="You write concise support answers using only the findings you are given.",
        tools=[], llm=llm, allow_delegation=False, verbose=False)

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
    Raises pydantic.ValidationError if the response does not conform."""
    TOOL_CALLS.clear()
    result = build_crew(question).kickoff()
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
        "What is the status of APP-0002?",                                            # lookup (+ RAG finds nothing)
        "Can you check APP-0010 and tell me the notice period policy?",              # both tools
    ]
    for q in samples:
        out = run_crew(q)
        print("=" * 78)
        print("Q:", q)
        print("Tools invoked:", [(c["tool"], c["input"]) for c in out["tool_calls"]])
        print("Draft answer :", out["draft"])