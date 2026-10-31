"""CrewAI tool wrappers (Task 7). Every call is logged so the transcript can prove
which tool actually ran on which query."""
import json

from pydantic import BaseModel, Field
from crewai.tools import BaseTool

from lookup_tool import check_job_application_status

RAG_COLLECTION = "kb_fixed"      # chunking strategy recommended in Task 5
TOOL_CALLS = []                  # [{"tool": ..., "input": ...}], cleared per request by the crew runner


class RagInput(BaseModel):
    query: str = Field(..., description="The policy question to search the knowledge base for.")


class StatusInput(BaseModel):
    record_id: str = Field(..., description="Application record id, e.g. APP-0002.")


_collection_cache = {}


def _get_collection():
    if "col" not in _collection_cache:
        from rag_index import get_collections, COLLECTION_FIXED
        fixed, sent = get_collections()
        _collection_cache["col"] = fixed if RAG_COLLECTION == COLLECTION_FIXED else sent
    return _collection_cache["col"]


class RagTool(BaseTool):
    name: str = "rag_lookup"
    description: str = ("Search the Naukri.com hiring-policy knowledge base and return a grounded "
                        "answer with its source documents, or grounded=false if nothing relevant.")
    args_schema: type[BaseModel] = RagInput

    def _run(self, query: str) -> str:
        from grounded import grounded_answer
        TOOL_CALLS.append({"tool": self.name, "input": query})
        result = grounded_answer(query, _get_collection(), RAG_COLLECTION)
        return json.dumps({"answer": result["answer"], "grounded": result["grounded"],
                           "sources": result["sources"]})


class StatusTool(BaseTool):
    name: str = "check_job_application_status"
    description: str = ("Look up one job application by record_id and return its status, expected "
                        "salary in INR, and escalation score.")
    args_schema: type[BaseModel] = StatusInput

    def _run(self, record_id: str) -> str:
        TOOL_CALLS.append({"tool": self.name, "input": record_id})
        return json.dumps(check_job_application_status(record_id))