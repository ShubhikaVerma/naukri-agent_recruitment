"""Task 8 transcripts: (A) state carried across turns, (B) a fresh conversation with state absent."""
from conversation import chat, get_session_history


def show_turn(n: int, out: dict, question: str):
    print("Turn {}  [session={}]".format(n, out["session_id"]))
    print("  User            : {}".format(question))
    if out["resolved_question"] != question:
        print("  Resolved to     : {}".format(out["resolved_question"]))
    print("  Tools invoked   : {}".format([(c["tool"], c["input"]) for c in out["tool_calls"]]))
    print("  Agent           : {}".format(out["answer"]))
    print("  Messages stored : {}\n".format(out["history_length"]))


def run(session_id: str, questions: list):
    for i, q in enumerate(questions, 1):
        show_turn(i, chat(session_id, q), q)


if __name__ == "__main__":
    print("#" * 78)
    print("# TRANSCRIPT A: multi-turn conversation, state carried across turns")
    print("#" * 78 + "\n")
    run("session-A", [
        "What is the status of APP-0002?",
        "Is it recommended for escalation, and what is its expected salary?",
        "What is the standard notice period when I join?",
        "What was my first question in this conversation?",
    ])

    print("#" * 78)
    print("# TRANSCRIPT B: separate FRESH conversation, state correctly absent")
    print("#" * 78 + "\n")
    run("session-B", [
        "What was my first question in this conversation?",
        "Is it recommended for escalation, and what is its expected salary?",
    ])

    print("Isolation check (sessions do not leak into each other):")
    for sid in ("session-A", "session-B"):
        print("  {} holds {} messages".format(sid, len(get_session_history(sid).messages)))