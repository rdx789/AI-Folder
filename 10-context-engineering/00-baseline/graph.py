"""Stage 00: one state field, one model node, every tool, full history.

Student reading order:
  1. AgentState        - what persists between turns
  2. call_model        - what the model receives
  3. should_continue   - when tools run
  4. graph wiring      - the complete loop

    START -> model[all tools + all messages] -> tools --+
                 | no tool calls                      |
                 +----------------> END <-------------+
"""

import sys
from pathlib import Path
from typing import Annotated, TypedDict

CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "00-agent-shared"))

from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

import tracing
from agent import ASSISTANT_PROMPT, get_model, invoke_answer
from evals.runner import main
from evals.tokens import METER


# 1. STATE -------------------------------------------------------------------

# The only persisted state is the complete message history.
class AgentState(TypedDict):
    """Everything persisted for one conversation thread.

    In this baseline, persisted state and model context are accidentally the same:
    every stored message is sent to every model call.
    """

    messages: Annotated[list, add_messages]


# 2. NODES AND ROUTING --------------------------------------------------------

def build_graph(tools: list):
    # The complete tool catalogue is bound once and remains visible on every turn.
    model = get_model().bind_tools(tools)
    all_tool_names = [tool.name for tool in tools]

    def call_model(state: AgentState) -> dict:
        # Baseline context = system prompt + every stored message.
        response = invoke_answer(
            model,
            [SystemMessage(ASSISTANT_PROMPT), *state["messages"]],
            meter=METER,
            exposed=all_tool_names,
            catalogue=all_tool_names,
            state=state,
        )
        return tracing.update("model", state, {"messages": [response]})

    def should_continue(state: AgentState) -> str:
        return "tools" if state["messages"][-1].tool_calls else END

    # 3. GRAPH WIRING ---------------------------------------------------------
    graph = StateGraph(AgentState)
    graph.add_node("model", call_model)
    graph.add_node("tools", tracing.tool_node(ToolNode(tools)))

    graph.add_edge(START, "model")
    graph.add_conditional_edges(
        "model",
        should_continue,
        {"tools": "tools", END: END},
    )
    graph.add_edge("tools", "model")

    # One thread_id maps to one persisted AgentState.
    return graph.compile(checkpointer=InMemorySaver())


if __name__ == "__main__":
    tracing.enable_from_argv()  # --trace: narrate every state change
    main(build_graph, "Stage 00 - full history and all tools on every turn.")
