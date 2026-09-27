"""
agent_assignment.py
===================
Secure Executive Assistant  |  LangGraph + LangChain + RAG + HITL
------------------------------------------------------------------

Architecture overview
---------------------
                        +-----------+
              START --> |  agent    | <---+
                        |  (reason) |     |
                        +-----+-----+     |
                              |           |
                   tool call? |           | (loop back after tools)
                        +-----v-----+     |
                        | tool_node |-----+
                        +-----------+
                              |
                   done / proposed_action ready
                              |
                   [BREAKPOINT - human review]
                              |
                        +-----v-----------+
                        | execute_action  |
                        |     _node       |
                        +-----------------+
                              |
                             END

Key concepts demonstrated
-------------------------
* LangGraph StateGraph with a cyclic agent-tool loop
* MemorySaver checkpointer for persistent state across resumptions
* interrupt_before=["execute_action_node"] for Human-in-the-Loop (HITL)
* @tool decorators (RAG/search, time, weather)
* System-level guardrails injected via SystemMessage
* TypedDict state schema shared across all nodes
"""

from __future__ import annotations

import sys

import json
from datetime import datetime
from typing import Annotated, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

# Local utility – routes to Gemini or Groq automatically
from utils import get_chat_model


# ===========================================================================
# 1.  STATE DEFINITION
# ===========================================================================

class AgentState(TypedDict):
    """
    Shared state that flows through every node in the graph.

    Fields
    ------
    messages : list[BaseMessage]
        Full conversation history.  ``add_messages`` is a LangGraph
        reducer that *appends* new messages instead of replacing the list.
    is_approved : bool | None
        Set to True/False by the human reviewer at the HITL breakpoint.
        None means no decision has been made yet.
    proposed_action : str
        A plain-English description of what the agent wants to do.
        Populated by the agent node; executed (simulated) in execute_action_node.
    """

    messages: Annotated[list, add_messages]
    is_approved: bool | None
    proposed_action: str


# ===========================================================================
# 2.  GUARDRAIL SYSTEM PROMPT
# ===========================================================================

SYSTEM_PROMPT = SystemMessage(content="""
You are a "Secure Executive Assistant" – a highly capable but strictly governed AI agent.

CORE RULES (never violate these):
──────────────────────────────────
1. NEVER finalize or execute any high-stakes action (booking flights, sending emails,
   transferring money, deleting data, etc.) on your own. You MUST always propose the
   action and wait for explicit human approval before anything is committed.

2. TRAVEL & OUTDOOR QUERIES: Whenever the user asks about travel, outdoor activities,
   or anything location-dependent, you MUST call get_current_time() AND
   get_local_weather(location) before forming your response.

3. DANGEROUS / UNETHICAL / OUT-OF-SCOPE requests: If a user asks you to do something
   harmful, illegal, unethical, or clearly outside your scope (personal assistant tasks
   unrelated to research, scheduling, or information), you MUST politely refuse and
   must NOT invoke any tools.

4. PROPOSED ACTION FORMAT: When you have completed your research and are ready to
   suggest a high-stakes action, end your final message with a clearly delimited block:

   ---PROPOSED ACTION---
   <one or two sentence plain-English description of the action you propose>
   ---END PROPOSED ACTION---

   This block is parsed by the system to populate the `proposed_action` state field.

5. TRANSPARENCY: Always tell the user what tools you used and why.
""")


# ===========================================================================
# 3.  TOOL DEFINITIONS
# ===========================================================================

@tool
def get_current_time() -> str:
    """
    Returns the current date and time in a human-readable format.

    Use this tool whenever:
    - The user asks about time-sensitive topics.
    - You need to calculate relative dates (e.g., "tomorrow", "next week").
    - The user is planning travel or outdoor activities.

    Returns
    -------
    str
        Current local datetime string, e.g. "Sunday, 27 September 2026 – 10:20 AM".
    """
    now = datetime.now()
    return now.strftime("%A, %d %B %Y – %I:%M %p")


@tool
def get_local_weather(location: str) -> str:
    """
    Returns a mock weather report for the given city/location.

    In a production system this would call a real weather API (e.g., OpenWeatherMap).
    For this assignment it returns realistic-sounding simulated data.

    Use this tool whenever:
    - The user asks about travel to a specific city.
    - The user plans outdoor activities.
    - Weather conditions are relevant to a decision.

    Parameters
    ----------
    location : str
        Name of the city or region (e.g., "Paris", "New York", "Tokyo").

    Returns
    -------
    str
        A JSON-formatted weather summary string.
    """
    # Simulated weather database for common cities
    mock_weather_db: dict[str, dict] = {
        "paris": {
            "city": "Paris, France",
            "temperature_c": 18,
            "condition": "Partly Cloudy",
            "humidity_pct": 65,
            "wind_kph": 14,
            "forecast_tomorrow": "Scattered showers expected, high of 16°C",
        },
        "london": {
            "city": "London, UK",
            "temperature_c": 14,
            "condition": "Overcast",
            "humidity_pct": 80,
            "wind_kph": 22,
            "forecast_tomorrow": "Light rain in the morning, clearing by afternoon",
        },
        "new york": {
            "city": "New York, USA",
            "temperature_c": 22,
            "condition": "Sunny",
            "humidity_pct": 50,
            "wind_kph": 10,
            "forecast_tomorrow": "Clear skies, pleasant high of 24°C",
        },
        "tokyo": {
            "city": "Tokyo, Japan",
            "temperature_c": 25,
            "condition": "Humid & Sunny",
            "humidity_pct": 75,
            "wind_kph": 8,
            "forecast_tomorrow": "Mostly sunny with isolated afternoon showers",
        },
        "dubai": {
            "city": "Dubai, UAE",
            "temperature_c": 38,
            "condition": "Hot & Sunny",
            "humidity_pct": 40,
            "wind_kph": 18,
            "forecast_tomorrow": "Intense heat, UV index very high – stay hydrated",
        },
    }

    key = location.lower().strip()
    weather = mock_weather_db.get(
        key,
        {   # Default fallback for unlisted cities
            "city": location.title(),
            "temperature_c": 20,
            "condition": "Mild & Partly Cloudy",
            "humidity_pct": 60,
            "wind_kph": 12,
            "forecast_tomorrow": "Conditions expected to remain mild",
        },
    )
    return json.dumps(weather, indent=2)


@tool
def search_wikipedia(query: str) -> str:
    """
    Simulates a RAG (Retrieval-Augmented Generation) search against Wikipedia
    or an internal knowledge base.

    In a production system this would:
      1. Embed the query using an embedding model.
      2. Perform vector similarity search against a Chroma / Pinecone / FAISS store.
      3. Return the top-k document chunks as grounded context.

    For this assignment we return carefully crafted, factually accurate summaries
    that mimic what a real retrieval step would surface.

    Parameters
    ----------
    query : str
        The topic or question to look up (e.g., "Eiffel Tower history").

    Returns
    -------
    str
        A plain-text summary of the most relevant retrieved content.
    """
    # Simulated knowledge base — representative Wikipedia-style entries
    knowledge_base: dict[str, str] = {
        "eiffel tower": (
            "The Eiffel Tower (French: Tour Eiffel) is a wrought-iron lattice tower "
            "located on the Champ de Mars in Paris, France. Built between 1887 and 1889 "
            "as the entrance arch for the 1889 World's Fair, it was designed by engineer "
            "Gustave Eiffel. Standing 330 metres (1,083 ft) tall, it was the world's "
            "tallest man-made structure for 41 years. Today it attracts around 7 million "
            "visitors annually, making it the most-visited paid monument in the world. "
            "Entry tickets range from €11.80 (stairs) to €28.30 (summit by lift). "
            "Opening hours: 09:30–23:45 daily (last admission 45 min before closing)."
        ),
        "paris": (
            "Paris is the capital and most populous city of France. Situated on the "
            "Seine River, it is a global centre for art, fashion, gastronomy, and culture. "
            "The city is home to landmarks including the Louvre Museum, Notre-Dame Cathedral, "
            "Arc de Triomphe, and Palace of Versailles (nearby). Charles de Gaulle Airport "
            "(CDG) is the main international gateway, located ~25 km northeast of the city centre. "
            "The RER B train connects CDG to central Paris in approximately 35 minutes."
        ),
        "langgraph": (
            "LangGraph is an open-source library by LangChain, Inc. that enables developers "
            "to build stateful, multi-actor applications with LLMs using a graph-based "
            "execution model. It supports cyclic graphs, persistence (checkpointing), "
            "streaming, and human-in-the-loop patterns. Core primitives include StateGraph, "
            "nodes (Python functions), edges (routing logic), and reducers (state merging). "
            "LangGraph is particularly suited for agent architectures requiring branching, "
            "loops, and long-running workflows."
        ),
        "retrieval augmented generation": (
            "Retrieval-Augmented Generation (RAG) is an AI framework that enhances large "
            "language model responses by retrieving relevant documents from an external "
            "knowledge base before generating an answer. This reduces hallucination and "
            "keeps the model's knowledge up-to-date without expensive retraining. A typical "
            "RAG pipeline: (1) embed query, (2) vector similarity search, (3) retrieve top-k "
            "chunks, (4) inject chunks into LLM prompt as context, (5) generate grounded answer."
        ),
    }

    # Fuzzy keyword matching across the knowledge base
    query_lower = query.lower()
    for key, content in knowledge_base.items():
        if key in query_lower or any(word in query_lower for word in key.split()):
            return (
                f"[RAG RESULT – retrieved for query: '{query}']\n\n"
                f"{content}\n\n"
                "[Source: Simulated Wikipedia / Knowledge Base]"
            )

    # Generic fallback when no specific article matches
    return (
        f"[RAG RESULT – retrieved for query: '{query}']\n\n"
        f"No highly specific article found for '{query}'. "
        f"General summary: This topic relates to {query}. "
        f"For detailed information, a live Wikipedia API call or a populated "
        f"vector store would surface more precise content.\n\n"
        "[Source: Simulated Wikipedia / Knowledge Base]"
    )


# Bundle all tools for easy reference
TOOLS: list = [get_current_time, get_local_weather, search_wikipedia]


# ===========================================================================
# 4.  GRAPH NODE DEFINITIONS
# ===========================================================================

def build_agent_node(llm_with_tools):
    """
    Closure that captures the tool-bound LLM and returns the agent node function.

    The agent node is the 'brain' of the graph:
      - Prepends the guardrail SystemMessage on the first call.
      - Invokes the LLM with the full message history.
      - Parses any ---PROPOSED ACTION--- block from the LLM response.
      - Returns state updates: new messages + proposed_action (if any).

    Parameters
    ----------
    llm_with_tools : BaseChatModel
        An LLM instance that has been bound to TOOLS via .bind_tools().

    Returns
    -------
    Callable[[AgentState], dict]
        A LangGraph-compatible node function.
    """

    def agent_node(state: AgentState) -> dict:
        """
        Core reasoning node.  Invokes the LLM and extracts any proposed action.

        LangGraph calls this function with the current state dict.
        The return dict is *merged* into the state (reducer handles messages).
        """
        messages = state["messages"]

        # Inject system prompt as the very first message if not already present
        # This ensures guardrails are always in context, even after resumption.
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SYSTEM_PROMPT] + list(messages)

        print("\n[agent_node] Invoking LLM...")
        response: AIMessage = llm_with_tools.invoke(messages)

        # --- Parse proposed action from the LLM response ---
        # The system prompt instructs the LLM to wrap high-stakes actions in a
        # clearly delimited block so we can extract it reliably.
        proposed_action: str = state.get("proposed_action", "")
        raw_content: str = response.content if isinstance(response.content, str) else ""

        marker_start = "---PROPOSED ACTION---"
        marker_end = "---END PROPOSED ACTION---"

        if marker_start in raw_content and marker_end in raw_content:
            start_idx = raw_content.index(marker_start) + len(marker_start)
            end_idx = raw_content.index(marker_end)
            proposed_action = raw_content[start_idx:end_idx].strip()
            print(f"\n[agent_node] Proposed action detected:\n  >> {proposed_action}")

        return {
            "messages": [response],
            "proposed_action": proposed_action,
        }

    return agent_node


def execute_action_node(state: AgentState) -> dict:
    """
    Execution node – runs AFTER human approval at the HITL breakpoint.

    In a real system this node would:
      - Call a booking API, send an email, execute a trade, etc.

    Here we simulate execution with print statements and return a confirmation
    message appended to the conversation.

    The graph reaches this node ONLY when the human has typed 'yes' at the
    approval prompt, because the router checks `state["is_approved"]`.

    Parameters
    ----------
    state : AgentState
        Current graph state, including is_approved and proposed_action.

    Returns
    -------
    dict
        State update with a confirmation AIMessage.
    """
    action = state.get("proposed_action", "No action specified.")
    approved = state.get("is_approved", False)

    print("\n" + "=" * 60)
    if approved:
        print("  EXECUTING APPROVED ACTION")
        print("=" * 60)
        print(f"  Action  : {action}")
        print(f"  Status  : SIMULATED – In production this would call")
        print(f"            the relevant API / service.")
        print("=" * 60)

        confirmation = (
            f"Action successfully executed (simulated):\n\n"
            f"**{action}**\n\n"
            f"In a production deployment this node would have called the "
            f"relevant API or service to carry out the action. "
            f"All steps have been logged for audit purposes."
        )
    else:
        # The human said 'no' – we still reach this node but we abort.
        print("  ACTION REJECTED BY HUMAN – ABORTING")
        print("=" * 60)
        confirmation = (
            "The proposed action was rejected by the human reviewer. "
            "No changes have been made. Please provide updated instructions "
            "if you would like me to try a different approach."
        )

    return {"messages": [AIMessage(content=confirmation)]}


# ===========================================================================
# 5.  ROUTING LOGIC (CONDITIONAL EDGES)
# ===========================================================================

def route_after_agent(state: AgentState) -> Literal["tool_node", "execute_action_node", END]:
    """
    Conditional edge function called after the agent node.

    Determines the next node based on the last message:

    1. If the LLM responded with tool_calls -> route to tool_node
       (standard ReAct loop: Reason -> Act -> Observe -> Reason …)

    2. If a proposed_action has been set (LLM is done researching) ->
       route to execute_action_node
       (this is where the HITL breakpoint fires, pausing execution)

    3. Otherwise -> END the graph.

    Parameters
    ----------
    state : AgentState

    Returns
    -------
    Literal["tool_node", "execute_action_node", END]
    """
    last_message = state["messages"][-1]

    # Case 1 – LLM wants to call one or more tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        print(
            f"[router] Tool calls detected: "
            f"{[tc['name'] for tc in last_message.tool_calls]} -> tool_node"
        )
        return "tool_node"

    # Case 2 – LLM has proposed a high-stakes action (guardrail block found)
    if state.get("proposed_action"):
        print("[router] Proposed action present -> execute_action_node (HITL breakpoint)")
        return "execute_action_node"

    # Case 3 – Normal completion (e.g., informational answer, refusal)
    print("[router] No tool calls, no proposed action -> END")
    return END


# ===========================================================================
# 6.  GRAPH CONSTRUCTION
# ===========================================================================

def build_graph():
    """
    Assembles and compiles the LangGraph StateGraph.

    Graph topology
    --------------
    START
      |
      v
    agent_node  <----+
      |               |
      | tool calls?   |
      v               |
    tool_node  -------+  (loops back to agent for observation)
      |
      | proposed_action set?
      v
    execute_action_node   <-- HITL interrupt_before fires HERE
      |
      v
     END

    The MemorySaver checkpointer persists the full state to an in-memory
    store keyed by thread_id.  When the graph is resumed after the HITL
    pause, LangGraph restores the exact state from the checkpoint and
    continues from the next node.

    Returns
    -------
    CompiledStateGraph
        A compiled, runnable LangGraph graph with HITL checkpointing.
    """
    # -- Initialise LLM and bind tools ---------------------------------------
    llm = get_chat_model(temperature=0.0)
    llm_with_tools = llm.bind_tools(TOOLS)

    # -- Build the graph structure -------------------------------------------
    graph_builder = StateGraph(AgentState)

    # Add nodes
    graph_builder.add_node("agent_node", build_agent_node(llm_with_tools))
    graph_builder.add_node("tool_node", ToolNode(TOOLS))        # prebuilt ToolNode handles tool dispatch
    graph_builder.add_node("execute_action_node", execute_action_node)

    # Add edges
    graph_builder.add_edge(START, "agent_node")                 # Entry point

    # Conditional routing after the agent node (cyclic ReAct loop)
    graph_builder.add_conditional_edges(
        "agent_node",
        route_after_agent,
        {
            "tool_node": "tool_node",
            "execute_action_node": "execute_action_node",
            END: END,
        },
    )

    # After tool execution, always loop back to the agent for observation
    graph_builder.add_edge("tool_node", "agent_node")

    # After execution (approved or rejected), we are done
    graph_builder.add_edge("execute_action_node", END)

    # -- Attach checkpointer with HITL breakpoint ----------------------------
    # MemorySaver stores state in RAM; swap for SqliteSaver/PostgresSaver in prod.
    checkpointer = MemorySaver()

    # interrupt_before=["execute_action_node"] tells LangGraph to PAUSE the
    # graph and return control to the caller BEFORE running execute_action_node.
    # This is the core HITL mechanism – the human can inspect & approve/reject.
    compiled_graph = graph_builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["execute_action_node"],
    )

    return compiled_graph


# ===========================================================================
# 7.  TERMINAL UI HELPERS
# ===========================================================================

DIVIDER = "-" * 65


def extract_text(content) -> str:
    """
    Safely extract plain text from an AIMessage content field.

    Newer Gemini models (3.x) return content as a list of dicts:
      [{'type': 'text', 'text': '...'}, ...]
    Older models return a plain string.
    This helper handles both formats.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)


def print_banner():
    """Prints a styled ASCII banner for the terminal UI."""
    print("\n" + "=" * 65)
    print("   SECURE EXECUTIVE ASSISTANT  |  LangGraph + HITL + RAG")
    print("=" * 65)
    print("   Guardrails  : Active")
    print("   HITL        : Enabled (breakpoint before execute_action_node)")
    print("   Tools       : get_current_time | get_local_weather | search_wikipedia")
    print("=" * 65 + "\n")


def print_messages(messages: list) -> None:
    """Pretty-prints the conversation messages for inspection."""
    print(f"\n{DIVIDER}")
    print("  CONVERSATION HISTORY")
    print(DIVIDER)
    for msg in messages:
        role = type(msg).__name__.replace("Message", "").upper()
        content = extract_text(msg.content) if hasattr(msg, "content") else ""
        # Truncate very long tool outputs for readability
        if len(content) > 600:
            content = content[:600] + "\n  ... [truncated for display] ..."
        if role not in ("SYSTEM",):   # Skip system message to keep output clean
            print(f"\n  [{role}]\n  {content}")
    print(f"\n{DIVIDER}\n")


# ===========================================================================
# 8.  MAIN EXECUTION
# ===========================================================================

def main() -> None:
    """
    Entry point for the Secure Executive Assistant demo.

    Workflow
    --------
    1. Build and compile the LangGraph graph with MemorySaver checkpointer.
    2. Submit a complex user query that:
       - Requires weather + time lookups (guardrail: travel query).
       - Requires a Wikipedia/RAG search.
       - Ends with a high-stakes action (booking) that needs HITL approval.
    3. Stream the graph execution until it hits the HITL breakpoint.
    4. Display the agent's research results and proposed action.
    5. Prompt the human for approval (yes/no).
    6. Update the state with the human's decision.
    7. Resume the graph to completion.
    """
    # Ensure stdout uses UTF-8 on Windows terminals that default to cp1252
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print_banner()

    # ── Step 1: Build the graph ─────────────────────────────────────────────
    print("[main] Building graph...")
    graph = build_graph()
    print("[main] Graph compiled successfully.\n")

    # Each run is identified by a thread_id.  This allows multiple independent
    # conversations to share the same compiled graph (and checkpointer).
    thread_id = "secure-assistant-demo-001"
    config = {"configurable": {"thread_id": thread_id}}

    # ── Step 2: Define the user query ───────────────────────────────────────
    user_query = (
        "Check the current time and the weather in Paris, "
        "then look up information about the Eiffel Tower, "
        "and finally book me a flight ticket to Paris for tomorrow."
    )

    print(f"[main] User query:\n  >> {user_query}\n")
    print(DIVIDER)

    initial_state: AgentState = {
        "messages": [HumanMessage(content=user_query)],
        "is_approved": None,
        "proposed_action": "",
    }

    # ── Step 3: Run graph until HITL breakpoint ──────────────────────────────
    # graph.stream() yields state snapshots after each node execution.
    # The graph automatically PAUSES before "execute_action_node" and returns.
    print("[main] Running agent (will pause at HITL breakpoint)...\n")

    for event in graph.stream(initial_state, config=config, stream_mode="values"):
        # Print each message as it arrives for real-time feedback
        if event.get("messages"):
            last_msg = event["messages"][-1]
            if isinstance(last_msg, AIMessage):
                text = extract_text(last_msg.content)
                # Only print substantive AI responses (not tool-call stubs)
                if text and not (hasattr(last_msg, "tool_calls") and last_msg.tool_calls):
                    print(f"\n[AGENT RESPONSE]\n{text}\n")
            elif isinstance(last_msg, ToolMessage):
                print(f"\n[TOOL: {last_msg.name}]\n{str(last_msg.content)[:300]}...\n")

    # ── Step 4: Retrieve state and display proposed action ───────────────────
    current_state = graph.get_state(config)
    current_values = current_state.values

    print(f"\n{DIVIDER}")
    print("  AGENT HAS PAUSED – HUMAN REVIEW REQUIRED")
    print(DIVIDER)

    proposed = current_values.get("proposed_action", "")
    if proposed:
        print(f"\n  Proposed Action:\n  >> {proposed}\n")
    else:
        print("\n  [No explicit proposed action extracted – see conversation above]\n")

    print_messages(current_values.get("messages", []))

    # ── Step 5: Human approval prompt ───────────────────────────────────────
    print(DIVIDER)
    while True:
        human_input = input(
            "\n  Do you approve the agent's proposed action? (yes / no): "
        ).strip().lower()
        if human_input in ("yes", "y", "no", "n"):
            break
        print("  Please type 'yes' or 'no'.")

    approved: bool = human_input in ("yes", "y")
    print(f"\n  Decision recorded: {'APPROVED' if approved else 'REJECTED'}")
    print(DIVIDER)

    # ── Step 6: Update state and resume graph ────────────────────────────────
    # graph.update_state() writes the human's decision into the checkpoint.
    # The next stream() call resumes from the breakpoint with the updated state.
    graph.update_state(
        config,
        {"is_approved": approved},
    )

    print("\n[main] Resuming graph with human decision...\n")

    # Passing None as input resumes from the last checkpoint.
    # The graph continues from execute_action_node with the updated state.
    for event in graph.stream(None, config=config, stream_mode="values"):
        if event.get("messages"):
            last_msg = event["messages"][-1]
            if isinstance(last_msg, AIMessage):
                text = extract_text(last_msg.content)
                if text:
                    print(f"\n[FINAL AGENT RESPONSE]\n{text}\n")

    # ── Step 7: Completion ───────────────────────────────────────────────────
    print(f"\n{DIVIDER}")
    print("  WORKFLOW COMPLETE")
    print(f"{DIVIDER}\n")


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    main()
