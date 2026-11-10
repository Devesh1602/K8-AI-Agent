# Think of this graph as the orchestrator or brain of your RAG agent.
# - Planner = Decision-maker.
# - Retriever = Knowledge gatherer.
# - Responder = Answer generator.
# - Router = Traffic controller.
# - AgentState = Shared working memory.
# - MemorySaver = Conversation checkpoint storage.
# - LangGraph = Coordinates the entire workflow.


# StateGraph: Creates the LangGraph workflow.
# END: Special marker indicating that the workflow has finished.
from langgraph.graph import StateGraph, END

# MemorySaver: Stores conversation state between executions
# so the agent can maintain context across multiple turns.
from langgraph.checkpoint.memory import MemorySaver

# Import the shared state structure used by all graph nodes.
from app.agents.state import AgentState

# Import the three nodes responsible for planning, retrieval and generation.
from app.agents.nodes.planner import planner_node
from app.agents.nodes.retriever import retrieve_node
from app.agents.nodes.responder import generate_node


# STEP 1: INITIALIZE THE STATE GRAPH

# Create a graph whose shared state follows the AgentState structure.
# Every node can read and update this shared state.
workflow = StateGraph(AgentState)


# STEP 2: REGISTER THE NODES

# Register the Planner node.
# It analyzes the user's intent and decides whether retrieval is needed.
workflow.add_node("planner", planner_node)

# Register the Retriever node.
# It searches Qdrant and reranks the retrieved documents using FlashRank.
workflow.add_node("retriever", retrieve_node)

# Register the Responder node.
# It generates the final answer using conversation history
# or retrieved technical context.
workflow.add_node("responder", generate_node)


# STEP 3: DEFINE CONDITIONAL ROUTING

def route_planner(state: AgentState):
    """
    Decides which node should execute after the Planner.

    Conversational query -> Responder
    Technical query      -> Retriever
    """

    # If the Planner identifies a conversational question,
    # skip document retrieval and go directly to the Responder.
    if state["current_query"] == "CONVERSATIONAL":
        return "responder"

    # Otherwise, send the query to the Retriever.
    return "retriever"


# Set the Planner as the first node executed when the graph starts.
workflow.set_entry_point("planner")


# STEP 4: CONNECT THE PLANNER TO THE CONDITIONAL ROUTER

# After the Planner finishes, route_planner() evaluates the state.
#
# If it returns "retriever", execute the Retriever node.
# If it returns "responder", execute the Responder node.
workflow.add_conditional_edges(
    "planner",
    route_planner,
    {
        "retriever": "retriever",
        "responder": "responder"
    }
)


# STEP 5: DEFINE THE REMAINING CONNECTIONS

# Once retrieval is complete, always pass the retrieved documents
# to the Responder for final answer generation.
workflow.add_edge("retriever", "responder")

# Once the Responder generates the answer, terminate the workflow.
workflow.add_edge("responder", END)


# STEP 6: ENABLE CONVERSATION MEMORY

# MemorySaver stores graph checkpoints in memory.
# Each conversation can maintain its own state using a thread_id.
checkpointer = MemorySaver()


# STEP 7: COMPILE THE GRAPH

# Compile the workflow into an executable LangGraph application.
# The checkpointer enables state persistence across turns
# when the same thread_id is reused.
rag_agent = workflow.compile(checkpointer=checkpointer)