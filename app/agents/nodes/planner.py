# Import the shared state structure used across LangGraph nodes.
from app.agents.state import AgentState

# Returns the LangChain-compatible LLM configured through Portkey.
from app.gateway import get_langchain_llm

import logfire


# Initialize the LLM for the planner.
# feature="planner" selects the planner-specific gateway configuration.
#
# Portkey provides gateway capabilities such as fallback, caching
# and retries while preserving the LangChain .invoke() interface.
llm = get_langchain_llm(feature="planner")


def planner_node(state: AgentState):
    """
    Planner Node:
    Understands the latest user message in the context of the
    entire conversation and decides whether retrieval is required.
    """

    # STEP 1: BUILD CONVERSATION HISTORY

    # Start with an empty string to collect previous messages.
    history = ""

    # Exclude the latest message because we process it separately below.
    for msg in state["messages"][:-1]:

        # Identify who sent each message.
        role = "User" if msg["role"] == "user" else "Assistant"

        # Append each previous message to the conversation history.
        history += f"{role}: {msg['content']}\n"

    # STEP 2: EXTRACT LATEST USER MESSAGE

    # Take the last message from the conversation.
    # If there are no messages, use an empty string.
    user_message = (
        state["messages"][-1]["content"]
        if state["messages"] else ""
    )

    # STEP 3: CREATE THE PLANNER PROMPT

    # Give the LLM both the previous conversation and latest message.
    # This helps it understand follow-up questions and user intent.
    prompt = f"""
    You are an intelligent Assistant Planner.
    Analyze the conversation history and the latest user message.

    CONVERSATION HISTORY:
    {history}

    LATEST MESSAGE:
    "{user_message}"

    Task:
    1. If the latest message is a greeting (hi, hello) or a question
       that can be answered using ONLY the conversation history above
       (e.g., "what is my name"), respond with 'CONVERSATIONAL'.

    2. If it is a technical question about Kubernetes, Intel, or
       Networking that requires fresh documentation, output a
       refined search query.

    Output ONLY 'CONVERSATIONAL' or the search query.
    """

    # STEP 4: ASK THE LLM TO MAKE A DECISION

    # Logfire creates a trace/span so we can monitor this planner operation.
    with logfire.span("🧠 Planner Decision"):

        # Send the prompt to the LLM.
        # .invoke() executes the model request.
        # .content extracts the actual text response.
        # .strip() removes unnecessary whitespace.
        decision = llm.invoke(prompt).content.strip()

        # Record the planner's decision in Logfire.
        logfire.info(f"Intent identified: {decision}")

    # STEP 5: HANDLE CONVERSATIONAL QUERIES

    # If the LLM determines that external retrieval is unnecessary:
    if decision == "CONVERSATIONAL":

        return {

            # Special marker telling downstream nodes to skip retrieval.
            "current_query": "CONVERSATIONAL",

            # Update the workflow status.
            "status": "Handling conversationally (using memory)...",

            # Record the reasoning/action plan.
            "plan": [
                "Intent: Conversational/Memory",
                "Retrieval: Skipped"
            ]
        }

    # STEP 6: HANDLE TECHNICAL / RESEARCH QUERIES

    # If the LLM returns a search query instead:
    return {

        # Store the refined query for the retrieval node.
        "current_query": decision,

        # Update the workflow status.
        "status": f"Technical research needed. Searching for: {decision}",

        # Store the execution plan.
        "plan": [
            "Intent: Technical",
            f"Search Term: {decision}"
        ]
    }