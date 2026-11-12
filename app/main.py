#This file connects your individual AI components into a usable backend API that can receive questions, apply safety checks, execute an intelligent RAG workflow, maintain conversation state and return answers to a frontend.
# ============================================================
# CRITICAL: Configure Logfire BEFORE importing application modules.
# This ensures that logging and tracing are initialized before
# other modules start creating spans or emitting logs.
# ============================================================

import logfire
import os

# Loads environment variables from the project's .env file.
from dotenv import load_dotenv

load_dotenv()

# Configure Logfire using the authentication token stored in .env.
# This enables observability: logs, traces and execution spans.
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))


# ============================================================
# APPLICATION IMPORTS
# Logfire is now configured, so we can safely import app modules.
# ============================================================

from fastapi import FastAPI, Response

# Import the compiled LangGraph RAG agent.
from app.agents.graph import rag_agent

# Import NeMo Guardrails initialization and request-checking functions.
from app.guardrails import initialize_rails, guard

from pydantic import BaseModel
from typing import Optional


# STEP 1: INITIALIZE FASTAPI

# Create the FastAPI application.
# This exposes HTTP endpoints through which users can interact
# with our Agentic RAG system.
app = FastAPI(title="Enterprise Agentic RAG API")


# STEP 2: APPLICATION STARTUP

@app.on_event("startup")
def startup_event():

    # Initialize NeMo Guardrails when the API server starts.
    # This prepares the safety and conversation-control layer.
    initialize_rails()


# STEP 3: DEFINE REQUEST STRUCTURE

class QueryRequest(BaseModel):

    # q: The user's question or message.
    # Required field.
    q: str

    # thread_id: Identifies a conversation.
    # Requests with the same thread_id share LangGraph checkpoint memory.
    # If not supplied, "default_user" is used.
    thread_id: Optional[str] = "default_user"


# STEP 4: HOME ENDPOINT

@app.get("/")
def home():

    # Simple health-check endpoint.
    # Confirms that the API is responding.
    return {
        "message": "Enterprise LangGraph RAG API is live."
    }


# STEP 5: GRAPH VISUALIZATION ENDPOINT

@app.get("/graph")
def get_graph_image():
    """
    Returns a PNG image representing the LangGraph workflow.
    """

    try:

        # Get the compiled agent's graph structure.
        # Convert its nodes and connections into a Mermaid diagram.
        # Render that diagram as PNG bytes.
        png_bytes = rag_agent.get_graph().draw_mermaid_png()

        # Return the generated image with the correct MIME type.
        return Response(
            content=png_bytes,
            media_type="image/png"
        )

    except Exception as e:

        # Return an error if graph visualization fails.
        return {
            "error": f"Could not generate graph image: {e}"
        }


# STEP 6: MAIN QUERY ENDPOINT

@app.post("/query")
def query(request: QueryRequest):
    """
    Receives a user query and executes the Agentic RAG pipeline.
    """

    # Extract the question from the incoming request.
    q = request.q

    # Extract the conversation ID for LangGraph memory.
    thread_id = request.thread_id

    # STEP 6A: INITIALIZE AGENT STATE

    # Create the initial shared state for this graph execution.
    initial_state = {

        # Store the user's message in conversation history.
        "messages": [
            {"role": "user", "content": q}
        ],

        # Initially, the current query is the original user question.
        # The Planner may later replace this with a refined search query.
        "current_query": q,

        # No documents have been retrieved yet.
        "documents": [],

        # Track the execution plan.
        "plan": ["Start"],

        # Track the current workflow status.
        "status": "Initializing Graph..."
    }

    # STEP 6B: CONFIGURE CONVERSATION MEMORY

    # Pass the thread_id to LangGraph's checkpointer.
    # Reusing the same thread_id allows the agent to access
    # previous conversation state.
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    try:

        # ====================================================
        # GATE 1: NEMO GUARDRAILS
        # ====================================================

        # Check the incoming question BEFORE running the RAG pipeline.
        #
        # Guardrails can identify disallowed requests, jailbreak attempts,
        # off-topic queries or dialog that should be handled directly.
        #
        # rail_fired: Whether a guardrail intercepted the request.
        # rail_response: The response produced by the guardrail.
        rail_fired, rail_response = guard(q)

        if rail_fired:

            # Log that the request was intercepted.
            logfire.info(
                f"🛡️ Request blocked by guardrails | thread={thread_id}"
            )

            # Return the guardrail response directly.
            # LangGraph retrieval and generation are skipped.
            return {
                "question": q,
                "answer": rail_response,
                "thought_process": [
                    "Intent: Guardrails Fired",
                    "Retrieval: Skipped"
                ],
                "status": "Blocked by guardrails.",
                "sources": []
            }

        # ====================================================
        # GATE 2: LANGGRAPH RAG PIPELINE
        # ====================================================

        # If guardrails allow the request, execute the compiled agent.
        #
        # LangGraph internally performs:
        # Planner -> Conditional Router
        #             -> Retriever (technical queries)
        #             -> Responder
        #
        # For conversational queries, it skips the Retriever.
        #
        # invoke() runs the graph synchronously.
        # Synchronous execution also preserves the current execution
        # context for Logfire instrumentation.
        final_output = rag_agent.invoke(
            initial_state,
            config=config
        )

        # STEP 6C: FORMAT THE API RESPONSE

        # Extract the final answer, execution plan, status and documents
        # from the final LangGraph state.
        return {
            "question": q,
            "answer": final_output.get("final_answer"),
            "thought_process": final_output.get("plan"),
            "status": final_output.get("status"),
            "sources": final_output.get("documents", [])
        }

    except Exception as e:

        # ====================================================
        # ERROR HANDLING
        # ====================================================

        # Log the internal error for debugging and monitoring.
        logfire.error(
            f"❌ Backend Execution Failed: {e}"
        )

        # Return a safe, user-friendly error response.
        # Avoid exposing internal exceptions to API consumers.
        return {
            "question": q,
            "answer": (
                "I apologize, but I encountered an internal error "
                "while processing your request. Please try again later."
            ),
            "thought_process": [
                "Error encountered during execution."
            ],
            "status": "error",
            "sources": []
        }