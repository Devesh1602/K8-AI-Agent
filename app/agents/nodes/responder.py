#The Generator is the final synthesis node. It takes the Planner's decision, combines the appropriate information, calls the LLM through Portkey, checks whether caching was used, and saves the final answer back into LangGraph's shared state.

import logfire

# Shared LangGraph state containing messages, query, documents,
# execution plan, status and final answer.
from app.agents.state import AgentState

# Native Portkey client lets us call the LLM and inspect
# response headers for cache information.
from app.gateway import portkey_client, extract_cache_status


def generate_node(state: AgentState):
    """
    Generation Node:
    Produces the final answer using either:
    1. Conversation history (conversational queries).
    2. Retrieved documents + conversation history (technical RAG queries).

    Uses native Portkey client to detect whether the response was cached.
    """

    # STEP 1: GET THE PLANNER'S DECISION
    # The Planner previously populated current_query.
    query = state["current_query"]

    # STEP 2: BUILD CONVERSATION HISTORY
    # Exclude the latest message because it is handled separately.
    history_str = ""

    for msg in state["messages"][:-1]:

        # Identify the sender of each previous message.
        role = "User" if msg["role"] == "user" else "Assistant"

        # Append previous messages to the history.
        history_str += f"{role}: {msg['content']}\n"

    # Extract the latest user message.
    user_msg = (
        state["messages"][-1]["content"]
        if state["messages"] else ""
    )

    # STEP 3: SELECT THE GENERATION PATH

    # PATH A: CONVERSATIONAL RESPONSE
    # If the Planner classified the request as conversational,
    # use conversation history without retrieving documents.
    if query == "CONVERSATIONAL":

        logfire.info("Generating conversational response using memory.")

        prompt = f"""
        You are a friendly and helpful Enterprise AI Assistant.
        Answer the user's latest message using the CONVERSATION HISTORY below.

        CONVERSATION HISTORY:
        {history_str}

        LATEST MESSAGE:
        "{user_msg}"
        """

    # PATH B: TECHNICAL RAG RESPONSE
    # If the Planner classified the request as technical,
    # use retrieved documentation as additional context.
    else:

        logfire.info("Generating technical RAG response.")

        # Set a character limit to avoid sending excessive context
        # to the LLM and exceeding token-per-minute limits.
        max_context_chars = 25000

        # Start with an empty context string.
        full_context = ""

        # Add retrieved document chunks until the character limit is reached.
        for doc in state["documents"]:

            if len(full_context) + len(doc) < max_context_chars:

                # Append the document chunk to the context.
                full_context += doc + "\n\n"

            else:

                # Stop adding documents when the context limit is reached.
                logfire.warning("Context truncated to fit Groq TPM limits.")
                break

        # Build the technical RAG prompt.
        # The LLM receives documentation and conversation history.
        prompt = f"""
        You are a Senior Technical Architect.
        Answer the question using the TECHNICAL CONTEXT provided.

        TECHNICAL CONTEXT:
        {full_context}

        CONVERSATION HISTORY:
        {history_str}

        USER QUESTION:
        "{user_msg}"
        """

    # STEP 4: CALL THE LLM THROUGH PORTKEY
    # Create a Logfire span to monitor the generation operation.
    with logfire.span("✍️ LLM Synthesis"):

        try:

            # Send the prompt to the LLM using Portkey's native client.
            #
            # messages: Supplies the prompt in chat-completion format.
            # temperature=0.1: Encourages more consistent, less creative answers.
            #
            # Unlike the planner's LangChain interface, this native client
            # allows us to inspect the Portkey response headers.
            response = portkey_client.chat.completions.create(
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )

            # Extract the generated answer from the API response.
            content = response.choices[0].message.content

            # STEP 5: CHECK PORTKEY CACHE STATUS
            # Inspect the response metadata/header to determine
            # whether Portkey served this response from its cache.
            cache_status = extract_cache_status(response)

            # True only when Portkey reports a cache HIT.
            is_cache_hit = cache_status == "HIT"

            if is_cache_hit:

                # Record cache hit in observability logs.
                logfire.info(
                    "⚡ Gateway Cache Hit — response served from Portkey cache."
                )

                # Append cache information to the existing execution plan.
                plan_update = state["plan"] + ["Cache: Hit ⚡"]

                status = "Cache hit — instant response."

            else:

                # Response was generated without a reported cache hit.
                logfire.info("✅ Response synthesised via LLM.")

                # Keep the existing execution plan unchanged.
                plan_update = state["plan"]

                status = "Response generated."

            # STEP 6: UPDATE LANGGRAPH STATE
            return {

                # Store the generated answer.
                "final_answer": content,

                # Update workflow status.
                "status": status,

                # Update execution plan, including cache information if applicable.
                "plan": plan_update,

                # Append the assistant response to conversation history.
                # AgentState's operator.add reducer preserves previous messages.
                "messages": [
                    {"role": "assistant", "content": content}
                ]
            }

        except Exception as e:

            # Log the generation error for debugging and monitoring.
            logfire.error(f"LLM Generation failed: {e}")

            # Re-raise the exception so LangGraph or its caller
            # can handle the failure.
            raise