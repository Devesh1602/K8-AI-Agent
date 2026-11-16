#This code acts as a security checkpoint in front of your RAG pipeline: it checks every incoming user message against NeMo Guardrails and either blocks the request with a guardrail response or allows it to proceed to LangGraph.


# Logfire is used for application observability, logging,
# and tracing execution of the guardrails pipeline.
import logfire

# ChatGroq provides the Groq-hosted LLM through LangChain.
# We use a smaller, faster model for guardrail classification.
from langchain_groq import ChatGroq

# RailsConfig loads the guardrail configuration (Colang + YAML).
# LLMRails executes the configured NeMo Guardrails.
from nemoguardrails import RailsConfig, LLMRails

# Import application settings.
# This includes configuration such as GROQ_API_KEY.
from app.config import settings

# Import the actual guardrail rules defined elsewhere.
# COLANG_CONTENT: Defines conversational/guardrail flows.
# YAML_CONTENT: Defines model and rail configuration.
# RAIL_INDICATORS: Strings used to identify a fired rail response.
from app.guardrails.colang_rules import (
    COLANG_CONTENT,
    YAML_CONTENT,
    RAIL_INDICATORS
)


# Type annotation:
# _rails can either contain an initialized LLMRails object
# or None if guardrails have not been initialized.
#
# It is a module-level variable, so it can be reused
# across requests instead of rebuilding the rails every time.
_rails: LLMRails | None = None


def initialize_rails() -> None:
    """
    Initialize NeMo Guardrails once during application startup.

    The guardrail model is llama-3.1-8b-instant.
    It is intentionally smaller and faster than the main
    RAG model, reducing the overhead of checking every query.
    """

    # We modify the module-level _rails variable.
    # Without global, Python would treat _rails as a local variable.
    global _rails

    # Create the LLM that NeMo Guardrails will use.
    guard_llm = ChatGroq(
        api_key=settings.GROQ_API_KEY,  # Groq API authentication
        model="llama-3.1-8b-instant",   # Lightweight guardrail model
        temperature=0                   # More deterministic responses
    )

    # Load the guardrail configuration from in-memory strings.
    #
    # COLANG_CONTENT contains the guardrail conversation flows.
    # YAML_CONTENT contains the configuration for those flows,
    # models, and other NeMo settings.
    config = RailsConfig.from_content(
        colang_content=COLANG_CONTENT,
        yaml_content=YAML_CONTENT
    )

    # Create the actual NeMo Guardrails engine.
    #
    # config = Rules and configuration
    # guard_llm = LLM used by the guardrails engine
    #
    # The resulting object is stored globally and reused.
    _rails = LLMRails(config, llm=guard_llm)

    # Log successful initialization for monitoring/debugging.
    logfire.info(
        "🛡️ NeMo Guardrails initialised (llama-3.1-8b-instant)."
    )


def guard(message: str) -> tuple[bool, str | None]:
    """
    Run a user's message through NeMo Guardrails.

    Returns:
        (True, rail_response):
            A configured rail appears to have fired.
            Return the guardrail response and skip RAG.

        (False, None):
            No configured rail indicator was detected.
            Allow the request to continue to LangGraph.
    """

    # Safety check:
    # If initialize_rails() was never called, _rails is None.
    if _rails is None:

        # Log the issue, but fail open:
        # The request is allowed to continue without guardrails.
        logfire.warning(
            "⚠️ Guardrails not initialised — skipping gate."
        )
        return False, None

    # Create an observability span.
    # This lets Logfire track how long the guardrail check takes
    # and associate its logs with this execution.
    with logfire.span("🛡️ Guardrails Check"):

        # Send the user's message to NeMo Guardrails.
        #
        # NeMo evaluates the message against the configured flows.
        # The LLM may also be invoked depending on the configuration.
        result = _rails.generate(
            messages=[
                {
                    "role": "user",
                    "content": message
                }
            ]
        )

        # Extract the assistant's response.
        #
        # Expected result:
        # {'role': 'assistant', 'content': '...'}
        #
        # If result is a dictionary, extract its content.
        # Otherwise, convert the entire result into a string.
        content = (
            result.get("content", "")
            if isinstance(result, dict)
            else str(result)
        )

        # Check whether the returned content contains
        # any predefined rail indicator.
        #
        # Example:
        # If RAIL_INDICATORS contains "I cannot help with that"
        # and the response contains this phrase,
        # fired becomes True.
        fired = any(
            indicator in content
            for indicator in RAIL_INDICATORS
        )

        # If an indicator is detected, treat the request
        # as blocked by a guardrail.
        if fired:

            # Log the event.
            # Only the first 80 characters of the query are logged.
            logfire.info(
                f"🛡️ Guardrails fired | query='{message[:80]}'"
            )

            # Return True to tell the calling application:
            # Do not continue to the RAG pipeline.
            # Return the guardrail's response to the user.
            return True, content

        # No indicator was detected.
        # The query is considered clean by this gate.
        logfire.info("✅ Guardrails passed.")

        # Tell the calling application to continue to LangGraph.
        return False, None
