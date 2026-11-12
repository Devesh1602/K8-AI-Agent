# ============================================================
# STEP 1: IMPORT REQUIRED LIBRARIES
# ============================================================

import os                  # File paths and environment variables
import streamlit as st     # Builds the frontend UI
import requests            # Sends HTTP requests to FastAPI backend
import time                # Creates the answer streaming effect
import uuid                # Generates unique conversation IDs
import logfire             # Observability: logs and traces
from dotenv import load_dotenv  # Loads environment variables from .env


# ============================================================
# STEP 2: LOAD ENVIRONMENT VARIABLES
# ============================================================

# Find the .env file in the project's root directory.
# __file__ represents the location of the current Python file.
# dirname(__file__) gets the frontend directory.
# ".." moves one directory up to the project root.
env_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".env")
)

# Load API keys, backend URL and Logfire configuration.
load_dotenv(dotenv_path=env_path)


# ============================================================
# STEP 3: INITIALIZE LOGFIRE
# ============================================================

try:
    # Read the Logfire authentication token from .env.
    token = os.getenv("LOGFIRE_TOKEN")

    # Warn if the token is missing.
    if not token:
        print("ERROR: LOGFIRE_TOKEN is empty or None!")

    # Connect this frontend application to Logfire.
    logfire.configure(token=token)

    # HTTP instrumentation is disabled here because of
    # the OpenTelemetry compatibility issue mentioned in your code.
    # logfire.instrument_requests()

    LOGFIRE_STATUS = "Connected & Tracing"

except Exception as e:

    # Keep the UI operational even if Logfire initialization fails.
    print(f"Logfire Init Error in UI: {e}")

    LOGFIRE_STATUS = f"Standby (Error: {e})"


# ============================================================
# STEP 4: CONFIGURE STREAMLIT PAGE
# ============================================================

st.set_page_config(
    page_title="Enterprise Agentic RAG",  # Browser tab title
    page_icon="🤖",                       # Browser tab icon
    layout="wide",                        # Use available screen width
)


# ============================================================
# STEP 5: DEFINE CHAT AVATARS
# ============================================================

AI_AVATAR = "🤖"
USER_AVATAR = "👤"


# ============================================================
# STEP 6: SESSION MANAGEMENT
# ============================================================

# Streamlit reruns the Python script after user interactions.
# session_state preserves selected data across those reruns.

# Generate a unique ID when a new browser session starts.
if "session_id" not in st.session_state:

    st.session_state.session_id = str(uuid.uuid4())

    logfire.info(
        f"✨ New User Session Created: {st.session_state.session_id}"
    )

# Initialize conversation history if it does not exist.
if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# STEP 7: SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🧠 Agent OS")
    st.markdown("---")

    # Display Logfire connection status.
    st.success(f"Logfire: {LOGFIRE_STATUS}")

    # Show the first 8 characters of the conversation ID.
    st.info(f"Memory ID: {st.session_state.session_id[:8]}")

    # Button to clear conversation history and start a new session.
    if st.button(
        "🗑️ Clear History & Memory",
        width="stretch",
        type="primary"
    ):

        # Record the memory reset event.
        logfire.warn(
            f"🗑️ Memory Wipe Triggered for session: "
            f"{st.session_state.session_id}"
        )

        # Clear frontend conversation history.
        st.session_state.messages = []

        # Generate a NEW ID so the backend starts a separate
        # LangGraph conversation thread.
        st.session_state.session_id = str(uuid.uuid4())

        # Rerun the Streamlit application with the new session.
        st.rerun()


# ============================================================
# STEP 8: MAIN CHAT INTERFACE
# ============================================================

st.title("🤖 Enterprise Agentic Assistant")


# Display all previous messages from Streamlit session memory.
for message in st.session_state.messages:

    # Select avatar based on the message sender.
    avatar = (
        AI_AVATAR
        if message["role"] == "assistant"
        else USER_AVATAR
    )

    # Display the historical message.
    with st.chat_message(message["role"], avatar=avatar):

        st.markdown(message["content"])


# ============================================================
# STEP 9: CAPTURE USER INPUT
# ============================================================

# Display a chat input box.
# The block executes only when the user submits a message.
if prompt := st.chat_input("Ask about your documentation..."):

    # Start a trace for the complete user interaction.
    with logfire.span(
        "💬 User Chat Interaction",
        user_query=prompt,
        session_id=st.session_state.session_id
    ):

        # Save the user message in frontend conversation history.
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        # Immediately display the user's message.
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(prompt)


        # ====================================================
        # STEP 10: CALL THE FASTAPI BACKEND
        # ====================================================

        with st.chat_message("assistant", avatar=AI_AVATAR):

            # Display a live status container while the agent works.
            with st.status(
                "🔍 Agent is thinking...",
                expanded=True
            ) as status:

                try:

                    # Create a nested trace for backend communication.
                    with logfire.span("📡 Calling RAG Backend"):

                        # Read backend URL from .env.
                        # Use localhost:8000 if BACKEND_URL is missing.
                        base_url = os.getenv(
                            "BACKEND_URL",
                            "http://localhost:8000"
                        )

                        # Build the query endpoint URL.
                        url = f"{base_url}/query"

                        # Send the question and conversation ID.
                        payload = {
                            "q": prompt,
                            "thread_id": st.session_state.session_id
                        }

                        # Send a POST request to FastAPI.
                        # Wait at most 60 seconds for a response.
                        response = requests.post(
                            url,
                            json=payload,
                            timeout=60
                        )

                        # Convert the JSON response into a Python dictionary.
                        data = response.json()

                    # =================================================
                    # STEP 11: DISPLAY AGENT EXECUTION PLAN
                    # =================================================

                    # Extract execution steps returned by LangGraph.
                    steps = data.get("thought_process", [])

                    # Display each step in the Streamlit status area.
                    for step in steps:
                        st.write(f"⚙️ {step}")

                    # Mark processing as complete.
                    status.update(
                        label="✅ Answer Synthesized",
                        state="complete",
                        expanded=False
                    )


                    # =================================================
                    # STEP 12: DISPLAY RETRIEVED SOURCES
                    # =================================================

                    sources = data.get("sources", [])

                    if sources:

                        # Show source documents inside an expandable section.
                        with st.expander(
                            "📄 View Retrieved Context (Sources)"
                        ):

                            for i, source in enumerate(sources):

                                # Generate a short preview for the chunk.
                                preview = (
                                    source[:100].replace("\n", " ")
                                    + "..."
                                )

                                # Create an individual expandable section.
                                with st.expander(
                                    f"Chunk {i+1}: {preview}"
                                ):
                                    st.info(source)


                except Exception as e:

                    # Log backend communication failures.
                    logfire.error(
                        f"❌ UI-Backend Connection Failed: {e}"
                    )

                    status.update(
                        label="❌ Connection Failed",
                        state="error"
                    )

                    st.error("Backend Offline.")

                    # Stop execution if the request failed.
                    st.stop()


            # =================================================
            # STEP 13: SIMULATE ANSWER STREAMING
            # =================================================

            # Create an empty UI placeholder for the answer.
            answer_placeholder = st.empty()

            # Extract the answer returned by FastAPI.
            full_answer = data.get("answer", "No response.")

            curr_text = ""

            # Display the answer character by character.
            # This simulates streaming even though the complete
            # answer has already been received from the backend.
            for char in full_answer:

                curr_text += char

                # Display a blinking cursor while rendering.
                answer_placeholder.markdown(curr_text + "▌")

                # Small delay between characters.
                time.sleep(0.005)

            # Display the final complete answer.
            answer_placeholder.markdown(full_answer)


            # =================================================
            # STEP 14: SAVE ASSISTANT RESPONSE
            # =================================================

            # Append assistant response to frontend chat history.
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_answer
            })

            # Log successful completion of the interaction.
            logfire.info("✅ Chat cycle completed successfully.")