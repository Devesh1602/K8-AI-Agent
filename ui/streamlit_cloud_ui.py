# ============================================================
# STEP 1: IMPORT REQUIRED LIBRARIES
# ============================================================

import os                  # Environment variables
import streamlit as st     # Frontend UI
import requests            # HTTP communication with FastAPI
import time                # Simulated answer streaming
import uuid                # Unique conversation IDs
import logfire             # Logging and distributed tracing


# ============================================================
# STEP 2: INITIALIZE LOGFIRE
# ============================================================

try:

    # Read Logfire token from Streamlit Secrets first.
    # If unavailable, fall back to the system environment variable.
    #
    # st.secrets is useful when deploying through Streamlit Cloud.
    logfire.configure(
        token=st.secrets.get(
            "LOGFIRE_TOKEN",
            os.getenv("LOGFIRE_TOKEN")
        )
    )

    # Instrument outgoing HTTP requests made using requests.
    #
    # This automatically propagates trace context to the FastAPI
    # backend through HTTP headers, allowing frontend and backend
    # operations to be correlated in distributed tracing.
    logfire.instrument_requests()

    LOGFIRE_STATUS = "Connected & Tracing"

except Exception:

    # Allow the frontend to continue even if Logfire is unavailable.
    LOGFIRE_STATUS = "Standby (No Token)"


# ============================================================
# STEP 3: CONFIGURE STREAMLIT PAGE
# ============================================================

st.set_page_config(
    page_title="Enterprise Agentic RAG",
    page_icon="🤖",
    layout="wide",
)


# ============================================================
# STEP 4: DEFINE CHAT AVATARS
# ============================================================

AI_AVATAR = "🤖"
USER_AVATAR = "👤"


# ============================================================
# STEP 5: SESSION MANAGEMENT
# ============================================================

# Streamlit reruns the application after user interactions.
# session_state preserves conversation data between those reruns.

if "session_id" not in st.session_state:

    # Generate a unique ID for this conversation.
    # This ID will also be used as the LangGraph thread_id.
    st.session_state.session_id = str(uuid.uuid4())

    logfire.info(
        f"✨ New User Session Created: {st.session_state.session_id}"
    )


if "messages" not in st.session_state:

    # Initialize an empty conversation history.
    st.session_state.messages = []


# ============================================================
# STEP 6: SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🧠 Agent OS")
    st.markdown("---")

    # Backend URL is configured directly here.
    # FastAPI is expected to run locally on port 8000.
    base_url = "http://localhost:8000"

    st.markdown("---")

    # Display Logfire connection status.
    st.success(f"Logfire: {LOGFIRE_STATUS}")

    # Display a shortened conversation ID.
    st.info(f"Memory ID: {st.session_state.session_id[:8]}")


    # Clear frontend history and start a completely new conversation.
    if st.button(
        "🗑️ Clear History & Memory",
        width="stretch",
        type="primary"
    ):

        # Log the reset event.
        logfire.warning(
            f"🗑️ Memory Wipe Triggered for session: "
            f"{st.session_state.session_id}"
        )

        # Clear messages stored in Streamlit.
        st.session_state.messages = []

        # Generate a new thread ID.
        # The new ID prevents this conversation from loading
        # the previous LangGraph checkpoint.
        st.session_state.session_id = str(uuid.uuid4())

        # Restart the Streamlit script.
        st.rerun()


# ============================================================
# STEP 7: MAIN CHAT INTERFACE
# ============================================================

st.title("🤖 Enterprise Agentic Assistant")


# Display previous conversation messages.
for message in st.session_state.messages:

    # Choose the appropriate avatar.
    avatar = (
        AI_AVATAR
        if message["role"] == "assistant"
        else USER_AVATAR
    )

    with st.chat_message(message["role"], avatar=avatar):

        st.markdown(message["content"])


# ============================================================
# STEP 8: CAPTURE USER INPUT
# ============================================================

if prompt := st.chat_input("Ask about your documentation..."):

    # Start a Logfire trace for this complete interaction.
    with logfire.span(
        "💬 User Chat Interaction",
        user_query=prompt,
        session_id=st.session_state.session_id
    ):

        # Save the user's question in frontend session history.
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        # Display the user's question immediately.
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(prompt)


        # ====================================================
        # STEP 9: ASSISTANT RESPONSE
        # ====================================================

        with st.chat_message("assistant", avatar=AI_AVATAR):

            # Initialize response data.
            # This prevents an undefined variable if execution fails.
            data = {}

            # Display an expandable progress/status container.
            with st.status(
                "🔍 Agent is thinking...",
                expanded=True
            ) as status:

                try:

                    # Start a nested trace for backend communication.
                    with logfire.span("📡 Calling RAG Backend"):

                        # Construct the FastAPI query endpoint.
                        url = f"{base_url}/query"

                        # Send the user's question and conversation ID.
                        payload = {
                            "q": prompt,
                            "thread_id": st.session_state.session_id
                        }

                        # Send POST request to FastAPI.
                        # Timeout prevents waiting indefinitely.
                        response = requests.post(
                            url,
                            json=payload,
                            timeout=60
                        )


                        # =================================================
                        # STEP 10: HTTP ERROR HANDLING
                        # =================================================

                        # Check whether the backend returned HTTP 200.
                        #
                        # If not, display the HTTP status and response body.
                        # st.stop() terminates the current Streamlit execution.
                        if response.status_code != 200:

                            st.error(
                                f"Backend Error: {response.status_code} "
                                f"- {response.text}"
                            )

                            st.stop()


                        # Convert the successful JSON response into
                        # a Python dictionary.
                        data = response.json()


                    # =================================================
                    # STEP 11: DISPLAY AGENT EXECUTION PLAN
                    # =================================================

                    # Get the execution steps returned by LangGraph.
                    steps = data.get("thought_process", [])

                    # Display each execution step in the status container.
                    # unsafe_allow_html=False prevents HTML rendering
                    # from these backend-provided strings.
                    for step in steps:

                        st.markdown(
                            f"⚙️ {step}",
                            unsafe_allow_html=False
                        )


                    # Mark agent processing as complete.
                    status.update(
                        label="✅ Answer Synthesized",
                        state="complete",
                        expanded=False
                    )


                except Exception as e:

                    # Log connection or backend execution errors.
                    logfire.error(
                        f"❌ UI-Backend Connection Failed: {e}"
                    )

                    # Update the progress indicator.
                    status.update(
                        label="❌ Connection Failed",
                        state="error"
                    )

                    # Show a user-friendly error.
                    st.error("Backend Offline.")

                    # Stop further execution.
                    st.stop()


            # =================================================
            # STEP 12: SIMULATE ANSWER STREAMING
            # =================================================

            # Create a placeholder where the answer will appear.
            #
            # IMPORTANT:
            # This is outside st.status(), so the answer remains
            # visible even after the thinking section collapses.
            answer_placeholder = st.empty()

            # Extract the answer returned by FastAPI.
            full_answer = data.get("answer", "No response.")

            curr_text = ""

            # Display the answer one character at a time.
            # This is simulated streaming because the full response
            # has already arrived from the backend.
            for char in full_answer:

                curr_text += char

                # Show the partially generated answer with a cursor.
                answer_placeholder.markdown(curr_text + "▌")

                # Add a small delay for the typing effect.
                time.sleep(0.005)

            # Display the final complete answer without the cursor.
            answer_placeholder.markdown(full_answer)


            # =================================================
            # STEP 13: DISPLAY RETRIEVED SOURCES
            # =================================================

            # Extract retrieved document chunks from FastAPI.
            sources = data.get("sources", [])

            if sources:

                # Display all retrieved chunks in one expandable section.
                #
                # This is also outside st.status(), so sources remain
                # visible after the progress section collapses.
                with st.expander(
                    f"📄 Retrieved Context ({len(sources)} chunks)"
                ):

                    for i, source in enumerate(sources):

                        # Display chunk number.
                        st.caption(f"Chunk {i + 1}")

                        # Display the actual retrieved document text.
                        st.info(source)

            else:

                # Conversational responses may not use document retrieval.
                st.caption(
                    "ℹ️ No context retrieved — conversational response."
                )


            # =================================================
            # STEP 14: SAVE ASSISTANT RESPONSE
            # =================================================

            # Append the final answer to Streamlit conversation history.
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_answer
            })

            # Log successful completion.
            logfire.info("✅ Chat cycle completed successfully.")