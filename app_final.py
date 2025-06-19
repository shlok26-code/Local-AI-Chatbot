from main_final import get_answer
import streamlit as st

st.set_page_config(page_title="Local QA Chatbot", page_icon="💬")

st.title(" Local QA Chatbot 💬")
st.write("Ask your questions about your documents!")

# Model selection
if "model_choice" not in st.session_state:
    st.session_state["model_choice"] = None
if "model_error" not in st.session_state:
    st.session_state["model_error"] = ""

# Chat Bot name display mapping
bot_name_map = {
    "1": "Mistral-7B-Instruct-v0.2",
    "2": "Meta-Llama-3-8B-Instruct"
}

if st.session_state["model_choice"] is None:
    st.write("Select which model to use for computation:")
    st.write("1. Mistral-7B-Instruct-v0.2")
    st.write("2. Meta-Llama-3-8B-Instruct")
    with st.form("model_select_form"):
        model_input = st.text_input("Enter 1 or 2 to select the model:")
        model_submit = st.form_submit_button("Select Model")
        if model_submit:
            if model_input == "1" or model_input == "2":
                st.session_state["model_choice"] = model_input
                st.session_state["model_error"] = ""
                st.rerun()
            else:
                st.session_state["model_error"] = "Please enter 1 or 2 only."
    if st.session_state["model_error"]:
        st.error(st.session_state["model_error"])
    st.stop()

# Display Chat Bot name based on model choice
st.write(f"**Chat Bot:** {bot_name_map.get(st.session_state['model_choice'],'Unknown')}")

# Session state to store the chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Streamlit form for input and submit button, which handles clearing input automatically
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Your question:")
    col1, col2 = st.columns([2, 1])
    with col1:
        submitted = st.form_submit_button("Submit")
    with col2:
        back = st.form_submit_button("Back to Bot Selection")
    if submitted and user_input:
        # Use model_choice as parameter to get_answer
        answer = get_answer(user_input, st.session_state["model_choice"])
        st.session_state["messages"].append({"user": user_input, "bot": answer})
    if back:
        st.session_state["model_choice"] = None
        st.session_state["messages"] = []
        st.rerun()

# Display chat history
for msg in st.session_state["messages"]:
    st.markdown(f"**Question:** {msg['user']}")
    st.markdown(f"**Answer:** {msg['bot']}")