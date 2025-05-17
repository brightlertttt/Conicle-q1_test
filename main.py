import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever
# st.title("Getting started streamlit")
# st.write("test")
# st.write("test")

model = OllamaLLM(model='llama3.2', temperature=0.5)
model_fallback = OllamaLLM(model='llama3.2', temperature=0.9)

template_normal = """
You are a helpful learning assistant chatbot.

Your task is to help users discover relevant competencies and suggest related course topics based on their input.

You are given a list of competencies and their descriptions below. Use this information to identify which competencies match the user's input.

Here is a list of competencies and their descriptions:
{competencies}

User input: {question}

Your response:
- If the input is casual or friendly (e.g., "hello", "hi", "how are you?"), respond politely with a light greeting.
- If the input seems unclear or nonsensical (e.g., "asdf", "??"), politely ask the user to clarify or rephrase.
- Otherwise, respond with:
  - 1 to 3 relevant competencies (just names)
  - 1 to 3 course topic suggestions related to those competencies
- Keep the response clear, focused, and helpful.
"""


template_fallback = """
You are a friendly and helpful learning assistant chatbot.

The user asked about a topic, but no matching competencies were found in the dataset provided to you.

Your job is to:
- If the input seems like a real learning goal or interest, even if no exact match was found, use your own knowledge to guess what competencies might be relevant, and suggest 1 to 3 course topics.

User input: {question}

Your response:
- Friendly and professional tone.
- Suggested competencies (if possible) and related course topics based on your best judgment.
"""




# Create two prompt chains
prompt_normal = ChatPromptTemplate.from_template(template_normal)
prompt_fallback = ChatPromptTemplate.from_template(template_fallback)

chain_normal = prompt_normal | model
chain_fallback = prompt_fallback | model_fallback
st.title("Learning Assistant Chatbot")
st.markdown("""
Welcome to the Learning Assistant Chatbot.
Describe a skill you're interested in, or name a competency (e.g., **3D modelling**, **communication**, or **financial reporting**).  
I’ll help you discover relevant competencies and suggest example courses.
""")

if "message" not in st.session_state:
    st.session_state.message = []
user_input = st.chat_input("What would you like to learn about?")
if st.button("Clear"):
    st.session_state.message = []
    st.rerun()
if user_input:
    # Add user message to history
    st.session_state.message.append(("user", user_input))

    # question = input("What would you like to learn about? (type 'q' to quit): ")
    competencies = retriever.invoke(user_input)
    # result = chain.invoke({
    #     "competencies": competencies, 
    #     "question": user_input
    # })
    # competencies = retriever.invoke(user_input)

    if not competencies:
        # fallback mode
        result = chain_fallback.invoke({
            "question": user_input
        })
    else:
        # normal mode
        result = chain_normal.invoke({
            "competencies": competencies,
            "question": user_input
        })
    # Add assistant message to history
    st.session_state.message.append(("assistant", result))

# Display chat history
for role, message in st.session_state.message:
    with st.chat_message(role):
        st.markdown(message)