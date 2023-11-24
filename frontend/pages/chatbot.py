import requests
import json
import streamlit as st
from streamlit_chat import message


url = "http://localhost:5000"

st.title('Patent Chatbot Engine')

if "messages" not in st.session_state:
    st.session_state.messages = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

with st.form('form', clear_on_submit=True):
    user_input = st.text_input('You: ', '', key='input')
    st.session_state.messages.append({"role": "user", "content": user_input})
    submitted = st.form_submit_button('Send')

if submitted and user_input:
    inputs = {"messages": st.session_state.messages}
    res = requests.post(f"{url}/api/chat",
                        json=json.dumps(inputs))
    full_response = ""
    message_placeholder = st.empty()
    if res.status_code == 200:
        res = eval(res.content.decode('utf-8'))['response']

        for token in res.split(" "):
            full_response += f" {token}"
    st.session_state.past.append(user_input)
    st.session_state.generated.append(full_response)

if st.session_state['generated']:
    for i in range(len(st.session_state['generated']) - 1, -1, -1):
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
        message(st.session_state["generated"][i], key=str(i))