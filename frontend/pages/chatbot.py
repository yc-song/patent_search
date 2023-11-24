import streamlit as st
import requests
import json


url = "http://localhost:5000"
st.title('Patent Chatbot Engine')

if "messages" not in st.session_state:
    st.session_state.messages = []

# st.session_state.messages로 현재 세션의 대화 기록 관리
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# streamlit은 새로고침이 일어날 때마다 script 전체를 다시 실행하는 것으로 알고 있음
# 그렇다면, input이 들어오고 답을 받아서 화면에 보여준 다음에는 다시
# 처음부터 rendering해서 또다시 input을 기다리는 것일 수도.
# 만약에 chat_input이 아니라 stt input을 받는다면 어떻게 해야 할지 고민.
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        inputs = {"messages": st.session_state.messages}
        res = requests.post(f"{url}/api/chat",
                            json=json.dumps(inputs))

        full_response = ""
        message_placeholder = st.empty()
        if res.status_code == 200:
            res = eval(res.content.decode('utf-8'))['response']

            for token in res.split(" "):
                full_response += f" {token}"
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        else:
            message_placeholder.markdown("Error: Unable to fetch data from the API.")
    st.session_state.messages.append({"role": "assistant", "content": full_response})
