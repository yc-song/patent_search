import requests
import json
import streamlit as st
from streamlit_chat import message


url = "http://localhost:5000"
# if st.session_state["authentication_status"]:
st.title('특허 챗봇 엔진')
st.markdown('서치를 한 이후의 결과에 대한 세부 질문 답변을 위한 챗봇입니다.')
number = st.number_input("서치 결과에서 관심 있는 결과의 번호를 입력해 주세요.", value=None, min_value=1, max_value=20, placeholder="숫자를 입력해 주세요.")
st.write('입력한 숫자 ', number)
if "messages" not in st.session_state:
    st.session_state.messages = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

# with st.form('form', clear_on_submit=True):
#     user_input = st.text_input('You: ', '', key='input')
#     st.session_state.messages.append({"role": "user", "content": user_input})
#     submitted = st.form_submit_button('Send')
if number:
    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})


        inputs = {"messages": st.session_state.messages, "number": number}

        res = requests.post(f"{url}/api/chat",
                            json=json.dumps(inputs))
        full_response = ""
        message_placeholder = st.empty()
        if res.status_code == 200:
            res = eval(res.content.decode('utf-8'))['response']

            for token in res.split(" "):
                full_response += f" {token}"
        st.session_state.past.append(prompt)
        st.session_state.generated.append(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})


    if st.session_state['generated']:
        for i in range(len(st.session_state['generated']) - 1, -1, -1):
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))
# elif st.session_state["authentication_status"] == False:
#     st.error('Username/password is incorrect')
# elif st.session_state["authentication_status"] == None:
#     st.warning('Please enter your username and password')

