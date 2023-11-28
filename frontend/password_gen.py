import streamlit_authenticator as stauth

def main():
    hashed_passwords = stauth.Hasher(['abc', 'def', 'ghi', 'jkl', 'mno']).generate()
    return hashed_passwords


if __name__ == '__main__':
    print(main())