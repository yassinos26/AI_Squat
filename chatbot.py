import openai
import streamlit as st

openai.api_key = "AIzaSyBRJovh1LGCsLN2pH_RYcwWu2itj53UyE8"

st.title("💬 Chatbot Assistant Squat")

user_input = st.text_input("Posez-moi une question sur votre squat !")

print("Chatbot Assistant Squat is running...")  # For debugging purposes
if user_input:
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Tu es une coach sportive spécialisée en analyse biomécanique de squat."},
            {"role": "user", "content": user_input}
        ]
    )
    response = completion.choices[0].message.content
    st.write(f"🤖 {response}")
    print(f"User input: {user_input}")  # For debugging purposes
    print(f"Chatbot response: {response}")  # For debugging purposes