import streamlit as st
from datetime import datetime
import openai

# Set your OpenAI API key
openai.api_key = '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'

# Create a title
st.title('四柱推命占い')

# Create a date input widget
min_date = datetime(1900, 1, 1)
dob = st.date_input('生年月日を選択してください',min_value=min_date)

# Create a time input widget
time_of_birth = st.text_input('生まれた時間を入力してください')

# Button to submit information
if st.button('占う'):
    # Concatenate the date and time into a single string
    dob_string = datetime.strftime(dob, '%Y-%m-%d') + ' ' + time_of_birth
    
    # Use the date and time of birth as input to the OpenAI API
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo-0613",
      messages=[
        {"role": "system", "content": "あなたは優秀な占い師です。"},
        {"role": "user", "content": f"以下の生年月日と生まれた時間から、四柱推命で恋愛、勉学、仕事について占ってください\n{dob_string}"},
        ],
      temperature=0.5,
      max_tokens=500
    )
        
    st.write(response.choices[0]['message']['content'])
