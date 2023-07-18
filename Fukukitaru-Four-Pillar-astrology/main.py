import streamlit as st
from datetime import datetime
import openai
import os

openai.api_key = os.getenv('OPENAI_API_KEY')

from langchain.callbacks import get_openai_callback

# Embedding用
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
# Vector 格納 / FAISS
from langchain.vectorstores import FAISS
# テキストファイルを読み込む
from langchain.document_loaders import TextLoader
# Q&A用Chain
from langchain.chains.question_answering import load_qa_chain
# ChatOpenAI GPT 3.5
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.prompts.chat import (
    # メッセージテンプレート
    ChatPromptTemplate,
    # System メッセージテンプレート
    SystemMessagePromptTemplate,
    # assistant メッセージテンプレート
    AIMessagePromptTemplate,
    # user メッセージテンプレート
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    # それぞれ GPT-3.5-turbo API の assistant, user, system role に対応
    AIMessage,
    HumanMessage,
    SystemMessage
)


loader = TextLoader(r'C:\Users\papa\Documents\Git-folder\Fukukitaru-Four-Pillar-astrology/Fukukitaru_Serihu_v2.txt', encoding='utf-8')
documents = loader.load()
text_splitter = CharacterTextSplitter(separator="\n",chunk_size=700, chunk_overlap=0,length_function=len)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

db = FAISS.from_documents(docs, embeddings)

query = "content:You are Machikanefukkital, a member of Uma Musume. Please answer the questions in a tone that is typical of Machikanefukkital. Answer in Japanese. user:四柱推命で恋愛、勉学、仕事について占ってください"

embedding_vector = embeddings.embed_query(query)
docs_and_scores = db.similarity_search_by_vector(embedding_vector)


# load_qa_chainを準備
chain = load_qa_chain(ChatOpenAI(model="gpt-3.5-turbo-0613",temperature=0,max_tokens=900), chain_type="stuff")

# Create a title
st.title('フクキタル四柱推命占い(非公式)')

# Create a date input widget
min_date = datetime(1900, 1, 1)
dob = st.date_input('生年月日を選択してください',min_value=min_date)

# Create a time input widget
time_of_birth = st.text_input('生まれた時間を入力してください')


# Button to submit information
if st.button('占う'):
    # Concatenate the date and time into a single string
    dob_string = datetime.strftime(dob, '%Y-%m-%d') + ' ' + time_of_birth
    
    prompt = f'''
    content:You are Machikanefukkital, a member of Uma Musume. Please answer the questions in a tone that is typical of Machikanefukkital. Answer in Japanese. 
    user:以下の生年月日と生まれた時間から、四柱推命で恋愛、勉学、仕事について占ってください
    {dob_string}
    '''     
    # Create a placeholder for the response
    placeholder = st.empty()
    placeholder.text('占い中...')
      
   
    
    with get_openai_callback() as cb: 
      # 質問応答の実行
      placeholder.write(chain({"input_documents": docs_and_scores, "question": prompt}, return_only_outputs=True))
      print(prompt)
      print(cb)    
            
    # Update the placeholder with the response
    #placeholder.write(response.choices[0]['message']['content'])