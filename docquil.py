#!/usr/bin/env python

# Import required modules
import os
from dotenv import load_dotenv, find_dotenv
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import GoogleDriveLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv(find_dotenv())
folder_id = os.getenv("FOLDER_ID")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Retrieve documents from Google Drive
loader = GoogleDriveLoader(folder_id=folder_id, recursive=False)
docs = loader.load()

# Split documents into sentences
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=4000, chunk_overlap=0, separators=[" ", ",", "\n"]
)

texts = text_splitter.split_documents(docs)
embeddings = OpenAIEmbeddings()
db = Chroma.from_documents(texts, embeddings)
retriever = db.as_retriever()

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

while True:
    question = input("Ask me something: ")
    answer = qa.run(question)
    print(answer)
