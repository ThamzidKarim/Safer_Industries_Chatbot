import os
from dotenv import load_dotenv

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings

import chainlit as cl
from typing import Optional

from google import genai
from google.genai import types

import json

from langchain.text_splitter import RecursiveCharacterTextSplitter

from chainlit.data.sql_alchemy import SQLAlchemyDataLayer

@cl.data_layer
def get_data_layer():
    conninfo = "postgresql+asyncpg://postgres:Safer123%21@db.dreowbtmqrfkcaxancpv.supabase.co:5432/postgres"
    return SQLAlchemyDataLayer(conninfo=conninfo)


VALID_USERNAME = "Intern"
VALID_PASSWORD = "safer123!"

@cl.password_auth_callback
def password_auth(username: str, password: str):
    if username == VALID_USERNAME and password == VALID_PASSWORD:
        return cl.User(identifier=username)
    return None

# Load env vars and configure Gemini
def setUpGoogleAPI():
    load_dotenv()
    api_key = os.getenv('GEMINI_API_KEY')
    client = genai.Client(api_key=api_key)
    cl.user_session.set('client', client)



# Load JSON file
with open('doc.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
big_text = data['text']

# Split the big text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
)

chunks = text_splitter.create_documents([big_text])
documents = [chunk.page_content for chunk in chunks]

print(f"Loaded and split into {len(documents)} chunks.")


# Load or create the vector DB
def loadVectorDataBase():
    chroma_client = chromadb.PersistentClient(path="./database/")
    db = chroma_client.get_or_create_collection(
        name="si_db", embedding_function=GeminiEmbeddingFunction()
    )
    if db.count() == 0:
        ids = [f"doc_{i}" for i in range(len(documents))]
        metadatas = [{"source": "intern_onboarding"}] * len(documents)

        db.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Added {len(documents)} documents to vector DB")
    else:
        print(f"Vector DB already contains {db.count()} documents")

    cl.user_session.set('db', db)

# Define Gemini embedding wrapper
class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        model = 'models/text-embedding-004'
        client = cl.user_session.get('client')
        response = client.models.embed_content(
            model=model,
            contents=input,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
        return [embedding.values for embedding in response.embeddings]



# Grab top matching chunks for a query
def get_relevant_passages(query, db, n_results=10):
    results = db.query(query_texts=[query], n_results=n_results)
    return results['documents'][0]

# Convert multiple passages into a context string
def convert_passages_to_string(passages):
    return "\n".join(passages)

# Build the final prompt
def make_prompt(query, relevant_text):
    return f"""
You are a helpful assistant answering based only on the provided information. When a user says "Hi" or equivalent, let them know what you can do.

Question: {query}

Relevant context:
{relevant_text}

If the context does not answer the question, respond with "OUT OF CONTEXT".
"""

@cl.on_chat_start
async def start():
    setUpGoogleAPI()
    loadVectorDataBase()
    cl.user_session.set("history", [])
    app_user = cl.user_session.get("user")
    await cl.Message(f"Hello!").send()

@cl.on_chat_resume
async def on_chat_resume(thread):
    pass

@cl.on_message
async def main(message: cl.Message):
    client = cl.user_session.get('client')
    db = cl.user_session.get('db')
    history = cl.user_session.get("history", [])

    question = message.content
    passages = get_relevant_passages(question, db)
    context = convert_passages_to_string(passages)
    prompt = make_prompt(question, context)

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt],
            config=types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=150,
            )
        )
        bot_reply = response.text.strip()
    except Exception as e:
        bot_reply = f"Gemini error: {str(e)}"

    history.append({"user": question})
    history.append({"bot": bot_reply})
    cl.user_session.set("history", history)

    await cl.Message(content=bot_reply).send()