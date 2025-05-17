from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

df = pd.read_csv("/Users/bright/Desktop/streamlit_llm/openai_result - competencies.csv")
embedding = OllamaEmbeddings(model="mxbai-embed-large")

# one time vectorize
df_location = "./langchain_db"
add_documents = not os.path.exists(df_location)

if add_documents:
    documents = []
    ids = []
    for i, row in df.iterrows():
        # รวม competency กับ description เพื่อให้เข้าใจความหมายมากขึ้น
        text = f"{row['competency']}: {row['description']}"
        
        doc = Document(
            page_content=text,
            metadata={"id": str(i), "competency": row['competency']}  # สามารถใช้ metadata ตอน retrieve ได้
        )
        documents.append(doc)
        ids.append(str(i))  # keep unique ID if necessary

vectordb = Chroma(
    persist_directory=df_location,
    embedding_function=embedding
)

if add_documents:
    vectordb.add_documents(documents=documents, ids = ids)
    # vectordb.persist()

retriever = vectordb.as_retriever(
    search_kwargs = {"k": 5}

)
