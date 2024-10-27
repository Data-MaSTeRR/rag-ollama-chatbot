from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def init_vectorstore(documents, embeddings):
    # 원래 임베딩 모델 사용
    embeddings_model = HuggingFaceEmbeddings(model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS")

    # 벡터 DB 초기화 및 임베딩 추가
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings_model,
        collection_name="consumer_case_qa",
        persist_directory="./chroma_db"
    )

    return vectorstore

def retrieve_documents(vectorstore, query, k=5):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(query)
