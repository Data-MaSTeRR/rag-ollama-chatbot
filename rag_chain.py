from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from rerank import rerank_documents

# LLM (예: Llama 모델 설정)
llm = ChatOllama(model="qwen2.5", temperature=0)


# 문서 리스트를 텍스트로 변환
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


# RAG 체인 실행
def build_rag_chain_with_rerank(vectorstore, query):
    # Step 1: 문서 검색
    retrieved_docs = vectorstore.as_retriever(search_kwargs={"k": 3}).invoke(query)

    # Step 2: 리랭크 모델 적용
    ranked_docs = rerank_documents(query, retrieved_docs)

    # Step 3: 생성 모델에 전달할 문서 선택 (상위 문서들만 사용)
    top_docs = ranked_docs[:5]

    # Step 4: 문서를 텍스트로 변환
    context = format_docs(top_docs)

    # Step 5: RAG 체인 설정 및 실행
    prompt_template = """
    당신은 한국어로 질문에 답변하는 전문가입니다. 다음 문서 내용을 바탕으로 **질문에 대한 구체적인 답변**을 해주세요. **질문에 관련된 정보**만 제공하고, **관련 없는 정보는 제외**하세요.

    문서 내용:
    {context}

    질문:
    {question}

    답변:
    """

    prompt = ChatPromptTemplate.from_template(prompt_template)

    # Step 6: Runnable 체인으로 구성
    rag_chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    # Return the generated answer (string)
    return rag_chain.invoke({"context": context, "question": query})

