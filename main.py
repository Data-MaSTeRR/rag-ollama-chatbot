from document_processing import process_documents  # 올바른 함수 이름으로 변경
from vectorstore import init_vectorstore
from rag_chain import build_rag_chain_with_rerank

def main():
    # Step 1: 문서 처리 및 임베딩 생성
    processed_docs, embeddings = process_documents('pdf_data')

    # Step 2: 벡터 DB 초기화 및 임베딩 저장
    vectorstore = init_vectorstore(processed_docs, embeddings)

    # Step 3: 질문에 대한 응답 생성
    #query = "고향사랑기부금 세액공제 적용 한도 확대된다고 하는데 얼마나 확대되는거야?"
    query = "버팀목 전세자금대출이 뭐야?"
    response = build_rag_chain_with_rerank(vectorstore, query)

    # Step 4: 결과 출력
    print(response)

if __name__ == "__main__":
    main()
