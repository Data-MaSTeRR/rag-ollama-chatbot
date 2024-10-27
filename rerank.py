from sentence_transformers import CrossEncoder

# Cross-Encoder 로드
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')

# 리랭크 함수
def rerank_documents(query, documents):
    # 각 문서와 쿼리 간의 유사도를 평가
    document_texts = [doc.page_content for doc in documents]
    pairs = [(query, doc_text) for doc_text in document_texts]

    # Cross-Encoder를 사용해 유사도 스코어 계산
    scores = cross_encoder.predict(pairs)

    # 유사도 스코어에 따라 문서들 리랭크
    ranked_documents = [doc for _, doc in sorted(zip(scores, documents), reverse=True)]
    return ranked_documents
