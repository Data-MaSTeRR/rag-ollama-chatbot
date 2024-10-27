import streamlit as st
from document_processing import process_documents
from vectorstore import init_vectorstore
from rag_chain import build_rag_chain_with_rerank

# Streamlit 앱
def main():
    st.title("RAG 기반 챗봇")

    # 상태 초기화 - PDF 처리 및 벡터 DB 생성
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    # 사이드바에서 PDF 임베딩 생성
    if st.sidebar.button("PDF 처리 및 벡터 DB 생성"):
        with st.spinner("PDF 처리 중..."):
            try:
                processed_docs, embeddings = process_documents('pdf_data')
                vectorstore = init_vectorstore(processed_docs, embeddings)
                st.session_state.vectorstore = vectorstore  # 세션 상태에 저장
                st.sidebar.success("PDF 임베딩 완료!")
            except Exception as e:
                st.sidebar.error(f"벡터 DB 생성 중 오류가 발생했습니다: {e}")

    # 사용자 질문 입력
    query = st.text_input("질문을 입력하세요:")

    if st.button("답변 받기"):
        if st.session_state.vectorstore is None:
            st.error("먼저 사이드바에서 PDF 처리 및 벡터 DB 생성을 완료해주세요.")
        elif not query.strip():
            st.warning("질문을 입력해 주세요!")
        else:
            st.info(f"질문: {query}")

            # 벡터 DB 검색 및 답변 생성
            with st.spinner("답변 생성 중..."):
                try:
                    response = build_rag_chain_with_rerank(st.session_state.vectorstore, query)
                    st.success("답변이 생성되었습니다!")
                    st.write(response)
                except Exception as e:
                    st.error(f"답변 생성 중 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    main()
