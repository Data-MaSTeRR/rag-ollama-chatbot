# RAG 기반 은행 AI검색기 프로젝트

## 개요
이 프로젝트는 RAG(Retrieval-Augmented Generation) 기반으로 PDF에 저장된 금융정책과 금융상품설명서 등을 검색하고 질문에 대한 답변을 생성하는 챗봇입니다.

## 기능
- PDF 문서에서 정보를 추출 및 청킹
- 벡터 검색 DB를 통해 관련 내용을 검색
- 검색된 정보를 바탕으로 답변 생성

## 설치
```bash
pip install -r requirements.txt
``` 

## 실행
```bash
ollama pull qwen2.5:1.5b
ollama serve
streamlit run app.py
``` 
