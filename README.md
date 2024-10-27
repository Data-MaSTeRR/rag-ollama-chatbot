# RAG 기반 소비자 분쟁조정 사례집 QA 챗봇

## 개요
이 프로젝트는 RAG(Retrieval-Augmented Generation) 기반으로 PDF에 저장된 소비자 분쟁조정 사례를 검색하고 질문에 대한 답변을 생성하는 챗봇입니다.

## 기능
- PDF 문서에서 사례 정보를 추출 및 청킹
- 벡터 검색 DB를 통해 관련 사례를 검색
- 검색된 사례 정보를 바탕으로 답변 생성

## 설치
```bash
pip install -r requirements.txt