from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
import os
import pickle
from glob import glob
from langchain_community.document_loaders import PyPDFLoader


def load_pdf_files(pdf_dir):
    return glob(os.path.join(pdf_dir, '*.pdf'))


def embed_document(text):
    # Hugging Face 임베딩 모델로 임베딩 생성
    embeddings_model = HuggingFaceEmbeddings(model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS")
    embedding = embeddings_model.embed_documents([text])
    return embedding


def process_documents(pdf_dir):
    # PDF 파일 로드
    pdf_files = load_pdf_files(pdf_dir)

    # 캐싱된 임베딩이 있는지 확인
    cache_dir = 'embeddings_cache'
    cache_file = os.path.join(cache_dir, 'embeddings_cache.pkl')

    # 디렉토리 생성 (필요할 경우)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    processed_docs = []  # 초기화

    if os.path.exists(cache_file):
        # 캐시에서 임베딩을 불러옴
        with open(cache_file, 'rb') as f:
            embeddings = pickle.load(f)

        # 캐시된 임베딩에 대한 processed_docs를 생성
        for pdf in pdf_files:
            loader = PyPDFLoader(pdf)
            pages = loader.load()
            text = "\n".join([page.page_content for page in pages])

            doc = Document(page_content=text, metadata={"source": pdf})
            processed_docs.append(doc)
    else:
        embeddings = []
        for pdf in pdf_files:
            # PDF 파일을 PyPDFLoader로 읽어오기
            loader = PyPDFLoader(pdf)
            pages = loader.load()
            text = "\n".join([page.page_content for page in pages])

            # 문서 임베딩 생성
            embedding = embed_document(text)
            embeddings.append(embedding)

            # Document 객체로 변환
            doc = Document(page_content=text, metadata={"source": pdf})
            processed_docs.append(doc)

        # 임베딩 캐싱
        with open(cache_file, 'wb') as f:
            pickle.dump(embeddings, f)

    return processed_docs, embeddings
