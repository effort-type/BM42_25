from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_community.document_transformers import LongContextReorder
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
import os
import shutil

from langchain_qdrant import QdrantVectorStore
from langchain_qdrant import FastEmbedSparse, RetrievalMode


def load_env():
    load_dotenv('.env')

    os.getenv("LANGCHAIN_TRACING_V2")
    os.getenv("LANGCHAIN_ENDPOINT")
    os.getenv("LANGCHAIN_API_KEY")

    os.getenv("OPENAI_API_KEY")
    os.getenv("ANTHROPIC_API_KEY")
    os.getenv("GOOGLE_API_KEY")
    os.getenv("UPSTAGE_API_KEY")

    os.getenv("LM_URL")
    os.getenv("LM_LOCAL_URL")


def docs_load():
    try:
        loader = TextLoader("corpus/정시 모집요강(동의대) 전처리 결과.txt", encoding="utf-8").load()
        return loader
    except FileNotFoundError:
        print("파일을 찾을 수 없습니다. 경로를 확인하세요.")
        return []
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")
        return []


def rc_text_split(corpus):
    """
    RecursiveCharacterTextSplitter를 사용하여 문서를 분할하도록 하는 함수
    :param corpus: 전처리 완료된 말뭉치
    :return: 분리된 청크
    """

    # 청크 사이즈 선택
    chunk_size_number = input("chunk_size를 선택해주세요. 기본값은 1500입니다.\n"
                              "1: 1500\n"
                              "2: 2000\n"
                              "3: 2500\n"
                              "4: 3000\n"
                              "5: 3500\n"
                              "6: 4000\n\n"
                              "선택 번호: ")

    chunk_size_checker = {
        '1': 1500,
        '2': 2000,
        '3': 2500,
        '4': 3000,
        '5': 3500,
        '6': 4000
    }

    chunk_size = chunk_size_checker.get(chunk_size_number, 1500)

    # 오버랩 사이즈 선택
    overlap_size_number = input("chunk_overlap를 선택해주세요. 기본값은 0입니다.\n"
                                "1: 0\n"
                                "2: 100\n"
                                "3: 200\n"
                                "4: 300\n"
                                "5: 400\n"
                                "6: 500\n\n"
                                "선택 번호: ")

    overlap_size_checker = {
        '1': 0,
        '2': 100,
        '3': 200,
        '4': 300,
        '5': 400,
        '6': 500
    }

    overlap_size = overlap_size_checker.get(overlap_size_number, 0)

    rc_text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        separators=["---", "\n\n", "\n"],
        chunk_size=chunk_size,
        chunk_overlap=overlap_size,
        model_name="gpt-4o"  # o200k_base
        # model_name="gpt-4"  # cl100k_base
    )

    text_documents = rc_text_splitter.split_documents(corpus)

    return text_documents


def embed_text(text_documents):
    model_name = "jhgan/ko-sroberta-multitask"
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': True}
    model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    return model


def document_embedding_basic(docs, model, save_directory: str):
    """
    Chroma 벡터저장소를 사용하여 문서를 임베딩하고, BM25Retriever의 기본적인 구조를 통해 문서를 키워드 위주의 임베딩을 진행하여 저장하는 함수
    :param model: 임베딩 모델 종류
    :param save_directory: 벡터저장소 저장 경로
    :param docs: 분할된 문서
    :return: 벡터저장소, BM25(기본)저장소
    """

    print("\n잠시만 기다려주세요.\n\n")

    # 벡터저장소가 이미 존재하는지 확인
    if os.path.exists(save_directory):
        shutil.rmtree(save_directory)
        print(f"디렉토리 {save_directory}가 삭제되었습니다.\n")

    print("문서 벡터화를 시작합니다. ")
    db = Chroma.from_documents(docs, model, persist_directory=save_directory)

    bm25_sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
    bm42_sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm42-all-minilm-l6-v2-attentions")

    # 여기에 BM25, BM42 추가하기
    bm25_db = QdrantVectorStore.from_documents(
        docs,
        embedding=db,
        sparse_embedding=bm25_sparse_embeddings,
        location=":memory:",
        collection_name="db_bm42",
        retrieval_mode=RetrievalMode.SPARSE,
        # retrieval_mode=RetrievalMode.DENSE,
    )

    bm42_db = QdrantVectorStore.from_documents(
        docs,
        embedding=db,
        sparse_embedding=bm42_sparse_embeddings,
        location=":memory:",
        collection_name="db_bm42",
        retrieval_mode=RetrievalMode.SPARSE,
    )

    print("새로운 Chroma 데이터베이스가 생성되었습니다.\n")

    return db, bm25_db, bm42_db


def document_embedding_kiwi(docs, model, save_directory: str):
    """
    Chroma 벡터저장소를 사용하여 문서를 임베딩하고, BM25Retriever에 한글 형태소 분석기(Kiki)를 통해 문서를 키워드 위주의 임베딩을 진행하여 저장하는 함수
    :param model: 임베딩 모델 종류
    :param save_directory: 벡터저장소 저장 경로
    :param docs: 분할된 문서
    :return: 벡터저장소, BM25(Kiwi 한글 형태소 분석기)저장소
    """

    print("\n잠시만 기다려주세요.\n\n")

    # 벡터저장소가 이미 존재하는지 확인
    if os.path.exists(save_directory):
        shutil.rmtree(save_directory)
        print(f"디렉토리 {save_directory}가 삭제되었습니다.\n")

    print("문서 벡터화를 시작합니다. ")
    db = Chroma.from_documents(docs, model, persist_directory=save_directory)

    # bm_db = BM25Retriever.from_documents(
    #     docs,
    #     preprocess_func=self.kiwi_tokenize
    # )
    print("새로운 Chroma 데이터베이스가 생성되었습니다.\n")

    return db


def chat_llm():
    """
    채팅에 사용되는 거대언어모델 생성 함수
    :return: 답변해주는 거대언어모델
    """
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
    )

    return llm


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


def reorder_documents(docs):
    # 재정렬
    reordering = LongContextReorder()
    reordered_docs = reordering.transform_documents(docs)
    combined = format_docs(reordered_docs)

    return combined


def db_qna_ensemble(llm, db, query):
    """
    llm, bm_db, db, query
    BM25Retriever와 Chroma 벡터스토어를 앙상블하여 문서 검색 후 적절한 답변을 찾아서 답하도록 하는 함수
    :param llm: 거대 언어 모델
    :param bm_db: BM25Retriever
    :param db: 벡터스토어
    :param query: 사용자 질문
    """
    db = db.as_retriever(
        search_kwargs={'k': 2},
    )
    # bm_db.k = 1  # BM25Retriever의 검색 결과 개수를 3로 설정
    #
    # # 앙상블 retriever를 초기화합니다.
    # ensemble_retriever = EnsembleRetriever(
    #     retrievers=[bm_db, db],
    #     weights=[0.3, 0.7],
    #     search_type="mmr",
    # )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a specialized AI for question-and-answer tasks.
                You must answer questions based solely on the Context provided.
                For questions about predicting successful applicants, base your answers on data from either the initial successful applicants or the final enrolled students.
                If no Context is provided, you must instruct to inquire at "https://ipsi.deu.ac.kr/main.do".

                Context: {context}
                """,
            ),
            ("human", "Question: {question}"),
        ]
    )

    chain = {
                # "context": ensemble_retriever | RunnableLambda(reorder_documents),
                "context": db | RunnableLambda(reorder_documents),
                "question": RunnablePassthrough()
            } | prompt | llm | StrOutputParser()

    response = chain.invoke(query)

    if not isinstance(llm, ChatOpenAI):
        print("\n\n{}".format(response))

    return response


def db_qna_ensemble_2(llm, db, bm25_db, bm42_db, query):
    """
    llm, bm_db, db, query
    BM25Retriever와 Chroma 벡터스토어를 앙상블하여 문서 검색 후 적절한 답변을 찾아서 답하도록 하는 함수
    :param llm: 거대 언어 모델
    :param bm_db: BM25Retriever
    :param db: 벡터스토어
    :param query: 사용자 질문
    """
    db = db.as_retriever(
        search_kwargs={'k': 2},
    )

    bm25 = bm25_db.as_retriever(
        search_kwargs={'k': 2},
    )

    bm42 = bm42_db.as_retriever(
        search_kwargs={'k': 2},
    )

    # bm_db.k = 1  # BM25Retriever의 검색 결과 개수를 3로 설정
    #
    # # 앙상블 retriever를 초기화합니다.
    # ensemble_retriever = EnsembleRetriever(
    #     retrievers=[bm25, db],
    #     weights=[0.7, 0.3],
    #     search_type="mmr",
    # )

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm42, db],
        weights=[0.3, 0.7],
        search_type="mmr",
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a specialized AI for question-and-answer tasks.
                You must answer questions based solely on the Context provided.
                For questions about predicting successful applicants, base your answers on data from either the initial successful applicants or the final enrolled students.
                If no Context is provided, you must instruct to inquire at "https://ipsi.deu.ac.kr/main.do".

                Context: {context}
                """,
            ),
            ("human", "Question: {question}"),
        ]
    )

    chain = {
                "context": ensemble_retriever | RunnableLambda(reorder_documents),
                "question": RunnablePassthrough()
            } | prompt | llm | StrOutputParser()

    response = chain.invoke(query)

    if not isinstance(llm, ChatOpenAI):
        print("\n\n{}".format(response))

    return response


def run():
    load_env()
    chunk = docs_load()
    text_documents = rc_text_split(chunk)
    print(text_documents)

    embedding_model = embed_text(text_documents)
    db, bm25_db, bm42_db = document_embedding_basic(text_documents, embedding_model, "chromadb_basic")
    llm = chat_llm()

    # 반복문 돌리기
    query = input("질문을 입력하세요: ")
    # db_qna_ensemble(llm, db, query)
    db_qna_ensemble_2(llm, db, bm25_db, bm42_db, query)


if __name__ == '__main__':
    run()
