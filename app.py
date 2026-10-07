"""DATA 폴더의 PDF를 사용하는 간단한 RAG 챗봇입니다.

실행 방법:
    uv run streamlit run app.py
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


# app.py가 있는 프로젝트 최상단을 기준으로 DATA 폴더를 찾습니다.
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "DATA"


def load_openai_api_key() -> str | None:
    """로컬 .env 또는 Streamlit Cloud Secrets에서 API 키를 읽습니다."""
    # 로컬에서는 프로젝트의 .env 파일을 사용합니다.
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("OPENAI_API_KEY")

    # Streamlit Cloud에서는 Settings > Secrets에 저장한 값을 사용합니다.
    if not api_key:
        try:
            api_key = st.secrets.get("OPENAI_API_KEY")
            if not api_key:
                openai_section = st.secrets.get("openai", {})
                api_key = openai_section.get("api_key")
        except (FileNotFoundError, KeyError):
            # 로컬에서 Secrets 파일이 없어도 .env만 있으면 정상 동작합니다.
            api_key = None

    if api_key:
        # LangChain OpenAI 클래스가 표준 환경변수를 사용할 수 있게 설정합니다.
        os.environ["OPENAI_API_KEY"] = str(api_key).strip()
        return os.environ["OPENAI_API_KEY"]
    return None


def load_pdf_documents() -> list[Document]:
    """DATA 폴더의 모든 PDF를 페이지별 LangChain Document로 읽습니다."""
    documents: list[Document] = []

    for pdf_path in sorted(DATA_DIR.glob("*.pdf")):
        reader = PdfReader(str(pdf_path))

        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue

            # 출처 표시를 위해 파일명과 페이지 번호를 메타데이터로 보관합니다.
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": pdf_path.name,
                        "page": page_number,
                    },
                )
            )

    return documents


@st.cache_resource(show_spinner="PDF를 읽고 검색 인덱스를 만드는 중입니다...")
def create_vector_store() -> tuple[InMemoryVectorStore, int, int]:
    """PDF를 청크로 나누고 OpenAI 임베딩으로 메모리 벡터 저장소를 만듭니다."""
    page_documents = load_pdf_documents()
    if not page_documents:
        raise FileNotFoundError("DATA 폴더에서 읽을 수 있는 PDF 문서를 찾지 못했습니다.")

    # 긴 페이지를 검색하기 좋은 크기의 겹치는 청크로 나눕니다.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    chunks = splitter.split_documents(page_documents)

    # 지정한 임베딩 모델로 InMemoryVectorStore를 구성합니다.
    # tiktoken 인코딩 파일을 별도로 내려받지 않고 OpenAI API에 직접 텍스트를 전달합니다.
    # 이렇게 하면 제한된 네트워크 환경에서도 임베딩 초기화가 막히지 않습니다.
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        tiktoken_enabled=False,
        check_embedding_ctx_length=False,
    )
    vector_store = InMemoryVectorStore(embedding=embeddings)
    vector_store.add_documents(documents=chunks)

    return vector_store, len(page_documents), len(chunks)


def make_evidence_sentence(text: str, max_length: int = 240) -> str:
    """검색된 청크에서 화면에 보여줄 짧은 근거 문장을 만듭니다."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?。！？])\s+", cleaned)
    evidence = next((sentence for sentence in sentences if sentence.strip()), cleaned)
    if len(evidence) > max_length:
        return evidence[: max_length - 1].rstrip() + "…"
    return evidence


def answer_question(vector_store: InMemoryVectorStore, question: str):
    """관련 문서를 검색하고, 검색 결과만 근거로 답변을 생성합니다."""
    # 검색 결과를 출처 표시에 재사용하기 위해 직접 검색합니다.
    retrieved_documents = vector_store.similarity_search(question, k=4)
    context_parts = []
    for index, document in enumerate(retrieved_documents, start=1):
        source = document.metadata.get("source", "알 수 없는 파일")
        page = document.metadata.get("page", "?")
        context_parts.append(
            f"[근거 {index}] 파일명: {source}, 페이지: {page}\n"
            f"{document.page_content}"
        )
    context = "\n\n".join(context_parts)

    system_prompt = """당신은 제공된 PDF 문서만 근거로 답변하는 한국어 RAG 챗봇입니다.

반드시 지킬 규칙:
1. 아래 [문서 근거]에 실제로 있는 내용만 사용하세요.
2. 문서에서 답을 확인할 수 없으면 정확히 '문서에서 확인할 수 없습니다.'라고 답하세요.
3. 일반 상식, 추측, 문서에 없는 숫자나 조건을 추가하지 마세요.
4. 답변은 초보자가 이해하기 쉬운 한국어로 간결하게 작성하세요.
5. 답변 본문에는 파일명이나 페이지를 임의로 만들지 마세요. 출처는 화면에서 별도로 표시합니다.
"""
    user_prompt = f"""[문서 근거]
{context}

[질문]
{question}
"""

    # 구버전 RetrievalQA나 ConversationChain 대신 최신 메시지 기반 호출을 사용합니다.
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    return str(response.content), retrieved_documents


def show_sources(documents: list[Document]) -> None:
    """답변 아래에 검색된 출처 파일명과 근거 문장을 표시합니다."""
    st.markdown("#### 출처 및 근거")
    seen: set[tuple[str, int]] = set()

    for document in documents:
        source = str(document.metadata.get("source", "알 수 없는 파일"))
        page = int(document.metadata.get("page", 0))
        source_key = (source, page)
        if source_key in seen:
            continue
        seen.add(source_key)
        evidence = make_evidence_sentence(document.page_content)
        st.markdown(f"- **{source}** (p. {page}) - {evidence}")


def main() -> None:
    """Streamlit 화면을 구성합니다."""
    st.set_page_config(page_title="공무원 여비 RAG 챗봇", page_icon="📚")
    st.title("📚 공무원 여비 RAG 챗봇")
    st.caption("DATA 폴더의 PDF 문서만 근거로 답변합니다.")

    if not load_openai_api_key():
        st.error(
            "로컬에서는 .env에 OPENAI_API_KEY를 입력하고, "
            "Streamlit Cloud에서는 Settings > Secrets에 API 키를 등록해 주세요."
        )
        st.stop()

    if not DATA_DIR.exists():
        st.error(f"DATA 폴더를 찾을 수 없습니다: {DATA_DIR}")
        st.stop()

    try:
        vector_store, page_count, chunk_count = create_vector_store()
    except Exception as error:
        st.error(f"문서 인덱스를 만드는 중 오류가 발생했습니다: {error}")
        st.stop()

    with st.sidebar:
        st.subheader("문서 정보")
        st.write(f"읽은 페이지: {page_count}개")
        st.write(f"검색 청크: {chunk_count}개")
        st.write("임베딩: text-embedding-3-small")
        st.write("답변: gpt-4o-mini")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                show_sources(message["sources"])

    question = st.chat_input("문서에 대해 질문해 주세요")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("문서에서 관련 내용을 찾는 중입니다..."):
            try:
                answer, source_documents = answer_question(vector_store, question)
            except Exception as error:
                st.error(f"답변을 생성하는 중 오류가 발생했습니다: {error}")
                return

        st.markdown(answer)
        show_sources(source_documents)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": source_documents,
            }
        )


if __name__ == "__main__":
    main()
