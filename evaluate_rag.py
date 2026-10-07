"""test_dataset.json의 질문을 실제 RAG 파이프라인으로 검증합니다."""

from __future__ import annotations

import json
from pathlib import Path
import re

from dotenv import load_dotenv
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import app


PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")

    # 앱과 동일한 문서 분할 및 임베딩 설정으로 테스트 인덱스를 만듭니다.
    pages = app.load_pdf_documents()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    ).split_documents(pages)
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        tiktoken_enabled=False,
        check_embedding_ctx_length=False,
    )
    vector_store = InMemoryVectorStore(embedding=embeddings)
    vector_store.add_documents(chunks)

    test_cases = json.loads((PROJECT_ROOT / "test_dataset.json").read_text(encoding="utf-8"))
    passed = 0
    results = []
    print(f"INDEX_READY pages={len(pages)} chunks={len(chunks)}")

    for case in test_cases:
        answer, source_documents = app.answer_question(vector_store, case["question"])
        refusal_detected = "문서에서 확인할 수 없습니다." in answer
        normalized_answer = re.sub(r"\s+", " ", answer).lower()
        expected_keywords = case.get("expected_keywords", [])
        matched_keywords = [
            keyword
            for keyword in expected_keywords
            if keyword.lower() in normalized_answer
        ]
        keyword_check = len(matched_keywords) >= case.get("min_keyword_matches", 0)
        citation_check = all(
            document.metadata.get("source") and document.metadata.get("page")
            for document in source_documents
        )
        refusal_check = refusal_detected == (not case["answerable_from_documents"])
        passed_case = refusal_check and keyword_check and citation_check
        passed += int(passed_case)
        compact_answer = " ".join(answer.split())[:180]
        status = "PASS" if passed_case else "CHECK"
        print(
            f"{status} ID={case['id']} "
            f"EXPECTED={'ANSWER' if case['answerable_from_documents'] else 'REFUSE'} "
            f"KEYWORDS={len(matched_keywords)}/{case.get('min_keyword_matches', 0)} "
            f"CITATIONS={'OK' if citation_check else 'FAIL'} "
            f"SOURCES={len(source_documents)} ANSWER={compact_answer}"
        )
        results.append(
            {
                "id": case["id"],
                "question": case["question"],
                "passed": passed_case,
                "refusal_check": refusal_check,
                "keyword_check": keyword_check,
                "matched_keywords": matched_keywords,
                "citation_check": citation_check,
                "sources": [
                    {
                        "file": document.metadata.get("source"),
                        "page": document.metadata.get("page"),
                    }
                    for document in source_documents
                ],
                "answer": answer,
            }
        )

    print(f"SUMMARY passed={passed} total={len(test_cases)}")
    report = {
        "total": len(test_cases),
        "passed": passed,
        "failed": len(test_cases) - passed,
        "results": results,
    }
    (PROJECT_ROOT / "evaluation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("REPORT_WRITTEN evaluation_report.json")


if __name__ == "__main__":
    main()
