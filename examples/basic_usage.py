"""Basic usage example: ingest a PDF and ask a question."""

from local_rag_stack.config import Settings
from local_rag_stack.pipeline import RAGPipeline

settings = Settings(
    text_embed_model="qwen3-embedding:4b",
    vision_embed_backend="none",  # disable visual embeddings for this example
    rerank_backend="none",
)

pipeline = RAGPipeline(settings)
try:
    doc = pipeline.ingest("path/to/document.pdf")
    print(f"Ingested {doc.document_name} with {len(doc.chunks)} chunks")

    response = pipeline.query("What are the key terms of this contract?")
    print(response.answer)
    for citation in response.citations:
        print(f"  [{citation.source_id}] {citation.document_name} (page {citation.page_number})")
finally:
    pipeline.close()
