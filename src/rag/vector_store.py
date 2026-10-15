"""Vector store: Pinecone (serverless) when PINECONE_API_KEY is set, otherwise in-memory fallback."""
import time
from functools import lru_cache
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import (EMBED_DIM, EMBED_MODEL, KB_DIR, PINECONE_API_KEY, PINECONE_CLOUD,
                        PINECONE_INDEX, PINECONE_REGION)
from src.observability import log


@lru_cache(maxsize=1)
def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL, encode_kwargs={"normalize_embeddings": True})


def load_chunks():
    docs = []
    for p in sorted(KB_DIR.glob("*.md")):
        docs.append(Document(page_content=p.read_text(), metadata={"source": p.name, "topic": p.stem}))
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=60).split_documents(docs)
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = f"{c.metadata['source']}-{i}"
    return chunks


def _pinecone_index_ready(pc):
    from pinecone import ServerlessSpec
    if PINECONE_INDEX not in [i.name for i in pc.list_indexes()]:
        pc.create_index(name=PINECONE_INDEX, dimension=EMBED_DIM, metric="cosine",
                        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION))
        while not pc.describe_index(PINECONE_INDEX).status["ready"]:
            time.sleep(1)
        return True  # newly created
    return False


def ingest():
    """Upsert KB chunks (idempotent: deterministic ids). Returns the vector store."""
    chunks, emb = load_chunks(), get_embeddings()
    if PINECONE_API_KEY:
        from pinecone import Pinecone
        from langchain_pinecone import PineconeVectorStore
        _pinecone_index_ready(Pinecone(api_key=PINECONE_API_KEY))
        vs = PineconeVectorStore.from_documents(chunks, emb, index_name=PINECONE_INDEX, ids=[c.metadata["chunk_id"] for c in chunks])
        log.info("Upserted %d chunks to Pinecone index %s", len(chunks), PINECONE_INDEX)
        return vs
    from langchain_core.vectorstores import InMemoryVectorStore
    vs = InMemoryVectorStore(emb)
    vs.add_documents(chunks)
    log.info("Pinecone key missing -> using in-memory store with %d chunks", len(chunks))
    return vs


@lru_cache(maxsize=1)
def get_vector_store():
    if PINECONE_API_KEY:
        from pinecone import Pinecone
        from langchain_pinecone import PineconeVectorStore
        if _pinecone_index_ready(Pinecone(api_key=PINECONE_API_KEY)):
            return ingest()
        return PineconeVectorStore(index_name=PINECONE_INDEX, embedding=get_embeddings())
    return ingest()


if __name__ == "__main__":
    ingest()
