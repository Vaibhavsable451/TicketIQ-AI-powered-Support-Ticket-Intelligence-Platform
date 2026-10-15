"""Vector store: Pinecone (serverless) when PINECONE_API_KEY is set, otherwise in-memory / TF-IDF fallback."""
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


class TfidfStore:
    """Dependency-free fallback retriever used when sentence-transformers is not installed (e.g. Streamlit Cloud)."""

    def __init__(self, chunks):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.chunks = chunks
        self.vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self.M = self.vec.fit_transform([c.page_content for c in chunks])

    def similarity_search(self, query, k=3):
        from sklearn.metrics.pairwise import cosine_similarity
        sims = cosine_similarity(self.vec.transform([query]), self.M)[0]
        return [self.chunks[i] for i in sims.argsort()[::-1][:k]]


def _embeddings_available() -> bool:
    """Check without triggering the full (broken) import chain."""
    try:
        from importlib.util import find_spec
        # torch is required for sentence-transformers to function;
        # if it's missing we already know embeddings won't work.
        if find_spec("torch") is None:
            return False
        if find_spec("sentence_transformers") is None:
            return False
        # Both packages exist – do a real import of torch to be sure it loads.
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def load_chunks():
    if not list(KB_DIR.glob("*.md")):
        from src.data.collect import write_knowledge_base
        write_knowledge_base()
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
    chunks = load_chunks()
    if not _embeddings_available():
        log.info("sentence-transformers not installed -> using TF-IDF retriever")
        return TfidfStore(chunks)
    emb = get_embeddings()
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
    if PINECONE_API_KEY and _embeddings_available():
        from pinecone import Pinecone
        from langchain_pinecone import PineconeVectorStore
        if _pinecone_index_ready(Pinecone(api_key=PINECONE_API_KEY)):
            return ingest()
        return PineconeVectorStore(index_name=PINECONE_INDEX, embedding=get_embeddings())
    return ingest()


if __name__ == "__main__":
    ingest()
