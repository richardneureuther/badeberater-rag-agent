"""
This part builds a ChromaDB vector index from bodensee_swimming.jsonl.
The code creates one chunk per bathing spot since we deal with fairly short texts

"""
import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

#config, setup data paht, directory collection names and the embedding model
JSONL_PATH = Path("bodensee_swimming.jsonl")
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "bodensee_spots"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

#read the jsonl file and return a list of dicts, one per line at a time
def load_records(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records

#convert the scraped records into a Langchain document 
def record_to_document(rec: dict) -> Document:
    metadata = {
        "id": rec.get("id", ""),
        "title": rec.get("title", ""),
        "url": rec.get("url", ""),
        "plz_ort": rec.get("plz_ort", ""),
        "telefon": rec.get("telefon", ""),
        "source": rec.get("source", ""),
    }
    # Drop empty metadata fields
    metadata = {k: v for k, v in metadata.items() if v}

    return Document(
        page_content=rec["text"],
        metadata=metadata,
    )


def main():
    #read records and print counts 
    print(f"Loading records from {JSONL_PATH} ...")
    records = load_records(JSONL_PATH)
    print(f"  loaded {len(records)} records")
    #convert to langchain documents
    print("Converting to LangChain Documents ...")
    docs = [record_to_document(r) for r in records]

    #fetch the embedding model from huggingface 
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    #get documents, embed content and store in Chroma data base and write to dir
    print(f"Building Chroma index at ./{CHROMA_DIR} ...")
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )

    print(f"Done. Indexed {len(docs)} documents.")
    print(f"Vector store persisted to ./{CHROMA_DIR}")


if __name__ == "__main__":
    main()