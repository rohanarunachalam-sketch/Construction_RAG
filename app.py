import os
import re
import json
import zipfile
import base64
import hashlib
import pickle
from pathlib import Path
from datetime import datetime

import streamlit as st
import fitz

from dotenv import load_dotenv

from langchain_openai import (
    ChatOpenAI,
    OpenAIEmbeddings,
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from langchain_community.vectorstores import FAISS


# ============================================================
# OPTIONAL BM25 / ENSEMBLE IMPORTS
# ============================================================

try:
    from langchain_community.retrievers import BM25Retriever
except ImportError:
    BM25Retriever = None


try:
    from langchain.retrievers.ensemble import EnsembleRetriever
except ImportError:
    try:
        from langchain_community.retrievers import EnsembleRetriever
    except ImportError:
        try:
            from langchain.retrievers import EnsembleRetriever
        except ImportError:
            EnsembleRetriever = None


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Construction Project AI Assistant",
    page_icon="🏗️",
    layout="wide",
)


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY is missing in your .env file.")
    st.stop()


# ============================================================
# MODEL SETTINGS
# ============================================================

MODEL_NAME = "gpt-5.6-luna"

EMBEDDING_MODEL = "text-embedding-3-small"

MAX_VISION_PAGES = 2

VISION_IMAGE_SCALE = 1.0

MAX_TEXT_CONTEXT_CHARS = 24000


# ============================================================
# DEBUG SETTINGS
# ============================================================

DEBUG_LLM_RESPONSES = True


def debug_log_response(label, response):

    if not DEBUG_LLM_RESPONSES:
        return

    try:

        print("=" * 60)
        print(f"[DEBUG] {label}")
        print("RAW RESPONSE OBJECT:", repr(response))

        metadata = getattr(
            response,
            "response_metadata",
            None,
        )

        print("RESPONSE_METADATA:", metadata)

        finish_reason = None

        if isinstance(metadata, dict):
            finish_reason = metadata.get(
                "finish_reason"
            )

        print("FINISH REASON:", finish_reason)

        content = getattr(
            response,
            "content",
            None,
        )

        print("CONTENT TYPE:", type(content))
        print("CONTENT VALUE:", content)

        usage = getattr(
            response,
            "usage_metadata",
            None,
        )

        print("USAGE METADATA:", usage)

        print("=" * 60)

    except Exception as debug_error:

        print(
            f"[DEBUG] Failed to log response: {debug_error}"
        )


# ============================================================
# VERSION CONTROL
# ============================================================

CACHE_VERSION = "v6_debug_reinforcement_vision_answer"

INDEX_VERSION = "v5_reinforcement_visual_index"


# ============================================================
# STORAGE
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

STORAGE_DIR = BASE_DIR / "storage"

STORAGE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CACHE_FILE = BASE_DIR / "answer_cache.json"


# ============================================================
# CACHE FILE
# ============================================================

def initialize_cache_file():

    try:

        if not CACHE_FILE.exists():

            CACHE_FILE.write_text(
                "{}",
                encoding="utf-8",
            )

    except Exception as e:

        st.warning(
            f"Could not create answer_cache.json: {e}"
        )


initialize_cache_file()


def load_answer_cache():

    initialize_cache_file()

    try:

        data = json.loads(
            CACHE_FILE.read_text(
                encoding="utf-8"
            )
        )

        if isinstance(data, dict):
            return data

        return {}

    except Exception:

        return {}


def save_answer_cache(cache_data):

    try:

        temp_file = CACHE_FILE.with_suffix(".tmp")

        temp_file.write_text(
            json.dumps(
                cache_data,
                indent=4,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        temp_file.replace(CACHE_FILE)

        return True

    except Exception as e:

        st.warning(
            f"Could not save answer_cache.json: {e}"
        )

        return False


def create_project_id(uploaded_file):

    if not uploaded_file:
        return None

    return hashlib.sha256(
        uploaded_file.getvalue()
    ).hexdigest()


def normalize_question(question):

    if not question:
        return ""

    return re.sub(
        r"\s+",
        " ",
        question.lower(),
    ).strip()


def create_cache_key(
    project_id,
    question,
):

    normalized = normalize_question(question)

    raw = (
        f"{CACHE_VERSION}::"
        f"{project_id}::"
        f"{normalized}"
    )

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()


def get_cached_answer(
    project_id,
    question,
):

    if not project_id:
        return None

    cache = load_answer_cache()

    key = create_cache_key(
        project_id,
        question,
    )

    cached = cache.get(key)

    if not isinstance(cached, dict):
        return None

    answer = cached.get(
        "answer",
        "",
    )

    if not answer or not answer.strip():
        return None

    return cached


# ============================================================
# SOURCE SERIALIZATION
# ============================================================

def serialize_sources(documents):

    result = []

    seen = set()

    for doc in documents:

        if hasattr(doc, "metadata"):
            meta = doc.metadata
        else:
            meta = {}

        key = (
            meta.get("file_name"),
            meta.get("page"),
        )

        if key in seen:
            continue

        seen.add(key)

        result.append(
            {
                "file_name": meta.get("file_name"),
                "page": meta.get("page"),
                "discipline": meta.get("discipline"),
                "document_type": meta.get("document_type"),
                "source_path": meta.get("source_path"),
            }
        )

    return result


def cache_answer(
    project_id,
    question,
    answer,
    sources,
    drawing_images,
):

    if not project_id:
        return False

    if not answer or not answer.strip():
        return False

    cache = load_answer_cache()

    key = create_cache_key(
        project_id,
        question,
    )

    drawing_sources = []

    seen = set()

    for item in drawing_images or []:

        meta = item.get(
            "metadata",
            {},
        )

        source_key = (
            meta.get("file_name"),
            meta.get("page"),
        )

        if source_key in seen:
            continue

        seen.add(source_key)

        drawing_sources.append(
            {
                "file_name": meta.get("file_name"),
                "page": meta.get("page"),
                "source_path": meta.get("source_path"),
                "discipline": meta.get("discipline"),
            }
        )

    cache[key] = {

        "cache_version": CACHE_VERSION,

        "project_id": project_id,

        "question": question,

        "normalized_question": normalize_question(
            question
        ),

        "answer": answer,

        "sources": serialize_sources(
            sources
        ),

        "drawing_sources": drawing_sources,

        "created_at": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
    }

    return save_answer_cache(cache)


# ============================================================
# PERSISTENT PROJECT STORAGE
# ============================================================

def get_project_dir(project_id):

    path = STORAGE_DIR / project_id

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


def is_project_persisted(project_id):

    project_dir = get_project_dir(
        project_id
    )

    metadata_file = (
        project_dir / "metadata.pkl"
    )

    faiss_dir = (
        project_dir / "faiss_index"
    )

    if (
        not faiss_dir.exists()
        or not metadata_file.exists()
    ):
        return False

    try:

        with open(
            metadata_file,
            "rb",
        ) as file:

            payload = pickle.load(file)

        return (
            payload.get("index_version")
            == INDEX_VERSION
        )

    except Exception:

        return False


def save_persisted_project(
    project_id,
    vectorstore,
    all_documents,
    chunks,
    stats,
    drawing_pages,
):

    project_dir = get_project_dir(
        project_id
    )

    vectorstore.save_local(
        str(
            project_dir / "faiss_index"
        )
    )

    payload = {

        "index_version": INDEX_VERSION,

        "all_documents": all_documents,

        "chunks": chunks,

        "stats": stats,

        "drawing_pages": drawing_pages,
    }

    with open(
        project_dir / "metadata.pkl",
        "wb",
    ) as file:

        pickle.dump(
            payload,
            file,
        )


def load_persisted_project(
    project_id,
    embedding_model,
):

    project_dir = get_project_dir(
        project_id
    )

    vectorstore = FAISS.load_local(
        str(
            project_dir / "faiss_index"
        ),
        embedding_model,
        allow_dangerous_deserialization=True,
    )

    with open(
        project_dir / "metadata.pkl",
        "rb",
    ) as file:

        payload = pickle.load(file)

    return (

        vectorstore,

        payload["all_documents"],

        payload["chunks"],

        payload["stats"],

        payload.get(
            "drawing_pages",
            [],
        ),
    )


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {

    "project_loaded": False,

    "project_id": None,

    "project_dir": None,

    "all_documents": [],

    "chunks": [],

    "vectorstore": None,

    "bm25_retriever": None,

    "ensemble_retriever": None,

    "drawing_pages": [],

    "project_stats": {},

    "chat_history": [],
}


for key, value in DEFAULT_STATE.items():

    if key not in st.session_state:

        st.session_state[key] = value


# ============================================================
# GENERAL HELPERS
# ============================================================

def normalize_text(text):

    if not text:
        return ""

    text = text.replace(
        "\x00",
        " ",
    )

    text = text.replace(
        "\\",
        "/",
    )

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def format_response_content(content):

    if content is None:
        return ""

    if isinstance(content, str):

        return content.strip()

    if isinstance(content, list):

        parts = []

        for item in content:

            if isinstance(item, str):

                parts.append(item)

            elif isinstance(item, dict):

                item_type = item.get("type")

                if item_type == "text":

                    text_value = item.get(
                        "text",
                        "",
                    )

                    if text_value:
                        parts.append(
                            str(text_value)
                        )

        return "\n".join(parts).strip()

    return str(content).strip()


def extract_llm_answer(response):

    if response is None:
        return ""

    content = getattr(
        response,
        "content",
        None,
    )

    answer = format_response_content(
        content
    )

    if answer:
        return answer

    text_value = getattr(
        response,
        "text",
        None,
    )

    answer = format_response_content(
        text_value
    )

    if answer:
        return answer

    additional_kwargs = getattr(
        response,
        "additional_kwargs",
        {},
    )

    if isinstance(
        additional_kwargs,
        dict,
    ):

        for key in [
            "text",
            "output_text",
            "content",
        ]:

            value = additional_kwargs.get(
                key
            )

            answer = format_response_content(
                value
            )

            if answer:
                return answer

    return ""


# ============================================================
# DOCUMENT TYPE
# ============================================================

def detect_document_type(file_path):

    path_text = normalize_text(
        str(file_path)
    ).lower()

    file_name = (
        Path(file_path)
        .name
        .lower()
    )

    if any(
        x in path_text
        for x in [
            "/drawings/",
            "/drawing/",
        ]
    ):

        return "Drawing"

    if (
        "/mail/" in path_text
        or "/emails/" in path_text
    ):

        return "Mail"

    if (
        "/rfi/" in path_text
        or "/rfis/" in path_text
    ):

        return "RFI"

    if (
        "/specifications/" in path_text
        or "/specification/" in path_text
    ):

        return "Specification"

    if any(
        x in file_name
        for x in [
            "rfi",
            "r.f.i",
        ]
    ):

        return "RFI"

    if "spec" in file_name:

        return "Specification"

    if (
        "mail" in file_name
        or "email" in file_name
    ):

        return "Mail"

    if any(
        x in file_name
        for x in [
            "drawing",
            "dwg",
            "plan",
            "section",
            "elevation",
            "detail",
        ]
    ):

        return "Drawing"

    return "Unknown"


# ============================================================
# DISCIPLINE
# ============================================================

def detect_discipline(
    file_name,
    file_path,
):

    text = normalize_text(
        f"{file_name} {file_path}"
    ).lower()

    patterns = {

        "Structural": [
            r"\bstructural\b",
            r"\bstructure\b",
            r"\bstruct\b",
            r"\bstr\b",
            r"\bstr[-_ ]",
            r"\bfoundation\b",
            r"\bfooting\b",
            r"\bbeam\b",
            r"\bcolumn\b",
            r"\bslab\b",
            r"\brebar\b",
            r"\breinforcement\b",
        ],

        "Architectural": [
            r"\barchitectural\b",
            r"\barchitecture\b",
            r"\barch\b",
            r"\barch[-_ ]",
            r"\bfloor plan\b",
            r"\belevation\b",
        ],

        "MEP": [
            r"\bmep\b",
            r"\bmechanical\b",
            r"\bplumbing\b",
        ],

        "Electrical": [
            r"\belectrical\b",
            r"\belectric\b",
            r"\belec\b",
        ],

        "Fire": [
            r"\bfire\b",
            r"\bfirefighting\b",
            r"\bsprinkler\b",
            r"\bhydrant\b",
        ],
    }

    for discipline, patterns_list in patterns.items():

        for pattern in patterns_list:

            if re.search(
                pattern,
                text,
            ):

                return discipline

    return "Unknown"


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pages(pdf_path):

    pages = []

    try:

        pdf = fitz.open(
            pdf_path
        )

        for (
            page_number,
            page,
        ) in enumerate(
            pdf,
            start=1,
        ):

            text = page.get_text(
                "text"
            ) or ""

            pages.append(
                {
                    "page": page_number,

                    "text": normalize_text(
                        text
                    ),
                }
            )

        pdf.close()

    except Exception as e:

        st.warning(
            f"Could not extract "
            f"{Path(pdf_path).name}: {e}"
        )

    return pages


def get_pdf_info(pdf_path):

    info = {

        "pages": 0,

        "text_pages": 0,

        "image_only_pages": 0,
    }

    try:

        pdf = fitz.open(
            pdf_path
        )

        info["pages"] = len(pdf)

        for page in pdf:

            text = normalize_text(
                page.get_text("text")
            )

            if len(text) >= 20:

                info["text_pages"] += 1

            else:

                info["image_only_pages"] += 1

        pdf.close()

    except Exception:

        pass

    return info


def process_pdf(
    pdf_path,
    relative_path,
):

    file_name = Path(
        pdf_path
    ).name

    document_type = detect_document_type(
        relative_path
    )

    discipline = detect_discipline(
        file_name,
        relative_path,
    )

    pdf_info = get_pdf_info(
        pdf_path
    )

    pages = extract_pages(
        pdf_path
    )

    documents = []

    for page_data in pages:

        metadata = {

            "document_type": document_type,

            "discipline": discipline,

            "file_name": file_name,

            "page": page_data["page"],

            "page_label": str(
                page_data["page"]
            ),

            "source_path": str(
                relative_path
            ),

            "total_pages": pdf_info["pages"],
        }

        documents.append(
            {
                "text": page_data["text"],

                "metadata": metadata,
            }
        )

    return (
        documents,
        pdf_info,
    )


# ============================================================
# CHUNKING
# ============================================================

def create_chunks(
    all_documents
):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = []

    for doc in all_documents:

        text = doc["text"].strip()

        if len(text) < 20:
            continue

        page_chunks = splitter.split_text(
            text
        )

        for (
            chunk_number,
            chunk_text,
        ) in enumerate(
            page_chunks,
            start=1,
        ):

            metadata = doc["metadata"].copy()

            metadata["chunk"] = chunk_number

            metadata["total_chunks_in_page"] = len(
                page_chunks
            )

            chunks.append(
                {
                    "text": chunk_text,

                    "metadata": metadata,
                }
            )

    return chunks


# ============================================================
# OPENAI EMBEDDINGS
# ============================================================

@st.cache_resource
def create_embedding_model():

    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
    )


# ============================================================
# OPENAI CHAT MODEL
# ============================================================

@st.cache_resource
def create_chat_model():

    return ChatOpenAI(
        model=MODEL_NAME,
        temperature=0.1,
        max_tokens=4096,
        api_key=OPENAI_API_KEY,
    )


# ============================================================
# HYBRID RETRIEVER
# ============================================================

def setup_hybrid_retriever(
    chunks,
    vectorstore,
):

    bm25 = None

    if BM25Retriever:

        try:

            docs = [

                Document(
                    page_content=item["text"],
                    metadata=item["metadata"],
                )

                for item in chunks
            ]

            bm25 = BM25Retriever.from_documents(
                docs
            )

            bm25.k = 10

        except Exception:

            bm25 = None

    ensemble = None

    if (
        bm25
        and EnsembleRetriever
    ):

        try:

            faiss_retriever = (
                vectorstore.as_retriever(
                    search_kwargs={
                        "k": 10
                    }
                )
            )

            ensemble = EnsembleRetriever(
                retrievers=[
                    bm25,
                    faiss_retriever,
                ],
                weights=[
                    0.5,
                    0.5,
                ],
            )

        except Exception:

            ensemble = None

    return (
        bm25,
        ensemble,
    )


# ============================================================
# SEARCH QUERY
# ============================================================

def prepare_search_query(
    question
):

    history = st.session_state.chat_history

    if not history:
        return question

    recent_questions = [

        item["content"]

        for item in history[-4:]

        if item.get("role") == "user"
    ]

    if not recent_questions:
        return question

    return (
        "Previous related question: "
        f"{recent_questions[-1]}\n"
        "Current question: "
        f"{question}"
    )


# ============================================================
# TEXT RETRIEVAL
# ============================================================

def retrieve_documents(
    query,
    k=10,
):

    ensemble = (
        st.session_state.ensemble_retriever
    )

    bm25 = (
        st.session_state.bm25_retriever
    )

    vectorstore = (
        st.session_state.vectorstore
    )

    if ensemble is not None:

        try:

            results = ensemble.invoke(
                query
            )

            return results[:k]

        except Exception:

            pass

    if (
        bm25 is not None
        and vectorstore is not None
    ):

        try:

            bm25_docs = bm25.invoke(
                query
            )

            faiss_docs = (
                vectorstore.similarity_search(
                    query,
                    k=k,
                )
            )

            scores = {}

            doc_map = {}

            for (
                rank,
                doc,
            ) in enumerate(
                bm25_docs
            ):

                key = (

                    doc.metadata.get(
                        "file_name"
                    ),

                    doc.metadata.get(
                        "page"
                    ),

                    doc.page_content[:80],
                )

                doc_map[key] = doc

                scores[key] = (
                    scores.get(
                        key,
                        0,
                    )
                    + 1 / (60 + rank)
                )

            for (
                rank,
                doc,
            ) in enumerate(
                faiss_docs
            ):

                key = (

                    doc.metadata.get(
                        "file_name"
                    ),

                    doc.metadata.get(
                        "page"
                    ),

                    doc.page_content[:80],
                )

                doc_map[key] = doc

                scores[key] = (
                    scores.get(
                        key,
                        0,
                    )
                    + 1 / (60 + rank)
                )

            ranked = sorted(
                scores,
                key=scores.get,
                reverse=True,
            )

            return [
                doc_map[key]
                for key in ranked[:k]
            ]

        except Exception:

            pass

    if vectorstore is not None:

        try:

            return vectorstore.similarity_search(
                query,
                k=k,
            )

        except Exception:

            pass

    return []


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(
    documents,
    max_chars=MAX_TEXT_CONTEXT_CHARS,
):

    if not documents:

        return (
            "No text documents were retrieved."
        )

    parts = []

    current_chars = 0

    for index, doc in enumerate(
        documents,
        start=1,
    ):

        meta = doc.metadata

        block = f"""
SOURCE {index}

File:
{meta.get("file_name")}

Document Type:
{meta.get("document_type")}

Discipline:
{meta.get("discipline")}

Page:
{meta.get("page")}

Source Path:
{meta.get("source_path")}

Content:
{doc.page_content}
"""

        if (
            current_chars
            + len(block)
            > max_chars
        ):
            break

        parts.append(block)

        current_chars += len(block)

    return "\n".join(parts)


# ============================================================
# VISUAL QUESTION DETECTION
# ============================================================

VISUAL_TERMS = {

    "drawing",
    "drawings",

    "dimension",
    "dimensions",
    "dimensioning",

    "rebar",
    "reinforcement",
    "reinforcing",

    "bar",
    "bars",

    "beam",
    "beams",

    "column",
    "columns",

    "slab",
    "slabs",

    "footing",
    "footings",

    "foundation",
    "foundations",

    "grid",
    "grids",

    "section",
    "sections",

    "elevation",

    "detail",
    "details",

    "plan",
    "layout",

    "level",

    "thickness",
    "cover",
    "spacing",
    "diameter",
    "dia",

    "stirrup",
    "stirrups",

    "mesh",

    "opening",
    "openings",

    "void",
    "voids",

    "door",
    "doors",

    "window",
    "windows",

    "location",
    "shown",
    "where",

    "schedule",
    "callout",

    "mark",
    "marks",

    "symbol",
    "symbols",

    "steel",
    "deck",

    "mezzanine",
    "roof",

    "wall",
    "walls",
}


def is_visual_question(
    question
):

    words = set(
        re.findall(
            r"[a-z0-9]+",
            question.lower(),
        )
    )

    if words & VISUAL_TERMS:
        return True

    phrases = [

        "which page",

        "which pages",

        "where is",

        "where are",

        "shown in",

        "shown on",

        "look at the drawing",

        "look in the drawing",

        "in the drawing",

        "in the drawings",

        "from the drawing",

        "from the drawings",
    ]

    question_lower = question.lower()

    return any(
        phrase in question_lower
        for phrase in phrases
    )


# ============================================================
# REINFORCEMENT QUESTION DETECTION
# ============================================================

REINFORCEMENT_TERMS = {

    "rebar",
    "reinforcement",
    "reinforcements",
    "reinforcing",

    "reinforced",

    "bar",
    "bars",

    "steel",

    "stirrup",
    "stirrups",

    "ties",

    "main bar",
    "main bars",

    "top bar",
    "top bars",

    "bottom bar",
    "bottom bars",

    "distribution bar",
    "distribution bars",

    "starter bar",
    "starter bars",

    "dowel",
    "dowels",

    "mesh",

    "tmt",

    "dia",
    "diameter",

    "spacing",

    "c/c",

    "center to center",

    "cover",
}


def is_reinforcement_question(
    question
):

    q = normalize_question(
        question
    )

    if any(
        term in q
        for term in [
            "reinforcement",
            "reinforcements",
            "reinforcing",
            "rebar",
            "reinforced concrete",
            "reinforcement detail",
            "reinforcement details",
            "rebar detail",
            "rebar details",
        ]
    ):
        return True

    words = set(
        re.findall(
            r"[a-z0-9]+",
            q,
        )
    )

    if (
        words & {
            "bar",
            "bars",
            "stirrup",
            "stirrups",
            "dowel",
            "dowels",
            "mesh",
        }
        and words
        & {
            "beam",
            "beams",
            "column",
            "columns",
            "slab",
            "slabs",
            "footing",
            "footings",
            "foundation",
            "foundations",
            "drawing",
            "drawings",
        }
    ):
        return True

    return False


# ============================================================
# TOKENIZER
# ============================================================

def tokenize(text):

    return set(

        word

        for word in re.findall(
            r"[a-z0-9]+",
            text.lower(),
        )

        if len(word) >= 2
    )


# ============================================================
# REINFORCEMENT KEYWORDS
# ============================================================

REINFORCEMENT_SEARCH_TERMS = [

    "reinforcement",

    "reinforcing",

    "rebar",

    "reinforced concrete",

    "reinforcement details",

    "reinforcement detail",

    "reinforcement drawing",

    "rebar detail",

    "rebar schedule",

    "bar schedule",

    "bar mark",

    "main bars",

    "top bars",

    "bottom bars",

    "distribution bars",

    "stirrups",

    "ties",

    "dowels",

    "starter bars",

    "TMT",

    "dia",

    "diameter",

    "spacing",

    "c/c",

    "cover",

]


# ============================================================
# DRAWING PAGE SCORING
# ============================================================

def score_drawing_page(
    question,
    page_record,
):

    meta = page_record["metadata"]

    page_text = page_record.get(
        "text",
        "",
    )

    query_words = tokenize(
        question
    )

    page_words = tokenize(
        page_text
    )

    file_text = (
        f'{meta.get("file_name", "")} '
        f'{meta.get("source_path", "")} '
        f'{meta.get("discipline", "")}'
    )

    file_words = tokenize(
        file_text
    )

    score = 0.0

    score += (
        len(
            query_words
            & page_words
        )
        * 5.0
    )

    score += (
        len(
            query_words
            & file_words
        )
        * 3.0
    )

    visual_query_words = (
        query_words
        & VISUAL_TERMS
    )

    score += (
        len(
            visual_query_words
            & page_words
        )
        * 1.5
    )

    if is_reinforcement_question(
        question
    ):

        page_lower = page_text.lower()

        file_lower = file_text.lower()

        reinforcement_hits = 0

        for term in REINFORCEMENT_SEARCH_TERMS:

            if term.lower() in page_lower:

                reinforcement_hits += 1

            elif term.lower() in file_lower:

                reinforcement_hits += 0.5

        score += (
            reinforcement_hits
            * 8.0
        )

        discipline = (
            str(
                meta.get(
                    "discipline",
                    "",
                )
            )
            .lower()
        )

        if "struct" in discipline:

            score += 15.0

        drawing_name = (
            f'{meta.get("file_name", "")} '
            f'{meta.get("source_path", "")}'
        ).lower()

        for term in [
            "structural",
            "foundation",
            "footing",
            "beam",
            "column",
            "slab",
            "rebar",
            "reinforcement",
            "rc",
            "reinforced",
        ]:

            if term in drawing_name:

                score += 4.0

    if page_text:

        score += min(
            len(page_text) / 3000.0,
            1.0,
        )

    return score


# ============================================================
# PAGE KEY
# ============================================================

def page_key(
    page_record
):

    meta = page_record["metadata"]

    return (
        meta.get("source_path"),
        meta.get("page"),
    )


# ============================================================
# LOCAL REINFORCEMENT PAGE SEARCH
# ============================================================

def find_reinforcement_pages(
    question,
    max_pages=12,
):

    drawing_pages = (
        st.session_state.drawing_pages
    )

    if not drawing_pages:
        return []

    scored = []

    q_words = tokenize(
        question
    )

    for record in drawing_pages:

        meta = record["metadata"]

        text = record.get(
            "text",
            "",
        )

        lower_text = text.lower()

        file_text = (
            f'{meta.get("file_name", "")} '
            f'{meta.get("source_path", "")} '
            f'{meta.get("discipline", "")}'
        ).lower()

        score = 0.0

        score += (
            len(
                q_words
                & tokenize(text)
            )
            * 4.0
        )

        for term in REINFORCEMENT_SEARCH_TERMS:

            term_lower = term.lower()

            if term_lower in lower_text:

                score += 12.0

            elif term_lower in file_text:

                score += 5.0

        discipline = str(
            meta.get(
                "discipline",
                "",
            )
        ).lower()

        if "struct" in discipline:

            score += 20.0

        name_lower = file_text

        for term in [
            "foundation",
            "footing",
            "beam",
            "column",
            "slab",
            "rebar",
            "reinforcement",
            "structural",
            "rc",
        ]:

            if term in name_lower:

                score += 5.0

        if text:

            score += min(
                len(text) / 2000.0,
                2.0,
            )

        scored.append(
            (
                score,
                record,
            )
        )

    scored.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    selected = []

    seen = set()

    for score, record in scored:

        key = page_key(
            record
        )

        if key in seen:
            continue

        seen.add(key)

        selected.append(
            {
                "text": record.get(
                    "text",
                    "",
                ),

                "metadata": record[
                    "metadata"
                ].copy(),

                "score": score,
            }
        )

        if len(selected) >= max_pages:
            break

    return selected


# ============================================================
# SELECT DRAWING PAGES
# ============================================================

def select_drawing_pages(
    question,
    retrieved_documents,
    max_pages=MAX_VISION_PAGES,
):

    drawing_pages = (
        st.session_state.drawing_pages
    )

    if not drawing_pages:
        return []

    if is_reinforcement_question(
        question
    ):

        local_candidates = (
            find_reinforcement_pages(
                question,
                max_pages=12,
            )
        )

    else:

        local_candidates = []

    candidate_boosts = {}

    for rank, doc in enumerate(
        retrieved_documents
    ):

        meta = doc.metadata

        if (
            meta.get(
                "document_type"
            )
            != "Drawing"
        ):

            continue

        key = (

            meta.get(
                "source_path"
            ),

            meta.get(
                "page"
            ),
        )

        candidate_boosts[key] = (
            100.0 - rank
        )

    for rank, record in enumerate(
        local_candidates
    ):

        key = page_key(
            record
        )

        candidate_boosts[key] = (
            candidate_boosts.get(
                key,
                0,
            )
            + 80.0
            - rank
        )

    scored = []

    for page_record in drawing_pages:

        score = score_drawing_page(
            question,
            page_record,
        )

        key = page_key(
            page_record
        )

        if key in candidate_boosts:

            score += candidate_boosts[key]

        scored.append(
            (
                score,
                page_record,
            )
        )

    scored.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    selected = []

    seen = set()

    for score, page_record in scored:

        key = page_key(
            page_record
        )

        if key in seen:
            continue

        seen.add(key)

        selected.append(
            {
                "text": page_record.get(
                    "text",
                    "",
                ),

                "metadata": page_record[
                    "metadata"
                ].copy(),

                "score": score,
            }
        )

        if len(selected) >= max_pages:
            break

    return selected


# ============================================================
# FIND SOURCE PDF
# ============================================================

def find_source_pdf(
    metadata
):

    project_dir = (
        st.session_state.project_dir
    )

    if not project_dir:
        return None

    source_path = metadata.get(
        "source_path"
    )

    if source_path:

        candidate = (
            Path(project_dir)
            / source_path
        )

        if candidate.exists():

            return str(candidate)

    file_name = metadata.get(
        "file_name"
    )

    if file_name:

        matches = list(
            Path(project_dir).rglob(
                file_name
            )
        )

        if matches:

            return str(matches[0])

    return None


# ============================================================
# RENDER PDF PAGE
# ============================================================

def render_pdf_page(
    pdf_path,
    page_number,
    scale=VISION_IMAGE_SCALE,
):

    try:

        pdf = fitz.open(
            pdf_path
        )

        page_index = (
            int(page_number) - 1
        )

        if (
            page_index < 0
            or page_index >= len(pdf)
        ):

            pdf.close()

            return None

        page = pdf[page_index]

        matrix = fitz.Matrix(
            scale,
            scale,
        )

        pix = page.get_pixmap(
            matrix=matrix,
            alpha=False,
        )

        image_bytes = pix.tobytes(
            "jpeg"
        )

        pdf.close()

        return image_bytes

    except Exception:

        return None


# ============================================================
# RENDER SELECTED DRAWINGS
# ============================================================

def render_selected_drawing_pages(
    page_records
):

    rendered = []

    for record in page_records:

        meta = record["metadata"]

        pdf_path = find_source_pdf(
            meta
        )

        if not pdf_path:
            continue

        image_bytes = render_pdf_page(
            pdf_path,
            meta.get(
                "page",
                1,
            ),
            scale=VISION_IMAGE_SCALE,
        )

        if image_bytes:

            rendered.append(
                {
                    "bytes": image_bytes,

                    "metadata": meta,

                    "score": record.get(
                        "score"
                    ),
                }
            )

    return rendered


# ============================================================
# CACHED DRAWING RENDER
# ============================================================

def render_cached_drawing_sources(
    drawing_sources
):

    rendered = []

    for source in drawing_sources or []:

        metadata = {

            "file_name": source.get(
                "file_name"
            ),

            "page": source.get(
                "page"
            ),

            "source_path": source.get(
                "source_path"
            ),

            "discipline": source.get(
                "discipline"
            ),
        }

        pdf_path = find_source_pdf(
            metadata
        )

        if not pdf_path:
            continue

        image_bytes = render_pdf_page(
            pdf_path,
            metadata.get(
                "page",
                1,
            ),
            scale=VISION_IMAGE_SCALE,
        )

        if image_bytes:

            rendered.append(
                {
                    "bytes": image_bytes,

                    "metadata": metadata,
                }
            )

    return rendered


# ============================================================
# BUILD REINFORCEMENT PROMPT
# ============================================================

def build_reinforcement_prompt(
    original_question,
    search_query,
    context,
):

    return f"""
You are an expert structural construction drawing analyst.

The user is asking a REINFORCEMENT-related question.

USER QUESTION:
{original_question}

SEARCH QUERY:
{search_query}

TEXT RETRIEVED FROM THE PROJECT:
{context}

The attached images are selected pages from the uploaded construction
drawings.

IMPORTANT:

You MUST inspect every attached drawing image carefully.

The user wants the reinforcement information that is actually
available in the uploaded drawings.

Your answer MUST be a TEXT ANSWER.

NEVER return an empty response.

Do NOT answer with only images.

Do NOT say that you cannot analyze the drawings if the information
is visible in the attached images.

Extract only information that is actually visible or supported by
the supplied project text.

For reinforcement, look specifically for:

- Member type
- Beam reinforcement
- Column reinforcement
- Slab reinforcement
- Footing reinforcement
- Foundation reinforcement
- Wall reinforcement
- Main bars
- Top bars
- Bottom bars
- Distribution bars
- Stirrups
- Ties
- Starter bars
- Dowels
- Bar diameter
- Number of bars
- Bar spacing
- c/c spacing
- Reinforcement mesh
- Reinforcement marks
- Bar marks
- Cover
- Reinforcement schedules
- Detail references
- Section/detail references
- Any other reinforcement information explicitly shown

For EVERY reinforcement detail you identify:

1. State the member/type if identifiable.
2. State the reinforcement information.
3. State the exact drawing file name.
4. State the page number.
5. If a value is unclear, say "not legible" rather than guessing.

If the question asks "what reinforcement details are available",
give an INVENTORY of the reinforcement information found across
the supplied drawing pages.

If several reinforcement details are found on different pages,
list them separately.

Example structure:

## Reinforcement details found

| Element | Reinforcement detail | Drawing | Page |
|---|---|---|---|
| Beam | ... | ... | ... |
| Column | ... | ... | ... |
| Slab | ... | ... | ... |
| Footing | ... | ... | ... |

Then add:

## Notes
- Mention anything that is visible but not fully legible.
- Do not invent missing values.

If no reinforcement detail can be read from the supplied pages,
say:

"No reinforcement detail could be read clearly from the selected
drawing pages."

Then identify the drawing pages inspected.

Do not use general engineering knowledge to fill missing values.

Do not claim that the reinforcement is structurally adequate,
safe, compliant, or correctly designed.

Use only the uploaded project as the source.

RETURN A TEXT ANSWER.
"""


# ============================================================
# BUILD GENERAL VISUAL PROMPT
# ============================================================

def build_general_visual_prompt(
    original_question,
    search_query,
    context,
):

    return f"""
You are an expert Construction Project AI Assistant specializing in
architectural, structural and engineering project documents.

USER QUESTION:
{original_question}

SEARCH QUERY USED:
{search_query}

TEXT RETRIEVED FROM PROJECT:
{context}

The attached images are selected candidate drawing pages from the
uploaded project.

You MUST inspect every attached drawing image before answering.

MOST IMPORTANT:

You MUST return a TEXT ANSWER.

NEVER return an empty response.

Do NOT answer with images only.

IMPORTANT DRAWING RULES:

1. Read visible dimensions, labels, notes, schedules, symbols,
   reinforcement callouts, grid lines and detail references directly
   from the drawing when relevant.

2. Do not assume that information missing from extracted PDF text
   is absent from the drawing.

3. If a value is visible in an attached drawing, use it and cite
   the drawing file and page.

4. If a drawing value is too small or unclear to read, say that
   it is not legible instead of guessing.

5. Do not invent reinforcement sizes, spacing, cover, dimensions,
   materials or levels.

6. Distinguish between information explicitly shown and engineering
   interpretation.

7. If the text and drawing appear to conflict, identify the conflict.

8. If the question asks "which page", give the exact file name and
   page number when identifiable.

9. Use only the uploaded project as the primary source.

10. Do not claim a design is safe, adequate or code-compliant unless
    the uploaded project explicitly establishes that.

11. Answer in practical English with headings and bullets when useful.

12. Keep the answer focused on the user's question.

RETURN A TEXT ANSWER.
"""


# ============================================================
# CREATE MULTIMODAL MESSAGE
# ============================================================

def create_multimodal_message(
    original_question,
    search_query,
    context,
    drawing_images,
):

    if is_reinforcement_question(
        original_question
    ):

        prompt = build_reinforcement_prompt(
            original_question,
            search_query,
            context,
        )

    else:

        prompt = build_general_visual_prompt(
            original_question,
            search_query,
            context,
        )

    content = []

    content.append(
        {
            "type": "text",

            "text": prompt,
        }
    )

    for index, drawing in enumerate(
        drawing_images,
        start=1,
    ):

        meta = drawing["metadata"]

        content.append(
            {
                "type": "text",

                "text": (
                    f"\nDRAWING IMAGE {index}\n"
                    f"File: {meta.get('file_name')}\n"
                    f"Page: {meta.get('page')}\n"
                    f"Discipline: {meta.get('discipline')}\n"
                    f"Source Path: {meta.get('source_path')}\n"
                    "Inspect this image and use it as evidence.\n"
                ),
            }
        )

        encoded = base64.b64encode(
            drawing["bytes"]
        ).decode(
            "utf-8"
        )

        content.append(
            {
                "type": "image_url",

                "image_url": {
                    "url": (
                        "data:image/jpeg;base64,"
                        f"{encoded}"
                    )
                },
            }
        )

    return HumanMessage(
        content=content
    )


# ============================================================
# RETRY MESSAGE
# ============================================================

def create_retry_message(
    original_question,
    drawing_images,
):

    content = []

    retry_prompt = f"""
You must answer this construction drawing question:

{original_question}

The previous answer-generation attempt produced no text.

This is a recovery attempt.

YOU MUST RETURN A TEXT ANSWER.

Inspect the attached drawing images carefully.

If the question concerns reinforcement, identify every visible
reinforcement detail you can read and provide the drawing file name
and page number.

Do not return images only.

Do not return an empty response.

Do not guess unreadable values.

If a value is unclear, say "not legible".

Use this format when possible:

## Details found

- Element:
- Reinforcement:
- Drawing:
- Page:

Repeat for each identifiable detail.

If no relevant detail is visible, clearly say so and identify the
pages inspected.
"""

    content.append(
        {
            "type": "text",
            "text": retry_prompt,
        }
    )

    for index, drawing in enumerate(
        drawing_images,
        start=1,
    ):

        meta = drawing["metadata"]

        content.append(
            {
                "type": "text",

                "text": (
                    f"IMAGE {index}: "
                    f"{meta.get('file_name')} "
                    f"- Page {meta.get('page')}"
                ),
            }
        )

        encoded = base64.b64encode(
            drawing["bytes"]
        ).decode(
            "utf-8"
        )

        content.append(
            {
                "type": "image_url",

                "image_url": {
                    "url": (
                        "data:image/jpeg;base64,"
                        f"{encoded}"
                    )
                },
            }
        )

    return HumanMessage(
        content=content
    )


# ============================================================
# MAIN QUESTION ANSWER
# ============================================================

def ask_project_assistant(
    question
):

    search_query = prepare_search_query(
        question
    )

    documents = retrieve_documents(
        search_query,
        k=10,
    )

    visual = is_visual_question(
        question
    )

    reinforcement = is_reinforcement_question(
        question
    )

    drawing_images = []

    if visual or reinforcement:

        selected_pages = select_drawing_pages(
            question,
            documents,
            max_pages=MAX_VISION_PAGES,
        )

        drawing_images = render_selected_drawing_pages(
            selected_pages
        )

    if (
        not documents
        and not drawing_images
    ):

        return (
            "I could not find relevant information "
            "in the uploaded project.",
            [],
            [],
        )

    context = build_context(
        documents
    )

    llm = create_chat_model()

    message = create_multimodal_message(
        original_question=question,
        search_query=search_query,
        context=context,
        drawing_images=drawing_images,
    )

    response = llm.invoke(
        [message]
    )

    debug_log_response(
        "FIRST ATTEMPT",
        response,
    )

    answer = extract_llm_answer(
        response
    )

    # ========================================================
    # EMPTY RESPONSE RECOVERY
    # ========================================================

    if not answer:

        retry_message = create_retry_message(
            question,
            drawing_images,
        )

        retry_response = llm.invoke(
            [retry_message]
        )

        debug_log_response(
            "RETRY ATTEMPT",
            retry_response,
        )

        answer = extract_llm_answer(
            retry_response
        )

    # ========================================================
    # FINAL FALLBACK
    # ========================================================

    if not answer:

        if drawing_images:

            page_lines = []

            for image in drawing_images:

                meta = image["metadata"]

                page_lines.append(
                    f"- {meta.get('file_name')} "
                    f"— Page {meta.get('page')}"
                )

            answer = (
                "I found relevant drawing pages, but the AI "
                "could not produce a readable text analysis.\n\n"
                "### Drawing pages inspected\n"
                + "\n".join(page_lines)
                + "\n\n"
                "Please try the question again."
            )

        else:

            answer = (
                "I could not generate a text answer "
                "from the retrieved project information."
            )

    return (
        answer,
        documents,
        drawing_images,
    )


# ============================================================
# SOURCE DISPLAY
# ============================================================

def source_label(source):
    document_type = (
        source.get("document_type")
        or "DOC"
    )

    file_name = (
        source.get("file_name")
        or "Unknown file"
    )

    page = (
        source.get("page")
        or "-"
    )

    discipline = (
        source.get("discipline")
        or "Unknown"
    )

    return (
        f"{document_type} | "
        f"{file_name} | "
        f"Page {page} | "
        f"{discipline}"
    )


# ============================================================
# DISPLAY ASSISTANT RESPONSE
# ============================================================

def render_assistant_turn(
    answer,
    sources=None,
    drawing_images=None,
    is_cached=False,
):

    st.markdown(answer)

    # --------------------------------------------------------
    # DRAWINGS INSPECTED
    # --------------------------------------------------------

    if drawing_images:

        st.markdown("### 🖼️ Drawing Pages Inspected")

        image_columns = st.columns(
            min(len(drawing_images), 3)
        )

        for index, image in enumerate(drawing_images):

            with image_columns[index % len(image_columns)]:

                meta = image["metadata"]

                st.image(
                    image["bytes"],
                    caption=(
                        f'{meta.get("file_name", "Unknown")} '
                        f'— Page {meta.get("page", "-")}'
                    ),
                    use_container_width=True,
                )

                st.caption(
                    f'Discipline: '
                    f'{meta.get("discipline", "Unknown")}'
                )

    # --------------------------------------------------------
    # TEXT SOURCES
    # --------------------------------------------------------

    if sources:

        st.markdown("### 📚 Retrieved Document Sources")

        for source in sources:

            document_type = (
                source.get("document_type")
                or "Document"
            )

            file_name = (
                source.get("file_name")
                or "Unknown file"
            )

            page = (
                source.get("page")
                or "-"
            )

            discipline = (
                source.get("discipline")
                or "Unknown"
            )

            with st.container(border=True):

                col1, col2, col3 = st.columns(
                    [1.2, 3.5, 1.3]
                )

                with col1:
                    st.caption("TYPE")
                    st.write(document_type)

                with col2:
                    st.caption("FILE")
                    st.write(file_name)

                with col3:
                    st.caption("PAGE")
                    st.write(f"Page {page}")

                st.caption(
                    f"Discipline: {discipline}"
                )

    # --------------------------------------------------------
    # CACHE
    # --------------------------------------------------------

    if is_cached:

        st.caption(
            "⚡ Response served from answer_cache.json"
        )

    # --------------------------------------------------------
    # DISCLAIMER
    # --------------------------------------------------------

    st.caption(
        "⚠️ AI results must be verified against the "
        "approved construction drawings and specifications "
        "before execution."
    )


# ============================================================
# CLEAR PROJECT
# ============================================================

def clear_project():

    for (
        key,
        default,
    ) in DEFAULT_STATE.items():

        if isinstance(
            default,
            list,
        ):

            st.session_state[key] = []

        elif isinstance(
            default,
            dict,
        ):

            st.session_state[key] = {}

        else:

            st.session_state[key] = default

    st.rerun()


# ============================================================
# STREAMLIT-ONLY USER INTERFACE
# ============================================================

# No custom HTML
# No custom CSS
# No generated images
# No external frontend framework
# All UI is built with native Streamlit widgets.

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🏗️ Construction AI")
    st.caption("Project intelligence workspace")

    st.divider()

    if st.session_state.project_loaded:

        st.success("Project Ready")

        st.subheader("Project Overview")

        stats = st.session_state.project_stats

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "PDFs",
                stats.get("pdf_files", 0),
            )

        with col2:
            st.metric(
                "Pages",
                stats.get("pages", 0),
            )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Drawings",
                stats.get("drawing_pages", 0),
            )

        with col2:
            st.metric(
                "Chunks",
                stats.get("chunks", 0),
            )

        st.divider()

        st.subheader("Project Controls")

        if st.button(
            "🗑️ Clear Conversation",
            use_container_width=True,
            key="sidebar_clear_conversation",
        ):

            st.session_state.chat_history = []

            st.rerun()

        if st.button(
            "🔄 Reset Project",
            use_container_width=True,
            key="sidebar_reset_project",
        ):

            clear_project()

    else:

        st.info(
            "No project loaded."
        )

        st.subheader("Workflow")

        st.write(
            "1️⃣ Upload Project"
        )

        st.write(
            "2️⃣ Process & Index"
        )

        st.write(
            "3️⃣ Ask Questions"
        )

        st.write(
            "4️⃣ Verify Sources"
        )

    st.divider()

    st.subheader("System")

    st.caption(
        f"Cached answers: "
        f"{len(load_answer_cache())}"
    )

    st.caption(
        f"Vision pages/question: "
        f"maximum {MAX_VISION_PAGES}"
    )

    st.caption(
        "Hybrid retrieval: BM25 + vector search"
    )

    st.caption(
        "Reinforcement-aware drawing retrieval: enabled"
    )

    if DEBUG_LLM_RESPONSES:

        st.warning(
            "Debug logging is ON. "
            "Check the Streamlit terminal for raw LLM responses."
        )


# ============================================================
# MAIN HEADER
# ============================================================

st.title(
    "🏗️ Construction Project AI Assistant"
)

st.subheader(
    "Build smarter with AI"
)

st.info(
    "Upload your project → Process it once → Ask your question → "
    "Read the answer → Check sources and drawings."
)

st.write(
    "Upload your construction project and ask questions "
    "about drawings, reinforcement, specifications, RFIs, "
    "emails and other project documents."
)

st.divider()


# ============================================================
# WORKFLOW
# ============================================================

st.subheader(
    "How it works"
)

workflow_columns = st.columns(4)

with workflow_columns[0]:

    st.info(
        "### 1️⃣ Upload Project"
    )

    st.write(
        "Upload a ZIP containing your construction "
        "project documents."
    )

with workflow_columns[1]:

    st.info(
        "### 2️⃣ Process & Index"
    )

    st.write(
        "Extract PDFs, create chunks and build "
        "the searchable project knowledge base."
    )

with workflow_columns[2]:

    st.info(
        "### 3️⃣ Ask Questions"
    )

    st.write(
        "Ask questions in natural language about "
        "drawings and project information."
    )

with workflow_columns[3]:

    st.info(
        "### 4️⃣ Verify Sources"
    )

    st.write(
        "Review exact file names, page numbers and "
        "relevant drawing pages."
    )


st.divider()


# ============================================================
# UPLOAD PROJECT
# ============================================================

st.subheader(
    "📤 Upload Construction Project"
)

st.caption(
    "Upload a ZIP containing drawings, specifications, "
    "RFIs, emails and other project PDFs."
)

uploaded_zip = st.file_uploader(
    "Choose Project ZIP",
    type=["zip"],
    help=(
        "The system creates a persistent project index. "
        "If the same project was already indexed, it will "
        "be loaded instead of being processed again."
    ),
)

if uploaded_zip:

    st.success(
        f"Selected: {uploaded_zip.name}"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "File size",
            f"{uploaded_zip.size / (1024 * 1024):.2f} MB",
        )

    with col2:

        current_project_id = create_project_id(
            uploaded_zip
        )

        already_indexed = is_project_persisted(
            current_project_id
        )

        st.metric(
            "Project status",
            "Already indexed"
            if already_indexed
            else "New project",
        )

    # ========================================================
    # AUTOMATICALLY LOAD EXISTING PROJECT
    # ========================================================

    if (
        not st.session_state.project_loaded
        and already_indexed
        and st.session_state.project_id
        != current_project_id
    ):

        with st.spinner(
            "Loading existing project index..."
        ):

            try:

                embedding_model = (
                    create_embedding_model()
                )

                (
                    vectorstore,
                    all_documents,
                    chunks,
                    stats,
                    drawing_pages,
                ) = load_persisted_project(
                    current_project_id,
                    embedding_model,
                )

                (
                    bm25,
                    ensemble,
                ) = setup_hybrid_retriever(
                    chunks,
                    vectorstore,
                )

                project_dir = (
                    get_project_dir(
                        current_project_id
                    )
                    / "extracted"
                )

                st.session_state.vectorstore = (
                    vectorstore
                )

                st.session_state.all_documents = (
                    all_documents
                )

                st.session_state.chunks = (
                    chunks
                )

                st.session_state.bm25_retriever = (
                    bm25
                )

                st.session_state.ensemble_retriever = (
                    ensemble
                )

                st.session_state.drawing_pages = (
                    drawing_pages
                )

                st.session_state.project_stats = (
                    stats
                )

                st.session_state.project_id = (
                    current_project_id
                )

                st.session_state.project_dir = str(
                    project_dir
                )

                st.session_state.project_loaded = (
                    True
                )

                st.session_state.chat_history = []

                st.rerun()

            except Exception as e:

                st.error(
                    "Could not load the existing project index."
                )

                st.exception(e)


    # ========================================================
    # PROJECT SWITCH
    # ========================================================

    if (
        st.session_state.project_id
        != current_project_id
    ):

        st.session_state.project_loaded = False

        st.session_state.vectorstore = None

        st.session_state.all_documents = []

        st.session_state.chunks = []

        st.session_state.bm25_retriever = None

        st.session_state.ensemble_retriever = None

        st.session_state.drawing_pages = []

        st.session_state.chat_history = []


    # ========================================================
    # PROCESS NEW PROJECT
    # ========================================================

    if not st.session_state.project_loaded:

        if st.button(
            "🚀 Process & Index Project",
            type="primary",
            use_container_width=True,
            key="process_project_button",
        ):

            with st.status(
                "Processing construction project...",
                expanded=True,
            ) as status:

                project_dir = get_project_dir(
                    current_project_id
                )

                extract_dir = (
                    project_dir / "extracted"
                )

                extract_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                zip_path = (
                    project_dir
                    / uploaded_zip.name
                )

                zip_path.write_bytes(
                    uploaded_zip.getbuffer()
                )

                st.write(
                    "📦 Extracting ZIP..."
                )

                with zipfile.ZipFile(
                    zip_path,
                    "r",
                ) as zip_file:

                    zip_file.extractall(
                        extract_dir
                    )

                pdf_files = list(
                    extract_dir.rglob(
                        "*.pdf"
                    )
                )

                if not pdf_files:

                    status.update(
                        label="No PDF files found.",
                        state="error",
                    )

                    st.stop()

                st.write(
                    f"📄 Found {len(pdf_files)} PDF files."
                )

                all_documents = []

                drawing_pages = []

                total_pages = 0

                text_pages = 0

                image_pages = 0

                progress = st.progress(
                    0,
                    text="Reading PDF files...",
                )

                for (
                    index,
                    pdf_path,
                ) in enumerate(
                    pdf_files
                ):

                    relative_path = (
                        pdf_path.relative_to(
                            extract_dir
                        )
                    )

                    (
                        documents,
                        pdf_info,
                    ) = process_pdf(
                        str(pdf_path),
                        relative_path,
                    )

                    all_documents.extend(
                        documents
                    )

                    total_pages += (
                        pdf_info["pages"]
                    )

                    text_pages += (
                        pdf_info["text_pages"]
                    )

                    image_pages += (
                        pdf_info["image_only_pages"]
                    )

                    if (
                        detect_document_type(
                            relative_path
                        )
                        == "Drawing"
                    ):

                        for document in documents:

                            drawing_pages.append(
                                document
                            )

                    progress.progress(
                        (index + 1)
                        / len(pdf_files),
                        text=(
                            f"Reading "
                            f"{index + 1}/"
                            f"{len(pdf_files)} PDFs"
                        ),
                    )

                st.write(
                    f"🖼️ Indexed {len(drawing_pages)} "
                    "drawing pages."
                )

                chunks = create_chunks(
                    all_documents
                )

                if not chunks:

                    status.update(
                        label=(
                            "No usable text chunks found."
                        ),
                        state="error",
                    )

                    st.stop()

                st.write(
                    "🧠 Creating OpenAI embeddings..."
                )

                embedding_model = (
                    create_embedding_model()
                )

                documents_for_faiss = [

                    Document(
                        page_content=item["text"],
                        metadata=item["metadata"],
                    )

                    for item in chunks
                ]

                vectorstore = (
                    FAISS.from_documents(
                        documents_for_faiss,
                        embedding_model,
                    )
                )

                st.write(
                    "🔎 Building hybrid text retriever..."
                )

                (
                    bm25,
                    ensemble,
                ) = setup_hybrid_retriever(
                    chunks,
                    vectorstore,
                )

                stats = {

                    "pdf_files": len(
                        pdf_files
                    ),

                    "pages": total_pages,

                    "text_pages": text_pages,

                    "image_pages": image_pages,

                    "drawing_pages": len(
                        drawing_pages
                    ),

                    "chunks": len(
                        chunks
                    ),
                }

                st.write(
                    "💾 Saving persistent project index..."
                )

                save_persisted_project(
                    current_project_id,
                    vectorstore,
                    all_documents,
                    chunks,
                    stats,
                    drawing_pages,
                )

                st.session_state.vectorstore = (
                    vectorstore
                )

                st.session_state.all_documents = (
                    all_documents
                )

                st.session_state.chunks = (
                    chunks
                )

                st.session_state.bm25_retriever = (
                    bm25
                )

                st.session_state.ensemble_retriever = (
                    ensemble
                )

                st.session_state.drawing_pages = (
                    drawing_pages
                )

                st.session_state.project_stats = (
                    stats
                )

                st.session_state.project_id = (
                    current_project_id
                )

                st.session_state.project_dir = (
                    str(extract_dir)
                )

                st.session_state.project_loaded = (
                    True
                )

                st.session_state.chat_history = []

                status.update(
                    label=(
                        "✅ Project indexed successfully."
                    ),
                    state="complete",
                )

            st.rerun()


# ============================================================
# PROJECT DASHBOARD
# ============================================================
# QUESTION & ANSWER WORKSPACE
# ============================================================

def clear_question_box():
    """Clear the question text area through a Streamlit callback."""
    st.session_state.question_box = ""


if st.session_state.project_loaded:

    st.divider()

    st.header("💬 Ask Your Construction Project")

    st.write(
        "Type your question below. The AI will search your project "
        "documents and show the answer, sources and relevant drawings."
    )

    # --------------------------------------------------------
    # LARGE, VISIBLE QUESTION BOX
    # --------------------------------------------------------

    with st.container(border=True):

        st.subheader("🔎 Ask a Question")

        st.caption(
            "Ask about reinforcement, footing details, beam dimensions, "
            "drawing pages, RFIs, specifications and more."
        )

        question_input = st.text_area(
            "Your question",
            height=100,
            placeholder=(
                "Example: Give me the reinforcement details for F1 footing "
                "and tell me which drawing page contains them."
            ),
            key="question_box",
        )

        ask_col, clear_col = st.columns([4, 1])

        with ask_col:
            ask_clicked = st.button(
                "🔎 Get Answer",
                type="primary",
                use_container_width=True,
                key="get_answer_button",
            )

        with clear_col:
            st.button(
                "Clear",
                use_container_width=True,
                key="question_clear_button",
                on_click=clear_question_box,
            )

    # --------------------------------------------------------
    # QUICK QUESTIONS
    # --------------------------------------------------------

    st.write("**Quick questions**")

    examples = [
        "How many F1 footings are there?",
        "Give me the reinforcement details for F1 footing.",
        "Which page has the beam details?",
        "Show the relevant structural drawing pages.",
    ]

    example_columns = st.columns(4)

    for example_index, (column, example) in enumerate(
        zip(example_columns, examples)
    ):

        with column:

            if st.button(
                example,
                use_container_width=True,
                key=f"quick_question_{example_index}",
            ):

                question_input = example
                ask_clicked = True

    # --------------------------------------------------------
    # PROCESS QUESTION
    # --------------------------------------------------------

    if ask_clicked:

        question = (question_input or "").strip()

        if not question:

            st.warning(
                "Please type a question before clicking "
                "'Get Answer'."
            )

        else:

            st.session_state.chat_history.append(
                {
                    "role": "user",
                    "content": question,
                }
            )

            cached = get_cached_answer(
                st.session_state.project_id,
                question,
            )

            if cached:

                answer = cached.get(
                    "answer",
                    "",
                )

                sources = cached.get(
                    "sources",
                    [],
                )

                drawing_images = (
                    render_cached_drawing_sources(
                        cached.get(
                            "drawing_sources",
                            [],
                        )
                    )
                )

                is_cached = True

            else:

                with st.spinner(
                    "🔎 Searching your project and inspecting relevant drawings..."
                ):

                    try:

                        (
                            answer,
                            documents,
                            drawing_images,
                        ) = ask_project_assistant(
                            question
                        )

                        sources = serialize_sources(
                            documents
                        )

                        is_cached = False

                        cache_answer(
                            st.session_state.project_id,
                            question,
                            answer,
                            documents,
                            drawing_images,
                        )

                    except Exception as e:

                        st.error(
                            "AI analysis failed."
                        )

                        st.exception(e)

                        answer = (
                            "I could not generate an answer "
                            "because an error occurred."
                        )

                        sources = []
                        drawing_images = []
                        is_cached = False

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "drawing_images": drawing_images,
                    "is_cached": is_cached,
                }
            )

            # Do not modify st.session_state.question_box here.
            # The widget has already been instantiated in this run.
            # The next rerun will preserve the current text safely.
            st.rerun()

    # --------------------------------------------------------
    # LATEST QUESTION + ANSWER
    # --------------------------------------------------------

    latest_user = None
    latest_assistant = None

    for message in reversed(
        st.session_state.chat_history
    ):

        if latest_assistant is None and message.get("role") == "assistant":
            latest_assistant = message

        elif latest_user is None and message.get("role") == "user":
            latest_user = message

        if latest_user and latest_assistant:
            break

    if latest_assistant:

        st.divider()

        st.header("✅ Latest Answer")

        if latest_user:

            with st.container(border=True):

                st.caption("YOUR QUESTION")

                st.write(
                    latest_user.get(
                        "content",
                        "",
                    )
                )

        with st.container(border=True):

            st.subheader("🤖 AI Answer")

            st.markdown(
                latest_assistant.get(
                    "content",
                    "",
                )
            )

            if latest_assistant.get(
                "is_cached",
                False,
            ):

                st.caption(
                    "⚡ Answer served from answer_cache.json"
                )

        # ----------------------------------------------------
        # SOURCES
        # ----------------------------------------------------

        sources = latest_assistant.get(
            "sources",
            [],
        )

        if sources:

            st.subheader(
                "📚 Sources Used"
            )

            st.caption(
                "The documents below were retrieved to support the answer."
            )

            for index, source in enumerate(
                sources,
                start=1,
            ):

                with st.container(border=True):

                    source_col1, source_col2, source_col3 = st.columns(
                        [0.7, 3.8, 1.0]
                    )

                    with source_col1:

                        st.write(
                            f"**{index}**"
                        )

                    with source_col2:

                        st.caption("DOCUMENT")

                        st.write(
                            source.get(
                                "file_name",
                                "Unknown file",
                            )
                        )

                        st.caption(
                            f"Type: "
                            f"{source.get('document_type', 'Document')} "
                            f"• Discipline: "
                            f"{source.get('discipline', 'Unknown')}"
                        )

                    with source_col3:

                        st.caption("PAGE")

                        st.write(
                            f"**Page "
                            f"{source.get('page', '-')}**"
                        )

        # ----------------------------------------------------
        # DRAWINGS
        # ----------------------------------------------------

        latest_drawings = latest_assistant.get(
            "drawing_images",
            [],
        )

        if latest_drawings:

            st.subheader(
                "🖼️ Relevant Drawing Pages"
            )

            st.caption(
                "These drawing pages were inspected by the AI."
            )

            drawing_columns = st.columns(
                min(
                    len(latest_drawings),
                    2,
                )
            )

            for index, drawing in enumerate(
                latest_drawings
            ):

                with drawing_columns[
                    index % len(drawing_columns)
                ]:

                    meta = drawing.get(
                        "metadata",
                        {},
                    )

                    st.image(
                        drawing["bytes"],
                        caption=(
                            f'{meta.get("file_name", "Unknown")} '
                            f'— Page {meta.get("page", "-")}'
                        ),
                        use_container_width=True,
                    )

    # --------------------------------------------------------
    # OLD CONVERSATION HISTORY
    # --------------------------------------------------------

    if len(
        st.session_state.chat_history
    ) > 2:

        st.divider()

        with st.expander(
            "🕘 Previous Questions & Answers",
            expanded=False,
        ):

            for message in (
                st.session_state.chat_history[:-2]
            ):

                if message.get("role") == "user":

                    st.markdown(
                        f"**👤 Question:** "
                        f"{message.get('content', '')}"
                    )

                else:

                    st.markdown(
                        f"**🤖 Answer:** "
                        f"{message.get('content', '')}"
                    )

                    old_sources = message.get(
                        "sources",
                        [],
                    )

                    if old_sources:

                        st.caption(
                            "Sources: "
                            + ", ".join(
                                [
                                    (
                                        f'{s.get("file_name", "Unknown")} '
                                        f'(Page {s.get("page", "-")})'
                                    )
                                    for s in old_sources
                                ]
                            )
                        )




# ============================================================
# FOOTER
# ============================================================
st.divider()

st.caption(
    "🏗️ Your AI partner for smarter construction projects."
)

st.caption(
    "⚠️ AI results must be verified against approved "
    "construction drawings and specifications before execution."
)