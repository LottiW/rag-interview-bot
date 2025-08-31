from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from typing import List
from langchain_core.documents import Document
import os
from pinecone import Pinecone, ServerlessSpec
import time
import logging
import traceback
from .ocr_utils import extract_text_with_ocr, is_ocr_available, should_use_ocr
import streamlit as st

# Configure detailed logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

try:
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
except KeyError:
    logger.error("PINECONE_API_KEY not found in Streamlit secrets")
    PINECONE_API_KEY = None
PINECONE_CLOUD = st.secrets.get("PINECONE_CLOUD", "aws")
PINECONE_REGION = st.secrets.get("PINECONE_REGION", "us-east-1")
INDEX_NAME = st.secrets.get("PINECONE_INDEX_NAME", "recruiter-rag-index")
DIMENSION = 1536
PINECONE_ENVIRONMENT = st.secrets.get("PINECONE_ENVIRONMENT", None)
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", None)


# Validate configuration
def validate_pinecone_config():
    """Validate Pinecone configuration"""
    config_issues = []

    if not PINECONE_API_KEY:
        config_issues.append("PINECONE_API_KEY is not set")

    if not INDEX_NAME:
        config_issues.append("PINECONE_INDEX_NAME is not set")

    if PINECONE_ENVIRONMENT is None and (not PINECONE_CLOUD or not PINECONE_REGION):
        config_issues.append("For serverless: PINECONE_CLOUD and PINECONE_REGION must be set")

    if config_issues:
        logger.error("Pinecone configuration issues:")
        for issue in config_issues:
            logger.error(f"  - {issue}")
        return False

    logger.info("Pinecone configuration validated successfully")
    logger.info(f"  - Index Name: {INDEX_NAME}")
    logger.info(f"  - Cloud: {PINECONE_CLOUD}")
    logger.info(f"  - Region: {PINECONE_REGION}")
    logger.info(f"  - Environment: {PINECONE_ENVIRONMENT or 'Not set (using serverless)'}")
    return True


# Initialize components
try:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    logger.info("Text splitter initialized")

    embedding_function = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    logger.info("OpenAI embeddings initialized")
except Exception as e:
    logger.error(f"Error initializing basic components: {e}")
    raise

# Initialize Pinecone with detailed error handling
try:
    if not validate_pinecone_config():
        raise ValueError("Pinecone configuration validation failed")

    pc = Pinecone(api_key=PINECONE_API_KEY)
    logger.info("Pinecone client initialized successfully")
except Exception as e:
    logger.error(f"Error initializing Pinecone client: {e}")
    logger.error(f"Full traceback: {traceback.format_exc()}")
    pc = None


def initialize_pinecone_index():
    """Initialize Pinecone index with detailed debugging"""
    try:
        if pc is None:
            raise ValueError("Pinecone client not initialized")

        logger.info(f"Checking if index '{INDEX_NAME}' exists...")

        # List existing indexes with error handling
        try:
            existing_indexes = pc.list_indexes().names()
            logger.info(f"Found existing indexes: {existing_indexes}")
        except Exception as e:
            logger.error(f"Error listing indexes: {e}")
            logger.error(f"This might indicate API key or permission issues")
            raise

        if INDEX_NAME not in existing_indexes:
            logger.info(f"Index '{INDEX_NAME}' not found, creating new index...")

            try:
                # For Serverless (modern approach)
                if PINECONE_ENVIRONMENT is None:
                    logger.info(f"Creating serverless index with cloud={PINECONE_CLOUD}, region={PINECONE_REGION}")
                    pc.create_index(
                        name=INDEX_NAME,
                        dimension=DIMENSION,
                        metric="cosine",
                        spec=ServerlessSpec(
                            cloud=PINECONE_CLOUD,
                            region=PINECONE_REGION
                        )
                    )
                else:
                    # For Pod-based indexes (legacy)
                    logger.info(f"Creating pod-based index in environment={PINECONE_ENVIRONMENT}")
                    from pinecone import PodSpec
                    pc.create_index(
                        name=INDEX_NAME,
                        dimension=DIMENSION,
                        metric="cosine",
                        spec=PodSpec(
                            environment=PINECONE_ENVIRONMENT,
                            pod_type="p1.x1"
                        )
                    )

                logger.info("Index creation request sent, waiting for index to be ready...")

                # Wait for index to be ready with timeout
                max_wait_time = 60  # seconds
                wait_time = 0
                while wait_time < max_wait_time:
                    try:
                        index = pc.Index(INDEX_NAME)
                        # Test if index is ready
                        stats = index.describe_index_stats()
                        logger.info(f"Index is ready! Stats: {stats}")
                        break
                    except Exception as e:
                        logger.info(f"Index not ready yet, waiting... (error: {e})")
                        time.sleep(5)
                        wait_time += 5

                if wait_time >= max_wait_time:
                    raise TimeoutError("Index creation timed out")

            except Exception as e:
                logger.error(f"Error creating index: {e}")
                logger.error(f"Full traceback: {traceback.format_exc()}")
                raise
        else:
            logger.info(f"Index '{INDEX_NAME}' already exists")

        # Connect to the index
        logger.info("Connecting to Pinecone index...")
        index = pc.Index(INDEX_NAME)

        # Get detailed index information
        try:
            index_stats = index.describe_index_stats()
            logger.info(f"Index connection successful!")
            logger.info(f"Index host: {index._config.host}")
            logger.info(f"Index stats: {index_stats}")

            # Test a simple operation
            logger.info("Testing index with a simple query...")
            test_results = index.query(
                vector=[0.0] * DIMENSION,
                top_k=1,
                include_metadata=True
            )
            logger.info(f"Test query successful, found {len(test_results.matches)} matches")

        except Exception as e:
            logger.error(f"Error getting index stats or testing query: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            # Don't raise here, index might still be usable

        return index

    except Exception as e:
        logger.error(f"Critical error initializing Pinecone index: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise


# Initialize the index and vector store with extensive error handling
pinecone_index = None
vectorstore = None

try:
    logger.info("Starting Pinecone initialization...")
    pinecone_index = initialize_pinecone_index()

    logger.info("Creating PineconeVectorStore...")
    vectorstore = PineconeVectorStore(
        index=pinecone_index,
        embedding=embedding_function,
        text_key="text"
    )
    logger.info("Pinecone vector store initialized successfully!")

except Exception as e:
    logger.error(f"Failed to initialize Pinecone vector store: {e}")
    logger.error(f"Full traceback: {traceback.format_exc()}")
    logger.error("Application will continue but document indexing will not work")
    vectorstore = None


def load_and_split_document(file_path: str) -> List[Document]:
    """Load and split document with detailed debugging"""
    logger.info(f"Loading document: {file_path}")

    try:
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Check file size
        file_size = os.path.getsize(file_path)
        logger.info(f"File size: {file_size} bytes")

        if file_size == 0:
            raise ValueError(f"File is empty: {file_path}")

        # Load based on file type
        if file_path.endswith('.pdf'):
            logger.info("Loading PDF document...")
            loader = PyPDFLoader(file_path)
            documents = loader.load()

        elif file_path.endswith('.docx'):
            logger.info("Loading DOCX document...")
            loader = Docx2txtLoader(file_path)
            documents = loader.load()

        elif file_path.endswith('.html'):
            logger.info("Loading HTML document...")
            loader = UnstructuredHTMLLoader(file_path)
            documents = loader.load()

        elif file_path.endswith(('.txt', '.md')):
            logger.info("Loading text/markdown document...")
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                documents = [Document(page_content=content, metadata={"source": file_path})]
        else:
            raise ValueError(f"Unsupported file type: {file_path}")

        logger.info(f"Loaded {len(documents)} document(s)")

        # Enhanced content analysis
        total_content_length = 0
        for i, doc in enumerate(documents):
            doc_length = len(doc.page_content) if doc.page_content else 0
            total_content_length += doc_length
            logger.debug(f"Document {i}: {doc_length} characters")

            # Log first few characters of each document for debugging
            if doc.page_content:
                preview = doc.page_content[:100].strip()
                if preview:
                    logger.debug(f"Document {i} preview: {preview}...")
                else:
                    logger.debug(f"Document {i}: content exists but appears to be whitespace only")
            else:
                logger.debug(f"Document {i}: no content (None or empty)")

        logger.info(f"Total content length across all documents: {total_content_length} characters")

        # Enhanced content preview
        if documents and total_content_length > 0:
            # Find first non-empty document for preview
            for doc in documents:
                if doc.page_content and doc.page_content.strip():
                    preview = doc.page_content.strip()[:200] + "..." if len(
                        doc.page_content.strip()) > 200 else doc.page_content.strip()
                    logger.info(f"First non-empty content preview: {preview}")
                    break
            else:
                logger.warning("All documents appear to have empty or whitespace-only content")
        else:
            logger.warning("No meaningful content found in any document")

        # Split documents only if we have content
        if total_content_length > 0:
            logger.info("Splitting documents into chunks...")
            splits = text_splitter.split_documents(documents)
            logger.info(f"Created {len(splits)} chunks")

            # Log chunk details
            if splits:
                chunk_sizes = [len(split.page_content) for split in splits]
                non_empty_chunks = [s for s in chunk_sizes if s > 0]
                logger.info(f"Chunk analysis: {len(non_empty_chunks)} non-empty chunks out of {len(splits)} total")
                if non_empty_chunks:
                    logger.info(
                        f"Chunk sizes: min={min(non_empty_chunks)}, max={max(non_empty_chunks)}, avg={sum(non_empty_chunks) / len(non_empty_chunks):.1f}")

                # Preview first non-empty chunk
                for chunk in splits:
                    if chunk.page_content.strip():
                        chunk_preview = chunk.page_content.strip()[:150] + "..." if len(
                            chunk.page_content.strip()) > 150 else chunk.page_content.strip()
                        logger.debug(f"First chunk preview: {chunk_preview}")
                        break
            else:
                logger.warning("Text splitter created no chunks despite having content")
        else:
            logger.info("Skipping text splitting due to no content")
            splits = []

        return splits

    except Exception as e:
        logger.error(f"Error loading document {file_path}: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return []


def index_document_to_pinecone(file_path: str, file_id: int, use_ocr_fallback: bool = True) -> bool:
    """
    Enhanced version with proper OCR fallback logic
    """
    try:
        logger.info(f"Starting indexing process for file_id {file_id}: {file_path}")

        # Validate prerequisites
        if vectorstore is None:
            logger.error("Vector store not initialized")
            return False

        # First stage: Standard text extraction
        documents = load_and_split_document(file_path)

        # Calculate total content from standard extraction
        total_content = ""
        for doc in documents:
            if doc.page_content:
                total_content += doc.page_content

        logger.info(f"Standard extraction yielded {len(total_content)} characters from {len(documents)} documents")

        # OCR fallback logic - trigger if we have little/no content
        ocr_used = False
        if len(total_content.strip()) < 50 and use_ocr_fallback and should_use_ocr(file_path) and is_ocr_available():
            logger.info(
                f"Standard text extraction yielded minimal content ({len(total_content)} chars). Attempting OCR...")

            ocr_text, ocr_metadata = extract_text_with_ocr(file_path)

            if ocr_text and len(ocr_text.strip()) > 50:
                logger.info(f"OCR successful: extracted {len(ocr_text)} characters")
                ocr_used = True

                # Create new Document with OCR text
                documents = [Document(
                    page_content=ocr_text,
                    metadata={
                        "source": file_path,
                        "file_id": file_id,
                        "extraction_method": "mistral_ocr",
                        "ocr_metadata": str(ocr_metadata) if ocr_metadata else None
                    }
                )]
                total_content = ocr_text  # Update total content
            else:
                logger.warning(f"OCR also failed to extract meaningful text from {file_path}")

        # Final check - do we have any usable content?
        if not documents or len(total_content.strip()) < 10:
            logger.error(f"No usable content found after both standard and OCR extraction: {file_path}")
            return False

        # Update metadata for all documents
        for doc in documents:
            if "file_id" not in doc.metadata:
                doc.metadata["file_id"] = file_id
            if "extraction_method" not in doc.metadata:
                doc.metadata["extraction_method"] = "mistral_ocr" if ocr_used else "standard"

        # Split documents into chunks (only if we have content)
        logger.info(f"Splitting documents into chunks...")
        chunks = text_splitter.split_documents(documents)

        if not chunks:
            logger.error(f"No chunks created from documents: {file_path}")
            return False

        logger.info(f"Created {len(chunks)} chunks")

        # Validate chunks have content
        valid_chunks = []
        for i, chunk in enumerate(chunks):
            if chunk.page_content.strip():
                # Ensure metadata is properly set
                if "file_id" not in chunk.metadata:
                    chunk.metadata["file_id"] = file_id
                chunk.metadata["chunk_id"] = f"{file_id}_{i}"
                valid_chunks.append(chunk)
            else:
                logger.warning(f"Skipping empty chunk {i}")

        if not valid_chunks:
            logger.error("No valid chunks with content found")
            return False

        logger.info(f"Processing {len(valid_chunks)} valid chunks")

        # Log extraction method used
        extraction_method = "OCR (Mistral)" if ocr_used else "Standard PDF parsing"
        logger.info(f"Final extraction method used: {extraction_method}")

        # Log chunk content preview
        if valid_chunks:
            first_chunk_preview = valid_chunks[0].page_content[:200] + "..." if len(
                valid_chunks[0].page_content) > 200 else valid_chunks[0].page_content
            logger.info(f"First chunk preview: {first_chunk_preview}")

        # Index chunks to Pinecone
        logger.info("Starting Pinecone indexing...")

        try:
            # Add documents to vector store
            vectorstore.add_documents(valid_chunks)
            logger.info(f"Successfully indexed {len(valid_chunks)} chunks to Pinecone using {extraction_method}")

            # Wait a bit for indexing to complete
            time.sleep(1)

            # Verify indexing by searching for a document with this file_id
            logger.info("Verifying indexing...")
            verification_results = vectorstore.similarity_search(
                query="test",
                k=1,
                filter={"file_id": file_id}
            )

            if verification_results:
                logger.info(f"Indexing verification successful: found {len(verification_results)} documents")
            else:
                logger.warning("Indexing verification failed: no documents found with file_id filter")
                # Don't fail here, as the documents might still be indexed but filter might not work as expected

            return True

        except Exception as e:
            logger.error(f"Error during Pinecone indexing: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False

    except Exception as e:
        logger.error(f"Error indexing document to Pinecone: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return False


def delete_doc_from_pinecone(file_id: int) -> bool:
    """Delete document from Pinecone with detailed debugging"""
    logger.info(f"Starting deletion process for file_id: {file_id}")

    try:
        if vectorstore is None:
            logger.error("Vector store not initialized")
            return False

        # First, check what we're about to delete
        logger.info(f"Searching for documents with file_id {file_id} before deletion...")
        try:
            existing_docs = vectorstore.similarity_search("", k=100, filter={"file_id": file_id})
            logger.info(f"Found {len(existing_docs)} documents to delete")
        except Exception as e:
            logger.warning(f"Could not count existing documents: {e}")

        # Get the underlying Pinecone index for deletion
        index = vectorstore.index
        logger.info(f"Using Pinecone index: {index._config.host}")

        # Delete by metadata filter
        logger.info(f"Executing delete operation for file_id {file_id}...")
        delete_response = index.delete(filter={"file_id": {"$eq": file_id}})

        logger.info(f"Delete operation completed. Response: {delete_response}")

        # Wait a bit for the deletion to propagate
        time.sleep(2)

        # Verify deletion
        logger.info("Verifying deletion...")
        try:
            remaining_docs = vectorstore.similarity_search("", k=10, filter={"file_id": file_id})
            if remaining_docs:
                logger.warning(f"Deletion verification failed: still found {len(remaining_docs)} documents")
            else:
                logger.info("Deletion verification successful: no documents found")
        except Exception as e:
            logger.warning(f"Could not verify deletion: {e}")

        logger.info(f"Successfully deleted all documents with file_id {file_id}")
        return True

    except Exception as e:
        logger.error(f"Error deleting document with file_id {file_id}: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return False


def get_retriever(search_kwargs=None):
    """Get Pinecone retriever with debugging"""
    logger.debug("Creating Pinecone retriever...")

    if vectorstore is None:
        logger.error("Cannot create retriever: vector store not initialized")
        raise ValueError("Vector store not initialized")

    if search_kwargs is None:
        search_kwargs = {
            "k": 6,
            "score_threshold": 0.6
        }

    logger.debug(f"Retriever search kwargs: {search_kwargs}")

    return vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs=search_kwargs
    )


def search_documents(query: str, k: int = 4, filter_metadata: dict = None) -> List[Document]:
    """Search documents with debugging"""
    logger.info(f"Searching documents: query='{query[:50]}...', k={k}, filter={filter_metadata}")

    try:
        if vectorstore is None:
            logger.error("Vector store not initialized")
            return []

        if filter_metadata:
            logger.info("Using metadata filtering")
            results = vectorstore.similarity_search(
                query=query,
                k=k,
                filter=filter_metadata
            )
        else:
            results = vectorstore.similarity_search(query=query, k=k)

        logger.info(f"Found {len(results)} results")
        return results

    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return []


def get_index_stats():
    """Get Pinecone index statistics with error handling"""
    logger.debug("Getting Pinecone index statistics...")

    try:
        if vectorstore is None:
            logger.error("Vector store not initialized")
            return {"error": "Vector store not initialized"}

        index = vectorstore.index
        stats = index.describe_index_stats()

        logger.debug(f"Index stats retrieved: {stats}")

        return {
            "total_vector_count": stats.get('total_vector_count', 0),
            "dimension": stats.get('dimension', 0),
            "index_fullness": stats.get('index_fullness', 0.0),
            "namespaces": stats.get('namespaces', {})
        }
    except Exception as e:
        logger.error(f"Failed to get index stats: {e}")
        return {"error": f"Failed to get index stats: {str(e)}"}


def test_pinecone_connection():
    """Test Pinecone connection with comprehensive diagnostics"""
    logger.info("Testing Pinecone connection...")

    try:
        if vectorstore is None:
            return {"status": "failed", "message": "Vector store not initialized"}

        # Test 1: Basic connection
        logger.info("Test 1: Testing basic similarity search...")
        test_query = "test connection"
        results = vectorstore.similarity_search(test_query, k=1)
        logger.info(f"Basic search successful, found {len(results)} results")

        # Test 2: Get statistics
        logger.info("Test 2: Getting index statistics...")
        stats = get_index_stats()
        logger.info(f"Statistics retrieved: {stats}")

        # Test 3: Test embedding
        logger.info("Test 3: Testing embedding generation...")
        test_embedding = embedding_function.embed_query("test")
        logger.info(f"Embedding successful, dimension: {len(test_embedding)}")

        return {
            "status": "success",
            "message": "All Pinecone tests passed",
            "stats": stats,
            "tests_passed": ["similarity_search", "index_stats", "embedding_generation"]
        }

    except Exception as e:
        logger.error(f"Pinecone connection test failed: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")

        return {
            "status": "failed",
            "message": f"Pinecone connection test failed: {str(e)}",
            "error_details": traceback.format_exc()
        }