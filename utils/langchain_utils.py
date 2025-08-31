# langchain_utils.py - Fixed version with proper error handling
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from typing import List, Optional
from langchain_core.documents import Document
import os
import streamlit as st
import logging

# Setup logging
logger = logging.getLogger(__name__)

# Get API key with proper error handling
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is empty")
    if not OPENAI_API_KEY.startswith('sk-'):
        raise ValueError("OPENAI_API_KEY format is invalid")
except KeyError:
    logger.error("OPENAI_API_KEY not found in Streamlit secrets")
    OPENAI_API_KEY = None
except Exception as e:
    logger.error(f"Error getting OPENAI_API_KEY: {e}")
    OPENAI_API_KEY = None

# Import retriever with proper error handling
retriever = None
vectorstore = None

try:
    if OPENAI_API_KEY:
        # GEÄNDERT: Importiere sowohl get_retriever als auch vectorstore
        from .pinecone_utils import get_retriever, vectorstore as pinecone_vectorstore

        retriever = get_retriever()

        # Vectorstore verfügbar machen
        vectorstore = pinecone_vectorstore

        logger.info("Retriever and vectorstore initialized successfully")
    else:
        logger.error("Cannot initialize retriever without valid OpenAI API key")
except ImportError as e:
    logger.error(f"Import error for pinecone_utils: {e}")
    logger.info("Running in fallback mode without Pinecone retriever")
except Exception as e:
    logger.error(f"Error initializing retriever: {e}")
    logger.info("Running in fallback mode")

output_parser = StrOutputParser()

# Improved contextualization prompt
contextualize_q_system_prompt = (
    "Du bist ein AI-Assistant, der Recruitern dabei hilft, umfassende Informationen über Charlotte Wangemann zu erhalten. "
    "Gegeben ist eine Chat-Historie und die neueste Benutzerfrage, die sich möglicherweise auf den Kontext "
    "in der Chat-Historie bezieht. Formuliere eine präzise, eigenständige Frage, die alle relevanten Details "
    "aus dem Dokument abrufen kann und eine vollständige Antwort in Fließtext ermöglicht. "
    "Beantworte die Frage NICHT, sondern erweitere sie bei Bedarf um wichtige Aspekte wie Zeiträume, "
    "spezifische Aufgaben, Erfolge und Qualifikationen. "
    "Fokussiere auf Recruiter-relevante Informationen wie berufliche Erfahrungen, Fähigkeiten, "
    "Ausbildung, Projekte und Erfolge."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Specialized QA prompt for recruiter requests
qa_system_prompt = """Du bist ein professioneller AI-Assistant für Charlotte Wangemann, der Recruitern detaillierte, narrative Antworten in natürlichem Fließtext bereitstellt.

HAUPTZIEL: Erstelle umfassende, flüssig lesbare Antworten, die alle relevanten Details aus dem Kontext in zusammenhängenden Absätzen präsentieren.

ANTWORT-STIL:
- Verwende AUSSCHLIESSLICH natürlichen Fließtext in gut strukturierten Absätzen
- KEINE Aufzählungspunkte, Listen oder Bullet Points
- KEINE Emojis oder spezielle Formatierungen
- Schreibe wie ein professioneller Recruiter-Brief oder Bewerbungstext
- Verbinde zusammengehörige Informationen in logischen Absätzen
- Verwende Übergänge und Bindewörter für flüssigen Lesefluss

INHALTLICHE ANFORDERUNGEN:
1. NUR FAKTEN: Verwende AUSSCHLIESSLICH Informationen aus dem bereitgestellten Kontext
2. KEINE ERFINDUNGEN: Erfinde NIEMALS Details, die nicht explizit im Kontext stehen
3. VOLLSTÄNDIGKEIT: Integriere ALLE relevanten Details aus dem Kontext in die Antwort
4. ZUSAMMENHANG: Verknüpfe verwandte Informationen thematisch in Absätzen
5. ZEITLICHER KONTEXT: Erwähne Zeiträume und chronologische Zusammenhänge

STRUKTUR FÜR UMFASSENDE ANTWORTEN:
- Einleitender Überblick über das angefragte Thema
- Detaillierte Darstellung aller relevanten Aspekte in zusammenhängenden Absätzen
- Chronologische oder thematische Gruppierung von Informationen
- Abschließende Einordnung oder Zusammenfassung

BEISPIEL-TRANSFORMATION:
❌ Schlecht (Listen-Format):
"Charlotte hat folgende Positionen:
• Werkstudentin bei SAP SE (2024-2025)
• Studentische Hilfskraft Uni Leipzig (2023-2024)"

✅ Gut (Fließtext-Format):
"Charlotte sammelte ihre beruflichen Erfahrungen in zwei wesentlichen Bereichen. Von Oktober 2023 bis September 2024 war sie als studentische Hilfskraft am Lehrstuhl für Arbeits- und Organisationspsychologie der Universität Leipzig tätig, wo sie bei Literaturrecherchen, der Erstellung von Literaturverzeichnissen und der Vorbereitung von Meta-Analysen mitwirkte. Parallel dazu begann sie im Oktober 2024 ihre Tätigkeit als Werkstudentin im Transformation Office bei SAP SE, wo sie eine organisationsweite Cloud-ERP-Transformation unterstützt und eigenständig Interviews mit Führungskräften konzipiert und analysiert."

KONTEXTINFORMATIONEN:
{context}

WICHTIGE REGELN:
- Bei unvollständigen Informationen: Sage klar was verfügbar ist und was fehlt
- Bei fehlenden Informationen: "Diese spezifische Information liegt mir nicht vor"
- Verwende Formulierungen wie "basierend auf den vorliegenden Informationen"
- Zitiere niemals wörtlich, sondern integriere Informationen natürlich
- Schreibe immer in der dritten Person über Charlotte"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])


def get_rag_chain(model="gpt-4o-mini"):
    """Creates a RAG chain with enhanced validation and better retrieval"""

    if not OPENAI_API_KEY:
        raise ValueError("OpenAI API key not available")

    # Verwende ein präziseres Modell für wichtige Anfragen
    if "karriere" in model.lower() or "beruf" in model.lower():
        model = "gpt-4o"

    llm = ChatOpenAI(
        model=model,
        temperature=0.1,
        max_tokens=4096,
        api_key=OPENAI_API_KEY
    )

    # Check if retriever is available
    if retriever is None:
        logger.warning("Retriever not available, creating fallback chain")
        return create_fallback_chain(llm)

    try:
        # HIER IST DIE ÄNDERUNG: Enhanced retriever mit mehr Kontext
        enhanced_retriever = get_enhanced_retriever()

        # History-aware retriever for contextual search
        history_aware_retriever = create_history_aware_retriever(
            llm,
            enhanced_retriever,  # Verwende den enhanced retriever
            contextualize_q_prompt
        )

        # Question-answer chain with specialized prompt
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

        # Complete RAG chain
        rag_chain = create_retrieval_chain(
            history_aware_retriever,
            question_answer_chain
        )

        logger.info("RAG chain created successfully with enhanced retriever")
        return rag_chain

    except Exception as e:
        logger.error(f"Error creating RAG chain: {e}")
        logger.info("Falling back to simple chain without retriever")
        return create_fallback_chain(llm)


def get_enhanced_retriever():
    """Enhanced retriever with more comprehensive search parameters"""
    if vectorstore is None:
        logger.error("Cannot create retriever: vector store not initialized")
        raise ValueError("Vector store not initialized")

    # HIER SIND DIE SEARCH_KWARGS ANPASSUNGEN:
    enhanced_search_kwargs = {
        "k": 8,  # Mehr Dokumente abrufen für umfassendere Antworten
        "score_threshold": 0.6  # Niedrigere Schwelle für mehr Kontext
    }

    logger.debug(f"Enhanced retriever search kwargs: {enhanced_search_kwargs}")

    return vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs=enhanced_search_kwargs
    )


# Zusätzlich: Update der bestehenden get_retriever Funktion
def get_retriever(search_kwargs=None):
    """Get Pinecone retriever with debugging - updated with better defaults"""
    logger.debug("Creating Pinecone retriever...")

    if vectorstore is None:
        logger.error("Cannot create retriever: vector store not initialized")
        raise ValueError("Vector store not initialized")

    if search_kwargs is None:
        # HIER AUCH ANGEPASST: Bessere Default-Werte für umfassendere Antworten
        search_kwargs = {
            "k": 6,  # Erhöht von 4 auf 6
            "score_threshold": 0.6  # Gesenkt von 0.5 auf 0.6
        }

    logger.debug(f"Retriever search kwargs: {search_kwargs}")

    return vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs=search_kwargs
    )


def create_fallback_chain(llm):
    """Create a simple chain when retriever is not available"""

    fallback_prompt = ChatPromptTemplate.from_messages([
        ("system", """Du bist ein professioneller AI-Assistant für Recruiter. 
        Du hilfst bei Fragen zu einem Kandidaten, aber hast derzeit keinen Zugriff auf spezifische Dokumente.

        Gib ehrlich zu, dass du keine spezifischen Dokumentinformationen abrufen kannst, 
        aber biete trotzdem hilfreiche allgemeine Antworten und Vorschläge für direkten Kontakt."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    return fallback_prompt | llm | output_parser


def get_standard_responses():
    """Predefined responses for common recruiter questions"""
    return {
        "kontakt": {
            "keywords": ["kontakt", "email", "telefon", "erreichen"],
            "response": """
📧 **Kontaktinformationen:**
- E-Mail: charly.wangemann@icloud.com
- LinkedIn: [LinkedIn Profil](https://www.linkedin.com/in/charlotte-wangemann/)

Ich freue mich über Ihre Kontaktaufnahme und stehe gerne für ein persönliches Gespräch zur Verfügung!
            """
        }
    }


def check_for_standard_response(query: str) -> Optional[str]:
    """
    Check if a standard response exists for the query

    Args:
        query (str): The user query

    Returns:
        str: Standard response or None
    """
    query_lower = query.lower()
    standard_responses = get_standard_responses()

    for response_type, response_data in standard_responses.items():
        if any(keyword in query_lower for keyword in response_data["keywords"]):
            return response_data["response"]

    return None


def get_retriever_status():
    """Get status of the retriever for debugging"""
    return {
        "retriever_initialized": retriever is not None,
        "retriever_type": type(retriever).__name__ if retriever else None,
        "openai_key_available": OPENAI_API_KEY is not None,
        "openai_key_valid_format": OPENAI_API_KEY.startswith('sk-') if OPENAI_API_KEY else False
    }


def test_rag_chain():
    """Test function to validate RAG chain functionality"""
    try:
        status = get_retriever_status()
        logger.info(f"RAG Chain Status: {status}")

        if not OPENAI_API_KEY:
            return {"status": "error", "message": "OpenAI API key not available"}

        # Try to create a chain
        chain = get_rag_chain()

        # Test with a simple query
        test_response = chain.invoke({
            "input": "Hallo, können Sie sich vorstellen?",
            "chat_history": []
        })

        return {
            "status": "success",
            "message": "RAG chain working",
            "retriever_available": retriever is not None,
            "test_response": test_response if isinstance(test_response, str) else str(test_response)[:100]
        }

    except Exception as e:
        logger.error(f"RAG chain test failed: {e}")
        return {"status": "error", "message": f"RAG chain test failed: {str(e)}"}