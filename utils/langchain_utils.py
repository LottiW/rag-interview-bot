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
try:
    if OPENAI_API_KEY:
        from .pinecone_utils import get_retriever

        retriever = get_retriever(
            search_kwargs={
                "k": 3,
                "score_threshold": 0.7
            }
        )
        logger.info("Retriever initialized successfully")
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
    "Du bist ein AI-Assistant, der Recruitern dabei hilft, Informationen über einen Kandidaten zu erhalten. "
    "Gegeben ist eine Chat-Historie und die neueste Benutzerfrage, die sich möglicherweise auf den Kontext "
    "in der Chat-Historie bezieht. Formuliere eine eigenständige Frage, die ohne die Chat-Historie "
    "verstanden werden kann. "
    "Beantworte die Frage NICHT, sondern formuliere sie nur um, falls nötig, ansonsten gib sie unverändert zurück. "
    "Achte besonders auf Recruiter-spezifische Begriffe wie 'Skills', 'Erfahrung', 'Verfügbarkeit', etc."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Specialized QA prompt for recruiter requests
qa_system_prompt = """Du bist ein AI-Assistant für Charlotte Wangemann, der Recruitern AUSSCHLIESSLICH verifizierte Informationen bereitstellt.

🚨 KRITISCHE REGELN - NIEMALS BRECHEN:
1. ✅ NUR FAKTEN: Verwende AUSSCHLIESSLICH Informationen aus dem bereitgestellten Kontext
2. ❌ KEINE ERFINDUNGEN: Erfinde NIEMALS Details, die nicht explizit im Kontext stehen
3. 🔍 VERIFIZIERUNG: Wenn eine Information nicht im Kontext steht, sage: "Diese Information liegt mir nicht vor"
4. 📝 WÖRTLICH: Bei konkreten Fakten (Positionen, Zeiträume, Projekte) bleibe EXAKT beim Kontext

WICHTIGE UNTERSCHEIDUNGEN:
- Werkstudium ≠ Praktikum (verwende die exakte Bezeichnung aus dem Kontext)
- Verwende nur die tatsächlichen Projektnamen aus dem Kontext
- Nenne nur die Ehrenämter, die explizit erwähnt werden

ANTWORTSTRUKTUR:
1. Prüfe zuerst: Steht die gefragte Information im Kontext?
   - JA → Antworte präzise mit den vorhandenen Informationen
   - NEIN → "Diese spezifische Information liegt mir nicht vor. Für Details kontaktieren Sie bitte Charlotte direkt."

2. Bei teilweisen Informationen:
   - Sage klar, was du weißt: "Basierend auf den vorliegenden Informationen..."
   - Sage klar, was fehlt: "Zu [spezifischer Aspekt] liegen mir keine Details vor."

KONTEXTINFORMATIONEN:
{context}

NIEMALS:
- Interpretiere oder extrapoliere über den Kontext hinaus
- Füge plausible aber nicht belegte Details hinzu
- Rate oder vermute bei fehlenden Informationen
- Verwende Standardantworten, wenn spezifische Infos fehlen

IMMER:
- Zitiere nur exakt aus dem Kontext
- Kennzeichne Unsicherheiten deutlich
- Verweise bei fehlenden Infos auf direkten Kontakt"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])


def get_rag_chain(model="gpt-4o-mini"):
    """Creates a RAG chain with enhanced validation"""

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
        # History-aware retriever for contextual search
        history_aware_retriever = create_history_aware_retriever(
            llm,
            retriever,
            contextualize_q_prompt
        )

        # Question-answer chain with specialized prompt
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

        # Complete RAG chain
        rag_chain = create_retrieval_chain(
            history_aware_retriever,
            question_answer_chain
        )

        logger.info("RAG chain created successfully with retriever")
        return rag_chain

    except Exception as e:
        logger.error(f"Error creating RAG chain: {e}")
        logger.info("Falling back to simple chain without retriever")
        return create_fallback_chain(llm)


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
        },
        "verfügbarkeit": {
            "keywords": ["verfügbar", "kündigungsfrist", "start", "wechsel"],
            "response": """
📅 **Verfügbarkeit:**
- Kündigungsfrist: 3 Monate (verhandelbar je nach Situation)
- Bereitschaft für Remote-Work: Ja
- Bereitschaft für Umzug: Je nach Standort

Für spezifische Zeitpläne können wir gerne direkt sprechen!
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