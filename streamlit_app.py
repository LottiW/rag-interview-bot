# streamlit_app.py - Standalone Version für Streamlit Cloud
import streamlit as st
import os
from datetime import datetime
import uuid

# Personal Information Secrets
# Diese müssen in den Streamlit Cloud Secrets konfiguriert werden:
PERSON_NAME = st.secrets.get("PERSON_NAME", "Charlotte Wangemann")
PERSON_EMAIL = st.secrets.get("PERSON_EMAIL", "charly.wangemann@icloud.com")
PERSON_LINKEDIN = st.secrets.get("PERSON_LINKEDIN", "https://www.linkedin.com/in/charlotte-wangemann/")

# Page config
st.set_page_config(
    page_title=f"{PERSON_NAME} - AI Assistant",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Direct imports statt API calls
from utils.langchain_utils import get_rag_chain, check_for_standard_response
from utils.db_utils import (
    insert_application_logs,
    get_chat_history,
    get_all_documents,
    insert_document_record,
    delete_document_record
)
from utils.pinecone_utils import (
    index_document_to_pinecone,
    delete_doc_from_pinecone
)

# Environment variables für Streamlit Cloud
# Diese müssen in den Streamlit Cloud Settings konfiguriert werden:
# OPENAI_API_KEY, PINECONE_API_KEY, DATABASE_URL, MISTRAL_API_KEY

def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": f"Hallo! Ich kann Ihnen Fragen zu {PERSON_NAME}s beruflichem Werdegang beantworten."
        }]

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    if "model" not in st.session_state:
        st.session_state.model = "gpt-4o-mini"


def chat_with_assistant(question: str, session_id: str, model: str):
    """Direkte Chat-Funktion statt API call"""
    try:
        # Standard responses prüfen
        standard_response = check_for_standard_response(question)

        if standard_response:
            answer = standard_response
        else:
            # RAG Chain verwenden
            chat_history = get_chat_history(session_id)
            rag_chain = get_rag_chain(model)

            result = rag_chain.invoke({
                "input": question,
                "chat_history": chat_history
            })

            answer = result['answer']

        # Log interaction
        insert_application_logs(session_id, question, answer, model)

        return {"answer": answer, "session_id": session_id, "model": model}

    except Exception as e:
        st.error(f"Fehler bei der Antwortgenerierung: {str(e)}")
        return {
            "answer": "Entschuldigung, es gab ein technisches Problem. Bitte versuchen Sie es erneut.",
            "session_id": session_id,
            "model": model
        }


def upload_document_direct(uploaded_file):
    """Direkter File Upload ohne FastAPI - temporäre Verarbeitung"""
    try:
        # Erstelle temporäre Datei im /tmp Verzeichnis (Streamlit Cloud kompatibel)
        temp_path = f"/tmp/temp_{uuid.uuid4()}_{uploaded_file.name}"

        # Schreibe Upload in temporäre Datei
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())  # getvalue() für UploadedFile

        # Database record erstellen
        file_id = insert_document_record(
            uploaded_file.name,
            uploaded_file.size,
            os.path.splitext(uploaded_file.name)[1]
        )

        # Extrahiere und indexiere Content zu Pinecone
        success = index_document_to_pinecone(temp_path, file_id, use_ocr_fallback=True)

        # WICHTIG: Temporäre Datei sofort löschen (kein persistenter Storage nötig)
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except:
            pass  # Cleanup Fehler ignorieren

        if success:
            return {
                "message": f"'{uploaded_file.name}' erfolgreich verarbeitet und indexiert!",
                "file_id": file_id,
                "status": "success",
                "note": "Datei wurde verarbeitet und sicher gelöscht. Content ist in der Wissensdatenbank verfügbar."
            }
        else:
            # Bei Fehler auch DB Record löschen
            delete_document_record(file_id)
            return {"message": "Indexierung fehlgeschlagen", "status": "error"}

    except Exception as e:
        # Cleanup bei Fehler
        try:
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.remove(temp_path)
        except:
            pass
        return {"message": f"Fehler bei Upload: {str(e)}", "status": "error"}


def main():
    initialize_session_state()

    # Vollständiges CSS Design aus der ursprünglichen Version
    st.markdown("""
    <style>
    /* Clean Background */
    .stApp {
        background-color: #FAFAFA;
    }

    /* Simple Header */
    .main-header {
        font-size: 2.5rem;
        color: #2C3E50;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 300;
        letter-spacing: -1px;
    }

    .sub-header {
        font-size: 1.1rem;
        color: #7F8C8D;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }

    /* Clean Info Box */
    .info-box {
        background: #FFFFFF;
        color: #2C3E50;
        padding: 2rem;
        border-radius: 8px;
        margin-bottom: 2rem;
        border: 1px solid #E8E8E8;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }

    .info-box h4 {
        color: #2C3E50;
        margin-bottom: 1rem;
        font-weight: 400;
    }

    .info-box ul li {
        color: #5D6D7E;
        margin-bottom: 0.5rem;
        font-size: 0.95rem;
    }

    /* Minimal Chat Styling */
    .stChatMessage {
        background: #FFFFFF !important;
        border: 1px solid #E8E8E8 !important;
        border-radius: 6px !important;
        margin-bottom: 1rem !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
    }

    /* Clean Input */
    .stChatInput > div > div > input {
        background-color: #FFFFFF !important;
        color: #2C3E50 !important;
        border: 1px solid #D5D8DC !important;
        border-radius: 6px !important;
        padding: 12px 16px !important;
        font-size: 0.95rem !important;
    }

    .stChatInput > div > div > input:focus {
        border-color: #3498DB !important;
        box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.1) !important;
    }

    /* Minimal Buttons */
    .stButton > button {
        background: #FFFFFF !important;
        color: #2C3E50 !important;
        border: 1px solid #D5D8DC !important;
        border-radius: 6px !important;
        padding: 0.6rem 1.2rem !important;
        font-weight: 400 !important;
        font-size: 0.9rem !important;
        transition: all 0.2s ease !important;
        box-shadow: none !important;
    }

    .stButton > button:hover {
        background: #F8F9FA !important;
        border-color: #3498DB !important;
        color: #3498DB !important;
    }

    /* Sidebar Clean */
    .css-1d391kg {
        background: #FFFFFF !important;
        border-right: 1px solid #E8E8E8 !important;
    }

    /* Text */
    .stMarkdown {
        color: #2C3E50 !important;
        line-height: 1.6 !important;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #2C3E50 !important;
        font-weight: 400 !important;
    }

    p {
        color: #5D6D7E !important;
        line-height: 1.6 !important;
    }

    /* Clean Code Blocks */
    .stCode {
        background: #F8F9FA !important;
        color: #2C3E50 !important;
        border: 1px solid #E8E8E8 !important;
        border-radius: 4px !important;
    }

    /* Subtle Alerts */
    .stSuccess {
        background: #F7F7F7 !important;
        color: #27AE60 !important;
        border: 1px solid #D5E8D4 !important;
        border-radius: 4px !important;
    }

    .stError {
        background: #F7F7F7 !important;
        color: #E74C3C !important;
        border: 1px solid #F5C6CB !important;
        border-radius: 4px !important;
    }

    .stInfo {
        background: #F7F7F7 !important;
        color: #3498DB !important;
        border: 1px solid #D1ECF1 !important;
        border-radius: 4px !important;
    }

    /* Contact info styling */
    .contact-info {
        background: #F8F9FA;
        color: #2C3E50;
        padding: 1.5rem;
        border-radius: 6px;
        margin: 1rem 0;
        border: 1px solid #E8E8E8;
    }

    .contact-info h4 {
        margin-bottom: 1rem;
        color: #2C3E50;
        font-weight: 400;
    }

    .contact-info p {
        margin: 0.5rem 0;
        color: #5D6D7E;
        font-size: 0.9rem;
    }

    .contact-info a {
        color: #3498DB;
        text-decoration: none;
    }

    /* Main content area */
    .main .block-container {
        color: #2C3E50;
        max-width: 1200px;
        padding-top: 2rem;
    }

    /* Lists */
    ul, ol, li {
        color: #5D6D7E !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header mit Sub-Header
    st.markdown(f'<h1 class="main-header">🤖 {PERSON_NAME} - AI Assistant</h1>',
                unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Stellen Sie mir Fragen zu meinem beruflichen Werdegang, Fähigkeiten und Erfahrungen</p>',
        unsafe_allow_html=True)

    # Clean Info Box wie in der ursprünglichen Version
    st.markdown("""
    <div class="info-box">
    <h4>👋 Willkommen liebe Recruiter!</h4>
    <p>Dieser AI-Assistant kann Ihnen bei folgenden Fragen helfen:</p>
    <ul>
    <li>📄 Beruflicher Werdegang und Erfahrungen</li>
    <li>🛠️ Technische Fähigkeiten und Kenntnisse</li>
    <li>🎓 Ausbildung und Zertifikate</li>
    <li>💼 Projektbeispiele und Erfolge</li>
    <li>📧 Kontaktinformationen</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar - vollständig wie in der ursprünglichen Version
    with st.sidebar:
        st.markdown("## ⚙️ Einstellungen")

        # Model selection
        model = st.selectbox(
            "🤖 AI-Modell",
            options=["gpt-4o-mini", "gpt-4o"],
            key="model",
            help="Wählen Sie das AI-Modell für die Antworten"
        )

        # Kontaktinformationen mit Styling
        st.markdown("---")
        st.markdown(f"""
        <div class="contact-info">
        <h4>📞 Direkter Kontakt</h4>
        <p><strong>{PERSON_NAME}</strong></p>
        <p>📧 {PERSON_EMAIL}</p>
        <p>🔗 <a href="{PERSON_LINKEDIN}">LinkedIn Profil</a></p>
        </div>
        """, unsafe_allow_html=True)

        # Admin mode für Document Management
        if st.checkbox("📁 Admin-Modus", help="Nur für Dokumentenverwaltung"):
            st.markdown("---")
            st.markdown("## 📚 Dokumentenverwaltung")
            st.markdown("### ⬆️ Dokument hochladen")

            uploaded_file = st.file_uploader(
                "Datei auswählen",
                type=["pdf", "docx", "html", "txt", "png", "jpg", "jpeg"],
                help="Laden Sie persönliche Dokumente hoch (CV, Zeugnisse, Portfolio, etc.)"
            )

            if uploaded_file and st.button("📤 Hochladen", use_container_width=True):
                with st.spinner("Wird hochgeladen..."):
                    result = upload_document_direct(uploaded_file)
                    if result["status"] == "success":
                        st.success(result["message"])
                        if "note" in result:
                            st.info(result["note"])
                    else:
                        st.error(result["message"])

            # Document List Display (vereinfacht für standalone)
            st.markdown("### 📋 Hochgeladene Dokumente")
            if st.button("🔄 Liste aktualisieren"):
                try:
                    documents = get_all_documents()
                    if documents:
                        st.write(f"📊 {len(documents)} Dokumente gefunden")
                        for doc in documents[-5:]:  # Letzte 5 anzeigen
                            upload_date = doc['upload_timestamp'][:10]
                            st.text(f"• {doc['filename']} ({upload_date})")
                    else:
                        st.info("Keine Dokumente gefunden")
                except Exception as e:
                    st.error(f"Fehler beim Laden der Dokumentliste: {str(e)}")

        # Statistiken
        st.markdown("---")
        st.markdown("## 📊 Statistiken")

        total_messages = len(st.session_state.messages) if "messages" in st.session_state else 0
        st.metric("💬 Nachrichten", total_messages)

        if st.session_state.session_id:
            st.metric("🔗 Session aktiv", "Ja")

        # Reset conversation
        if st.button("🔄 Gespräch zurücksetzen", use_container_width=True):
            st.session_state.messages = [{
                "role": "assistant",
                "content": f"Hallo! Ich kann Ihnen Fragen zu {PERSON_NAME}s beruflichem Werdegang beantworten."
            }]
            st.session_state.session_id = str(uuid.uuid4())
            st.rerun()

    # Quick Questions - vollständig wie in der ursprünglichen Version
    st.markdown("### 💡 Häufige Fragen")
    col1, col2, col3 = st.columns(3)

    # Erste Reihe
    with col1:
        if st.button("🎓 Ausbildung & Qualifikationen", key="btn_education"):
            st.session_state.messages.append({
                "role": "user",
                "content": "Erzählen Sie mir von Ihrer Ausbildung und Ihren Qualifikationen."
            })
            process_quick_question = True

    with col2:
        if st.button("💼 Berufserfahrung", key="btn_experience"):
            st.session_state.messages.append({
                "role": "user",
                "content": "Welche beruflichen Erfahrungen haben Sie gesammelt?"
            })
            process_quick_question = True

    with col3:
        if st.button("🛠️ Technische Skills", key="btn_skills"):
            st.session_state.messages.append({
                "role": "user",
                "content": "Welche technischen Fähigkeiten und Tools beherrschen Sie?"
            })
            process_quick_question = True

    # Zweite Reihe
    col4, col5, col6 = st.columns(3)

    with col4:
        if st.button("🚀 Aktuelle Projekte", key="btn_projects"):
            st.session_state.messages.append({
                "role": "user",
                "content": "An welchen Projekten arbeiten Sie aktuell?"
            })
            process_quick_question = True

    with col5:
        if st.button("📈 Karriereziele", key="btn_goals"):
            st.session_state.messages.append({
                "role": "user",
                "content": "Was sind Ihre beruflichen Ziele?"
            })
            process_quick_question = True

    with col6:
        if st.button("📧 Kontakt", key="btn_contact"):
            st.session_state.messages.append({
                "role": "user",
                "content": "Wie kann ich Sie kontaktieren?"
            })
            process_quick_question = True

    # Process quick question if one was clicked
    if 'process_quick_question' in locals():
        with st.spinner("Antwort wird generiert..."):
            last_question = st.session_state.messages[-1]["content"]
            response = chat_with_assistant(last_question, st.session_state.session_id, model)
            st.session_state.messages.append({
                "role": "assistant",
                "content": response["answer"]
            })
        st.rerun()

    # Chat Interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input
    if prompt := st.chat_input("Ihre Frage..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Antwort wird generiert..."):
                response = chat_with_assistant(prompt, st.session_state.session_id, model)
                st.markdown(response["answer"])
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response["answer"]
                })


if __name__ == "__main__":
    main()