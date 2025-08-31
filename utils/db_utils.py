# db_utils.py - PostgreSQL version for Neon hosting
import psycopg2
import psycopg2.extras
from datetime import datetime, timedelta
from typing import List, Dict, Any
import os
from contextlib import contextmanager
import logging
import streamlit as st

try:
    DATABASE_URL = st.secrets["DATABASE_URL"]
except KeyError:
    raise ValueError("DATABASE_URL not found in Streamlit secrets. Please add it to your secrets configuration.")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@contextmanager
def get_db_connection():
    """Context manager for database connections with automatic cleanup"""
    conn = None
    try:
        conn = psycopg2.connect(
            DATABASE_URL,
            cursor_factory=psycopg2.extras.RealDictCursor  # Returns dict-like rows
        )
        yield conn
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        if conn:
            conn.close()


def create_application_logs():
    """Creates the table for chat logs"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS application_logs
                       (
                           id
                           SERIAL
                           PRIMARY
                           KEY,
                           session_id
                           TEXT,
                           user_query
                           TEXT,
                           gpt_response
                           TEXT,
                           model
                           TEXT,
                           response_time
                           REAL
                           DEFAULT
                           0,
                           created_at
                           TIMESTAMP
                           DEFAULT
                           CURRENT_TIMESTAMP
                       )
                       ''')

        # Create indexes for better performance
        cursor.execute('''
                       CREATE INDEX IF NOT EXISTS idx_application_logs_session_id
                           ON application_logs(session_id)
                       ''')
        cursor.execute('''
                       CREATE INDEX IF NOT EXISTS idx_application_logs_created_at
                           ON application_logs(created_at)
                       ''')

        conn.commit()


def create_document_store():
    """Creates the table for document metadata"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS document_store
                       (
                           id
                           SERIAL
                           PRIMARY
                           KEY,
                           filename
                           TEXT,
                           file_size
                           INTEGER
                           DEFAULT
                           0,
                           file_type
                           TEXT,
                           document_category
                           TEXT
                           DEFAULT
                           'other',
                           upload_timestamp
                           TIMESTAMP
                           DEFAULT
                           CURRENT_TIMESTAMP
                       )
                       ''')

        # Create index
        cursor.execute('''
                       CREATE INDEX IF NOT EXISTS idx_document_store_category
                           ON document_store(document_category)
                       ''')

        conn.commit()


def create_feedback_table():
    """Creates the table for recruiter feedback"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS recruiter_feedback
                       (
                           id
                           SERIAL
                           PRIMARY
                           KEY,
                           session_id
                           TEXT,
                           rating
                           INTEGER,
                           comment
                           TEXT,
                           feedback_timestamp
                           TIMESTAMP
                           DEFAULT
                           CURRENT_TIMESTAMP
                       )
                       ''')

        # Create index
        cursor.execute('''
                       CREATE INDEX IF NOT EXISTS idx_recruiter_feedback_session_id
                           ON recruiter_feedback(session_id)
                       ''')

        conn.commit()


def create_session_analytics():
    """Creates the table for session analytics"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS session_analytics
                       (
                           id
                           SERIAL
                           PRIMARY
                           KEY,
                           session_id
                           TEXT
                           UNIQUE,
                           first_interaction
                           TIMESTAMP
                           DEFAULT
                           CURRENT_TIMESTAMP,
                           last_interaction
                           TIMESTAMP
                           DEFAULT
                           CURRENT_TIMESTAMP,
                           total_queries
                           INTEGER
                           DEFAULT
                           0,
                           user_agent
                           TEXT,
                           ip_address
                           TEXT
                       )
                       ''')

        # Create unique index on session_id
        cursor.execute('''
                       CREATE UNIQUE INDEX IF NOT EXISTS idx_session_analytics_session_id
                           ON session_analytics(session_id)
                       ''')

        conn.commit()


def insert_application_logs(session_id: str, user_query: str, gpt_response: str, model: str, response_time: float = 0):
    """Adds a chat log entry"""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        try:
            # Add chat log
            cursor.execute('''
                           INSERT INTO application_logs (session_id, user_query, gpt_response, model, response_time)
                           VALUES (%s, %s, %s, %s, %s)
                           ''', (session_id, user_query, gpt_response, model, response_time))

            # Update Session Analytics
            cursor.execute('''
                           SELECT total_queries
                           FROM session_analytics
                           WHERE session_id = %s
                           ''', (session_id,))
            result = cursor.fetchone()

            if result:
                # Update existing session
                cursor.execute('''
                               UPDATE session_analytics
                               SET last_interaction = CURRENT_TIMESTAMP,
                                   total_queries    = total_queries + 1
                               WHERE session_id = %s
                               ''', (session_id,))
            else:
                # Create new session record using INSERT ... ON CONFLICT
                cursor.execute('''
                               INSERT INTO session_analytics (session_id, total_queries)
                               VALUES (%s, 1) ON CONFLICT (session_id) 
                    DO
                               UPDATE SET
                                   last_interaction = CURRENT_TIMESTAMP,
                                   total_queries = session_analytics.total_queries + 1
                               ''', (session_id,))

            conn.commit()
            logger.info(f"Successfully logged interaction for session {session_id}")

        except Exception as e:
            conn.rollback()
            logger.error(f"Error inserting application log: {e}")
            raise


def get_chat_history(session_id: str, limit: int = 10) -> List[Dict[str, str]]:
    """Gets chat history for a session"""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        cursor.execute('''
                       SELECT user_query, gpt_response
                       FROM application_logs
                       WHERE session_id = %s
                       ORDER BY created_at DESC
                           LIMIT %s
                       ''', (session_id, limit * 2))

        messages = []
        rows = cursor.fetchall()

        # Reverse to get chronological order
        for row in reversed(rows):
            messages.extend([
                {"role": "human", "content": row['user_query']},
                {"role": "ai", "content": row['gpt_response']}
            ])

        return messages


def insert_document_record(filename: str, file_size: int = 0, file_type: str = None) -> int:
    """Adds a document entry and returns the ID"""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Determine document category based on filename
        filename_lower = filename.lower()
        if any(word in filename_lower for word in ['cv', 'lebenslauf', 'resume']):
            category = 'cv'
        elif any(word in filename_lower for word in ['zertifikat', 'certificate', 'zeugnis']):
            category = 'certificate'
        elif any(word in filename_lower for word in ['portfolio', 'projekt']):
            category = 'portfolio'
        elif any(word in filename_lower for word in ['referenz', 'reference', 'empfehlung']):
            category = 'reference'
        else:
            category = 'other'

        cursor.execute('''
                       INSERT INTO document_store (filename, file_size, file_type, document_category)
                       VALUES (%s, %s, %s, %s) RETURNING id
                       ''', (filename, file_size, file_type, category))

        file_id = cursor.fetchone()['id']
        conn.commit()
        logger.info(f"Document record created with ID {file_id}: {filename}")
        return file_id


def delete_document_record(file_id: int) -> bool:
    """Deletes a document entry"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute('DELETE FROM document_store WHERE id = %s', (file_id,))
            deleted_rows = cursor.rowcount
            conn.commit()

            if deleted_rows > 0:
                logger.info(f"Document record deleted: ID {file_id}")
                return True
            else:
                logger.warning(f"No document found with ID {file_id}")
                return False

        except Exception as e:
            conn.rollback()
            logger.error(f"Error deleting document {file_id}: {e}")
            return False


def get_all_documents() -> List[Dict[str, Any]]:
    """Gets all documents with extended information"""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        cursor.execute('''
                       SELECT id, filename, file_size, file_type, document_category, upload_timestamp
                       FROM document_store
                       ORDER BY upload_timestamp DESC
                       ''')

        documents = cursor.fetchall()
        return [dict(doc) for doc in documents]


def insert_feedback(session_id: str, rating: int, comment: str = ""):
    """Adds recruiter feedback"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute('''
                           INSERT INTO recruiter_feedback (session_id, rating, comment)
                           VALUES (%s, %s, %s)
                           ''', (session_id, rating, comment))
            conn.commit()
            logger.info(f"Feedback recorded for session {session_id}: rating {rating}")
        except Exception as e:
            conn.rollback()
            logger.error(f"Error inserting feedback: {e}")
            raise


def get_interaction_stats() -> Dict[str, Any]:
    """Gets interaction statistics for dashboard"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        stats = {}

        try:
            # Total statistics
            cursor.execute('SELECT COUNT(*) as total_queries FROM application_logs')
            stats['total_queries'] = cursor.fetchone()['total_queries']

            cursor.execute('SELECT COUNT(DISTINCT session_id) as unique_sessions FROM application_logs')
            stats['unique_sessions'] = cursor.fetchone()['unique_sessions']

            cursor.execute('SELECT COUNT(*) as total_documents FROM document_store')
            stats['total_documents'] = cursor.fetchone()['total_documents']

            # Today's statistics
            cursor.execute('''
                           SELECT COUNT(*) as today_queries
                           FROM application_logs
                           WHERE DATE (created_at) = CURRENT_DATE
                           ''')
            stats['today_queries'] = cursor.fetchone()['today_queries']

            cursor.execute('''
                           SELECT COUNT(DISTINCT session_id) as today_sessions
                           FROM application_logs
                           WHERE DATE (created_at) = CURRENT_DATE
                           ''')
            stats['today_sessions'] = cursor.fetchone()['today_sessions']

            # Last 7 days
            cursor.execute('''
                           SELECT COUNT(*) as week_queries
                           FROM application_logs
                           WHERE created_at >= CURRENT_DATE - INTERVAL '7 days'
                           ''')
            stats['week_queries'] = cursor.fetchone()['week_queries']

            # Top used models
            cursor.execute('''
                           SELECT model, COUNT(*) as usage_count
                           FROM application_logs
                           GROUP BY model
                           ORDER BY usage_count DESC LIMIT 5
                           ''')
            stats['top_models'] = [dict(row) for row in cursor.fetchall()]

            # Average queries per session
            cursor.execute('''
                           SELECT AVG(total_queries) as avg_queries_per_session
                           FROM session_analytics
                           ''')
            result = cursor.fetchone()
            stats['avg_queries_per_session'] = round(float(result['avg_queries_per_session'] or 0), 2)

            # Feedback statistics
            cursor.execute('''
                           SELECT AVG(rating) as avg_rating, COUNT(*) as total_feedback
                           FROM recruiter_feedback
                           ''')
            feedback_stats = cursor.fetchone()
            stats['avg_rating'] = round(float(feedback_stats['avg_rating'] or 0), 2)
            stats['total_feedback'] = feedback_stats['total_feedback']

            # Document categories
            cursor.execute('''
                           SELECT document_category, COUNT(*) as count
                           FROM document_store
                           GROUP BY document_category
                           ORDER BY count DESC
                           ''')
            stats['document_categories'] = [dict(row) for row in cursor.fetchall()]

            # Recent activities
            cursor.execute('''
                           SELECT session_id, user_query, created_at
                           FROM application_logs
                           ORDER BY created_at DESC LIMIT 5
                           ''')
            stats['recent_queries'] = [dict(row) for row in cursor.fetchall()]

        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            stats = {"error": "Failed to retrieve statistics"}

        return stats


def get_session_details(session_id: str) -> Dict[str, Any]:
    """Gets detailed information about a session"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        details = {}

        try:
            # Session info
            cursor.execute('''
                           SELECT first_interaction, last_interaction, total_queries
                           FROM session_analytics
                           WHERE session_id = %s
                           ''', (session_id,))

            session_info = cursor.fetchone()
            if session_info:
                details['session_info'] = dict(session_info)

            # All queries for this session
            cursor.execute('''
                           SELECT user_query, gpt_response, model, created_at, response_time
                           FROM application_logs
                           WHERE session_id = %s
                           ORDER BY created_at
                           ''', (session_id,))

            details['queries'] = [dict(row) for row in cursor.fetchall()]

            # Feedback for this session
            cursor.execute('''
                           SELECT rating, comment, feedback_timestamp
                           FROM recruiter_feedback
                           WHERE session_id = %s
                           ''', (session_id,))

            details['feedback'] = [dict(row) for row in cursor.fetchall()]

        except Exception as e:
            logger.error(f"Error retrieving session details: {e}")
            details = {"error": f"Failed to retrieve session details: {e}"}

        return details


def cleanup_old_data(days_to_keep: int = 90):
    """Cleans up old data (optional for maintenance)"""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)

            # Delete old logs
            cursor.execute('''
                           DELETE
                           FROM application_logs
                           WHERE created_at < %s
                           ''', (cutoff_date,))
            logs_deleted = cursor.rowcount

            # Delete old session analytics
            cursor.execute('''
                           DELETE
                           FROM session_analytics
                           WHERE first_interaction < %s
                           ''', (cutoff_date,))
            sessions_deleted = cursor.rowcount

            # Delete old feedback
            cursor.execute('''
                           DELETE
                           FROM recruiter_feedback
                           WHERE feedback_timestamp < %s
                           ''', (cutoff_date,))
            feedback_deleted = cursor.rowcount

            conn.commit()

            logger.info(
                f"Cleanup completed: {logs_deleted} logs, {sessions_deleted} sessions, {feedback_deleted} feedback entries deleted")

            return {
                "logs_deleted": logs_deleted,
                "sessions_deleted": sessions_deleted,
                "feedback_deleted": feedback_deleted
            }

        except Exception as e:
            conn.rollback()
            logger.error(f"Error during cleanup: {e}")
            raise


def test_connection():
    """Test database connection"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT version()')
            version = cursor.fetchone()
            logger.info(f"PostgreSQL connection successful: {version['version']}")
            return True
    except Exception as e:
        logger.error(f"PostgreSQL connection failed: {e}")
        return False


def initialize_database():
    """Initialize all database tables"""
    try:
        logger.info("Initializing PostgreSQL database...")
        create_application_logs()
        create_document_store()
        create_feedback_table()
        create_session_analytics()
        logger.info("Database initialization completed successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise


# Initialize database on import
if __name__ == "__main__":
    # Test connection first
    if test_connection():
        initialize_database()
    else:
        logger.error("Database initialization skipped due to connection failure")
else:
    # Auto-initialize when imported
    initialize_database()