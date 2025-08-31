# ocr_utils.py - OCR functionality using Mistral AI
import os
import base64
import logging
from typing import Optional, Tuple, Dict, Any
from mistralai import Mistral
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

logger = logging.getLogger(__name__)


class MistralOCRProcessor:
    """OCR processor using Mistral AI's OCR service"""

    def __init__(self):
        self.api_key = st.secrets["MISTRAL_API_KEY"]
        if not self.api_key:
            logger.warning("MISTRAL_API_KEY not found. OCR functionality will be disabled.")
            self.client = None
        else:
            self.client = Mistral(api_key=self.api_key)
            logger.info("Mistral OCR client initialized successfully")

    def is_available(self) -> bool:
        """Check if OCR service is available"""
        return self.client is not None

    def encode_file_to_base64(self, file_path: str) -> Optional[str]:
        """Encode file to base64 string"""
        try:
            with open(file_path, "rb") as file:
                return base64.b64encode(file.read()).decode('utf-8')
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            return None
        except Exception as e:
            logger.error(f"Error encoding file {file_path}: {str(e)}")
            return None

    def determine_mime_type(self, file_path: str) -> str:
        """Determine MIME type based on file extension"""
        suffix = Path(file_path).suffix.lower()
        mime_types = {
            '.pdf': 'application/pdf',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp',
            '.tiff': 'image/tiff',
            '.webp': 'image/webp',
            '.avif': 'image/avif',
            '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        }
        return mime_types.get(suffix, 'application/octet-stream')

    def extract_text_from_file(self, file_path: str, include_images: bool = False) -> Tuple[
        Optional[str], Optional[Dict[str, Any]]]:
        """
        Extract text from file using Mistral OCR

        Args:
            file_path: Path to the file to process
            include_images: Whether to include image data in response

        Returns:
            Tuple of (extracted_text, metadata)
        """
        if not self.is_available():
            logger.error("OCR service not available - MISTRAL_API_KEY missing")
            return None, None

        try:
            # Encode file to base64
            base64_content = self.encode_file_to_base64(file_path)
            if not base64_content:
                return None, None

            # Determine file type and MIME type
            mime_type = self.determine_mime_type(file_path)
            file_suffix = Path(file_path).suffix.lower()

            # Determine document type for Mistral API
            if file_suffix in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp', '.avif']:
                document_type = "image_url"
                document_url = f"data:{mime_type};base64,{base64_content}"
                document_config = {
                    "type": document_type,
                    "image_url": document_url
                }
            else:  # PDF, DOCX, PPTX, etc.
                document_type = "document_url"
                document_url = f"data:{mime_type};base64,{base64_content}"
                document_config = {
                    "type": document_type,
                    "document_url": document_url
                }

            logger.info(f"Starting OCR processing for {file_path} (type: {document_type})")

            # Call Mistral OCR API
            ocr_response = self.client.ocr.process(
                model="mistral-ocr-latest",
                document=document_config,
                include_image_base64=include_images
            )

            print(ocr_response)

            # Extract text content
            extracted_text = ""
            metadata = {
                "file_path": file_path,
                "document_type": document_type,
                "mime_type": mime_type,
                "ocr_model": "mistral-ocr-latest",
                "page_count": 0,
                "images_found": 0
            }

            # Process OCR response
            if hasattr(ocr_response, 'pages') and ocr_response.pages:
                metadata["page_count"] = len(ocr_response.pages)
                for page in ocr_response.pages:
                    if hasattr(page, 'markdown') and page.markdown:
                        extracted_text += page.markdown + "\n\n"

                    # Count images if present
                    if hasattr(page, 'images') and page.images:
                        metadata["images_found"] += len(page.images)

            # Clean up extracted text
            extracted_text = extracted_text.strip()

            if extracted_text:
                logger.info(
                    f"OCR successful: extracted {len(extracted_text)} characters from {metadata['page_count']} pages")
                return extracted_text, metadata
            else:
                logger.warning(f"OCR completed but no text found in {file_path}")
                return None, metadata

        except Exception as e:
            logger.error(f"OCR processing failed for {file_path}: {str(e)}")
            return None, {"error": str(e), "file_path": file_path}

    def process_image_url(self, image_url: str, include_images: bool = False) -> Tuple[
        Optional[str], Optional[Dict[str, Any]]]:
        """
        Process image from URL using Mistral OCR

        Args:
            image_url: URL of the image to process
            include_images: Whether to include image data in response

        Returns:
            Tuple of (extracted_text, metadata)
        """
        if not self.is_available():
            logger.error("OCR service not available - MISTRAL_API_KEY missing")
            return None, None

        try:
            logger.info(f"Starting OCR processing for image URL: {image_url}")

            # Call Mistral OCR API
            ocr_response = self.client.ocr.process(
                model="mistral-ocr-latest",
                document={
                    "type": "image_url",
                    "image_url": image_url
                },
                include_image_base64=include_images
            )

            # Extract text content
            extracted_text = ""
            metadata = {
                "image_url": image_url,
                "document_type": "image_url",
                "ocr_model": "mistral-ocr-latest",
                "page_count": 0
            }

            # Process OCR response
            if hasattr(ocr_response, 'content') and ocr_response.content:
                for page in ocr_response.content:
                    if hasattr(page, 'text') and page.text:
                        extracted_text += page.text + "\n\n"
                        metadata["page_count"] += 1

            # Clean up extracted text
            extracted_text = extracted_text.strip()

            if extracted_text:
                logger.info(f"OCR successful: extracted {len(extracted_text)} characters")
                return extracted_text, metadata
            else:
                logger.warning(f"OCR completed but no text found in image: {image_url}")
                return None, metadata

        except Exception as e:
            logger.error(f"OCR processing failed for image URL {image_url}: {str(e)}")
            return None, {"error": str(e), "image_url": image_url}


# Convenience functions
def extract_text_with_ocr(file_path: str, include_images: bool = False) -> Tuple[
    Optional[str], Optional[Dict[str, Any]]]:
    """
    Convenience function to extract text from a file using OCR

    Args:
        file_path: Path to the file to process
        include_images: Whether to include image data in response

    Returns:
        Tuple of (extracted_text, metadata)
    """
    processor = MistralOCRProcessor()
    return processor.extract_text_from_file(file_path, include_images)


def is_ocr_available() -> bool:
    """Check if OCR service is available"""
    processor = MistralOCRProcessor()
    return processor.is_available()


def should_use_ocr(file_path: str) -> bool:
    """
    Determine if OCR should be used for a file based on its extension

    Args:
        file_path: Path to the file

    Returns:
        True if OCR should be used, False otherwise
    """
    ocr_extensions = {'.pdf', '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp', '.avif', '.pptx', '.docx'}
    suffix = Path(file_path).suffix.lower()
    return suffix in ocr_extensions


# Example usage and testing
if __name__ == "__main__":
    # Setup logging for testing
    logging.basicConfig(level=logging.INFO)

    # Test OCR functionality
    processor = MistralOCRProcessor()

    if processor.is_available():
        print("OCR service is available")

        # Test with a sample file (replace with actual file path for testing)
        test_file = "sample_document.pdf"
        if os.path.exists(test_file):
            text, metadata = processor.extract_text_from_file(test_file)
            if text:
                print(f"Extracted text ({len(text)} chars):")
                print(text[:500] + "..." if len(text) > 500 else text)
                print(f"Metadata: {metadata}")
            else:
                print("No text extracted")
        else:
            print(f"Test file {test_file} not found")
    else:
        print("OCR service not available - check MISTRAL_API_KEY")