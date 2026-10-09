"""
utils/attachment_processor.py
=============================
Processor for Jira attachments — downloads, routes, and extracts content.
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import httpx

from config.settings import Settings
from utils.logger import get_logger

log = get_logger(__name__)

class AttachmentProcessor:
    """
    Downloads and extracts semantic content/text from Jira story attachments.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.download_dir = settings.attachment_dir
        self.auth_token = base64.b64encode(
            f"{settings.jira_email}:{settings.jira_api_token}".encode()
        ).decode()

    def process(self, attachments: List[Dict[str, Any]], story_key: str) -> List[Dict[str, Any]]:
        """
        Process a list of attachments: download them, extract content, and return insights.
        """
        insights = []
        if not attachments:
            return insights

        # Ensure directory for this specific story exists
        story_dir = self.download_dir / story_key
        story_dir.mkdir(parents=True, exist_ok=True)

        for attachment in attachments:
            filename = attachment.get("filename") or "unnamed"
            content_type = attachment.get("content_type") or ""
            url = attachment.get("url") or ""
            att_id = attachment.get("id") or "unknown"

            if not url:
                log.warning(f"[AttachmentProcessor] Skipping attachment {filename} due to empty URL.")
                continue

            dest_path = story_dir / f"{att_id}_{filename}"
            log.info(f"[AttachmentProcessor] Processing attachment: {filename} ({content_type})")

            # 1. Download file
            downloaded = self._download_file(url, dest_path)
            if not downloaded:
                log.warning(f"[AttachmentProcessor] Failed to download {filename}. Skipping.")
                continue

            # 2. Extract content based on MIME type
            extracted_text = ""
            source = "none"

            try:
                if content_type == "application/pdf" or filename.lower().endswith(".pdf"):
                    extracted_text = self._extract_pdf(dest_path)
                    source = "pypdf"
                elif content_type in ["text/plain", "text/csv", "text/markdown"] or filename.lower().endswith((".txt", ".csv", ".md")):
                    extracted_text = dest_path.read_text(encoding="utf-8", errors="ignore")
                    source = "text_file"
                elif content_type.startswith("image/") or filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".gif")):
                    extracted_text = self._extract_image_vision(dest_path, content_type)
                    source = "llm_vision" if extracted_text else "none"
                else:
                    # Optional: support for docx if we want to add later
                    log.info(f"[AttachmentProcessor] Skipping text extraction for unsupported type: {content_type}")
            except Exception as e:
                log.error(f"[AttachmentProcessor] Error extracting content from {filename}: {e}", exc_info=True)

            if extracted_text.strip():
                insights.append({
                    "id": att_id,
                    "filename": filename,
                    "content_type": content_type,
                    "extracted_text": extracted_text.strip(),
                    "source": source,
                    "local_path": str(dest_path)
                })
                log.info(f"[AttachmentProcessor] Successfully extracted insight from {filename} via {source}")

        return insights

    def _download_file(self, url: str, dest_path: Path) -> bool:
        """Download file from Jira REST API with authentication."""
        try:
            # We construct a custom headers dict mimicking JiraAgent basic auth
            headers = {
                "Authorization": f"Basic {self.auth_token}",
                "X-Atlassian-Token": "no-check",
            }
            with httpx.Client(follow_redirects=True, timeout=60.0) as client:
                response = client.get(url, headers=headers)
                response.raise_for_status()
                dest_path.write_bytes(response.content)
            return True
        except Exception as e:
            log.error(f"[AttachmentProcessor] Failed to download {url}: {e}")
            return False

    def _extract_pdf(self, file_path: Path) -> str:
        """Extract text from a PDF file using pypdf."""
        try:
            import pypdf
            reader = pypdf.PdfReader(str(file_path))
            text_parts = []
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(f"--- Page {i+1} ---\n{page_text}")
            return "\n".join(text_parts)
        except ImportError:
            log.error("[AttachmentProcessor] 'pypdf' package not installed, skipping PDF extraction.")
            return ""
        except Exception as e:
            log.error(f"[AttachmentProcessor] Failed reading PDF: {e}")
            return ""

    def _extract_image_vision(self, file_path: Path, content_type: str) -> str:
        """Extract layout and UI details from image using configured LLM Vision API."""
        provider = self.settings.llm_provider
        api_key = self.settings.llm_api_key
        model = self.settings.llm_model

        if not provider or not api_key:
            log.warning("[AttachmentProcessor] LLM Provider or API Key not configured. Skipping image vision analysis.")
            return ""

        # Normalize provider and get model
        provider = provider.lower().strip()
        
        try:
            image_bytes = file_path.read_bytes()
            base64_image = base64.b64encode(image_bytes).decode("utf-8")
            prompt = self.settings.llm_vision_prompt

            if "openai" in provider:
                # OpenAI implementation
                from openai import OpenAI
                client = OpenAI(api_key=api_key)
                target_model = model or "gpt-4o"
                
                # Standard OpenAI chat completion with inline image block
                response = client.chat.completions.create(
                    model=target_model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{content_type};base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=self.settings.llm_max_tokens,
                    temperature=self.settings.llm_temperature
                )
                return response.choices[0].message.content or ""

            elif "google" in provider or "gemini" in provider:
                # Google Gemini implementation
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                
                # Default model mapping
                target_model = model or "gemini-1.5-flash"
                model_inst = genai.GenerativeModel(target_model)
                
                # Standard Gemini vision prompt call with parts
                response = model_inst.generate_content([
                    prompt,
                    {
                        "mime_type": content_type,
                        "data": image_bytes
                    }
                ])
                return response.text or ""

            elif "anthropic" in provider or "claude" in provider:
                # Anthropic implementation
                from anthropic import Anthropic
                client = Anthropic(api_key=api_key)
                # claude-3-5-sonnet-20240620 was retired 2025-10-28; claude-opus-4-8 is the
                # current default vision-capable model.
                target_model = model or "claude-opus-4-8"
                
                # Anthropic message with source base64 image media block
                response = client.messages.create(
                    model=target_model,
                    max_tokens=self.settings.llm_max_tokens,
                    temperature=self.settings.llm_temperature,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": content_type,
                                        "data": base64_image,
                                    },
                                },
                                {
                                    "type": "text",
                                    "text": prompt
                                }
                            ],
                        }
                    ]
                )
                return response.content[0].text or ""
            else:
                log.warning(f"[AttachmentProcessor] Unsupported LLM provider for vision analysis: {provider}")
                return ""

        except Exception as e:
            log.error(f"[AttachmentProcessor] Failed image vision analysis: {e}", exc_info=True)
            return ""
