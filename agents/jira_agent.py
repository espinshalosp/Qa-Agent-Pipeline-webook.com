"""
agents/jira_agent.py
====================
Jira Agent — Entry point of the QA pipeline.

Responsibilities
----------------
1. Authenticate with Jira REST API v3 using Basic Auth (email + API token).
2. Execute the configured JQL query with full pagination support.
3. For each matching issue, fetch all required fields:
   - Summary, Description, Issue Type, Priority, Status
   - Assignee, Reporter, Labels, Components
   - Acceptance Criteria (custom field — configurable)
   - Comments (paginated), Attachments, Linked Issues
   - Epic / Parent, Sprint
4. Normalize each raw Jira response into the story.json schema (v1.0.0).
5. Write artifacts/story.json for each issue.
6. Return the list of written story paths to the pipeline manager.

Usage (via pipeline)
--------------------
    from config.settings import get_settings
    from agents.jira_agent import JiraAgent

    agent = JiraAgent(get_settings())
    story_paths = agent.run()

Error Handling
--------------
- 401 Unauthorized → fatal (check email / api_token)
- 403 Forbidden    → fatal (check project permissions)
- 404 Not Found    → fatal (check base_url / issue key)
- 429 Too Many Requests → retry with exponential backoff
- 503 Service Unavailable → retry with exponential backoff
- Network errors   → retry with exponential backoff
"""
from __future__ import annotations

import base64
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from config.settings import Settings
from utils.logger import get_logger
from utils.retry import retry_on_failure
from utils.story_target import is_pipeline_generated_comment

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Fields requested from Jira on every issue fetch
# ---------------------------------------------------------------------------
_ISSUE_FIELDS = [
    "summary",
    "description",
    "issuetype",
    "priority",
    "status",
    "assignee",
    "reporter",
    "labels",
    "components",
    "comment",
    "attachment",
    "issuelinks",
    "created",
    "updated",
    "parent",           # Epic parent (Next-gen projects)
    "customfield_10014",  # Epic Link (classic projects)
    "customfield_10020",  # Sprint
    "customfield_10045",  # Acceptance Criteria (default — configurable)
]


# ===========================================================================
# JiraAgent
# ===========================================================================
class JiraAgent:
    """
    Fetches and normalises Jira issues into story.json artifacts.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client: httpx.Client | None = None

    # -----------------------------------------------------------------------
    # Public entry point
    # -----------------------------------------------------------------------
    def run(self) -> List[Path]:
        """
        Execute the full Jira fetch pipeline.

        Returns
        -------
        list[Path]
            Paths to all written story.json files (one per issue found).
            Empty list if no issues match the JQL query.
        """
        log.info("=" * 60)
        log.info("[JiraAgent] Starting")
        log.info(f"[JiraAgent] Base URL : {self.settings.jira_base_url}")
        log.info(f"[JiraAgent] Email    : {self.settings.jira_email}")
        log.info(f"[JiraAgent] JQL      : {self.settings.jira_jql}")
        log.info("=" * 60)

        written_paths: List[Path] = []

        with self._build_client() as client:
            self._client = client

            # Step 1: Discover all matching issues via JQL
            log.info("[JiraAgent] Running JQL search...")
            issue_stubs = self._discover_issues(self.settings.jira_jql)

            if not issue_stubs:
                log.info("[JiraAgent] No QA stories available.")
                return []

            log.info(f"[JiraAgent] Found {len(issue_stubs)} issue(s) matching JQL.")

            # Step 2: Fetch full details + normalise + write for each issue
            for stub in issue_stubs:
                key = stub["key"]
                log.info(f"[JiraAgent] Processing issue: {key}")
                try:
                    raw = self._fetch_issue(key)
                    story = self._normalize(raw)
                    path = self._write_story(story, key)
                    written_paths.append(path)
                    log.info(f"[JiraAgent] ✓ Written: {path}")
                except httpx.HTTPStatusError as exc:
                    log.error(
                        f"[JiraAgent] HTTP {exc.response.status_code} fetching {key}: {exc}"
                    )
                except Exception as exc:
                    log.error(f"[JiraAgent] Unexpected error fetching {key}: {exc}")

        self._client = None
        log.info(f"[JiraAgent] Complete. {len(written_paths)} story file(s) written.")
        return written_paths

    # -----------------------------------------------------------------------
    # HTTP client setup
    # -----------------------------------------------------------------------
    def _build_client(self) -> httpx.Client:
        """Build an authenticated httpx client."""
        token = base64.b64encode(
            f"{self.settings.jira_email}:{self.settings.jira_api_token}".encode()
        ).decode()

        return httpx.Client(
            base_url=self.settings.jira_base_url,
            headers={
                "Authorization": f"Basic {token}",
                "Accept": "application/json",
                "Content-Type": "application/json",
                "X-Atlassian-Token": "no-check",
            },
            timeout=30.0,
            follow_redirects=True,
        )

    # -----------------------------------------------------------------------
    # JQL Search (paginated)
    # -----------------------------------------------------------------------
    @retry_on_failure(max_retries=3, initial_delay=1.0, backoff_factor=2, retry_codes={429, 503})
    def _search_page(self, jql: str, next_page_token: str | None = None) -> Dict[str, Any]:
        """Fetch a single page of JQL search results (cursor-based pagination)."""
        assert self._client is not None
        params: Dict[str, Any] = {
            "jql": jql,
            "maxResults": self.settings.jira_max_results_per_page,
            "fields": "summary,status,issuetype,priority,key",
        }
        if next_page_token:
            params["nextPageToken"] = next_page_token

        response = self._client.get("/rest/api/3/search/jql", params=params)
        response.raise_for_status()
        return response.json()

    def _discover_issues(self, jql: str) -> List[Dict[str, Any]]:
        """
        Run a JQL query and return all matching issue stubs.
        Uses cursor-based pagination (nextPageToken) from the search/jql endpoint.
        """
        all_issues: List[Dict[str, Any]] = []
        next_page_token: str | None = None
        page_num = 0

        while True:
            page_num += 1
            log.debug(f"[JiraAgent] Fetching search page {page_num}")
            page = self._search_page(jql, next_page_token)

            issues_on_page = page.get("issues", [])
            all_issues.extend(issues_on_page)

            log.debug(
                f"[JiraAgent] Page {page_num}: {len(issues_on_page)} issue(s) "
                f"(total fetched: {len(all_issues)})"
            )

            # Cursor-based pagination: stop when isLast=True or no nextPageToken
            is_last = page.get("isLast", True)
            next_page_token = page.get("nextPageToken")

            if is_last or not next_page_token or not issues_on_page:
                break

        return all_issues

    # -----------------------------------------------------------------------
    # Single Issue Fetch
    # -----------------------------------------------------------------------
    def _build_fields_param(self) -> str:
        """Build the comma-separated fields list, including the configured AC field."""
        fields = list(_ISSUE_FIELDS)
        ac_field = self.settings.jira_acceptance_criteria_field
        if ac_field and ac_field not in fields:
            fields.append(ac_field)
        return ",".join(fields)

    @retry_on_failure(max_retries=3, initial_delay=1.0, backoff_factor=2, retry_codes={429, 503})
    def _fetch_issue(self, key: str) -> Dict[str, Any]:
        """Fetch a single issue with all required fields."""
        assert self._client is not None
        fields = self._build_fields_param()
        response = self._client.get(
            f"/rest/api/3/issue/{key}",
            params={"fields": fields, "expand": "renderedFields,names"},
        )
        response.raise_for_status()
        return response.json()

    # -----------------------------------------------------------------------
    # Normalisation — Jira raw response → story.json schema
    # -----------------------------------------------------------------------
    def _normalize(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map a raw Jira REST API v3 issue response to the story.json schema v1.0.0.
        """
        fields = raw.get("fields", {})
        key = raw.get("key", "")

        # -- Core fields --
        summary = fields.get("summary") or ""
        description = self._extract_adf_text(fields.get("description"))
        issue_type = _safe_name(fields.get("issuetype"))
        priority = _safe_name(fields.get("priority"))
        status = _safe_name(fields.get("status"))
        assignee = _safe_display_name(fields.get("assignee"))
        reporter = _safe_display_name(fields.get("reporter"))

        # -- Labels & components --
        labels: List[str] = fields.get("labels") or []
        components: List[str] = [c.get("name", "") for c in (fields.get("components") or [])]

        # -- Acceptance Criteria --
        acceptance_criteria = self._extract_acceptance_criteria(fields)

        # -- Comments --
        comments = self._extract_comments(fields.get("comment", {}))

        # -- Attachments --
        attachments = self._extract_attachments(fields.get("attachment") or [])

        # -- Linked Issues --
        linked_issues = self._extract_linked_issues(fields.get("issuelinks") or [])

        # -- Epic --
        epic_key, epic_summary = self._extract_epic(fields)

        # -- Sprint --
        sprint = self._extract_sprint(fields.get("customfield_10020"))

        # -- Timestamps --
        created_at = fields.get("created")
        updated_at = fields.get("updated")
        fetched_at = datetime.now(timezone.utc).isoformat()

        return {
            "_schema_version": "1.0.0",
            "_status": "populated",
            "_description": "Normalized Jira story artifact. Populated by the Jira Agent.",
            "story": {
                "id": raw.get("id"),
                "key": key,
                "summary": summary,
                "description": description,
                "story_type": issue_type,
                "status": status,
                "priority": priority,
                "assignee": assignee,
                "reporter": reporter,
                "sprint": sprint,
                "epic_key": epic_key,
                "epic_summary": epic_summary,
                "labels": labels,
                "components": components,
                "acceptance_criteria": acceptance_criteria,
                "linked_issues": linked_issues,
                "attachments": attachments,
                "comments": comments,
                "created_at": created_at,
                "updated_at": updated_at,
                "fetched_at": fetched_at,
            },
            "_metadata": {
                "source_agent": "jira_agent",
                "pipeline_run_id": None,  # Set by pipeline manager
                "jira_base_url": self.settings.jira_base_url,
                "jql_used": self.settings.jira_jql,
            },
        }

    # -----------------------------------------------------------------------
    # Acceptance Criteria extraction
    # -----------------------------------------------------------------------
    def _extract_acceptance_criteria(self, fields: Dict[str, Any]) -> List[str]:
        """
        Try multiple strategies to extract acceptance criteria:
        1. Configured custom field (e.g., customfield_10045)
        2. Any field whose value is ADF content and whose key contains 'acceptance'
        3. Pattern matching in description text ("Acceptance Criteria:" section)
        """
        # Strategy 1: Configured custom field
        ac_field = self.settings.jira_acceptance_criteria_field
        ac_value = fields.get(ac_field)
        if ac_value:
            if isinstance(ac_value, dict):  # ADF format
                text = self._extract_adf_text(ac_value)
                if text:
                    return [line.strip() for line in text.splitlines() if line.strip()]
            elif isinstance(ac_value, str) and ac_value.strip():
                return [line.strip() for line in ac_value.splitlines() if line.strip()]

        # Strategy 2: Scan for any custom field with 'acceptance' in the name
        for field_key, field_val in fields.items():
            if "acceptance" in field_key.lower() and field_val:
                if isinstance(field_val, dict):
                    text = self._extract_adf_text(field_val)
                    if text:
                        return [line.strip() for line in text.splitlines() if line.strip()]

        # Strategy 3: Parse from description
        description_text = self._extract_adf_text(fields.get("description"))
        if description_text:
            return self._parse_ac_from_text(description_text)

        return []

    @staticmethod
    def _parse_ac_from_text(text: str) -> List[str]:
        """Extract acceptance criteria bullet points from free-text description."""
        lines = text.splitlines()
        capturing = False
        criteria: List[str] = []

        for line in lines:
            stripped = line.strip()
            lower = stripped.lower()

            if any(marker in lower for marker in ["acceptance criteria", "acceptance criterion", "ac:"]):
                capturing = True
                continue

            # Stop capturing at the next heading-like line (after we started)
            if capturing and stripped and (
                stripped.startswith("#")
                or (stripped.isupper() and len(stripped) > 3)
                or lower.startswith(("definition of done", "out of scope", "notes", "assumptions"))
            ):
                break

            if capturing and stripped:
                # Strip common list markers
                clean = stripped.lstrip("-•*>1234567890.) ").strip()
                if clean:
                    criteria.append(clean)

        return criteria

    # -----------------------------------------------------------------------
    # Atlassian Document Format (ADF) text extraction
    # -----------------------------------------------------------------------
    @classmethod
    def _extract_adf_text(cls, node: Any) -> str:
        """
        Recursively extract plain text from an Atlassian Document Format (ADF) node.
        Returns an empty string if node is None or not ADF.
        """
        if node is None:
            return ""
        if isinstance(node, str):
            return node

        if not isinstance(node, dict):
            return ""

        node_type = node.get("type", "")
        texts: List[str] = []

        # Text leaf node
        if node_type == "text":
            return node.get("text", "")

        # Hard break → newline
        if node_type == "hardBreak":
            return "\n"

        # Recurse into children
        for child in node.get("content", []):
            child_text = cls._extract_adf_text(child)
            if child_text:
                texts.append(child_text)

        # Decide separator based on the node type
        if node_type in ("doc", "bulletList", "orderedList", "blockquote"):
            return "\n".join(texts)
        elif node_type in ("paragraph", "heading"):
            return "".join(texts)
        elif node_type == "listItem":
            return "\n".join(texts)
        else:
            return "".join(texts)

    # -----------------------------------------------------------------------
    # Comments
    # -----------------------------------------------------------------------
    @classmethod
    def _extract_comments(cls, comment_field: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract all comments from the comment field object."""
        if not comment_field:
            return []

        result: List[Dict[str, str]] = []
        for c in comment_field.get("comments", []):
            author = c.get("author", {})
            result.append({
                "author": author.get("displayName") or author.get("emailAddress") or "Unknown",
                "body": cls._extract_adf_text(c.get("body")),
                "created": c.get("created") or "",
                "updated": c.get("updated") or "",
            })
        return result

    # -----------------------------------------------------------------------
    # Attachments
    # -----------------------------------------------------------------------
    @staticmethod
    def _extract_attachments(attachment_list: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Extract attachment metadata."""
        result: List[Dict[str, str]] = []
        for a in attachment_list:
            result.append({
                "id": a.get("id") or "",
                "filename": a.get("filename") or "",
                "content_type": a.get("mimeType") or "",
                "size_bytes": str(a.get("size") or 0),
                "url": a.get("content") or "",
                "author": _safe_display_name(a.get("author")),
                "created": a.get("created") or "",
            })
        return result

    # -----------------------------------------------------------------------
    # Linked Issues
    # -----------------------------------------------------------------------
    @staticmethod
    def _extract_linked_issues(links: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Extract linked issue metadata."""
        result: List[Dict[str, str]] = []
        for link in links:
            link_type = link.get("type", {}).get("name") or "relates to"
            if "inwardIssue" in link:
                issue = link["inwardIssue"]
                result.append({
                    "key": issue.get("key") or "",
                    "summary": (issue.get("fields") or {}).get("summary") or "",
                    "link_type": link_type,
                    "direction": "inward",
                    "status": _safe_name((issue.get("fields") or {}).get("status")),
                })
            if "outwardIssue" in link:
                issue = link["outwardIssue"]
                result.append({
                    "key": issue.get("key") or "",
                    "summary": (issue.get("fields") or {}).get("summary") or "",
                    "link_type": link_type,
                    "direction": "outward",
                    "status": _safe_name((issue.get("fields") or {}).get("status")),
                })
        return result

    # -----------------------------------------------------------------------
    # Epic
    # -----------------------------------------------------------------------
    @staticmethod
    def _extract_epic(fields: Dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
        """Return (epic_key, epic_summary) using multiple strategies."""
        # Next-gen projects: parent field
        parent = fields.get("parent")
        if parent:
            parent_type = _safe_name((parent.get("fields") or {}).get("issuetype"))
            if parent_type and parent_type.lower() == "epic":
                epic_key = parent.get("key")
                epic_summary = (parent.get("fields") or {}).get("summary")
                return epic_key, epic_summary

        # Classic projects: Epic Link custom field
        epic_link = fields.get("customfield_10014")
        if epic_link:
            return epic_link, None  # Summary would require a second API call

        return None, None

    # -----------------------------------------------------------------------
    # Sprint
    # -----------------------------------------------------------------------
    @staticmethod
    def _extract_sprint(sprint_field: Any) -> Optional[str]:
        """Extract sprint name from the sprint custom field."""
        if not sprint_field:
            return None
        # Sprint field is a list of sprint objects
        if isinstance(sprint_field, list) and sprint_field:
            sprint_obj = sprint_field[-1]  # Most recent sprint
            if isinstance(sprint_obj, dict):
                return sprint_obj.get("name")
            if isinstance(sprint_obj, str):
                # Sometimes returned as a string like "com.atlassian.greenhopper...name=Sprint 1,..."
                for part in sprint_obj.split(","):
                    if "name=" in part:
                        return part.split("name=")[-1].strip().rstrip("]")
        return None

    # -----------------------------------------------------------------------
    # Artifact writing
    # -----------------------------------------------------------------------
    def _extract_normalized_module(self, story_data: Dict[str, Any]) -> str:
        """Extract and normalize the module name from the story data."""
        import re
        from urllib.parse import urlparse, parse_qs
        story = story_data.get("story", {})
        summary = story.get("summary", "")
        description = story.get("description", "")
        # Exclude this pipeline's own past comments — a story that has already been run
        # through the pipeline before shouldn't have its module name influenced by its
        # own prior output (e.g. file names from a previous run).
        comments = [c.get("body", "") for c in story.get("comments", []) if not is_pipeline_generated_comment(c.get("body", ""))]

        all_text = f"{summary} {description} {' '.join(comments)}"
        url_matches = re.findall(r'https?://[a-zA-Z0-9./?=&_-]+', all_text)

        # Only extract a module name from URLs that point at the configured application
        # (not a fixed "webook.com" literal) — works for whatever app_base_url is set for
        # this deployment, instead of a single hardcoded domain.
        configured_host = urlparse(self.settings.application_base_url or "").netloc.lower()

        for url in url_matches:
            parsed_host = urlparse(url).netloc.lower().split(":")[0]
            is_app_url = bool(configured_host) and (
                parsed_host == configured_host.split(":")[0] or parsed_host.endswith("." + configured_host.split(":")[0])
            )
            if is_app_url:
                try:
                    parsed = urlparse(url)
                    qs = parse_qs(parsed.query)
                    if "category" in qs:
                        return qs["category"][0].replace("-", "_").lower()
                    elif "date" in qs:
                        return qs["date"][0].replace("-", "_").lower()
                    elif qs:
                        for k, v in qs.items():
                            if v and v[0]:
                                return v[0].replace("-", "_").lower()
                    last_seg = parsed.path.strip("/").split("/")[-1]
                    if last_seg and last_seg.lower() not in ["en", "explore"]:
                        return last_seg.replace("-", "_").lower()
                except Exception:
                    pass
        
        components = story.get("components", [])
        if components:
            return components[0].replace(" ", "_").lower()
            
        # Fallback to JIRA summary/title (first clean word representing the module)
        if summary:
            first_word = re.findall(r'[a-zA-Z0-9]+', summary)
            if first_word:
                return first_word[0].lower()

        return "unknown"

    def _write_story(self, story: Dict[str, Any], issue_key: str) -> Path:
        """
        Write the normalised story dict to artifacts/<story_key>_<normalized_module>_<YYYYMMDD_HHMMSS>/story.json.
        """
        normalized_module = self._extract_normalized_module(story)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.settings.artifact_dir / f"{issue_key}_{normalized_module}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        output_path = run_dir / "story.json"
        output_path.write_text(
            json.dumps(story, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        return output_path


# ===========================================================================
# Helpers
# ===========================================================================
def _safe_name(obj: Any) -> Optional[str]:
    """Safely extract the 'name' key from a Jira field object."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get("name") or obj.get("value")
    return str(obj)


def _safe_display_name(obj: Any) -> Optional[str]:
    """Safely extract displayName or emailAddress from a Jira user object."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get("displayName") or obj.get("emailAddress")
    return str(obj)
