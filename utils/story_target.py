"""
utils/story_target.py
======================
Resolves the effective base URL to explore/execute against for a given story.

Different Jira stories can legitimately target different applications or
environments (e.g. a story testing a third-party integration flow), so a single
globally-configured application_base_url is not always correct. This module lets a
story explicitly say which URL to use, while guarding against blindly following any
URL incidentally mentioned in free text (which would let arbitrary comment content
redirect the automated browser, including to internal/link-local hosts).
"""
from __future__ import annotations

import re
from typing import Optional, Tuple
from urllib.parse import urlparse

# Loopback / private / link-local hosts — never treated as a valid explicit target,
# even when named via an explicit "url = " directive.
_PRIVATE_HOST_RE = re.compile(
    r'^(localhost'
    r'|127\.\d{1,3}\.\d{1,3}\.\d{1,3}'
    r'|10\.\d{1,3}\.\d{1,3}\.\d{1,3}'
    r'|172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}'
    r'|192\.168\.\d{1,3}\.\d{1,3}'
    r'|169\.254\.\d{1,3}\.\d{1,3}'
    r'|0\.0\.0\.0'
    r'|::1)$',
    re.IGNORECASE,
)

# Matches an explicit, unambiguous directive like "url = https://example.com" or
# "URL: https://example.com" — deliberately narrow syntax so it can't be triggered by
# an incidental link mentioned in ordinary prose.
_URL_DIRECTIVE_RE = re.compile(r'\burl\s*[:=]\s*(https?://[^\s,;]+)', re.IGNORECASE)


def resolve_story_base_url(
    story_text: str,
    configured_base_url: Optional[str],
    default_base_url: str,
) -> Tuple[str, bool]:
    """
    Determine which base URL to use for this story's crawl/execution.

    Priority:
      1. An explicit "url = <URL>" directive in the story text — a deliberate
         instruction from whoever authored the story, trusted even when it points
         somewhere other than the configured default. Rejected (falls through) if it
         resolves to a private/loopback/link-local host.
      2. The configured application base URL.
      3. The provided default.

    Returns (base_url, is_explicit_directive). A bare URL mentioned incidentally
    elsewhere in free text (without the "url =" directive syntax) is intentionally
    NOT used to choose the navigation target.
    """
    match = _URL_DIRECTIVE_RE.search(story_text or "")
    if match:
        candidate = match.group(1).rstrip('.,;)')
        parsed = urlparse(candidate)
        host = parsed.netloc.split(":")[0].lower()
        if parsed.scheme in ("http", "https") and host and not _PRIVATE_HOST_RE.match(host):
            # Trust the author's explicit choice of HOST (that's the actual instruction —
            # "test this site"), but not necessarily their choice of plain http: virtually
            # every real production site now requires/redirects to https, so a "url =
            # http://..." directive is almost always a typo rather than a deliberate
            # instruction to exercise the insecure scheme — and following it literally
            # causes real navigation timeouts against hosts that don't answer on http at
            # all. Upgrading the scheme doesn't change which site is targeted.
            scheme = "https" if parsed.scheme == "http" else parsed.scheme
            return f"{scheme}://{parsed.netloc}", True
        # Invalid or disallowed directive — ignore it rather than trust it, fall through.

    if configured_base_url:
        return configured_base_url, False

    return default_base_url, False


# Common UI-flow keywords used to derive a fallback domain/module name directly from
# the story when no crawled route scores a confident match (e.g. a login modal that
# isn't exposed as a top-level nav route to crawl at all).
_FALLBACK_KEYWORD_RE = re.compile(
    r'\b(login|log[\s-]?in|sign[\s-]?in|sign[\s-]?up|signup|register|registration|'
    r'checkout|profile|account|password[\s-]?reset|forgot[\s-]?password|payment|'
    r'search|booking|cart|wishlist|dashboard)\b',
    re.IGNORECASE,
)


# Fixed markers this pipeline's own Jira Update Agent puts on every comment it posts
# (see agents/jira_update_agent.py). Used to keep the tool's own past output out of
# domain/terminology analysis of "what is this story about" — without this, a story that
# has been run through the pipeline before (and had a summary comment posted back to it)
# feeds its own prior mistakes back into itself: e.g. a wrongly-picked "Explore" domain
# shows up in later comments as file names like "TC-EXPLORE-018_trace.zip", which then
# re-matches the literal word "explore" in every subsequent run, entrenching the same
# wrong answer indefinitely.
_PIPELINE_COMMENT_MARKERS = (
    "📊 QA Pipeline Run",
    "Report generated automatically by Antigravity QA Agent Platform",
)


def is_pipeline_generated_comment(body: str) -> bool:
    """True if a Jira comment body was posted by this pipeline itself, not a human."""
    if not body:
        return False
    return any(marker in body for marker in _PIPELINE_COMMENT_MARKERS)


def filter_human_comments(comments: list) -> list:
    """
    Given a list of comment dicts (as in story.json's story.comments, each with a
    "body" key), return only the ones NOT posted by this pipeline's own Jira Update
    Agent — i.e. the comments that reflect genuine human/story context.
    """
    return [c for c in comments if not is_pipeline_generated_comment(c.get("body", ""))]


def derive_fallback_domain(story_text: str, summary_text: str = "") -> str:
    """
    Derive a fallback domain/module keyword directly from the story when route-based
    matching found no confident candidate — instead of defaulting to a fixed,
    application-specific page name.
    """
    match = _FALLBACK_KEYWORD_RE.search(story_text or "") or _FALLBACK_KEYWORD_RE.search(summary_text or "")
    if match:
        return re.sub(r'[\s-]+', '-', match.group(1).strip().lower())

    words = re.findall(r'[a-zA-Z]{4,}', summary_text or "")
    if words:
        return words[0].lower()

    return "home"
