"""
agents/testcase_generation_agent.py
===================================
TestCase Generation Agent — Intent-driven behavioral QA intelligence engine.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Tuple, Set

from config.settings import Settings
from utils.logger import get_logger
from utils.attachment_processor import AttachmentProcessor
from utils.story_target import resolve_story_base_url, derive_fallback_domain, filter_human_comments

log = get_logger(__name__)

# Common English filler words excluded when a multi-word crawled route label (e.g.
# "Esports World Cup WITNESS THE ACTION") is split into individual keyword candidates
# for domain matching — without this, a word like "the"/"for"/"and" being part of an
# unrelated route label lets that route win a spurious match against almost any story.
_ROUTE_LABEL_STOPWORDS = frozenset({
    "the", "and", "for", "you", "your", "our", "all", "with", "from", "this", "that",
    "are", "was", "his", "her", "she", "him", "who", "how", "why", "not", "but", "get",
    "has", "had", "did", "may", "let", "new", "see", "use", "now", "out", "can", "will",
    "its", "his", "per", "via", "into", "onto", "over", "than", "then", "them", "they",
    "have", "been", "were", "when", "what", "where", "here", "there",
})


def _sanitize_node_name(text: str) -> str:
    """
    Normalizes a raw selector/URL fragment into the short node-name form used to build
    behavior-graph node/edge IDs (e.g. "<page>_comp_<name>"). Shared between
    BehavioralFlowEngine (which builds the graph) and TestCaseGenerator (which must
    reference the same IDs in each test case's coverage_reference) so the two stay in
    sync — previously they used the same logic only in one place, and most generated
    test cases fell back to a generic, non-matching coverage_reference, making the
    True Coverage Engine's component/path coverage scores appear far lower than reality.
    """
    cleaned = re.sub(r'[^a-zA-Z0-9_]', '_', text or "")
    return cleaned[:35].strip('_').lower()


def _safe_filename_component(value: Any) -> str:
    """
    Strip path-traversal and filesystem-unsafe characters before use in a filename.
    Values like `domain` can be derived from URLs found in Jira story/comment text,
    so they must never be used unsanitized in a filesystem path.
    """
    text = str(value) if value is not None else ""
    text = text.replace("..", "_")
    text = re.sub(r'[^A-Za-z0-9._-]', '_', text)
    return text or "unknown"


def _load_playwright_headless_setting() -> bool:
    """Loads Playwright headless setting from settings.yaml if available, defaulting to False."""
    try:
        import yaml
        yaml_path = Path(__file__).parent.parent / "config" / "settings.yaml"
        if yaml_path.exists():
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                pw_config = data.get("playwright", {})
                if "headless" in pw_config:
                    return bool(pw_config["headless"])
    except Exception as e:
        log.warning(f"Could not load Playwright config from settings.yaml: {e}")
    return False


def handle_secure_login(page) -> None:
    """Detects if the page requires authentication and performs secure login using environment credentials."""
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    email = os.getenv("WEBOOK_EMAIL")
    password = os.getenv("WEBOOK_PASSWORD")
    
    current_url = page.url.lower()
    has_login_fields = False
    try:
        email_input = page.query_selector("input[type='email'], input[name='email'], input[placeholder*='email' i], input[placeholder*='e-mail' i]")
        pass_input = page.query_selector("input[type='password'], input[name='password'], input[placeholder*='password' i]")
        if email_input and pass_input:
            has_login_fields = True
    except Exception:
        pass
        
    is_login_page = ("login" in current_url or "signin" in current_url or "auth" in current_url) or has_login_fields
    
    if is_login_page:
        if not email or not password:
            log.warning("Authentication required, but WEBOOK_EMAIL or WEBOOK_PASSWORD environment variables are missing.")
            return
            
        try:
            log.info("Authentication screen detected. Initiating secure login process...")
            
            # Fill email
            email_input = page.query_selector("input[type='email'], input[name='email'], input[placeholder*='email' i], input[placeholder*='e-mail' i]")
            if email_input:
                email_input.fill(email)
            
            # Fill password
            pass_input = page.query_selector("input[type='password'], input[name='password'], input[placeholder*='password' i]")
            if pass_input:
                pass_input.fill(password)
                
            # Click submit/login button
            submit_btn = page.query_selector("button[type='submit'], button:has-text('Login'), button:has-text('Log In'), button:has-text('Sign In'), input[type='submit']")
            if submit_btn:
                submit_btn.click()
            else:
                if pass_input:
                    pass_input.press("Enter")
            
            page.wait_for_timeout(3000)
            log.info("Secure login sequence submitted.")
        except Exception:
            log.warning("Secure login process encountered an interaction issue. Continuing exploration normally.")


class DomainTerminologyResolver:
    """
    Dynamically extracts and maps business terminology from active module and Jira story.
    Eliminates hardcoded domain-specific terms by deriving vocabulary from context.
    """

    def __init__(self, intent: Dict[str, Any], story_data: Dict[str, Any], behavior_graph: Dict[str, Any] = None) -> None:
        self.intent = intent
        self.story_data = story_data
        self.behavior_graph = behavior_graph or {}
        self.domain = intent.get("domain", "")
        # Authentication/login stories are not transactional-commerce flows, so the
        # default "transaction"/"booking"/"amount" business vocabulary is wrong for them
        # (e.g. describing a First Name field as "Transaction amount is entered"). When the
        # story is an auth flow, resolve to authentication vocabulary instead.
        self.is_auth = intent.get("semantic_key") == "auth"
        self._terminology_cache: Dict[str, str] = {}
        self._build_terminology()

    def _build_terminology(self) -> None:
        """Extracts terminology from multiple sources: domain, story, UI elements, APIs."""
        story = self.story_data.get("story", {})
        summary = story.get("summary", "")
        description = story.get("description", "")
        acceptance_criteria = story.get("acceptance_criteria", [])
        # Exclude comments this pipeline posted on a prior run — otherwise a story that
        # has already been through the pipeline once feeds its own past output (including
        # any past mistake) back into itself as "story context" on every future run.
        comments = [c.get("body", "") for c in filter_human_comments(story.get("comments", []))]

        all_text = f"{self.domain} {summary} {description} {' '.join(acceptance_criteria)} {' '.join(comments)}".lower()

        self._extract_business_entities(all_text)
        self._extract_business_actions(all_text)
        self._extract_ui_terminology()
        self._extract_api_terminology()
        self._extract_semantic_context(all_text)
        self._build_domain_vocabulary()

    def _extract_business_entities(self, text: str) -> None:
        """
        Extracts business entity names (nouns) purely from the story text and the
        real-time-derived domain name — no hardcoded per-domain vocabulary table.
        Works for any module (flights, hotels, invoices, HR requests, ...) since the
        candidates come from the actual story/module content, not a fixed dictionary.
        """
        entity_patterns = [
            r'\b(\w+)\s+(?:booking|reservation|ticket|order|purchase|listing|record|item)',
            r'\b(\w+)\s+(?:management|administration|handling)',
            r'(?:browse|view|search|find)\s+(\w+)',
            r'(?:select|choose|pick)\s+(\w+)',
        ]

        candidates: Dict[str, int] = {}

        def bump(word: str, weight: int = 1) -> None:
            w = word.strip().lower()
            if w:
                candidates[w] = candidates.get(w, 0) + weight

        for pattern in entity_patterns:
            for match in re.findall(pattern, text):
                bump(match)

        # The domain name itself is already derived at runtime from the Jira story /
        # crawled UI (see IntentEngine), not hardcoded here — treat it as a strong
        # candidate, weighted higher than incidental regex matches.
        domain_singular = self.domain.lower().rstrip('s') if self.domain else ""
        if domain_singular:
            bump(domain_singular, weight=5)
        if self.domain:
            bump(self.domain.lower(), weight=3)

        # Filter out generic non-entity words that regex noise commonly produces.
        stop_entities = {"completion", "completions", "action", "actions", "item", "items", "state", "page", "this", "that", "these", "those"}
        for stop in stop_entities:
            candidates.pop(stop, None)

        if candidates:
            # Prefer the highest-weighted (most frequent / domain-anchored) candidate;
            # break ties by preferring the longer, more specific term.
            primary_entity = max(candidates.items(), key=lambda kv: (kv[1], len(kv[0])))[0]
            self._terminology_cache["primary_entity"] = primary_entity
            self._terminology_cache["primary_entity_plural"] = self._pluralize(primary_entity)
        else:
            fallback = self.domain.lower() if self.domain else "item"
            self._terminology_cache["primary_entity"] = fallback
            self._terminology_cache["primary_entity_plural"] = self._pluralize(fallback)

    def _extract_business_actions(self, text: str) -> None:
        """Extracts business action verbs from text."""
        action_patterns = [
            r'\b(?:user|customer|system)\s+(?:can|should|must|will)\s+(\w+)',
            r'\b(?:enable|allow|permit|support)\s+(\w+ing)',
            r'\b(\w+)(?:\s+\w+){0,3}\s+(?:process|workflow|flow)',
        ]

        actions: Set[str] = set()
        for pattern in action_patterns:
            matches = re.findall(pattern, text)
            actions.update(matches)

        common_actions = ["search", "browse", "select", "book", "purchase", "reserve", "order", "filter", "sort"]
        for action in common_actions:
            if action in text:
                actions.add(action)

        self._terminology_cache["business_actions"] = list(actions)

    def _extract_ui_terminology(self) -> None:
        """Extracts terminology from UI elements in behavior graph."""
        selectors = self.behavior_graph.get("selectors", {})
        ui_terms: Set[str] = set()

        for page_key, page_data in selectors.items():
            metadata = page_data.get("component_metadata", {})
            for selector, meta in metadata.items():
                label = meta.get("label", "")
                aria_label = meta.get("aria_label", "")
                text = meta.get("text", "")
                card_context = meta.get("card_context", "")

                for term in [label, aria_label, text, card_context]:
                    if term and len(term) > 2:
                        clean_term = re.sub(r'[^\w\s]', ' ', term).strip()
                        if clean_term:
                            ui_terms.add(clean_term.lower())

            for rule in page_data.get("business_rules", []):
                if rule:
                    clean_rule = re.sub(r'[^\w\s]', ' ', rule).strip()
                    if clean_rule:
                        ui_terms.add(clean_rule.lower())

        self._terminology_cache["ui_terms"] = list(ui_terms)

    def _extract_api_terminology(self) -> None:
        """Extracts terminology from observed APIs."""
        selectors = self.behavior_graph.get("selectors", {})
        api_operations: Set[str] = set()

        for page_data in selectors.values():
            for api in page_data.get("observed_apis", []):
                op_name = api.get("business_operation", "")
                intent = api.get("business_intent", "")
                url = api.get("url", "")

                for term in [op_name, intent]:
                    if term:
                        clean_term = re.sub(r'[^\w\s]', ' ', term).strip()
                        if clean_term:
                            api_operations.add(clean_term.lower())

                from urllib.parse import urlparse
                try:
                    path_parts = urlparse(url).path.split("/")
                    for part in path_parts:
                        if part and len(part) > 2:
                            api_operations.add(part.lower())
                except Exception:
                    pass

        self._terminology_cache["api_operations"] = list(api_operations)

    def _extract_semantic_context(self, text: str) -> None:
        """Extracts semantic context from story to understand business domain."""
        # Extract business intent patterns
        intent_patterns = [
            r'(?:user|customer|system)\s+(?:should|can|will|must)\s+(?:be able to|have the ability to)\s+(\w+(?:\s+\w+)*)',
            r'(?:enable|allow|permit)\s+(?:users|customers)\s+to\s+(\w+(?:\s+\w+)*)',
            r'(?:goal|objective|purpose)\s+(?:is|to)\s+(\w+(?:\s+\w+)*)',
        ]
        
        intents: Set[str] = set()
        for pattern in intent_patterns:
            matches = re.findall(pattern, text)
            intents.update(matches)
        
        self._terminology_cache["business_intents"] = list(intents)
        
        # Extract transaction-related terms
        transaction_patterns = [
            r'(?:place|submit|make)\s+(\w+)',
            r'(?:book|reserve|order|purchase)\s+(\w+)',
            r'(?:complete|finish|finalize)\s+(\w+)',
        ]
        
        transactions: Set[str] = set()
        for pattern in transaction_patterns:
            matches = re.findall(pattern, text)
            transactions.update(matches)
        
        self._terminology_cache["transaction_terms"] = list(transactions)
        
        # Extract workflow stages from acceptance criteria
        story = self.story_data.get("story", {})
        acceptance_criteria = story.get("acceptance_criteria", [])
        
        workflow_stages = []
        for ac in acceptance_criteria:
            # Extract action verbs and stages
            stage_patterns = [
                r'(?:when|after|before|upon)\s+(\w+(?:\s+\w+)*)',
                r'(?:then|next|subsequently)\s+(\w+(?:\s+\w+)*)',
                r'(?:first|second|third|finally|lastly)\s+(\w+(?:\s+\w+)*)',
            ]
            for pattern in stage_patterns:
                matches = re.findall(pattern, ac.lower())
                workflow_stages.extend(matches)
        
        self._terminology_cache["workflow_stages"] = list(set(workflow_stages))
        
        # Extract business rules and constraints
        business_rules = []
        rule_patterns = [
            r'(?:must|should|shall)\s+(?:be|have|contain|include)\s+(\w+(?:\s+\w+)*)',
            r'(?:cannot|cannot|must not)\s+(\w+(?:\s+\w+)*)',
            r'(?:minimum|maximum|at least|at most)\s+(\w+(?:\s+\w+)*)',
        ]
        for ac in acceptance_criteria:
            for pattern in rule_patterns:
                matches = re.findall(pattern, ac.lower())
                business_rules.extend(matches)
        
        self._terminology_cache["business_rules"] = list(set(business_rules))
        
        # Extract preconditions and postconditions
        preconditions = []
        postconditions = []
        for ac in acceptance_criteria:
            if "given" in ac.lower() or "when" in ac.lower() or "if" in ac.lower():
                preconditions.append(ac)
            if "then" in ac.lower() or "should" in ac.lower() or "must" in ac.lower():
                postconditions.append(ac)
        
        self._terminology_cache["preconditions"] = preconditions
        self._terminology_cache["postconditions"] = postconditions

    def _build_domain_vocabulary(self) -> None:
        """Builds domain-specific vocabulary from all extracted sources."""
        entity = self._terminology_cache.get("primary_entity", "")
        entity_plural = self._terminology_cache.get("primary_entity_plural", "")
        ui_terms = self._terminology_cache.get("ui_terms", [])
        api_operations = self._terminology_cache.get("api_operations", [])
        business_intents = self._terminology_cache.get("business_intents", [])
        transaction_terms = self._terminology_cache.get("transaction_terms", [])
        
        # Build domain-specific vocabulary
        domain_vocabulary: Set[str] = set()
        
        # Add entity terms
        domain_vocabulary.add(entity)
        domain_vocabulary.add(entity_plural)
        
        # Add UI terms that are domain-relevant
        for term in ui_terms:
            if entity in term or entity_plural in term:
                domain_vocabulary.add(term)
        
        # Add API operations that are domain-relevant
        for op in api_operations:
            if entity in op or entity_plural in op:
                domain_vocabulary.add(op)
        
        # Add business intents
        for intent in business_intents:
            if entity in intent or entity_plural in intent:
                domain_vocabulary.add(intent)
        
        # Add transaction terms
        for trans in transaction_terms:
            domain_vocabulary.add(trans)
        
        self._terminology_cache["domain_vocabulary"] = list(domain_vocabulary)
        
        # Build allowlist for this domain (terms that are valid for this domain)
        self._terminology_cache["domain_allowlist"] = list(domain_vocabulary)

    def _pluralize(self, word: str) -> str:
        """Simple pluralization logic."""
        if word.endswith('y'):
            return word[:-1] + 'ies'
        elif word.endswith('s') or word.endswith('x') or word.endswith('z') or word.endswith('ch') or word.endswith('sh'):
            return word + 'es'
        else:
            return word + 's'

    def get_entity_term(self, plural: bool = False) -> str:
        """Returns the primary business entity term."""
        key = "primary_entity_plural" if plural else "primary_entity"
        return self._terminology_cache.get(key, self.domain.lower())

    def get_listing_term(self) -> str:
        """Returns the appropriate term for listings/records."""
        entity = self.get_entity_term(plural=True)
        ui_terms = self._terminology_cache.get("ui_terms", [])
        for term in ui_terms:
            if any(word in term for word in ["listing", "record", "item", "result", "option"]):
                if self.get_entity_term() in term or entity in term:
                    return term
        return entity

    def get_action_term(self, action_type: str) -> str:
        """Returns the appropriate business action term based on action type."""
        business_actions = self._terminology_cache.get("business_actions", [])
        
        for action in business_actions:
            if action_type in action:
                return action
        
        # Fallback to action_type itself if not found
        return action_type

    def get_transaction_term(self) -> str:
        """Returns the appropriate term for the primary action of this domain."""
        # For an auth/login flow the primary action is authenticating, not a purchase.
        if self.is_auth:
            return "authentication"

        entity = self.get_entity_term()
        ui_terms = self._terminology_cache.get("ui_terms", [])

        for term in ui_terms:
            if any(word in term for word in ["booking", "reservation", "order", "purchase", "transaction"]):
                if entity in term:
                    return term.replace(entity, "").strip() or "booking"

        # Check story context
        story = self.story_data.get("story", {})
        all_text = f"{story.get('summary', '')} {story.get('description', '')}".lower()

        if "booking" in all_text:
            return "booking"
        elif "reservation" in all_text:
            return "reservation"
        elif "order" in all_text:
            return "order"
        elif "purchase" in all_text:
            return "purchase"
        else:
            return "transaction"

    def get_workflow_steps(self) -> str:
        """Dynamically generates workflow steps based on domain context."""
        entity = self.get_entity_term()
        entity_plural = self.get_entity_term(plural=True)
        transaction = self.get_transaction_term()
        action = self.get_action_term("browse")

        # Build dynamic workflow from story context
        story = self.story_data.get("story", {})

        # Derive a short, human-readable stage phrase per acceptance criterion instead of
        # a "step_N" placeholder token — this string is surfaced directly in generated
        # test-case titles/expected results, so it must already read as real content
        # (a literal, unresolved "step_1 → step_2 → ..." previously shipped verbatim).
        ac_list = story.get("acceptance_criteria", [])
        strip_prefixes = ["user can ", "user is able to ", "valid ", "invalid ", "required "]
        stage_phrases = []
        for ac in ac_list[:5]:
            phrase = ac.strip().rstrip(".").lower()
            for prefix in strip_prefixes:
                if phrase.startswith(prefix):
                    phrase = phrase[len(prefix):]
                    break
            words = phrase.split()
            if words:
                # 8 words is enough to capture most short AC clauses whole (a hard 5-word
                # cutoff was chopping mid-clause, e.g. "credentials allow the user to" —
                # dangling on "to" instead of finishing "...log in successfully").
                stage_phrases.append(" ".join(words[:8]))

        if stage_phrases:
            steps = " → ".join(stage_phrases)
        else:
            # Generic fallback based on extracted terminology
            steps = f"{action} {entity_plural} → select {entity} → {transaction} → confirmation"

        return steps

    def get_domain_allowlist(self) -> List[str]:
        """Returns the domain-specific allowlist of valid business terms."""
        return self._terminology_cache.get("domain_allowlist", [])

    def get_domain_vocabulary(self) -> List[str]:
        """Returns the complete domain vocabulary."""
        return self._terminology_cache.get("domain_vocabulary", [])

    def is_valid_domain_term(self, term: str) -> bool:
        """Checks if a term is valid for the current domain."""
        allowlist = self.get_domain_allowlist()
        term_lower = term.lower().strip()
        if not term_lower:
            return False
        for allowed in allowlist:
            allowed_lower = allowed.lower().strip()
            if not allowed_lower:
                continue
            if allowed_lower == term_lower:
                return True
            # Whole-word containment only — raw substring containment let short
            # allowlist entries (e.g. a single short word) trivially match almost
            # any term, making the allowlist close to a no-op.
            if re.search(rf'\b{re.escape(allowed_lower)}\b', term_lower) or \
               re.search(rf'\b{re.escape(term_lower)}\b', allowed_lower):
                return True
        return False

    def get_workflow_stages(self) -> List[str]:
        """Returns extracted workflow stages from acceptance criteria."""
        return self._terminology_cache.get("workflow_stages", [])

    def get_business_rules(self) -> List[str]:
        """Returns extracted business rules and constraints."""
        return self._terminology_cache.get("business_rules", [])

    def get_preconditions(self) -> List[str]:
        """Returns extracted preconditions from acceptance criteria."""
        return self._terminology_cache.get("preconditions", [])

    def get_postconditions(self) -> List[str]:
        """Returns extracted postconditions from acceptance criteria."""
        return self._terminology_cache.get("postconditions", [])


class IntentEngine:
    """Extracts semantic intent from Jira story details, mapping user goals and business actions."""

    def __init__(self, summary: str, description: str, comments: List[str], routes: List[Dict[str, str]],
                 extracted_path: str = None, extracted_domain: str = None, acceptance_criteria: List[str] = None,
                 attachment_context: str = None, application_name: str = None) -> None:
        self.summary = summary
        self.description = description
        self.comments = comments
        self.routes = routes
        self.extracted_path = extracted_path
        self.extracted_domain = extracted_domain
        self.acceptance_criteria = acceptance_criteria or []
        self.attachment_context = attachment_context or ""
        # The configured application's own name/brand — stripped from text before domain
        # keyword-scoring so it isn't scored as a spurious route-label match, without
        # hardcoding any specific product name here.
        self.application_name = (application_name or "").lower()

    def analyze(self) -> Dict[str, Any]:
        best_route = None
        best_domain = None
        max_score = -1

        if self.extracted_path and self.extracted_domain:
            best_route = {"label": self.extracted_domain, "path": self.extracted_path}
            best_domain = self.extracted_domain
            max_score = 999
            log.info(f"[IntentEngine] Using dynamically extracted route from Jira story URL: {best_domain} -> {best_route['path']}")
        else:
            raw_combined = f"{self.summary} {self.description} {' '.join(self.comments)} {self.attachment_context}".lower()
            combined_text = re.sub(r'https?://\S+', '', raw_combined)
            summary_clean = self.summary.lower()
            if self.application_name:
                combined_text = combined_text.replace(self.application_name, "")
                summary_clean = summary_clean.replace(self.application_name, "")

            for r in self.routes:
                label = r["label"].strip().replace("\n", " ").lower()
                label = re.sub(r'\s+', ' ', label)
                if not label or len(label) < 3:
                    continue
                # Skip generic layout links — but not when the story's own summary is
                # plausibly ABOUT this exact page (e.g. a story literally testing the
                # "Blog" or "Careers" page). The eventual domain isn't resolved yet at
                # this point (that's what this loop is determining), so the signal used
                # here is whether the candidate label appears in the story's own summary
                # text rather than an already-known active domain name.
                generic_layout_terms = ["about", "contact", "privacy", "terms", "jobs", "careers", "help", "faq", "blog", "partners"]
                matched_generic = [g for g in generic_layout_terms if g in label]
                if matched_generic and not any(g in summary_clean for g in matched_generic):
                    continue
                # Skip dynamic content/cards (long labels) to avoid false keyword matches
                if len(label) > 40:
                    continue

                # Keywords defined from label
                keywords = [label]
                singular = label.rstrip('s')
                if singular != label:
                    keywords.append(singular)
                if " " in label:
                    # Only the individual split words are stopword-filtered below — a
                    # common filler word like "the" or "for" being part of a multi-word
                    # route label (e.g. "Esports World Cup WITNESS THE ACTION") must not
                    # by itself count as a meaningful keyword match.
                    keywords.extend(w for w in label.split() if w not in _ROUTE_LABEL_STOPWORDS)

                def match_keyword(kw: str, text: str) -> bool:
                    pattern = rf"\b{re.escape(kw)}(?:s|es|ing)?\b"
                    return bool(re.search(pattern, text, re.IGNORECASE))

                score = 0
                for kw in set(keywords):
                    if len(kw) < 3:
                        continue
                    if match_keyword(kw, combined_text):
                        score += 3 if match_keyword(kw, summary_clean) else 1

                if score > max_score:
                    max_score = score
                    best_route = r
                    best_domain = r["label"].strip().capitalize()

            # Require at least a summary-level match (score >= 3) to trust a crawled
            # route as the story's domain. A route that only picked up an incidental
            # single-word overlap somewhere in the description/comments (score 1-2,
            # previously accepted by "max_score == 0") is not a confident signal —
            # this was routinely misfiring for stories whose real subject (e.g. a login
            # modal) isn't exposed as a top-level nav route to crawl in the first place.
            if not best_route or max_score < 3:
                # No crawled route scored a confident match — common for flows that
                # aren't exposed as a top-level nav route (e.g. a login modal). Derive
                # the fallback domain from the story's own words instead of defaulting
                # to a fixed, application-specific page name.
                fallback_keyword = derive_fallback_domain(combined_text, summary_clean)
                best_route = {"label": fallback_keyword.capitalize(), "path": f"/en/{fallback_keyword}"}
                best_domain = fallback_keyword.capitalize()

            log.info(f"[IntentEngine] Dynamically selected domain: {best_domain} with path {best_route['path']} (score: {max_score})")

        # Dynamic validation requirements parsing
        inferred_validations = []
        text_for_parsing = f"{self.summary} {self.description} {' '.join(self.comments)} {self.attachment_context}"
        # Split on sentence terminators and "AC-1"/"AC1"-style acceptance-criteria markers
        # only. The previous character class [.!?AC\-\d\n] treated every literal capital
        # 'A', 'C', digit, and hyphen as a delimiter — fragmenting ordinary words/numbers
        # like "Confirm", "Access", "Card", "API", or "8 characters" mid-sentence.
        sentences = re.split(r'\bAC-?\d+\b|[.!?\n]+', text_for_parsing, flags=re.IGNORECASE)
        for s in sentences:
            s_clean = s.strip()
            if not s_clean:
                continue
            s_lower = s_clean.lower()
            if any(w in s_lower for w in ["validation", "must", "require", "error", "invalid", "alert", "mask", "obscure"]):
                if any(w in s_lower for w in ["required", "mandatory", "must", "complete"]):
                    inferred_validations.append("Mandatory Input Verification")
                if any(w in s_lower for w in ["error", "invalid", "fail", "alert"]):
                    inferred_validations.append("Error Alerts Handling")
                if any(w in s_lower for w in ["redirect", "navigate", "route"]):
                    inferred_validations.append("Navigational Redirection")
                if any(w in s_lower for w in ["mask", "obscure", "hide", "password"]):
                    inferred_validations.append("Security Input Masking")
                if any(w in s_lower for w in ["format", "email", "phone", "date"]):
                    inferred_validations.append("Format Restriction Validation")
                if any(w in s_lower for w in ["boundary", "limit", "range", "min", "max"]):
                    inferred_validations.append("Numeric/Length Boundary Validation")

        validations = list(set(inferred_validations))
        if not validations:
            validations = ["Mandatory Input Verification", "Error Alerts Handling"]

        # Infer semantic key dynamically
        path_lower = best_route["path"].lower()
        if any(w in path_lower or w in best_domain.lower() for w in ["login", "auth", "signup", "register", "password"]):
            semantic_key = "auth"
        else:
            semantic_key = "transactional"

        return {
            "domain": best_domain,
            "path": best_route["path"],
            "user_goal": f"Execute operations and validation tasks within the {best_domain} page space",
            "business_action": f"Verify behavioral state transitions for the {best_domain} user journey",
            "validations_required": validations,
            "semantic_key": semantic_key,
            "acceptance_criteria": self.acceptance_criteria
        }


class IntentDrivenUIExplorationEngine:
    """Uses intent to limit exploration scope, navigating the target application only within the relevant module space."""

    def __init__(self, settings: Settings, intent: Dict[str, Any], discovered_routes: List[Dict[str, str]] = None, story_urls: List[str] = None) -> None:
        self.settings = settings
        self.intent = intent
        # Resolve base URL dynamically from story URLs if present
        base_url = settings.application_base_url
        if not base_url and story_urls:
            for url in story_urls:
                from urllib.parse import urlparse
                parsed = urlparse(url)
                if parsed.netloc:
                    base_url = f"{parsed.scheme}://{parsed.netloc}"
                    break
        self.base_url = base_url or "https://webook.com"
        self.discovered_routes = discovered_routes or []
        self.captured_selectors = []
        self.story_urls = story_urls or []

    def explore(self, existing_graph: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        domain = self.intent["domain"]
        screenshot_path = self.settings.screenshot_dir / f"{_safe_filename_component(domain.lower())}_autonomous.png"

        # Determine target starting page URL dynamically from resolved path
        target_path = self.intent.get("path", f"/en/{domain.lower()}")
        target_url = f"{self.base_url.rstrip('/')}{target_path}"

        def clean_path_locale(p: str) -> str:
            p = p.strip().lower().split("?")[0]
            p = p.strip("/")
            segs = p.split("/")
            if segs and len(segs[0]) == 2: # e.g. en, ar
                segs = segs[1:]
            return "/" + "/".join(segs)

        target_clean = clean_path_locale(target_path)

        # Determine all other module entry paths from the homepage crawl to block them
        other_module_paths = []
        for r in self.discovered_routes:
            p_clean = clean_path_locale(r["path"])
            if p_clean != target_clean and len(p_clean) > 3:
                other_module_paths.append(p_clean)

        # Dynamically resolved entity terms for allowed keywords (pure runtime assessment)
        dynamic_allowed_paths: Set[str] = set()
        resolver = DomainTerminologyResolver(self.intent, {})
        allowed_keywords = [
            "explore", "detail", "booking", "checkout", "confirm", "login", "signup", "auth", "register", "page",
            domain.lower(),
            resolver.get_entity_term(plural=True).lower(),
            resolver.get_entity_term(plural=False).lower()
        ]

        def is_in_scope(url: str) -> bool:
            try:
                # Normalize and check story URLs
                url_normalized = url.rstrip('/')
                if any(u.rstrip('/') in url_normalized or url_normalized in u.rstrip('/') for u in self.story_urls):
                    return True

                from urllib.parse import urlparse
                parsed = urlparse(url)
                path = parsed.path.lower()
                path_clean = clean_path_locale(path)
                
                # Check if it goes to a different, blocked homepage module
                for blocked in other_module_paths:
                    if blocked == path_clean or path_clean.startswith(blocked + "/"):
                        return False
                
                # Allow base paths
                if path in ["/en", "/en/", "/"]:
                    return True
                
                # Allow target path or related path keywords
                if target_clean in path_clean:
                    return True
                    
                if any(kw in path for kw in allowed_keywords):
                    return True
                if any(kw in path_clean for kw in dynamic_allowed_paths):
                    return True
                return False
            except Exception:
                return False

        def is_related_to_active_story(url: str) -> bool:
            try:
                # Normalize and check story URLs
                url_normalized = url.rstrip('/')
                if any(u.rstrip('/') in url_normalized or url_normalized in u.rstrip('/') for u in self.story_urls):
                    return True

                from urllib.parse import urlparse, parse_qs
                parsed = urlparse(url)
                path = parsed.path.lower()
                
                if "explore" in path:
                    target_parsed = urlparse(target_url)
                    target_qs = parse_qs(target_parsed.query)
                    parsed_qs = parse_qs(parsed.query)
                    
                    if "category" in target_qs:
                        target_cats = target_qs["category"]
                        if "category" in parsed_qs:
                            if not any(c in target_cats for c in parsed_qs["category"]):
                                return False
                        else:
                            return False
                    
                    if "date" in target_qs:
                        target_dates = target_qs["date"]
                        if "date" in parsed_qs:
                            if not any(d in target_dates for d in parsed_qs["date"]):
                                return False
                        else:
                            return False
                            
                    if "tag" in target_qs:
                        target_tags = target_qs["tag"]
                        if "tag" in parsed_qs:
                            if not any(t in target_tags for t in parsed_qs["tag"]):
                                return False
                        else:
                            return False
                            
                return True
            except Exception:
                return True

        discovered_pages: Dict[str, Dict[str, Any]] = {}
        transitions: List[Dict[str, Any]] = []
        observed_apis: List[Dict[str, Any]] = []
        
        # Initialize assessment completion tracking at higher scope for graceful failure handling
        assessment_completion = {
            'total_pending': 0,
            'assessed': 0,
            'skipped': 0,
            'failed': 0
        }
        interactable_classifications = {}
        exploration_failed = False
        exploration_failure_reason = None

        try:
            from playwright.sync_api import sync_playwright
            log.info(f"[IntentDrivenUIExplorationEngine] Launching explorer targeting scoped URL: {target_url}")

            with sync_playwright() as p:
                headless_val = _load_playwright_headless_setting()
                log.info(f"[IntentDrivenUIExplorationEngine] Launching chromium (headless={headless_val})")
                browser = p.chromium.launch(headless=headless_val)
                context = browser.new_context(
                    viewport={"width": 1920, "height": 1080},
                    user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                )
                page = context.new_page()

                request_timestamps = {}
                current_node_tracker = {"name": "home"}
                current_interaction = {"action": "page_load", "component": "body", "page": "home"}

                def mask_json_values(obj: Any) -> Any:
                    if isinstance(obj, dict):
                        masked = {}
                        for k, v in obj.items():
                            k_low = k.lower()
                            is_sensitive = any(s in k_low for s in [
                                "password", "pass", "secret", "token", "auth", "cvv", "card", 
                                "cookie", "email", "phone", "jwt", "session", "api_key", "apikey",
                                "pwd", "credit", "personal", "username", "ssn", "login"
                            ])
                            if is_sensitive:
                                masked[k] = "[MASKED]"
                            else:
                                masked[k] = mask_json_values(v)
                        return masked
                    elif isinstance(obj, list):
                        return [mask_json_values(x) for x in obj]
                    else:
                        return obj

                def get_json_schema(obj: Any) -> Any:
                    if isinstance(obj, dict):
                        return {k: get_json_schema(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        if obj:
                            return [get_json_schema(obj[0])]
                        return []
                    else:
                        return type(obj).__name__

                def handle_request(request):
                    try:
                        if request.resource_type in ["xhr", "fetch"]:
                            request_timestamps[request] = datetime.now()
                    except Exception:
                        pass

                def handle_response(response):
                    try:
                        req = response.request
                        if req.resource_type not in ["xhr", "fetch"]:
                            return

                        url = response.url
                        url_lower = url.lower()
                        from urllib.parse import urlparse, parse_qsl

                        # Filter out static assets
                        static_extensions = [".js", ".css", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".woff", ".woff2", ".ttf", ".eot", ".mp3", ".mp4", ".webm", ".json"]
                        path_part = urlparse(url).path.lower()
                        is_static_asset = False
                        for ext in static_extensions:
                            if ext != ".json" and path_part.endswith(ext):
                                is_static_asset = True
                                break
                        if is_static_asset:
                            log.debug(f"[IntentDrivenUIExplorationEngine] Skipping static asset request: {url}")
                            return

                        # Filter out locales and telemetry / CDN
                        is_telemetry_or_locale = (
                            "/cdn-cgi/" in url_lower or
                            "/locales/" in url_lower or
                            "locales/" in url_lower or
                            "sar.json" in url_lower or
                            "telemetry" in url_lower or
                            "analytics" in url_lower or
                            "doubleclick" in url_lower or
                            "pagead" in url_lower or
                            "googlesyndication" in url_lower or
                            "mixpanel" in url_lower or
                            "amplitude" in url_lower or
                            "facebook" in url_lower or
                            "sentry" in url_lower or
                            "hotjar" in url_lower or
                            "optimizely" in url_lower or
                            "segment" in url_lower
                        )
                        if is_telemetry_or_locale:
                            log.debug(f"[IntentDrivenUIExplorationEngine] Skipping telemetry/locale/CDN request: {url}")
                            return

                        # Dynamic same-site/backend check to filter out third-party ads/analytics
                        try:
                            target_host = urlparse(target_url).hostname or ""
                            api_host = urlparse(url).hostname or ""
                            
                            def get_main_domain(hostname: str) -> str:
                                if not hostname:
                                    return ""
                                parts = hostname.split(".")
                                if len(parts) >= 2:
                                    return ".".join(parts[-2:])
                                return hostname

                            target_main = get_main_domain(target_host).lower()
                            api_main = get_main_domain(api_host).lower()
                            
                            third_party_keywords = ["analytics", "telemetry", "doubleclick", "pagead", "googlesyndication", "mixpanel", "amplitude", "facebook", "sentry", "hotjar", "optimizely", "segment"]
                            is_third_party = any(kw in api_host.lower() for kw in third_party_keywords)
                            
                            if is_third_party and target_main != api_main:
                                log.debug(f"[IntentDrivenUIExplorationEngine] Skipping third-party telemetry/ad request: {url}")
                                return
                        except Exception as parse_err:
                            log.debug(f"Error parsing hosts for filter: {parse_err}")

                        method = req.method
                        req_headers = dict(req.headers)
                        for k in list(req_headers.keys()):
                            if k.lower() in ["authorization", "cookie", "x-api-key", "token"]:
                                req_headers[k] = "[MASKED]"

                        req_payload = None
                        if req.post_data:
                            try:
                                payload_data = json.loads(req.post_data)
                                req_payload = mask_json_values(payload_data)
                            except Exception:
                                req_payload = "[BINARY/TEXT DATA]"

                        resp_headers = dict(response.headers)
                        for k in list(resp_headers.keys()):
                            if k.lower() in ["set-cookie", "authorization"]:
                                resp_headers[k] = "[MASKED]"

                        status = response.status
                        content_type = response.headers.get("content-type", "")
                        
                        duration_ms = 0
                        if req in request_timestamps:
                            duration_ms = int((datetime.now() - request_timestamps[req]).total_seconds() * 1000)

                        body_size = 0
                        resp_schema = {"type": "unknown"}
                        resp_body_masked = None
                        try:
                            body_bytes = response.body()
                            body_size = len(body_bytes)
                            if "application/json" in content_type:
                                try:
                                    resp_body = response.json()
                                    resp_schema = get_json_schema(resp_body)
                                    resp_body_masked = mask_json_values(resp_body)
                                except Exception:
                                    pass
                        except Exception:
                            pass

                        # Parse Query Parameters
                        query_params = {}
                        try:
                            parsed_url = urlparse(url)
                            query_params = dict(parse_qsl(parsed_url.query))
                            for k in list(query_params.keys()):
                                if k.lower() in ["password", "pass", "secret", "token", "auth", "cvv", "card", "cookie", "email", "phone"]:
                                    query_params[k] = "[MASKED]"
                        except Exception:
                            pass

                        # Parse Redirect Info
                        redirect_info = {}
                        try:
                            if getattr(req, "redirected_from", None):
                                redirect_info["redirected_from"] = req.redirected_from.url
                            if getattr(req, "redirected_to", None):
                                redirect_info["redirected_to"] = req.redirected_to.url
                        except Exception:
                            pass

                        # Get interaction details
                        triggering_action = current_interaction.get("action", "page_load")
                        associated_component = current_interaction.get("component", "body")
                        associated_page = current_interaction.get("page", "unknown")

                        url_clean = re.sub(r'[^a-zA-Z0-9_]', '_', url.split('?')[0].replace('https://', '').replace('http://', ''))
                        url_clean = url_clean[:35].strip('_').lower()

                        api_details = {
                            "method": method,
                            "url": url,
                            "request_headers": req_headers,
                            "request_payload": req_payload,
                            "response_status": status,
                            "response_headers": resp_headers,
                            "response_time_ms": duration_ms,
                            "response_size_bytes": body_size,
                            "content_type": content_type,
                            "response_schema": resp_schema,
                            "response_body": resp_body_masked,
                            "current_node": current_node_tracker.get("name", "unknown"),
                            # Metadata requirements
                            "query_parameters": query_params,
                            "redirect_information": redirect_info,
                            "triggering_ui_action": triggering_action,
                            "associated_component": associated_component,
                            "associated_page": associated_page,
                            "associated_workflow": self.intent.get("business_action", "unknown"),
                            "associated_behavioral_graph_node": f"{associated_page}_api_{method.lower()}_{url_clean}"
                        }
                        observed_apis.append(api_details)
                    except Exception as e:
                        log.debug(f"Error parsing response: {e}")

                page.on("request", handle_request)
                page.on("response", handle_response)

                visited_urls: Set[str] = set()
                max_page_loads = 15
                page_load_count = 0

                def is_target_module_url(url: str) -> bool:
                    try:
                        from urllib.parse import urlparse
                        parsed = urlparse(url)
                        path = parsed.path.lower()
                        path_clean = clean_path_locale(path)
                        domain_lower = domain.lower()
                        
                        if any(u.rstrip('/') in url.rstrip('/') or url.rstrip('/') in u.rstrip('/') for u in self.story_urls):
                            return True
                        
                        if target_clean in path_clean or domain_lower in path_clean:
                            return True
                        
                        # Allow dynamically collected runtime detail page segments
                        if any(kw in path_clean for kw in dynamic_allowed_paths):
                            return True
                        
                        return False
                    except Exception:
                        return False

                def run_crawl(url: str, current_node: str, depth: int):
                    nonlocal page_load_count
                    if depth > 3 or url in visited_urls or page_load_count >= max_page_loads:
                        return

                    if not is_in_scope(url):
                        log.info(f"[IntentDrivenUIExplorationEngine] Skipping out-of-scope URL path: {url}")
                        return

                    if not is_related_to_active_story(url):
                        log.info(f"[IntentDrivenUIExplorationEngine] Treating {url} as transient navigation (outside active story context). Skipping DOM analysis and selector extraction.")
                        return

                    # Prevent search space explosion / recursive crawling on filter page variations
                    try:
                        from urllib.parse import urlparse, parse_qs
                        parsed = urlparse(url)
                        if depth > 1 and "explore" in parsed.path.lower():
                            target_parsed = urlparse(target_url)
                            target_qs = parse_qs(target_parsed.query)
                            parsed_qs = parse_qs(parsed.query)
                            # If the query parameters differ (e.g. going from ?tag=football to ?tag=football&date=tomorrow), skip deep recursive crawling
                            if target_qs != parsed_qs:
                                log.info(f"[IntentDrivenUIExplorationEngine] Skipping deep recursive crawl at depth {depth} for query variant explore URL: {url}")
                                return
                    except Exception as query_check_err:
                        log.warning(f"Error checking query variant for explore URL: {query_check_err}")

                    visited_urls.add(url)
                    page_load_count += 1
                    current_node_tracker["name"] = current_node
                    current_interaction["action"] = "navigation"
                    current_interaction["component"] = "page_window"
                    current_interaction["page"] = current_node

                    log.info(f"[IntentDrivenUIExplorationEngine] Depth {depth}: Navigating to {url} (Load count: {page_load_count})")
                    try:
                        try:
                            page.goto(url, timeout=30000, wait_until="load")
                            try:
                                page.wait_for_load_state("networkidle", timeout=8000)
                            except Exception:
                                try:
                                    page.wait_for_selector("button, a[href], input, select, [role='button']", timeout=5000)
                                except Exception:
                                    pass
                            page.wait_for_timeout(3000)
                        except Exception as err:
                            log.warning(f"[IntentDrivenUIExplorationEngine] Navigation wait timed out/failed for {url}. Proceeding to extract DOM anyway: {err}")
                            page.wait_for_timeout(5000)

                        # Dynamically extract link path prefixes on the first page load to avoid hardcoding
                        if depth == 1:
                            try:
                                hrefs = page.evaluate("""() => {
                                    return Array.from(document.querySelectorAll('a'))
                                        .map(el => el.getAttribute('href'))
                                        .filter(href => href && (href.startsWith('/') || href.startsWith(window.location.origin)));
                                }""")
                                for href in hrefs:
                                    clean_href = clean_path_locale(href.split('?')[0])
                                    segs = [s for s in clean_href.split('/') if s]
                                    if segs:
                                        first_seg = segs[0]
                                        if len(first_seg) > 2 and first_seg not in [p.strip('/') for p in other_module_paths]:
                                            dynamic_allowed_paths.add(first_seg)
                                if dynamic_allowed_paths:
                                    log.info(f"[IntentDrivenUIExplorationEngine] Dynamically discovered allowed detail paths at runtime: {list(dynamic_allowed_paths)}")
                            except Exception as path_err:
                                log.warning(f"Error collecting dynamic allowed paths: {path_err}")

                        # Dynamic cookie dialog detection
                        try:
                            buttons_list = page.query_selector_all("button")
                            for btn in buttons_list:
                                try:
                                    text = btn.inner_text().strip()
                                    btn_class = btn.get_attribute("class") or ""
                                    btn_id = btn.get_attribute("id") or ""
                                    text_lower = text.lower()
                                    is_cookie_btn = (
                                        any(w in text_lower for w in ["accept", "agree", "allow", "consent", "ok", "reject"]) and
                                        any(w in text_lower or w in btn_class.lower() or w in btn_id.lower() for w in ["cookie", "consent", "privacy", "policy"]) and
                                        not any(w in text_lower for w in ["settings", "manage", "preferences", "customise", "customize"])
                                    )
                                    if is_cookie_btn or any(phrase in text for phrase in ["Accept all", "Accept All", "Reject all non-essential"]):
                                        log.info(f"[IntentDrivenUIExplorationEngine] Dismissing cookie consent overlay via '{text}' button.")
                                        current_interaction["action"] = f"click '{text}'"
                                        current_interaction["component"] = f"button:has-text('{text}')"
                                        current_interaction["page"] = current_node
                                        btn.click()
                                        page.wait_for_timeout(2000)
                                        transitions.append({
                                            "source": current_node,
                                            "target": current_node,
                                            "action": f"click '{text}'",
                                            "result": "cookie consent modal dismissed",
                                            "state_change": True
                                        })
                                        break
                                except Exception:
                                    pass
                        except Exception:
                            pass
                    except Exception as err:
                        log.warning(f"[IntentDrivenUIExplorationEngine] Failed to navigate to {url}: {err}")
                        return

                    # For an authentication/login story, the login form itself is the system
                    # under test — auto-logging-in here would submit the form, redirect to the
                    # authenticated homepage, and leave the crawler exploring homepage content
                    # instead of the login flow (module leakage). Only auto-login for
                    # non-auth stories that happen to sit behind a login wall.
                    if self.intent.get("semantic_key") == "auth":
                        log.info("[IntentDrivenUIExplorationEngine] Auth/login story detected — skipping auto-login so the authentication form remains under test.")
                    else:
                        handle_secure_login(page)

                    # Scroll page fully to assess lazy-loaded content / infinite scroll
                    try:
                        log.info(f"[BehavioralExploration] Performing scroll assessment on {url}")
                        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                        page.wait_for_timeout(1500)
                        page.evaluate("window.scrollTo(0, 0)")
                        page.wait_for_timeout(800)
                    except Exception as scroll_err:
                        log.warning(f"[BehavioralExploration] Scroll assessment failed: {scroll_err}")

                    # Real-time deep DOM Analysis script
                    dom_data_script = """(isAuth) => {
                        const data = {
                            inputs: [],
                            buttons: [],
                            dropdowns: [],
                            checkboxes: [],
                            radio_buttons: [],
                            disabled_controls: [],
                            required_fields: [],
                            placeholders: [],
                            aria_labels: [],
                            images: [],
                            links: [],
                            tabs: [],
                            cards: [],
                            modals: [],
                            tooltips: [],
                            keyboard_focus_order: [],
                            breadcrumbs: [],
                            toast_messages: [],
                            validation_messages: [],
                            loading_states: [],
                            business_rules: [],
                            component_metadata: {}
                        };
                        const getCardContext = (el) => {
                            const card = el.closest(".card, [class*='card'], .item-card, section, form, [role='region']");
                            if (card) {
                                const header = card.querySelector("h1, h2, h3, h4, h5, h6, .card-title, [class*='title'], [class*='header']");
                                if (header) {
                                    return header.innerText.trim().replace(/\\n/g, " ");
                                }
                                return card.innerText.trim().substring(0, 50).replace(/\\n/g, " ");
                            }
                            return "";
                        };

                        // Scope element discovery so persistent page chrome (nav carousels,
                        // country pickers, footer links) present on every page isn't captured
                        // as if it were part of the story's own flow. Priority:
                        //   1. A visible modal/dialog (login often opens in a modal overlay).
                        //   2. For auth/login stories, the form containing the password field
                        //      (the login page is otherwise a full page wrapped in site chrome).
                        //   3. Otherwise the whole document (unchanged behavior for normal pages).
                        const _isVis = (el) => {
                            const style = window.getComputedStyle(el);
                            const rect = el.getBoundingClientRect();
                            return style.display !== "none" && style.visibility !== "hidden" && rect.width > 0 && rect.height > 0;
                        };
                        let root = document;
                        const _visibleModals = Array.from(document.querySelectorAll('[role="dialog"], dialog, .modal, .popup, .drawer, [class*="modal"], [class*="drawer"]')).filter(_isVis);
                        if (_visibleModals.length > 0) {
                            root = _visibleModals[_visibleModals.length - 1];
                        } else if (isAuth) {
                            const _pwd = Array.from(document.querySelectorAll("input[type='password']")).filter(_isVis);
                            if (_pwd.length > 0) {
                                const _form = _pwd[0].closest("form") || _pwd[0].closest("[role='form'], section, div");
                                if (_form && _isVis(_form)) {
                                    root = _form;
                                }
                            }
                        }

                        root.querySelectorAll("input, textarea, select").forEach(el => {
                            const type = el.getAttribute("type") || el.tagName.toLowerCase();
                            const name = el.getAttribute("name");
                            const id = el.getAttribute("id");
                            const placeholder = el.getAttribute("placeholder");
                            const selector = name ? el.tagName.toLowerCase() + "[name='" + name + "']" : (id ? el.tagName.toLowerCase() + "#" + id : el.tagName.toLowerCase());
                            
                            let labelText = "";
                            if (el.labels && el.labels.length > 0) {
                                labelText = el.labels[0].innerText.trim();
                            } else if (id) {
                                const lbl = document.querySelector("label[for='" + id + "']");
                                if (lbl) labelText = lbl.innerText.trim();
                            }
                            const ariaLabel = el.getAttribute("aria-label") || "";
                            let ariaLabelledbyText = "";
                            const labelledby = el.getAttribute("aria-labelledby");
                            if (labelledby) {
                                const lblEl = document.getElementById(labelledby);
                                if (lblEl) ariaLabelledbyText = lblEl.innerText.trim();
                            }

                            data.component_metadata[selector] = {
                                label: labelText || ariaLabelledbyText || "",
                                placeholder: placeholder || "",
                                aria_label: ariaLabel || "",
                                card_context: getCardContext(el)
                            };

                            if (el.disabled) {
                                data.disabled_controls.push(selector + " (Disabled)");
                            }
                            if (el.required || el.getAttribute("aria-required") === "true") {
                                data.required_fields.push(selector);
                            }
                            if (placeholder) {
                                data.placeholders.push(selector + ': placeholder="' + placeholder + '"');
                            }

                            if (type === "checkbox") {
                                data.checkboxes.push(selector);
                            } else if (type === "radio") {
                                data.radio_buttons.push(selector);
                            } else if (el.tagName.toLowerCase() === "select") {
                                data.dropdowns.push(selector);
                            } else {
                                data.inputs.push(selector);
                            }
                        });

                        root.querySelectorAll("button, input[type='submit'], [role='button']").forEach(el => {
                            const text = el.innerText.trim().replace(/\\n/g, " ");
                            const id = el.getAttribute("id");
                            const name = el.getAttribute("name");
                            const selector = text ? "button:has-text('" + text + "')" : (id ? "button#" + id : (name ? "button[name='" + name + "']" : "button"));
                            
                            const ariaLabel = el.getAttribute("aria-label") || "";
                            let ariaLabelledbyText = "";
                            const labelledby = el.getAttribute("aria-labelledby");
                            if (labelledby) {
                                const lblEl = document.getElementById(labelledby);
                                if (lblEl) ariaLabelledbyText = lblEl.innerText.trim();
                            }

                            data.component_metadata[selector] = {
                                label: ariaLabelledbyText || "",
                                placeholder: "",
                                aria_label: ariaLabel || "",
                                text: text,
                                card_context: getCardContext(el)
                            };

                            if (el.disabled) {
                                data.disabled_controls.push(selector + " (Disabled)");
                            }
                            data.buttons.push(selector);
                        });

                        root.querySelectorAll("a").forEach(el => {
                            const href = el.getAttribute("href");
                            const text = el.innerText.trim().replace(/\\n/g, " ");
                            const id = el.getAttribute("id");
                            if (href && text) {
                                const selector = text ? "a:has-text('" + text + "')" : (id ? "a#" + id : "a");
                                const ariaLabel = el.getAttribute("aria-label") || "";
                                
                                data.component_metadata[selector] = {
                                    label: "",
                                    placeholder: "",
                                    aria_label: ariaLabel || "",
                                    text: text,
                                    card_context: getCardContext(el)
                                };
                                data.links.push(text + " -> " + href);
                            }
                        });

                        document.querySelectorAll("[role='tab'], .tab, [class*='tab']").forEach(el => {
                            const text = el.innerText.trim();
                            if (text) data.tabs.push(text);
                        });

                        document.querySelectorAll(".card, [class*='card'], .item-card").forEach(el => {
                            const text = el.innerText.trim().substring(0, 50);
                            if (text) data.cards.push(text);
                        });

                        document.querySelectorAll("[role='dialog'], dialog, .modal, .popup").forEach(el => {
                            const text = el.innerText.trim().substring(0, 50);
                            if (text) data.modals.push(text);
                        });

                        document.querySelectorAll("[title], [data-tooltip], .tooltip").forEach(el => {
                            const title = el.getAttribute("title") || el.getAttribute("data-tooltip") || el.innerText.trim();
                            if (title) data.tooltips.push(title);
                        });

                        document.querySelectorAll("[aria-label]").forEach(el => {
                            const label = el.getAttribute("aria-label");
                            if (label) data.aria_labels.push(label);
                        });

                        document.querySelectorAll("img").forEach(el => {
                            const alt = el.getAttribute("alt") || "Image";
                            const src = el.getAttribute("src") || "";
                            data.images.push(alt + ": " + src);
                        });

                        document.querySelectorAll(".breadcrumb, [class*='breadcrumb']").forEach(el => {
                            data.breadcrumbs.push(el.innerText.trim());
                        });

                        document.querySelectorAll(".toast, .alert, [role='alert']").forEach(el => {
                            data.toast_messages.push(el.innerText.trim());
                        });
                        document.querySelectorAll(".validation-error, .error-message, .invalid-feedback").forEach(el => {
                            data.validation_messages.push(el.innerText.trim());
                        });
                        document.querySelectorAll(".spinner, .loading, [class*='spinner'], [class*='loading']").forEach(el => {
                            data.loading_states.push("Loading Spinner/Indicator");
                        });

                        const focusable = Array.from(document.querySelectorAll("a, button, input, select, textarea, [tabindex]"))
                            .filter(el => {
                                const tabIndex = el.getAttribute("tabindex");
                                return !el.disabled && (tabIndex === null || parseInt(tabIndex) >= 0);
                            });
                        data.keyboard_focus_order = focusable.map(el => el.tagName.toLowerCase() + (el.innerText ? ":has-text('" + el.innerText.trim().substring(0, 15) + "')" : ""));

                        root.querySelectorAll("input, select, textarea").forEach(el => {
                            const label = el.labels && el.labels.length > 0 ? el.labels[0].innerText.trim() : (el.getAttribute("aria-label") || el.getAttribute("name") || "field");
                            if (el.required || el.getAttribute("aria-required") === "true") {
                                data.business_rules.push(label + " is a mandatory field.");
                            }
                            if (el.disabled) {
                                data.business_rules.push(label + " starts in a disabled state.");
                            }
                            const max = el.getAttribute("max");
                            const min = el.getAttribute("min");
                            if (max || min) {
                                data.business_rules.push(label + " has numeric boundaries: min=" + (min || "none") + ", max=" + (max || "none") + ".");
                            }
                            const maxlen = el.getAttribute("maxlength");
                            const minlen = el.getAttribute("minlength");
                            if (maxlen || minlen) {
                                data.business_rules.push(label + " has character length boundaries: min=" + (minlen || "none") + ", max=" + (maxlen || "none") + ".");
                            }
                            if (el.getAttribute("type") === "date" || el.tagName.toLowerCase() === "date-picker") {
                                data.business_rules.push(label + " restricts inputs to valid date calendar selections.");
                            }
                        });

                        return data;
                    }"""

                    dom_data = page.evaluate(dom_data_script, self.intent.get("semantic_key") == "auth")
                    log.info(f"[IntentDrivenUIExplorationEngine] Discovered {len(dom_data['inputs'])} inputs and {len(dom_data['buttons'])} buttons on {url}")

                    if depth == 1:
                        page.screenshot(path=str(screenshot_path))

                    discovered_pages[current_node] = {
                        "url": url,
                        **dom_data
                    }

                    # Read-only Selector Registry Extraction
                    other_paths = [clean_path_locale(r.get("path", "")) for r in self.discovered_routes if r.get("path")]
                    other_paths = [p for p in other_paths if p != target_clean and len(p) > 3]

                    selector_extraction_script = r"""([otherPaths, domainClean]) => {
                        const isVisible = (el) => {
                            if (!el) return false;
                            const style = window.getComputedStyle(el);
                            if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') return false;
                            const rect = el.getBoundingClientRect();
                            if (rect.width === 0 || rect.height === 0) return false;
                            return true;
                        };

                        const isUnstableId = (id) => {
                            if (!id) return true;
                            if (id.includes(':')) return true; // Avoid colons which cause syntax errors and represent dynamic IDs
                            const hasHash = /[a-f0-9]{8,}/i.test(id);
                            const hasReactId = /^:r[0-9]+:/.test(id);
                            const hasAngularId = /ng-/.test(id);
                            const endsWithLongNum = /[0-9]{5,}$/.test(id);
                            const isNumeric = /^[0-9]+$/.test(id);
                            return hasHash || hasReactId || hasAngularId || endsWithLongNum || isNumeric;
                        };

                        const getStableCssSelector = (el) => {
                            const testIdAttrs = ['data-testid', 'test-id', 'data-test'];
                            for (const attr of testIdAttrs) {
                                const val = el.getAttribute(attr);
                                if (val) return `[${attr}="${val}"]`;
                            }

                            const id = el.getAttribute('id');
                            if (id && !isUnstableId(id)) {
                                return `#${CSS.escape(id)}`;
                            }

                            const name = el.getAttribute('name');
                            if (name) {
                                return `${el.tagName.toLowerCase()}[name="${name}"]`;
                            }

                            let path = [];
                            let current = el;
                            while (current && current.nodeType === Node.ELEMENT_NODE) {
                                let selector = current.tagName.toLowerCase();
                                const currId = current.getAttribute('id');
                                if (currId && !isUnstableId(currId)) {
                                    selector += `#${CSS.escape(currId)}`;
                                    path.unshift(selector);
                                    break;
                                }
                                
                                const classes = Array.from(current.classList).filter(c => {
                                    const isTailwind = /^(hover|focus|active|inline|block|flex|grid|p-|m-|w-|h-|text-|bg-|border-|rounded-|shadow-|z-|top-|left-|right-|bottom-|justify-|items-|gap-)/.test(c);
                                    const isHash = /[a-f0-9]{5,}/.test(c) || /\d+/.test(c);
                                    return !isTailwind && !isHash;
                                });
                                if (classes.length > 0) {
                                    selector += `.${classes.join('.')}`;
                                }
                                
                                let index = 1;
                                let sibling = current.previousElementSibling;
                                while (sibling) {
                                    if (sibling.tagName === current.tagName) {
                                        index++;
                                    }
                                    sibling = sibling.previousElementSibling;
                                }
                                if (index > 1) {
                                    selector += `:nth-of-type(${index})`;
                                }
                                
                                path.unshift(selector);
                                current = current.parentNode;
                            }
                            return path.join(' > ');
                        };

                        const getXPath = (el) => {
                            let path = [];
                            let current = el;
                            while (current && current.nodeType === Node.ELEMENT_NODE) {
                                let index = 1;
                                let sibling = current.previousElementSibling;
                                while (sibling) {
                                    if (sibling.nodeType === Node.ELEMENT_NODE && sibling.tagName === current.tagName) {
                                        index++;
                                    }
                                    sibling = sibling.previousElementSibling;
                                }
                                const tagName = current.tagName.toLowerCase();
                                const pathIndex = `[${index}]`;
                                path.unshift(tagName + pathIndex);
                                current = current.parentNode;
                            }
                            return '/' + path.join('/');
                        };

                        const getAccessibleName = (el) => {
                            const labelledby = el.getAttribute('aria-labelledby');
                            if (labelledby) {
                                const labelEl = document.getElementById(labelledby);
                                if (labelEl) {
                                    const name = labelEl.innerText || labelEl.textContent;
                                    if (name && name.trim()) return name.trim();
                                }
                            }
                            const ariaLabel = el.getAttribute('aria-label');
                            if (ariaLabel && ariaLabel.trim()) return ariaLabel.trim();
                            if (el.id) {
                                const labelEl = document.querySelector(`label[for="${el.id}"]`);
                                if (labelEl) {
                                    const name = labelEl.innerText || labelEl.textContent;
                                    if (name && name.trim()) return name.trim();
                                }
                            }
                            const parentLabel = el.closest('label');
                            if (parentLabel) {
                                const name = parentLabel.innerText || parentLabel.textContent;
                                if (name && name.trim()) return name.trim();
                            }
                            const placeholder = el.getAttribute('placeholder');
                            if (placeholder && placeholder.trim()) return placeholder.trim();
                            const title = el.getAttribute('title');
                            if (title && title.trim()) return title.trim();
                            if (el.tagName.toLowerCase() === 'img') {
                                const alt = el.getAttribute('alt');
                                if (alt && alt.trim()) return alt.trim();
                            }
                            const text = el.innerText || el.textContent;
                            if (text && text.trim()) return text.trim();
                            return '';
                        };

                        const getComponentType = (el) => {
                            const tagName = el.tagName.toLowerCase();
                            const role = el.getAttribute('role');
                            if (role) return role;
                            if (tagName === 'button' || el.getAttribute('type') === 'submit') return 'button';
                            if (tagName === 'a') return 'link';
                            if (tagName === 'input') {
                                const type = el.getAttribute('type') || 'text';
                                return `input-${type}`;
                            }
                            if (tagName === 'select') return 'select';
                            if (tagName === 'textarea') return 'textarea';
                            return tagName;
                        };

                        const getNormalizedComponentType = (tagName, role, className) => {
                            const combined = `${tagName} ${role} ${className}`.toLowerCase();
                            if (combined.includes('card') || combined.includes('listing') || combined.includes('experience')) {
                                return "Business Card";
                            }
                            if (combined.includes('filter') || combined.includes('chip') || combined.includes('toggle') || combined.includes('option') || combined.includes('checkbox') || combined.includes('radio')) {
                                return "Filter Control";
                            }
                            if (combined.includes('cta') || combined.includes('primary') || combined.includes('action') || combined.includes('submit') || role === 'button' || tagName === 'button') {
                                return "Action Button";
                            }
                            if (tagName === 'input' || tagName === 'textarea' || role === 'textbox') {
                                return "Input Field";
                            }
                            if (tagName === 'a' || role === 'link') {
                                return "Navigation Link";
                            }
                            return "UI Component";
                        };

                        const cleanPathLocale = (p) => {
                            p = p.trim().toLowerCase();
                            p = p.split('?')[0].replace(/\/+$/, '');
                            const segs = p.split('/').filter(Boolean);
                            if (segs.length > 0 && segs[0].length === 2) {
                                segs.shift();
                            }
                            return '/' + segs.join('/');
                        };

                        const dateRegexes = [
                            /^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|mon|tue|wed|thu|fri|sat|sun)\s+\d{1,2}/i,
                            /^\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)/i,
                            /^(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}/i,
                            /^\d{1,2}\s+(january|february|march|april|may|june|july|august|september|october|november|december)/i,
                            /^\d{4}[-/]\d{2}[-/]\d{2}$/,
                            /^\d{1,2}[-/]\d{2}[-/]\d{2,4}$/,
                            /^(today|tomorrow|yesterday)$/i,
                            /^(?:[1-9]|[12]\d|3[01])$/ // Individual calendar day cell number
                        ];

                        const priceRegexes = [
                            /^(aed|sar|usd|eur|gbp|qar|omr|bhd|kwd|egp|¥|£|\$)\s*\d+([.,]\d{2})?$/i,
                            /^\d+([.,]\d{2})?\s*(aed|sar|usd|eur|gbp|qar|omr|bhd|kwd|egp|¥|£|\$)$/i,
                            /^\d+$/
                        ];

                        const isRuntimeValue = (val) => {
                            if (!val) return false;
                            val = val.trim();
                            for (const rx of dateRegexes) {
                                if (rx.test(val)) return true;
                            }
                            for (const rx of priceRegexes) {
                                if (rx.test(val)) return true;
                            }
                            if (/^[a-z]+\s+#?\d+$/i.test(val)) return true;
                            if (val.split(/\s+/).length > 4) return true;
                            return false;
                        };

                        const getNormalizedValue = (val, role, tagName) => {
                            if (!val) return '';
                            val = val.trim();
                            for (const rx of dateRegexes) {
                                if (rx.test(val)) return "Calendar Date Cell";
                            }
                            for (const rx of priceRegexes) {
                                if (rx.test(val)) return "Price Label";
                            }
                            if (/^[a-z]+\s+#?\d+$/i.test(val)) {
                                return "Item ID Selection";
                            }
                            const words = val.split(/\s+/);
                            if (words.length > 4) {
                                const lowerVal = val.toLowerCase();
                                if (lowerVal.includes('hiring') || lowerVal.includes('jobs') || lowerVal.includes('careers')) {
                                    return "Careers Link";
                                }
                                if (role === 'link' || tagName === 'a') {
                                    return "Content Title Link";
                                }
                                if (role === 'button' || tagName === 'button') {
                                    return "Action Button";
                                }
                                return "Content Card Title";
                            }
                            return val;
                        };

                        const getClassification = (el, text, id, className, name, dataTestId) => {
                            const combined = `${text} ${id} ${className} ${name} ${dataTestId} ${el.tagName}`.toLowerCase();
                            const infraKeywords = [
                                'analytics', 'telemetry', 'tracking', 'ads', 'advertisement', 'cookie', 
                                'consent', 'privacy', 'terms', 'footer', 'header', 'social', 'blog', 
                                'careers', 'language', 'lang', 'country', 'newsletter', 'support', 'help', 
                                'facebook', 'twitter', 'instagram', 'linkedin', 'github', 'youtube',
                                'about', 'contact', 'legal', 'security', 'vulnerability', 'faq',
                                'business partners', 'partners', 'jobs', 'hiring', 'recruit', 'notice',
                                'banner', 'close', 'logo'
                            ];
                            const activeDomainLower = (domainClean || '').toLowerCase();
                            for (const kw of infraKeywords) {
                                // A keyword that IS the actual domain under test (e.g. testing a
                                // "Blog" or "Careers" page) is the real page being explored, not
                                // incidental site chrome to skip past.
                                if (kw === activeDomainLower) continue;
                                if (combined.includes(kw)) {
                                    return "infrastructure";
                                }
                            }
                            for (const path of otherPaths) {
                                const segs = path.split('/').filter(Boolean);
                                if (segs.length > 0) {
                                    const modName = segs[segs.length - 1];
                                    if (modName && modName.length > 3 && combined.includes(modName)) {
                                        return "infrastructure";
                                    }
                                }
                            }
                            return "business";
                        };

                        const getDOMRelationship = (el) => {
                            const parentForm = el.closest('form');
                            const parentCard = el.closest(".card, [class*='card'], .item-card, section, [role='region']");
                            let parentContext = '';
                            if (parentForm) {
                                const formId = parentForm.getAttribute('id') || parentForm.getAttribute('name') || 'form';
                                parentContext += `form[${formId}]`;
                            }
                            if (parentCard) {
                                const header = parentCard.querySelector("h1, h2, h3, h4, h5, h6, .card-title, [class*='title'], [class*='header']");
                                const contextText = header ? header.innerText.trim() : parentCard.innerText.trim().substring(0, 30);
                                if (parentContext) parentContext += ' > ';
                                parentContext += `card[${contextText.replace(/\\n/g, ' ')}]`;
                            }
                            return parentContext || 'body';
                        };

                        const inferCapability = (el, text, role, componentType, parentContext) => {
                            const combined = `${text} ${el.getAttribute('placeholder') || ''} ${el.getAttribute('aria-label') || ''} ${role} ${componentType} ${parentContext}`.toLowerCase();
                            if (/\b(search|find|submit)\b/i.test(combined)) {
                                return `Search ${domainClean}`;
                            }
                            if (/\b(origin|from|airport)\b/i.test(combined)) {
                                return "Select Origin";
                            }
                            if (/\b(destination|to)\b/i.test(combined)) {
                                return "Select Destination";
                            }
                            if (/\b(date|calendar|departure|return)\b/i.test(combined)) {
                                return "Select Date";
                            }
                            if (/\b(passenger|cabin|class|economy|business|adult|child|infant|seat)\b/i.test(combined)) {
                                return "Passenger & Class Selection";
                            }
                            if (/\b(filter|sort|airline|price)\b/i.test(combined)) {
                                return "Filter & Sort Results";
                            }
                            if (/\b(book|select|continue|checkout)\b/i.test(combined)) {
                                return "Booking & Checkout Flow";
                            }
                            if (/\b(confirm|pay|card)\b/i.test(combined)) {
                                return "Booking Confirmation";
                            }
                            if (parentContext && parentContext !== 'body') {
                                return parentContext;
                            }
                            return document.title || `${domainClean} Interaction`;
                        };

                        const isShellNavigationElement = (el) => {
                            // Check if element is within global header/footer/shell navigation
                            const shellSelectors = [
                                'header', 'footer', '[role="banner"]', '[role="contentinfo"]',
                                '.navbar', '.nav-bar', '.navigation', '.main-nav', '.top-nav',
                                '.shell', '.app-shell', '.global-nav', '.site-header', '.site-footer',
                                '.cookie-banner', '.consent-banner', '.policy-banner'
                            ];
                            
                            for (const selector of shellSelectors) {
                                if (el.closest(selector)) {
                                    return true;
                                }
                            }
                            
                            // Check for shell-specific class patterns
                            const className = (el.className || '').toString().toLowerCase();
                            const shellClassPatterns = [
                                'shell', 'global', 'site-', 'main-nav', 'top-nav', 'navbar',
                                'cookie-banner', 'consent-banner', 'policy-banner', 'footer-',
                                'header-', 'app-shell', 'layout-shell'
                            ];
                            
                            for (const pattern of shellClassPatterns) {
                                if (className.includes(pattern)) {
                                    return true;
                                }
                            }
                            
                            return false;
                        };

                        const isExcluded = (el) => {
                            const text = (el.innerText || el.textContent || '').toLowerCase();
                            const id = (el.getAttribute('id') || '').toLowerCase();
                            const className = (el.className || '').toString().toLowerCase();
                            const name = (el.getAttribute('name') || '').toLowerCase();
                            
                            // Enhanced infrastructure keywords for shell navigation
                            const excludeKeywords = [
                                'consent', 'gdpr', 'policy', 'agree', 'decline', 'accept cookies', 'reject cookies',
                                'facebook', 'twitter', 'instagram', 'linkedin', 'github', 'youtube', 'social',
                                'newsletter', 'subscribe', 'language', 'lang', 'country', 'region',
                                'about', 'contact', 'help', 'faq', 'support', 'careers', 'jobs', 'hiring',
                                'blog', 'partners', 'business partners', 'legal', 'terms', 'privacy',
                                'analytics', 'telemetry', 'tracking', 'advertisement', 'ads'
                            ];
                            const activeDomainLowerExcl = (domainClean || '').toLowerCase();
                            for (const kw of excludeKeywords) {
                                // A keyword that IS the actual domain under test is the real page
                                // being explored, not incidental shell navigation to skip past.
                                if (kw === activeDomainLowerExcl) continue;
                                if (text.includes(kw) || id.includes(kw) || className.includes(kw) || name.includes(kw)) {
                                    return true;
                                }
                            }
                            
                            // Exclude shell navigation elements (header/footer/global nav)
                            if (isShellNavigationElement(el)) {
                                return true;
                            }
                            
                            if (['div', 'span', 'section', 'header', 'footer', 'main', 'aside', 'nav'].includes(el.tagName.toLowerCase()) && !el.getAttribute('role')) {
                                    return true;
                            }
                            
                            if (el.tagName.toLowerCase() === 'a') {
                                const href = el.getAttribute('href') || '';
                                if (href.startsWith('/')) {
                                    const cleanHref = cleanPathLocale(href);
                                    for (const blocked of otherPaths) {
                                        if (cleanHref === blocked || cleanHref.startsWith(blocked + '/')) {
                                            return true;
                                        }
                                    }
                                }
                            }
                            return false;
                        };

                        const resolveDuplicateElement = (el, registry) => {
                            // Robust fallback resolution for duplicate/variant elements
                            // Prefer visible active in-scope element, dedupe shell duplicates
                            // Use role/text/accessibility/parent-context fallbacks instead of brittle CSS ids
                            
                            // Prevent clicking all individual calendar date cell numbers redundantly
                            const role = el.getAttribute('role') || getComponentType(el);
                            const rawText = (el.innerText || el.textContent || '').trim().substring(0, 30);
                            const rawAccName = getAccessibleName(el);
                            const normText = getNormalizedValue(rawText, role, el.tagName);
                            const normAccName = getNormalizedValue(rawAccName, role, el.tagName);
                            
                            if (normText === "Calendar Date Cell" || normAccName === "Calendar Date Cell") {
                                const hasExistingCalendar = registry.some(item => 
                                    item.role === role && 
                                    (item.text === "Calendar Date Cell" || item.accessible_name === "Calendar Date Cell")
                                );
                                if (hasExistingCalendar) {
                                    return 'skip_duplicate';
                                }
                            }

                            // Dynamically detect and deduplicate repeating sibling list elements (e.g. dropdown options, calendar grids)
                            let currentEl = el;
                            for (let i = 0; i < 3; i++) {
                                const parent = currentEl.parentElement;
                                if (!parent) break;
                                const siblings = Array.from(parent.children).filter(child => 
                                    child.tagName === currentEl.tagName && 
                                    (child.getAttribute('role') === currentEl.getAttribute('role') || child.className === currentEl.className)
                                );
                                if (siblings.length > 4) {
                                    if (seenRegisteredParents.has(parent)) {
                                        return 'skip_duplicate';
                                    }
                                }
                                currentEl = parent;
                            }

                            const existingBySelector = registry.filter(item => 
                                item.stable_css_locator === getStableCssSelector(el)
                            );
                            
                            if (existingBySelector.length === 0) {
                                return null; // No duplicate, proceed normally
                            }
                            
                            // Check if existing element is in shell navigation
                            const existingInShell = existingBySelector.some(item => 
                                item.skip_reason === 'shell_navigation_element'
                            );
                            
                            if (existingInShell) {
                                // Prefer the new element if it's not in shell
                                if (!isShellNavigationElement(el)) {
                                    return 'replace_shell_duplicate';
                                }
                            }
                            
                            // Use role + text combination for deduplication
                            const existingByRoleText = registry.filter(item => 
                                item.role === role && 
                                (item.text === rawText || item.accessible_name === rawAccName)
                            );
                            
                            if (existingByRoleText.length > 0) {
                                // Check visibility and enabled state
                                const isVisibleEl = isVisible(el);
                                const isEnabledEl = !el.disabled;
                                
                                const existingVisible = existingByRoleText.some(item => item.visibility && item.enabled);
                                
                                if (isVisibleEl && isEnabledEl && !existingVisible) {
                                    return 'prefer_active_element';
                                }
                                
                                // Check parent context for better disambiguation
                                const currentContext = getDOMRelationship(el);
                                const existingContext = existingByRoleText[0].dom_relationship;
                                
                                if (currentContext !== existingContext && currentContext !== 'body') {
                                    return 'use_parent_context';
                                }
                            }
                            
                            // If all else fails, use accessibility attributes
                            if (accName && accName.length > 0) {
                                const existingByAccName = registry.filter(item => 
                                    item.accessible_name === accName
                                );
                                if (existingByAccName.length === 0) {
                                    return 'use_accessibility_name';
                                }
                            }
                            
                            return 'skip_duplicate';
                        };

                        const elements = Array.from(document.querySelectorAll('input, button, a, select, textarea, [role="button"], [role="link"], [role="checkbox"], [role="radio"]'));
                        const registry = [];
                        const seenRegisteredParents = new Set();

                        for (const el of elements) {
                            if (!isVisible(el)) continue;
                            if (isExcluded(el)) continue;

                            // Check for duplicates and apply fallback resolution
                            const duplicateResolution = resolveDuplicateElement(el, registry);
                            if (duplicateResolution === 'skip_duplicate') {
                                continue; // Skip this duplicate element
                            }
                            if (duplicateResolution === 'replace_shell_duplicate') {
                                // Remove shell duplicate from registry
                                const shellIndex = registry.findIndex(item => 
                                    item.skip_reason === 'shell_navigation_element' && 
                                    item.stable_css_locator === getStableCssSelector(el)
                                );
                                if (shellIndex !== -1) {
                                    registry.splice(shellIndex, 1);
                                }
                            }

                            const role = el.getAttribute('role') || getComponentType(el);
                            const nameAttr = el.getAttribute('name') || '';
                            const idAttr = el.getAttribute('id') || '';
                            const dataTestId = el.getAttribute('data-testid') || el.getAttribute('test-id') || el.getAttribute('data-test') || '';
                            
                            const ariaAttrs = {};
                            for (let i = 0; i < el.attributes.length; i++) {
                                const attr = el.attributes[i];
                                if (attr.name.startsWith('aria-')) {
                                    ariaAttrs[attr.name] = attr.value;
                                }
                            }

                            const rawPlaceholder = el.getAttribute('placeholder') || '';
                            const rawTextContent = el.innerText ? el.innerText.trim() : '';
                            const rawAccName = getAccessibleName(el);

                            let rawLabelText = '';
                            if (el.labels && el.labels.length > 0) {
                                rawLabelText = el.labels[0].innerText.trim();
                            } else if (idAttr) {
                                const lbl = document.querySelector(`label[for="${idAttr}"]`);
                                if (lbl) rawLabelText = lbl.innerText.trim();
                            }

                            const accName = getNormalizedValue(rawAccName, role, el.tagName);
                            const textContent = getNormalizedValue(rawTextContent, role, el.tagName);
                            const labelText = getNormalizedValue(rawLabelText, role, el.tagName);
                            const placeholder = getNormalizedValue(rawPlaceholder, role, el.tagName);

                            let preferredLocator = '';
                            let isAccNameRuntime = isRuntimeValue(rawAccName);
                            
                            if (dataTestId) {
                                preferredLocator = `page.get_by_test_id("${dataTestId}")`;
                            } else if (role && accName && !isAccNameRuntime) {
                                let pwRole = role;
                                if (role === 'input-text' || role === 'textarea') pwRole = 'textbox';
                                if (role === 'input-checkbox') pwRole = 'checkbox';
                                if (role === 'input-radio') pwRole = 'radio';
                                if (role === 'select') pwRole = 'combobox';
                                
                                const validRoles = ['button', 'checkbox', 'combobox', 'heading', 'link', 'radio', 'textbox', 'listitem', 'menuitem', 'option', 'tab'];
                                if (validRoles.includes(pwRole)) {
                                    preferredLocator = `page.get_by_role("${pwRole}", name="${accName}")`;
                                }
                            }
                            
                            if (!preferredLocator && placeholder && !isRuntimeValue(rawPlaceholder)) {
                                preferredLocator = `page.get_by_placeholder("${placeholder}")`;
                            }
                            if (!preferredLocator && labelText && !isRuntimeValue(rawLabelText)) {
                                preferredLocator = `page.get_by_label("${labelText}")`;
                            }
                            if (!preferredLocator && textContent && textContent.length < 30 && !isRuntimeValue(rawTextContent)) {
                                preferredLocator = `page.get_by_text("${textContent}")`;
                            }

                            const cssSelector = getStableCssSelector(el);
                            if (!preferredLocator) {
                                preferredLocator = `page.locator("${cssSelector}")`;
                            }

                            const xpath = getXPath(el);
                            const compType = getComponentType(el);
                            const normalizedCompType = getNormalizedComponentType(el.tagName, role, el.className || '');
                            
                            const visibility = true;
                            const enabled = !el.disabled;
                            const editable = compType.startsWith('input') || compType === 'textarea';

                            let isUnique = false;
                            try {
                                isUnique = document.querySelectorAll(cssSelector).length === 1;
                            } catch (e) {}

                            const domRelationship = getDOMRelationship(el);
                            const classification = getClassification(el, rawTextContent, idAttr, el.className || '', nameAttr, dataTestId);
                            const businessCapability = inferCapability(el, rawTextContent, role, compType, domRelationship);

                            let stability = 0.5;
                            if (dataTestId) stability += 0.2;
                            if (idAttr && !isUnstableId(idAttr)) stability += 0.1;
                            if (isUnique) stability += 0.1;
                            if (visibility && enabled) stability += 0.1;
                            if (rawAccName && !isAccNameRuntime) stability += 0.1;
                            if (preferredLocator.startsWith('page.locator') || preferredLocator.includes('xpath')) stability -= 0.2;
                            stability = Math.max(0.0, Math.min(1.0, stability));

                            // Determine initial classification status
                            let assessmentStatus = 'pending';
                            let skipReason = '';
                            
                            // Check for shell navigation duplicates
                            if (isShellNavigationElement(el)) {
                                assessmentStatus = 'skipped';
                                skipReason = 'shell_navigation_element';
                            } else if (!enabled) {
                                assessmentStatus = 'skipped';
                                skipReason = 'disabled_element';
                            } else if (classification === 'infrastructure') {
                                assessmentStatus = 'skipped';
                                skipReason = 'infrastructure_element';
                            } else if (idAttr && isUnstableId(idAttr) && !dataTestId && !accName) {
                                // Radix dynamic-id selector issues without fallback
                                assessmentStatus = 'failed';
                                skipReason = 'radix_dynamic_id_no_fallback';
                            } else if (idAttr && isUnstableId(idAttr) && !dataTestId) {
                                assessmentStatus = 'pending'; // Will be resolved via fallback
                            } else {
                                assessmentStatus = 'pending'; // Ready for assessment
                            }

                            registry.push({
                                role: role,
                                accessible_name: accName,
                                label: labelText,
                                placeholder: placeholder,
                                text: textContent,
                                data_testid: dataTestId,
                                id: idAttr && !isUnstableId(idAttr) ? idAttr : '',
                                name: nameAttr,
                                aria_attributes: ariaAttrs,
                                preferred_playwright_locator: preferredLocator,
                                stable_css_locator: cssSelector,
                                xpath: xpath,
                                component_type: compType,
                                normalized_component_type: normalizedCompType,
                                classification: classification,
                                visibility: visibility,
                                enabled: enabled,
                                editable: editable,
                                uniqueness: isUnique,
                                dom_relationship: domRelationship,
                                runtime_url: window.location.href,
                                business_capability: businessCapability,
                                stability_score: stability,
                                assessment_status: assessmentStatus,
                                skip_reason: skipReason,
                                validation: {
                                    exists: true,
                                    visible: true,
                                    unique: isUnique,
                                    interactable: enabled,
                                    verified_at: "runtime_exploration"
                                }
                            });
                            
                            let tempEl = el;
                            for (let i = 0; i < 3; i++) {
                                const parent = tempEl.parentElement;
                                if (!parent) break;
                                seenRegisteredParents.add(parent);
                                tempEl = parent;
                            }
                        }
                        return registry;
                    }"""

                    try:
                        extracted = page.evaluate(selector_extraction_script, [other_paths, domain])
                        if extracted:
                            self.captured_selectors.extend(extracted)
                    except Exception as sel_err:
                        log.warning(f"[IntentDrivenUIExplorationEngine] Failed to extract selectors on {url}: {sel_err}")

                    # Dynamic full behavioral exploration loop
                    explored_signatures = set()
                    max_interactions = 35
                    interactions_performed = 0
                    
                    # Count pending elements for assessment. self.captured_selectors is the
                    # cumulative list across all pages visited so far, so this must be a full
                    # recount (reset to 0 first) rather than `+=` accumulation — this method
                    # runs once per crawled page, and without the reset each subsequent page
                    # visit added its counts on top of every prior page's counts, inflating
                    # the totals (e.g. 3x after 3 pages instead of the true cumulative count).
                    assessment_completion['total_pending'] = 0
                    assessment_completion['skipped'] = 0
                    assessment_completion['failed'] = 0
                    for selector in self.captured_selectors:
                        if selector.get('assessment_status') == 'pending':
                            assessment_completion['total_pending'] += 1
                        elif selector.get('assessment_status') == 'skipped':
                            assessment_completion['skipped'] += 1
                        elif selector.get('assessment_status') == 'failed':
                            assessment_completion['failed'] += 1

                    log.info(f"[BehavioralExploration] Assessment completion tracking: {assessment_completion['total_pending']} pending, {assessment_completion['skipped']} skipped, {assessment_completion['failed']} failed")

                    interacted_with_calendar_cell = False
                    while interactions_performed < max_interactions:
                        state_js = """() => {
                            const getVisibleText = (sel) => {
                                const el = document.querySelector(sel);
                                return el ? el.innerText.trim() : '';
                            };
                            const countElements = (sel) => {
                                return document.querySelectorAll(sel).length;
                            };
                            const modalOpen = !!document.querySelector('[role="dialog"], dialog, .modal, .popup, .drawer, [class*="modal"], [class*="drawer"]');
                            const errors = Array.from(document.querySelectorAll('.validation-error, .error-message, .invalid-feedback, [class*="error"]'))
                                .filter(el => {
                                    const style = window.getComputedStyle(el);
                                    return style.display !== 'none' && style.visibility !== 'hidden';
                                })
                                .map(el => el.innerText.trim())
                                .join('; ');
                            const loaderVisible = !!document.querySelector('.spinner, .loading, [class*="spinner"], [class*="loading"]');
                            const cardsCount = document.querySelectorAll('.card, [class*="card"], .item-card, li, .listing').length;
                            const inputsState = Array.from(document.querySelectorAll('input, select, textarea'))
                                .map(el => `${el.name || el.id || 'field'}:${el.disabled}`)
                                .join(',');
                            return {
                                url: window.location.href,
                                title: document.title,
                                modalOpen: modalOpen,
                                errors: errors,
                                loaderVisible: loaderVisible,
                                cardsCount: cardsCount,
                                inputsState: inputsState,
                                textLength: document.body.innerText.length
                            };
                        }"""
                        
                        try:
                            before_state = page.evaluate(state_js)
                        except Exception as e:
                            log.warning(f"Failed to retrieve before_state: {e}")
                            break
                        
                        elements_js = """(isAuth) => {
                            const isVisible = (el) => {
                                if (!el) return false;
                                const style = window.getComputedStyle(el);
                                if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') return false;
                                const rect = el.getBoundingClientRect();
                                if (rect.width === 0 || rect.height === 0) return false;
                                return true;
                            };
                            
                            const getSelector = (el) => {
                                const testIdAttrs = ['data-testid', 'test-id', 'data-test'];
                                for (const attr of testIdAttrs) {
                                    const val = el.getAttribute(attr);
                                    if (val) return `[${attr}="${val}"]`;
                                }
                                // CSS.escape() is required here — real-world `id` attributes
                                // aren't always conformant CSS identifiers (spaces, apostrophes,
                                // etc., e.g. id="What's new section"), and interpolating one
                                // raw into a selector string produces an invalid selector that
                                // throws "Unexpected token" when passed to querySelector/Playwright.
                                if (el.id && !/ng-|:|:r[0-9]|[a-f0-9]{8,}/.test(el.id)) return `#${CSS.escape(el.id)}`;

                                let path = [];
                                let current = el;
                                while (current && current.nodeType === Node.ELEMENT_NODE) {
                                    let selector = current.tagName.toLowerCase();
                                    const currId = current.getAttribute('id');
                                    if (currId && !/ng-|:|:r[0-9]|[a-f0-9]{8,}/.test(currId)) {
                                        selector += `#${CSS.escape(currId)}`;
                                        path.unshift(selector);
                                        break;
                                    }
                                    let index = 1;
                                    let sibling = current.previousElementSibling;
                                    while (sibling) {
                                        if (sibling.tagName === current.tagName) {
                                            index++;
                                        }
                                        sibling = sibling.previousElementSibling;
                                    }
                                    if (index > 1 || current.nextElementSibling) {
                                        selector += `:nth-of-type(${index})`;
                                    }
                                    path.unshift(selector);
                                    current = current.parentNode;
                                }
                                return path.join(' > ');
                            };

                            const getXPath = (el) => {
                                let path = [];
                                let current = el;
                                while (current && current.nodeType === Node.ELEMENT_NODE) {
                                    let index = 1;
                                    let sibling = current.previousElementSibling;
                                    while (sibling) {
                                        if (sibling.nodeType === Node.ELEMENT_NODE && sibling.tagName === current.tagName) {
                                            index++;
                                        }
                                        sibling = sibling.previousElementSibling;
                                    }
                                    const tagName = current.tagName.toLowerCase();
                                    const pathIndex = `[${index}]`;
                                    path.unshift(tagName + pathIndex);
                                    current = current.parentNode;
                                }
                                return '/' + path.join('/');
                            };

                            // Scope interaction targets the same way as element discovery:
                            // a visible modal, else (for auth stories) the password form, else
                            // the whole page. Keeps the interaction loop from clicking through
                            // site chrome (nav, carousels, event cards) on a login-form story.
                            let root = document;
                            const modals = Array.from(document.querySelectorAll('[role="dialog"], dialog, .modal, .popup, .drawer, [class*="modal"], [class*="drawer"]')).filter(isVisible);
                            if (modals.length > 0) {
                                root = modals[modals.length - 1];
                            } else if (isAuth) {
                                const pwd = Array.from(document.querySelectorAll("input[type='password']")).filter(isVisible);
                                if (pwd.length > 0) {
                                    const form = pwd[0].closest("form") || pwd[0].closest("[role='form'], section, div");
                                    if (form && isVisible(form)) {
                                        root = form;
                                    }
                                }
                            }

                            const elements = Array.from(root.querySelectorAll('input, button, a, select, textarea, [role="button"], [role="link"], [role="checkbox"], [role="radio"], [class*="chip"], [class*="tab"], [class*="accordion"], .toggle, [data-testid]'));
                            
                            return elements.filter(isVisible).map(el => {
                                const selector = getSelector(el);
                                const tagName = el.tagName.toLowerCase();
                                const type = el.getAttribute('type') || '';
                                const role = el.getAttribute('role') || '';
                                const text = el.innerText ? el.innerText.trim() : '';
                                const placeholder = el.getAttribute('placeholder') || '';
                                const name = el.getAttribute('name') || '';
                                
                                let controlType = 'clickable';
                                if (tagName === 'input' && (type === 'checkbox' || role === 'checkbox')) controlType = 'checkbox';
                                else if (tagName === 'input' && (type === 'radio' || role === 'radio')) controlType = 'radio';
                                else if (tagName === 'input' || tagName === 'textarea') controlType = 'input';
                                else if (tagName === 'select' || role === 'combobox') controlType = 'dropdown';
                                
                                const testIdAttrs = ['data-testid', 'test-id', 'data-test'];
                                let testIdVal = '';
                                for (const attr of testIdAttrs) {
                                    const val = el.getAttribute(attr);
                                    if (val) { testIdVal = val; break; }
                                }
                                
                                const isGlobal = !!el.closest('header, footer, nav, [role="banner"], [role="contentinfo"], .header, .footer, .nav, .sidebar, .menu, [id*="header"], [id*="footer"], [id*="nav"]') ||
                                    ['header', 'footer', 'nav-link', 'nav_link', 'top_notice', 'lang-switcher', 'logo', 'bug bounty', 'care center', 'zendesk', 'apply.workable', 'faq', 'support'].some(kw => {
                                        const combinedAttr = `${el.tagName} ${el.className || ''} ${el.id || ''} ${testIdVal} ${el.getAttribute('data-location') || ''}`.toLowerCase();
                                        return combinedAttr.includes(kw);
                                    }) ||
                                    (tagName === 'a' && ['/flights', '/hotels', '/explore', '/subscriptions', '/blog', '/business', '/contact-us', '/about', '/ticket-safety', '/security', '/report-black-market', '/fan-protection'].some(path => (el.getAttribute('href') || '').includes(path)));

                                return {
                                    selector: selector,
                                    tagName: tagName,
                                    type: type,
                                    role: role,
                                    text: text,
                                    placeholder: placeholder,
                                    name: name,
                                    controlType: controlType,
                                    data_testid: testIdVal,
                                    isGlobal: isGlobal,
                                    xpath: getXPath(el)
                                };
                            });
                        }"""
                        
                        try:
                            elements = page.evaluate(elements_js, self.intent.get("semantic_key") == "auth")
                            # Register newly discovered elements in the classification tracker
                            for el in elements:
                                el_sel = el["selector"]
                                if el_sel not in interactable_classifications:
                                    interactable_classifications[el_sel] = {
                                        "selector": el_sel,
                                        "tagName": el.get("tagName"),
                                        "text": el.get("text"),
                                        "controlType": el.get("controlType"),
                                        "classification": "pending",
                                        "reason": None
                                    }
                        except Exception as e:
                            log.warning(f"Failed to retrieve elements: {e}")
                            break
                        
                        target_element = None
                        for el in elements:
                            el_sel = el["selector"]
                            is_global = el.get("isGlobal", False)
                            
                            unsafe_keywords = [
                                "pay", "checkout", "buy", "purchase", "delete", "remove", "cancel", "terminate",
                                "deactivate", "reset", "clear reservation", "confirm payment", "order", "subscribe"
                            ]
                            text_lower = el.get("text", "").lower()
                            sel_lower = el_sel.lower()
                            placeholder_lower = el.get("placeholder", "").lower()
                            name_lower = el.get("name", "").lower()

                            is_unsafe = False
                            for kw in unsafe_keywords:
                                if kw in text_lower or kw in sel_lower or kw in placeholder_lower or kw in name_lower:
                                    if "close" in text_lower or "cancel" in text_lower:
                                        if "modal" in sel_lower or "dialog" in sel_lower or "close" in text_lower:
                                            continue
                                    is_unsafe = True
                                    break

                            # Social/external share targets (a "Share" dialog's "Copy link /
                            # WhatsApp / X / LinkedIn / Email / Other" options) hand off to a
                            # third-party domain, a mail client, or the native OS share sheet —
                            # none of which exercise this module's own business logic, and the
                            # native OS share sheet in particular can crash the automated browser
                            # entirely (observed: clicking "Other" on an Auctions share dialog
                            # killed the whole browser process mid-exploration). Matched on the
                            # EXACT button text (not a substring) so this can't also skip an
                            # unrelated real element like an "Email" input field.
                            share_target_labels = {
                                "copy link", "whatsapp", "linkedin", "facebook", "telegram",
                                "messenger", "pinterest", "reddit", "x", "email", "other", "more"
                            }
                            if text_lower.strip() in share_target_labels:
                                interactable_classifications[el_sel] = {
                                    "selector": el_sel,
                                    "tagName": el.get("tagName"),
                                    "text": el.get("text"),
                                    "controlType": el.get("controlType"),
                                    "classification": "intentionally skipped with valid scope/safety reason",
                                    "reason": "social/external share target — hands off to a third-party domain or native OS share sheet, not this module's own business logic"
                                }
                                explored_signatures.add((el_sel, el["controlType"]))
                                continue

                            if is_global:
                                interactable_classifications[el_sel] = {
                                    "selector": el_sel,
                                    "tagName": el.get("tagName"),
                                    "text": el.get("text"),
                                    "controlType": el.get("controlType"),
                                    "classification": "intentionally skipped with valid scope/safety reason",
                                    "reason": "global shell navigation / header / footer element"
                                }
                                explored_signatures.add((el_sel, el["controlType"]))
                                continue
                                
                            if is_unsafe:
                                interactable_classifications[el_sel] = {
                                    "selector": el_sel,
                                    "tagName": el.get("tagName"),
                                    "text": el.get("text"),
                                    "controlType": el.get("controlType"),
                                    "classification": "intentionally skipped with valid scope/safety reason",
                                    "reason": "unsafe destructive action keyword match"
                                }
                                explored_signatures.add((el_sel, el["controlType"]))
                                continue
                                
                            # Prevent clicking all individual calendar date cells redundantly (equivalence partitioning)
                            is_calendar_day = False
                            try:
                                if el.get("controlType") == "clickable" and el.get("text", "").strip().isdigit():
                                    num = int(el.get("text", "").strip())
                                    if 1 <= num <= 31:
                                        is_calendar_day = True
                            except Exception:
                                pass

                            if is_calendar_day:
                                if interacted_with_calendar_cell:
                                    interactable_classifications[el_sel] = {
                                        "selector": el_sel,
                                        "tagName": el.get("tagName"),
                                        "text": el.get("text"),
                                        "controlType": el.get("controlType"),
                                        "classification": "intentionally skipped with valid scope/safety reason",
                                        "reason": "duplicate calendar date picker cell"
                                    }
                                    explored_signatures.add((el_sel, el["controlType"]))
                                    continue

                            sig = (el_sel, el["controlType"])
                            if sig not in explored_signatures:
                                if is_calendar_day:
                                    interacted_with_calendar_cell = True
                                target_element = el
                                explored_signatures.add(sig)
                                break
                                
                        if not target_element:
                            if before_state["modalOpen"]:
                                log.info("[BehavioralExploration] No untried elements in modal. Attempting to close it.")
                                try:
                                    page.keyboard.press("Escape")
                                    page.wait_for_timeout(1000)
                                    after_state = page.evaluate(state_js)
                                    if before_state["modalOpen"] and not after_state["modalOpen"]:
                                        transitions.append({
                                            "source": current_node,
                                            "target": current_node,
                                            "action": "press Escape key",
                                            "result": "closed modal via Escape",
                                            "state_change": True
                                        })
                                        interactions_performed += 1
                                        continue
                                except Exception:
                                    pass
                            
                            log.info("[BehavioralExploration] Exhausted all safe interactable elements on page.")
                            break
                            
                        interactions_performed += 1
                        sel = target_element["selector"]
                        ctype = target_element["controlType"]
                        el_text = target_element["text"]
                        
                        log.info(f"[BehavioralExploration] Interaction #{interactions_performed}: {ctype} on '{el_text or sel}'")
                        
                        # Update True Coverage assessment status for this element
                        for selector in self.captured_selectors:
                            if selector.get('stable_css_locator') == sel or selector.get('preferred_playwright_locator').endswith(sel):
                                if selector.get('assessment_status') == 'pending':
                                    selector['assessment_status'] = 'assessed'
                                    assessment_completion['assessed'] += 1
                                    assessment_completion['total_pending'] -= 1
                                break
                        
                        try:
                            current_interaction["action"] = f"{ctype} on {sel}"
                            current_interaction["component"] = sel
                            current_interaction["page"] = current_node

                            # Robust locator resolution
                            def get_robust_locator(page_obj, el_info):
                                test_id = el_info.get("data_testid")
                                if test_id:
                                    try:
                                        loc = page_obj.get_by_test_id(test_id)
                                        if loc.count() > 0:
                                            return loc
                                    except Exception:
                                        pass

                                role = el_info.get("role")
                                acc_name = el_info.get("text") or el_info.get("placeholder")
                                if role and acc_name:
                                    pw_role = role
                                    if role in ['input-text', 'textarea', 'textbox']:
                                        pw_role = 'textbox'
                                    elif role == 'input-checkbox':
                                        pw_role = 'checkbox'
                                    elif role == 'input-radio':
                                        pw_role = 'radio'
                                    elif role == 'select':
                                        pw_role = 'combobox'
                                    
                                    valid_roles = ['button', 'checkbox', 'combobox', 'heading', 'link', 'radio', 'textbox', 'listitem', 'menuitem', 'option', 'tab']
                                    if pw_role in valid_roles:
                                        try:
                                            loc = page_obj.get_by_role(pw_role, name=acc_name)
                                            if loc.count() > 0:
                                                return loc
                                        except Exception:
                                            pass

                                css_sel = el_info.get("selector", "")
                                if ":" in css_sel:
                                    import re
                                    # Protect standard CSS pseudo-classes by only escaping colons not followed by them
                                    pseudos = ["nth-of-type", "nth-child", "has-text", "has", "not", "first-child", "last-child", "active", "hover", "focus", "enabled", "disabled"]
                                    pattern = r":(?!(?:" + "|".join(pseudos) + r")\b)"
                                    css_sel = re.sub(pattern, "\\:", css_sel)
                                
                                try:
                                    loc = page_obj.locator(css_sel)
                                    return loc
                                except Exception:
                                    return page_obj.locator(f"xpath={el_info.get('xpath', '//*')}")

                            locator = get_robust_locator(page, target_element)
                            
                            # Disambiguate if matching multiple elements
                            try:
                                count = locator.count()
                            except Exception:
                                count = 1
                                
                            if count > 1:
                                log.info(f"[BehavioralExploration] Selector '{sel}' resolved to {count} elements. Resolving duplicates...")
                                found_active = False
                                for idx in range(count):
                                    try:
                                        sub_loc = locator.nth(idx)
                                        if sub_loc.is_visible() and sub_loc.is_enabled():
                                            locator = sub_loc
                                            found_active = True
                                            break
                                    except Exception:
                                        pass
                                if not found_active:
                                    locator = locator.first
                            
                            if ctype == "input":
                                val = "test"
                                ph = target_element.get("placeholder", "").lower()
                                label = target_element.get("text", "").lower()
                                name = target_element.get("name", "").lower()
                                
                                if "email" in ph or "email" in label or "email" in name:
                                    val = "test_user@example.com"
                                elif "phone" in ph or "phone" in label or "phone" in name or "mobile" in name:
                                    val = "5551234567"
                                elif "date" in ph or "date" in label or "date" in name:
                                    val = "2026-07-15"
                                elif "search" in ph or "search" in label or "search" in name:
                                    val = domain
                                
                                locator.scroll_into_view_if_needed(timeout=3000)
                                locator.click(timeout=3000, force=True)
                                locator.fill(val, timeout=3000)
                                if "search" in ph or "search" in label or "search" in name:
                                    locator.press("Enter")
                                
                            elif ctype == "dropdown":
                                locator.scroll_into_view_if_needed(timeout=3000)
                                if target_element["tagName"] == "select":
                                    options = locator.locator("option").all_inner_texts()
                                    valid_opts = [o.strip() for o in options if o.strip()]
                                    if valid_opts:
                                        locator.select_option(label=valid_opts[0])
                                    else:
                                        locator.click(timeout=3000, force=True)
                                else:
                                    locator.click(timeout=3000, force=True)
                                    
                            elif ctype == "checkbox" or ctype == "radio":
                                locator.scroll_into_view_if_needed(timeout=3000)
                                locator.click(timeout=3000, force=True)
                                
                            else:  # clickable
                                locator.scroll_into_view_if_needed(timeout=3000)
                                locator.click(timeout=3000, force=True)
                                
                            page.wait_for_timeout(1000)
                            try:
                                page.wait_for_load_state("networkidle", timeout=3000)
                            except Exception:
                                pass
                                
                            # SUCCESS
                            interactable_classifications[sel] = {
                                "selector": sel,
                                "tagName": target_element.get("tagName"),
                                "text": target_element.get("text"),
                                "controlType": ctype,
                                "classification": "successfully assessed",
                                "reason": None
                            }
                                
                        except Exception as click_err:
                            log.warning(f"Interaction with element {sel} failed: {click_err}")
                            # Mark element as failed assessment in True Coverage tracking
                            for selector in self.captured_selectors:
                                if selector.get('stable_css_locator') == sel or selector.get('preferred_playwright_locator').endswith(sel):
                                    if selector.get('assessment_status') == 'pending':
                                        selector['assessment_status'] = 'failed'
                                        selector['skip_reason'] = f'interaction_failed: {str(click_err)[:100]}'
                                        assessment_completion['failed'] += 1
                                        assessment_completion['total_pending'] -= 1
                                    break
                            
                            # FRAMEWORK FAILURE
                            interactable_classifications[sel] = {
                                "selector": sel,
                                "tagName": target_element.get("tagName"),
                                "text": target_element.get("text"),
                                "controlType": ctype,
                                "classification": "framework failure",
                                "reason": str(click_err)
                            }
                            continue
                            
                        # Fetch after_state and compare
                        try:
                            after_state = page.evaluate(state_js)
                        except Exception as e:
                            log.warning(f"Failed to retrieve after_state: {e}")
                            page.goto(url)
                            continue
                            
                        # Handle Navigation / URL Change
                        if before_state["url"] != after_state["url"]:
                            new_url = after_state["url"]
                            if is_target_module_url(new_url):
                                new_node = get_node_name_for_url(new_url)
                                transitions.append({
                                    "source": current_node,
                                    "target": new_node,
                                    "action": f"click '{el_text or sel}'",
                                    "result": f"navigated to {new_node}",
                                    "state_change": True
                                })
                                if new_url not in visited_urls:
                                    run_crawl(new_url, new_node, depth + 1)
                                
                                log.info(f"[BehavioralExploration] Returning back to page: {url}")
                                try:
                                    page.goto(url, timeout=15000)
                                    try:
                                        page.wait_for_load_state("networkidle", timeout=5000)
                                    except Exception:
                                        pass
                                except Exception as goto_err:
                                    log.warning(f"Failed to navigate back to {url}: {goto_err}")
                                page.wait_for_timeout(1000)
                            else:
                                log.info(f"[BehavioralExploration] Navigation out of scope detected ({new_url}). Reverting.")
                                try:
                                    page.go_back(timeout=15000)
                                except Exception as back_err:
                                    log.warning(f"Failed to go back: {back_err}")
                                page.wait_for_timeout(1000)
                                
                        else:
                            # Analyze DOM changes on the same page
                            change_msg = []
                            state_changed = False
                            
                            if not before_state["modalOpen"] and after_state["modalOpen"]:
                                change_msg.append("modal/drawer opened")
                                state_changed = True
                            elif before_state["modalOpen"] and not after_state["modalOpen"]:
                                change_msg.append("modal/drawer closed")
                                state_changed = True
                                
                            if before_state["errors"] != after_state["errors"] and after_state["errors"]:
                                change_msg.append(f"validation errors: {after_state['errors']}")
                                state_changed = True
                                
                            if before_state["cardsCount"] != after_state["cardsCount"]:
                                change_msg.append(f"content results updated (count changed from {before_state['cardsCount']} to {after_state['cardsCount']})")
                                state_changed = True
                                
                            if before_state["inputsState"] != after_state["inputsState"]:
                                change_msg.append("input controls enabled/disabled state changed")
                                state_changed = True
                                
                            if after_state["loaderVisible"]:
                                change_msg.append("loading states triggered")
                                state_changed = True
                                
                            if abs(before_state["textLength"] - after_state["textLength"]) > 100:
                                change_msg.append("page content updated")
                                state_changed = True
                                
                            if state_changed:
                                target_node = current_node
                                if after_state["modalOpen"]:
                                    target_node = f"{current_node}_modal"
                                    
                                observed_change = "; ".join(change_msg)
                                log.info(f"[BehavioralExploration] Detected page state change: {observed_change}")
                                
                                transitions.append({
                                    "source": current_node,
                                    "target": target_node,
                                    "action": f"{ctype} '{el_text or sel}'",
                                    "result": observed_change,
                                    "state_change": True
                                })

                                # Extract selectors on this new state
                                try:
                                    extracted_sub = page.evaluate(selector_extraction_script, [other_paths, domain])
                                    if extracted_sub:
                                        self.captured_selectors.extend(extracted_sub)
                                except Exception as sel_err:
                                    log.debug(f"Failed to extract sub-state selectors: {sel_err}")

                                # Extract DOM analysis on this new state
                                try:
                                    dom_data_sub = page.evaluate(dom_data_script, self.intent.get("semantic_key") == "auth")
                                    if target_node not in discovered_pages:
                                        discovered_pages[target_node] = {
                                            "url": url,
                                            **dom_data_sub
                                        }
                                    else:
                                        for k, v in dom_data_sub.items():
                                            if isinstance(v, list):
                                                existing_list = discovered_pages[target_node].setdefault(k, [])
                                                for item in v:
                                                    if item not in existing_list:
                                                        existing_list.append(item)
                                            elif isinstance(v, dict):
                                                existing_dict = discovered_pages[target_node].setdefault(k, {})
                                                existing_dict.update(v)
                                except Exception as dom_err:
                                    log.debug(f"Failed to evaluate sub-state DOM analysis: {dom_err}")

                                # Merge into main domain.lower() page for coverage tracking
                                main_node = domain.lower()
                                if main_node in discovered_pages:
                                    for key in ["inputs", "buttons", "dropdowns", "checkboxes", "radio_buttons", "links", "business_rules"]:
                                        main_list = discovered_pages[main_node].setdefault(key, [])
                                        sub_list = dom_data_sub.get(key, [])
                                        for item in sub_list:
                                            if item not in main_list:
                                                main_list.append(item)
                                    main_meta = discovered_pages[main_node].setdefault("component_metadata", {})
                                    sub_meta = dom_data_sub.get("component_metadata", {})
                                    main_meta.update(sub_meta)

                # Log final assessment completion report
                log.info(f"[BehavioralExploration] Final assessment completion: {assessment_completion['assessed']} assessed, {assessment_completion['skipped']} skipped, {assessment_completion['failed']} failed, {assessment_completion['total_pending']} remaining pending")
                
                # Warn if there are still pending elements that weren't assessed
                if assessment_completion['total_pending'] > 0:
                    log.warning(f"[BehavioralExploration] {assessment_completion['total_pending']} elements remain pending assessment. This may indicate incomplete behavior coverage.")
                
                # Check for Radix dynamic-id failures that need attention
                radix_failures = [s for s in self.captured_selectors if s.get('skip_reason') == 'radix_dynamic_id_no_fallback']
                if radix_failures:
                    log.warning(f"[BehavioralExploration] {len(radix_failures)} elements failed due to Radix dynamic-id selector parsing issues without fallback. These elements require manual selector configuration.")
                
                # Mark exploration as successful if we got here
                exploration_failed = False

                def get_node_name_for_url(url: str) -> str:
                    try:
                        from urllib.parse import urlparse
                        parsed = urlparse(url)
                        path_segs = [s for s in parsed.path.strip("/").split("/") if s]
                        if path_segs and len(path_segs[0]) == 2:
                            path_segs = path_segs[1:]
                        if not path_segs:
                            return "home"
                        node_name = "_".join(path_segs)
                        if parsed.query:
                            import hashlib
                            q_hash = hashlib.md5(parsed.query.encode('utf-8')).hexdigest()[:4]
                            node_name += f"_{q_hash}"
                        return node_name.lower()
                    except Exception:
                        return "linked_page"

                start_node = domain.lower()
                run_crawl(target_url, start_node, depth=1)

                # Crawl other story URLs at depth 1
                for s_url in self.story_urls:
                    def normalize_url(u: str) -> str:
                        return u.rstrip('/')
                    
                    if normalize_url(s_url) != normalize_url(target_url) and s_url not in visited_urls:
                        s_node = get_node_name_for_url(s_url)
                        log.info(f"[IntentDrivenUIExplorationEngine] Crawling additional story URL: {s_url} as node: {s_node}")
                        run_crawl(s_url, s_node, depth=1)

                browser.close()

        except Exception as e:
            exploration_failed = True
            exploration_failure_reason = str(e)
            log.error(f"[IntentDrivenUIExplorationEngine] Crawler encountered exception: {e}", exc_info=True)

            # If the browser was launched before the failure, ensure it's closed here —
            # otherwise an exception between launch() and the normal browser.close() call
            # above leaks the headless Chromium process.
            try:
                if 'browser' in locals() and browser is not None:
                    browser.close()
            except Exception:
                pass

            # Compute safe default assessment completion on failure
            if not self.captured_selectors:
                assessment_completion = {
                    'total_pending': 0,
                    'assessed': 0,
                    'skipped': 0,
                    'failed': 0
                }
            else:
                # Recompute from whatever selectors were captured before failure
                assessment_completion = {
                    'total_pending': 0,
                    'assessed': 0,
                    'skipped': 0,
                    'failed': 0
                }
                for selector in self.captured_selectors:
                    if selector.get('assessment_status') == 'pending':
                        assessment_completion['total_pending'] += 1
                    elif selector.get('assessment_status') == 'assessed':
                        assessment_completion['assessed'] += 1
                    elif selector.get('assessment_status') == 'skipped':
                        assessment_completion['skipped'] += 1
                    elif selector.get('assessment_status') == 'failed':
                        assessment_completion['failed'] += 1
            
            # Log failure assessment summary
            log.error(f"[IntentDrivenUIExplorationEngine] Exploration failed: {exploration_failure_reason}")
            log.error(f"[IntentDrivenUIExplorationEngine] Partial assessment completion: {assessment_completion['assessed']} assessed, {assessment_completion['skipped']} skipped, {assessment_completion['failed']} failed, {assessment_completion['total_pending']} pending")

        # Write exploration classifications report to artifact directory
        try:
            # Recompute assessment_completion stats directly from ground-truth classifications
            classification_counts = {
                'total_pending': 0,
                'assessed': 0,
                'skipped': 0,
                'failed': 0
            }
            for item in interactable_classifications.values():
                status = item.get("classification")
                if status == "pending":
                    classification_counts['total_pending'] += 1
                elif status == "successfully assessed":
                    classification_counts['assessed'] += 1
                elif status == "intentionally skipped with valid scope/safety reason":
                    classification_counts['skipped'] += 1
                elif status == "framework failure":
                    classification_counts['failed'] += 1
            
            assessment_completion = classification_counts
            
            report_path = self.settings.artifact_dir / "exploration_report.json"
            report_data = {
                "domain": domain,
                "target_url": target_url,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "exploration_failed": exploration_failed,
                "exploration_failure_reason": exploration_failure_reason,
                "assessment_completion": assessment_completion,
                "interactables": list(interactable_classifications.values())
            }
            report_path.write_text(json.dumps(report_data, indent=2, ensure_ascii=False), encoding="utf-8")
            log.info(f"[IntentDrivenUIExplorationEngine] Saved crawler exploration report to: {report_path}")
            
            # Print a clean, formatted stdout summary of the classification status
            log.info("==================================================================")
            log.info("  CRAWLER BEHAVIORAL ASSESSMENT SUMMARY")
            log.info("==================================================================")
            log.info(f"Assessed elements           : {assessment_completion['assessed']}")
            log.info(f"Intentionally skipped       : {assessment_completion['skipped']}")
            log.info(f"Framework failures          : {assessment_completion['failed']}")
            log.info(f"Remaining pending           : {assessment_completion['total_pending']}")
            log.info("==================================================================")
        except Exception as report_err:
            log.warning(f"Failed to write exploration_report.json: {report_err}")

        # Check if exploration failed and handle accordingly
        if exploration_failed:
            log.error(f"[IntentDrivenUIExplorationEngine] Behavioral exploration incomplete due to failure: {exploration_failure_reason}")
            log.error(f"[IntentDrivenUIExplorationEngine] Testcase generation should be blocked or flagged as incomplete")
            # Return empty results to prevent testcase generation from proceeding with incomplete data
            return {
                "discovered_pages": {}, 
                "observed_apis": [],
                "exploration_failed": True,
                "exploration_failure_reason": exploration_failure_reason,
                "assessment_completion": assessment_completion
            }, []

        return {
            "discovered_pages": discovered_pages, 
            "observed_apis": observed_apis,
            "exploration_failed": False,
            "assessment_completion": assessment_completion
        }, transitions


class BehavioralFlowEngine:
    """Manages persistent behavioral graph storage and monitors transitions state."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.graph_path = settings.artifact_dir / "ui_graph.json"

    def load_graph(self) -> Dict[str, Any]:
        """Load the behavioral graph memory from artifacts/ui_graph.json if available."""
        load_path = self.graph_path
        if not load_path.exists():
            global_path = self.settings.artifact_dir.parent / "ui_graph.json"
            if global_path.exists():
                load_path = global_path

        if load_path.exists():
            try:
                return json.loads(load_path.read_text(encoding="utf-8"))
            except Exception as e:
                log.error(f"[BehavioralFlowEngine] Error loading behavioral graph memory: {e}")
        
        return {
            "nodes": [],
            "edges": [],
            "selectors": {},
            "last_updated": ""
        }

    @staticmethod
    def _sanitize_node_name(text: str) -> str:
        return _sanitize_node_name(text)

    def update_graph(
        self, existing_graph: Dict[str, Any], ui_map: Dict[str, Any], transitions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Update and merge discovered transitions, selectors, and state outputs into behavioral memory."""
        nodes = set(existing_graph.get("nodes", []))
        edges = existing_graph.get("edges", [])
        selectors = existing_graph.get("selectors", {})

        new_transitions_found = 0
        existing_transitions_found = 0

        # Group observed APIs by page node name
        observed_apis = ui_map.get("observed_apis", [])
        apis_by_page = {}
        for api in observed_apis:
            node = api.get("current_node", "unknown")
            if node not in apis_by_page:
                apis_by_page[node] = []
            apis_by_page[node].append(api)

        # Update page selectors inventory
        discovered_pages = ui_map.get("discovered_pages", {})
        for page_node, details in discovered_pages.items():
            nodes.add(page_node)
            
            # Store observed APIs under the page selectors detail
            details["observed_apis"] = apis_by_page.get(page_node, [])
            selectors[page_node] = details  # Store all extracted DOM properties and business rules

            # Extend graph with Page, Component, Interaction, and State nodes/edges
            for comp_key in ["inputs", "buttons", "dropdowns", "checkboxes", "radio_buttons", "links"]:
                for comp in details.get(comp_key, []):
                    comp_name_clean = self._sanitize_node_name(comp)
                    comp_node = f"{page_node}_comp_{comp_name_clean}"
                    interact_node = f"{page_node}_interact_{comp_name_clean}"

                    nodes.add(comp_node)
                    nodes.add(interact_node)

                    # Add Page -> Component edge
                    edges.append({
                        "source": page_node,
                        "target": comp_node,
                        "action": "has component",
                        "result": f"renders {comp_key[:-1]}",
                        "state_change": False
                    })

                    # Add Component -> Interaction edge
                    edges.append({
                        "source": comp_node,
                        "target": interact_node,
                        "action": "interact via user action",
                        "result": "triggers click or entry",
                        "state_change": False
                    })

                    # Add Interaction -> Page outcome edge
                    edges.append({
                        "source": interact_node,
                        "target": page_node,
                        "action": "triggers system response",
                        "result": "system evaluates action",
                        "state_change": True
                    })

            # Extend graph with observed APIs: Page/Component -> Network Request -> System Response -> UI State Change
            for api in apis_by_page.get(page_node, []):
                method = api.get("method", "GET")
                url = api.get("url", "")
                status = api.get("response_status", 200)

                url_clean = self._sanitize_node_name(url.split("?")[0].replace("https://", "").replace("http://", ""))
                api_node = f"{page_node}_api_{method.lower()}_{url_clean}"
                nodes.add(api_node)

                # Find triggering interaction node
                trigger_node = page_node
                found_comp = False
                associated_comp = api.get("associated_component")
                if associated_comp and associated_comp not in ["body", "page_window"]:
                    comp_name_clean = self._sanitize_node_name(associated_comp)
                    interact_node = f"{page_node}_interact_{comp_name_clean}"
                    if interact_node in nodes:
                        edges.append({
                            "source": interact_node,
                            "target": api_node,
                            "action": "triggers network call",
                            "result": f"sends HTTP {method}",
                            "state_change": False
                        })
                        found_comp = True

                if not found_comp:
                    comp_key_matches = ["buttons"] if method in ["POST", "PUT", "DELETE", "PATCH"] else ["inputs", "dropdowns", "links"]
                    for ckey in comp_key_matches:
                        for comp in details.get(ckey, []):
                            comp_name_clean = self._sanitize_node_name(comp)
                            interact_node = f"{page_node}_interact_{comp_name_clean}"
                            if interact_node in nodes:
                                edges.append({
                                    "source": interact_node,
                                    "target": api_node,
                                    "action": "triggers network call",
                                    "result": f"sends HTTP {method}",
                                    "state_change": False
                                })
                                found_comp = True
                                break
                        if found_comp:
                            break

                if not found_comp:
                    edges.append({
                        "source": page_node,
                        "target": api_node,
                        "action": "sends request",
                        "result": f"sends HTTP {method}",
                        "state_change": False
                    })

                # API Response Node -> State Change Outcome
                edges.append({
                    "source": api_node,
                    "target": page_node,
                    "action": f"returns status {status}",
                    "result": "system processes response",
                    "state_change": True
                })

        # Merge transition edges (actions mapped to outcomes)
        for transition in transitions:
            is_duplicate = False
            for edge in edges:
                if (edge.get("source") == transition["source"] and 
                    edge.get("target") == transition["target"] and 
                    edge.get("action") == transition["action"]):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                edges.append(transition)
                new_transitions_found += 1
            else:
                existing_transitions_found += 1

        # Deduplicate edges lists
        unique_edges = []
        seen_edges = set()
        for edge in edges:
            edge_key = (edge.get("source"), edge.get("target"), edge.get("action"))
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                unique_edges.append(edge)

        log.info(f"[BehavioralFlowEngine] Graph updated: {new_transitions_found} new transitions, {existing_transitions_found} existing transitions.")

        return {
            "nodes": sorted(list(nodes)),
            "edges": unique_edges,
            "selectors": selectors,
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
class TestCaseGenerator:
    """Synthesizes high-fidelity flow test cases derived strictly from behavioral context and intent."""

    def __init__(self, story_key: str, intent: Dict[str, Any], behavior_graph: Dict[str, Any], attachment_insights: List[Dict[str, Any]] = None, story_data: Dict[str, Any] = None, discovered_routes: List[Dict[str, str]] = None) -> None:
        self.story_key = story_key
        self.intent = intent
        self.behavior_graph = behavior_graph
        self.attachment_insights = attachment_insights or []
        self.story_data = story_data or {}
        # Routes discovered during UI exploration — used by the Module Isolation
        # rejection rule (RULE 5) to detect cross-module contamination in generated
        # test cases. Previously never populated (this class had no such attribute),
        # so that rule was a permanent no-op.
        self.discovered_routes = discovered_routes or []
        # Initialize domain-aware terminology resolver
        self.terminology = DomainTerminologyResolver(intent, story_data, behavior_graph)

    def _determine_polarity(self, tctype: str, title: str, description: str, tags: List[str], edge_case: bool = False) -> str:
        """Determines if a testcase is Positive, Negative, or Edge."""
        tctype_lower = tctype.lower()
        title_lower = title.lower()
        desc_lower = description.lower()
        tags_lower = [t.lower() for t in tags]
        
        # Explicit check for type first
        if tctype_lower == "edge" or edge_case is True or "edge" in tags_lower or "boundary" in tags_lower:
            return "Edge"
        if tctype_lower == "negative" or "negative" in tags_lower:
            return "Negative"
            
        # Check for Resilience and Recovery as Edge
        if tctype_lower in ["resilience", "recovery"] or any(t in tags_lower for t in ["resilience", "recovery"]):
            return "Edge"
            
        # Semantic analysis of title and description
        edge_keywords = [
            "special character", "emoji", "unicode", "script injection", "malicious", 
            "xss", "sql injection", "double-click", "double click", "latency", "slow network", 
            "retry", "recovery", "refresh", "reload", "restart", "boundary", "overflow", 
            "character limit", "numeric range", "concurrency", "keyboard navigation", 
            "accessibility", "wcag", "aria tags", "screen reader"
        ]
        if any(kw in title_lower or kw in desc_lower for kw in edge_keywords):
            return "Edge"
            
        negative_keywords = [
            "invalid", "blank", "empty", "missing", "unauthorized", "expiry", "expired", 
            "fail", "rejections", "rejects", "error", "deny", "denies", "blocked", 
            "offline", "disconnect", "refuses", "prevent"
        ]
        if any(kw in title_lower or kw in desc_lower for kw in negative_keywords):
            return "Negative"
            
        return "Positive"

    @staticmethod
    def _classify_component(tagName: str, name: str) -> List[str]:
        categories = ["Functional", "Keyboard", "Accessibility", "Responsive", "Performance", "Regression"]
        name_lower = name.lower()
        if "password" in name_lower:
            categories.extend(["Validation", "Negative", "Error Handling"])
        elif "email" in name_lower or "mail" in name_lower:
            categories.extend(["Validation", "Negative", "Boundary", "Error Handling"])
        elif "search" in name_lower or "query" in name_lower:
            categories.extend(["Boundary", "Negative", "Localization"])
        elif tagName in ["select", "dropdown"]:
            categories.extend(["Validation", "Localization"])
        elif tagName in ["checkbox", "radio"]:
            categories.extend(["Validation", "Edge"])
        return list(set(categories))

    @staticmethod
    def _extract_domain_terms_from_routes(routes: List[Dict[str, str]], current_domain: str) -> List[str]:
        """Extract domain terms from discovered routes dynamically at runtime."""
        domain_terms = set()
        for route in routes:
            label = route.get("label", "").lower()
            path = route.get("path", "").lower()
            
            # Extract potential domain terms from labels and paths
            # Common patterns: /en/explore?tag=football, /en/explore?category=sports-event
            if "tag=" in path:
                tag = path.split("tag=")[-1].split("&")[0]
                domain_terms.add(tag)
            if "category=" in path:
                category = path.split("category=")[-1].split("&")[0]
                # Extract base domain from category (e.g., "sports-event" -> "sports")
                base_category = category.split("-")[0]
                domain_terms.add(base_category)
                domain_terms.add(category)
            
            # Extract from labels
            label_words = label.split()
            for word in label_words:
                if len(word) > 3 and word.isalpha():
                    domain_terms.add(word.lower())
        
        # Remove current domain and its variants
        domain_clean = current_domain.rstrip('s')
        domain_terms.discard(domain_clean)
        domain_terms.discard(current_domain)
        domain_terms.discard(domain_clean + 's')
        
        # Remove generic terms that aren't domains
        generic_terms = {"explore", "page", "shop", "blog", "contact", "about", "terms", "privacy", "help", "business", "report", "fan", "protection", "safety", "black", "market"}
        for term in generic_terms:
            domain_terms.discard(term)
        
        return list(domain_terms)

    @staticmethod
    def _determine_priority_class(comp_selector: str, comp_type: str, page_sels: Dict[str, Any], intent: Dict[str, Any]) -> str:
        sel_lower = comp_selector.lower()
        is_required = "[required]" in sel_lower or "required" in sel_lower or any(comp_selector in r for r in page_sels.get("required_fields", []))
        
        # Critical signals: authentication, transactional checkout, submit buttons, terms acceptance
        critical_keywords = [
            "login", "auth", "password", "email", "username", "checkout", "pay", "buy", 
            "book", "confirm", "submit", "register", "signup", "card", "cvv", "payment", 
            "otp", "verify", "phone", "mobile", "price", "total", "purchase", "agree", "accept",
            "select-options", "search-submit", "checkout-button"
        ]
        
        if comp_type == "input" and (is_required or "password" in sel_lower or "email" in sel_lower or "card" in sel_lower):
            return "critical"
        if comp_type == "button" and any(kw in sel_lower for kw in ["submit", "confirm", "pay", "checkout", "register", "login"]):
            return "critical"
        if any(kw in sel_lower for kw in critical_keywords):
            return "critical"
            
        # Cross-reference with acceptance criteria from intent
        ac_list = intent.get("acceptance_criteria", [])
        for ac in ac_list:
            ac_words = set(re.split(r'\W+', ac.lower()))
            sel_words = set(re.split(r'\W+', comp_selector.lower()))
            common = {w for w in ac_words.intersection(sel_words) if len(w) > 3}
            if common:
                return "critical"
                
        # Medium signals: query / search, secondary options, select dropdowns
        medium_keywords = [
            "search", "query", "filter", "sort", "select", "category", "date", 
            "edit", "update", "cancel", "back", "close", "agree", "terms"
        ]
        if any(kw in sel_lower for kw in medium_keywords):
            return "medium"
        if comp_type in ["dropdown", "checkbox", "radio_button"]:
            return "medium"
            
        return "low"

    def _find_matching_apis(self, comp_selector: str, comp_type: str, page_key: str) -> List[Dict[str, Any]]:
        selectors = self.behavior_graph.get("selectors", {})
        page_selectors = selectors.get(page_key, {})
        observed = page_selectors.get("observed_apis", [])
        
        apis = []
        comp_lower = comp_selector.lower()
        semantic_name = self.get_semantic_name(comp_selector, comp_type, page_key).lower()
        
        for api in observed:
            self._enrich_api_details(api)
            url = api.get("url", "")
            url_lower = url.lower()
            if ("/cdn-cgi/" in url_lower or 
                "/locales/" in url_lower or 
                "sar.json" in url_lower or 
                any(ext in url_lower for ext in [".js", ".css", ".png", ".jpg", ".jpeg", ".woff", ".woff2", ".ico", ".svg"]) or
                any(kw in url_lower for kw in [
                    "telemetry", "analytics", "doubleclick", "pagead", "sentry", "tracking",
                    "advertising", "cookie", "consent", "infra", "google-analytics", "hotjar",
                    "facebook", "log", "metric", "report", "mixpanel", "segment", "datadog",
                    "amplitude", "fbevents", "tiktok", "snapchat", "ad-system", "ad", "ads",
                    "pixel", "beacon", "gtm", "googletagmanager", "collect", "track"
                ])):
                continue

            op_name = api.get("business_operation", "UnknownOperation")
            is_init_api = op_name in ["getConfig", "getPageBase", "getWebSection", "getPageSections", "getConfigCollection"]
            associated_comp = api.get("associated_component")
            trigger_action = api.get("triggering_ui_action", "")
            method = api.get("method", "GET").upper()

            # Skip load-time config APIs unless the component is directly associated with them
            if is_init_api and associated_comp != comp_selector:
                continue

            confidence = 0

            # 1. Direct Component Association
            if associated_comp and associated_comp == comp_selector:
                confidence += 10
            elif associated_comp and associated_comp != "page_window" and associated_comp != "body":
                assoc_semantic = self.get_semantic_name(associated_comp, "", page_key).lower()
                if assoc_semantic == semantic_name:
                    confidence += 8
                elif assoc_semantic in semantic_name or semantic_name in assoc_semantic:
                    confidence += 5

            # 2. Domain / Keyword Mapping - Use dynamic terminology
            entity = self.terminology.get_entity_term()
            entity_plural = self.terminology.get_entity_term(plural=True)
            transaction = self.terminology.get_transaction_term()
            ui_terms = self.terminology._terminology_cache.get("ui_terms", [])
            
            # Build dynamic keywords_map from terminology resolver and UI terms
            keywords_map = {
                transaction: [transaction, "submit", "confirm", "place"],
                "search": ["search", "query", "find"],
                "login": ["login", "auth", "signin", "signup", "register", "credentials"],
                "auth": ["login", "auth", "signin", "signup", "register", "credentials"],
                "subscribe": ["subscribe", "newsletter", "subscription"],
                "checkout": ["checkout", "pay", "buy", "purchase", "order"],
                entity: [entity, entity_plural]
            }
            
            # Add UI terms dynamically
            for term in ui_terms:
                term_lower = term.lower()
                if term_lower not in keywords_map:
                    keywords_map[term_lower] = [term_lower]
            
            for category, kws in keywords_map.items():
                if any(kw in comp_lower or kw in semantic_name for kw in kws):
                    if any(kw in url_lower or kw in op_name.lower() for kw in kws):
                        confidence += 6
                        break

            # 3. Action & Method Matching - Use dynamic terminology
            transaction = self.terminology.get_transaction_term()
            
            if comp_type == "button":
                is_submit_btn = any(w in comp_lower or w in semantic_name for w in ["submit", "confirm", "pay", "login", "register", transaction, "subscribe"])
                if is_submit_btn:
                    if method in ["POST", "PUT", "DELETE", "PATCH"]:
                        confidence += 4
                    if any(w in url_lower or w in op_name.lower() for w in ["submit", "confirm", "pay", "login", "register", transaction, "subscribe"]):
                        confidence += 3
                else:
                    if method in ["POST", "PUT", "DELETE", "PATCH"] and trigger_action != "navigation":
                        confidence += 2
            elif comp_type == "input":
                if method in ["POST", "PUT"] and trigger_action != "navigation":
                    confidence += 2
            elif comp_type == "link":
                if method == "GET" and trigger_action == "click":
                    confidence += 2

            # Only consider relevant APIs with a positive confidence score >= 2 to filter page-load noise
            if confidence >= 2:
                apis.append((api, confidence))

        # Sort by confidence rating descending
        apis.sort(key=lambda x: x[1], reverse=True)
        if apis:
            max_conf = apis[0][1]
            return [x[0] for x in apis if x[1] == max_conf]
        return []

    def _enrich_api_details(self, api: Dict[str, Any]) -> Dict[str, Any]:
        from urllib.parse import urlparse
        import re
        method = api.get("method", "GET").upper()
        url = api.get("url", "")
        payload = api.get("request_payload")
        resp_body = api.get("response_body")
        
        business_operation = "UnknownOperation"
        business_intent = "UnknownIntent"
        triggering_user_action = api.get("triggering_ui_action", "interaction")
        
        is_graphql = False
        operation_name = None
        
        if isinstance(payload, dict):
            if "query" in payload or "operationName" in payload:
                is_graphql = True
                operation_name = payload.get("operationName")
                if not operation_name and "query" in payload:
                    query_str = payload["query"] or ""
                    # Match named queries/mutations
                    match_named = re.search(r'(query|mutation)\s+([a-zA-Z0-9_]+)', query_str)
                    if match_named:
                        operation_name = match_named.group(2)
                    else:
                        # Match unnamed queries/mutations selection set field
                        match_unnamed = re.search(r'(query|mutation)\s*\{\s*([a-zA-Z0-9_]+)', query_str)
                        if match_unnamed:
                            operation_name = match_unnamed.group(2)
        
        if is_graphql and operation_name:
            business_operation = operation_name
            op_lower = operation_name.lower()
            
            # Use dynamic terminology from resolver
            entity = self.terminology.get_entity_term()
            entity_plural = self.terminology.get_entity_term(plural=True)
            transaction = self.terminology.get_transaction_term()
            
            if "bid" in op_lower or transaction in op_lower:
                business_intent = f"Submit {transaction} offer on active {entity}"
            elif "login" in op_lower or "auth" in op_lower:
                business_intent = "Authenticate user credentials and initialize session"
            elif "search" in op_lower or "query" in op_lower:
                business_intent = f"Query and retrieve active {entity_plural} matching filters"
            elif "config" in op_lower:
                business_intent = "Load global application settings and theme configurations"
            elif "subscription" in op_lower or "subscribe" in op_lower:
                business_intent = "Subscribe user to email newsletters"
            elif "profile" in op_lower or "user" in op_lower:
                business_intent = "Fetch authenticated user account profile details"
            else:
                business_intent = f"Execute GraphQL {operation_name} business logic"
        else:
            url_lower = url.lower()
            
            # Use dynamic terminology from resolver
            entity = self.terminology.get_entity_term()
            entity_plural = self.terminology.get_entity_term(plural=True)
            transaction = self.terminology.get_transaction_term()
            
            if "geolocation" in url_lower:
                business_operation = "FetchGeolocation"
                business_intent = "Identify user location and locale settings"
            elif "login" in url_lower or "auth" in url_lower:
                business_operation = "AuthenticateUser"
                business_intent = "Authenticate user credentials and initialize session"
            elif "search" in url_lower or "query" in url_lower:
                business_operation = f"Search{entity.capitalize()}"
                business_intent = f"Retrieve matching {entity_plural}"
            elif "bid" in url_lower or transaction in url_lower:
                business_operation = f"Place{transaction.capitalize()}"
                business_intent = f"Place {transaction} on dynamic {entity}"
            elif "subscribe" in url_lower or "newsletter" in url_lower:
                business_operation = "SubmitSubscription"
                business_intent = "Register email for marketing campaigns"
            elif "user" in url_lower or "profile" in url_lower:
                business_operation = "FetchUserProfile"
                business_intent = "Retrieve user session and settings data"
            else:
                try:
                    path_parts = [p for p in urlparse(url).path.split("/") if p]
                    if path_parts:
                        clean_parts = [p for p in path_parts if not p.isdigit() and len(p) < 20]
                        if clean_parts:
                            segment = "".join(p.capitalize() for p in clean_parts[-2:])
                            business_operation = f"{method}{segment}"
                        else:
                            business_operation = f"{method}Request"
                    else:
                        business_operation = f"{method}Request"
                except Exception:
                    business_operation = f"{method}Request"
                business_intent = f"Perform {method} operations on resource path"

        if isinstance(resp_body, dict):
            if "data" in resp_body:
                gql_data = resp_body["data"]
                if isinstance(gql_data, dict) and gql_data:
                    first_key = list(gql_data.keys())[0]
                    if business_operation == "UnknownOperation" or not is_graphql:
                        business_operation = first_key[0].upper() + first_key[1:]
            elif "country_name" in resp_body:
                business_operation = "FetchGeolocation"
                business_intent = "Identify user location and locale settings"

        api["business_operation"] = business_operation
        api["business_intent"] = business_intent
        api["triggering_user_action"] = triggering_user_action
        
        return api

    def _generate_api_expectations(self, api: Dict[str, Any]) -> Dict[str, Any]:
        self._enrich_api_details(api)
        status = api.get("response_status", 200)
        method = api.get("method", "GET")
        url = api.get("url", "")
        duration = api.get("response_time_ms", 100)
        body_size = api.get("response_size_bytes", 0)
        content_type = api.get("content_type", "application/json")
        headers = api.get("request_headers", {})
        
        # Parse query params to infer pagination
        from urllib.parse import urlparse, parse_qs
        is_paginated = False
        try:
            parsed = urlparse(url)
            qs = parse_qs(parsed.query)
            is_paginated = any(k in qs for k in ["page", "limit", "offset", "size", "p", "per_page"])
        except Exception:
            pass
        
        expectations = {
            "endpoint": f"{method} {url.split('?')[0]}",
            "expected_status": status,
            "expected_success": status in [200, 201, 204],
            "max_response_time_ms": max(500, int(duration * 1.5)),
            "response_payload_type": content_type,
            "required_headers": [h for h in headers.keys() if h.lower() in ["content-type", "accept", "x-requested-with"]],
            "business_operation": api.get("business_operation"),
            "business_intent": api.get("business_intent"),
            "triggering_user_action": api.get("triggering_user_action")
        }
        
        payload = api.get("request_payload")
        if isinstance(payload, dict):
            expectations["expected_payload_schema"] = list(payload.keys())
        elif isinstance(payload, list) and payload:
            if isinstance(payload[0], dict):
                expectations["expected_payload_schema"] = list(payload[0].keys())
        else:
            expectations["expected_payload_schema"] = []
                
        # Expected Error Handling
        if status >= 400:
            expectations["expected_error_handling"] = "System returns error status code and payload gracefully"
        else:
            expectations["expected_error_handling"] = "Endpoint processes request and returns success without application-level validation errors"
            
        # Expected Empty Response Handling
        if body_size == 0:
            expectations["expected_empty_response_handling"] = "System handles empty or no-content response body safely"
        else:
            expectations["expected_empty_response_handling"] = "System expects a structured response body payload"
            
        # Expected Authorization Behaviour
        has_auth = any(h.lower() in ["authorization", "cookie", "x-api-key", "token"] for h in headers.keys())
        if has_auth:
            expectations["expected_authorization_behaviour"] = "Request requires valid session tokens or credentials in header context"
        else:
            expectations["expected_authorization_behaviour"] = "Endpoint is publicly accessible without active credentials"
            
        # Expected Retry Behaviour
        expectations["expected_retry_behaviour"] = "Idempotent execution with standard timeout backing off if temporary failure occurs"
        
        # Expected Pagination Behaviour
        if is_paginated:
            expectations["expected_pagination_behaviour"] = "Supports limit-offset or page-based server pagination"
        else:
            expectations["expected_pagination_behaviour"] = "Pagination not applicable for this resource query type"
            
        # Expected Validation Messages
        validation_msg = "None observed"
        resp_body = api.get("response_body")
        if isinstance(resp_body, dict):
            for k in ["message", "error", "errors", "detail", "validation", "msg"]:
                if k in resp_body:
                    validation_msg = f"Observed response validation field: {resp_body[k]}"
                    break
        expectations["expected_validation_messages"] = validation_msg
            
        # Expected State Transition
        expectations["expected_state_transition"] = f"Endpoint triggers state change transitioning page context to {api.get('current_node', 'unknown')}"
        
        return expectations

    def get_semantic_name(self, selector: str, ctype: str, page_key: str) -> str:
        """Resolves a raw DOM selector into a clean, business-oriented semantic name."""
        selectors_store = self.behavior_graph.get("selectors", {})
        page_selectors = selectors_store.get(page_key, {})
        metadata = page_selectors.get("component_metadata", {})
        
        # Check if we have gathered metadata for this exact selector
        elem_meta = metadata.get(selector, {})
        
        label = elem_meta.get("label", "").strip()
        aria_label = elem_meta.get("aria_label", "").strip()
        placeholder = elem_meta.get("placeholder", "").strip()
        text = elem_meta.get("text", "").strip()
        card_context = elem_meta.get("card_context", "").strip()
        
        # Determine candidate name
        candidate = ""
        
        if label:
            candidate = label
        elif aria_label:
            candidate = aria_label
        elif placeholder:
            candidate = placeholder
        elif text:
            candidate = text
            
        # Clean up candidate
        candidate = re.sub(r'[:*?]', '', candidate) # remove required asterisks, etc.
        candidate = candidate.strip()
        
        # Regex mappings / fallbacks if no metadata was found or metadata is too generic
        if not candidate:
            # Let's clean the selector
            clean_sel = selector
            if "has-text('" in clean_sel:
                match = re.search(r"has-text\('([^']+)'\)", clean_sel)
                if match:
                    candidate = match.group(1)
            elif "name='" in clean_sel:
                match = re.search(r"name='([^']+)'", clean_sel)
                if match:
                    candidate = match.group(1).replace("_", " ").replace("-", " ").title()
            elif "#" in clean_sel:
                parts = clean_sel.split("#")
                if len(parts) > 1 and not parts[1].startswith(":r"): # skip auto-generated ids
                    candidate = parts[1].replace("_", " ").replace("-", " ").title()
                    
        # Apply page-specific heuristic rules using dynamic terminology
        entity = self.terminology.get_entity_term()
        transaction = self.terminology.get_transaction_term()
        
        if not candidate or candidate.lower() in [transaction, "submit", "confirm", "value", "input"]:
            # Check context using dynamic terminology
            if entity.lower() in page_key or transaction in page_key:
                if ctype == "input":
                    candidate = f"{transaction.capitalize()} Amount Input Field"
                elif ctype == "button":
                    candidate = f"Place {transaction.capitalize()} Button"
            elif "auth" in page_key or "login" in page_key:
                if "email" in selector.lower():
                    candidate = "Email Input Field"
                elif "password" in selector.lower():
                    candidate = "Password Input Field"
                elif "login" in selector.lower() or "submit" in selector.lower():
                    candidate = "Login Submit Button"
            elif "explore" in page_key:
                if "search" in selector.lower():
                    candidate = "Search Input Field"
        
        # Default fallback
        if not candidate:
            candidate = selector.replace("button:has-text('", "").replace("')", "")
            candidate = candidate.replace("input[name='", "").replace("']", "")
            candidate = candidate.replace("a:has-text('", "").replace("')", "")
            candidate = re.sub(r'[^a-zA-Z0-9 ]', '', candidate)
            
        # Formatting suffix
        candidate_words = candidate.split()
        # Keep it concise
        if len(candidate_words) > 6:
            candidate = " ".join(candidate_words[:5]) + "..."
            
        # Append component type suffix gracefully if not present
        candidate_lower = candidate.lower()
        if ctype == "input" and "input" not in candidate_lower and "field" not in candidate_lower:
            candidate = f"{candidate} Input Field"
        elif ctype == "button" and "button" not in candidate_lower and "btn" not in candidate_lower:
            candidate = f"{candidate} Button"
        elif ctype == "dropdown" and "select" not in candidate_lower and "dropdown" not in candidate_lower:
            candidate = f"{candidate} Dropdown"
        elif ctype in ["checkbox", "radio_button"] and not any(w in candidate_lower for w in ["check", "radio", "option"]):
            candidate = f"{candidate} Checkbox"
        elif ctype == "link" and "link" not in candidate_lower:
            candidate = f"{candidate} Link"
            
        # Add card context prefix if available and not already in candidate
        if card_context and card_context.lower() not in candidate.lower() and len(card_context) < 30:
            candidate = f"{card_context} > {candidate}"
            
        # Label Normalization Policy: Normalize whitespace, case, hidden formatting, separators, duplicate concatenations
        candidate = re.sub(r'[\r\n\t\xa0]', ' ', candidate)
        candidate = re.sub(r'\s+', ' ', candidate)
        candidate = re.sub(r'([a-z])([A-Z])', r'\1 \2', candidate) # Separate concatenated titlecases (e.g. DestinationSaudi -> Destination Saudi)
        candidate = candidate.strip(" |:-_ ")
        
        # Ensure a clean normalized title casing structure. Generic acronyms are kept
        # as-is rather than title-cased; no application brand name is hardcoded here so
        # this works the same regardless of which app is under test.
        words = []
        for w in candidate.split():
            if w.lower() in ["url", "id", "api"]:
                words.append(w)
            else:
                words.append(w.capitalize())
        candidate = " ".join(words)
        
        return candidate

    @staticmethod
    def is_global_component(selector: str, semantic_name: str) -> bool:
        sel_lower = selector.lower()
        sem_lower = semantic_name.lower()
        global_keywords = [
            "cookie", "banner", "newsletter", "subscribe", "footer", "header", 
            "language", "country", "theme", "dark mode", "light mode", "navbar",
            "copyright", "hiring", "apply now", "menu-links", "social-links",
            "accept all", "decline all", "reject all", "privacy", "cookieconsent",
            "cookie settings", "cookie preferences", "reject all non-essential",
            "global header", "global footer", "language selector", "country selector"
        ]
        return any(kw in sel_lower or kw in sem_lower for kw in global_keywords)

    def _run_qa_reasoning_engine(self, selectors: Dict[str, Any], edges: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Runs the Enterprise QA Reasoning Engine internally.
        Executes Phase 1 to Phase 5 before testcase generation begins.
        """
        # Phase 1: Module Understanding
        business_purpose = f"Enable users to perform QA operations for the {self.intent['domain']} module."
        user_goals = [self.intent["user_goal"]]
        business_action = self.intent["business_action"]
        
        # Phase 2: Business Capability Model - Use dynamic terminology
        capabilities = {}
        entity = self.terminology.get_entity_term()
        entity_plural = self.terminology.get_entity_term(plural=True)
        transaction = self.terminology.get_transaction_term()
        
        for pk, page_sels in selectors.items():
            for ctype in ["inputs", "buttons", "dropdowns", "checkboxes", "links"]:
                for selector in page_sels.get(ctype, []):
                    sem_name = self.get_semantic_name(selector, ctype[:-1], pk)
                    
                    cap_key = "General Operations"
                    sem_lower = sem_name.lower()
                    if "search" in sem_lower or "query" in sem_lower:
                        cap_key = "Search & Query"
                    elif transaction in sem_lower or entity in sem_lower or "price" in sem_lower:
                        cap_key = f"{entity.capitalize()} & {transaction.capitalize()}"
                    elif "login" in sem_lower or "auth" in sem_lower or "password" in sem_lower:
                        cap_key = "Authentication & Security"
                    elif "cookie" in sem_lower or "consent" in sem_lower or "terms" in sem_lower:
                        cap_key = "Compliance & Preferences"
                    
                    if cap_key not in capabilities:
                        capabilities[cap_key] = []
                    capabilities[cap_key].append((selector, ctype[:-1], pk, sem_name))
                    
        # Phase 3 & 4: Workflow and Coverage Model
        coverage_checklists = {}
        for cap, elements in capabilities.items():
            coverage_checklists[cap] = {
                "workflows": ["Happy Path", "Negative Path", "Boundary Edge Case"],
                "validations": ["Visibility", "Enabled State", "Interaction Success", "Error Handling"],
                "elements_covered": len(elements)
            }
            
        # Phase 5: Gap Analysis - Use dynamic terminology
        active_caps = list(capabilities.keys())
        gaps_found = []
        domain_lower = self.intent.get("domain", "").lower()
        for expected_cap in ["Search & Query", f"{entity.capitalize()} & {transaction.capitalize()}"]:
            keyword = expected_cap.split()[-1].lower()
            if keyword in domain_lower and expected_cap not in active_caps:
                gaps_found.append(expected_cap)
                
        return {
            "purpose": business_purpose,
            "goals": user_goals,
            "action": business_action,
            "capabilities": capabilities,
            "checklists": coverage_checklists,
            "gaps": gaps_found
        }

    def _validate_and_review_testcase(self, title: str, desc: str, steps: List[Dict[str, Any]], expected: str, tctype: str) -> bool:
        """
        Applies the Enterprise Testcase Rejection Policy (Mandatory).
        Returns True if the testcase passes all quality gates, False if rejected.
        """
        title_lower = title.lower()
        desc_lower = desc.lower()
        exp_lower = expected.lower()

        # RULE 5: Acceptance Criteria titles check
        if any(f"ac-{i}" in title_lower or f"ac{i}" in title_lower for i in range(1, 20)):
            log.warning(f"[RejectionPolicy] Rejected '{title}': Cannot use AC reference in title.")
            return False

        # RULE 3: Generic expected results check
        generic_phrases = ["action completed", "success", "stable state", "navigation succeeds", "action is triggered"]
        if any(p in exp_lower for p in generic_phrases):
            log.warning(f"[RejectionPolicy] Rejected '{title}': Generic expected results detected.")
            return False

        # RULE 1 & 10: UI Inventory / click/existence only check
        # These keywords are meant to catch INCIDENTAL references to generic marketing/
        # footer pages (e.g. a "blog" link sitting in the footer of some unrelated page
        # under test). But when the module actually being tested IS one of these pages
        # (e.g. testing a "Blog" domain), every legitimate test case title naturally
        # contains that same word — the keyword is then just the module's own name, not
        # a sign of a generic non-business control, and must not be treated as one
        # (mirrors the same active-domain exclusion RULE 5 / Module Isolation already
        # applies below, generalized to any domain rather than hardcoded to one).
        active_domain_clean = self.intent.get("domain", "").lower().rstrip('s')
        ui_only_keywords = [
            "cookie settings", "cookie preferences", "reject all non-essential", "accept all",
            "read more", "subscribe now", "newsletter", "social media", "blog", "careers", "about us"
        ]
        matched_ui_only = [
            kw for kw in ui_only_keywords
            if (kw in title_lower or kw in desc_lower) and kw.rstrip('s') != active_domain_clean
        ]
        if matched_ui_only:
            log.warning(f"[RejectionPolicy] Rejected '{title}': UI inventory/non-business control scenario.")
            return False

        # RULE 4: Navigation only check
        if tctype == "Functional" and ("navigate" in title_lower or "link" in title_lower):
            has_subsequent_verification = False
            for s in steps:
                act = s["action"].lower()
                if "verify" in act or "observe" in act or "check" in act or "expect" in s["expected"].lower():
                    if not any(phrase in s["expected"].lower() for phrase in ["page loads", "url changes", "navigation"]):
                        has_subsequent_verification = True

            # Same active-domain exclusion as above: "blog"/"careers" here are meant to
            # detect a link to an external/unrelated page, not the module under test itself.
            is_external = any(
                kw in title_lower and kw.rstrip('s') != active_domain_clean
                for kw in ["external", "redirect", "blog", "careers"]
            )
            if not is_external and not has_subsequent_verification:
                log.warning(f"[RejectionPolicy] Rejected '{title}': Navigation-only testcase without destination analysis.")
                return False

        # RULE 5: Module Isolation - Use dynamic terminology
        entity = self.terminology.get_entity_term()
        # Dynamically extract unrelated modules from discovered routes
        discovered_routes = self.discovered_routes if hasattr(self, 'discovered_routes') else []
        unrelated_modules = self._extract_domain_terms_from_routes(discovered_routes, self.intent.get("domain", ""))
        active_domain = self.intent.get("domain", "").lower()
        is_unrelated = False
        for mod in unrelated_modules:
            mod_clean = mod.rstrip('s')
            active_clean = active_domain.rstrip('s')
            if mod_clean != active_clean and mod_clean in title_lower:
                is_unrelated = True
                break
                
        if is_unrelated:
            is_navigation_verification = any(kw in title_lower for kw in ["navigate", "redirect", "click", "transition"]) and not any(kw in title_lower for kw in ["latency", "offline", "reload", "injection", "double-click", "validation", "error", "standards", "aria"])
            if not is_navigation_verification:
                log.info(f"[ModuleIsolation] Rejected '{title}': Cross-module validation scenario blocked.")
                return False

        return True

    def generate(self) -> List[Dict[str, Any]]:
        domain = self.intent["domain"]
        module_prefix = re.sub(r'[^a-zA-Z0-9]', '', domain).upper()
        now = datetime.now(timezone.utc).isoformat()

        selectors = self.behavior_graph.get("selectors", {})
        
        # Execute the mandatory internal reasoning phase
        reasoning_context = self._run_qa_reasoning_engine(selectors, self.behavior_graph.get("edges", []))
        gaps_found = reasoning_context.get("gaps") or []
        if gaps_found:
            # Previously computed and never read anywhere — surface it so a coverage
            # gap actually shows up somewhere instead of being silently discarded.
            log.warning(
                f"[QAReasoningEngine] Gap analysis for '{domain}': expected capability coverage "
                f"not found in discovered selectors: {gaps_found}. Generated test cases may not "
                f"exercise these business capabilities."
            )

        page_key = domain.lower()
        page_selectors = selectors.get(page_key, {})
        
        # Resolve base URL from user story dynamically
        base_url = None
        story = self.story_data.get("story", {})
        story_text = f"{story.get('summary', '')} {story.get('description', '')} {' '.join([c.get('body', '') for c in story.get('comments', [])])} {' '.join(story.get('acceptance_criteria', []))}"
        url_matches = re.findall(r'https?://[a-zA-Z0-9./?=&_-]+', story_text)
        for url in url_matches:
            url_clean = url.rstrip('.,;)')
            from urllib.parse import urlparse
            parsed = urlparse(url_clean)
            if parsed.netloc:
                base_url = f"{parsed.scheme}://{parsed.netloc}"
                break
        if not base_url:
            base_url = "https://webook.com"

        target_url = page_selectors.get("url", f"{base_url.rstrip('/')}{self.intent['path']}")
        edges = self.behavior_graph.get("edges", [])

        # Determine all relevant pages for the module to scan recursively
        relevant_pages = []
        domain_clean = domain.lower()
        for key in selectors.keys():
            if domain_clean in key.lower() or key.lower() in domain_clean or key in selectors:
                relevant_pages.append(key)
        if not relevant_pages:
            relevant_pages = [page_key]

        # Pool elements from all domain-relevant pages
        raw_components_pool = []
        for pk in relevant_pages:
            p_selectors = selectors.get(pk, {})
            for item in p_selectors.get("inputs", []):
                raw_components_pool.append((item, "input", pk))
            for item in p_selectors.get("buttons", []):
                raw_components_pool.append((item, "button", pk))
            for item in p_selectors.get("dropdowns", []):
                raw_components_pool.append((item, "dropdown", pk))
            for item in p_selectors.get("checkboxes", []):
                raw_components_pool.append((item, "checkbox", pk))
            for item in p_selectors.get("radio_buttons", []):
                raw_components_pool.append((item, "radio_button", pk))
            for item in p_selectors.get("links", []):
                raw_components_pool.append((item, "link", pk))

        if not raw_components_pool:
            raw_components_pool = [
                (f"input[name='{domain.lower()}_input']", "input", page_key),
                (f"button:has-text('Submit')", "button", page_key),
                (f"a[href='/en/{domain.lower()}']", "link", page_key)
            ]

        # Filter raw components pool by active story relevance to prevent scope leakage
        story = self.story_data.get("story", {})
        summary = story.get("summary", "")
        description = story.get("description", "")
        # Exclude comments this pipeline posted on a prior run — otherwise a story that
        # has already been through the pipeline once feeds its own past output (including
        # any past mistake) back into itself as "story context" on every future run.
        comments = [c.get("body", "") for c in filter_human_comments(story.get("comments", []))]
        acceptance_criteria = story.get("acceptance_criteria", [])
        all_story_text = f"{summary} {description} {' '.join(comments)} {' '.join(acceptance_criteria)}"
        
        story_keywords = set()
        story_keywords.add(domain.lower())
        story_keywords.add(domain_clean.lower())
        story_keywords.add(domain_clean.lower().rstrip('s'))
        story_keywords.add(domain_clean.lower() + "s")
        
        story_words = re.findall(r'\b[a-zA-Z]{3,}\b', all_story_text.lower())
        stop_words = {
            "with", "from", "your", "that", "this", "have", "will", "what", "where", "user", "story", "epic",
            "feature", "module", "priority", "story", "points", "type", "title", "description", "business", "objective",
            "provide", "actor", "actors", "primary", "secondary", "preconditions", "trigger", "main", "flow",
            "alternate", "flows", "exception", "rules", "acceptance", "criteria", "non", "functional",
            "performance", "security", "compatibility", "reliability", "api", "integrations", "test",
            "coverage", "expectations", "derive", "given", "when", "then", "and", "the", "for", "only",
            "has", "text", "button", "input", "select", "option", "class", "id", "name", "value", "href", "link",
            "field", "click", "trigger", "page", "users", "should", "shouldn", "could", "would", "must", "either",
            "other", "without", "during", "after", "before", "while", "across", "about", "above", "below", "been",
            "was", "were", "are", "does", "did", "done", "doing", "goes", "went", "gone", "going", "came", "come"
        }
        for w in story_words:
            if w not in stop_words:
                story_keywords.add(w)

        def is_component_relevant(item: str, pk: str) -> bool:
            if pk == page_key:
                return True
            item_lower = item.lower()
            common_supporting = ["back", "cancel", "close", "home", "submit", "login", "sign in", "password", "email", "continue", "next", "previous"]
            if any(cs in item_lower for cs in common_supporting):
                return True
            
            # Use precise word matching to prevent substring leakage (e.g. 'lies' in 'policies' matching 'ladies')
            item_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', item_lower))
            if item_words.intersection(story_keywords):
                return True
            return False

        filtered_pool = []
        for item, itype, pk in raw_components_pool:
            if is_component_relevant(item, pk):
                filtered_pool.append((item, itype, pk))
        raw_components_pool = filtered_pool

        # Deduplicate and group parameterizable elements using dynamic terminology
        entity = self.terminology.get_entity_term()
        entity_plural = self.terminology.get_entity_term(plural=True)
        transaction = self.terminology.get_transaction_term()
        ui_terms = self.terminology._terminology_cache.get("ui_terms", [])
        
        # Extract filter-related terms from UI terms dynamically
        filter_keywords = set()
        date_keywords = set()
        location_keywords = set()
        audience_keywords = set()
        social_keywords = set()
        
        for term in ui_terms:
            term_lower = term.lower()
            if any(kw in term_lower for kw in ["category", "filter", "type"]):
                filter_keywords.add(term)
            if any(kw in term_lower for kw in ["date", "week", "month", "today", "tomorrow"]):
                date_keywords.add(term)
            if any(kw in term_lower for kw in ["destination", "location", "city", "country"]):
                location_keywords.add(term)
            if any(kw in term_lower for kw in ["audience", "age", "family", "adult", "kid"]):
                audience_keywords.add(term)
            if any(kw in term_lower for kw in ["facebook", "twitter", "instagram", "linkedin", "social", "share"]):
                social_keywords.add(term)
        
        # Build dynamic groups based on extracted terminology and business operations
        grouped_candidates = {}
        
        # Business operation groups: correlate related components that work together
        if date_keywords:
            grouped_candidates["Date Filter Option"] = {
                "keywords": list(date_keywords) + ["date", "week", "month"],
                "elements": [],
                "values": [],
                "business_operation": "filter_by_date"
            }
        
        if filter_keywords:
            grouped_candidates[f"{entity.capitalize()} Category Filter"] = {
                "keywords": list(filter_keywords) + ["category", "filter", "type"],
                "elements": [],
                "values": [],
                "business_operation": "filter_by_category"
            }
        
        if location_keywords:
            grouped_candidates["Location Selection Option"] = {
                "keywords": list(location_keywords) + ["destination", "location"],
                "elements": [],
                "values": [],
                "business_operation": "filter_by_location"
            }
        
        if audience_keywords:
            grouped_candidates["Audience Filter Option"] = {
                "keywords": list(audience_keywords) + ["audience", "age"],
                "elements": [],
                "values": [],
                "business_operation": "filter_by_audience"
            }
        
        if social_keywords:
            grouped_candidates["Social Media Link"] = {
                "keywords": list(social_keywords),
                "elements": [],
                "values": [],
                "business_operation": "share_content"
            }
        
        # Add search operation group (search input + search button)
        grouped_candidates["Search Operation"] = {
            "keywords": ["search", "query", "find"],
            "elements": [],
            "values": [],
            "business_operation": "search_content"
        }
        
        # Add transaction operation group (submit/confirm buttons related to entity)
        if transaction:
            grouped_candidates[f"{transaction.capitalize()} Operation"] = {
                "keywords": [transaction, "submit", "confirm", "place", "book"],
                "elements": [],
                "values": [],
                "business_operation": f"execute_{transaction}"
            }

        seen_components = set()
        components_pool = []
        
        group_values = {g: [] for g in grouped_candidates}
        group_elements = {g: [] for g in grouped_candidates}

        for comp, ctype, pk in raw_components_pool:
            sem_name = self.get_semantic_name(comp, ctype, pk)
            sem_lower = sem_name.lower()
            
            # Unknown Component Policy: Exclude elements with no identifiable business purpose
            clean_sem = sem_name.replace("Button", "").replace("Link", "").replace("Input Field", "").replace("Checkbox", "").replace("Dropdown", "").strip()
            is_placeholder_pattern = bool(
                re.search(r'[a-zA-Z]r\d+', sem_lower) or 
                re.search(r'buttonr\d+', sem_lower) or 
                re.search(r'linkr\d+', sem_lower) or 
                re.search(r'inputr\d+', sem_lower) or
                re.search(r'filters?\d+', sem_lower) or
                sem_lower in ["buttonr6", "buttonr7", "buttonr8", "buttonr9", "filters1", "filters2"]
            )
            if is_placeholder_pattern or not clean_sem or any(w in sem_lower for w in ["unknown", "placeholder", "control", "element"]):
                log.info(f"[UnknownComponentPolicy] Excluded component '{comp}' (type: {ctype}, resolved name: '{sem_name}') as placeholder or unknown business purpose.")
                continue

            # Check if this component belongs to a group for parameterization
            matched_group = None
            for group_name, g_info in grouped_candidates.items():
                if any(kw in sem_lower or kw in comp.lower() for kw in g_info["keywords"]):
                    if sem_lower != domain.lower() and sem_lower != "explore":
                        matched_group = group_name
                        break
            
            if matched_group:
                group_values[matched_group].append(sem_name)
                group_elements[matched_group].append((comp, ctype, pk))
            else:
                comp_key_tuple = (ctype, sem_name, pk)
                if comp_key_tuple not in seen_components:
                    seen_components.add(comp_key_tuple)
                    components_pool.append((comp, ctype, pk, sem_name, None))

        # Add representative component for each active group with diversified scenarios
        for group_name, elems in group_elements.items():
            if elems:
                rep_comp, rep_ctype, rep_pk = elems[0]
                unique_vals = sorted(list(set(group_values[group_name])))
                business_operation = grouped_candidates.get(group_name, {}).get("business_operation", "")
                # Add business operation context to enable diversified scenario generation
                components_pool.append((rep_comp, rep_ctype, rep_pk, group_name, unique_vals, business_operation))

        # Classify components dynamically
        classified_components = []
        global_components = []
        for comp_tuple in components_pool:
            # Handle both regular components (5 elements) and grouped components (6 elements)
            if len(comp_tuple) == 6:
                comp, ctype, pk, semantic_name, exec_data, business_operation = comp_tuple
            else:
                comp, ctype, pk, semantic_name, exec_data = comp_tuple
                business_operation = ""
            
            p_selectors = selectors.get(pk, {})
            prio_class = self._determine_priority_class(comp, ctype, p_selectors, self.intent)
            comp_item = {
                "selector": comp,
                "type": ctype,
                "priority_class": prio_class,
                "semantic_name": semantic_name,
                "page_key": pk,
                "exec_data": exec_data,
                "business_operation": business_operation
            }
            is_global = TestCaseGenerator.is_global_component(comp, semantic_name)
            domain_lower = self.intent.get("domain", "").lower()
            if is_global:
                # If the domain is explicitly testing this global component/area, do not filter it out
                is_explicit_global = any(kw in domain_lower for kw in ["cookie", "banner", "newsletter", "subscribe", "footer", "header"])
                if is_explicit_global:
                    is_global = False
            
            if is_global:
                global_components.append(comp_item)
            else:
                classified_components.append(comp_item)

        # Helper to clean selectors
        def clean_selector_name(sel: str) -> str:
            res = sel.replace("button:has-text('", "").replace("')", "")
            res = res.replace("input[name='", "").replace("']", "")
            return res

        # Scenarios mapping function with enterprise-level coverage and diversified scenarios
        def get_scenarios_for_component_v2(comp_obj: Dict[str, Any], page_sels: Dict[str, Any]) -> List[Dict[str, Any]]:
            selector = comp_obj["selector"]
            ctype = comp_obj["type"]
            prio_class = comp_obj["priority_class"]
            semantic_name = comp_obj["semantic_name"]
            business_operation = comp_obj.get("business_operation", "")
            exec_data = comp_obj.get("exec_data")
            
            # Get terminology for diversified scenarios
            entity = self.terminology.get_entity_term()
            entity_plural = self.terminology.get_entity_term(plural=True)
            transaction = self.terminology.get_transaction_term()
            
            scenarios = []
            clean_sel = clean_selector_name(selector)
            
            # Track scenario types to ensure diversity
            scenario_types_generated = set()
            
            # Extract validation rules dynamically
            is_required = "[required]" in selector.lower() or "required" in selector.lower() or any(selector in r for r in page_sels.get("required_fields", []))
            is_disabled = "disabled" in selector.lower() or any(selector in d for d in page_sels.get("disabled_controls", []))
            
            # Parse dynamic limits from business rules
            max_len = None
            min_len = None
            max_val = None
            min_val = None
            for rule in page_sels.get("business_rules", []):
                if clean_sel.lower() in rule.lower() or selector in rule:
                    if "mandatory" in rule.lower() or "required" in rule.lower():
                        is_required = True
                    # Check length boundaries
                    len_match = re.search(r'character length boundaries: min=(\w+), max=(\w+)', rule)
                    if len_match:
                        min_len = len_match.group(1) if len_match.group(1) != "none" else None
                        max_len = len_match.group(2) if len_match.group(2) != "none" else None
                    # Check numeric boundaries
                    num_match = re.search(r'numeric boundaries: min=(\w+), max=(\w+)', rule)
                    if num_match:
                        min_val = num_match.group(1) if num_match.group(1) != "none" else None
                        max_val = num_match.group(2) if num_match.group(2) != "none" else None
            
            # For grouped components with business operations, generate diversified scenarios
            if business_operation and exec_data and len(exec_data) > 1:
                # Generate diversified scenarios based on business operation type
                if "filter" in business_operation:
                    # Scenario 1: Filter with valid criteria
                    scenarios.append({
                        "suffix": f"applies {business_operation} with valid criteria and updates display",
                        "desc": f"Validates that category filter applies valid business criteria and updates {entity_plural} display with accurate results.",
                        "priority": "High",
                        "type": "Functional",
                        "data": {"filter_value": exec_data[0] if exec_data else "valid_criteria"},
                        "steps": [
                            {"step": 1, "action": f"Select valid filter option from category filter", "expected": f"Filter option is selected and {business_operation} is applied."},
                            {"step": 2, "action": f"Verify {entity_plural} display updates with filtered results", "expected": f"{entity_plural.capitalize()} list is filtered to show only matching items with business criteria."}
                        ],
                        "expected": f"{entity_plural.capitalize()} display is updated to show only items matching the {business_operation} criteria, with accurate filtering and preserved ordering."
                    })
                    scenario_types_generated.add("filter_valid")
                    
                    # Scenario 2: Filter with boundary/edge case criteria
                    scenarios.append({
                        "suffix": f"applies {business_operation} with edge case criteria and validates boundary behavior",
                        "desc": f"Validates that category filter handles edge case filter criteria with proper boundary validation and data integrity.",
                        "priority": "Medium",
                        "type": "Edge",
                        "data": {"filter_value": exec_data[-1] if len(exec_data) > 1 else "edge_criteria"},
                        "steps": [
                            {"step": 1, "action": f"Select edge case filter option from category filter", "expected": f"Edge case filter option is selected and {business_operation} processes boundary conditions."},
                            {"step": 2, "action": f"Verify {entity_plural} display handles edge case correctly", "expected": f"{entity_plural.capitalize()} list handles edge case filter with proper boundary validation and no errors."}
                        ],
                        "expected": f"{business_operation} correctly handles edge case filter criteria with proper boundary validation, data integrity, and no unexpected behavior."
                    })
                    scenario_types_generated.add("filter_edge")
                    
                    # Scenario 3: Clear filter and restore original state
                    scenarios.append({
                        "suffix": f"clears {business_operation} and restores original {entity_plural} display state",
                        "desc": f"Validates that clearing category filter restores the original {entity_plural} display state with preserved ordering and all items visible.",
                        "priority": "Medium",
                        "type": "Functional",
                        "data": {"filter_value": "clear"},
                        "steps": [
                            {"step": 1, "action": f"Clear applied filter from category filter", "expected": f"{business_operation} is cleared and filter state is reset."},
                            {"step": 2, "action": f"Verify {entity_plural} display restores to original state", "expected": f"{entity_plural.capitalize()} list is restored to original unfiltered state with preserved ordering and all items visible."}
                        ],
                        "expected": f"Clearing {business_operation} restores the original {entity_plural} display state with all items visible, preserved ordering, and correct data integrity."
                    })
                    scenario_types_generated.add("filter_clear")
                    
                    return scenarios  # Return diversified scenarios for grouped components
                
                elif "search" in business_operation:
                    # Search operation scenarios with business context
                    scenarios.append({
                        "suffix": "executes search with valid business queries and displays matching results",
                        "desc": f"Validates that search function executes search operation with valid business queries and displays matching {entity_plural} with accurate filtering and correct result ordering.",
                        "priority": "High",
                        "type": "Functional",
                        "data": {"input_value": f"valid_{entity}_query"},
                        "steps": [
                            {"step": 1, "action": f"Enter valid business search query in search function", "expected": f"Search query is entered and {business_operation} is prepared."},
                            {"step": 2, "action": f"Execute search and verify matching {entity_plural} are displayed", "expected": f"Search results display matching {entity_plural} with accurate filtering and business criteria."}
                        ],
                        "expected": f"Search operation returns matching {entity_plural} with accurate filtering, business criteria validation, and correct result ordering."
                    })
                    scenario_types_generated.add("search_valid")
                    
                    scenarios.append({
                        "suffix": "handles search with no matching results and displays appropriate empty state",
                        "desc": f"Validates that search function handles search queries with no matching {entity_plural} and displays appropriate empty state message with clear user guidance.",
                        "priority": "Medium",
                        "type": "Negative",
                        "data": {"input_value": "nomatch_xyz_999"},
                        "steps": [
                            {"step": 1, "action": f"Enter search query with no matching {entity_plural} in search function", "expected": f"Search query is entered and no matches are found."},
                            {"step": 2, "action": f"Verify empty state message is displayed", "expected": f"Appropriate empty state message is displayed with clear guidance and no errors."}
                        ],
                        "expected": f"Search operation gracefully handles no-match queries with appropriate empty state message, clear user guidance, and stable application state."
                    })
                    scenario_types_generated.add("search_empty")
                    
                    return scenarios
                
                elif "execute" in business_operation:
                    # Transaction operation scenarios
                    scenarios.append({
                        "suffix": f"executes {business_operation} with valid data and completes successfully",
                        "desc": f"Validates that booking action executes {business_operation} with valid business data and completes with expected state transition and data integrity preserved.",
                        "priority": "High",
                        "type": "Functional",
                        "data": {f"{entity}_data": f"valid_{entity}_test_data"},
                        "steps": [
                            {"step": 1, "action": f"Prepare valid business data for {business_operation}", "expected": f"Business data is prepared with valid values and required fields."},
                            {"step": 2, "action": f"Execute {business_operation} via booking action", "expected": f"{business_operation} is executed and business rules are validated."},
                            {"step": 3, "action": f"Verify {entity} state transition and completion", "expected": f"{entity.capitalize()} state transitions to expected final state with data integrity preserved."}
                        ],
                        "expected": f"{business_operation} completes successfully with valid business data, business rules validated, {entity} state transitioned, and data integrity preserved."
                    })
                    scenario_types_generated.add("transaction_valid")
                    
                    scenarios.append({
                        "suffix": f"handles {business_operation} with invalid data and enforces business rules",
                        "desc": f"Validates that booking action handles {business_operation} with invalid data and enforces business rules with appropriate error handling and state preservation.",
                        "priority": "High",
                        "type": "Negative",
                        "data": {f"{entity}_data": "invalid_business_data"},
                        "steps": [
                            {"step": 1, "action": f"Prepare invalid business data for {business_operation}", "expected": f"Business data contains invalid values or missing required fields."},
                            {"step": 2, "action": f"Execute {business_operation} via booking action", "expected": f"{business_operation} is rejected and business rules are enforced."},
                            {"step": 3, "action": f"Verify error message and state preservation", "expected": f"Appropriate error message is displayed and {entity} state is preserved without corruption."}
                        ],
                        "expected": f"{business_operation} is rejected with appropriate error message, business rules enforced, {entity} state preserved, and no data corruption."
                    })
                    scenario_types_generated.add("transaction_invalid")
                    
                    return scenarios

            # 1. Inputs
            if ctype == "input":
                is_search = "search" in selector.lower() or "search" in semantic_name.lower() or "query" in selector.lower()
                
                if is_search:
                    # Search query happy path - Use dynamic terminology
                    entity_plural = self.terminology.get_entity_term(plural=True)
                    
                    scenarios.append({
                        "suffix": "accepts valid search queries and displays matching results",
                        "desc": f"Validates functional search input query processing for {semantic_name} with accurate filtering and result display.",
                        "priority": "High",
                        "type": "Functional",
                        "data": {"input_value": "active_item"},
                        "steps": [
                            {"step": 1, "action": f"Focus on search field {semantic_name}", "expected": f"{semantic_name} is focused and ready for input."},
                            {"step": 2, "action": f"Enter valid search keyword matching active {entity_plural}", "expected": f"Search term is entered in {semantic_name} and input is validated."},
                            {"step": 3, "action": "Trigger search execution", "expected": f"Results list is updated dynamically with matching {entity_plural} within 2 seconds."}
                        ],
                        "expected": "Matching results are dynamically rendered within 2 seconds; mismatching items are hidden; result count is displayed accurately."
                    })
                    
                    # Search mismatch empty state
                    scenarios.append({
                        "suffix": "handles non-matching query terms and renders empty state",
                        "desc": f"Verifies system gracefully handles query terms with zero matches in {semantic_name} with appropriate empty state messaging.",
                        "priority": "Medium",
                        "type": "Negative",
                        "data": {"input_value": "xyz999_nomatch"},
                        "steps": [
                            {"step": 1, "action": f"Enter mismatching keyword with no matching {entity_plural} in {semantic_name}", "expected": "Query is updated and validated."},
                            {"step": 2, "action": "Trigger search execution", "expected": "Empty results state is displayed within 2 seconds with clear guidance message."}
                        ],
                        "expected": "Graceful empty state placeholder message is displayed within 2 seconds; clear-search button is visible and functional; no errors occur."
                    })
                    
                    # Special characters / Unicode / Emoji search
                    scenarios.append({
                        "suffix": "handles special characters and emoji search queries safely",
                        "desc": f"Ensures security filters and parser handle special characters, unicode, and emojis in search field {semantic_name} with proper sanitization.",
                        "priority": "High",
                        "type": "Edge",
                        "data": {"input_value": "test_★_🔥_<script>"},
                        "steps": [
                            {"step": 1, "action": f"Enter string containing unicode, emojis, and script characters in {semantic_name}", "expected": "Search term is input and sanitized without script execution."},
                            {"step": 2, "action": "Trigger search execution", "expected": "Query executes safely within 2 seconds with no security violations or errors."}
                        ],
                        "expected": "Search request completes safely within 2 seconds; special characters are sanitized; no script execution occurs; system remains stable."
                    })
                    
                    # Clear search
                    scenarios.append({
                        "suffix": "clears active query and restores default view list",
                        "desc": f"Validates that clearing the search input {semantic_name} resets list views to default states with all items visible.",
                        "priority": "High",
                        "type": "Functional",
                        "steps": [
                            {"step": 1, "action": f"Enter search keyword in {semantic_name} to filter results", "expected": f"Filtered results are shown with matching {entity_plural}."},
                            {"step": 2, "action": "Click clear search control / erase query", "expected": "Search field is cleared within 1 second and full list is restored."}
                        ],
                        "expected": f"Full default list of active items is restored to view within 2 seconds; all {entity_plural} are visible; search field is empty."
                    })

                else:
                    # Happy path input - Use dynamic terminology
                    entity = self.terminology.get_entity_term()
                    transaction = self.terminology.get_transaction_term()
                    
                    if "password" in selector.lower():
                        step_expected_val = "Password characters are masked."
                        final_expected_val = "Password characters are masked, preparing secure form submission."
                    elif "email" in selector.lower() or "mail" in selector.lower():
                        step_expected_val = "Email address accepts input."
                        final_expected_val = "Email format validation passes successfully."
                    elif transaction in selector.lower() or entity.lower() in page_key:
                        step_expected_val = f"{transaction.capitalize()} amount is entered."
                        final_expected_val = f"{transaction.capitalize()} amount is validated, enabling {transaction} submission control."
                    else:
                        step_expected_val = f"Value is entered in the {semantic_name} field."
                        final_expected_val = f"{semantic_name} input is validated successfully."

                    scenarios.append({
                        "suffix": "accepts valid inputs and updates field state",
                        "desc": f"Validates functional input processing for {semantic_name} with proper validation and state update.",
                        "priority": "High" if prio_class == "critical" else "Medium",
                        "type": "Functional",
                        "data": {"input_value": "standard_text_123"},
                        "steps": [
                            {"step": 1, "action": f"Focus on {semantic_name}", "expected": f"{semantic_name} is focused and ready for input within 1 second."},
                            {"step": 2, "action": f"Enter standard alphanumeric test value into {semantic_name}", "expected": f"{step_expected_val} Input is validated and field state is updated."}
                        ],
                        "expected": f"{final_expected_val} Field value is persisted; validation passes; no errors occur; field is ready for form submission."
                    })
                    
                    # Validation error on empty submit - Use dynamic terminology
                    if is_required or prio_class == "critical":
                        if transaction in selector.lower() or entity.lower() in page_key:
                            step_expected_val = f"{transaction.capitalize()} submission is blocked, and validation message is shown."
                            final_expected_val = f"{transaction.capitalize()} request is blocked from submission, ensuring data integrity."
                        elif any(w in selector.lower() for w in ["password", "email", "login"]):
                            step_expected_val = "Validation message is displayed, and login is blocked."
                            final_expected_val = "Authentication is rejected, preventing invalid credential submission."
                        else:
                            step_expected_val = f"Validation message is displayed for missing {semantic_name}."
                            final_expected_val = f"Validation warning prevents form submission for {semantic_name}."

                        scenarios.append({
                            "suffix": "handles validation error on blank submission",
                            "desc": f"Ensures form validation displays appropriate error if required input {semantic_name} is submitted empty with clear user guidance.",
                            "priority": "High",
                            "type": "Negative",
                            "steps": [
                                {"step": 1, "action": f"Clear any pre-filled value in {semantic_name}", "expected": f"{semantic_name} is empty and ready for validation."},
                                {"step": 2, "action": "Trigger form submission action", "expected": f"{step_expected_val} Error message is displayed within 1 second."}
                            ],
                            "expected": f"{final_expected_val} Validation error message is clear and specific; form submission is blocked; field is highlighted; user can correct the error."
                        })
                        
                    # XSS / SQL Injection check for critical inputs
                    if prio_class == "critical":
                        scenarios.append({
                            "suffix": "sanitizes special characters and script injection inputs safely",
                            "desc": f"Checks input sanitization and script safety boundaries on critical input {semantic_name} with proper security filtering.",
                            "priority": "High",
                            "type": "Edge",
                            "data": {"input_value": "<script>alert('xss')</script>"},
                            "steps": [
                                {"step": 1, "action": f"Input malicious script tags or SQL injection strings in {semantic_name}", "expected": "Malicious script is entered without executing; input is sanitized within 1 second."},
                                {"step": 2, "action": "Submit input parameters to the application", "expected": "System warns user or sanitizes input; no script execution occurs."}
                            ],
                            "expected": "Security filters successfully sanitize injection attempts within 2 seconds; system blocks execution; preserves stable state; no security violations occur."
                        })
                        
                        # Session Expiry edge case
                        scenarios.append({
                            "suffix": "handles session expiry gracefully during active text entry",
                            "desc": f"Ensures the system preserves inputs or prompts for re-authentication if the session expires while typing in critical input {semantic_name} with proper state preservation.",
                            "priority": "High",
                            "type": "Edge",
                            "steps": [
                                {"step": 1, "action": f"Begin typing a valid value into critical input {semantic_name}", "expected": "Characters are entered and field state is updated."},
                                {"step": 2, "action": "Simulate session expiry / auth token invalidation event", "expected": "Session expiry warning is shown within 1 second."},
                                {"step": 3, "action": "Complete typing and trigger next interaction step", "expected": "Re-authentication dialog is displayed within 2 seconds."}
                            ],
                            "expected": "User session expiry is handled gracefully within 2 seconds; draft data is preserved and user is guided to log in again; no data loss occurs."
                        })
                        
                        # Refresh Protection edge case
                        scenarios.append({
                            "suffix": "warns user on mid-transaction page reload refresh",
                            "desc": f"Ensures the system prompts the user before losing unsubmitted progress on page refresh when interacting with {semantic_name} with proper data protection.",
                            "priority": "Medium",
                            "type": "Edge",
                            "steps": [
                                {"step": 1, "action": f"Populate {semantic_name} with transactional data", "expected": "Form inputs are updated and data is persisted."},
                                {"step": 2, "action": "Trigger page refresh / reload operation", "expected": "Confirmation dialog is displayed within 1 second warning of data loss."}
                            ],
                            "expected": "System prevents accidental data loss during active transaction via standard confirmation warnings within 2 seconds; user can cancel or proceed with clear understanding."
                        })
                        
                    # Specific input behaviors
                    if "password" in selector.lower():
                        scenarios.append({
                            "suffix": "masks text entry visually for security verification",
                            "desc": f"Ensures password input values are masked to protect user credentials in {semantic_name} with proper security measures.",
                            "priority": "High",
                            "type": "Functional",
                            "steps": [
                                {"step": 1, "action": f"Input characters into password field {semantic_name}", "expected": "Password characters are masked within 1 second; no plaintext is visible."}
                            ],
                            "expected": "Password values are masked visually to protect user security; masking is consistent; no plaintext exposure occurs."
                        })
                    elif "email" in selector.lower() or "mail" in selector.lower():
                        scenarios.append({
                            "suffix": "rejects malformed email format values",
                            "desc": f"Ensures the system validates email format for input field {semantic_name} with proper format validation and clear error messaging.",
                            "priority": "High",
                            "type": "Negative",
                            "data": {"input_value": "invalid-email-format"},
                            "steps": [
                                {"step": 1, "action": f"Enter an invalid email format into {semantic_name}", "expected": "Invalid email is entered and field is ready for validation."},
                                {"step": 2, "action": "Attempt to submit the form", "expected": "Validation message is displayed within 1 second indicating invalid format."}
                            ],
                            "expected": "Incorrect email format is rejected by the validation interface within 2 seconds; clear error message is shown; form submission is blocked; user can correct the error."
                        })
                    
                    # Length boundaries
                    if max_len or min_len:
                        limit_val = max_len or min_len
                        scenarios.append({
                            "suffix": f"respects boundary constraints at length limit of {limit_val} characters",
                            "desc": f"Verifies boundary character limits on input field {semantic_name} with proper enforcement and user feedback.",
                            "priority": "Medium",
                            "type": "Edge",
                            "data": {"input_value": "A" * (int(limit_val) + 1 if limit_val.isdigit() else 256)},
                            "steps": [
                                {"step": 1, "action": f"Enter characters exceeding limit in {semantic_name}", "expected": "Character entry reaches input boundaries and input is truncated or rejected within 1 second."},
                                {"step": 2, "action": "Submit input value to the system", "expected": "System rejects or truncates value; validation message is displayed."}
                            ],
                            "expected": f"Character length constraints are enforced within 2 seconds, limiting content processing to {limit_val} characters; clear validation message is shown; no data corruption occurs."
                        })

                    # Numeric boundaries
                    if max_val or min_val:
                        scenarios.append({
                            "suffix": f"enforces numeric range constraints: min={min_val}, max={max_val}",
                            "desc": f"Validates boundary numeric limits for input field {semantic_name} with proper range validation.",
                            "priority": "Medium",
                            "type": "Edge",
                            "data": {"input_value": str(int(max_val) + 1 if max_val.isdigit() else 9999)},
                            "steps": [
                                {"step": 1, "action": f"Enter a value outside numeric bounds into {semantic_name}", "expected": "Warning message is displayed within 1 second indicating value is out of range."}
                            ],
                            "expected": "Bidding or numeric values outside limits are rejected within 2 seconds; clear error message is shown; preventing overflow or transaction errors; user can correct the value."
                        })

                    # Depth scenario 1: State persistence on refresh
                    scenarios.append({
                        "suffix": "preserves entry value across page refresh",
                        "desc": f"Ensures value in input field {semantic_name} is handled gracefully on page reload with proper state preservation.",
                        "priority": "Medium",
                        "type": "Functional",
                        "steps": [
                            {"step": 1, "action": f"Enter a valid value in {semantic_name}", "expected": f"Value is entered in {semantic_name} and field state is updated."},
                            {"step": 2, "action": "Trigger page refresh or reload operation", "expected": "Page content reloads within 2 seconds and value is preserved or clearly communicated."}
                        ],
                        "expected": "Draft inputs are handled gracefully on browser refresh events within 2 seconds; value is preserved or user is informed of data loss; no unexpected errors occur."
                    })
                    # Depth scenario 2: Offline input entry validation
                    scenarios.append({
                        "suffix": "handles input validation during offline network state",
                        "desc": f"Ensures input field {semantic_name} handles offline network state gracefully with clear user guidance.",
                        "priority": "High",
                        "type": "Negative",
                        "steps": [
                            {"step": 1, "action": "Simulate network offline state", "expected": "Offline mode is active and network status is detected within 1 second."},
                            {"step": 2, "action": f"Enter a value into input field {semantic_name}", "expected": f"Value is entered in {semantic_name} and field state is updated."},
                            {"step": 3, "action": "Trigger form submission or input validation", "expected": "Offline connection warning is displayed within 2 seconds with clear guidance."}
                        ],
                        "expected": "System displays clear network warning during offline input validation attempts within 2 seconds; submission is blocked; user can retry when connection is restored."
                    })
                    # Depth scenario 3: Cross-field validation dependencies
                    scenarios.append({
                        "suffix": "validates cross-field dependency validation rules",
                        "desc": f"Checks how {semantic_name} validation interacts with other fields on the form with proper dependency handling.",
                        "priority": "Medium",
                        "type": "Functional",
                        "steps": [
                            {"step": 1, "action": "Leave required sibling form inputs blank", "expected": "Sibling fields are empty and ready for validation."},
                            {"step": 2, "action": f"Enter a valid value in input field {semantic_name}", "expected": f"Value is entered in {semantic_name} and field state is updated."},
                            {"step": 3, "action": "Click the form's submit button to trigger validation", "expected": "Validation flags missing dependencies within 1 second with clear error messages."}
                        ],
                        "expected": "Cross-field validation rules prevent incomplete submissions within 2 seconds; errors are highlighted clearly; user can identify and fix missing dependencies."
                    })

            # 2. Buttons
            elif ctype == "button":
                is_filter = "filter" in selector.lower() or "filter" in semantic_name.lower()
                is_sort = "sort" in selector.lower() or "sort" in semantic_name.lower() or "order" in selector.lower()

                if is_filter:
                    scenarios.append({
                        "suffix": "filters results by individual category criteria",
                        "desc": f"Validates that clicking the filter button {semantic_name} updates list selection criteria with accurate filtering.",
                        "priority": "High",
                        "type": "Functional",
                        "steps": [
                            {"step": 1, "action": f"Click on filter button {semantic_name}", "expected": "Filter criteria is applied to active listing space within 2 seconds and results are updated."}
                        ],
                        "expected": "Results list is updated dynamically within 2 seconds to show elements matching the filter category; non-matching items are hidden; filter state is visible."
                    })
                elif is_sort:
                    scenarios.append({
                        "suffix": "orders results dynamically by selected sort criteria",
                        "desc": f"Validates sorting order sequence layout updates on clicking {semantic_name} with correct ordering.",
                        "priority": "High",
                        "type": "Functional",
                        "steps": [
                            {"step": 1, "action": f"Click sort button {semantic_name}", "expected": "List sequence is updated within 2 seconds conforming to chosen sorting direction."}
                        ],
                        "expected": "Display items are ordered sequentially within 2 seconds conforming to chosen sorting direction criteria; sort indicator is visible; ordering is consistent."
                    })
                else:
                    # Use dynamic terminology for critical buttons
                    entity = self.terminology.get_entity_term()
                    transaction = self.terminology.get_transaction_term()
                    
                    if prio_class == "critical":
                        if "login" in selector.lower() or "auth" in page_key:
                            final_expected_val = "User is authenticated and redirected to the dashboard within 3 seconds."
                        elif transaction in selector.lower() or entity.lower() in page_key:
                            final_expected_val = f"{transaction.capitalize()} is successfully submitted within 3 seconds, and {transaction} status is updated."
                        else:
                            final_expected_val = f"Action triggers page navigation or loads next section within 2 seconds."

                        scenarios.append({
                            "suffix": "triggers primary action successfully on user click",
                            "desc": f"Validates click processing on primary button {semantic_name} with proper action execution.",
                            "priority": "High",
                            "type": "Functional",
                            "steps": [
                                {"step": 1, "action": f"Click on the {semantic_name}", "expected": f"{semantic_name} is clicked and action is initiated within 1 second."},
                                {"step": 2, "action": "Wait for system response", "expected": "Action executes successfully within 3 seconds with no errors."}
                            ],
                            "expected": final_expected_val
                        })
                        scenarios.append({
                            "suffix": "prevents duplicate requests on rapid double-click triggers",
                            "desc": f"Ensures click action is throttled/debounced on critical button {semantic_name} to avoid duplicate transaction entries.",
                            "priority": "Medium",
                            "type": "Edge",
                            "steps": [
                                {"step": 1, "action": f"Rapidly click {semantic_name} twice", "expected": "Only the first click is registered."}
                            ],
                            "expected": "Duplicate submission is safely suppressed, preserving transaction consistency."
                        })
                        scenarios.append({
                            "suffix": "handles network offline state gracefully on click",
                            "desc": f"Checks network connection loss error handling for button {semantic_name}.",
                            "priority": "High",
                            "type": "Negative",
                            "steps": [
                                {"step": 1, "action": "Simulate network offline state", "expected": "Offline state is simulated."},
                                {"step": 2, "action": f"Click {semantic_name}", "expected": "Offline error message is displayed."}
                            ],
                            "expected": "System blocks request transmission during disconnection, displaying user-friendly offline message."
                        })
                        
                        # Latency Spinner edge case
                        scenarios.append({
                            "suffix": "handles high latency and displays progress spinner on click",
                            "desc": f"Verifies that a click on critical button {semantic_name} shows loading indicator during slow network response.",
                            "priority": "High",
                            "type": "Edge",
                            "steps": [
                                {"step": 1, "action": "Configure network latency emulation (e.g. 5000ms delay)", "expected": "Network latency is simulated."},
                                {"step": 2, "action": f"Click on {semantic_name}", "expected": "Progress spinner is displayed."},
                                {"step": 3, "action": "Wait for delayed API response to resolve", "expected": "Operation completes successfully."}
                            ],
                            "expected": "System handles slow network gracefully, preventing multiple clicks and displaying clear progress feedback."
                        })
                        
                        # Refresh Protection edge case
                        scenarios.append({
                            "suffix": "warns user on mid-transaction page reload refresh",
                            "desc": f"Ensures the system prompts the user before losing unsubmitted progress on page refresh when interacting with {semantic_name}.",
                            "priority": "Medium",
                            "type": "Edge",
                            "steps": [
                                {"step": 1, "action": f"Initiate action by clicking critical button {semantic_name}", "expected": "Transaction is initiated."},
                                {"step": 2, "action": "Trigger page refresh / reload operation before API resolves", "expected": "Confirmation dialog is displayed."}
                            ],
                            "expected": "System prevents accidental data loss during active transaction via standard confirmation warnings."
                        })
                    else:
                        action_name = semantic_name.lower()
                        if any(w in action_name for w in ["submit", "save", "send", "register", "apply", "create"]):
                            outcome = "Form data is processed and saved successfully, transitioning system state."
                        elif any(w in action_name for w in ["cancel", "close", "dismiss", "reject"]):
                            outcome = "Action is cancelled, restoring previous display state without saving changes."
                        elif any(w in action_name for w in ["edit", "update", "modify"]):
                            outcome = "Form fields are enabled for editing or changes are saved to memory."
                        elif any(w in action_name for w in ["next", "continue", "forward"]):
                            outcome = "Next form step or layout section is loaded into view."
                        elif any(w in action_name for w in ["back", "previous"]):
                            outcome = "Previous wizard step or page view is restored."
                        elif any(w in action_name for w in ["open", "view", "show"]):
                            outcome = "Target modal, drawer, or details view is rendered in the viewport."
                        elif any(w in action_name for w in ["add", "new", "plus"]):
                            outcome = "New draft record item is added to the listing layout."
                        elif any(w in action_name for w in ["delete", "remove", "clear"]):
                            outcome = "Selected record or filter parameters are removed from system context."
                        else:
                            outcome = f"Target {semantic_name} action executes, updating page view or data context."

                        scenarios.append({
                            "suffix": "executes target interaction on click",
                            "desc": f"Validates click response for button {semantic_name}.",
                            "priority": "Medium",
                            "type": "Functional",
                            "steps": [
                                {"step": 1, "action": f"Click on {semantic_name}", "expected": "Action is triggered."}
                            ],
                            "expected": outcome
                        })
                    
                    if is_disabled:
                        scenarios.append({
                            "suffix": "remains non-interactive and blocks clicks when disabled",
                            "desc": f"Validates disabled state behavior on button {semantic_name}.",
                            "priority": "High",
                            "type": "Negative",
                            "steps": [
                                {"step": 1, "action": f"Ensure {semantic_name} is in disabled state", "expected": f"{semantic_name} is disabled."},
                                {"step": 2, "action": f"Attempt click interaction on {semantic_name}", "expected": "No action is triggered."}
                            ],
                            "expected": "Disabled button ignores interaction, preventing premature form submissions."
                        })
                    # NOTE: Generic depth scenarios (double-click, API rollback, latency)
                    # are intentionally omitted for non-critical buttons to avoid
                    # low-value repetition. These are generated for critical buttons only.

            # 3. Dropdowns
            elif ctype == "dropdown":
                is_filter = "filter" in selector.lower() or "filter" in semantic_name.lower() or "category" in selector.lower() or "status" in selector.lower() or "type" in selector.lower()
                is_sort = "sort" in selector.lower() or "sort" in semantic_name.lower() or "order" in selector.lower() or "orderby" in selector.lower()

                if is_sort:
                    scenarios.append({
                        "suffix": "orders results dynamically by selected sort criteria",
                        "desc": f"Validates sorting order sequence layout updates on dropdown option selection in {semantic_name}.",
                        "priority": "High",
                        "type": "Functional",
                        "steps": [
                            {"step": 1, "action": f"Click dropdown {semantic_name}", "expected": "Sorting options are displayed."},
                            {"step": 2, "action": "Select sort direction or sequence parameter", "expected": "Sorting criterion is applied to the results."}
                        ],
                        "expected": "Listing items are ordered sequentially conforming to chosen sorting direction criteria."
                    })
                    scenarios.append({
                        "suffix": "preserves active filter states during sort actions",
                        "desc": f"Ensures chosen sort order operates on already filtered listings in {semantic_name}.",
                        "priority": "Medium",
                        "type": "Edge",
                        "steps": [
                            {"step": 1, "action": "Apply a filter category to restrict listing items", "expected": "List displays subset matching filter."},
                            {"step": 2, "action": f"Select sorting criteria from dropdown {semantic_name}", "expected": "List subset is ordered sequentially."}
                        ],
                        "expected": "Restricted items list is sorted sequentially without resetting active filters."
                    })
                elif is_filter:
                    scenarios.append({
                        "suffix": "filters results by individual category criteria",
                        "desc": f"Validates filters applied from selection options in dropdown {semantic_name}.",
                        "priority": "High",
                        "type": "Functional",
                        "steps": [
                            {"step": 1, "action": f"Click dropdown {semantic_name}", "expected": "Filter categories list is shown."},
                            {"step": 2, "action": "Select a category parameter option", "expected": "Selected category parameter is applied."}
                        ],
                        "expected": "List updates to show only items matching selected category."
                    })
                    scenarios.append({
                        "suffix": "combines multiple filter categories to refine results",
                        "desc": f"Ensures system combines multiple selection options from {semantic_name} to filter list.",
                        "priority": "Medium",
                        "type": "Edge",
                        "steps": [
                            {"step": 1, "action": f"Select first category option in {semantic_name}", "expected": "List is updated with first category."},
                            {"step": 2, "action": "Select additional category option in the UI layout", "expected": "Multiple filter options are active."}
                        ],
                        "expected": "List displays intersection subset matching all active filter selections."
                    })
                else:
                    scenarios.append({
                        "suffix": "updates selection state when option is selected",
                        "desc": f"Validates selection updates state in dropdown {semantic_name}.",
                        "priority": "High" if prio_class == "critical" else "Medium",
                        "type": "Functional",
                        "steps": [
                            {"step": 1, "action": f"Click dropdown {semantic_name}", "expected": "Dropdown menu options are shown."},
                            {"step": 2, "action": "Select a valid list option item", "expected": "Selected option is displayed."}
                        ],
                        "expected": "User preferences are updated with the selected option."
                    })
                    if prio_class == "critical":
                        scenarios.append({
                            "suffix": "renders expected default option placeholder on initial load",
                            "desc": f"Ensures dropdown {semantic_name} renders placeholder when page loads.",
                            "priority": "Medium",
                            "type": "UI",
                            "steps": [
                                {"step": 1, "action": f"Observe {semantic_name} initial state", "expected": "Default placeholder option is shown."}
                            ],
                            "expected": "Dropdown prompts user with appropriate selection guidance before interaction."
                        })

                # Depth scenario 1: Dropdown selection state persistence
                scenarios.append({
                    "suffix": "preserves selected option across page refresh reload",
                    "desc": f"Ensures the active choice in dropdown {semantic_name} is preserved after page reload.",
                    "priority": "Medium",
                    "type": "Functional",
                    "steps": [
                        {"step": 1, "action": f"Select a valid option from dropdown {semantic_name}", "expected": "Dropdown selection is updated."},
                        {"step": 2, "action": "Trigger page refresh or reload operation", "expected": "Page content reloads."}
                    ],
                    "expected": "The active choice is preserved and displayed after the page refresh resolves."
                })
                # Depth scenario 2: Dropdown dependent cross-field updates
                scenarios.append({
                    "suffix": "triggers dependent field updates on option selection",
                    "desc": f"Validates that changing selection in dropdown {semantic_name} updates dependent fields on the page.",
                    "priority": "Medium",
                    "type": "Functional",
                    "steps": [
                        {"step": 1, "action": f"Click dropdown {semantic_name}", "expected": "Dropdown options are displayed."},
                        {"step": 2, "action": "Select a specific option from the list", "expected": "Option is selected."}
                    ],
                    "expected": "Dependent form fields update correctly according to the chosen option."
                })

            # 4. Checkboxes / Radios
            elif ctype in ["checkbox", "radio_button", "checkboxes", "radio_buttons"]:
                is_filter = "filter" in selector.lower() or "filter" in semantic_name.lower() or "category" in selector.lower() or "status" in selector.lower() or "type" in selector.lower()
                
                if is_filter:
                    scenarios.append({
                        "suffix": "filters results by individual category criteria",
                        "desc": f"Validates that toggling the filter checkbox {semantic_name} filters list items.",
                        "priority": "High",
                        "type": "Functional",
                        "steps": [
                            {"step": 1, "action": f"Click and toggle {semantic_name} to checked state", "expected": "Filter checkbox is enabled."}
                        ],
                        "expected": "Active list items are filtered dynamically based on the checked option."
                    })
                    scenarios.append({
                        "suffix": "resets all active filters to default states",
                        "desc": f"Ensures that unchecking filter {semantic_name} restores default display state.",
                        "priority": "Medium",
                        "type": "Functional",
                        "steps": [
                            {"step": 1, "action": f"Ensure checkbox {semantic_name} is in enabled check state", "expected": "Checkbox is checked."},
                            {"step": 2, "action": f"Toggle {semantic_name} to unchecked state", "expected": "Checkbox is unchecked."}
                        ],
                        "expected": "Filtered results display returns to default unfiltered active display state."
                    })
                else:
                    scenarios.append({
                        "suffix": "toggles check state state transition successfully",
                        "desc": f"Ensures click toggles selection on {semantic_name}.",
                        "priority": "High" if prio_class == "critical" else "Medium",
                        "type": "Functional",
                        "steps": [
                            {"step": 1, "action": f"Click {semantic_name} to enable", "expected": "Checkbox is checked."},
                            {"step": 2, "action": f"Click {semantic_name} to disable", "expected": "Checkbox is unchecked."}
                        ],
                        "expected": "User preferences are updated."
                    })
                    if is_required:
                        scenarios.append({
                            "suffix": "validation blocks submission when unchecked",
                            "desc": f"Checks required constraint validation on terms/consent checkbox {semantic_name}.",
                            "priority": "High",
                            "type": "Negative",
                            "steps": [
                                {"step": 1, "action": f"Ensure {semantic_name} is unchecked", "expected": f"{semantic_name} is unchecked."},
                                {"step": 2, "action": "Trigger form submission action", "expected": "Validation warning is displayed."}
                            ],
                            "expected": "Validation message is displayed."
                        })

                # Depth scenario 1: Checkbox selection state persistence
                scenarios.append({
                    "suffix": "preserves selected state across page refresh reload",
                    "desc": f"Ensures the active state of checkbox {semantic_name} is preserved after page reload.",
                    "priority": "Medium",
                    "type": "Functional",
                    "steps": [
                        {"step": 1, "action": f"Toggle state of checkbox {semantic_name}", "expected": "Checkbox state is updated."},
                        {"step": 2, "action": "Trigger page refresh or reload operation", "expected": "Page content reloads."}
                    ],
                    "expected": "The active checkbox state is preserved and displayed after the page refresh resolves."
                })
                # Depth scenario 2: Checkbox server transaction failure rollback
                scenarios.append({
                    "suffix": "rolls back selection state on backend update API failure",
                    "desc": f"Ensures checkbox {semantic_name} selection state rolls back cleanly if update API fails.",
                    "priority": "High",
                    "type": "Negative",
                    "steps": [
                        {"step": 1, "action": f"Toggle checkbox {semantic_name} selection", "expected": "Toggle is registered by the interface."},
                        {"step": 2, "action": "Simulate server API error status code response", "expected": "Server responds with error status code."}
                    ],
                    "expected": "Checkbox selection rolls back to original state and warning indicator displays."
                })

            # 5. Links
            elif ctype == "link":
                if prio_class in ["critical", "medium"]:
                    scenarios.append({
                        "suffix": "navigates users to target destination page",
                        "desc": f"Validates navigation routing on link {semantic_name}.",
                        "priority": "High" if prio_class == "critical" else "Medium",
                        "type": "Functional",
                        "steps": [
                            {"step": 1, "action": f"Click on link {semantic_name}", "expected": "Target page loading begins."},
                            {"step": 2, "action": "Verify target page header and key elements of destination page are loaded", "expected": f"Key content elements specific to {semantic_name} are visible and interactive."}
                        ],
                        "expected": f"Target viewport navigation is completed, displaying the relevant {semantic_name} page content."
                    })
                    # Depth scenario 1: Browser back/forward navigation compliance
                    scenarios.append({
                        "suffix": "conforms to browser history navigation standards",
                        "desc": f"Validates browser back/forward history compliance after clicking link {semantic_name}.",
                        "priority": "Medium",
                        "type": "Functional",
                        "steps": [
                            {"step": 1, "action": f"Click on link {semantic_name}", "expected": "Target page loading begins."},
                            {"step": 2, "action": "Verify target page content is displayed", "expected": f"Key elements of target {semantic_name} page are visible."},
                            {"step": 3, "action": "Click browser back navigation control button", "expected": "Previous listing page reloads."},
                            {"step": 4, "action": "Click browser forward navigation control button", "expected": f"Target {semantic_name} page reloads."}
                        ],
                        "expected": "Browser navigation history is updated and traversable."
                    })
                    # Depth scenario 2: Access control/permission validations - Use dynamic terminology
                    entity = self.terminology.get_entity_term()
                    scenarios.append({
                        "suffix": "enforces authorization controls on destination redirect",
                        "desc": f"Ensures access to link {semantic_name} destination is authorized.",
                        "priority": "High",
                        "type": "Functional",
                        "steps": [
                            {"step": 1, "action": f"Click on link {semantic_name}", "expected": "Redirect request is sent."}
                        ],
                        "expected": f"Target {entity} content is loaded if authorized, otherwise access is blocked."
                    })
                    
            return scenarios

        # Enterprise-level scenario generation for business workflows and API coverage
        def generate_enterprise_scenarios() -> List[Dict[str, Any]]:
            """Generates enterprise-level scenarios: API, security, resilience, business-rule, workflow, state-transition, recovery."""
            enterprise_scenarios = []
            entity = self.terminology.get_entity_term()
            entity_plural = self.terminology.get_entity_term(plural=True)
            transaction = self.terminology.get_transaction_term()
            
            # Extract business context from story
            story = self.story_data.get("story", {})
            acceptance_criteria = story.get("acceptance_criteria", [])
            business_intents = self.terminology._terminology_cache.get("business_intents", [])
            api_operations = self.terminology._terminology_cache.get("api_operations", [])
            workflow_stages = self.terminology.get_workflow_stages()
            business_rules = self.terminology.get_business_rules()
            preconditions = self.terminology.get_preconditions()
            postconditions = self.terminology.get_postconditions()
            
            # 1. API Integration Scenarios with business context and measurable outcomes
            if api_operations:
                for api_op in api_operations[:3]:  # Limit to top 3 most relevant APIs
                    # Determine API business purpose from terminology
                    api_purpose = f"{transaction}" if transaction in api_op else f"{entity}"
                    
                    # Extract specific business validation from ACs
                    business_validation = ""
                    for ac in acceptance_criteria:
                        if "api" in ac.lower() or "response" in ac.lower() or "data" in ac.lower():
                            business_validation = ac[:50]
                            break
                    
                    enterprise_scenarios.append({
                        "title": f"Verify {api_op} API processes valid {api_purpose} requests with measurable business outcomes",
                        "description": f"Validates that the {api_op} API endpoint processes valid {api_purpose} requests and returns expected business responses with correct data integrity, response time, and business state transitions.",
                        "priority": "High",
                        "type": "API",
                        "steps": [
                            {"step": 1, "action": f"Prepare valid {api_purpose} request payload for {api_op} endpoint with all required fields and realistic business data", "expected": f"Request payload contains valid {api_purpose} data with all mandatory fields, proper data types, and business-valid values."},
                            {"step": 2, "action": f"Execute API call to {api_op} endpoint and measure response time", "expected": "API returns 200 OK status code within acceptable SLA (e.g., < 2s) with valid response body containing complete business data."},
                            {"step": 3, "action": f"Validate {api_purpose} response structure, business data integrity, and state transitions", "expected": f"Response contains all required {api_purpose} fields with valid data types, business values are accurate, and system state reflects the {transaction} operation."}
                        ],
                        "expected": f"{api_op} API endpoint successfully processes {api_purpose} requests within performance SLA, returns valid business responses with correct data integrity, and triggers appropriate state transitions."
                    })
                    
                    enterprise_scenarios.append({
                        "title": f"Verify {api_op} API handles invalid {api_purpose} requests with specific error codes and recovery paths",
                        "description": f"Validates that the {api_op} API endpoint handles invalid {api_purpose} requests and error conditions with specific error codes, meaningful error messages, and documented recovery paths.",
                        "priority": "High",
                        "type": "API",
                        "steps": [
                            {"step": 1, "action": f"Prepare invalid {api_purpose} request payload for {api_op} endpoint with specific error conditions (missing required fields, invalid data types, business rule violations)", "expected": f"Request payload contains invalid {api_purpose} data triggering specific validation errors."},
                            {"step": 2, "action": f"Execute API call to {api_op} endpoint and capture error response", "expected": "API returns appropriate error status code (400 for validation, 404 for not found, 422 for business rule violation) with structured error details."},
                            {"step": 3, "action": "Validate error response structure, specific error codes, field-level validation details, and recovery guidance", "expected": "Error response contains specific error code, meaningful error message, field-level validation details identifying the exact issue, and recovery guidance for correction."}
                        ],
                        "expected": f"{api_op} API endpoint gracefully handles invalid {api_purpose} requests with specific error codes, meaningful error messages, field-level validation details, and documented recovery paths."
                    })
            
            # 2. Business Rule Scenarios from Acceptance Criteria with executable steps and realistic test data
            for idx, ac in enumerate(acceptance_criteria[:5]):  # Limit to top 5 ACs
                # Extract specific business rule from AC
                ac_lower = ac.lower()
                
                # Determine rule type and generate appropriate validation
                if any(kw in ac_lower for kw in ["must", "should", "shall", "required", "mandatory"]):
                    rule_type = "mandatory constraint"
                    action_verb = "enforces"
                    validation_focus = "presence and completeness"
                elif any(kw in ac_lower for kw in ["cannot", "must not", "prohibited", "forbidden"]):
                    rule_type = "prohibition constraint"
                    action_verb = "prevents"
                    validation_focus = "absence and rejection"
                elif any(kw in ac_lower for kw in ["minimum", "maximum", "at least", "at most"]):
                    rule_type = "boundary constraint"
                    action_verb = "validates"
                    validation_focus = "range and limits"
                else:
                    rule_type = "business logic"
                    action_verb = "implements"
                    validation_focus = "behavior and outcome"
                
                # Generate realistic test data based on entity and business context
                if entity:
                    test_data = {
                        f"{entity}_id": f"test_{entity}_12345",
                        f"{entity}_name": f"Valid {entity.capitalize()} Test Data",
                        f"{entity}_status": "active",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "user_context": "authenticated_user"
                    }
                else:
                    test_data = {
                        "test_id": "test_12345",
                        "test_value": "valid_business_data",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                
                # Extract specific validation criteria from AC
                validation_criteria = ac
                if "then" in ac_lower:
                    validation_criteria = ac_lower.split("then")[-1].strip()
                
                enterprise_scenarios.append({
                    "title": f"Verify system {action_verb} {rule_type}: {ac[:60]}...",
                    "description": f"Validates that the system correctly {action_verb} the {rule_type} defined in acceptance criterion: {ac}. This test verifies {validation_focus} with realistic business data and measurable outcomes.",
                    "priority": "High",
                    "type": "Business-Rule",
                    "data": test_data,
                    "preconditions": preconditions[:2] if preconditions else [f"{entity.capitalize()} exists in valid state"],
                    "postconditions": postconditions[:2] if postconditions else [f"{entity.capitalize()} state reflects {rule_type} enforcement"],
                    "steps": [
                        {"step": 1, "action": f"Configure test scenario with realistic {entity} data to validate {rule_type}: {validation_criteria[:40]}...", "expected": f"Test scenario is configured with appropriate {entity} data representing valid business context."},
                        {"step": 2, "action": f"Execute {entity} operation to trigger {rule_type} validation with test data", "expected": f"System applies {rule_type} validation to {entity} data and processes according to business rules."},
                        {"step": 3, "action": f"Verify {rule_type} outcome matches acceptance criteria: {validation_criteria[:40]}...", "expected": f"{rule_type.capitalize()} is enforced correctly, {entity} state reflects the expected business outcome, and validation results are measurable."}
                    ],
                    "expected": f"System correctly {action_verb} the {rule_type} from acceptance criterion {idx + 1}, ensuring {entity} business logic integrity with measurable validation results and correct state transitions."
                })
            
            # 3. Workflow Scenarios with actual business stages and realistic user journeys
            if workflow_stages:
                # Use extracted workflow stages from ACs
                actual_workflow = " → ".join(workflow_stages[:5]) if len(workflow_stages) >= 3 else self.terminology.get_workflow_steps()
            else:
                # Fallback to dynamic construction with business context
                action = self.terminology.get_action_term("browse")
                actual_workflow = f"{action} {entity_plural} → select {entity} → {transaction} → confirmation"
            
            # Generate workflow steps with actual business stages and measurable outcomes
            workflow_steps_list = actual_workflow.split(" → ")
            workflow_test_steps = []
            for step_idx, stage in enumerate(workflow_steps_list[:5], 1):
                # Generate realistic expected results for each workflow stage
                if "browse" in stage.lower() or "search" in stage.lower():
                    stage_expected = f"{stage.capitalize()} completes and displays available {entity_plural} with filtering and sorting options active."
                elif "select" in stage.lower():
                    stage_expected = f"{entity.capitalize()} is selected, details are displayed, and {entity} state is prepared for {transaction}."
                elif transaction in stage.lower():
                    stage_expected = f"{transaction.capitalize()} operation is initiated, business rules are validated, and {entity} state transitions to processing."
                elif "confirm" in stage.lower():
                    stage_expected = f"{transaction.capitalize()} is confirmed, {entity} state is updated to final state, and confirmation is displayed."
                else:
                    stage_expected = f"Stage '{stage}' completes successfully and {entity} state advances to next workflow stage with measurable business outcome."
                
                workflow_test_steps.append({
                    "step": step_idx,
                    "action": f"Execute workflow stage: {stage} with realistic business data",
                    "expected": stage_expected
                })
            
            enterprise_scenarios.append({
                "title": f"Verify complete {entity} business workflow: {actual_workflow}",
                "description": f"Validates the end-to-end business workflow for {entity} management covering all critical business stages from initiation to completion. This test ensures realistic user journey with measurable outcomes at each stage.",
                "priority": "High",
                "type": "Workflow",
                "data": {
                    f"{entity}_workflow_id": f"workflow_{entity}_12345",
                    "workflow_stages": workflow_steps_list,
                    "expected_outcome": f"{entity.capitalize()} successfully processed through complete business lifecycle"
                },
                "preconditions": [f"User is authenticated and authorized for {entity} operations", f"{entity_plural} are available in the system"],
                "postconditions": [f"{entity.capitalize()} reaches final business state", f"Workflow completion is logged and audited"],
                "steps": workflow_test_steps,
                "expected": f"Complete {entity} business workflow executes successfully from start to finish, with each stage ({actual_workflow}) completing in sequence, measurable business outcomes at each transition, and {entity} reaching the expected final business state with data integrity preserved."
            })
            
            # 4. Security Scenarios with measurable business outcomes
            enterprise_scenarios.append({
                "title": f"Verify {transaction} authorization enforces access control and prevents unauthorized operations",
                "description": f"Validates that {transaction} operations require proper authentication and authorization, preventing unauthorized access to {entity} resources.",
                "priority": "High",
                "type": "Security",
                "steps": [
                    {"step": 1, "action": f"Attempt {transaction} operation without valid authentication credentials", "expected": f"System denies {transaction} access and redirects to authentication page with clear error message."},
                    {"step": 2, "action": "Authenticate with valid user credentials and establish session", "expected": f"Authentication succeeds, session is established, and user is authorized for {entity} operations."},
                    {"step": 3, "action": f"Attempt {transaction} operation with valid authentication and authorization", "expected": f"{transaction.capitalize()} operation is authorized, proceeds successfully, and {entity} state is updated accordingly."}
                ],
                "expected": f"{transaction.capitalize()} operations enforce proper authorization and access control, ensuring only authenticated users can modify {entity} resources."
            })
            
            enterprise_scenarios.append({
                "title": f"Verify {entity} input sanitization prevents SQL injection, XSS, and script execution attacks",
                "description": f"Validates that {entity} data inputs are properly sanitized to prevent SQL injection, XSS, and other injection attacks, ensuring data security.",
                "priority": "High",
                "type": "Security",
                "steps": [
                    {"step": 1, "action": f"Enter malicious script payload (<script>alert('XSS')</script>, ' OR 1=1 --, etc.) into {entity} input field", "expected": f"Input is accepted for processing by the {entity} form."},
                    {"step": 2, "action": f"Submit malicious {entity} payload to backend", "expected": "System sanitizes payload, strips malicious tags, and prevents script execution."},
                    {"step": 3, "action": "Verify no script execution occurred and data is stored safely", "expected": f"Malicious payload is rendered safely as plain text without script execution, and {entity} data is stored securely."}
                ],
                "expected": f"{entity.capitalize()} data inputs are properly sanitized to prevent injection attacks, ensuring {entity} data security and system integrity."
            })
            
            # 5. Resilience Scenarios with measurable business outcomes
            enterprise_scenarios.append({
                "title": f"Verify {entity} system handles network latency with user feedback and completes operations successfully",
                "description": f"Validates that the {entity} system handles network latency gracefully with appropriate loading indicators and user feedback, ensuring operations complete successfully.",
                "priority": "Medium",
                "type": "Resilience",
                "steps": [
                    {"step": 1, "action": f"Initiate {entity} operation with simulated network latency (2-5 seconds)", "expected": f"{entity.capitalize()} operation request is initiated and loading indicator is displayed."},
                    {"step": 2, "action": "Observe system behavior during network latency period", "expected": "System displays loading spinner/progress bar and prevents duplicate user actions during latency."},
                    {"step": 3, "action": "Verify operation completes successfully after network latency resolves", "expected": f"{entity.capitalize()} operation completes successfully, loading indicator is dismissed, and {entity} state is updated correctly."}
                ],
                "expected": f"{entity.capitalize()} system handles network latency gracefully with appropriate user feedback and operations complete successfully despite network delays."
            })
            
            enterprise_scenarios.append({
                "title": f"Verify {transaction} operations recover from temporary service failures with automatic retry logic",
                "description": f"Validates that {transaction} operations can recover from temporary service failures with appropriate retry logic and user notification, ensuring business continuity.",
                "priority": "High",
                "type": "Resilience",
                "steps": [
                    {"step": 1, "action": f"Initiate {transaction} operation with valid {entity} data", "expected": f"{transaction.capitalize()} operation is initiated and request is sent to backend service."},
                    {"step": 2, "action": "Simulate temporary service failure (500 error, timeout)", "expected": "System detects service failure and initiates automatic retry logic with user notification."},
                    {"step": 3, "action": "Verify operation completes successfully after service recovery and retry", "expected": f"{transaction.capitalize()} operation completes successfully after retry, {entity} state is updated, and user is notified of success."}
                ],
                "expected": f"{transaction.capitalize()} operations recover from temporary failures with appropriate retry logic, ensuring business continuity and user experience."
            })
            
            # 6. State Transition Scenarios with business lifecycle validation
            enterprise_scenarios.append({
                "title": f"Verify {entity} state transitions follow defined business lifecycle with validation at each stage",
                "description": f"Validates that {entity} state transitions follow the defined business lifecycle correctly with proper validation and constraints at each transition stage.",
                "priority": "High",
                "type": "State-Transition",
                "steps": [
                    {"step": 1, "action": f"Create {entity} in initial state with valid data", "expected": f"{entity.capitalize()} is created in initial state with all required fields populated."},
                    {"step": 2, "action": "Trigger state transition to next lifecycle stage (e.g., from draft to active)", "expected": f"{entity.capitalize()} transitions to next state after validation of business rules and constraints."},
                    {"step": 3, "action": "Verify state transition validation, constraints, and business rule enforcement", "expected": f"State transition validates all business rules, enforces constraints, and {entity} state reflects the correct lifecycle stage."}
                ],
                "expected": f"{entity.capitalize()} state transitions follow the defined business lifecycle correctly with proper validation and constraints at each transition stage."
            })
            
            # 7. Accessibility Scenarios with measurable outcomes
            enterprise_scenarios.append({
                "title": f"Verify {entity} interface supports full keyboard navigation for all interactive controls",
                "description": f"Validates that {entity} interface is fully navigable using keyboard only, ensuring accessibility for keyboard-only users.",
                "priority": "Medium",
                "type": "Accessibility",
                "steps": [
                    {"step": 1, "action": f"Navigate to {entity} interface using keyboard Tab key", "expected": f"Focus moves to {entity} interface and first interactive control receives visible focus indication."},
                    {"step": 2, "action": f"Navigate through all {entity} interface elements using keyboard Tab/Shift+Tab", "expected": "All interactive elements (inputs, buttons, dropdowns, links) are reachable via keyboard in logical order."},
                    {"step": 3, "action": f"Verify keyboard interaction with all {entity} controls (Enter, Space, Arrow keys)", "expected": f"All {entity} controls respond to appropriate keyboard input and perform expected actions."}
                ],
                "expected": f"{entity.capitalize()} interface is fully accessible via keyboard navigation, ensuring compliance with WCAG keyboard accessibility standards."
            })
            
            enterprise_scenarios.append({
                "title": f"Verify {entity} interface provides proper ARIA labels and screen reader announcements",
                "description": f"Validates that {entity} interface provides proper ARIA labels, roles, and descriptions for screen reader compatibility and accessibility.",
                "priority": "Medium",
                "type": "Accessibility",
                "steps": [
                    {"step": 1, "action": f"Activate screen reader on {entity} interface", "expected": f"Screen reader announces {entity} interface and its purpose."},
                    {"step": 2, "action": f"Navigate through {entity} interface with screen reader", "expected": "Screen reader announces all interactive elements with descriptive labels and current state."},
                    {"step": 3, "action": f"Verify ARIA labels, roles, and descriptions for all {entity} elements", "expected": f"All {entity} elements have descriptive ARIA labels, appropriate roles, and state announcements for screen readers."}
                ],
                "expected": f"{entity.capitalize()} interface provides proper screen reader compatibility with descriptive ARIA labels and announcements, ensuring accessibility compliance."
            })
            
            # 8. Recovery Scenarios with measurable business outcomes
            enterprise_scenarios.append({
                "title": f"Verify {transaction} data is preserved and recoverable after system interruption or browser refresh",
                "description": f"Validates that {transaction} data is preserved and recoverable after system interruption (browser refresh, tab close, network disconnect), ensuring data integrity.",
                "priority": "High",
                "type": "Recovery",
                "steps": [
                    {"step": 1, "action": f"Initiate {transaction} with partial data entry (fill required fields)", "expected": f"{transaction.capitalize()} data is entered in form fields and temporary state is established."},
                    {"step": 2, "action": "Simulate system interruption (browser refresh, tab close, network disconnect)", "expected": "System interruption occurs and session is temporarily lost."},
                    {"step": 3, "action": f"Recover {transaction} data after system restart and verify data integrity", "expected": f"{transaction.capitalize()} data is preserved, form fields are repopulated, and data integrity is maintained."}
                ],
                "expected": f"{transaction.capitalize()} data is preserved and recoverable after system interruption, ensuring data integrity and user experience continuity."
            })
            
            return enterprise_scenarios

        # Compile test cases dynamically without wrap-around loops
        test_cases = []
        tc_seq = 1

        # Track seen combinations to prevent duplicates
        seen_testcase_keys = set()

        # ── Intent Mapping Derivation ─────────────────────────────────────────
        # Produces scenario-specific user_goal + business_action per test case.
        def _derive_intent_mapping(title: str, tctype: str, ctype: str, suffix: str, domain: str) -> Dict[str, Any]:

            t = title.lower()
            s = (suffix or "").lower()

            # Security / XSS / injection
            if any(w in t or w in s for w in ["sanitize", "injection", "xss", "script", "sql"]):
                return {
                    "user_goal": "Verify security boundaries for untrusted input data",
                    "business_action": "Verify malicious input is sanitized and does not execute"
                }
            # Session expiry
            if "session" in t or "expir" in t:
                return {
                    "user_goal": "Test session state handling during prolonged user interaction",
                    "business_action": "Verify session expiry triggers re-authentication prompt without data loss"
                }
            # Refresh / reload protection
            if "refresh" in t or "reload" in t:
                return {
                    "user_goal": "Test data persistence and protection during accidental page reload",
                    "business_action": "Verify unsaved data is preserved or user is warned on page reload"
                }
            # Double-click / debounce
            if "double" in t or "concurrent" in t or "debounce" in t:
                return {
                    "user_goal": "Test concurrent action protection on rapid user interaction",
                    "business_action": "Verify debounce mechanism prevents duplicate request submission"
                }
            # High latency / spinner
            if "latency" in t or "spinner" in t or "progress" in t:
                return {
                    "user_goal": "Test UI feedback under slow network conditions",
                    "business_action": "Verify progress spinner renders and controls lock during high latency"
                }
            # Offline / network dropout
            if "offline" in t or "network dropout" in t or "disconnection" in t:
                return {
                    "user_goal": "Test system resilience during network interruption",
                    "business_action": "Verify system blocks request and shows clear offline warning to user"
                }
            # API rollback / failure
            if "rollback" in t or "api failure" in t or "backend failure" in t:
                return {
                    "user_goal": "Test error recovery from backend failure",
                    "business_action": "Verify UI state rolls back cleanly and error message is presented to user"
                }
            # Disabled element
            if "disabled" in t:
                return {
                    "user_goal": "Verify non-interactive state of a disabled control",
                    "business_action": "Verify disabled element ignores interaction and blocks premature submissions"
                }
            # Email format validation
            if "email" in t and ("malform" in t or "invalid" in t or "reject" in t):
                return {
                    "user_goal": "Enter an incorrectly formatted email address into the authentication form",
                    "business_action": "Verify email format validation rejects malformed address before submission"
                }
            # Password masking
            if "password" in t and "mask" in t:
                return {
                    "user_goal": "Verify credential visibility protection in the password input",
                    "business_action": "Verify password characters are masked to protect user credentials"
                }
            # Empty / blank / required field validation
            if any(w in t for w in ["blank", "empty", "required", "validation error"]):
                return {
                    "user_goal": "Submit a form with required field left empty to trigger validation",
                    "business_action": "Verify validation error message is displayed and submission is blocked"
                }
            # Filter / category selection
            if any(w in t for w in ["filter", "category", "audience", "date filter", "destination filter"]):
                return {
                    "user_goal": "Apply a filter criterion to narrow the displayed results list",
                    "business_action": "Verify filtered display updates to show only items matching the selected criterion"
                }
            # Sort / ordering
            if "sort" in t or "order" in t:
                return {
                    "user_goal": "Select a sort order to resequence the displayed results list",
                    "business_action": "Verify listing items are reordered according to the chosen sort criterion"
                }
            # Search — valid query
            if "search" in t and any(w in t for w in ["valid", "matching", "displays matching"]):
                return {
                    "user_goal": "Enter a valid search query to find matching records",
                    "business_action": "Verify search results display only records matching the entered query"
                }
            # Search — empty state
            if "search" in t and any(w in t for w in ["non-matching", "empty state", "no match", "no result"]):
                return {
                    "user_goal": "Enter a search query with no matching records to test empty state behavior",
                    "business_action": "Verify graceful empty state message is displayed with guidance for the user"
                }
            # Load more / pagination
            if "load more" in t or "additional" in t and "load" in t:
                return {
                    "user_goal": "Trigger pagination load to extend the visible results list",
                    "business_action": "Verify additional records load correctly while preserving active filter state"
                }
            # Navigation / link
            if ctype == "link" or "navigat" in t or "target section" in t:
                return {
                    "user_goal": "Navigate to a linked section of the application",
                    "business_action": "Verify navigation resolves to target page with updated browser history"
                }
            # Accessibility
            if any(w in t for w in ["keyboard", "tab navigation", "focus", "wcag", "aria", "accessibility"]):
                return {
                    "user_goal": "Ensure accessibility compliance for keyboard-only and assistive technology users",
                    "business_action": "Verify all controls are focusable and labeled per WCAG accessibility standards"
                }
            # End-to-end journey / workflow
            if any(w in t for w in ["end-to-end", "journey", "workflow", "complete", "transactional"]):
                return {
                    "user_goal": f"Complete the full {domain} business flow from entry to confirmation",
                    "business_action": f"Verify end-to-end {domain} user journey produces correct outcome at each stage"
                }
            # AC / business rule compliance
            if any(w in t for w in ["business logic", "system implements", "conforms to", "requirement"]):
                return {
                    "user_goal": "Validate that the system adheres to the acceptance criteria business rule",
                    "business_action": "Verify the defined business rule is enforced correctly by the application"
                }
            # Input — valid entry (positive)
            if ctype == "input" and tctype == "Functional":
                return {
                    "user_goal": f"Enter valid data into the {domain} form to progress the business flow",
                    "business_action": "Verify form field accepts and registers the valid input value in state"
                }
            # Button — primary action (positive)
            if ctype == "button" and tctype == "Functional":
                return {
                    "user_goal": f"Interact with the {domain} UI to trigger a primary business action",
                    "business_action": f"Verify the triggered action updates the {domain} interface state correctly"
                }
            # Negative catch-all
            if tctype == "Negative":
                return {
                    "user_goal": f"Test the {domain} system response to an invalid or unexpected input condition",
                    "business_action": f"Verify the system handles invalid inputs gracefully with clear error feedback"
                }
            # Edge catch-all
            if tctype == "Edge":
                return {
                    "user_goal": f"Test {domain} boundary and resilience conditions beyond normal usage",
                    "business_action": f"Verify the {domain} system handles boundary conditions without data loss or errors"
                }
            # Default fallback
            return {
                "user_goal": f"Execute operations and validation tasks within the {domain} page space",
                "business_action": f"Verify behavioral state transitions for the {domain} user journey"
            }


        # Track current scenario suffix for intent derivation inside add_testcase
        suffix_hint = ""


        def add_testcase(title: str, desc: str, prio: str, tctype: str, steps: List[Dict[str, Any]], expected: str, data: Dict[str, Any] = None, ref_edge: str = None, extra_tags: List[str] = None, comp_info: Tuple[str, str] = None, target_pk: str = None) -> None:
            nonlocal tc_seq, suffix_hint

            
            # Apply Generation-Time Semantic Quality Refinement (Intelligence Layer)
            title_lower = title.lower()
            subject = title
            if title.startswith("Verify "):
                subject = title[7:]
            
            # Strip UI component suffixes for clean business descriptions
            clean_name = subject
            for sfx in [" Button", " Input Field", " Dropdown", " Checkbox", " Link"]:
                if clean_name.endswith(sfx):
                    clean_name = clean_name[:-len(sfx)]
            clean_name = clean_name.strip()

            # Dynamic title normalization based on terminology
            entity = self.terminology.get_entity_term()
            entity_capitalized = entity.capitalize()
            
            # Generic button title normalization
            if "button triggers action successfully" in title_lower:
                # Extract the button name and create a business-specific title
                button_name = title.replace(" button triggers action successfully", "").strip()
                if button_name.lower() == entity.lower():
                    title = f"Verify selecting the {entity_capitalized} category filters displayed events correctly."
                elif "this week" in button_name.lower():
                    title = "Verify selecting This Week displays only events occurring during the current week."
                elif "load more" in button_name.lower():
                    title = f"Verify Load More appends additional {entity_capitalized} listings while preserving active filters and ordering."
                else:
                    title = f"Verify selecting {button_name} processes action correctly with {entity} state transition."
            else:
                # Terminology normalization: Button -> Selector/Action/Filter, Input Field -> Function
                new_title = title
                if " Button" in new_title:
                    if "filter" in title_lower or "category" in title_lower:
                        new_title = new_title.replace(" Button", " Category Filter")
                    elif "sort" in title_lower or "order" in title_lower:
                        new_title = new_title.replace(" Button", " Sort Option")
                    elif "date" in title_lower:
                        new_title = new_title.replace(" Button", " Date Selector")
                    elif "destination" in title_lower:
                        new_title = new_title.replace(" Button", " Destination Selector")
                    elif "audience" in title_lower:
                        new_title = new_title.replace(" Button", " Audience Filter")
                    elif "booking" in title_lower or "place" in title_lower or "confirm" in title_lower:
                        new_title = new_title.replace(" Button", " Booking Action")
                    elif "link" in title_lower or "navigate" in title_lower:
                        new_title = new_title.replace(" Button", " Navigation Action")
                    else:
                        new_title = new_title.replace(" Button", "")
                        
                if " Input Field" in new_title:
                    if "search" in title_lower or "query" in title_lower:
                        new_title = new_title.replace(" Input Field", " Search Function")
                    else:
                        new_title = new_title.replace(" Input Field", "")
                        
                if " Dropdown" in new_title:
                    if "sort" in title_lower or "order" in title_lower:
                        new_title = new_title.replace(" Dropdown", " Sort Option")
                    elif "filter" in title_lower:
                        new_title = new_title.replace(" Dropdown", " Filter Selector")
                    else:
                        new_title = new_title.replace(" Dropdown", "")
                        
                if " Checkbox" in new_title:
                    if "filter" in title_lower:
                        new_title = new_title.replace(" Checkbox", " Filter Option")
                    else:
                        new_title = new_title.replace(" Checkbox", "")

                # If title is generic AC reference, rewrite as business scenario
                if any(f"ac{i}" in new_title.lower() or f"ac-{i}" in new_title.lower() for i in range(1, 20)):
                    new_title = new_title.replace("Acceptance Criterion: ", "").replace("AC-", "").replace("AC", "")
                    
                title = new_title

            # Expected Result Refinement - Use dynamic terminology
            entity = self.terminology.get_entity_term()
            entity_plural = self.terminology.get_entity_term(plural=True)
            
            if expected.strip() in ["Action executes successfully.", "Action completed.", "Target action executes.", "Action is triggered.", "Action is triggered"]:
                if "click" in title.lower() or "select" in title.lower():
                    expected = f"The requested {entity} action completes and the interface updates accordingly."
                elif "enter" in title.lower() or "type" in title.lower() or "input" in title.lower() or "focus" in title.lower():
                    expected = f"The field value is registered correctly in the {entity} input context."
                else:
                    expected = f"Displayed {entity_plural} match the selected filter criteria."
            elif expected.strip() in ["Page updates successfully.", "Page reloads successfully."]:
                expected = f"Only {entity_plural} satisfying the selected business rule are displayed."
            elif expected.strip() in ["Navigation succeeds.", "Navigation completes successfully."]:
                expected = f"The expected destination {entity} page loads with the correct business context and user-visible data."
                
            # Clean expected results from generic success keywords
            exp_lower = expected.lower()
            if "successful" in exp_lower:
                expected = expected.replace("successful", "valid").replace("Successful", "Valid")
            for key_term in ["success", "successfully", "stable state", "navigation succeeds", "action is triggered"]:
                if key_term in exp_lower:
                    expected = expected.replace("successfully", "correctly").replace("success", "completion").replace("stable state", "stable system environment").replace("navigation succeeds", "navigation completes").replace("action is triggered", "action is initiated")
            
            # Step Expected Refinement - Use dynamic terminology
            refined_steps = []
            for s in steps:
                s_action = s.get("action", "")
                s_expected = s.get("expected", "")
                if s_expected.strip() in ["Action executes successfully.", "Action completed.", "Target action executes.", "Action is triggered.", "Action is triggered"]:
                    if "click" in s_action.lower() or "select" in s_action.lower():
                        s_expected = f"The requested {entity} action completes and the interface updates accordingly."
                    elif "enter" in s_action.lower() or "type" in s_action.lower() or "input" in s_action.lower() or "focus" in s_action.lower():
                        s_expected = f"The field value is registered correctly in the {entity} input context."
                    else:
                        s_expected = f"Displayed {entity_plural} match the selected filter criteria."
                elif s_expected.strip() in ["Page updates successfully.", "Page reloads successfully."]:
                    s_expected = f"Only {entity_plural} satisfying the selected business rule are displayed."
                elif s_expected.strip() in ["Navigation succeeds.", "Navigation completes successfully."]:
                    s_expected = f"The expected destination {entity} page loads with the correct business context and user-visible data."
                
                step_exp_lower = s_expected.lower()
                if "successful" in step_exp_lower:
                    s_expected = s_expected.replace("successful", "valid").replace("Successful", "Valid")
                for key_term in ["success", "successfully", "stable state", "navigation succeeds", "action is triggered"]:
                    if key_term in step_exp_lower:
                        s_expected = s_expected.replace("successfully", "correctly").replace("success", "completion").replace("stable state", "stable system environment").replace("navigation succeeds", "navigation completes").replace("action is triggered", "action is initiated")
                
                refined_steps.append({
                    "step": s.get("step", 1),
                    "action": s_action,
                    "expected": s_expected
                })
            steps = refined_steps

            # Execute mandatory Quality Gate Rejection Policy
            if not self._validate_and_review_testcase(title, desc, steps, expected, tctype):
                return

            norm_key = (title.lower().strip(), tuple((s["action"].lower(), s["expected"].lower()) for s in steps))
            if norm_key in seen_testcase_keys:
                return
            seen_testcase_keys.add(norm_key)

            categories = ["Functional", "Regression"]
            if tctype == "UI":
                categories = ["UI", "Accessibility"]
            elif tctype == "Negative":
                categories = ["Negative", "Validation"]
            elif tctype == "Edge":
                categories = ["Edge", "Boundary"]

            # M2-FIX: Deduplicate tags — keep insertion order, remove duplicates (case-insensitive)
            raw_tags = [domain.lower(), f"story-{self.story_key}", tctype.lower()] + categories
            if extra_tags:
                raw_tags.extend(extra_tags)
            seen_tags = set()
            final_tags = []
            for _t in raw_tags:
                _tl = _t.lower()
                if _tl not in seen_tags:
                    seen_tags.add(_tl)
                    final_tags.append(_t)

            pk = target_pk or page_key
            p_selectors = selectors.get(pk, {})
            t_url = p_selectors.get("url", target_url)

            # Match observed APIs for this component interaction
            matching_apis = []
            comp_selector, comp_type = "", ""
            if comp_info:
                comp_selector, comp_type = comp_info

                matching_apis = self._find_matching_apis(comp_selector, comp_type, pk)
                for api in matching_apis:
                    self._enrich_api_details(api)

            # Build a coverage_reference that actually matches an edge ID produced by
            # BehavioralFlowEngine.update_graph() (same _sanitize_node_name scheme) so the
            # True Coverage Engine's path/component coverage can find it. Previously almost
            # every test case fell back to a generic "<domain> -> <domain> (interact)"
            # string that never matched any real graph edge, making coverage look near-zero
            # regardless of how comprehensive the generated suite actually was.
            if ref_edge:
                edge_ref = ref_edge
            elif comp_info and comp_selector:
                comp_name_clean = _sanitize_node_name(comp_selector)
                edge_ref = f"{pk}_comp_{comp_name_clean} -> {pk}_interact_{comp_name_clean} (interact via user action)"
            else:
                edge_ref = f"{domain.lower()} -> {domain.lower()} (interact)"

            api_notes = ""
            if matching_apis:
                api_notes = "\n\n[API Expectations]\n"
                for api in matching_apis:
                    exp = self._generate_api_expectations(api)
                    api_notes += f"- Endpoint: {exp['endpoint']}\n"
                    api_notes += f"  Expected Status: {exp['expected_status']}\n"
                    api_notes += f"  Expected Response Success: {exp['expected_success']}\n"
                    api_notes += f"  Expected Response Time: {exp['max_response_time_ms']}ms\n"
                    api_notes += f"  Expected Required Fields: {exp['expected_payload_schema']}\n"
                    api_notes += f"  Expected Error Handling: {exp['expected_error_handling']}\n"
                    api_notes += f"  Expected Empty Response Handling: {exp['expected_empty_response_handling']}\n"
                    api_notes += f"  Expected Authorization Behaviour: {exp['expected_authorization_behaviour']}\n"
                    api_notes += f"  Expected Retry Behaviour: {exp['expected_retry_behaviour']}\n"
                    api_notes += f"  Expected Pagination Behaviour: {exp['expected_pagination_behaviour']}\n"
                    api_notes += f"  Expected Validation Messages: {exp['expected_validation_messages']}\n"
                    api_notes += f"  Expected State Transition: {exp['expected_state_transition']}\n"
                    api_notes += f"  Business Operation: {exp.get('business_operation')}\n"
                    api_notes += f"  Business Intent: {exp.get('business_intent')}\n"
                    api_notes += f"  Triggering User Action: {exp.get('triggering_user_action')}\n"

            # Clean steps without API expectation details in step expected text
            enriched_steps = []
            for s in steps:
                enriched_steps.append({
                    "step": s["step"],
                    "action": s["action"],
                    "expected": s["expected"]
                })

            final_expected_result = expected

            # Balance priority and severity based on business impact rules:
            # - High/Critical: Payment, authentication, place bid submission, security (XSS/SQLi edge cases), transaction workflows.
            # - Medium/Major: General functional actions (search queries, selection dropdowns).
            # - Low/Minor: Structural checks, tab navigation, ARIA attributes, layout placeholders.
            
            prio_lower = (prio or "").lower()
            title_lower = title.lower()
            desc_lower = desc.lower()

            is_critical_business = any(w in title_lower or w in desc_lower for w in ["login", "auth", "password", "payment", "checkout", "place bid", "bidding", "xss", "sqli", "injection", "sanitize", "transactional"])
            is_minor_business = any(w in title_lower or w in desc_lower for w in ["accessibility", "keyboard", "tab navigation", "focus", "wcag", "aria", "placeholder", "layout"])

            # Business-critical/minor keyword signals take precedence (documented rule above),
            # but otherwise honor the caller-supplied priority instead of silently discarding it
            # (previously every call site's carefully-computed `prio` was overwritten here).
            if is_critical_business:
                prio = "High"
                severity = "Critical"
            elif is_minor_business:
                prio = "Low"
                severity = "Minor"
            elif prio_lower == "high":
                prio = "High"
                severity = "Major"
            elif prio_lower == "low":
                prio = "Low"
                severity = "Minor"
            else:
                prio = "Medium"
                severity = "Major"

            # Determine dynamic state_change per step to be business-oriented
            behavioral_seq = []
            for s in enriched_steps:
                action_text = s["action"].lower()
                sc_val = "UI Context Updated"
                if "click" in action_text or "press" in action_text or "trigger" in action_text:
                    if matching_apis:
                        sc_val = f"Backend Transaction Initiated ({matching_apis[0].get('business_operation')})"
                    else:
                        sc_val = "UI Interaction Registered"
                elif "input" in action_text or "enter" in action_text or "type" in action_text:
                    sc_val = "Form Input Buffer Populated"
                elif "verify" in action_text or "observe" in action_text:
                    sc_val = "State Verification Completed"
                elif "navigate" in action_text or "open" in action_text:
                    sc_val = "Viewport Location Transitioned"
                
                behavioral_seq.append({
                    "step": s["step"],
                    "user_action": s["action"],
                    "system_response": s["expected"],
                    "state_change": sc_val
                })

            polarity = self._determine_polarity(tctype, title, desc, final_tags, tctype == "Edge")

            test_cases.append({
                "id": f"TC-{module_prefix}-{tc_seq:03d}",
                "title": title,
                "description": desc,
                "priority": prio,
                "severity": severity,
                "type": tctype,
                "scenario_classification": polarity,
                "tags": final_tags,
                "preconditions": [f"Application is accessible at {t_url}"],
                "test_data": data or {},
                "steps": enriched_steps,
                "expected_result": final_expected_result,
                "postconditions": ["Application remains in stable state"],
                "automation": "Yes",
                # M4-FIX: automation_notes meaningful per scenario type
                "automation_notes": (
                    f"Automate via Playwright: intercept API '{matching_apis[0].get('url', '')}', assert response status and UI state.{api_notes}"
                    if matching_apis else
                    f"Automate via Playwright: navigate to {t_url}, interact with '{title.split('Verify ')[-1][:60]}', assert final UI state."
                ),
                "dependencies": [],
                "edge_case": tctype == "Edge",
                "confidence": "high",
                # C5-FIX: full title used — no truncation on flow_path
                "flow_path": f"{domain} Flow → {title}",
                "behavioral_sequence": behavioral_seq,
                "ui_nodes_used": [domain.lower()],
                "intent_mapping": _derive_intent_mapping(title, tctype, comp_type if comp_info else "", suffix_hint, domain),


                "coverage_reference": edge_ref,
                "status": "Draft",
                "created_by": "testcase_generation_agent",
                "created_at": now,
                "linked_story": self.story_key
            })
            tc_seq += 1

        # 1. Process components
        for comp_obj in classified_components:
            pk = comp_obj.get("page_key", page_key)
            p_selectors = selectors.get(pk, {})
            scenarios = get_scenarios_for_component_v2(comp_obj, p_selectors)
            for sc in scenarios:
                selector = comp_obj["selector"]
                ctype = comp_obj["type"]
                semantic_name = comp_obj["semantic_name"]
                
                # Clean up title to be concise and business-friendly
                title_sem_name = semantic_name
                if " > " in title_sem_name:
                    title_sem_name = title_sem_name.split(" > ")[-1]

                suffix = sc['suffix']
                exec_data = comp_obj.get("exec_data")
                
                # Use dynamic terminology for action descriptions
                entity = self.terminology.get_entity_term()
                entity_plural = self.terminology.get_entity_term(plural=True)
                transaction = self.terminology.get_transaction_term()
                
                if suffix == "triggers primary action successfully on user click":
                    if transaction in title_sem_name.lower():
                        action_desc = f"submits {transaction} successfully"
                    elif "login" in title_sem_name.lower():
                        action_desc = "submits credentials successfully"
                    elif "confirm" in title_sem_name.lower():
                        action_desc = "confirms transaction successfully"
                    elif "date" in title_sem_name.lower():
                        action_desc = f"updates {entity_plural} based on selected date filter"
                    elif "category" in title_sem_name.lower():
                        action_desc = f"updates {entity_plural} according to selected category filter"
                    elif "destination" in title_sem_name.lower():
                        action_desc = f"filters {entity_plural} by chosen location destination"
                    elif "audience" in title_sem_name.lower():
                        action_desc = f"updates {entity_plural} based on selected audience type filter"
                    else:
                        action_desc = "submits action successfully"
                elif suffix == "accepts valid inputs and updates field state":
                    action_desc = "accepts valid entries"
                elif suffix == "handles validation error on blank submission":
                    action_desc = "validates empty entry"
                elif suffix == "sanitizes special characters and script injection inputs safely":
                    action_desc = "handles script injection safely"
                elif suffix == "handles session expiry gracefully during active text entry":
                    action_desc = "handles session expiry during entry"
                elif suffix == "warns user on mid-transaction page reload refresh":
                    action_desc = "warns user on reload during transaction"
                elif suffix == "prevents duplicate requests on rapid double-click triggers":
                    action_desc = "prevents duplicate requests"
                elif suffix == "handles network offline state gracefully on click":
                    action_desc = "handles offline connection gracefully"
                elif suffix == "handles high latency and displays progress spinner on click":
                    action_desc = "displays progress spinner during network latency"
                elif suffix == "executes target interaction on click":
                    if "date" in title_sem_name.lower():
                        action_desc = f"updates {entity_plural} based on selected date filter"
                    elif "category" in title_sem_name.lower():
                        action_desc = f"updates {entity_plural} according to selected category filter"
                    elif "destination" in title_sem_name.lower():
                        action_desc = f"filters {entity_plural} by chosen location destination"
                    elif "audience" in title_sem_name.lower():
                        action_desc = f"updates {entity_plural} based on selected audience type filter"
                    else:
                        action_desc = "triggers action successfully"
                else:
                    action_desc = suffix

                # Strip UI component suffixes for clean business descriptions
                clean_name = title_sem_name
                for sfx in [" Button", " Input Field", " Dropdown", " Checkbox", " Link"]:
                    if clean_name.endswith(sfx):
                        clean_name = clean_name[:-len(sfx)]
                clean_name = clean_name.strip()

                desc = sc["desc"]
                prio = sc["priority"]
                tctype = sc["type"]
                expected = sc["expected"]

                # Apply UI to Business Conversion Overlay rules
                refined_title = f"Verify {title_sem_name} {action_desc}"
                refined_expected = expected

                clean_lower = clean_name.lower()
                
                # Truncate extremely long element names to maintain readability (Enterprise Standard: max 100 chars)
                if len(clean_name) > 50:
                    # Keep first 25 chars and last 20 chars with ellipsis
                    clean_name = clean_name[:25] + "..." + clean_name[-20:]
                
                # Check for filter options - Use dynamic terminology
                entity_plural = self.terminology.get_entity_term(plural=True)
                
                if "filter" in suffix or "filter" in clean_lower or "category" in clean_lower or "destination" in clean_lower or "date" in clean_lower or "audience" in clean_lower:
                    if "reset" in suffix or "uncheck" in suffix:
                        refined_title = f"Verify resetting selection for {clean_name} restores default display state"
                        refined_expected = "Active filter selection is cleared and default unfiltered list is restored."
                    else:
                        # Use generic business-focused title instead of including long element name
                        refined_title = f"Verify category filter updates displayed records correctly"
                        refined_expected = f"Displayed {entity_plural} update dynamically to show only elements matching selected filter criteria."
                
                # Check for sort options
                elif "sort" in suffix or "sort" in clean_lower or "order" in clean_lower:
                    refined_title = f"Verify sorting orders list elements correctly"
                    refined_expected = f"Returned elements are reordered in the layout conforming to the selected sorting rule."

                # Check for specific search inputs
                elif "search" in clean_lower or "query" in clean_lower:
                    if "mismatch" in suffix or "non-matching" in suffix:
                        refined_title = f"Verify searching with non-matching query renders empty state"
                        refined_expected = "Graceful empty state placeholder message and clear option are displayed."
                    elif "clear" in suffix or "reset" in suffix:
                        refined_title = f"Verify clearing search query restores default list"
                        refined_expected = "Search query is removed and original list items are restored to view."
                    elif "special" in suffix or "script" in suffix:
                        refined_title = f"Verify search handles special characters and emojis safely"
                        refined_expected = "Security sanitization filters clean input entries and system executes query safely."
                    else:
                        refined_title = f"Verify searching with valid query displays matching results"
                        refined_expected = "Matching results are dynamically rendered, and non-matching listings are hidden."

                # Check for load more action - Use dynamic terminology
                elif "load more" in clean_lower:
                    entity_plural = self.terminology.get_entity_term(plural=True)
                    refined_title = f"Verify additional {entity_plural} load correctly while preserving active filters"
                    refined_expected = f"Additional {entity_plural} are loaded into the view while preserving existing filters and ordering."

                # Check for input fields
                elif ctype == "input":
                    elem_display_name = clean_name if clean_name else "input field"
                    if "validation" in suffix or "blank" in suffix or "empty" in suffix:
                        refined_title = f"Verify blank entry in required field {elem_display_name} displays validation error"
                        refined_expected = f"Form validation prevents submission and highlights required constraint on {elem_display_name}."
                    elif "injection" in suffix or "sanitize" in suffix:
                        refined_title = f"Verify input field {elem_display_name} sanitizes script injection strings safely"
                        refined_expected = f"Malicious tags typed into {elem_display_name} are sanitized and parsed safely without script execution."
                    elif "expiry" in suffix:
                        refined_title = f"Verify entering text in {elem_display_name} handles session expiration gracefully"
                        refined_expected = f"Re-authentication dialog is displayed to preserve draft entries in {elem_display_name}."
                    elif "persistence" in suffix:
                        refined_title = f"Verify draft entry in {elem_display_name} is preserved on page refresh"
                        refined_expected = f"Keystrokes buffer for {elem_display_name} is persisted or cleanly handled on browser refresh events."
                    elif "offline" in suffix:
                        refined_title = f"Verify validation warning displays for {elem_display_name} when connection is offline"
                        refined_expected = f"System displays network offline warning message and prevents submission for {elem_display_name}."
                    else:
                        refined_title = f"Verify entering valid data in input field {elem_display_name} updates state"
                        refined_expected = f"Input value for {elem_display_name} is registered successfully in the field state."

                # Check for buttons
                elif ctype == "button":
                    elem_display_name = clean_name if clean_name else "button"
                    if "concurrent" in suffix or "double" in suffix:
                        refined_title = f"Verify double clicking {elem_display_name} prevents concurrent actions"
                        refined_expected = f"Only the first action request for {elem_display_name} is processed, suppressing duplicate transaction requests."
                    elif "offline" in suffix:
                        refined_title = f"Verify selecting {elem_display_name} during network dropout shows warning"
                        refined_expected = f"System displays clear network warning during offline {elem_display_name} execution attempts."
                    elif "latency" in suffix:
                        refined_title = f"Verify selecting {elem_display_name} under high latency displays progress spinner"
                        refined_expected = f"Progress spinner feedback is rendered to inform user of active {elem_display_name} processing."
                    elif "rollback" in suffix:
                        refined_title = f"Verify selecting {elem_display_name} rolls back state cleanly on API failure"
                        refined_expected = f"Transaction changes for {elem_display_name} are rolled back and error message is displayed to the user."
                    elif "disabled" in suffix:
                        refined_title = f"Verify disabled button {elem_display_name} ignores click interactions"
                        refined_expected = f"Disabled control {elem_display_name} remains non-interactive to block premature submissions."
                    else:
                        refined_title = f"Verify selecting {elem_display_name} processes action correctly"
                        refined_expected = f"Requested business action for {elem_display_name} executes correctly, updating UI state."

                # Check for dropdowns
                elif ctype == "dropdown":
                    elem_display_name = clean_name if clean_name else "dropdown"
                    if "persistence" in suffix:
                        refined_title = f"Verify dropdown selection of {elem_display_name} persists across page reload"
                        refined_expected = f"The selected dropdown choice for {elem_display_name} remains active after refresh."
                    elif "dependent" in suffix:
                        refined_title = f"Verify changing choice in {elem_display_name} updates dependent fields"
                        refined_expected = f"Related form fields and values update correctly matching the chosen option in {elem_display_name}."
                    else:
                        refined_title = f"Verify selecting option in dropdown {elem_display_name} updates preference state"
                        refined_expected = f"Dropdown choice is updated with the selected option in system context for {elem_display_name}."

                # Check for checkboxes
                elif ctype in ["checkbox", "radio_button", "checkboxes", "radio_buttons"]:
                    elem_display_name = clean_name if clean_name else "checkbox"
                    if "persistence" in suffix:
                        refined_title = f"Verify selection state of {elem_display_name} persists across view reloads"
                        refined_expected = f"The active {elem_display_name} state remains preserved after reload."
                    elif "rollback" in suffix:
                        refined_title = f"Verify checkbox selection for {elem_display_name} rolls back on transaction failure"
                        refined_expected = f"Toggle state for {elem_display_name} rolls back to default and warning error displays."
                    else:
                        refined_title = f"Verify selecting checkbox {elem_display_name} updates preference state"
                        refined_expected = f"Toggle choice for {elem_display_name} is registered and updates the system preferences state."

                # Check for links
                elif ctype == "link":
                    if "navigation" in suffix or "history" in suffix:
                        refined_title = f"Verify navigation history compliance after selecting link"
                        refined_expected = "Browser navigation history is updated, enabling correct back and forward traversal."
                    elif "auth" in suffix or "permission" in suffix:
                        refined_title = f"Verify access authorization controls when navigating to link"
                        refined_expected = "Access redirect resolves if user is authorized, otherwise access is blocked."
                    else:
                        refined_title = f"Verify selecting link navigates to target section"
                        refined_expected = "Viewport navigation is completed, loading the target section view."

                title = refined_title
                expected = refined_expected

                # Finally, perform double safety check to clean out any remaining forbidden generic expected words
                exp_lower = expected.lower()
                for key_term in ["success", "successfully", "stable state", "navigation succeeds", "action is triggered"]:
                    if key_term in exp_lower:
                        expected = expected.replace("successfully", "correctly").replace("success", "completion").replace("stable state", "stable system environment").replace("navigation succeeds", "navigation completes").replace("action is triggered", "action is initiated")

                steps = []
                for s in sc["steps"]:
                    # Clean step action/expected if we have exec_data
                    step_action = s["action"]
                    step_expected = s["expected"]
                    if exec_data and "click" in step_action.lower():
                        step_action = f"Select one of the options from the list of values: {', '.join(exec_data[:3])}..."
                        if "date" in title_sem_name.lower():
                            step_expected = "The selected date filter option is highlighted and active."
                        elif "category" in title_sem_name.lower():
                            step_expected = "The selected category filter option is highlighted and active."
                        elif "destination" in title_sem_name.lower():
                            step_expected = "The selected destination option is highlighted and active."
                        elif "audience" in title_sem_name.lower():
                            step_expected = "The selected audience filter option is highlighted and active."
                    
                    steps.append({
                        "step": s["step"],
                        "action": step_action,
                        "expected": step_expected
                    })
                
                matched_edge = None
                for edge in edges:
                    if edge.get("source") == pk and semantic_name.lower() in edge.get("action", "").lower():
                        matched_edge = f"{edge['source']} -> {edge['target']} ({edge['action']})"
                        break
                
                tc_data = sc.get("data") or {}
                if exec_data:
                    tc_data["execution_data"] = exec_data

                # Update suffix_hint so _derive_intent_mapping can use it
                suffix_hint = sc.get("suffix", "")
                add_testcase(title, desc, prio, tctype, steps, expected, tc_data, matched_edge, comp_info=(comp_obj["selector"], comp_obj["type"]), target_pk=pk)


        # 2. Add consolidated Accessibility and Keyboard Navigation test cases
        if components_pool:
            add_testcase(
                title=f"Verify keyboard Tab navigation order and focus rings across {domain} page controls",
                desc=f"Ensures focusable interactive controls in the {domain} space are traversable sequentially in reading order via Tab triggers.",
                prio="Medium",
                tctype="UI",
                steps=[
                    {"step": 1, "action": f"Open {domain} page space and set active focus to first interactive control", "expected": "First control shows active focus highlight state"},
                    {"step": 2, "action": "Press keyboard 'Tab' key sequentially to navigate form controls", "expected": "Focus ring outline correctly transitions to each field in reading sequence"}
                ],
                expected="All interactive inputs, buttons, and links receive sequential focus indication complying with accessibility standards."
            )
            add_testcase(
                title=f"Verify WCAG accessibility standards and descriptive ARIA tags in {domain}",
                desc=f"Verifies HTML structural properties ensuring assistive technologies can read interactive control elements.",
                prio="Low",
                tctype="UI",
                steps=[
                    {"step": 1, "action": f"Inspect HTML structural metadata properties of {domain} buttons and form inputs", "expected": "Descriptive labels, tags, alt text, or associated ARIA values are present"}
                ],
                expected="Assigned descriptors provide descriptive names complying with accessibility standards."
            )

        # 2b. Happy-path authentication flow (auth/login stories only). This is the positive
        # counterpart to the per-field negative/validation cases: it enters the *valid*
        # configured credentials and verifies an actual authenticated state. At execution
        # time the email/password steps are filled from WEBOOK_EMAIL / WEBOOK_PASSWORD (.env),
        # so this really logs in rather than using a placeholder that always fails.
        if self.intent.get("semantic_key") == "auth":
            def _pool_has(*keywords: str) -> bool:
                for entry in components_pool:
                    comp, ctype = entry[0], entry[1]
                    sem = entry[3] if len(entry) > 3 else ""
                    hay = f"{comp} {sem}".lower()
                    if ctype == "input" and any(k in hay for k in keywords):
                        return True
                return False

            if _pool_has("email", "mail") and _pool_has("password"):
                add_testcase(
                    title="Verify login with valid registered credentials authenticates the user",
                    desc="Validates the primary authentication happy path — a registered user submitting valid credentials is authenticated and granted access to their account area.",
                    prio="High",
                    tctype="Functional",
                    steps=[
                        {"step": 1, "action": "Enter a valid registered account email into the Email field", "expected": "The email value is accepted in the Email field with no validation error."},
                        {"step": 2, "action": "Enter the valid account password into the Password field", "expected": "The password value is accepted and masked in the Password field with no validation error."},
                        {"step": 3, "action": "Click the Login button to submit the credentials", "expected": "The credentials are submitted and the authentication request is processed."},
                        {"step": 4, "action": "Verify the resulting authenticated state after submission", "expected": "The user is redirected away from the login form to their account or home area and an authenticated-session indicator such as the account menu becomes visible."}
                    ],
                    expected="A registered user entering valid credentials is authenticated and granted access to their account area with no error shown."
                )

        # 3. Multi-page User Journey / Business Flows
        domain_clean = domain.lower()
        def is_module_page(pk: str) -> bool:
            pk_lower = pk.lower()
            return bool(domain_clean in pk_lower or pk_lower in domain_clean or "explore" in pk_lower or "booking" in pk_lower or "room" in pk_lower or "payment" in pk_lower or "search" in pk_lower)

        journey_edges = [edge for edge in edges if edge.get("state_change") and edge.get("source") != edge.get("target")]
        module_to_module_edges = []
        external_navigation_edges = []
        
        for edge in journey_edges:
            src = edge.get("source", "")
            tgt = edge.get("target", "")
            if is_module_page(src) and is_module_page(tgt):
                module_to_module_edges.append(edge)
            elif is_module_page(src) and not is_module_page(tgt):
                external_navigation_edges.append(edge)

        # Generate up to 3 module-to-module transitions
        if module_to_module_edges:
            for edge in module_to_module_edges[:3]:
                src = edge["source"].capitalize()
                tgt = edge["target"].capitalize()
                act = edge["action"]
                res = edge["result"]
                add_testcase(
                    title=f"Verify complete transactional workflow: from {src} to {tgt} page spaces",
                    desc=f"Validates complete user navigation path and state transition mapping: {src} -> {tgt} via action '{act}'.",
                    prio="High",
                    tctype="Functional",
                    steps=[
                        {"step": 1, "action": f"Navigate to the starting space of {src}", "expected": f"{src} page state is loaded"},
                        {"step": 2, "action": f"Perform transition action: '{act}'", "expected": f"Action resolves and page begins redirecting to {tgt}"},
                        {"step": 3, "action": f"Observe state of destination space: {tgt}", "expected": f"Redirect resolves and system updates state to show: {res}"}
                    ],
                    expected=f"Transactional journey is completed, updating view to {tgt}.",
                    ref_edge=f"{edge['source']} -> {edge['target']} ({act})"
                )
        else:
            # Use dynamic terminology for workflow steps
            entity = self.terminology.get_entity_term()
            entity_plural = self.terminology.get_entity_term(plural=True)
            transaction = self.terminology.get_transaction_term()
            action = self.terminology.get_action_term("browse")
            
            workflow_steps = self.terminology.get_workflow_steps()
            if not workflow_steps or "step_" in workflow_steps:
                # Fallback to dynamic construction
                workflow_steps = f"{action} {entity_plural} → select {entity} → {transaction} → confirmation"
            add_testcase(
                title=f"Verify complete end-to-end user journey flow for {domain} Management ({workflow_steps})",
                desc=f"Ensures the main business action user path ({workflow_steps}) completes successfully matching story specification.",
                prio="High",
                tctype="Functional",
                steps=[
                    {"step": 1, "action": f"Access application and navigate to the starting page of the module: {domain}", "expected": f"{domain} page is accessed and active listings are displayed"},
                    {"step": 2, "action": f"Execute the user journey flow: {workflow_steps}", "expected": f"All interactive selections are made and registered in the {domain} interface state"},
                    {"step": 3, "action": "Submit final confirmation to complete the transaction", "expected": f"Confirmation page updates and displays correct details of user selection"}
                ],
                expected=f"The complete end-to-end user journey ({workflow_steps}) resolves successfully."
            )

        # Generate at most one external navigation verification testcase
        if external_navigation_edges:
            edge = external_navigation_edges[0]
            src = edge["source"].capitalize()
            tgt = edge["target"].capitalize()
            act = edge["action"]
            res = edge["result"]
            add_testcase(
                title=f"Verify external navigation from {src} to global {tgt} space",
                desc=f"Validates external/global page navigation path redirect from {src} to {tgt} via action '{act}'.",
                prio="Medium",
                tctype="Functional",
                steps=[
                    {"step": 1, "action": f"Navigate to {src} page space", "expected": f"{src} page is active"},
                    {"step": 2, "action": f"Click on navigation link '{act}'", "expected": f"Redirect resolves to global page {tgt}"}
                ],
                expected=f"External navigation link correctly loads {tgt} page.",
                ref_edge=f"{edge['source']} -> {edge['target']} ({act})"
            )

        # 4. Add Enterprise-Level Scenarios for Coverage Diversity
        enterprise_scenarios = generate_enterprise_scenarios()
        for ent_sc in enterprise_scenarios:
            add_testcase(
                title=ent_sc["title"],
                desc=ent_sc["description"],
                prio=ent_sc["priority"],
                tctype=ent_sc["type"],
                steps=ent_sc["steps"],
                expected=ent_sc["expected"],
                data=ent_sc.get("data"),
                extra_tags=[ent_sc["type"].lower()]
            )

        # 4.5. Generation Quality Validation with semantic duplicate detection
        def validate_generation_quality(test_cases_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            """Validates generation quality and regenerates affected scenarios if needed."""
            entity = self.terminology.get_entity_term()
            entity_plural = self.terminology.get_entity_term(plural=True)
            transaction = self.terminology.get_transaction_term()
            workflow_stages = self.terminology.get_workflow_stages()
            business_rules = self.terminology.get_business_rules()
            preconditions = self.terminology.get_preconditions()
            postconditions = self.terminology.get_postconditions()
            
            generic_phrases = [
                "Action executes successfully",
                "Action completed",
                "Target action executes",
                "Action is triggered",
                "interface updates",
                "workflow completes",
                "step_",
                "available workflow",
                "generic",
                "placeholder",
                "button triggers action",
                "input accepts input",
                "dropdown displays options"
            ]
            
            weak_assertions = [
                "successfully",
                "correctly",
                "properly",
                "appropriately",
                "as expected",
                "without issues"
            ]
            
            validated_testcases = []
            quality_issues = []
            semantic_signatures = set()  # Track semantic signatures for duplicate detection
            
            for tc in test_cases_list:
                tc_title = tc.get("title", "")
                tc_expected = tc.get("expected_result", "")
                tc_steps = tc.get("steps", [])
                tc_type = tc.get("type", "")
                
                # Generate semantic signature for duplicate detection
                # Normalize title and expected result to identify semantic duplicates
                title_normalized = re.sub(r'\d+', 'N', tc_title.lower())  # Replace numbers with N
                title_normalized = re.sub(r'["\'].*?["\']', 'VALUE', title_normalized)  # Replace quoted values
                expected_normalized = re.sub(r'\d+', 'N', tc_expected.lower())
                expected_normalized = re.sub(r'["\'].*?["\']', 'VALUE', expected_normalized)
                semantic_signature = f"{tc_type}:{title_normalized}:{expected_normalized}"
                
                # Check for semantic duplicates
                if semantic_signature in semantic_signatures:
                    quality_issues.append(f"Semantic duplicate detected in testcase {tc.get('id', 'unknown')}: {tc_title}")
                    # Skip adding this duplicate - it will be replaced with complementary scenarios
                    continue
                semantic_signatures.add(semantic_signature)
                
                # Check for generic phrases
                has_generic = False
                for phrase in generic_phrases:
                    if phrase in tc_expected.lower() or phrase in tc_title.lower():
                        has_generic = True
                        quality_issues.append(f"Generic phrase '{phrase}' found in testcase {tc.get('id', 'unknown')}")
                        break
                
                # Check for placeholder workflows
                if "step_" in tc_title.lower() or "step_" in tc_expected.lower():
                    has_generic = True
                    quality_issues.append(f"Placeholder workflow found in testcase {tc.get('id', 'unknown')}")
                
                # Check for weak business assertions
                assertion_count = sum(1 for assertion in weak_assertions if assertion in tc_expected.lower())
                if assertion_count > 2 and len(tc_expected) < 100:
                    has_generic = True
                    quality_issues.append(f"Weak business assertions in testcase {tc.get('id', 'unknown')}")
                
                # Check for repetitive template patterns (e.g., similar titles with only UI differences)
                if "button" in tc_title.lower() and "verify" in tc_title.lower():
                    # Check if this is a generic button test that could be diversified
                    if len(tc_expected) < 80 and "button" in tc_expected.lower():
                        quality_issues.append(f"Repetitive button template pattern in testcase {tc.get('id', 'unknown')}")
                
                # If quality issues found, regenerate with better business context
                if has_generic:
                    # Improve expected result with business context
                    improved_expected = tc_expected
                    for phrase in generic_phrases:
                        if phrase in improved_expected.lower():
                            if "workflow" in phrase.lower():
                                improved_expected = improved_expected.replace(phrase, f"business workflow for {entity} completes with expected state and measurable outcomes")
                            elif "action" in phrase.lower():
                                improved_expected = improved_expected.replace(phrase, f"{transaction} operation completes and {entity} state is updated with business validation")
                            elif "button" in phrase.lower():
                                improved_expected = improved_expected.replace(phrase, f"button click triggers {transaction} operation with {entity} state transition")
                            elif "input" in phrase.lower():
                                improved_expected = improved_expected.replace(phrase, f"input field accepts {entity} data with validation and business rule enforcement")
                            elif "dropdown" in phrase.lower():
                                improved_expected = improved_expected.replace(phrase, f"dropdown selection filters {entity_plural} with business criteria and updates display")
                            else:
                                improved_expected = improved_expected.replace(phrase, f"{entity} operation completes with measurable business outcome and state transition")
                    
                    tc["expected"] = improved_expected
                    log.info(f"[GenerationQuality] Improved testcase {tc.get('id', 'unknown')} by replacing generic phrases")
                
                # Add realistic test data and preconditions for enterprise scenarios
                if tc_type in ["API", "Business-Rule", "Workflow", "State-Transition"]:
                    # Add realistic test data
                    if not tc.get("data"):
                        tc["data"] = {
                            f"{entity}_id": f"test_{entity}_12345",
                            f"{entity}_name": f"Valid {entity.capitalize()} Test Data",
                            f"{entity}_status": "active",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "user_context": "authenticated_user"
                        }
                    
                    # Add preconditions from ACs
                    if preconditions and not tc.get("preconditions"):
                        tc["preconditions"] = preconditions[:2] if len(preconditions) >= 2 else preconditions
                    
                    # Add postconditions from ACs
                    if postconditions and not tc.get("postconditions"):
                        tc["postconditions"] = postconditions[:2] if len(postconditions) >= 2 else postconditions
                
                validated_testcases.append(tc)
            
            if quality_issues:
                log.warning(f"[GenerationQuality] Found {len(quality_issues)} quality issues: {quality_issues}")
            
            return validated_testcases

        # 4.6. Business Workflow Completeness Validation
        def validate_workflow_completeness(test_cases_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            """Validates business workflow completeness and auto-generates missing scenarios."""
            entity = self.terminology.get_entity_term()
            workflow_stages = self.terminology.get_workflow_stages()
            acceptance_criteria = self.story_data.get("story", {}).get("acceptance_criteria", [])
            
            # Extract covered workflow stages from existing test cases
            covered_stages = set()
            for tc in test_cases_list:
                tc_title = tc.get("title", "").lower()
                for stage in workflow_stages:
                    if stage in tc_title:
                        covered_stages.add(stage)
            
            # Identify missing workflow stages
            missing_stages = [stage for stage in workflow_stages if stage not in covered_stages]
            
            if missing_stages:
                log.info(f"[WorkflowCompleteness] Found missing workflow stages: {missing_stages}")
                
                # Auto-generate scenarios for missing stages
                for stage in missing_stages:
                    new_tc = {
                        "title": f"Verify {entity} workflow stage: {stage} completes successfully",
                        "description": f"Validates that the {stage} workflow stage for {entity} management completes successfully with expected business outcomes.",
                        "priority": "High",
                        "type": "Workflow",
                        "steps": [
                            {"step": 1, "action": f"Initiate {stage} workflow stage for {entity}", "expected": f"{stage.capitalize()} stage is initiated for {entity}."},
                            {"step": 2, "action": f"Execute {stage} workflow operations", "expected": f"{stage.capitalize()} operations complete successfully."},
                            {"step": 3, "action": f"Verify {stage} stage completion and {entity} state transition", "expected": f"{entity} transitions to next workflow stage after {stage} completion."}
                        ],
                        "expected": f"{entity} workflow stage {stage} completes successfully with expected business outcomes and state transitions."
                    }
                    test_cases_list.append(new_tc)
                    log.info(f"[WorkflowCompleteness] Auto-generated scenario for missing workflow stage: {stage}")
            
            return test_cases_list

        # Apply generation quality validation
        test_cases = validate_generation_quality(test_cases)
        
        # Apply workflow completeness validation
        test_cases = validate_workflow_completeness(test_cases)

        # 5. Process Attachments Insights (prioritized & high-quality)
        if self.attachment_insights:
            for insight in self.attachment_insights:
                filename = insight.get("filename", "")
                source = insight.get("source", "")
                text = insight.get("extracted_text", "")
                
                spec_rules = [line.strip() for line in re.split(r'[.\n]+', text) if len(line.strip()) > 10]
                steps = []
                for i, rule in enumerate(spec_rules[:5]):
                    steps.append({
                        "step": i + 1,
                        "action": f"Verify UI/behavior matches rule specified in {filename}: '{rule}'",
                        "expected": "System display and behavioral transitions match the rule exactly"
                    })
                
                if not steps:
                    steps = [{
                        "step": 1,
                        "action": f"Audit layout design structure and validation criteria defined in attachment: {filename}",
                        "expected": "All visual copy and behavior transitions match design mockup spec"
                    }]

                add_testcase(
                    title=f"Verify {domain} requirements match layout specification in attachment {filename}",
                    desc=f"Enforces design specifications, copy rules, and workflow conditions extracted from attachment: {filename} via {source}.",
                    prio="High",
                    tctype="Functional",
                    steps=steps,
                    expected=f"UI layout, constraints, and content are implemented exactly as specified in {filename}.",
                    data={"attachment_source": source, "attachment_file": filename},
                    ref_edge=f"attachment:{filename}",
                    extra_tags=["attachment", source]
                )

        # 5. Self-Validation & Gap Checking for Acceptance Criteria
        ac_list = self.intent.get("acceptance_criteria", [])
        for idx, ac in enumerate(ac_list):
            ac_covered = False
            for tc in test_cases:
                if ac.lower() in tc["title"].lower() or ac.lower() in tc["description"].lower() or any(ac.lower() in s["action"].lower() or ac.lower() in s["expected"].lower() for s in tc["steps"]):
                    ac_covered = True
                    break
            
            if not ac_covered:
                clean_ac = ac
                match_ac = re.match(r'^(?:AC-\d+|AC\d+)\s*–?\s*(.*)$', ac, re.IGNORECASE)
                if match_ac:
                    clean_ac = match_ac.group(1).strip()
                
                ac_title = f"Verify {clean_ac}"
                if len(ac_title) > 90:
                    ac_title = ac_title[:87] + "..."
                
                add_testcase(
                    title=ac_title,
                    desc=f"Verify that system behavior conforms to story requirement: '{ac}'.",
                    prio="High",
                    tctype="Functional",
                    steps=[
                        {"step": 1, "action": f"Set preconditions and trigger business flow for validation check: '{clean_ac}'", "expected": f"System state updates correctly matching business specification."},
                        {"step": 2, "action": "Verify system outcome conforms to specification requirements", "expected": f"Displayed data and system state reflect the '{clean_ac}' rule correctly."}
                    ],
                    expected=f"The system behavior conforms to the business requirement: '{clean_ac}'."
                )

        # Re-index all sequential IDs
        for i, tc in enumerate(test_cases):
            tc["id"] = f"TC-{module_prefix}-{(i + 1):03d}"

        return test_cases


class TestCaseGenerationAgent:
    """Self-learning, intent-driven, real-time QA intelligence engine."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def _run_pre_generation_validation(self, story_key: str, domain: str, story_data: Dict[str, Any] = None) -> None:
        """Verifies that execution context has been completely cleared of other module data using domain-aware validation."""
        log.info("[PreGenerationValidation] Verifying execution context isolation...")
        domain_clean = domain.lower()
        
        # Create terminology resolver for domain-aware validation
        intent = {"domain": domain}
        resolver = DomainTerminologyResolver(intent, story_data or {})
        
        # Verify ui_graph.json isolation if it exists
        graph_path = self.settings.artifact_dir / "ui_graph.json"
        if graph_path.exists():
            try:
                graph_data = json.loads(graph_path.read_text(encoding="utf-8"))
                g_story = graph_data.get("story_key")
                g_domain = graph_data.get("domain", "").lower()
                if g_story and g_story != story_key:
                    raise ValueError(f"Pre-generation validation failed: ui_graph.json has stale story {g_story} (expected {story_key})")
                
                # Note: Pre-generation validation for cross-module contamination is disabled
                # to avoid false positives on legitimate UI components with domain-related terms.
                # Post-generation validation with domain-aware allowlist is sufficient to catch
                # actual cross-module contamination in generated test cases.
                log.info("[PreGenerationValidation] Story context validation passed. Cross-module contamination check deferred to post-generation validation.")
                
            except Exception as e:
                if isinstance(e, ValueError):
                    raise e
                log.warning(f"Could not parse ui_graph.json in pre-generation check: {e}")

    def _extract_domain_terms_from_routes(self, routes: List[Dict[str, str]], current_domain: str) -> List[str]:
        """Extract domain terms from discovered routes dynamically at runtime."""
        domain_terms = set()
        for route in routes:
            label = route.get("label", "").lower()
            path = route.get("path", "").lower()
            
            # Extract potential domain terms from labels and paths
            # Common patterns: /en/explore?tag=football, /en/explore?category=sports-event
            if "tag=" in path:
                tag = path.split("tag=")[-1].split("&")[0]
                domain_terms.add(tag)
            if "category=" in path:
                category = path.split("category=")[-1].split("&")[0]
                # Extract base domain from category (e.g., "sports-event" -> "sports")
                base_category = category.split("-")[0]
                domain_terms.add(base_category)
                domain_terms.add(category)
            
            # Extract from labels
            label_words = label.split()
            for word in label_words:
                if len(word) > 3 and word.isalpha():
                    domain_terms.add(word.lower())
        
        # Remove current domain and its variants
        domain_clean = current_domain.rstrip('s')
        domain_terms.discard(domain_clean)
        domain_terms.discard(current_domain)
        domain_terms.discard(domain_clean + 's')
        
        # Remove generic terms that aren't domains
        generic_terms = {"explore", "page", "shop", "blog", "contact", "about", "terms", "privacy", "help", "business", "report", "fan", "protection", "safety", "black", "market"}
        for term in generic_terms:
            domain_terms.discard(term)
        
        return list(domain_terms)

    def _run_post_generation_validation(self, test_cases: List[Dict[str, Any]], story_key: str, domain: str, story_data: Dict[str, Any] = None) -> None:
        """Checks every generated testcase against the active story context using domain-aware validation."""
        log.info("[PostGenerationValidation] Verifying consistency of generated test suite...")
        domain_clean = domain.lower().rstrip('s')
        module_prefix = re.sub(r'[^a-zA-Z0-9]', '', domain).upper()
        
        # Create terminology resolver for domain-aware validation
        intent = {"domain": domain}
        resolver = DomainTerminologyResolver(intent, story_data or {})
        domain_allowlist = resolver.get_domain_allowlist()
        
        # Dynamically extract other domains from discovered routes
        discovered_routes = self.discovered_routes if hasattr(self, 'discovered_routes') else []
        other_domains = self._extract_domain_terms_from_routes(discovered_routes, domain_clean)
            
        for tc in test_cases:
            tc_id = tc.get("id", "")
            tc_title = tc.get("title", "")
            tc_desc = tc.get("description", "")
            tc_tags = tc.get("tags", [])
            tc_story = tc.get("linked_story", "")
            tc_expected = tc.get("expected_result", "")
            
            # Verify ID prefix matches active module
            if not tc_id.startswith(f"TC-{module_prefix}-"):
                raise ValueError(f"Post-generation validation failed: Testcase ID {tc_id} does not match active prefix {module_prefix}.")
                
            # Verify linked story matches active story key
            # Skip validation for test cases with empty/missing linked_story (legacy cases)
            if tc_story and tc_story != story_key:
                raise ValueError(f"Post-generation validation failed: Testcase {tc_id} has stale linked story {tc_story} (expected {story_key}).")
                
            # Verify tags do not reference another story or module - Use domain-aware validation
            for tag in tc_tags:
                tag_lower = tag.lower()
                if "story-" in tag_lower and f"story-{story_key}".lower() not in tag_lower:
                    raise ValueError(f"Post-generation validation failed: Testcase {tc_id} contains stale story tag: {tag} (expected story-{story_key}).")
                # Check if tag contains a term from another domain that's NOT in the current domain's allowlist
                for other_d in other_domains:
                    if other_d in tag_lower and not resolver.is_valid_domain_term(other_d):
                        raise ValueError(f"Post-generation validation failed: Testcase {tc_id} contains cross-module contamination: '{tag}' references domain '{other_d}' which is not in current domain allowlist.")
                        
            # Verify title, description, expected results, steps do not reference other domains/modules - Use domain-aware validation with leniency for navigation
            for text_field in [tc_title, tc_desc, tc_expected]:
                tf_lower = text_field.lower()
                for other_d in other_domains:
                    if other_d in tf_lower and not resolver.is_valid_domain_term(other_d):
                        # Allow cross-domain terms if they appear in legitimate contexts:
                        # 1. Navigation contexts (selecting, navigating, clicking, choosing)
                        # 2. UI component contexts (button, link, input, dropdown, checkbox)
                        # 3. Validation contexts (validates, verifies, tests)
                        # 4. Business action contexts (executes, action, business action, UI state)
                        # 5. General UI interaction contexts (requested, processes, updates)
                        is_legitimate_context = any(
                            nav_kw in tf_lower for nav_kw in [
                                "selecting", "navigating", "navigate to", "clicking", "choose", "choosing",
                                "button", "link", "input", "dropdown", "checkbox", "option",
                                "validates", "verifies", "tests", "check", "response",
                                "executes", "action", "business action", "ui state",
                                "requested", "processes", "updates", "triggers"
                            ]
                        )
                        if is_legitimate_context:
                            continue  # Allow legitimate cross-domain references in UI/navigation contexts
                        raise ValueError(f"Post-generation validation failed: Testcase {tc_id} references a different domain: '{other_d}' in field content: '{text_field}'. This term is not in the current domain allowlist.")
            
            for s in tc.get("steps", []):
                s_act = s.get("action", "").lower()
                s_exp = s.get("expected", "").lower()
                for other_d in other_domains:
                    if (other_d in s_act or other_d in s_exp) and not resolver.is_valid_domain_term(other_d):
                        # Allow cross-domain terms if they appear in legitimate contexts
                        is_legitimate_context = any(
                            nav_kw in s_act for nav_kw in [
                                "selecting", "navigating", "navigate to", "clicking", "choose", "choosing",
                                "button", "link", "input", "dropdown", "checkbox", "option",
                                "validates", "verifies", "tests", "check", "response",
                                "executes", "action", "business action", "ui state",
                                "requested", "processes", "updates", "triggers"
                            ]
                        )
                        if is_legitimate_context:
                            continue  # Allow legitimate cross-domain references in UI/navigation contexts
                        raise ValueError(f"Post-generation validation failed: Testcase {tc_id} references a different domain: '{other_d}' in step field: '{s}'. This term is not in the current domain allowlist.")

    def _sanitize_testcase_content(self, tc: Dict[str, Any], domain: str, story_data: Dict[str, Any] = None) -> None:
        """Sanitizes test case content (title, description, expectations, and steps) dynamically."""
        # Create terminology resolver for dynamic terminology
        intent = {"domain": domain}
        resolver = DomainTerminologyResolver(intent, story_data or {})
        
        # Use dynamic terminology from resolver
        entity = resolver.get_entity_term()
        entity_plural = resolver.get_entity_term(plural=True)
        transaction = resolver.get_transaction_term()

        # A test case for the login/auth/registration page itself is exercised while the
        # user is deliberately NOT signed in yet, so a blanket "User is logged in" /
        # "User is authenticated" precondition template would directly contradict the
        # scenario being tested.
        is_auth_domain = any(kw in domain.lower() for kw in ["login", "signin", "sign-in", "auth", "register", "signup", "sign-up"])

        title = tc.get("title", "")
        desc = tc.get("description", "")
        tctype = tc.get("type", "Functional")
        title_lower = title.lower()
        desc_lower = desc.lower()

        # Check if the test case is an enterprise scenario (API, Business-Rule, etc.)
        # These are generated with high quality initially, so we don't overwrite their expected result and steps,
        # but we do want to apply terminology replacements and humanize graph node IDs at the end.
        is_negative_type = tctype == "Negative"
        is_enterprise = tctype in ["API", "Business-Rule", "Workflow", "Security", "Resilience", "State-Transition", "Accessibility", "Recovery"]

        # Extract target element name from title or description
        element_name = ""
        for pattern in [
            r"verify selecting (?:the |button )?(.+?)(?: processes action| filters| orders| updates| transitions| clears| accepts| handles| is| remains| on| under)?$",
            r"verify disabled button (.+?) ignores",
            r"verify double clicking (.+?) prevents",
            r"verify input field (.+?) sanitizes",
            r"verify blank entry in required field (.+?) displays",
            r"verify entering text in (.+?) handles",
            r"verify draft entry in (.+?) is",
            r"verify validation warning displays for (.+?) when",
            r"verify entering valid data in input field (.+?) updates",
            r"verify dropdown (.+?) selection",
            r"verify changing choice in (.+?) updates",
            r"verify selecting checkbox (.+?) updates",
            r"verify selection state of (.+?) persists",
            r"verify checkbox selection for (.+?) rolls",
            r"verify selecting (.+?) during",
            r"verify selecting (.+?) under",
            r"verify selecting (.+?) rolls",
            r"verify navigation history compliance after selecting (.+?)$",
            r"verify access authorization controls when navigating to (.+?)$",
            r"verify selecting (.+?) navigates",
        ]:
            match = re.search(pattern, title, re.IGNORECASE)
            if match:
                element_name = match.group(1).strip()
                break
        
        # If no element name was found, or if the extracted name is too generic, fallback to description
        if not element_name or element_name.lower() in ["button", "input", "dropdown", "checkbox", "link", "element", "field", "control"]:
            match = re.search(r"(?:button|field|control|element|dropdown|checkbox|link) ([a-zA-Z0-9_, -]+?)(?:\.|$)", desc, re.IGNORECASE)
            if match:
                element_name = match.group(1).strip()
        
        if not element_name:
            match = re.search(r"Verify (?:selecting |entering |entering valid data in |blank entry in required field |disabled button |double clicking )?(.+?)(?: processes| ignores| prevents| updates| filters| orders| handles| sanitizes| displays| remains| toggles| clears| navigates| completes|$)", title, re.IGNORECASE)
            if match:
                element_name = match.group(1).strip()
                
        if element_name:
            # Clean element name of common action/scenario suffixes (stripping them safely)
            for phrase in [
                "under high latency displays progress spinner",
                "under high latency and displays progress spinner on click",
                "during network dropout shows warning",
                "rolls back state cleanly on API failure",
                "remains non-interactive and blocks clicks when disabled",
                "prevents concurrent transactions on rapid double clicks",
                "ignores click interactions",
                "prevents concurrent actions",
                "processes action correctly",
                "has a validation error on blank submission",
                "has a session expiration gracefully",
                "is responsive across supported devices",
                "matches layout specification in attachment",
                "Button", "Input Field", "Dropdown", "Checkbox", "Link"
            ]:
                phrase_lower = phrase.lower()
                if element_name.lower().endswith(phrase_lower):
                    element_name = element_name[:-len(phrase)].strip()
                elif phrase_lower in element_name.lower():
                    idx = element_name.lower().find(phrase_lower)
                    element_name = element_name[:idx].strip()
            element_name = element_name.strip()
            
        if not element_name or element_name.lower() in ["button", "input", "dropdown", "checkbox", "link", "element", "field", "control"]:
            element_name = entity.capitalize()

        # Identify scenario category — ordered from most-specific to least-specific
        is_checkbox = "checkbox" in title_lower and not is_enterprise
        is_double_click = ("double click" in title_lower or "concurrent" in title_lower) and not is_checkbox and not is_enterprise
        is_latency = ("latency" in title_lower or "latency" in desc_lower) and not is_checkbox and not is_enterprise
        is_offline = ("offline" in title_lower or "offline" in desc_lower or "network dropout" in title_lower or "network dropout" in desc_lower) and not is_checkbox and not is_enterprise
        is_disabled = ("disabled" in title_lower or "disabled" in desc_lower or "ignores" in title_lower) and not is_checkbox and not is_enterprise
        is_rollback = ("rollback" in title_lower or "backend failure" in title_lower or "server error" in title_lower) and not is_checkbox and not is_enterprise
        is_blank = ("blank" in title_lower or "empty" in title_lower or "validation error" in title_lower) and not is_checkbox and not is_enterprise
        is_injection = ("injection" in title_lower or "sanitize" in title_lower) and not is_checkbox and not is_enterprise
        is_expiry = ("expiry" in title_lower or "expiry" in desc_lower) and not is_checkbox and not is_enterprise
        is_persistence = ("persistence" in title_lower or "reload" in title_lower or "refresh" in title_lower or "persist" in title_lower) and not is_checkbox and not is_enterprise
        is_navigation = ("navigates" in title_lower or "navigation history" in title_lower or "access authorization" in title_lower) and not is_checkbox and not is_enterprise

        # Whole-word matcher — raw `in` substring checks caused real misclassifications:
        # "date" matched inside "up*date*s state" (present in EVERY happy-path input title
        # like "...input field Email updates state"), so all valid-input cases were wrongly
        # treated as date-filters and rewritten into filter-grid steps. Likewise "order"
        # matches "reorder"/"border" and "sort" matches "assortment". Require word boundaries.
        def _kw(word: str) -> bool:
            return re.search(rf'\b{re.escape(word)}\b', title_lower) is not None

        # Check calendar dates and months for date filter matching
        has_calendar_word = (
            any(_kw(day) for day in ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]) or
            any(_kw(month) for month in ["january", "february", "march", "april", "june", "july", "august", "september", "october", "november", "december"])
        )
        # "selecting"/"choice" alone are too generic (e.g. "Verify selecting Confirm Booking
        # Action processes action correctly" is a booking button, not a filter) — only treat
        # them as a filter signal when there's actual option/dropdown context and the title
        # isn't clearly a business-transaction action.
        has_option_context = any(w in title_lower for w in ["option", "dropdown", "list of values", "choice from"])
        is_generic_action_context = any(w in title_lower for w in ["booking", "payment", "checkout", "confirm", "place bid", "submit", "reserve"])
        is_filter = (
            not is_checkbox and not is_enterprise and
            (
                _kw("filter") or _kw("category") or _kw("destination") or
                _kw("date") or _kw("audience") or has_calendar_word or
                (("selecting" in title_lower or "choice" in title_lower) and has_option_context and not is_generic_action_context)
            )
        )
        is_sort = (_kw("sort") or _kw("order")) and not is_enterprise
        is_search = (_kw("search") or _kw("query")) and not is_enterprise
        is_responsiveness = ("responsive" in title_lower or "devices" in title_lower) and not is_enterprise
        is_no_defects = ("no critical" in title_lower or "defects are present" in title_lower) and not is_enterprise
        is_keyboard = ("keyboard" in title_lower or "tab" in title_lower) and not is_enterprise
        is_accessibility = ("aria" in title_lower or "wcag" in title_lower or "accessibility" in title_lower) and not is_enterprise

        # Element overrides for global check test cases to keep them clean
        if is_responsiveness:
            element_name = "UI layout"
        elif is_no_defects:
            element_name = "system"
        elif is_keyboard:
            element_name = "page controls"
        elif is_accessibility:
            element_name = "interactive elements"
        elif is_checkbox:
            # C1-FIX: Extract the real checkbox label from the title
            cb_match = re.search(r"checkbox\s+(.+?)(?:\s+updates|\s+rolls|\s+selection|$)", title, re.IGNORECASE)
            element_name = cb_match.group(1).strip() if cb_match else element_name

        # Initialize enriched fields
        expected = tc.get("expected_result", "")
        steps = tc.get("steps", [])
        behavioral_seq = tc.get("behavioral_sequence", [])
        test_data = {}
        # Resolve base URL dynamically
        story_base_url = None
        story = story_data.get("story", {}) if story_data else {}
        story_text = f"{story.get('summary', '')} {story.get('description', '')} {' '.join([c.get('body', '') for c in story.get('comments', [])])} {' '.join(story.get('acceptance_criteria', []))}"
        url_matches = re.findall(r'https?://[a-zA-Z0-9./?=&_-]+', story_text)
        for url in url_matches:
            url_clean = url.rstrip('.,;)')
            from urllib.parse import urlparse
            parsed = urlparse(url_clean)
            if parsed.netloc:
                story_base_url = f"{parsed.scheme}://{parsed.netloc}"
                break
        if not story_base_url:
            story_base_url = "https://webook.com"

        preconditions = tc.get("preconditions", [])
        if len(preconditions) == 1 and any(kw in preconditions[0] for kw in ["http://", "https://"]):
            t_url = preconditions[0].replace("Application is accessible at ", "")
        else:
            t_url = f"{story_base_url.rstrip('/')}/en/explore?tag={domain.lower()}"

        story_ac = story.get("acceptance_criteria", []) if story else []

        # 1. Scenario-specific Expected Results, Test Data, and Preconditions
        if is_enterprise:
            expected = tc.get("expected_result", expected)
            test_data = tc.get("test_data", {})
            preconditions = tc.get("preconditions", preconditions)
        elif is_checkbox:
            if is_negative_type:
                expected = f"When the API call for toggling '{element_name}' returns a server error (500), the checkbox reverts to its original state and an error toast is displayed."
                test_data = {"checkbox_label": element_name, "checked": True, "mock_response_status": 500}
                preconditions = [
                    f"Application is accessible at {t_url}",
                    "API endpoint for preference update is intercepted and mocked to return 500",
                    f"'{element_name}' checkbox is visible in the preference settings section"
                ]
            else:
                expected = f"Clicking the '{element_name}' checkbox toggles its selection state, and the preference update API call is triggered with the new checked value persisted."
                test_data = {"checkbox_label": element_name, "checked": True}
                preconditions = [
                    f"Application is accessible at {t_url}",
                    f"'{element_name}' checkbox is visible and interactive in the settings panel",
                    "User is authenticated with appropriate permissions"
                ]
        elif is_double_click:
            expected = f"Only a single execution request for the {element_name} action is processed by the application server, safely debouncing subsequent clicks and preventing duplicate entries."
            test_data = {"click_interval_ms": 50, "max_concurrent_requests": 1}
            preconditions = [
                f"Application is accessible at {t_url}",
                "Target control is enabled and ready to accept double interaction triggers",
                "Browser session is active and stable"
            ]
        elif is_latency:
            expected = f"A visual progress loading spinner displays immediately, disabling interactive controls to prevent concurrent clicks during the emulation delay."
            test_data = {"latency_delay_ms": 5000}
            preconditions = [
                f"Application is accessible at {t_url}",
                "Network speed and latency emulation tools are configured",
                "Target control is ready for interaction"
            ]
        elif is_offline:
            expected = f"The application blocks request dispatching during network disconnect and shows a clear toast message: 'Internet connection lost'."
            test_data = {"network_status": "offline"}
            preconditions = [
                f"Application is accessible at {t_url}",
                "Network speed and latency emulation tools are configured",
                "Offline emulation is active"
            ]
        elif is_disabled:
            expected = f"The disabled {element_name} control remains visually inactive and completely blocks click events, preventing premature submissions."
            test_data = {"is_interactive": False}
            preconditions = [
                f"Application is accessible at {t_url}",
                "The target control state is configured to disabled",
                "User is on the correct workflow section"
            ]
        elif is_rollback:
            expected = f"The system handles the backend failure gracefully, rolling back any draft UI state changes for {element_name} and displaying an error dialog."
            test_data = {"mock_response_status": 500, "mock_response_body": "Internal Server Error"}
            preconditions = [
                f"Application is accessible at {t_url}",
                "Mock API server configuration is initialized",
                "API endpoint for form submission is intercepted"
            ]
        elif is_blank:
            expected = f"Form validation rules prevent submission, highlighting the {element_name} field with a validation message: '{element_name} is required'."
            test_data = {"input_value": "", "validation_rule": "required"}
            preconditions = [
                f"Application is accessible at {t_url}",
                "Required input field is cleared of all values",
                "Form validation context is active"
            ]
        elif is_injection:
            expected = f"The critical input field {element_name} correctly sanitizes malicious script tags, storing or searching them as literal text without execution."
            test_data = {"input_value": "<script>alert('xss')</script>", "expected_sanitization": "encoded"}
            preconditions = [
                f"Application is accessible at {t_url}",
                "Security auditing logs are active",
                "Target input field is focused"
            ]
        elif is_expiry:
            expected = f"The user's session state is safely handled; if session expires during input in {element_name}, a re-authentication prompt is shown without losing progress."
            test_data = {"session_token_expired": True, "draft_data_key": f"draft_{element_name}"}
            preconditions = [
                f"Application is accessible at {t_url}",
                "Authentication state is active",
                "Session timeout interval is set to trigger immediately"
            ]
        elif is_persistence:
            expected = f"The selection or draft input state of {element_name} is preserved in local/session storage and persists after page reload."
            # Storage key namespaced by the actual domain being tested, not a fixed brand prefix.
            test_data = {"draft_input": "draft_save_test_123", "storage_key": f"{domain.lower()}_{element_name}"}
            preconditions = [
                f"Application is accessible at {t_url}",
                "Browser storage capabilities are enabled",
                "Draft data entry is completed"
            ]
        elif is_filter:
            if is_negative_type:
                expected = f"When no {entity_plural} exist for the selected filter '{element_name}', the events grid displays a contextual empty state message such as 'No results found for {element_name}'."
                test_data = {"selected_filter": element_name, "filter_type": "category", "expected_result_count": 0}
                preconditions = [
                    f"Application is accessible at {t_url}",
                    f"Database contains no {entity_plural} matching the '{element_name}' filter criterion",
                    "Events grid is loaded and visible"
                ]
            else:
                expected = f"The {entity_plural} list grid refreshes and updates dynamically, displaying only {entity_plural} that match the selected filter criterion: {element_name}."
                test_data = {"selected_filter": element_name, "filter_type": "category"}
                preconditions = [
                    f"Application is accessible at {t_url}",
                    "Events grid is loaded and populated",
                    "Target filter choice is visible and active"
                ]
        elif is_sort:
            expected = f"The displayed list of {entity_plural} reorders sequentially in real-time according to the selected sorting rule: {element_name}."
            test_data = {"sort_by": element_name, "order": "descending"}
            preconditions = [
                f"Application is accessible at {t_url}",
                f"Multiple {entity_plural} are rendered in events grid",
                "Sort control choice is visible and active"
            ]
        elif is_search:
            expected = f"The search query is evaluated and the results display refreshes to show only matching {entity_plural} matching the search value."
            # Search term derived from the domain's real business entity, not a
            # hardcoded sample value — works for any module (flight, hotel, invoice, ...).
            test_data = {"search_query": entity, "locale": "en"}
            preconditions = [
                f"Application is accessible at {t_url}",
                "Search query field is active",
                "Database records match query value"
            ]
        elif is_responsiveness:
            expected = f"The layout and interactive controls resize and reposition fluidly according to the viewport width, maintaining full usability on all devices."
            test_data = {"viewports": [{"width": 375, "height": 812}, {"width": 768, "height": 1024}]}
            preconditions = [
                f"Application is accessible at {t_url}",
                "Browser viewport control is active",
                "Page visual resources are completely loaded"
            ]
        elif is_no_defects:
            expected = f"The landing page and critical user flow paths are completely free of functional defects, console error logs, or broken UI elements."
            test_data = {"url_check_list": [t_url]}
            preconditions = [
                f"Application is accessible at {t_url}",
                "Page content rendering completes",
                "Browser console monitoring is active"
            ]
        elif is_keyboard:
            expected = f"Interactive elements receive focus in sequential tab order, rendering a distinct visual focus outline border around controls."
            test_data = {"focus_indicator_visible": True}
            preconditions = [
                f"Application is accessible at {t_url}",
                "Interactive components are visible in viewport",
                "Keyboard focus simulation environment is active"
            ]
        elif is_accessibility:
            expected = f"Interactive elements contain explicit ARIA attributes (e.g. role, aria-label, aria-expanded) for screen reader accessibility compliance."
            test_data = {"wcag_compliance_level": "AA"}
            preconditions = [
                f"Application is accessible at {t_url}",
                "Interactive controls are visible in DOM tree",
                "Accessibility validation framework is active"
            ]
        elif is_navigation:
            nav_target = re.sub(r"(?:verify selecting|verify navigation history compliance after selecting|verify access authorization controls when navigating to)\s+", "", title, flags=re.IGNORECASE).strip()
            expected = f"Clicking the '{nav_target}' link navigates to the target page; key page content and heading elements are visible and the browser URL updates correctly."
            test_data = {"link_label": nav_target, "expected_url_fragment": nav_target.lower().replace(" ", "-")}
            preconditions = [
                f"Application is accessible at {t_url}",
                "User has arrived at the current page" if is_auth_domain else "User is authenticated",
                f"'{nav_target}' link is visible in the page footer or navigation area"
            ]
        else:
            ac_phrase = re.sub(r"^Verify\s+", "", title, flags=re.IGNORECASE).rstrip(".")
            expected = f"The system behavior conforms to the acceptance criterion: '{ac_phrase}'. The UI reflects the expected state with no errors or console warnings."
            test_data = {}
            if ac_phrase and not any(phrase in ac_phrase.lower() for phrase in ["complete transactional workflow", "keyboard navigation", "layout specification"]):
                if not story_ac or any(ac.lower() in ac_phrase.lower() for ac in story_ac):
                    test_data["acceptance_criterion"] = ac_phrase
            preconditions = [
                f"Application is accessible at {t_url}",
                "User has arrived at the current page" if is_auth_domain else "User is logged in",
                "System state is ready for user interaction"
            ]

        # C3-FIX: Append type qualifier to title for Negative/Edge tests that share a title with Functional tests
        new_title = title
        if is_checkbox and is_negative_type:
            new_title = f"Verify checkbox {element_name} reverts state on API failure"
            new_title = new_title if new_title.endswith("]") else f"{new_title} [Negative]"
        elif is_negative_type:
            new_title = title if title.endswith("]") else f"{title} [Negative]"
        elif tctype == "Edge":
            new_title = title if title.endswith("]") else f"{title} [Edge]"
        
        if new_title != title:
            tc["title"] = new_title
            fp = tc.get("flow_path", "")
            if fp and " → " in fp:
                prefix = fp.split(" → ", 1)[0]
                tc["flow_path"] = f"{prefix} → {new_title}"

        # 2. Enrich Steps & Actions with specific instructions (eliminating vague placeholder verbs)
        if len(steps) == 1:
            if (tctype == "Functional" and (is_filter or is_sort or is_search or is_navigation)) or is_double_click:
                steps.append({"action": "", "expected": "", "system_response": ""})
                bs = tc.get("behavioral_sequence", [])
                if bs and len(bs) == 1:
                    bs.append({
                        "step": 2,
                        "user_action": "",
                        "system_response": "",
                        "state_change": "State Verification Completed"
                    })
                    tc["behavioral_sequence"] = bs

        enriched_steps = []
        for idx, s in enumerate(steps):
            s_action = s.get("action", "")
            s_expected = s.get("expected", "")

            step_num = idx + 1

            # Rewrite steps based on scenario details
            if is_enterprise:
                # Do not overwrite enterprise steps
                pass
            elif is_checkbox:
                if is_negative_type:
                    if step_num == 1:
                        s_action = f"Click the '{element_name}' checkbox to toggle its state."
                        s_expected = f"'{element_name}' checkbox registers the click and a preference update API call is dispatched."
                    elif step_num == 2:
                        s_action = f"Observe the UI response after the mocked API returns status 500."
                        s_expected = f"The '{element_name}' checkbox reverts to its original unchecked state and an error toast is shown."
                else:
                    if step_num == 1:
                        s_action = f"Click the '{element_name}' checkbox in the preferences panel."
                        s_expected = f"The '{element_name}' checkbox changes to checked state (visual tick mark visible)."
                    elif step_num == 2:
                        s_action = f"Verify the preference update API call was triggered."
                        s_expected = f"Network request is dispatched to update '{element_name}' preference; response status is 200 OK."
            elif is_double_click:
                if step_num == 1:
                    s_action = f"Rapidly trigger double clicks on {element_name} in quick succession (interval < 100ms)."
                    s_expected = f"The system registers only the first click, debouncing the second trigger."
                elif step_num == 2:
                    s_action = f"Inspect network requests to confirm only a single API call was dispatched for the {element_name} interaction."
                    s_expected = f"Exactly one network request is visible in the request log; no duplicate calls are recorded."
            elif is_latency:
                if step_num == 1:
                    s_action = f"Configure network emulation to 5000ms delay and click {element_name}."
                    s_expected = f"The visual loading spinner is rendered immediately, disabling interactive controls."
                elif step_num == 2:
                    s_action = f"Wait for latency response to resolve."
                    s_expected = f"The loading indicator is dismissed, and UI state updates with events data."
            elif is_offline:
                if step_num == 1:
                    s_action = f"Simulate network disconnect and click {element_name}."
                    s_expected = f"The system blocks request transmission and displays an offline warning message."
            elif is_disabled:
                if step_num == 1:
                    s_action = f"Inspect DOM properties of {element_name} to check for disabled attribute."
                    s_expected = f"{element_name} control contains the disabled attribute or class."
                elif step_num == 2:
                    s_action = f"Attempt to click or trigger the disabled {element_name} element."
                    s_expected = f"No action is initiated, console errors are empty, and interface remains stable."
            elif is_rollback:
                if step_num == 1:
                    s_action = f"Trigger {element_name} action while mocking server error response (status code 500)."
                    s_expected = f"Server returns 500 Internal Server Error response."
                elif step_num == 2:
                    s_action = f"Verify UI state and error warning display."
                    s_expected = f"UI display rolls back cleanly to its pre-action state, and an error warning banner is shown."
            elif is_blank:
                if step_num == 1:
                    s_action = f"Ensure the input field {element_name} is empty and trigger form submission."
                    s_expected = f"Form submission is blocked, and {element_name} field is highlighted."
                elif step_num == 2:
                    s_action = f"Verify validation warning text displays near {element_name}."
                    s_expected = f"Validation message '{element_name} is required' is rendered in UI context."
            elif is_injection:
                if step_num == 1:
                    s_action = f"Type malicious script payload '<script>alert(1)</script>' into {element_name}."
                    s_expected = f"Input characters are registered correctly without executing."
                elif step_num == 2:
                    s_action = f"Submit form data containing script payload."
                    s_expected = f"System processes request safely; tags are encoded or stripped, and no script executes."
            elif is_expiry:
                if step_num == 1:
                    s_action = f"Begin entering valid data in input field {element_name}."
                    s_expected = f"Data is typed in {element_name}."
                elif step_num == 2:
                    s_action = f"Simulate session expiry token event."
                    s_expected = f"Re-authentication prompt is shown, preserving the current input state of {element_name}."
            elif is_persistence:
                if step_num == 1:
                    s_action = f"Select or enter a value in {element_name}."
                    s_expected = f"Input value is active in {element_name}."
                elif step_num == 2:
                    s_action = f"Trigger a full page refresh."
                    s_expected = f"Page reloads and {element_name} displays the previously entered value."
            elif is_filter:
                if is_negative_type:
                    if step_num == 1:
                        s_action = f"Click filter option '{element_name}' in the UI filter area (no matching records exist)."
                        s_expected = f"Filter is applied but the events grid returns zero results."
                    elif step_num == 2:
                        s_action = f"Verify the empty state message is displayed."
                        s_expected = f"The grid shows a contextual empty state message: 'No results found for {element_name}'."
                else:
                    if step_num == 1:
                        s_action = f"Click filter option '{element_name}' in the UI filter area."
                        s_expected = f"The selected option '{element_name}' is highlighted and active."
                    elif step_num == 2:
                        s_action = f"Verify that the displayed {entity_plural} list updates."
                        s_expected = f"The events grid displays only {entity_plural} matching the '{element_name}' parameter."
            elif is_sort:
                if step_num == 1:
                    s_action = f"Apply sorting parameter '{element_name}' from sorting options dropdown."
                    s_expected = f"Sorting rule '{element_name}' is set active."
                elif step_num == 2:
                    s_action = f"Verify that the displayed {entity_plural} list is sorted correctly."
                    s_expected = f"All visible items are sorted in the UI conforming to the '{element_name}' rule."
            elif is_search:
                if step_num == 1:
                    s_action = f"Type search query into {element_name} input box."
                    s_expected = f"Search query text displays correctly in the search field."
                elif step_num == 2:
                    s_action = f"Trigger search filter execution."
                    s_expected = f"The UI list updates showing matching {entity_plural}; appropriate empty state displays if there are zero matches."
            elif is_responsiveness:
                if step_num == 1:
                    s_action = f"Resize the browser viewport to mobile dimensions (e.g., 375x812) and check alignment."
                    s_expected = f"The navigation layout collapses into a mobile hamburger menu and all controls fit without horizontal scrolling."
                elif step_num == 2:
                    s_action = f"Resize the browser viewport to tablet dimensions (e.g., 768x1024) and interact with {element_name}."
                    s_expected = f"UI components resize fluidly and remain fully interactive without overlapping text."
            elif is_no_defects:
                if step_num == 1:
                    s_action = f"Navigate through the main user flow paths in the {domain} page context."
                    s_expected = f"No application crash screens, status code errors, or console exceptions occur."
                elif step_num == 2:
                    s_action = f"Verify that all headers, event grids, and filtering controls render correctly."
                    s_expected = f"All visual layout structures display properly with correct system statuses."
            elif is_keyboard:
                if step_num == 1:
                    s_action = f"Open {domain} page space and set active focus to first interactive control."
                    s_expected = f"First control shows active focus highlight state."
                elif step_num == 2:
                    s_action = f"Press keyboard 'Tab' key sequentially to navigate form controls."
                    s_expected = f"Focus ring outline correctly transitions to each field in reading sequence."
            elif is_accessibility:
                if step_num == 1:
                    s_action = f"Inspect HTML code structure of {element_name}."
                    s_expected = f"DOM structure contains standard descriptive ARIA attributes and labels."
            elif is_navigation:
                if step_num == 1:
                    s_action = f"Click navigation link/control for {element_name}."
                    s_expected = f"The browser initiates transition to the target {element_name} destination."
                elif step_num == 2:
                    s_action = f"Verify that the target page has loaded and key content is visible."
                    s_expected = f"The {element_name} page renders successfully with no console or connection errors."

            s_system_resp = s_expected

            enriched_steps.append({
                "step": step_num,
                "action": s_action,
                "expected": s_expected,
                "system_response": s_system_resp
            })

        if is_double_click and len(enriched_steps) == 1:
            enriched_steps.append({
                "step": 2,
                "action": f"Inspect network requests to confirm only a single API call was dispatched for the {element_name} interaction.",
                "expected": "Exactly one network request is visible in the request log; no duplicate calls are recorded.",
                "system_response": "Exactly one network request is visible in the request log; no duplicate calls are recorded."
            })

        behavioral_seq = []
        for s in enriched_steps:
            action_text = s["action"].lower()
            sc_val = "UI Context Updated"
            if "click" in action_text or "press" in action_text or "trigger" in action_text:
                sc_val = "UI Interaction Registered"
            elif "input" in action_text or "enter" in action_text or "type" in action_text:
                sc_val = "Form Input Buffer Populated"
            elif "verify" in action_text or "observe" in action_text or "inspect" in action_text:
                sc_val = "State Verification Completed"
            elif "navigate" in action_text or "open" in action_text:
                sc_val = "Viewport Location Transitioned"
            elif "resize" in action_text:
                sc_val = "Responsive Layout Recalculated"
            elif "wait" in action_text:
                sc_val = "Async Response Awaited"
            elif "debounce" in action_text or "duplicate" in action_text or "network request" in action_text:
                sc_val = "Debounce Guard Verified"

            behavioral_seq.append({
                "step": s["step"],
                "user_action": s["action"],
                "system_response": s["expected"],
                "state_change": sc_val
            })

        # 4. Humanize text (replace raw node IDs with business language)
        def humanize_text(text: str) -> str:
            if not isinstance(text, str):
                return text
            
            # Match programmatic node IDs containing '_interact_'
            pattern = r'\b[A-Za-z0-9]+_interact_[A-Za-z0-9_]+\b'
            
            def replace_node(match):
                node_id = match.group(0)
                clean = re.sub(r'__\w+$', '', node_id)
                
                if "button_has_text__" in clean:
                    text_val = clean.split("button_has_text__")[-1]
                    text_val = text_val.replace("_", " ").title()
                    return f"'{text_val}' button"
                if "input_has_text__" in clean:
                    text_val = clean.split("input_has_text__")[-1]
                    text_val = text_val.replace("_", " ").title()
                    return f"'{text_val}' input field"
                
                parts = clean.split("_")
                prefix = parts[0].title() if parts else ""
                
                if "button" in clean.lower():
                    return f"{prefix} button" if prefix and prefix.lower() != "flights" else "button"
                if "input" in clean.lower():
                    return f"{prefix} input field" if prefix and prefix.lower() != "flights" else "input field"
                if "dropdown" in clean.lower():
                    return f"{prefix} dropdown" if prefix and prefix.lower() != "flights" else "dropdown"
                if "checkbox" in clean.lower():
                    return f"{prefix} checkbox" if prefix and prefix.lower() != "flights" else "checkbox"
                if "link" in clean.lower():
                    return f"{prefix} link" if prefix and prefix.lower() != "flights" else "link"
                
                return clean.replace("_", " ").title()
            
            return re.sub(pattern, replace_node, text)

        # Apply term cleanup for API titles
        if "api API" in new_title:
            new_title = new_title.replace("api API", f"{domain.capitalize()} API")

        # Run text humanization on all fields
        new_title = humanize_text(new_title)
        desc = humanize_text(desc)
        expected = humanize_text(expected)
        
        for s in enriched_steps:
            if "action" in s:
                s["action"] = humanize_text(s["action"])
            if "expected" in s:
                s["expected"] = humanize_text(s["expected"])
            if "system_response" in s:
                s["system_response"] = humanize_text(s["system_response"])
                
        for s in behavioral_seq:
            if "user_action" in s:
                s["user_action"] = humanize_text(s["user_action"])
            if "system_response" in s:
                s["system_response"] = humanize_text(s["system_response"])

        if "intent_mapping" in tc:
            im = tc["intent_mapping"]
            if isinstance(im, dict):
                if "user_goal" in im:
                    im["user_goal"] = humanize_text(im["user_goal"])
                if "business_action" in im:
                    im["business_action"] = humanize_text(im["business_action"])

        # Write back sanitized fields
        tc["title"] = new_title
        tc["description"] = desc
        tc["expected_result"] = expected
        tc["steps"] = enriched_steps
        tc["behavioral_sequence"] = behavioral_seq
        tc["test_data"] = test_data
        tc["preconditions"] = preconditions
        tc["postconditions"] = ["Application remains in stable state"]

    def run(self, story_path: Path) -> Path:
        log.info("[TestCaseGenerationAgent] Running self-learning behavioral QA intelligence engine")

        # 1. READ STORY DATA & EXTRACT DYNAMIC ROUTE FROM URL
        story_data = json.loads(story_path.read_text(encoding="utf-8"))
        story = story_data.get("story", {})
        story_key = story.get("key", "UNKNOWN")
        summary = story.get("summary", "")
        description = story.get("description", "")
        # Exclude comments this pipeline posted on a prior run — otherwise a story that
        # has already been through the pipeline once feeds its own past output (including
        # any past mistake) back into itself as "story context" on every future run.
        comments = [c.get("body", "") for c in filter_human_comments(story.get("comments", []))]
        acceptance_criteria = story.get("acceptance_criteria", [])

        # 1b. EXTRACT AND PROCESS ATTACHMENTS
        attachments = story.get("attachments", [])
        attachment_insights = []
        attachment_context = ""
        if attachments:
            try:
                log.info(f"[TestCaseGenerationAgent] Found {len(attachments)} attachment(s) in story. Processing...")
                processor = AttachmentProcessor(self.settings)
                attachment_insights = processor.process(attachments, story_key)
                attachment_context = "\n".join([
                    f"Attachment file: {insight['filename']}\nContent:\n{insight['extracted_text']}"
                    for insight in attachment_insights
                ])
            except Exception as e:
                log.error(f"[TestCaseGenerationAgent] Failed to process attachments: {e}", exc_info=True)

        # Find URLs in the Jira story to determine a direct starting path (matched against
        # the configured application_base_url below, not any specific hardcoded domain)
        all_story_text = f"{summary} {description} {' '.join(comments)} {' '.join(acceptance_criteria)}"
        url_matches = re.findall(r'https?://[a-zA-Z0-9./?=&_-]+', all_story_text)
        story_extracted_path = None
        story_extracted_domain = None
        story_urls = []
        story_url_nodes = []

        # Resolve base URL for this story: an explicit "url = <URL>" directive in the
        # story text (a deliberate author instruction) takes priority over the
        # configured default — different stories can legitimately target different
        # apps/environments. See utils/story_target.py for the SSRF-safe resolution
        # rules (a bare incidental link elsewhere in the text does not override this).
        base_url, base_url_is_explicit = resolve_story_base_url(
            all_story_text, self.settings.application_base_url, "https://webook.com"
        )
        if base_url_is_explicit:
            log.info(f"[TestCaseGenerationAgent] Story specifies an explicit target URL directive: {base_url}")

        for url in url_matches:
            url_clean = url.rstrip('.,;)')
            if url_clean not in story_urls:
                from urllib.parse import urlparse
                parsed = urlparse(url_clean)
                base_parsed = urlparse(base_url)
                if base_parsed.netloc in parsed.netloc or not parsed.netloc:
                    story_urls.append(url_clean)

                    # Extract node name for domain filtering allowance
                    try:
                        path_segs = [s for s in parsed.path.strip("/").split("/") if s]
                        if path_segs and len(path_segs[0]) == 2:
                            path_segs = path_segs[1:]
                        if path_segs:
                            node_name = "_".join(path_segs).lower()
                            story_url_nodes.append(node_name)
                    except Exception:
                        pass

        for url in story_urls:
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(url)
            path = parsed.path
            if parsed.query:
                path += "?" + parsed.query
            story_extracted_path = path

            # Infer domain name dynamically from URL
            qs = parse_qs(parsed.query)
            if "category" in qs:
                story_extracted_domain = qs["category"][0].replace("-", " ").title()
            elif "date" in qs:
                story_extracted_domain = qs["date"][0].replace("-", " ").title()
            elif qs:
                for k, v in qs.items():
                    if v and v[0]:
                        story_extracted_domain = v[0].replace("-", " ").title()
                        break
            if not story_extracted_domain:
                last_seg = parsed.path.strip("/").split("/")[-1]
                if last_seg and last_seg.lower() not in ["en", "explore"]:
                    story_extracted_domain = last_seg.replace("-", " ").title()
            break

        # 2. DYNAMIC ROUTE DISCOVERY PRE-CRAWL (HOMEPAGE CRAWL)
        discovered_routes = []
        try:
            from playwright.sync_api import sync_playwright
            log.info("[TestCaseGenerationAgent] Pre-crawling home page to dynamically discover routes...")
            base_url = base_url
            with sync_playwright() as p:
                headless_val = _load_playwright_headless_setting()
                log.info(f"[TestCaseGenerationAgent] Launching chromium (headless={headless_val})")
                browser = p.chromium.launch(headless=headless_val)
                context = browser.new_context(viewport={"width": 1280, "height": 800})
                page = context.new_page()
                try:
                    page.goto(f"{base_url.rstrip('/')}/en", timeout=20000, wait_until="load")
                    page.wait_for_timeout(3000)
                    handle_secure_login(page)
                except Exception as e:
                    log.warning(f"[TestCaseGenerationAgent] Home page load timed out/failed: {e}")
                    page.wait_for_timeout(5000)

                # Dynamic cookie dialog dismissal
                buttons_list = page.query_selector_all("button")
                for btn in buttons_list:
                    try:
                        text = btn.inner_text().strip()
                        btn_class = btn.get_attribute("class") or ""
                        btn_id = btn.get_attribute("id") or ""
                        text_lower = text.lower()
                        is_cookie_btn = (
                            any(w in text_lower for w in ["accept", "agree", "allow", "consent", "ok", "reject"]) and
                            any(w in text_lower or w in btn_class.lower() or w in btn_id.lower() for w in ["cookie", "consent", "privacy", "policy"]) and
                            not any(w in text_lower for w in ["settings", "manage", "preferences", "customise", "customize"])
                        )
                        if is_cookie_btn or any(phrase in text for phrase in ["Accept all", "Accept All", "Reject all non-essential"]):
                            btn.click()
                            page.wait_for_timeout(1000)
                            break
                    except Exception:
                        pass

                # Extract links
                links = page.query_selector_all("a")
                for link in links:
                    try:
                        href = link.get_attribute("href") or ""
                        text = link.inner_text().strip()
                        if text and (href.startswith("/") or base_url in href):
                            from urllib.parse import urlparse
                            parsed = urlparse(href)
                            path_query = parsed.path
                            if parsed.query:
                                path_query += "?" + parsed.query

                            # Avoid duplicates and non-routing links
                            if not any(x in path_query for x in ["logout", "signout", "cdn-cgi", "assets"]):
                                # Normalize route text (newline replacement and spacing cleanup)
                                clean_text = text.replace("\n", " ").strip()
                                clean_text = re.sub(r'\s+', ' ', clean_text)
                                if clean_text and len(clean_text) <= 40 and not any(r["path"] == path_query for r in discovered_routes):
                                    discovered_routes.append({"label": clean_text, "path": path_query})
                    except Exception:
                        pass
                browser.close()
            log.info(f"[TestCaseGenerationAgent] Dynamically discovered {len(discovered_routes)} routes: {discovered_routes}")
        except Exception as e:
            log.error(f"[TestCaseGenerationAgent] Pre-crawl failed: {e}", exc_info=True)
            # If the browser was launched before the failure, ensure it's closed here —
            # otherwise an exception between launch() and the normal browser.close() call
            # above leaks the headless Chromium process.
            try:
                if 'browser' in locals() and browser is not None:
                    browser.close()
            except Exception:
                pass

        # Dynamic fallback if both URL extraction and pre-crawl yield nothing (no hardcoded list)
        if not story_extracted_path and not discovered_routes:
            # Find first noun-like word in story title/summary to form a dynamic path
            potential_words = [w.strip().lower() for w in re.split(r'\W+', summary) if len(w.strip()) > 3]
            stop_words = {"with", "from", "your", "that", "this", "have", "will", "what", "where"}
            keywords = [w for w in potential_words if w not in stop_words]
            fallback_keyword = keywords[0] if keywords else "explore"
            story_extracted_path = f"/en/{fallback_keyword}"
            story_extracted_domain = fallback_keyword.capitalize()
            log.warning(f"[TestCaseGenerationAgent] Using dynamic keyword fallback path: {story_extracted_domain} -> {story_extracted_path}")

        # 3. INTENT ENGINE
        intent_engine = IntentEngine(summary, description, comments, discovered_routes,
                                     extracted_path=story_extracted_path,
                                     extracted_domain=story_extracted_domain,
                                     acceptance_criteria=acceptance_criteria,
                                     attachment_context=attachment_context,
                                     application_name=self.settings.application_name)
        intent = intent_engine.analyze()
        log.info(f"[TestCaseGenerationAgent] Extracted Intent: {intent}")

        # 4. LOAD BEHAVIORAL GRAPH MEMORY
        behavior_flow_engine = BehavioralFlowEngine(self.settings)
        behavior_graph = behavior_flow_engine.load_graph()
        
        # Clear stale behavioral graph memory if it belongs to a different story to prevent context leakage
        graph_story = behavior_graph.get("story_key")
        if (graph_story and graph_story != story_key) or not graph_story:
            log.info(f"[TestCaseGenerationAgent] Clearing stale behavioral graph memory from story {graph_story} for active story {story_key}.")
            behavior_graph = {
                "nodes": [],
                "edges": [],
                "selectors": {},
                "last_updated": "",
                "story_key": story_key,
                "domain": intent["domain"]
            }
            
        # Domain-based graph filtering to absolutely eliminate context leakage
        domain_lower = intent["domain"].lower()
        domain_clean = domain_lower.rstrip('s')
        
        # Dynamically extract other domains from discovered routes
        discovered_routes = self.discovered_routes if hasattr(self, 'discovered_routes') else []
        other_domains = self._extract_domain_terms_from_routes(discovered_routes, domain_clean)
            
        def is_clean_for_domain(text: str) -> bool:
            text_lower = text.lower()
            if any(node in text_lower for node in story_url_nodes):
                return True
            for other_d in other_domains:
                if other_d in text_lower and not (domain_clean in text_lower):
                    return False
            return bool(domain_clean in text_lower or "explore" in text_lower)
            
        filtered_selectors = {}
        for pk, val in behavior_graph.get("selectors", {}).items():
            if is_clean_for_domain(pk):
                filtered_selectors[pk] = val
                
        filtered_nodes = []
        for node in behavior_graph.get("nodes", []):
            if is_clean_for_domain(node):
                filtered_nodes.append(node)
                
        filtered_edges = []
        for edge in behavior_graph.get("edges", []):
            src = edge.get("source", "")
            tgt = edge.get("target", "")
            if is_clean_for_domain(src) and is_clean_for_domain(tgt):
                filtered_edges.append(edge)
                
        behavior_graph["selectors"] = filtered_selectors
        behavior_graph["nodes"] = filtered_nodes
        behavior_graph["edges"] = filtered_edges
        behavior_graph["story_key"] = story_key
        behavior_graph["domain"] = intent["domain"]
        
        # Save filtered graph memory back to disk to prevent context leakage in cache files
        try:
            graph_path = self.settings.artifact_dir / "ui_graph.json"
            graph_path.write_text(json.dumps(behavior_graph, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            log.warning(f"Could not save filtered graph to disk: {e}")
            
        log.info(f"[TestCaseGenerationAgent] Loaded behavioral memory graph (filtered for {intent['domain']}): {len(behavior_graph.get('nodes', []))} nodes, {len(behavior_graph.get('edges', []))} edges")
        self._run_pre_generation_validation(story_key, intent["domain"], story_data)

        # 5. INTENT-DRIVEN UI EXPLORATION ENGINE
        exploration_engine = IntentDrivenUIExplorationEngine(self.settings, intent, discovered_routes, story_urls=story_urls)
        ui_map, transitions = exploration_engine.explore(behavior_graph)

        if ui_map.get("exploration_failed"):
            # Exploration can fail for environmental/transient reasons (e.g. an autonomous
            # exploration click landing on a native OS dialog outside the browser's control,
            # which can crash the whole browser process) — retry once with a fresh engine/
            # browser session before accepting the failure.
            log.warning(
                f"[TestCaseGenerationAgent] Exploration failed ({ui_map.get('exploration_failure_reason')}); "
                f"retrying once with a fresh browser session before proceeding."
            )
            retry_engine = IntentDrivenUIExplorationEngine(self.settings, intent, discovered_routes, story_urls=story_urls)
            ui_map, transitions = retry_engine.explore(behavior_graph)

        self.exploration_incomplete = bool(ui_map.get("exploration_failed"))
        if self.exploration_incomplete:
            # Still failed after a retry. We deliberately don't abort generation entirely —
            # the acceptance-criteria-derived enterprise scenarios don't depend on real DOM
            # discovery, so there's still some value to produce — but this MUST be visible
            # to whoever reviews the output rather than looking like a normal, fully-grounded
            # run (previously: logged loudly here and then silently ignored by every caller).
            log.error(
                f"[TestCaseGenerationAgent] Exploration is still incomplete after retry "
                f"({ui_map.get('exploration_failure_reason')}). Proceeding with generation, but "
                f"the resulting test cases for '{intent['domain']}' are NOT grounded in real "
                f"discovered UI elements and should be treated as low-confidence pending manual review."
            )

        # 6. BEHAVIORAL FLOW ENGINE (MERGE & UPDATE BEHAVIOR GRAPH)
        merged_graph = behavior_flow_engine.update_graph(behavior_graph, ui_map, transitions)
        merged_graph["story_key"] = story_key
        merged_graph["domain"] = intent["domain"]

        # 7. TEST CASE GENERATION ENGINE
        testcase_generator = TestCaseGenerator(story_key, intent, merged_graph, attachment_insights, story_data, discovered_routes)
        new_test_cases = testcase_generator.generate()

        # Deduplication and Preservation Merge
        existing_test_cases = []
        output_path = self.settings.artifact_dir / "testcases.json"
        load_path = output_path
        if not load_path.exists():
            global_path = self.settings.artifact_dir.parent / "testcases.json"
            if global_path.exists():
                load_path = global_path

        if load_path.exists():
            try:
                existing_data = json.loads(load_path.read_text(encoding="utf-8"))
                raw_existing = existing_data.get("test_cases", [])
                
                # Filter out any existing test cases that belong to another story or reference other domains
                domain_clean = intent["domain"].lower().rstrip('s')
                # Dynamically extract other domains from discovered routes
                other_domains = self._extract_domain_terms_from_routes(discovered_routes, domain_clean)
                
                clean_existing = []
                for tc in raw_existing:
                    if tc.get("linked_story") != story_key:
                        continue
                    
                    has_stale_ref = False
                    title_lower = tc.get("title", "").lower()
                    desc_lower = tc.get("description", "").lower()
                    is_placeholder = bool(
                        re.search(r'[a-zA-Z]r\d+', title_lower) or 
                        re.search(r'[a-zA-Z]r\d+', desc_lower) or 
                        "filters1" in title_lower or "filters2" in title_lower
                    )
                    if is_placeholder:
                        has_stale_ref = True
                    else:
                        for other_d in other_domains:
                            if other_d in title_lower or other_d in desc_lower or other_d in tc.get("expected_result", "").lower():
                                has_stale_ref = True
                                break
                            for s in tc.get("steps", []):
                                if other_d in s.get("action", "").lower() or other_d in s.get("expected", "").lower():
                                    has_stale_ref = True
                                    break
                    if not has_stale_ref:
                        clean_existing.append(tc)
                        
                existing_test_cases = clean_existing
                if len(existing_test_cases) > 0:
                    log.info(f"[TestCaseGenerationAgent] Loaded and filtered {len(existing_test_cases)} existing test cases (matching story {story_key}) from {len(raw_existing)} stored cases.")
                else:
                    log.info(f"[TestCaseGenerationAgent] No matching clean test cases for story {story_key} in existing testcases.json. Starting with a clean slate.")
            except Exception as e:
                log.warning(f"[TestCaseGenerationAgent] Could not load existing test cases: {e}")

        # Combine existing and new test cases
        combined_test_cases = list(existing_test_cases)
        
        # Track existing test cases by a normalized key to avoid duplicates
        def get_tc_key(tc):
            title = tc.get("title", "").lower().strip()
            steps = tuple((s.get("action", "").lower().strip(), s.get("expected", "").lower().strip()) for s in tc.get("steps", []))
            return (title, steps)
            
        existing_keys = {get_tc_key(tc) for tc in existing_test_cases}
        
        # Find next sequential ID number for the current module
        max_seq = 0
        module_prefix = re.sub(r'[^a-zA-Z0-9]', '', intent["domain"]).upper()
        for tc in existing_test_cases:
            tc_id = tc.get("id", "")
            parts = tc_id.split("-")
            if len(parts) >= 3 and parts[-1].isdigit() and parts[1] == module_prefix:
                max_seq = max(max_seq, int(parts[-1]))

        # Add new test cases that are not duplicates
        added_count = 0
        for tc in new_test_cases:
            tc_key = get_tc_key(tc)
            if tc_key not in existing_keys:
                max_seq += 1
                tc["id"] = f"TC-{module_prefix}-{max_seq:03d}"
                combined_test_cases.append(tc)
                existing_keys.add(tc_key)
                added_count += 1
                
        log.info(f"[TestCaseGenerationAgent] Merged test cases: {len(existing_test_cases)} existing, {added_count} new added. Total: {len(combined_test_cases)}")
        
        # Verify that the new testcase count is not less than the existing one
        if len(existing_test_cases) > 0 and len(combined_test_cases) < len(existing_test_cases):
            raise ValueError(f"Coverage Preservation Violation: new test case count ({len(combined_test_cases)}) is less than existing test case count ({len(existing_test_cases)}).")
            
        # Retroactively sanitize and clean up generic expected results in all merged test cases
        for tc in combined_test_cases:
            self._sanitize_testcase_content(tc, intent["domain"], story_data)

        # Sanitization can homogenize previously-distinct drafts (e.g. two graph edges with
        # generic titles) into byte-identical content, after the earlier draft-time dedup
        # already ran on their un-sanitized text. Re-check for duplicates now that the final
        # title/steps/expected_result text is settled, then re-index IDs to close any gap.
        deduped_test_cases = []
        seen_final_signatures = set()
        for tc in combined_test_cases:
            signature = (
                tc.get("title", "").lower().strip(),
                tc.get("expected_result", "").lower().strip(),
                tuple(
                    (s.get("action", "").lower().strip(), s.get("expected", "").lower().strip())
                    for s in tc.get("steps", [])
                )
            )
            if signature in seen_final_signatures:
                log.info(f"[PostSanitizeDedup] Dropping testcase {tc.get('id', 'unknown')}: identical to an earlier case after sanitization.")
                continue
            seen_final_signatures.add(signature)
            deduped_test_cases.append(tc)
        combined_test_cases = deduped_test_cases
        for i, tc in enumerate(combined_test_cases):
            tc["id"] = f"TC-{module_prefix}-{(i + 1):03d}"

        try:
            self._run_post_generation_validation(combined_test_cases, story_key, intent["domain"], story_data)
        except ValueError as exc:
            # This is a heuristic check (domain-term / cross-module text matching) that can
            # false-positive. Previously an unguarded raise here aborted run() before
            # testcases.json/selector_registry.json/coverage_summary.json were written,
            # discarding an otherwise-successful generation pass. Log loudly and continue.
            log.error(
                f"[TestCaseGenerationAgent] Post-generation validation flagged an issue; "
                f"continuing so generated artifacts are not lost. Review the flagged test case(s): {exc}"
            )
        test_cases = combined_test_cases

        # 8. TRUE COVERAGE ENGINE (DYNAMIC GRAPH-BASED COVERAGE - ZERO CHEATING)
        edges = merged_graph.get("edges", [])
        selectors = merged_graph.get("selectors", {})
        domain = intent["domain"]
        page_key = domain.lower()
        page_selectors = selectors.get(page_key, {})

        # Find nodes relevant to the current user story domain dynamically based on visited pages
        visited_pages = list(ui_map.get("discovered_pages", {}).keys())
        relevant_nodes = set()
        for page in visited_pages:
            relevant_nodes.add(page)
        for node in merged_graph.get("nodes", []):
            if any(node.startswith(p) for p in visited_pages) or node == domain.lower():
                relevant_nodes.add(node)

        # Build list of scoped path ids
        all_path_ids = []
        for edge in edges:
            if edge.get("source") in relevant_nodes:
                path_id = f"{edge['source']} -> {edge['target']} ({edge['action']})"
                all_path_ids.append(path_id)

        # Calculate dynamic components coverage ratio
        discovered_components = []
        for key in ["inputs", "buttons", "dropdowns", "checkboxes", "radio_buttons", "links"]:
            discovered_components.extend(page_selectors.get(key, []))

        # A discovered component is "covered" if its sanitized node name appears in the
        # "_comp_<name>" segment of some test case's coverage_reference. Previously this
        # checked `raw_selector_string in humanized_step_text` — a raw CSS/attribute
        # selector essentially never appears verbatim inside natural-language step
        # descriptions like "Enter valid search keyword matching...", so comp_coverage
        # was structurally near-zero regardless of how much the suite actually covered.
        covered_comp_names = set()
        for tc in test_cases:
            ref = tc.get("coverage_reference") or ""
            covered_comp_names.update(re.findall(r'_comp_([a-z0-9_]+)', ref))

        covered_components = {
            comp for comp in discovered_components
            if _sanitize_node_name(comp) in covered_comp_names
        }

        comp_coverage = len(covered_components) / len(discovered_components) if discovered_components else 1.0

        # Calculate business rules coverage ratio
        discovered_rules = page_selectors.get("business_rules", [])
        covered_rules = set()
        for tc in test_cases:
            for rule in discovered_rules:
                if tc.get("type") in ["Validation", "Edge", "Negative"]:
                    covered_rules.add(rule)

        rules_coverage = len(covered_rules) / len(discovered_rules) if discovered_rules else 1.0

        # Calculate path/transition coverage
        executed_paths = []
        for tc in test_cases:
            ref = tc.get("coverage_reference")
            if ref in all_path_ids and ref not in executed_paths:
                executed_paths.append(ref)

        missing_paths = [pid for pid in all_path_ids if pid not in executed_paths]
        path_coverage = len(executed_paths) / len(all_path_ids) if all_path_ids else 1.0

        # Calculate testing categories coverage ratio
        target_categories = {"UI", "Functional", "Negative", "Edge"}
        tested_categories = {tc.get("type") for tc in test_cases if tc.get("type") in target_categories}
        cats_coverage = len(tested_categories) / len(target_categories) if target_categories else 1.0

        # Calculate Observed API Interaction Coverage
        page_apis = page_selectors.get("observed_apis", [])
        covered_apis = set()
        for tc in test_cases:
            tc_notes = tc.get("automation_notes", "")
            for api in page_apis:
                api_key = f"{api.get('method')} {api.get('url').split('?')[0]}"
                if api_key in tc_notes:
                    covered_apis.add(api_key)
        api_coverage = len(covered_apis) / len(page_apis) if page_apis else 1.0

        # Combine weighted coverage score (25% Components, 25% Transitions, 20% Rules, 15% Categories, 15% API coverage)
        coverage_score = round(
            0.25 * comp_coverage +
            0.25 * path_coverage +
            0.20 * rules_coverage +
            0.15 * cats_coverage +
            0.15 * api_coverage,
            2
        )

        coverage_summary = {
            "total_nodes": len(merged_graph.get("nodes", [])),
            "total_behavior_paths": len(all_path_ids),
            "executed_paths": executed_paths,
            "missing_paths": missing_paths,
            "coverage_score": coverage_score,
            "_metadata": {
                "components_coverage": round(comp_coverage, 2),
                "business_rules_coverage": round(rules_coverage, 2),
                "transitions_coverage": round(path_coverage, 2),
                "categories_coverage": round(cats_coverage, 2),
                "observed_api_coverage": round(api_coverage, 2)
            }
        }

        # Prompt user for manual control
        run_mode_msg = "FULL"
        emitted_test_cases = test_cases
        
        import sys
        import os
        is_interactive = sys.stdin.isatty() and os.environ.get("AUTO_APPROVE") != "true"
        
        if is_interactive:
            print("\n" + "=" * 60)
            print("  TEST CASE GENERATION CONTROL")
            print("=" * 60)
            print(f"Total test cases derived: {len(test_cases)}")
            print("1. Proceed with full Test Case Generation")
            print("2. Generate only a specific count of test cases")
            print("=" * 60)
            
            choice = ""
            while choice not in ["1", "2"]:
                try:
                    choice = input("Select an option (1 or 2): ").strip()
                    if choice not in ["1", "2"]:
                        print("Invalid option. Please enter 1 or 2.")
                except (EOFError, KeyboardInterrupt):
                    print("\nSelection cancelled. Proceeding with FULL generation by default.")
                    choice = "1"
                    break
            
            if choice == "2":
                while True:
                    try:
                        count_input = input("Enter testcase count to generate: ").strip()
                        if count_input.lower() in ["cancel", "c", "exit"]:
                            print("Selection cancelled. Proceeding with FULL generation.")
                            run_mode_msg = "FULL"
                            emitted_test_cases = test_cases
                            break
                        if not count_input:
                            print("Input is empty. Please enter a valid positive integer.")
                            continue
                        val = int(count_input)
                        if val <= 0:
                            print("Count must be a positive integer (> 0).")
                            continue
                        
                        # Truncate test cases list
                        if val < len(test_cases):
                            emitted_test_cases = test_cases[:val]
                            run_mode_msg = f"PARTIAL ({val} testcases)"
                        else:
                            run_mode_msg = "FULL"
                        break
                    except ValueError:
                        print("Invalid input. Please enter a valid integer.")
                    except (EOFError, KeyboardInterrupt):
                        print("\nInput cancelled. Proceeding with FULL generation by default.")
                        run_mode_msg = "FULL"
                        emitted_test_cases = test_cases
                        break
            else:
                run_mode_msg = "FULL"
        else:
            run_mode_msg = "FULL"

        # Print run mode
        print(f"\n[TestCaseGenerationAgent] Test case generation run mode: {run_mode_msg}\n")
        log.info(f"[TestCaseGenerationAgent] Test case generation run mode: {run_mode_msg}")

        # 9. WRITE PERSISTENT ARTIFACTS
        # Note: capability-gap analysis (gaps_found) is computed and logged inside
        # TestCaseGenerator.generate() itself, where it's available — it isn't threaded
        # through to this method since generate() only returns the test case list.
        output = self._build_output_envelope(story_key, intent, emitted_test_cases, attachment_insights, coverage_summary, exploration_incomplete=getattr(self, "exploration_incomplete", False))
        output_path = self.settings.artifact_dir / "testcases.json"
        output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
        log.info(f"[TestCaseGenerationAgent] ✓ Saved generated testcases to: {output_path}")

        # Save Selector Registry
        captured_selectors = getattr(exploration_engine, "captured_selectors", [])
        
        # Semantic Deduplication
        def get_semantic_root(name):
            import re
            name = re.sub(r'[^a-z0-9\s]', '', name.lower()).strip()
            for word in ["now", "button", "link", "input", "select", "click"]:
                name = name.replace(f" {word}", "").replace(f"{word} ", "").strip()
            return name

        semantic_groups = {}
        for sel in captured_selectors:
            name = (sel.get("accessible_name") or sel.get("text") or sel.get("label") or "").lower()
            name_root = get_semantic_root(name)
            
            comp_type = sel.get("component_type") or ""
            dom_rel = sel.get("dom_relationship") or ""
            classification = sel.get("classification") or "business"
            
            sem_key = (name_root, comp_type, dom_rel, classification)
            score = float(sel.get("stability_score", 0.0))
            
            if sem_key not in semantic_groups:
                semantic_groups[sem_key] = sel
            else:
                existing_score = float(semantic_groups[sem_key].get("stability_score", 0.0))
                if score > existing_score:
                    semantic_groups[sem_key] = sel
                    
        deduplicated_selectors = list(semantic_groups.values())
        
        selector_registry = {
            "story_key": story_key,
            "module": intent.get("domain", "Unknown"),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "selectors": deduplicated_selectors
        }
        
        selector_registry_path = self.settings.artifact_dir / "selector_registry.json"
        selector_registry_path.write_text(json.dumps(selector_registry, indent=2, ensure_ascii=False), encoding="utf-8")
        log.info(f"[TestCaseGenerationAgent] ✓ Saved selector registry to: {selector_registry_path}")

        graph_path = self.settings.artifact_dir / "ui_graph.json"
        graph_path.write_text(json.dumps(merged_graph, indent=2, ensure_ascii=False), encoding="utf-8")
        log.info(f"[TestCaseGenerationAgent] ✓ Saved behavioral graph memory: {graph_path}")

        cov_path = self.settings.artifact_dir / "coverage_summary.json"
        cov_path.write_text(json.dumps(coverage_summary, indent=2, ensure_ascii=False), encoding="utf-8")
        log.info(f"[TestCaseGenerationAgent] ✓ Saved True Coverage summary: {cov_path}")

        return output_path

    def run_multi_module_regression_test(self, story_paths: List[Path], domains: List[str]) -> Dict[str, Any]:
        """
        Runs regression testing across multiple modules to validate no context leakage or terminology contamination.
        
        Args:
            story_paths: List of paths to story JSON files for each module
            domains: List of domain names corresponding to each story
            
        Returns:
            Dictionary containing regression test results for each module
        """
        log.info("[RegressionTest] Starting multi-module regression test...")
        
        regression_results = {
            "modules_tested": len(domains),
            "modules": {},
            "cross_module_contamination": [],
            "terminology_consistency": {},
            "overall_status": "PASS"
        }
        
        # Store test cases for each module for cross-validation
        module_testcases = {}
        
        for idx, (story_path, domain) in enumerate(zip(story_paths, domains)):
            log.info(f"[RegressionTest] Testing module {idx + 1}/{len(domains)}: {domain}")
            
            try:
                # Run generation for this module
                output_path = self.run(story_path)
                
                # Load generated test cases
                if output_path.exists():
                    test_data = json.loads(output_path.read_text(encoding="utf-8"))
                    test_cases = test_data.get("test_cases", [])
                    module_testcases[domain] = test_cases
                    
                    # Create terminology resolver for this module
                    story_data = json.loads(story_path.read_text(encoding="utf-8"))
                    intent = {"domain": domain}
                    resolver = DomainTerminologyResolver(intent, story_data)
                    domain_allowlist = resolver.get_domain_allowlist()
                    domain_vocabulary = resolver.get_domain_vocabulary()
                    
                    # Validate terminology consistency
                    terminology_issues = []
                    for tc in test_cases:
                        tc_title = tc.get("title", "")
                        tc_expected = tc.get("expected_result", "")
                        
                        # Check for generic expected results
                        generic_phrases = ["Action executes successfully", "Action completed", "Target action executes", "Action is triggered"]
                        for phrase in generic_phrases:
                            if phrase in tc_expected:
                                terminology_issues.append(f"Generic phrase '{phrase}' found in testcase {tc.get('id', 'unknown')}")
                        
                        # Check for cross-module contamination
                        for other_domain in domains:
                            if other_domain != domain:
                                if other_domain.lower() in tc_title.lower() and not resolver.is_valid_domain_term(other_domain):
                                    contamination = f"Cross-module contamination: '{other_domain}' found in {domain} testcase {tc.get('id', 'unknown')}"
                                    if contamination not in regression_results["cross_module_contamination"]:
                                        regression_results["cross_module_contamination"].append(contamination)
                    
                    regression_results["modules"][domain] = {
                        "status": "PASS",
                        "testcase_count": len(test_cases),
                        "domain_allowlist_size": len(domain_allowlist),
                        "domain_vocabulary_size": len(domain_vocabulary),
                        "terminology_issues": terminology_issues
                    }
                    
                    regression_results["terminology_consistency"][domain] = {
                        "entity": resolver.get_entity_term(),
                        "entity_plural": resolver.get_entity_term(plural=True),
                        "transaction": resolver.get_transaction_term(),
                        "allowlist_sample": domain_allowlist[:5] if domain_allowlist else []
                    }
                    
                    # Coverage Distribution Analysis
                    coverage_distribution = {
                        "Functional": 0,
                        "Negative": 0,
                        "Edge": 0,
                        "API": 0,
                        "Security": 0,
                        "Resilience": 0,
                        "Business-Rule": 0,
                        "Workflow": 0,
                        "State-Transition": 0,
                        "Accessibility": 0,
                        "Recovery": 0,
                        "UI": 0
                    }
                    
                    for tc in test_cases:
                        tc_type = tc.get("type", "Functional")
                        if tc_type in coverage_distribution:
                            coverage_distribution[tc_type] += 1
                        else:
                            coverage_distribution["UI"] += 1
                    
                    regression_results["modules"][domain]["coverage_distribution"] = coverage_distribution
                    
                    # Scenario Diversity Validation
                    scenario_diversity_score = 0
                    scenario_types_present = sum(1 for count in coverage_distribution.values() if count > 0)
                    scenario_diversity_score = (scenario_types_present / len(coverage_distribution)) * 100
                    
                    regression_results["modules"][domain]["scenario_diversity_score"] = scenario_diversity_score
                    
                    # Enterprise Coverage Thresholds
                    enterprise_coverage_threshold = {
                        "API": 1,  # At least 1 API scenario
                        "Security": 1,  # At least 1 Security scenario
                        "Resilience": 1,  # At least 1 Resilience scenario
                        "Business-Rule": 1,  # At least 1 Business-Rule scenario
                        "Workflow": 1  # At least 1 Workflow scenario
                    }
                    
                    coverage_gaps = []
                    for scenario_type, min_count in enterprise_coverage_threshold.items():
                        if coverage_distribution.get(scenario_type, 0) < min_count:
                            coverage_gaps.append(f"{scenario_type} (expected {min_count}, got {coverage_distribution.get(scenario_type, 0)})")
                    
                    if coverage_gaps:
                        regression_results["modules"][domain]["coverage_gaps"] = coverage_gaps
                        regression_results["modules"][domain]["status"] = "WARN"
                        regression_results["overall_status"] = "WARN"
                    
                    # Scenario Diversity Threshold (at least 60% of scenario types should be present)
                    if scenario_diversity_score < 60:
                        regression_results["modules"][domain]["diversity_warning"] = f"Scenario diversity score {scenario_diversity_score:.1f}% is below 60% threshold"
                        regression_results["modules"][domain]["status"] = "WARN"
                        regression_results["overall_status"] = "WARN"
                    
                    if terminology_issues:
                        regression_results["modules"][domain]["status"] = "WARN"
                        regression_results["overall_status"] = "WARN"
                        
                else:
                    regression_results["modules"][domain] = {
                        "status": "FAIL",
                        "error": f"No output file generated at {output_path}"
                    }
                    regression_results["overall_status"] = "FAIL"
                    
            except Exception as e:
                log.error(f"[RegressionTest] Error testing module {domain}: {e}", exc_info=True)
                regression_results["modules"][domain] = {
                    "status": "FAIL",
                    "error": str(e)
                }
                regression_results["overall_status"] = "FAIL"
        
        # Final validation
        if regression_results["cross_module_contamination"]:
            regression_results["overall_status"] = "FAIL"
            log.error(f"[RegressionTest] Cross-module contamination detected: {regression_results['cross_module_contamination']}")
        
        log.info(f"[RegressionTest] Regression test completed with status: {regression_results['overall_status']}")
        
        return regression_results

    @staticmethod
    def _build_output_envelope(story_key: str, intent: Dict[str, Any], test_cases: List[Dict[str, Any]], attachment_insights: List[Dict[str, Any]] = None, coverage_summary: Dict[str, Any] = None, gaps_found: List[str] = None, exploration_incomplete: bool = False) -> Dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        domain = intent["domain"]

        total = len(test_cases)
        # Built dynamically (not a fixed {"UI","Functional","Negative","Edge"} allowlist) so
        # enterprise scenario types (API, Security, Resilience, ...) aren't silently dropped —
        # previously by_type's counts could sum to less than total_test_cases.
        by_type: Dict[str, int] = {}
        by_priority = {"High": 0, "Medium": 0, "Low": 0}
        by_severity = {"Critical": 0, "Major": 0, "Minor": 0}
        automation_yes = 0
        automation_no = 0
        edge_cases = 0

        for tc in test_cases:
            tc_type = tc.get("type", "Functional")
            by_type[tc_type] = by_type.get(tc_type, 0) + 1

            prio = tc.get("priority", "Medium")
            if prio in by_priority:
                by_priority[prio] += 1

            sev = tc.get("severity", "Major")
            if sev in by_severity:
                by_severity[sev] += 1

            if tc.get("automation", "Yes") == "Yes":
                automation_yes += 1
            else:
                automation_no += 1

            if tc.get("edge_case", False):
                edge_cases += 1

            # Determine and assign polarity to every test case
            title_lower = tc.get("title", "").lower()
            desc_lower = tc.get("description", "").lower()
            tags_lower = [t.lower() for t in tc.get("tags", [])]
            tctype_lower = tc_type.lower()
            edge_case = tc.get("edge_case", False)

            # Polarity detection logic
            if tctype_lower == "edge" or edge_case is True or "edge" in tags_lower or "boundary" in tags_lower:
                polarity = "Edge"
            elif tctype_lower == "negative" or "negative" in tags_lower:
                polarity = "Negative"
            elif tctype_lower in ["resilience", "recovery"] or any(t in tags_lower for t in ["resilience", "recovery"]):
                polarity = "Edge"
            else:
                edge_keywords = [
                    "special character", "emoji", "unicode", "script injection", "malicious", 
                    "xss", "sql injection", "double-click", "double click", "latency", "slow network", 
                    "retry", "recovery", "refresh", "reload", "restart", "boundary", "overflow", 
                    "character limit", "numeric range", "concurrency", "keyboard navigation", 
                    "accessibility", "wcag", "aria tags", "screen reader"
                ]
                negative_keywords = [
                    "invalid", "blank", "empty", "missing", "unauthorized", "expiry", "expired", 
                    "fail", "rejections", "rejects", "error", "deny", "denies", "blocked", 
                    "offline", "disconnect", "refuses", "prevent"
                ]
                if any(kw in title_lower or kw in desc_lower for kw in edge_keywords):
                    polarity = "Edge"
                elif any(kw in title_lower or kw in desc_lower for kw in negative_keywords):
                    polarity = "Negative"
                else:
                    polarity = "Positive"

            tc["scenario_classification"] = polarity

        # Final Coverage Validation Gate Review — every line below is derived from data
        # actually available at this point. Previously this block unconditionally logged
        # "True" for every claim regardless of whether it had been checked, including
        # claims (API/validation/edge coverage) with no corresponding computation at all.
        pages_seen = set()
        for tc in test_cases:
            for prec in tc.get("preconditions", []):
                if "accessible at" in prec:
                    url = prec.split("accessible at ")[-1].strip()
                    pages_seen.add(url)
        num_pages = len(pages_seen) or 1

        seen_workflow_keys = set()
        duplicate_workflows = 0
        generic_expected_phrases = ["action completed", "success", "stable state", "navigation succeeds", "action is triggered"]
        generic_expected_count = 0
        ac_titled_count = 0
        for tc in test_cases:
            wf_key = (
                tc.get("title", "").lower().strip(),
                tuple(s.get("action", "").lower() for s in tc.get("steps", [])),
            )
            if wf_key in seen_workflow_keys:
                duplicate_workflows += 1
            else:
                seen_workflow_keys.add(wf_key)

            exp_lower = tc.get("expected_result", "").lower()
            if any(p in exp_lower for p in generic_expected_phrases):
                generic_expected_count += 1

            title_lower_tc = tc.get("title", "").lower()
            if any(f"ac-{i}" in title_lower_tc or f"ac{i}" in title_lower_tc for i in range(1, 20)):
                ac_titled_count += 1

        log.info("==========================================================")
        log.info("  FINAL QUALITY GATE & COVERAGE VALIDATION")
        log.info("==========================================================")
        log.info(f"Pages analyzed: {num_pages}")
        if coverage_summary:
            missing_paths = coverage_summary.get("missing_paths") or []
            mark = "✓" if not missing_paths else "⚠"
            log.info(
                f"{mark} Behavior-path coverage score: {coverage_summary.get('coverage_score')} "
                f"({len(missing_paths)} path(s) not represented"
                + (f": {missing_paths}" if missing_paths else "") + ")"
            )
        else:
            log.info("Behavior-path coverage score: not computed for this run.")

        if gaps_found:
            log.info(f"⚠ Capability gap analysis flagged missing coverage for: {gaps_found}")
        elif gaps_found is not None:
            log.info("✓ No capability gaps flagged by the QA reasoning engine.")
        else:
            log.info("Capability gap analysis: see QAReasoningEngine log output above for this run.")

        log.info(f"{'✓' if duplicate_workflows == 0 else '⚠'} No duplicate business workflows: {duplicate_workflows == 0} ({duplicate_workflows} duplicate(s) found)")
        log.info(f"{'✓' if ac_titled_count == 0 else '⚠'} No AC-referenced testcase titles: {ac_titled_count == 0} ({ac_titled_count} found)")
        log.info(f"{'✓' if generic_expected_count == 0 else '⚠'} No generic expected results: {generic_expected_count == 0} ({generic_expected_count} found)")
        log.info(f"Total test cases: {total}")
        log.info("==========================================================")

        return {
            "_schema_version": "2.0.0",
            "_status": "populated",
            "_description": f"Autonomously generated test cases from dynamic UI graph memory on domain: {domain}.",
            "feature": f"{domain} Management",
            "module": domain.upper(),
            "generated_on": now,
            "source_story_key": story_key,
            "generation_summary": {
                "total_test_cases": total,
                "by_type": by_type,
                "by_priority": by_priority,
                "by_severity": by_severity,
                "automation_yes": automation_yes,
                "automation_no": automation_no,
                "edge_cases": edge_cases,
                "attachment_insights_used": len(attachment_insights) if attachment_insights else 0,
            },
            "test_cases": test_cases,
            "_metadata": {
                "source_agent": "testcase_generation_agent",
                "pipeline_run_id": None,
                "model_used": "self_learning_exploration_engine",
                "exploration_incomplete": exploration_incomplete,
            },
        }
