"""
agents/defect_agent.py
======================
Defect Agent — Analyzes failed test execution reports, classifies defect types,
performs root cause analysis (RCA), and generates defect artifacts.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Any, List

from config.settings import Settings
from utils.logger import get_logger

log = get_logger(__name__)


class DefectAgent:
    """
    Analyzes execution failures from current story run, performs RCA,
    groups failures into defect candidates, and writes defect reports.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.execution_dir = self.settings.execution_dir
        self.defect_dir = self.settings.defect_dir

    def _analyze_execution_evidence(
        self, 
        failing_step: Dict[str, Any], 
        exc_msg: str, 
        actual_msg: str,
        tc_title: str
    ) -> Dict[str, str]:
        """
        Deeply analyzes execution evidence to extract the exact problem description.
        Uses failed step text, expected vs actual, and error context to infer the actual issue.
        """
        # `.get(key, default)` only applies the default when the key is *missing* — an
        # execution report with an explicit `"action": null` would otherwise flow a None
        # into action_text.lower() below and crash. `.get(key) or default` also covers
        # explicit null/empty values.
        action_text = failing_step.get("action") or ""
        expected = failing_step.get("expected") or ""
        step_num = failing_step.get("step") or 1
        
        analysis = {
            "problem_type": "unknown",
            "target_element": "unknown",
            "action_attempted": "unknown",
            "failure_reason": "unknown",
            "business_impact": "unknown"
        }
        
        # Extract action attempted
        action_lower = action_text.lower()
        if "click" in action_lower:
            analysis["action_attempted"] = "click interaction"
        elif "type" in action_lower or "input" in action_lower or "enter" in action_lower:
            analysis["action_attempted"] = "text input"
        elif "select" in action_lower:
            analysis["action_attempted"] = "selection"
        elif "hover" in action_lower:
            analysis["action_attempted"] = "hover interaction"
        elif "navigate" in action_lower or "goto" in action_lower:
            analysis["action_attempted"] = "navigation"
        elif "wait" in action_lower:
            analysis["action_attempted"] = "wait operation"
        else:
            analysis["action_attempted"] = "interaction"
        
        # Extract target element from action
        quoted_match = re.search(r"'([^']+)'", action_text)
        if quoted_match:
            target = quoted_match.group(1)
            # Check if it's a date value
            if any(month in target.lower() for month in ['january', 'february', 'march', 'april', 'may', 'june', 
                                                       'july', 'august', 'september', 'october', 'november', 'december']):
                analysis["target_element"] = "date filter option"
            elif any(day in target.lower() for day in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']):
                analysis["target_element"] = "date filter option"
            else:
                analysis["target_element"] = target[:50]  # Truncate long values
        else:
            # Infer from action text
            if "filter" in action_lower:
                analysis["target_element"] = "filter control"
            elif "button" in action_lower:
                analysis["target_element"] = "button"
            elif "input" in action_lower or "field" in action_lower:
                analysis["target_element"] = "input field"
            elif "dropdown" in action_lower or "select" in action_lower:
                analysis["target_element"] = "dropdown"
            else:
                analysis["target_element"] = "ui element"
        
        # Analyze failure reason from exception and actual message
        combined = f"{exc_msg} {actual_msg}".lower()
        
        if "page.evaluate" in combined and ("undefined" in combined or "null" in combined):
            analysis["problem_type"] = "dom_resolution_failure"
            analysis["failure_reason"] = "target element could not be resolved in the page DOM during evaluation"
        elif "element not found" in combined or "unable to locate" in combined:
            analysis["problem_type"] = "element_not_found"
            analysis["failure_reason"] = "target element was not found on the page"
        elif "locator" in combined and ("mismatch" in combined or "resolution" in combined):
            analysis["problem_type"] = "locator_resolution_failure"
            analysis["failure_reason"] = "element locator could not be resolved to a valid DOM target"
        elif "target closed" in combined:
            analysis["problem_type"] = "target_closed"
            analysis["failure_reason"] = "target element or page context was closed before interaction"
        elif "timeout" in combined or "timed out" in combined:
            analysis["problem_type"] = "timeout"
            analysis["failure_reason"] = "operation timed out waiting for element or page state"
        elif "not visible" in combined or "is hidden" in combined:
            analysis["problem_type"] = "visibility_failure"
            analysis["failure_reason"] = "target element exists but is not visible or interactable"
        elif "not interactable" in combined or "is disabled" in combined or "is readonly" in combined:
            analysis["problem_type"] = "interactability_failure"
            analysis["failure_reason"] = "target element exists but cannot be interacted with in current state"
        elif "assertionerror" in combined or ("expected" in combined and "actual" in combined):
            analysis["problem_type"] = "assertion_failure"
            analysis["failure_reason"] = "expected result did not match actual result"
        elif "connection refused" in combined or "network" in combined:
            analysis["problem_type"] = "network_failure"
            analysis["failure_reason"] = "network or connection issue prevented operation"
        else:
            analysis["problem_type"] = "general_failure"
            analysis["failure_reason"] = "operation failed during execution"
        
        # Determine business impact
        if analysis["problem_type"] in ["dom_resolution_failure", "element_not_found", "locator_resolution_failure"]:
            analysis["business_impact"] = "user cannot complete the intended action on the target element"
        elif analysis["problem_type"] == "timeout":
            analysis["business_impact"] = "operation did not complete within expected time frame"
        elif analysis["problem_type"] == "assertion_failure":
            analysis["business_impact"] = "system behavior does not match expected business rules"
        elif analysis["problem_type"] == "network_failure":
            analysis["business_impact"] = "system is unavailable or unreachable"
        else:
            analysis["business_impact"] = "operation could not be completed as intended"
        
        return analysis

    def _generate_problem_oriented_title(
        self, 
        analysis: Dict[str, str], 
        classification: str,
        repeated_pattern: bool = False
    ) -> str:
        """
        Generates a problem-oriented title based on execution evidence analysis.
        """
        target = analysis["target_element"]
        action = analysis["action_attempted"]
        reason = analysis["failure_reason"]
        
        # Make it business-readable
        if "dom_resolution_failure" in analysis["problem_type"]:
            if repeated_pattern:
                return f"Repeated {target} interaction failure - element not resolved in page DOM"
            else:
                return f"{target} could not be interacted with - element not resolved in page DOM"
        
        elif "element_not_found" in analysis["problem_type"]:
            if repeated_pattern:
                return f"Repeated {target} interaction failure - element not found on page"
            else:
                return f"{target} not found on page - {action} could not be completed"
        
        elif "locator_resolution_failure" in analysis["problem_type"]:
            if repeated_pattern:
                return f"Repeated {target} interaction failure - locator could not resolve to target"
            else:
                return f"{target} locator could not resolve to valid DOM target"
        
        elif "visibility_failure" in analysis["problem_type"]:
            return f"{target} exists but is not visible - {action} blocked"
        
        elif "interactability_failure" in analysis["problem_type"]:
            return f"{target} is not interactable in current state - {action} blocked"
        
        elif "timeout" in analysis["problem_type"]:
            return f"{target} {action} timed out - element or page did not respond in time"
        
        elif "assertion_failure" in analysis["problem_type"]:
            return f"{target} validation failed - actual result did not match expected"
        
        elif "network_failure" in analysis["problem_type"]:
            return f"{target} {action} failed - network or connection issue"
        
        else:
            if repeated_pattern:
                return f"Repeated {target} interaction failure - {reason}"
            else:
                return f"{target} {action} failed - {reason}"

    def _generate_detailed_summary(
        self,
        analysis: Dict[str, str],
        action_text: str,
        expected: str,
        actual: str,
        tc_count: int = 1
    ) -> str:
        """
        Generates a detailed problem-oriented summary based on execution evidence.
        """
        target = analysis["target_element"]
        action = analysis["action_attempted"]
        reason = analysis["failure_reason"]
        impact = analysis["business_impact"]
        
        summary_parts = []
        
        # Describe what was attempted
        if tc_count > 1:
            summary_parts.append(f"In {tc_count} test case(s), the system attempted to perform a {action} on {target}.")
        else:
            summary_parts.append(f"The system attempted to perform a {action} on {target}.")
        
        # Describe what failed
        summary_parts.append(f"The operation failed because {reason}.")
        
        # Describe the impact (fix the "user from user cannot" issue)
        if impact.startswith("user cannot"):
            summary_parts.append(f"This {impact}.")
        else:
            summary_parts.append(f"This prevents the user from {impact}.")
        
        # Add specific action context if available
        if action_text and action_text != "unknown":
            summary_parts.append(f"The failing step was: '{action_text}'.")
        
        # Add expected vs actual if available
        if expected and actual and expected != actual:
            summary_parts.append(f"Expected: {expected}, but got: {actual}.")
        
        return " ".join(summary_parts)

    def run(self, story_path: Path) -> Path:
        """
        Runs the Defect Agent on the active story post-execution.
        """
        log.info("[DefectAgent] Starting post-execution defect analysis...")

        # Ensure output defects folder exists
        self.defect_dir.mkdir(parents=True, exist_ok=True)

        # 1. Load execution artifacts
        report_path = self.execution_dir / "execution_report.json"
        logs_path = self.execution_dir / "execution_logs.json"

        testcases: List[Dict[str, Any]] = []
        execution_logs: List[Dict[str, Any]] = []

        if report_path.exists():
            try:
                report_data = json.loads(
                    report_path.read_text(encoding="utf-8")
                )
                testcases = report_data.get("test_cases", [])
            except Exception as e:
                log.warning(f"Could not load execution report: {e}")

        if logs_path.exists():
            try:
                logs_data = json.loads(
                    logs_path.read_text(encoding="utf-8")
                )
                execution_logs = logs_data.get("logs", [])
            except Exception as e:
                log.warning(f"Could not load execution logs: {e}")

        # 2. Filter for failed executed test cases
        failed_cases = [tc for tc in testcases if tc.get("status") == "FAIL"]
        log.info(
            f"[DefectAgent] Found {len(failed_cases)} failed test case(s) "
            f"out of {len(testcases)} total."
        )

        # 3. Analyze failures and construct defect artifacts
        defect_candidates = self._analyze_failed_cases(
            failed_cases, execution_logs
        )
        root_cause_analysis = self._perform_root_cause_analysis(
            failed_cases, defect_candidates
        )
        defect_metrics = self._generate_defect_metrics(
            failed_cases, defect_candidates, root_cause_analysis
        )

        # 4. Write output JSON files
        candidates_file = self.defect_dir / "defect_candidates.json"
        rca_file = self.defect_dir / "root_cause_analysis.json"
        metrics_file = self.defect_dir / "defect_metrics.json"

        try:
            candidates_file.write_text(
                json.dumps(defect_candidates, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
            log.info(
                f"[DefectAgent] Candidates written to "
                f"{candidates_file}"
            )
        except Exception as e:
            log.error(f"Failed to write defect_candidates.json: {e}")

        try:
            rca_file.write_text(
                json.dumps(root_cause_analysis, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
            log.info(
                f"[DefectAgent] RCA written to "
                f"{rca_file}"
            )
        except Exception as e:
            log.error(f"Failed to write root_cause_analysis.json: {e}")

        try:
            metrics_file.write_text(
                json.dumps(defect_metrics, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
            log.info(
                f"[DefectAgent] Metrics written to "
                f"{metrics_file}"
            )
        except Exception as e:
            log.error(f"Failed to write defect_metrics.json: {e}")

        return candidates_file

    def _is_non_executable_step(self, action_text: str) -> bool:
        """
        Determines if a testcase step is non-executable/abstract (business-rule, setup,
        or not directly executable as a concrete UI interaction).
        """
        action_lower = (action_text or "").lower()
        
        # Patterns that indicate abstract/non-executable steps
        non_executable_patterns = [
            # Business rule/setup patterns
            "ensure that",
            "verify that",
            "confirm that",
            "check that",
            "validate that",
            "make sure",
            "should be",
            "must be",
            "is expected to",
            "is supposed to",
            
            # Setup/configuration patterns
            "setup",
            "configure",
            "initialize",
            "prepare",
            "establish",
            
            # Abstract business logic without concrete action
            "user should",
            "system should",
            "application should",
            "process should",
            
            # State/condition descriptions
            "is available",
            "is visible",
            "is enabled",
            "is disabled",
            "is present",
            "is displayed",
            "exists",
            
            # Generic verification without action
            "verify the",
            "check the",
            "confirm the",
            "validate the",
            
            # Business flow descriptions
            "complete the",
            "finish the",
            "proceed with",
            "continue to",
        ]
        
        # Check if action starts with non-executable patterns
        for pattern in non_executable_patterns:
            if action_lower.startswith(pattern):
                return True
        
        # Check if action is purely descriptive without concrete UI action
        concrete_actions = ["click", "type", "input", "select", "hover", "scroll", "drag", "drop", "fill", "enter", "press", "submit", "navigate", "goto", "wait"]
        has_concrete_action = any(action in action_lower for action in concrete_actions)
        
        # If no concrete action verb found, likely abstract
        if not has_concrete_action and len(action_lower) > 5:
            return True
        
        return False

    def _classify_defect(
        self, exception_msg: str, actual_msg: str, action_text: str = ""
    ) -> Dict[str, str]:
        """
        Heuristics to classify defect type and detailed root cause.
        """
        combined = f"{exception_msg} {actual_msg}".lower()

        # 1. Product Failure Check (assertion, validation, incorrect outcome)
        if "assertionerror" in combined or (
            "expected" in combined and "actual" in combined
        ) or "validation mismatch" in combined or "wrong ui/app outcome" in combined:
            return {
                "type": "Product",
                "classification": "Product - Assertion Mismatch"
            }

        # 2. Automation / Non-Executable Testcase Step Check
        # This check must come BEFORE UI element resolution check
        if self._is_non_executable_step(action_text):
            return {
                "type": "Automation",
                "classification": "Automation / Non-Executable Testcase Step"
            }

        # 3. Timeout Check — must come BEFORE the generic UI-automation pattern check below.
        # Genuine Playwright TimeoutError text ("Timeout 30000ms exceeded ... waiting for
        # locator(...)") also matches several of the broad ui_automation_patterns entries
        # ("waiting for", "timeout exceeded", "locator"), so checking UI-automation first
        # was misclassifying real timeouts as "Automation / UI Element Resolution Failure"
        # instead of "Timeout - Wait Timeout Exceeded" — producing internally inconsistent
        # defect records (a timeout-worded title paired with an Automation classification)
        # and undercounting real timeouts in RCA metrics.
        if "timeout" in combined or "timed out" in combined:
            return {
                "type": "Timeout",
                "classification": "Timeout - Wait Timeout Exceeded"
            }

        # 4. Automation / UI Element Resolution Failure Check
        # This check must come BEFORE API check to prevent misclassification
        ui_automation_patterns = [
            "page.evaluate",
            "eval at evaluate",
            "cannot read properties of undefined",
            "cannot read properties of null",
            "undefined (reading",
            "null (reading",
            "element not found",
            "unable to locate",
            "locator mismatch",
            "rediscovery failure",
            "locator.click",
            "locator.",
            "selector",
            "playwright interaction failure",
            "unresolved ui target",
            "target closed",
            "element is not visible",
            "waiting for locator",
            "waiting for selector",
            "element resolution failure",
            "strict mode violation",
            "element handle",
            "frame.evaluate",
            "jshandle",
            "dom error",
            "tagname",
            "node is detached",
            "document is not attached",
            "execution context was destroyed",
            "target closed",
            "page crashed",
            "console.error",
            "script timeout",
            "evaluation failed",
            "timeout exceeded",
            "waiting for",
            "element not attached to the dom",
            "element not visible",
            "element not interactable",
            "element is hidden",
            "element is disabled",
            "element is readonly",
            "element is covered",
            "element outside viewport",
            "playwright",
            "browser context",
            "page.",
            "frame.",
            "locator",
        ]
        
        # Check if any UI/automation pattern is present
        if any(pattern in combined for pattern in ui_automation_patterns):
            return {
                "type": "Automation",
                "classification": "Automation / UI Element Resolution Failure"
            }

        # 5. API Failure Check
        # Classify as API only when there is explicit HTTP/API evidence
        # This check must come AFTER Automation/UI check to prevent misclassification
        api_evidence_patterns = [
            "api request failed",
            "api response error",
            "failed api assertion",
            "api assertion failed",
            "http status code",
            "status code 4",
            "status code 5",
            "response status: 4",
            "response status: 5",
            "request failed with status",
            "server returned 4",
            "server returned 5",
            "network error: 4",
            "network error: 5",
            "axios error",
            "fetch error",
            "xhr error",
            "ajax error",
            "api call failed",
            "endpoint error",
            "rest api error",
            "graphql error",
        ]
        
        # Check for explicit API evidence with context (not just status codes alone)
        has_explicit_api_evidence = any(pattern in combined for pattern in api_evidence_patterns)
        
        # Only match status codes if they appear with API-related context
        has_status_code_with_context = (
            ("status" in combined or "response" in combined or "error" in combined) and
            re.search(r'\b(4\d{2}|5\d{2})\b', combined) is not None
        )
        
        if has_explicit_api_evidence or has_status_code_with_context:
            return {
                "type": "API",
                "classification": "API - Server Response Error (5xx/4xx)"
            }

        # 5. Environment Failure Check
        if (
            "connection refused" in combined
            or "dns" in combined
            or "net::err" in combined
            or "failed to load" in combined
            or "browser launch failure" in combined
            or "failed to launch browser" in combined
            or "chrome failed to start" in combined
            or "executable not found" in combined
            or "config error" in combined
            or "auth/environment" in combined
            or "external network outage" in combined
            or "dependency unavailability" in combined
            or "service unavailable" in combined
            or "network unreachable" in combined
            or "host unreachable" in combined
        ):
            return {
                "type": "Environment",
                "classification": "Environment - Network / DNS Host Unreachable"
            }

        # 6. Fallback
        return {
            "type": "Unknown",
            "classification": "Unknown - Unclassified Exception"
        }

    def _normalize_component(self, action_text: str, tc_title: str = "") -> str:
        """
        Normalizes component names from action text to avoid literal UI values.
        Returns a normalized component family name for grouping.
        """
        action_lower = (action_text or "").lower()
        title_lower = (tc_title or "").lower()
        
        # Common UI component patterns to normalize
        component_patterns = {
            # Date/time filters
            r'\d{1,2}\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}': 'date filter option',
            r'(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday),\s*\d{1,2}': 'date filter option',
            r'\d{1,2}/\d{1,2}/\d{4}': 'date filter option',
            r'\d{4}-\d{2}-\d{2}': 'date filter option',
            
            # Generic filter options
            r'filter\s+option': 'filter option',
            r'filter\s+by': 'filter control',
            
            # Navigation elements
            r'nav\s+\w+': 'navigation item',
            r'menu\s+\w+': 'menu item',
            r'tab\s+\w+': 'tab',
            
            # Form elements
            r'button\s+\w+': 'button',
            r'input\s+\w+': 'input field',
            r'textbox\s+\w+': 'textbox',
            r'checkbox\s+\w+': 'checkbox',
            r'select\s+\w+': 'dropdown',
            r'radio\s+\w+': 'radio button',
            
            # Content elements
            r'link\s+\w+': 'link',
            r'image\s+\w+': 'image',
            r'text\s+\w+': 'text element',
        }
        
        # Try to match against patterns
        for pattern, normalized in component_patterns.items():
            if re.search(pattern, action_lower, re.IGNORECASE):
                return normalized
        
        # Extract from action text if it contains quotes
        quoted_match = re.search(r"'([^']+)'", action_text)
        if quoted_match:
            raw_value = quoted_match.group(1).lower()
            
            # Check if it's a date pattern
            if any(month in raw_value for month in ['january', 'february', 'march', 'april', 'may', 'june', 
                                                       'july', 'august', 'september', 'october', 'november', 'december']):
                return 'date filter option'
            
            # Check if it's a day pattern
            if any(day in raw_value for day in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']):
                return 'date filter option'
            
            # Check if it's a numeric value that might be a filter option
            if raw_value.isdigit() and len(raw_value) <= 4:
                return 'filter option'
        
        # Infer from action type
        if 'click' in action_lower:
            return 'button'
        elif 'input' in action_lower or 'type' in action_lower:
            return 'input field'
        elif 'select' in action_lower:
            return 'dropdown'
        elif 'hover' in action_lower:
            return 'hover target'
        
        # Infer from testcase title
        if 'filter' in title_lower:
            return 'filter component'
        elif 'date' in title_lower:
            return 'date component'
        elif 'button' in title_lower:
            return 'button'
        elif 'form' in title_lower:
            return 'form element'
        
        # Fallback to generic component
        return 'ui component'

    def _extract_target_text(self, action_text: str, component: str) -> str:
        """
        Extracts the actual target text from action text for precise clustering.
        Returns the specific quoted value if available, otherwise infers from action patterns.
        This ensures different targets like "Things To Do", "Most Popular", "Load More" are not merged.
        """
        action_lower = action_text.lower()
        
        # Extract quoted value - this is the actual target text
        quoted_match = re.search(r"'([^']+)'", action_text)
        if quoted_match:
            target_value = quoted_match.group(1)
            
            # Normalize date patterns to a common value to group date filters together
            target_lower = target_value.lower()
            if any(month in target_lower for month in ['january', 'february', 'march', 'april', 'may', 'june', 
                                                       'july', 'august', 'september', 'october', 'november', 'december']):
                return 'date_filter'
            if any(day in target_lower for day in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']):
                return 'date_filter'
            if re.search(r'\d{1,2}/\d{1,2}/\d{4}', target_lower):
                return 'date_filter'
            if re.search(r'\d{4}-\d{2}-\d{2}', target_lower):
                return 'date_filter'
            
            # Return the actual target text for non-date values
            # This ensures "Things To Do", "Most Popular", etc. are kept separate
            return target_value.lower()[:50]  # Truncate if too long
        
        # Infer from action text patterns when no quoted value
        if 'load more' in action_lower or 'show more' in action_lower:
            return 'load_more'
        elif 'date' in action_lower:
            return 'date_filter'
        elif 'filter' in action_lower:
            return 'filter_control'
        elif 'category' in action_lower:
            return 'category_filter'
        elif 'search' in action_lower:
            return 'search_action'
        elif 'sort' in action_lower:
            return 'sort_action'
        
        # Fallback to component type
        if 'button' in component.lower():
            return 'button'
        elif 'input' in component.lower() or 'field' in component.lower():
            return 'input_field'
        elif 'dropdown' in component.lower() or 'select' in component.lower():
            return 'dropdown'
        
        return 'ui_element'

    def _normalize_failure_signature(self, exc_msg: str, actual_msg: str, 
                                     component: str, classification: str, action_text: str) -> str:
        """
        Creates a normalized failure signature for grouping similar failures.
        Uses actual target text for precise clustering to prevent merging different targets.
        """
        combined = f"{exc_msg} {actual_msg}".lower()
        
        # Extract the core error pattern
        error_patterns = [
            ('page.evaluate', 'page.evaluate error'),
            ('cannot read properties of undefined', 'undefined property access'),
            ('cannot read properties of null', 'null property access'),
            ('element not found', 'element not found'),
            ('unable to locate', 'element not found'),
            ('locator mismatch', 'locator resolution failure'),
            ('target closed', 'target closed'),
            ('timeout', 'timeout error'),
            ('assertionerror', 'assertion failure'),
            ('connection refused', 'connection error'),
        ]
        
        core_error = 'unknown error'
        for pattern, normalized in error_patterns:
            if pattern in combined:
                core_error = normalized
                break
        
        # Extract actual target text for precise clustering
        target_text = self._extract_target_text(action_text, component)
        
        # Create signature: classification + core_error + target_text
        # This ensures different targets like "Things To Do", "Most Popular", "Load More" are not merged
        signature = f"{classification}|{core_error}|{target_text}"
        return signature

    def _analyze_failed_cases(
        self,
        failed_cases: List[Dict[str, Any]],
        execution_logs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Groups failed test cases into defect candidates by error signature.
        """
        if not failed_cases:
            return []

        # Find the failing step in each testcase
        analyzed_cases = []
        for tc in failed_cases:
            tc_id = tc.get("id", "TC")
            steps = tc.get("steps", [])
            failing_step = {}
            for s in steps:
                if s.get("status") == "FAIL":
                    failing_step = s
                    break

            # Determine component name from action using normalization
            action_text = failing_step.get("action") or ""
            tc_title = tc.get("title") or ""
            component = self._normalize_component(action_text, tc_title)

            exc_msg = tc.get("exception", "") or ""
            actual_msg = tc.get("actual_result", "") or ""
            classification = self._classify_defect(exc_msg, actual_msg, action_text)
            
            # Perform deep execution evidence analysis
            execution_analysis = self._analyze_execution_evidence(
                failing_step, exc_msg, actual_msg, tc_title
            )

            # Parse logs specific to this test case
            tc_logs = []
            for entry in execution_logs:
                msg = entry.get("message", "")
                if f"[{tc_id}]" in msg:
                    tc_logs.append(
                        f"{entry.get('timestamp')} "
                        f"[{entry.get('level')}] {msg}"
                    )

            # Collect evidence references
            screenshots = []
            screenshot_rel = tc.get("screenshot")
            if screenshot_rel:
                screenshots.append(screenshot_rel)

            # Add step screenshots if they exist and failed
            if failing_step:
                if failing_step.get("before_screenshot"):
                    screenshots.append(failing_step.get("before_screenshot"))
                if failing_step.get("after_screenshot"):
                    screenshots.append(failing_step.get("after_screenshot"))

            # Trace references
            traces = []
            trace_path = (
                self.settings.artifact_dir / "traces" / f"{tc_id}_trace.zip"
            )
            if trace_path.exists():
                traces.append(f"traces/{tc_id}_trace.zip")

            analyzed_cases.append({
                "tc_id": tc_id,
                "title": tc.get("title") or "",
                # `.get(key, default)` only substitutes when the key is missing, not when
                # it's explicitly null — an execution report with "severity": null would
                # otherwise carry None through to downstream `.lower()` calls and crash.
                "severity": tc.get("severity") or "Major",
                "priority": tc.get("priority") or "Medium",
                "failing_step_num": failing_step.get("step") or 1,
                "failing_step_action": (
                    action_text or "No failing step recorded."
                ),
                "expected": failing_step.get("expected") or "",
                "actual": actual_msg,
                "defect_type": classification["type"],
                "classification": classification["classification"],
                "component": component,
                "exc_msg": exc_msg,
                "execution_analysis": execution_analysis,
                "screenshots": screenshots,
                "traces": traces,
                "logs": tc_logs
            })

        # Grouping algorithm: group by normalized failure signature
        candidates: List[Dict[str, Any]] = []
        candidate_map: Dict[str, Dict[str, Any]] = {}

        for ac in analyzed_cases:
            # Create normalized signature for grouping
            failure_signature = self._normalize_failure_signature(
                ac["exc_msg"], 
                ac["actual"], 
                ac["component"], 
                ac["classification"],
                ac["failing_step_action"]
            )
            group_key = failure_signature
            if group_key not in candidate_map:
                candidate_map[group_key] = {
                    "defect_id": f"DF-{len(candidate_map) + 1:03d}",
                    "linked_testcase_ids": [],
                    "title": (
                        f"Verify failure on component '{ac['component']}' "
                        f"due to {ac['defect_type']} error"
                    ),
                    "failure_step": (
                        f"Step {ac['failing_step_num']}: "
                        f"{ac['failing_step_action']}"
                    ),
                    "expected_vs_actual": (
                        f"Expected: {ac['expected'] or 'Successful execution'}"
                        f" -> Actual: {ac['actual']}"
                    ),
                    "defect_type": ac["defect_type"],
                    "severity_suggestion": ac["severity"],
                    "priority_suggestion": ac["priority"],
                    "evidence_references": {
                        "screenshots": [],
                        "traces": [],
                        "logs": []
                    }
                }

            # Merge into candidate
            c = candidate_map[group_key]
            c["linked_testcase_ids"].append(ac["tc_id"])
            c["evidence_references"]["screenshots"].extend(ac["screenshots"])
            c["evidence_references"]["traces"].extend(ac["traces"])
            c["evidence_references"]["logs"].extend(ac["logs"])
            
            # Store execution analysis for later use in title generation
            if "execution_analyses" not in c:
                c["execution_analyses"] = []
            c["execution_analyses"].append(ac["execution_analysis"])

            # Suggest highest severity / priority in group
            def get_weight(val: str) -> int:
                val_lower = (val or "").lower()
                if (
                    "critical" in val_lower
                    or "blocker" in val_lower
                    or "high" in val_lower
                ):
                    return 3
                if "major" in val_lower or "medium" in val_lower:
                    return 2
                if "minor" in val_lower or "low" in val_lower or "trivial" in val_lower:
                    return 1
                # Unrecognized severity/priority vocabulary (e.g. "Moderate", "P1") would
                # otherwise silently fall through to the lowest weight and could suppress a
                # legitimately higher value during "pick the highest in group" selection.
                log.warning(f"[DefectAgent] Unrecognized severity/priority value '{val}' — treating as lowest weight.")
                return 1

            if (
                get_weight(ac["severity"])
                > get_weight(c["severity_suggestion"])
            ):
                c["severity_suggestion"] = ac["severity"]
            if (
                get_weight(ac["priority"])
                > get_weight(c["priority_suggestion"])
            ):
                c["priority_suggestion"] = ac["priority"]

        # Finalize title / candidates list
        for key, c in candidate_map.items():
            # Clean up duplicate lists
            c["evidence_references"]["screenshots"] = sorted(
                list(set(c["evidence_references"]["screenshots"]))
            )
            c["evidence_references"]["traces"] = sorted(
                list(set(c["evidence_references"]["traces"]))
            )
            # Truncate logs if they are too long (keep first 20 entries)
            c["evidence_references"]["logs"] = (
                c["evidence_references"]["logs"][:20]
            )

            # Generate problem-oriented title using execution evidence
            execution_analyses = c.get("execution_analyses", [])
            
            if execution_analyses:
                # Since clustering now uses actual target text, all failures in a cluster
                # have the same target. Use first analysis as representative.
                analysis = execution_analyses[0]
                classification = c.get("defect_type", "Unknown")
                repeated_pattern = len(execution_analyses) > 1
                
                c["title"] = self._generate_problem_oriented_title(
                    analysis, classification, repeated_pattern
                )
                
                # Generate detailed summary
                tc_count = len(c["linked_testcase_ids"])
                action_text = c.get("failure_step", "")
                expected = c.get("expected_vs_actual", "")
                
                # Extract expected and actual from the expected_vs_actual field
                expected_val = ""
                actual_val = ""
                if " -> " in expected:
                    parts = expected.split(" -> ")
                    if len(parts) == 2:
                        expected_val = parts[0].replace("Expected: ", "")
                        actual_val = parts[1].replace("Actual: ", "")
                
                c["summary"] = self._generate_detailed_summary(
                    analysis, action_text, expected_val, actual_val, tc_count
                )
            else:
                # Fallback to original title format if no analysis available
                signature_parts = key.split('|')
                if len(signature_parts) >= 3:
                    classification = signature_parts[0]
                    core_error = signature_parts[1]
                    component_family = signature_parts[2]
                    
                    # Match the actual classification text produced by _classify_defect
                    # ("Automation / UI Element Resolution Failure") — the previous check
                    # ("UI Interaction") never appeared in any real classification string.
                    if "Automation" in classification and "UI Element Resolution Failure" in classification:
                        c["title"] = (
                            f"{component_family.capitalize()} interaction fails due to "
                            f"{core_error}"
                        )
                    else:
                        c["title"] = (
                            f"{component_family.capitalize()} fails due to "
                            f"{core_error}"
                        )
                else:
                    tcs_str = ", ".join(c["linked_testcase_ids"])
                    c["title"] = (
                        f"Component failed with {c['defect_type']} ({tcs_str})"
                    )
                c["summary"] = c.get("expected_vs_actual", "No detailed summary available.")
            
            # Remove temporary execution_analyses field
            if "execution_analyses" in c:
                del c["execution_analyses"]
            
            candidates.append(c)

        return candidates

    def _perform_root_cause_analysis(
        self,
        failed_cases: List[Dict[str, Any]],
        candidates: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Extracts RCA mappings and calculates splits and hotspot counts.
        """
        if not failed_cases:
            return {
                "root_cause_classifications": {},
                "root_cause_breakdown_summary": (
                    "All tests executed successfully. "
                    "No root causes to analyze."
                ),
                "rc_metrics": {
                    "product_vs_automation_vs_environment_split": {},
                    "repeated_failure_hotspots": {},
                    "dominant_failure_causes": {}
                }
            }

        classifications = {}
        hotspots = {}
        causes = {}
        prod_count = 0
        auto_count = 0
        api_count = 0
        env_count = 0

        # Classify each testcase
        for tc in failed_cases:
            tc_id = tc.get("id", "TC")
            exc = tc.get("exception", "") or ""
            actual = tc.get("actual_result", "") or ""
            
            # Extract action text from failing step for better classification
            action_text = ""
            steps = tc.get("steps", [])
            for s in steps:
                if s.get("status") == "FAIL":
                    action_text = s.get("action") or ""
                    break

            res = self._classify_defect(exc, actual, action_text)

            classifications[tc_id] = {
                "root_cause_classification": res["classification"],
                "reason": actual
            }

            # Gather metrics
            cause_name = res["classification"]
            causes[cause_name] = causes.get(cause_name, 0) + 1

            if res["type"] == "Product":
                prod_count += 1
            elif res["type"] in ["Automation", "Timeout"]:
                auto_count += 1
            elif res["type"] == "API":
                api_count += 1
            else:
                env_count += 1

        # Hotspots from candidates - use specific target text from clustering
        for c in candidates:
            title = c["title"]
            failure_step = c.get("failure_step", "")
            summary = c.get("summary", "")
            
            hotspot_name = "Unknown"
            
            # Extract quoted values from failure step (these are the actual UI elements)
            quoted_match = re.search(r"'([^']+)'", failure_step)
            if quoted_match:
                specific_value = quoted_match.group(1)
                # Don't use date values as hotspot names
                if not any(month in specific_value.lower() for month in ['january', 'february', 'march', 'april', 'may', 'june', 
                                                                           'july', 'august', 'september', 'october', 'november', 'december']):
                    if not any(day in specific_value.lower() for day in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']):
                        # Use the specific value if it's not a date
                        hotspot_name = specific_value[:30]  # Truncate if too long
            
            # If no specific value found, extract from title with more context
            if hotspot_name == "Unknown":
                # Extract meaningful words from title (avoid generic terms)
                words = title.split()
                meaningful_words = []
                for word in words:
                    word_lower = word.lower()
                    if word_lower not in ['component', 'element', 'interaction', 'fails', 'due', 'to', 'the', 'a', 'an', 'in', 'at', 'for', 'with', 'repeated']:
                        meaningful_words.append(word)
                        if len(meaningful_words) >= 3:
                            break
                
                if meaningful_words:
                    hotspot_name = " ".join(meaningful_words[:3]).lower()
                elif words:
                    hotspot_name = words[0].lower()
            
            # If still unknown, extract from summary
            if hotspot_name == "Unknown" or len(hotspot_name) < 3:
                # Look for specific UI elements in summary
                summary_lower = summary.lower()
                if "button" in summary_lower:
                    hotspot_name = "button"
                elif "filter" in summary_lower and "date" in summary_lower:
                    hotspot_name = "date filter"
                elif "filter" in summary_lower:
                    hotspot_name = "filter control"
                elif "input" in summary_lower or "field" in summary_lower:
                    hotspot_name = "input field"
                elif "dropdown" in summary_lower or "select" in summary_lower:
                    hotspot_name = "dropdown"
            
            # Final fallback to component family if still unknown
            if hotspot_name == "Unknown" or len(hotspot_name) < 3:
                action_words = failure_step.split()
                if action_words:
                    # Find the first meaningful word (not step number, common words)
                    for word in action_words:
                        word_lower = word.lower().strip(":")
                        if word_lower and word_lower not in ['step', 'click', 'on', 'the', 'a', 'an', 'in', 'at', 'for', 'with']:
                            hotspot_name = word_lower[:30]
                            break
            
            # Final fallback if still unknown
            if hotspot_name == "Unknown":
                words = title.split()
                if words:
                    component_family = words[0].lower()
                    if component_family in ['date', 'filter']:
                        hotspot_name = 'date filter'
                    elif component_family in ['button', 'input', 'dropdown', 'checkbox']:
                        hotspot_name = component_family
                    else:
                        hotspot_name = component_family
            
            count = len(c["linked_testcase_ids"])
            hotspots[hotspot_name] = hotspots.get(hotspot_name, 0) + count

        total = len(failed_cases)
        split = {
            "product": round(prod_count / total, 2) if total > 0 else 0.0,
            "automation": round(auto_count / total, 2) if total > 0 else 0.0,
            "api": round(api_count / total, 2) if total > 0 else 0.0,
            "environment": round(env_count / total, 2) if total > 0 else 0.0
        }

        # Build detailed RCA summary with exact problem identification
        hotspot_list = [f"'{k}' ({v} failures)" for k, v in hotspots.items()]
        causes_list = [f"'{k}' ({v} instances)" for k, v in causes.items()]
        
        # Add detailed problem analysis to the summary
        problem_analysis = []
        if "Automation / UI Element Resolution Failure" in causes:
            problem_analysis.append("UI element resolution failures indicate that target elements could not be located or resolved in the page DOM during execution.")
        if "Automation / Non-Executable Testcase Step" in causes:
            problem_analysis.append("Non-executable step failures indicate abstract business logic steps that could not be executed as concrete UI interactions.")
        if "Product - Assertion Mismatch" in causes:
            problem_analysis.append("Assertion failures indicate system behavior does not match expected business rules or validation criteria.")
        # Key must match _classify_defect's actual classification string exactly
        # ("API - Server Response Error (5xx/4xx)") — the previous shorter string never
        # matched, so this sentence silently never appeared even when API failures dominated.
        if "API - Server Response Error (5xx/4xx)" in causes:
            problem_analysis.append("API failures indicate server-side errors or network issues preventing successful API communication.")
        if "Timeout - Wait Timeout Exceeded" in causes:
            problem_analysis.append("Timeout failures indicate operations did not complete within the expected wait period, suggesting slow-loading elements or unresponsive application state.")
        if "Environment - Network / DNS Host Unreachable" in causes:
            problem_analysis.append("Environment failures indicate infrastructure or connectivity issues preventing test execution.")
        
        analysis_text = " ".join(problem_analysis) if problem_analysis else ""
        
        breakdown_summary = (
            f"Analyzed {total} failure(s). "
            f"The primary failure hotspot(s) were: "
            f"{', '.join(hotspot_list) if hotspot_list else 'None'}. "
            f"The dominant cause(s) identified were: "
            f"{', '.join(causes_list) if causes_list else 'None'}. "
            f"{analysis_text}"
        )

        return {
            "root_cause_classifications": classifications,
            "root_cause_breakdown_summary": breakdown_summary,
            "rc_metrics": {
                "product_vs_automation_vs_environment_split": split,
                "repeated_failure_hotspots": hotspots,
                "dominant_failure_causes": causes
            }
        }

    def _generate_defect_metrics(
        self,
        failed_cases: List[Dict[str, Any]],
        candidates: List[Dict[str, Any]],
        rca: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Computes run-level defect metrics distributions and densities.
        """
        sev_dist = {}
        type_dist = {}
        rc_dist = rca["rc_metrics"]["dominant_failure_causes"]

        for tc in failed_cases:
            sev = tc.get("severity") or "Major"
            sev_dist[sev] = sev_dist.get(sev, 0) + 1

        for c in candidates:
            dtype = c["defect_type"]
            type_dist[dtype] = (
                type_dist.get(dtype, 0) + len(c["linked_testcase_ids"])
            )

        density_summary = (
            f"Generated {len(candidates)} unique defect candidate(s) "
            f"from {len(failed_cases)} failures."
        )

        return {
            "total_failed_testcases": len(failed_cases),
            "total_defect_candidates": len(candidates),
            "severity_distribution": sev_dist,
            "defect_type_distribution": type_dist,
            "root_cause_distribution": rc_dist,
            "defect_density_summary": density_summary
        }
