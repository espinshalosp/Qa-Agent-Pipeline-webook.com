"""
agents/jira_update_agent.py
===========================
Jira Update Agent — Final reporting agent in the QA pipeline.

Responsibilities
----------------
1. Read the final run artifacts from the current run directory.
2. Upload actual run artifacts as attachments to the Jira story.
3. Build a concise QA summary from available outputs (generation, execution, coverage, defects)
   detailing only successful uploads.
4. Post the QA summary back to the same Jira story as a Jira comment (using ADF format).
5. Run safely and adapt to the presence/absence of run artifacts.
"""
from __future__ import annotations

import base64
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from config.settings import Settings
from utils.logger import get_logger
from utils.retry import retry_on_failure

log = get_logger(__name__)


# ===========================================================================
# ADF Parser Helper
# ===========================================================================
def parse_inline_text(text: str) -> List[Dict[str, Any]]:
    """
    Parse text with basic markdown-like formatting (**bold**, `code`, [text](url))
    into a list of ADF text nodes.
    """
    if not text:
        return []
    
    # Pattern to match bold (**bold**), code (`code`), or links ([text](url))
    pattern = re.compile(r'(\*\*.*?\*\*|`[^`]+`|\[[^\]]+\]\([^)]+\))')
    parts = pattern.split(text)
    nodes = []
    
    for part in parts:
        if not part:
            continue
        if part.startswith('**') and part.endswith('**'):
            nodes.append({
                "type": "text",
                "text": part[2:-2],
                "marks": [{"type": "strong"}]
            })
        elif part.startswith('`') and part.endswith('`'):
            nodes.append({
                "type": "text",
                "text": part[1:-1],
                "marks": [{"type": "code"}]
            })
        elif part.startswith('[') and ']' in part and '(' in part and part.endswith(')'):
            match = re.match(r'\[([^\]]+)\]\(([^)]+)\)', part)
            if match:
                nodes.append({
                    "type": "text",
                    "text": match.group(1),
                    "marks": [{
                        "type": "link",
                        "attrs": {
                            "href": match.group(2)
                        }
                    }]
                })
            else:
                nodes.append({"type": "text", "text": part})
        else:
            nodes.append({"type": "text", "text": part})
            
    return nodes


class ADFBuilder:
    """
    Helper to construct Atlassian Document Format (ADF) payloads.
    """
    def __init__(self) -> None:
        self.content: List[Dict[str, Any]] = []

    def add_heading(self, text: str, level: int = 2) -> ADFBuilder:
        self.content.append({
            "type": "heading",
            "attrs": {"level": level},
            "content": parse_inline_text(text)
        })
        return self

    def add_paragraph(self, text: str) -> ADFBuilder:
        lines = text.split("\n")
        for line in lines:
            if not line.strip():
                continue
            self.content.append({
                "type": "paragraph",
                "content": parse_inline_text(line)
            })
        return self

    def add_bullet_list(self, items: List[str]) -> ADFBuilder:
        list_content = []
        for item in items:
            list_content.append({
                "type": "listItem",
                "content": [
                    {
                        "type": "paragraph",
                        "content": parse_inline_text(item)
                    }
                ]
            })
        self.content.append({
            "type": "bulletList",
            "content": list_content
        })
        return self

    def add_code_block(self, code: str, language: str = "text") -> ADFBuilder:
        self.content.append({
            "type": "codeBlock",
            "attrs": {"language": language},
            "content": [
                {
                    "type": "text",
                    "text": code
                }
            ]
        })
        return self

    def build(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "type": "doc",
            "content": self.content
        }


# ===========================================================================
# Hotspot cleanup helper
# ===========================================================================
def _clean_hotspot(text: str) -> str:
    """
    Cleans truncated and internal-looking hotspot texts to be business-readable.
    """
    text = text.strip()
    truncations = {
        "workflo": "workflow",
        "interactio": "interaction",
        "validatio": "validation",
        "configuratio": "configuration",
        "transactio": "transaction",
        "authenticatio": "authentication",
        "registratio": "registration",
    }
    for trunc, replacement in truncations.items():
        if text.endswith(trunc):
            text = text[:-len(trunc)] + replacement
            break
            
    if text:
        text = text[0].upper() + text[1:]
    return text


# ===========================================================================
# JiraUpdateAgent
# ===========================================================================
class JiraUpdateAgent:
    """
    Gathers pipeline run artifacts, creates a QA summary, and posts a Jira comment.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def run(self, story_path: Path, phase: str = "all") -> Optional[str]:
        """
        Gathers results from the story's run directory, uploads attachments, and posts comments.

        Parameters
        -------
        story_path: Path
            Path to the story.json file inside the run directory.
        phase: str
            The pipeline phase for which to update Jira ("generation", "execution", "defect", or "all").

        Returns
        -------
        str | None
            The response ID/key of the posted comment or message.
        """
        log.info("=" * 60)
        log.info(f"[JiraUpdateAgent] Updating details and attachments on Jira for {phase} phase...")
        log.info(f"[JiraUpdateAgent] Target run dir: {story_path.parent}")
        log.info("=" * 60)

        # 1. Read story.json to find issue key
        story_data = self._read_json(story_path)
        if not story_data:
            log.error(f"[JiraUpdateAgent] Could not load story.json at {story_path}. Aborting.")
            return None

        story_obj = story_data.get("story", {})
        issue_key = story_obj.get("key")
        if not issue_key:
            log.error("[JiraUpdateAgent] Story does not have a valid issue key. Aborting.")
            return None

        # 2. Collect other artifacts if they exist
        run_dir = story_path.parent
        testcases_path = run_dir / "testcases.json"
        coverage_path = run_dir / "coverage_summary.json"
        exec_summary_path = run_dir / "execution" / "execution_summary.json"
        exec_report_path = run_dir / "execution" / "execution_report.json"
        defect_metrics_path = run_dir / "defects" / "defect_metrics.json"
        rca_path = run_dir / "defects" / "root_cause_analysis.json"
        defect_candidates_path = run_dir / "defects" / "defect_candidates.json"

        # Determine which files exist and need to be uploaded based on phase
        upload_files: List[Path] = []
        gen_paths = [story_path, testcases_path, coverage_path]
        exec_paths = [exec_summary_path, exec_report_path]
        defect_paths = [defect_candidates_path, rca_path, defect_metrics_path]

        if phase == "generation":
            target_paths = gen_paths
        elif phase == "execution":
            target_paths = exec_paths
        elif phase == "defect":
            target_paths = defect_paths
        else:
            target_paths = gen_paths + exec_paths + defect_paths

        for path in target_paths:
            if path.exists():
                upload_files.append(path)

        # 3. Zip Allure Report and collect trace files
        allure_zip_path = run_dir / "reports" / "allure-report.zip"
        allure_zip_created = False

        try:
            if phase in ("execution", "all"):
                # Zip Allure Report if it exists and contains content
                allure_report_dir = run_dir / "reports" / "allure-report"
                if allure_report_dir.is_dir() and any(allure_report_dir.iterdir()):
                    import shutil
                    try:
                        allure_zip_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.make_archive(
                            str(run_dir / "reports" / "allure-report"),
                            "zip",
                            root_dir=str(allure_report_dir)
                        )
                        if allure_zip_path.exists():
                            upload_files.append(allure_zip_path)
                            allure_zip_created = True
                            log.info(f"[JiraUpdateAgent] Packaged Allure report into: {allure_zip_path.name}")
                    except Exception as exc:
                        log.error(f"[JiraUpdateAgent] Failed to zip Allure report: {exc}")

                # Collect trace zip files
                traces_dir = run_dir / "traces"
                if traces_dir.is_dir():
                    trace_files = [p for p in traces_dir.iterdir() if p.is_file() and p.name.endswith(".zip")]
                    
                    # Check if testcase execution status is available in execution_report.json
                    exec_report_data = self._read_json(exec_report_path)
                    failed_tc_ids = set()
                    status_available = False
                    
                    if exec_report_data and isinstance(exec_report_data.get("test_cases"), list):
                        status_available = True
                        for tc in exec_report_data["test_cases"]:
                            tc_id = tc.get("id")
                            tc_status = str(tc.get("status", "")).upper()
                            if tc_id and tc_status in ("FAIL", "FAILED", "FAILURE"):
                                failed_tc_ids.add(tc_id)
                    
                    if status_available:
                        # Attach only the failed testcase trace zip files
                        for tf in trace_files:
                            if any(failed_id in tf.name for failed_id in failed_tc_ids):
                                upload_files.append(tf)
                                log.info(f"[JiraUpdateAgent] Added failed trace zip to upload: {tf.name}")
                    else:
                        # Attach all trace zip files if status is not available
                        for tf in trace_files:
                            upload_files.append(tf)
                            log.info(f"[JiraUpdateAgent] No execution status available. Added all trace zip to upload: {tf.name}")

            # 4. Perform Uploads
            successful_uploads: List[str] = []
            
            if self.settings.dry_run:
                log.info(f"[JiraUpdateAgent] [DRY RUN] Would upload {len(upload_files)} attachment(s): {[f.name for f in upload_files]}")
                successful_uploads = [f.name for f in upload_files]
            else:
                for file_path in upload_files:
                    try:
                        success = self._upload_attachment(issue_key, file_path)
                        if success:
                            successful_uploads.append(file_path.name)
                    except Exception as exc:
                        log.error(f"[JiraUpdateAgent] Failed to upload {file_path.name}: {exc}")

            # 5. Build ADF Comment Body
            testcases_data = self._read_json(testcases_path)
            coverage_data = self._read_json(coverage_path)
            exec_summary_data = self._read_json(exec_summary_path)
            defect_metrics_data = self._read_json(defect_metrics_path)
            rca_data = self._read_json(rca_path)

            adf_body = self._build_comment_payload(
                issue_key=issue_key,
                run_dir=run_dir,
                story_data=story_obj,
                testcases_data=testcases_data,
                coverage_data=coverage_data,
                exec_summary_data=exec_summary_data,
                defect_metrics_data=defect_metrics_data,
                rca_data=rca_data,
                uploaded_files=successful_uploads,
                phase=phase
            )

            # 6. Post comment
            comment_id = None
            if self.settings.dry_run:
                log.info(f"[JiraUpdateAgent] [DRY RUN] Would post comment to {issue_key} (phase: {phase}):")
                log.info(json.dumps(adf_body, indent=2))
                comment_id = "dry_run_success"
            else:
                try:
                    comment_id = self._post_comment(issue_key, adf_body)
                    log.info(f"[JiraUpdateAgent] ✓ Posted update comment to {issue_key} (Comment ID: {comment_id})")
                except Exception as exc:
                    log.error(f"[JiraUpdateAgent] Failed to post comment to Jira: {exc}")

        finally:
            # 7. Clean up allure zip ONLY after upload attempt finishes
            if allure_zip_created and allure_zip_path.exists():
                try:
                    allure_zip_path.unlink()
                    log.debug(f"[JiraUpdateAgent] Cleaned up temporary file: {allure_zip_path.name}")
                except Exception as exc:
                    log.warning(f"[JiraUpdateAgent] Could not delete temporary zip {allure_zip_path.name}: {exc}")

        return comment_id

    # -----------------------------------------------------------------------
    # Helper to build ADF payload
    # -----------------------------------------------------------------------
    def _build_comment_payload(
        self,
        issue_key: str,
        run_dir: Path,
        story_data: Dict[str, Any],
        testcases_data: Optional[Dict[str, Any]],
        coverage_data: Optional[Dict[str, Any]],
        exec_summary_data: Optional[Dict[str, Any]],
        defect_metrics_data: Optional[Dict[str, Any]],
        rca_data: Optional[Dict[str, Any]],
        uploaded_files: List[str],
        phase: str = "all"
    ) -> Dict[str, Any]:
        """
        Assemble the structured ADF JSON body based on what data exists and the target phase.
        """
        builder = ADFBuilder()

        # Title
        if phase == "generation":
            builder.add_heading(f"📊 QA Pipeline Run — Test Case Generation Summary — {issue_key}", level=1)
        elif phase == "execution":
            builder.add_heading(f"📊 QA Pipeline Run — Test Execution Summary — {issue_key}", level=1)
        elif phase == "defect":
            builder.add_heading(f"📊 QA Pipeline Run — Defect Analysis Summary — {issue_key}", level=1)
        else:
            builder.add_heading(f"📊 QA Pipeline Run Summary — {issue_key}", level=1)

        # Gather basic variables
        total_generated = testcases_data.get("generation_summary", {}).get("total_test_cases", 0) if testcases_data else 0
        executed_count = exec_summary_data.get("summary", {}).get("total_test_cases", 0) if exec_summary_data else 0

        # Determine execution scope text
        is_selected_run = False
        if exec_summary_data:
            if total_generated > 0 and executed_count < total_generated:
                is_selected_run = True
                scope_str = f"Selected execution run ({executed_count} of {total_generated} generated testcases executed)"
            else:
                scope_str = f"Full execution run ({executed_count} testcases executed)"
        else:
            scope_str = "No execution performed (Generation Only)"

        # 1. QA Run Summary
        builder.add_heading("QA Run Summary", level=2)
        if phase == "generation":
            status_str = "🟡 Generated (Execution Pending)"
        elif phase == "execution":
            if exec_summary_data:
                summary = exec_summary_data.get("summary", {})
                failed = summary.get("failed", 0)
                status_str = "🟢 Passed" if failed == 0 else "🔴 Failed"
            else:
                status_str = "🟡 Executed (Summary Missing)"
        elif phase == "defect":
            if defect_metrics_data:
                defect_candidates = defect_metrics_data.get("total_defect_candidates", 0)
                status_str = "🔴 Defect Candidates Identified" if defect_candidates > 0 else "🟢 No Defect Candidates"
            else:
                status_str = "🔴 Defect Analysis Completed"
        else:
            if exec_summary_data:
                summary = exec_summary_data.get("summary", {})
                failed = summary.get("failed", 0)
                status_str = "🟢 Passed" if failed == 0 else "🔴 Failed"
            else:
                status_str = "🟡 Generated (No Execution)"

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H-%M-%S UTC")
        summary_bullets = [
            f"**Status:** {status_str}",
            f"**Run Timestamp:** {timestamp}",
            f"**Target Run Directory:** `{run_dir.name}`",
            f"**Feature:** {story_data.get('summary', 'Unknown')}"
        ]
        builder.add_bullet_list(summary_bullets)

        # 2. Test Case Generation Summary
        if phase in ("generation", "all") and testcases_data:
            builder.add_heading("Test Generation Summary", level=2)
            gen_summary = testcases_data.get("generation_summary", {})
            
            gen_bullets = [f"**Total Test Cases Generated:** `{total_generated}`"]
            
            by_type = gen_summary.get("by_type", {})
            if by_type:
                type_str = ", ".join(f"{k}: {v}" for k, v in by_type.items())
                gen_bullets.append(f"**By Type:** {type_str}")
                
            by_priority = gen_summary.get("by_priority", {})
            if by_priority:
                prio_str = ", ".join(f"{k}: {v}" for k, v in by_priority.items())
                gen_bullets.append(f"**By Priority:** {prio_str}")
                
            builder.add_bullet_list(gen_bullets)

        # 3. Coverage Summary (Business-readable)
        if phase in ("generation", "all"):
            builder.add_heading("Coverage Summary", level=2)

            # Derive status from the actual coverage_summary.json data (coverage_score /
            # missing_paths) instead of unconditionally claiming "Fully Covered" whenever
            # any test cases exist — previously coverage_data was accepted as a parameter
            # here but never read, so this section always reported 100% regardless of
            # what was actually generated.
            missing_paths = (coverage_data or {}).get("missing_paths") or []
            coverage_score = (coverage_data or {}).get("coverage_score")
            if coverage_score is not None:
                score_pct = round(coverage_score * 100, 1)
                if coverage_score >= 0.95 and not missing_paths:
                    cov_status = f"🟢 Fully Covered ({score_pct}%)"
                elif coverage_score >= 0.6:
                    cov_status = f"🟡 Partially Covered ({score_pct}%)"
                else:
                    cov_status = f"🔴 Coverage Gaps Detected ({score_pct}%)"
            elif testcases_data:
                cov_status = "🟡 Generated (Coverage Score Not Available)"
            else:
                cov_status = "🟡 Generated (No Execution)"

            ac_list = story_data.get("acceptance_criteria", [])
            if ac_list:
                # Reports the count found in the story, not a fabricated coverage
                # percentage — this pipeline does not map individual test cases back to
                # specific acceptance criteria, so a per-criterion coverage % cannot be
                # honestly computed here.
                ac_cov_str = f"{len(ac_list)} Acceptance Criteria identified in story (per-criterion mapping not independently verified)"
            else:
                ac_cov_str = "N/A"

            cov_bullets = [
                f"**Coverage Status:** {cov_status}",
                f"**Execution Scope:** {scope_str}",
                f"**Acceptance Criteria Coverage:** {ac_cov_str}"
            ]

            if missing_paths:
                cov_bullets.append(
                    f"⚠️ **Uncovered behavior paths:** {len(missing_paths)} discovered "
                    f"path(s) not represented by any generated test case."
                )

            # Add coverage note if it's a selected execution run
            if is_selected_run:
                cov_bullets.append("⚠️ **Note:** This run targeted only selected test cases for validation, not the full regression suite.")
            else:
                cov_bullets.append("All generated test cases were target of this run.")

            builder.add_bullet_list(cov_bullets)

        # 4. Test Execution Summary
        if phase in ("execution", "all") and exec_summary_data:
            builder.add_heading("Test Execution Summary", level=2)
            summary = exec_summary_data.get("summary", {})
            passed = summary.get("passed", 0)
            failed = summary.get("failed", 0)
            skipped = summary.get("skipped", 0)
            pass_rate = summary.get("pass_rate", 0.0)
            duration = summary.get("total_duration_seconds", 0.0)

            exec_bullets = [
                f"**Execution Scope:** {scope_str}",
                f"**Passed:** `🟢 {passed}`",
                f"**Failed:** `🔴 {failed}`",
                f"**Skipped:** `🟡 {skipped}`",
                f"**Pass Rate:** `{pass_rate}%`",
                f"**Duration:** `{round(duration, 2)} seconds`"
            ]
            builder.add_bullet_list(exec_bullets)

        # 5. Defect Summary (Polished business-readable wording)
        if phase in ("defect", "all") and (defect_metrics_data or rca_data):
            builder.add_heading("Defect Summary", level=2)
            defect_bullets = []
            
            if defect_metrics_data:
                defect_candidates = defect_metrics_data.get("total_defect_candidates", 0)
                defect_bullets.append(f"**Defect Candidates Identified:** `{defect_candidates}`")
                
                density = defect_metrics_data.get("defect_density_summary", "")
                if density:
                    defect_bullets.append(f"**Defect Density:** {density}")
            
            if rca_data:
                rc_metrics = rca_data.get("rc_metrics", {})
                
                # Dominant failure cause
                dominant_causes = rc_metrics.get("dominant_failure_causes", {})
                if dominant_causes:
                    dominant_str = ", ".join(f"'{k}' ({v} instances)" for k, v in dominant_causes.items())
                    defect_bullets.append(f"**Dominant Failure Cause:** {dominant_str}")
                    
                # Repeated failure hotspots / Primary Failure Area
                hotspots = rc_metrics.get("repeated_failure_hotspots", {})
                cleaned_hotspots = []
                for k, v in hotspots.items():
                    cleaned_k = _clean_hotspot(k)
                    cleaned_hotspots.append(f"'{cleaned_k}' ({v} occurrences)")
                    
                if len(cleaned_hotspots) == 1:
                    defect_bullets.append(f"**Primary Failure Area:** {cleaned_hotspots[0]}")
                elif len(cleaned_hotspots) > 1:
                    defect_bullets.append(f"**Repeated Hotspots:** {', '.join(cleaned_hotspots)}")

            if defect_bullets:
                builder.add_bullet_list(defect_bullets)

        # 6. Attached QA Artifacts (Only show successful uploads)
        builder.add_heading("Attached QA Artifacts", level=2)
        builder.add_paragraph("The following run artifacts have been successfully uploaded as attachments directly to this Jira issue:")
        
        ref_bullets = []
        for file_name in uploaded_files:
            ref_bullets.append(f"`{file_name}`")
            
        if ref_bullets:
            builder.add_bullet_list(ref_bullets)
        else:
            builder.add_paragraph("_No supporting artifacts were attached for this run._")

        # Footer
        builder.add_paragraph("---")
        builder.add_paragraph("_Report generated automatically by Antigravity QA Agent Platform._")

        return {"body": builder.build()}

    # -----------------------------------------------------------------------
    # Jira client setup & post calls
    # -----------------------------------------------------------------------
    def _build_client(self, content_type: Optional[str] = "application/json") -> httpx.Client:
        """Build an authenticated httpx client."""
        token = base64.b64encode(
            f"{self.settings.jira_email}:{self.settings.jira_api_token}".encode()
        ).decode()

        headers = {
            "Authorization": f"Basic {token}",
            "Accept": "application/json",
            "X-Atlassian-Token": "no-check",
        }
        if content_type:
            headers["Content-Type"] = content_type

        return httpx.Client(
            base_url=self.settings.jira_base_url,
            headers=headers,
            timeout=30.0,
            follow_redirects=True,
        )

    @retry_on_failure(max_retries=3, initial_delay=1.0, backoff_factor=2, retry_codes={429, 503})
    def _post_comment(self, issue_key: str, payload: Dict[str, Any]) -> str:
        """
        Send POST request to add a comment to Jira issue.
        """
        with self._build_client() as client:
            response = client.post(
                f"/rest/api/3/issue/{issue_key}/comment",
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            return data.get("id", "")

    @retry_on_failure(max_retries=3, initial_delay=1.0, backoff_factor=2, retry_codes={429, 503})
    def _upload_attachment(self, issue_key: str, file_path: Path) -> bool:
        """
        Upload a file as attachment to a Jira issue.
        """
        if not file_path.exists():
            log.warning(f"[JiraUpdateAgent] File {file_path} does not exist. Skipping upload.")
            return False

        log.info(f"[JiraUpdateAgent] Uploading attachment: {file_path.name} ({file_path.stat().st_size} bytes)")
        
        with self._build_client(content_type=None) as client:
            with open(file_path, "rb") as f:
                files = {"file": (file_path.name, f)}
                response = client.post(
                    f"/rest/api/3/issue/{issue_key}/attachments",
                    files=files
                )
                response.raise_for_status()
                return True

    # -----------------------------------------------------------------------
    # Util helper
    # -----------------------------------------------------------------------
    @staticmethod
    def _read_json(path: Path) -> Optional[Dict[str, Any]]:
        """Safely read and parse json file."""
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning(f"Could not parse JSON file {path.name}: {exc}")
            return None
