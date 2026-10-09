"""
manager/qa_pipeline.py
=======================
QA Pipeline Manager — Orchestrates the 3-agent pipeline (Phase 1).

Pipeline (current):
    1. Jira Agent         → artifacts/story.json
    2. Requirement Agent  → artifacts/requirements.json
    3. TestCase Agent     → artifacts/testcases.json

Future phases (Coverage → Automation → Execution → Defect → Jira Update)
are not wired yet.

Usage
-----
    # From project root:
    python -m manager.qa_pipeline

    # With debug logging:
    LOG_LEVEL=DEBUG python -m manager.qa_pipeline
"""
from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from config.settings import get_settings
from utils.logger import configure as configure_logging
from utils.logger import get_logger

log = get_logger(__name__)


def main() -> None:
    # -------------------------------------------------------------------------
    # 0. Bootstrap
    # -------------------------------------------------------------------------
    settings = get_settings()
    run_id = uuid.uuid4().hex[:8]
    pipeline_started_at = datetime.now(timezone.utc)

    configure_logging(
        log_level=settings.log_level,
        log_dir=settings.log_dir,
        run_id=run_id,
    )

    # Only ensure root base directories exist globally
    settings.artifact_dir.mkdir(parents=True, exist_ok=True)
    settings.log_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info(f"  QA Agent Platform — Pipeline Start")
    log.info(f"  Run ID   : {run_id}")
    log.info(f"  Pipeline : Jira → TestCase Generation → Manual Approval → Execution")
    log.info("=" * 60)

    # -------------------------------------------------------------------------
    # 1. Jira Agent — fetch & normalise all QA issues
    # -------------------------------------------------------------------------
    from agents.jira_agent import JiraAgent

    jira_agent = JiraAgent(settings)
    try:
        story_paths = jira_agent.run()
    except Exception as exc:
        # Every later pipeline stage is wrapped per-story so one failure doesn't take down
        # the whole run — Jira discovery itself (auth failure, bad JQL, network outage) was
        # the one exception, crashing main() with a raw traceback instead of exiting cleanly.
        log.error(f"[Pipeline] Jira discovery failed: {exc}", exc_info=True)
        sys.exit(1)

    if not story_paths:
        log.info("No QA stories available. Pipeline exiting gracefully.")
        sys.exit(0)

    log.info(f"Pipeline found {len(story_paths)} story file(s). Processing sequentially...")

    all_results = []

    for story_path in story_paths:
        # Read the issue key for logging context
        story_data = json.loads(story_path.read_text(encoding="utf-8"))
        issue_key = story_data.get("story", {}).get("key", "UNKNOWN")

        log.info("-" * 60)
        log.info(f"Processing: {issue_key}")
        log.info("-" * 60)

        result = _run_single_story(settings, story_path, issue_key, run_id)
        all_results.append(result)

    # Reconfigure global logging back to the original logs directory
    configure_logging(
        log_level=settings.log_level,
        log_dir=settings.log_dir,
        run_id=run_id,
    )

    # -------------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------------
    _write_pipeline_summary(settings, run_id, all_results, pipeline_started_at)

    success_count = sum(1 for r in all_results if r["status"] == "success")
    fail_count = len(all_results) - success_count

    log.info("=" * 60)
    log.info(f"  Pipeline Complete")
    log.info(f"  Stories processed : {len(all_results)}")
    log.info(f"  Succeeded         : {success_count}")
    log.info(f"  Failed            : {fail_count}")
    for r in all_results:
        s_path = r.get("story_path")
        if s_path:
            story_run_dir = Path(s_path).parent
            log.info(f"  Summary ({r['issue_key']}) : {story_run_dir / 'execution' / 'pipeline_summary.json'}")
    log.info("=" * 60)

    sys.exit(0 if fail_count == 0 else 1)


def _run_single_story(settings, story_path: Path, issue_key: str, run_id: str) -> dict:
    """Run the unified pipeline for a single story. Returns a result dict."""
    # EXECUTE_TESTCASE_IDS is set on os.environ by _request_manual_approval when the user
    # picks partial execution for a story, and the pipeline processes stories sequentially
    # within one process. Clear any value left over from a previous story here so it can't
    # silently leak into this story's execution (e.g. if this story skips the interactive
    # approval prompt in a non-interactive run). _request_manual_approval sets it fresh
    # below if this story's own approval gate selects partial execution.
    import os
    os.environ.pop("EXECUTE_TESTCASE_IDS", None)

    result = {
        "issue_key": issue_key,
        "story_path": str(story_path),
        "status": "success",
        "stages": {},
        "error": None,
    }

    # Scope all settings to the story-specific run directory
    run_dir = story_path.parent
    
    orig_artifact_dir = settings.artifact_dir
    orig_report_dir = settings.report_dir
    orig_screenshot_dir = settings.screenshot_dir
    orig_attachment_dir = settings.attachment_dir
    orig_log_dir = settings.log_dir
    orig_execution_dir = settings.execution_dir
    orig_defect_dir = settings.defect_dir
    
    settings.artifact_dir = run_dir
    settings.report_dir = run_dir / "reports"
    settings.screenshot_dir = run_dir / "screenshots"
    settings.attachment_dir = run_dir / "attachments"
    settings.log_dir = run_dir / "logs"
    settings.execution_dir = run_dir / "execution"
    settings.defect_dir = run_dir / "defects"
    
    settings.ensure_dirs()

    # Always ensure standardized subdirectories exist
    (run_dir / "traces").mkdir(parents=True, exist_ok=True)
    (run_dir / "videos").mkdir(parents=True, exist_ok=True)
    (run_dir / "reports" / "allure-report").mkdir(parents=True, exist_ok=True)
    (run_dir / "reports" / "allure-results").mkdir(parents=True, exist_ok=True)
    (run_dir / "execution").mkdir(parents=True, exist_ok=True)
    (run_dir / "defects").mkdir(parents=True, exist_ok=True)

    # Reconfigure root logging handlers to write to the run-scoped log directory
    configure_logging(
        log_level=settings.log_level,
        log_dir=settings.log_dir,
        run_id=run_id,
    )

    try:
        # ------------------------------------------------------------------
        # 2. TestCase Generation Agent
        # ------------------------------------------------------------------
        from agents.testcase_generation_agent import TestCaseGenerationAgent

        gen_agent = TestCaseGenerationAgent(settings)
        testcases_path = gen_agent.run(story_path)
        result["stages"]["testcase_generation_agent"] = {
            "status": "success",
            "output": str(testcases_path),
        }

        # Log the test case count for easy visibility
        testcases_data = json.loads(testcases_path.read_text(encoding="utf-8"))
        tc_count = testcases_data.get("generation_summary", {}).get("total_test_cases", 0)
        if testcases_data.get("_metadata", {}).get("exploration_incomplete"):
            # UI exploration crashed/failed and stayed failed even after a retry — the test
            # cases below aren't grounded in real discovered UI elements for this domain, so
            # this must not read the same as a normal, fully-grounded generation run.
            log.error(
                f"[Pipeline] ⚠ {issue_key}: exploration was INCOMPLETE — the {tc_count} generated "
                f"test case(s) are NOT grounded in real discovered UI elements and should be "
                f"treated as low-confidence pending manual review before execution."
            )
            result["exploration_incomplete"] = True
        log.info(f"[Pipeline] ✓ {issue_key} complete — {tc_count} test case(s) generated.")

        # ------------------------------------------------------------------
        # 3. Jira Update Approval Gate (Generation)
        # ------------------------------------------------------------------
        gen_jira_approved = _request_jira_update_approval(settings, story_path, issue_key)

        if gen_jira_approved:
            log.info(f"[Pipeline] Updating Jira with Test Case Generation details and attachments for {issue_key}...")
            from agents.jira_update_agent import JiraUpdateAgent
            jira_update_agent = JiraUpdateAgent(settings)
            gen_comment_id = jira_update_agent.run(story_path, phase="generation")
            result["stages"]["jira_update_generation"] = {
                "status": "success" if gen_comment_id else "failed",
                "output": gen_comment_id,
            }
        else:
            result["stages"]["jira_update_generation"] = {
                "status": "skipped",
                "output": None,
            }

        # ------------------------------------------------------------------
        # 4. Manual Approval Gate
        # ------------------------------------------------------------------
        approval = _request_manual_approval(issue_key, tc_count, testcases_data, testcases_path)
        result["stages"]["manual_approval_gate"] = {
            "status": "approved" if approval else "rejected",
        }

        if not approval:
            log.info(f"[Pipeline] Manual approval not received. Skipping execution for {issue_key}.")
            result["stages"]["jira_update_execution"] = {
                "status": "skipped",
                "output": None,
            }
            result["stages"]["jira_update_defect"] = {
                "status": "skipped",
                "output": None,
            }
            result["stages"]["jira_update_agent"] = {
                "status": "skipped",
                "output": None,
            }
            return result

        # ------------------------------------------------------------------
        # 5. TestCase Execution Agent
        # ------------------------------------------------------------------
        log.info(f"[Pipeline] Invoking Execution Agent for {issue_key}...")
        from agents.testcase_execution_agent import TestCaseExecutionAgent

        exec_agent = TestCaseExecutionAgent(settings)
        report_path = exec_agent.run(story_path)
        result["stages"]["execution_agent"] = {
            "status": "success",
            "output": str(report_path),
        }

        # ------------------------------------------------------------------
        # 6. Defect Agent
        # ------------------------------------------------------------------
        log.info(f"[Pipeline] Invoking Defect Agent for {issue_key}...")
        from agents.defect_agent import DefectAgent

        defect_agent = DefectAgent(settings)
        defect_report_path = defect_agent.run(story_path)
        result["stages"]["defect_agent"] = {
            "status": "success",
            "output": str(defect_report_path),
        }

        # ------------------------------------------------------------------
        # 7. Combined Jira Update Approval Gate (Execution & Defect)
        # ------------------------------------------------------------------
        post_jira_approved = _request_jira_update_approval(settings, story_path, issue_key)

        if post_jira_approved:
            log.info(f"[Pipeline] Updating Jira with Test Execution details and attachments for {issue_key}...")
            from agents.jira_update_agent import JiraUpdateAgent
            jira_update_agent = JiraUpdateAgent(settings)
            
            exec_comment_id = jira_update_agent.run(story_path, phase="execution")
            result["stages"]["jira_update_execution"] = {
                "status": "success" if exec_comment_id else "failed",
                "output": exec_comment_id,
            }

            log.info(f"[Pipeline] Updating Jira with Defect Analysis details and attachments for {issue_key}...")
            defect_comment_id = jira_update_agent.run(story_path, phase="defect")
            result["stages"]["jira_update_defect"] = {
                "status": "success" if defect_comment_id else "failed",
                "output": defect_comment_id,
            }
            # For backward compatibility, map the final status to jira_update_agent stage
            result["stages"]["jira_update_agent"] = {
                "status": "success" if defect_comment_id else "failed",
                "output": defect_comment_id,
            }
        else:
            result["stages"]["jira_update_execution"] = {
                "status": "skipped",
                "output": None,
            }
            result["stages"]["jira_update_defect"] = {
                "status": "skipped",
                "output": None,
            }
            result["stages"]["jira_update_agent"] = {
                "status": "skipped",
                "output": None,
            }



    except Exception as exc:
        log.error(f"[Pipeline] ERROR processing {issue_key}: {exc}", exc_info=True)
        result["status"] = "failed"
        result["error"] = str(exc)
    finally:
        # Restore settings
        settings.artifact_dir = orig_artifact_dir
        settings.report_dir = orig_report_dir
        settings.screenshot_dir = orig_screenshot_dir
        settings.attachment_dir = orig_attachment_dir
        settings.log_dir = orig_log_dir
        settings.execution_dir = orig_execution_dir
        settings.defect_dir = orig_defect_dir

    return result


def _request_manual_approval(issue_key: str, tc_count: int, testcases_data: dict, testcases_path: Path) -> bool:
    """
    Request manual approval for execution.
    
    Displays a summary of generated test cases and prompts for approval.
    Only an explicit 'YES' or 'Y' response proceeds.
    """
    log.info("=" * 60)
    log.info("  MANUAL APPROVAL GATE")
    log.info("=" * 60)
    log.info(f"Test case generation completed successfully for {issue_key}.")
    log.info(f"")
    log.info(f"Generated Test Cases: {tc_count}")
    log.info(f"")
    
    # Display summary of test case types
    test_cases = testcases_data.get("test_cases", [])
    type_counts = {}
    for tc in test_cases:
        tc_type = tc.get("type", "Functional")
        type_counts[tc_type] = type_counts.get(tc_type, 0) + 1
    
    if type_counts:
        log.info("Test Case Types:")
        for tc_type, count in sorted(type_counts.items()):
            log.info(f"  - {tc_type}: {count}")
    
    log.info(f"")
    log.info("You may now review, modify, remove, or add test cases.")
    log.info(f"Test cases location: {testcases_path}")
    log.info(f"")
    log.info("Execution Mode Selection:")
    log.info(f"")
    log.info("1. Full Execution")
    log.info("2. Execute only selected test cases")
    log.info("3. Exit")
    log.info("=" * 60)
    
    # Prompt for approval
    import os
    if os.environ.get("SKIP_EXECUTION") == "true":
        log.info(f"[Pipeline] SKIP_EXECUTION=true detected. Skipping execution.")
        return False

    if os.environ.get("AUTO_APPROVE") == "true" or not sys.stdin.isatty():
        if "EXECUTE_TESTCASE_IDS" in os.environ:
            log.info(f"[Pipeline] Non-interactive context detected with EXECUTE_TESTCASE_IDS={os.environ['EXECUTE_TESTCASE_IDS']}. Executing selected test cases.")
        else:
            log.info(f"[Pipeline] Non-interactive or auto-approve context detected. Auto-approving full execution for {issue_key}.")
        return True

    while True:
        try:
            choice = input("Select an option (1, 2, or 3): ").strip()
            if choice == "1":
                log.info(f"[Pipeline] Full execution selected. Proceeding to execution.")
                if "EXECUTE_TESTCASE_IDS" in os.environ:
                    del os.environ["EXECUTE_TESTCASE_IDS"]
                return True
            elif choice == "2":
                # Build mapping of available testcase numbers
                test_cases = testcases_data.get("test_cases", [])
                available_nums = {}
                import re
                for tc in test_cases:
                    matches = re.findall(r'\d+', tc.get("id", ""))
                    if matches:
                        num = int(matches[-1])
                        available_nums[num] = tc

                while True:
                    ids_input = input("Enter testcase ID numbers to execute (comma-separated): ").strip()
                    if ids_input.lower() in ["cancel", "c", "exit"]:
                        print("Selection cancelled.")
                        break
                    if not ids_input:
                        print("Input is empty. Please enter comma-separated numbers.")
                        continue
                    
                    selected_parts = ids_input.split(",")
                    selected_nums = []
                    warnings = []
                    
                    for part in selected_parts:
                        part_clean = part.strip()
                        if not part_clean:
                            continue
                        
                        if part_clean.isdigit():
                            selected_nums.append(int(part_clean))
                        elif "-" in part_clean:
                            subparts = part_clean.split("-")
                            if len(subparts) == 2 and subparts[0].strip().isdigit() and subparts[1].strip().isdigit():
                                start = int(subparts[0].strip())
                                end = int(subparts[1].strip())
                                if start <= end:
                                    selected_nums.extend(range(start, end + 1))
                                else:
                                    warnings.append(f"Invalid range: '{part_clean}'. Start of range must be less than or equal to end of range.")
                            else:
                                warnings.append(f"Invalid range format: '{part_clean}'. Expected 'start-end'.")
                        else:
                            warnings.append(f"Invalid numeric input: '{part_clean}'. Expected a positive integer or range.")
                    
                    # Deduplicate while preserving order
                    seen = set()
                    unique_nums = []
                    for n in selected_nums:
                        if n not in seen:
                            seen.add(n)
                            unique_nums.append(n)
                    selected_nums = unique_nums
                    
                    # Validate selections against available testcases
                    existing_selections = []
                    missing_nums = []
                    for num in selected_nums:
                        if num in available_nums:
                            existing_selections.append(num)
                        else:
                            missing_nums.append(num)
                    
                    # Output warnings for formatting/syntax issues
                    for warn in warnings:
                        print(f"Warning: {warn}")
                        log.warning(warn)
                    
                    # Output warnings for non-existent test case numbers
                    for num in missing_nums:
                        print(f"Warning: Test case number {num} does not exist in the loaded test cases. Ignoring.")
                        log.warning(f"Test case number {num} does not exist in the loaded test cases. Ignoring.")
                    
                    if not existing_selections:
                        print("All inputs are invalid or no matching test cases found. Please try again or type 'cancel' to exit.")
                        continue
                    
                    # Store selected numbers in env to pass to ExecutionAgent
                    selected_str = ",".join(str(n) for n in existing_selections)
                    os.environ["EXECUTE_TESTCASE_IDS"] = selected_str
                    log.info(f"[Pipeline] Partial execution selected (selected IDs: {selected_str}). Proceeding to execution.")
                    return True
            elif choice == "3":
                log.info(f"[Pipeline] Manual execution denied/cancelled for {issue_key}. Exiting.")
                return False
            else:
                log.warning("Invalid option. Please enter 1, 2, or 3.")
        except (EOFError, KeyboardInterrupt):
            log.info(f"[Pipeline] Input interrupted. Skipping execution for {issue_key}.")
            return False
def _request_jira_update_approval(settings, story_path: Path, issue_key: str) -> bool:
    """
    Request manual approval before posting to Jira.
    """
    import os
    if os.environ.get("SKIP_JIRA_UPDATE") == "true":
        log.info(f"[Pipeline] SKIP_JIRA_UPDATE=true detected. Skipping Jira update.")
        return False

    if os.environ.get("AUTO_APPROVE") == "true" or not sys.stdin.isatty():
        log.info(f"[Pipeline] Non-interactive or auto-approve context detected. Auto-approving Jira update for {issue_key}.")
        return True

    run_dir = story_path.parent
    testcases_path = run_dir / "testcases.json"
    exec_summary_path = run_dir / "execution" / "execution_summary.json"
    defect_metrics_path = run_dir / "defects" / "defect_metrics.json"

    # Read artifacts safely
    story_data = {}
    if story_path.exists():
        try:
            story_data = json.loads(story_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    testcases_data = {}
    if testcases_path.exists():
        try:
            testcases_data = json.loads(testcases_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    exec_summary_data = {}
    if exec_summary_path.exists():
        try:
            exec_summary_data = json.loads(exec_summary_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    defect_metrics_data = {}
    if defect_metrics_path.exists():
        try:
            defect_metrics_data = json.loads(defect_metrics_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    # Extract info
    story_title = story_data.get("story", {}).get("summary", "Unknown")
    tc_count = testcases_data.get("generation_summary", {}).get("total_test_cases", 0)

    exec_info = None
    if exec_summary_data:
        summary = exec_summary_data.get("summary", {})
        exec_info = {
            "total": summary.get("total_test_cases", 0),
            "passed": summary.get("passed", 0),
            "failed": summary.get("failed", 0),
            "skipped": summary.get("skipped", 0),
            "pass_rate": summary.get("pass_rate", 0.0)
        }

    defect_count = None
    if defect_metrics_data:
        defect_count = defect_metrics_data.get("total_defect_candidates", 0)

    log.info("=" * 60)
    log.info("  JIRA UPDATE MANUAL APPROVAL GATE")
    log.info("=" * 60)
    log.info(f"Story Key : {issue_key}")
    log.info(f"Module    : {story_title}")
    log.info(f"Testcases : {tc_count} generated")
    if exec_info:
        log.info(f"Execution : {exec_info['total']} executed (Passed: {exec_info['passed']}, Failed: {exec_info['failed']}, Skipped: {exec_info['skipped']}, Pass Rate: {exec_info['pass_rate']}%)")
    if defect_count is not None:
        log.info(f"Defects   : {defect_count} candidate(s) identified")
    log.info("=" * 60)
    log.info("Jira Update Selection:")
    log.info("1. Update Jira")
    log.info("2. Skip Jira update for this run")
    log.info("3. Exit")
    log.info("=" * 60)

    while True:
        try:
            choice = input("Select an option (1, 2, or 3): ").strip()
            if choice == "1":
                log.info(f"[Pipeline] Jira update approved for {issue_key}.")
                return True
            elif choice == "2":
                log.info(f"[Pipeline] Jira update skipped for {issue_key}.")
                return False
            elif choice == "3":
                log.info(f"[Pipeline] User selected Exit. Terminating pipeline execution.")
                sys.exit(0)
            else:
                log.warning("Invalid option. Please enter 1, 2, or 3.")
        except (EOFError, KeyboardInterrupt):
            log.info(f"[Pipeline] Input interrupted. Skipping Jira update for {issue_key}.")
            return False


def _write_pipeline_summary(settings, run_id: str, results: list, started_at: datetime) -> None:

    """Write a final pipeline summary JSON to artifacts/reports/."""
    summary = {
        "_schema_version": "1.0.0",
        "run_id": run_id,
        "pipeline_stages": ["jira_agent", "testcase_generation_agent", "execution_agent"],
        # Previously this was set to datetime.now() at write time — i.e. after every story
        # had finished processing — so it was actually the completion time mislabeled as
        # "started_at". Now takes the real start-of-run timestamp captured in main().
        "started_at": started_at.isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "total_stories": len(results),
        "succeeded": sum(1 for r in results if r["status"] == "success"),
        "failed": sum(1 for r in results if r["status"] == "failed"),
        "results": results,
    }

    # Write the pipeline summary only inside each story's run-scoped reports directory

    # Also write a copy to each story's run-scoped reports directory
    for r in results:
        s_path = r.get("story_path")
        if s_path:
            story_run_dir = Path(s_path).parent
            s_exec_dir = story_run_dir / "execution"
            s_exec_dir.mkdir(parents=True, exist_ok=True)
            try:
                (s_exec_dir / "pipeline_summary.json").write_text(
                    json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
                )
            except Exception as e:
                log.warning(f"Could not write pipeline summary to story run dir: {e}")


if __name__ == "__main__":
    main()
