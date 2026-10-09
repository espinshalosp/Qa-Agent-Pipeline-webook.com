# QA Agent Platform

> **Status**: 🏗️ Skeleton / Framework — Ready for Implementation
> **Version**: 0.1.0

An AI-powered, multi-agent QA automation platform that takes a Jira story as input and produces
test cases, Playwright automation scripts, executed results, and Jira defect tickets — fully
orchestrated by a chain of specialized AI agents.

---

## 🏛️ Architecture Overview

The platform is built on a **sequential multi-agent pipeline** where each agent has a single responsibility,
reads from well-defined input artifacts, and writes to well-defined output artifacts.

```
Jira Story (input)
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. Jira Agent          → Fetch & normalize story               │
│  2. Requirement Agent   → Extract testable requirements         │
│  3. TestCase Agent      → Generate structured test cases        │
│  4. Coverage Agent      → Validate coverage + gate check        │
│  5. Automation Agent    → Generate Playwright test scripts      │
│  6. Execution Agent     → Execute tests + capture screenshots   │
│  7. Defect Agent        → Analyze failures + build defect list  │
│  8. Jira Update Agent   → Create Jira tickets + post summary    │
└─────────────────────────────────────────────────────────────────┘
       │
       ▼
Pipeline Summary (artifacts/reports/pipeline_summary.json)
```

---

## 📁 Project Structure

```
qa-agent-platform/
│
├── agents/                    # Agent specification documents
│   ├── jira_agent.md          # Fetches Jira story data
│   ├── requirement_agent.md   # Extracts testable requirements
│   ├── testcase_agent.md      # Generates test cases
│   ├── coverage_agent.md      # Evaluates requirement coverage
│   ├── automation_agent.md    # Generates Playwright scripts
│   ├── execution_agent.md     # Runs tests, captures results
│   ├── defect_agent.md        # Classifies and reports defects
│   └── jira_update_agent.md   # Creates Jira tickets
│
├── skills/                    # Shared knowledge libraries used by agents
│   ├── jira_api.md            # Jira REST API patterns & field schemas
│   ├── playwright.md          # Playwright coding conventions & POM patterns
│   ├── qa_constitution.md     # Quality standards & acceptance criteria
│   ├── testcase_rules.md      # Test case schema & naming conventions
│   └── automation_rules.md    # Automation eligibility & code standards
│
├── artifacts/                 # Runtime data produced by agents
│   ├── story.json             # Normalized Jira story (→ Jira Agent)
│   ├── requirements.json      # Extracted requirements (→ Requirement Agent)
│   ├── testcases.json         # Generated test cases (→ TestCase Agent)
│   ├── automation/            # Generated Playwright scripts (→ Automation Agent)
│   ├── reports/               # All run reports (coverage, execution, defects, summary)
│   └── screenshots/           # Failure screenshots (→ Execution Agent)
│
├── manager/
│   └── qa_pipeline.md         # Master pipeline orchestration design
│
├── config/
│   ├── settings.yaml          # All configurable parameters
│   └── agents.yaml            # Agent registry with execution order & dependencies
│
├── logs/                      # Pipeline run logs
│
├── README.md                  # This file
└── requirements.txt           # Python dependencies
```

---

## 🤖 Agents

| # | Agent | Responsibility | Input | Output |
|---|-------|----------------|-------|--------|
| 1 | [Jira Agent](agents/jira_agent.md) | Fetch & normalize Jira story | Jira Issue Key | `story.json` |
| 2 | [Requirement Agent](agents/requirement_agent.md) | Extract requirements with AI | `story.json` | `requirements.json` |
| 3 | [TestCase Agent](agents/testcase_agent.md) | Generate test cases with AI | `requirements.json` | `testcases.json` |
| 4 | [Coverage Agent](agents/coverage_agent.md) | Evaluate coverage + gate | `requirements.json`, `testcases.json` | `coverage_report.json` |
| 5 | [Automation Agent](agents/automation_agent.md) | Generate Playwright scripts | `testcases.json` | `automation/*.py` |
| 6 | [Execution Agent](agents/execution_agent.md) | Execute tests + screenshots | `automation/*.py` | `execution_report.json` |
| 7 | [Defect Agent](agents/defect_agent.md) | Analyze failures + classify | `execution_report.json` | `defect_report.json` |
| 8 | [Jira Update Agent](agents/jira_update_agent.md) | Create tickets + summary | `defect_report.json` | Jira tickets + `pipeline_summary.json` |

---

## 🛠️ Skills

Shared reference libraries used by multiple agents:

| Skill | Used By |
|-------|---------|
| [Jira API](skills/jira_api.md) | Jira Agent, Jira Update Agent |
| [Playwright](skills/playwright.md) | Automation Agent, Execution Agent |
| [QA Constitution](skills/qa_constitution.md) | Requirement Agent, TestCase Agent, Coverage Agent, Defect Agent |
| [TestCase Rules](skills/testcase_rules.md) | TestCase Agent |
| [Automation Rules](skills/automation_rules.md) | Automation Agent |

---

## ⚙️ Configuration

### `config/settings.yaml`
All runtime parameters including Jira connection, LLM model selection, Playwright config,
coverage thresholds, and notification settings. See [settings.yaml](config/settings.yaml).

> ⚠️ **Security**: Never store API keys in `settings.yaml`. Use environment variables or a secrets manager.

### `config/agents.yaml`
Agent registry defining execution order, dependencies, inputs/outputs, and skill associations.
See [agents.yaml](config/agents.yaml).

---

## 🚀 Quick Start (Future)

Once implementation is complete, the pipeline will be invoked as:

```bash
# 1. Set up environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium

# 2. Configure credentials
cp .env.example .env
# Edit .env with your Jira credentials, LLM API key, and target app URL

# 3. Run the pipeline
python -m manager.qa_pipeline --story PROJ-123

# 4. (Optional) Dry run — logs actions without creating Jira tickets
python -m manager.qa_pipeline --story PROJ-123 --dry-run
```

---

## 🔄 Execution Flow

1. **Entry**: User provides a Jira issue key (e.g., `PROJ-123`).
2. **Fetch**: Jira Agent authenticates and fetches the story, writing to `artifacts/story.json`.
3. **Analyze**: Requirement Agent uses AI to extract testable requirements → `artifacts/requirements.json`.
4. **Generate**: TestCase Agent generates comprehensive test cases → `artifacts/testcases.json`.
5. **Gate**: Coverage Agent validates ≥80% requirement coverage. **Pipeline halts** if threshold not met.
6. **Automate**: Automation Agent generates Playwright scripts for eligible test cases → `artifacts/automation/`.
7. **Execute**: Execution Agent runs all scripts, capturing results and screenshots.
8. **Defects**: Defect Agent analyzes failures, classifies severity, prepares Jira payloads.
9. **Update**: Jira Update Agent creates bug tickets, links them to the story, posts execution summary.
10. **Done**: Final pipeline summary written to `artifacts/reports/pipeline_summary.json`.

---

## 📋 Pipeline Gates

| Gate | Condition | Action on Fail |
|------|-----------|----------------|
| Coverage Gate | `coverage ≥ 80%` | Halt pipeline, generate gap report |
| Execution Gate | `any test failures?` | Route to Defect Agent |
| Defect Gate | `defects found?` | Create Jira tickets; otherwise just post summary |

---

## 📦 Dependencies

See [requirements.txt](requirements.txt) for the full list. Key dependencies:

| Category | Libraries |
|----------|-----------|
| Jira API | `jira`, `requests`, `httpx` |
| AI/LLM | `langchain`, `langgraph` (provider-agnostic) |
| Browser Automation | `playwright`, `pytest-playwright` |
| Test Execution | `pytest`, `pytest-xdist`, `pytest-json-report` |
| Data & Reporting | `pandas`, `openpyxl`, `jinja2` |
| Validation | `pydantic`, `pydantic-settings` |
| CLI | `click`, `rich` |

---

## 🗺️ Roadmap

- [ ] **Phase 1**: Implement Jira Agent (API authentication + story fetch)
- [ ] **Phase 2**: Implement Requirement Agent (LLM prompt + extraction)
- [ ] **Phase 3**: Implement TestCase Agent (LLM prompt + schema validation)
- [ ] **Phase 4**: Implement Coverage Agent (mapping logic + gate check)
- [ ] **Phase 5**: Implement Automation Agent (Playwright script generation)
- [ ] **Phase 6**: Implement Execution Agent (pytest runner + result parsing)
- [ ] **Phase 7**: Implement Defect Agent (failure analysis + classification)
- [ ] **Phase 8**: Implement Jira Update Agent (ticket creation + summary)
- [ ] **Phase 9**: Implement QA Pipeline Manager (LangGraph orchestration)
- [ ] **Phase 10**: Build CLI + web dashboard

---

## 🤝 Contributing

1. Each agent lives in its own module under `agents/`.
2. All agents must read from and write to `artifacts/` only — no direct agent-to-agent calls.
3. All configuration must be loaded from `config/settings.yaml` via `pydantic-settings`.
4. All generated Playwright code must follow conventions in `skills/playwright.md`.
5. All test cases must conform to the schema in `skills/testcase_rules.md`.
6. Before implementing an agent, review its spec in `agents/<agent_name>.md`.

---

## 📄 License

<!-- TODO: Add your project license here -->
