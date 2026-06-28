# Agentic Clinical Decision Support — Proof of Concept

**Cotiviti GenAI Internship — Hackathon POC**
**Topic 2: Clinical Decision Making & Pattern Recognition in Health Care**
Author: Yash Bhamore

---

## What it does

Given a patient case (symptoms, age, vitals, history), an LLM-based **agentic**
assistant:

1. Runs a **deterministic, age-banded vitals check** (transparent rule layer)
2. Performs **chain-of-thought** differential diagnosis
3. Identifies the **top diagnosis** and suggests **ICD-10-CM codes** (the *Payment* in TPO)
4. **Flags anomalies / risk signals**, adjusted for the patient's age group
5. Returns a **structured clinical summary** (Pydantic schema enforced on Claude's output via the `instructor` library)

## Why the design is interesting

- **Hybrid, not just an API wrapper.** A rule-based vitals checker runs *first*
  and feeds the model verified, auditable facts. The model reasons on top of
  objective checks rather than replacing them — an AI-governance pattern that
  maps directly to Cotiviti's healthcare context.
- **Strict structured output.** The response always conforms to a Pydantic
  schema, so a live demo never breaks on malformed JSON.
- **Age-specific anomaly detection.** The same vitals are judged against
  infant / child / adolescent / adult / geriatric reference ranges.

## Run it

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="sk-ant-..."    # Windows: set ANTHROPIC_API_KEY=...

python clinical_decision_agent.py        # all 3 example cases
python clinical_decision_agent.py 2      # only case #2 (pediatric)
python clinical_decision_agent.py --model claude-sonnet-4-5-20250929   # override model
```

Three built-in cases (cardiac / pediatric / geriatric) are included for a live run.

## Architecture

```
Patient case ──► [Rule-based vitals check] ──► verified flags ─┐
                                                               ├─► [LLM agent] ──► Structured JSON
                          system prompt + case + flags ────────┘        │
                                                                        ├─ chain-of-thought
                                                                        ├─ differential dx
                                                                        ├─ ICD-10 codes
                                                                        ├─ risk flags
                                                                        └─ clinical summary
```

## ⚠️ Disclaimer

This is a **demonstration prototype** for a technical assessment. It is **not a
medical device**, is **not FDA-cleared**, and must **not** be used for real
clinical care. All outputs require review by a licensed clinician.
