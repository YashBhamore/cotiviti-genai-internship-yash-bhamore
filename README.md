# Cotiviti GenAI Internship Assessment

**Author:** Yash Bhamore | M.S. Data Science, University of North Texas

**Topic:** Clinical Decision Making & Pattern Recognition in Health Care

## Overview

This repository contains an agentic clinical decision support proof of concept for the Cotiviti GenAI internship assessment.

Given a patient case with symptoms, age, vitals, and history, the assistant:

1. Runs a deterministic, age-banded vitals check
2. Performs differential diagnosis reasoning
3. Identifies the top diagnosis and suggests ICD-10-CM codes
4. Flags anomalies and risk signals based on age group
5. Returns a structured clinical summary

## Why the design is interesting

- Hybrid, not just an API wrapper. A rule-based vitals checker runs first and feeds the model verified, auditable facts.
- Strict structured output. The response is designed to stay consistent and machine-readable.
- Age-specific anomaly detection. The same vitals are judged against infant, child, adolescent, adult, and geriatric reference ranges.

## Files

- `clinical_decision_agent.py` - Proof of concept code
- `README.md` - Project overview and run instructions
- `requirements.txt` - Python dependencies
- `.gitignore` - Local environment and cache exclusions

## How to Run the POC

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your_api_key"
python clinical_decision_agent.py
```

## Additional Notes

- The project includes example cases for demonstration.
- This is a prototype for assessment purposes and is not intended for clinical use.

## Disclaimer

This is a demonstration prototype for a technical assessment. It is not a medical device, is not FDA-cleared, and must not be used for real clinical care. All outputs require review by a licensed clinician.
