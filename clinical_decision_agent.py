"""
clinical_decision_agent.py
--------------------------
Author : Yash Bhamore
Course : Cotiviti GenAI Internship — Technical Assessment
Date   : June 2026

Overview
--------
This script is my proof-of-concept for applying agentic LLM reasoning to
clinical decision support. The core idea: instead of throwing raw patient
data at a model and hoping for the best, I split the work between a
deterministic vitals-checking layer (plain Python, fully auditable) and an
LLM that reasons on top of those verified facts.

The agent takes a patient case and returns:
  - A step-by-step chain-of-thought differential diagnosis
  - ICD-10-CM billing codes for the top diagnosis
  - Risk flags for abnormal vitals, cross-checked against age-specific ranges
  - A structured clinical summary enforced via a Pydantic schema

Architecture note
-----------------
Anthropic's Claude API doesn't support native structured outputs the way
OpenAI does, so I'm using the `instructor` library to enforce the Pydantic
schema through tool-calling. If the model returns a malformed response,
instructor automatically retries — the demo never crashes on bad output.

Three hardcoded cases are included (cardiac, pediatric, geriatric) so you
can run the demo cleanly without any setup beyond the API key.

Disclaimer
----------
Demonstration only. Not a medical device. Not FDA-cleared. Every output
requires review by a licensed clinician before any clinical use.

Usage
-----
  pip install anthropic instructor pydantic
  export ANTHROPIC_API_KEY="sk-ant-..."
  python clinical_decision_agent.py          # all 3 cases
  python clinical_decision_agent.py 2        # case #2 only
  python clinical_decision_agent.py --model claude-sonnet-4-5
"""

import os
import sys
import json
from typing import List, Optional, Literal

from pydantic import BaseModel, Field
import instructor
from anthropic import Anthropic

# Default to the cheap, fast Haiku model for a live demo; override with --model
DEFAULT_MODEL = "claude-haiku-4-5"
# Anthropic requires an explicit output-token budget on every call.
MAX_TOKENS = 4096


# ===========================================================================
# 1. INPUT SCHEMA  — the patient case we send to the agent
#    (We construct these ourselves, so normal Python defaults are fine here.)
# ===========================================================================
class Vitals(BaseModel):
    heart_rate: Optional[int] = None          # beats per minute
    systolic_bp: Optional[int] = None         # mmHg
    diastolic_bp: Optional[int] = None        # mmHg
    respiratory_rate: Optional[int] = None    # breaths per minute
    temperature_f: Optional[float] = None     # degrees Fahrenheit
    spo2: Optional[int] = None                # oxygen saturation %


class PatientCase(BaseModel):
    case_id: str
    age: int
    sex: str
    chief_complaint: str
    symptoms: List[str]
    history: str
    vitals: Vitals


# ===========================================================================
# 2. OUTPUT SCHEMA  — what the LLM must return (enforced via instructor)
#    Keep every field REQUIRED so the schema is strict and the demo is rich.
# ===========================================================================
class ReasoningStep(BaseModel):
    """One explicit step in the model's chain-of-thought."""
    step_number: int
    thought: str


class DifferentialDiagnosis(BaseModel):
    condition: str
    likelihood: Literal["high", "moderate", "low"]
    supporting_evidence: List[str]
    reasoning: str


class ICD10Code(BaseModel):
    code: str                 # e.g. "I20.0"
    description: str          # e.g. "Unstable angina"
    rationale: str            # why this code fits the top diagnosis


class RiskFlag(BaseModel):
    signal: str                                              # e.g. "Tachycardia"
    detail: str                                              # the observed value + context
    severity: Literal["low", "moderate", "high", "critical"]
    recommended_action: str


class ClinicalAssessment(BaseModel):
    """The full structured output of one agent run."""
    reasoning_steps: List[ReasoningStep]
    differential_diagnosis: List[DifferentialDiagnosis]
    top_diagnosis: str
    suggested_icd10_codes: List[ICD10Code]
    risk_flags: List[RiskFlag]
    recommended_next_steps: List[str]
    clinical_summary: str
    confidence: Literal["high", "moderate", "low"]
    disclaimer: str


# ===========================================================================
# 3. DETERMINISTIC VITALS CHECK  — the rule-based, explainable layer
#    Age-banded reference ranges (simplified for demo). This runs in plain
#    Python so the flags are auditable and reproducible, independent of the LLM.
# ===========================================================================
def _ranges_for_age(age: int) -> dict:
    """Return simplified normal vital-sign ranges for an age band.

    Ranges are approximate teaching values for demonstration only.
    """
    if age < 1:        # infant
        return dict(hr=(100, 160), rr=(30, 60), sbp=(70, 100), spo2=(95, 100))
    elif age < 12:     # child
        return dict(hr=(70, 120), rr=(18, 30), sbp=(80, 110), spo2=(95, 100))
    elif age < 18:     # adolescent
        return dict(hr=(60, 100), rr=(12, 20), sbp=(90, 120), spo2=(95, 100))
    elif age < 65:     # adult
        return dict(hr=(60, 100), rr=(12, 20), sbp=(90, 120), spo2=(95, 100))
    else:              # geriatric
        return dict(hr=(60, 100), rr=(12, 20), sbp=(90, 130), spo2=(94, 100))


def rule_based_vitals_flags(case: PatientCase) -> List[str]:
    """Flag out-of-range vitals using transparent, age-specific rules.

    Returns a list of human-readable flag strings. These are passed to the
    LLM as VERIFIED facts so the model reasons on top of objective checks.
    """
    r = _ranges_for_age(case.age)
    v = case.vitals
    flags: List[str] = []

    def check(value, low, high, label, unit):
        if value is None:
            return
        if value < low:
            flags.append(f"{label} LOW: {value}{unit} (age-normal {low}-{high}{unit})")
        elif value > high:
            flags.append(f"{label} HIGH: {value}{unit} (age-normal {low}-{high}{unit})")

    check(v.heart_rate, *r["hr"], "Heart rate", " bpm")
    check(v.respiratory_rate, *r["rr"], "Respiratory rate", " /min")
    check(v.systolic_bp, *r["sbp"], "Systolic BP", " mmHg")
    check(v.spo2, *r["spo2"], "SpO2", "%")

    # A couple of fixed, universally-meaningful thresholds
    if v.temperature_f is not None and v.temperature_f >= 100.4:
        flags.append(f"Fever: {v.temperature_f}F (>=100.4F)")
    if v.temperature_f is not None and v.temperature_f <= 95.0:
        flags.append(f"Hypothermia: {v.temperature_f}F (<=95.0F)")
    if v.spo2 is not None and v.spo2 < 92:
        flags.append(f"Hypoxemia: SpO2 {v.spo2}% (<92% warrants urgent attention)")

    return flags


# ===========================================================================
# 4. THE AGENT  — system prompt + LLM call with schema-enforced output
# ===========================================================================
SYSTEM_PROMPT = """You are an agentic clinical decision-support assistant used \
for medical education and operational triage support. You DO NOT replace a \
licensed clinician; your role is to reason transparently and surface options.

For each patient case you must:
1. Think step by step (explicit chain-of-thought) about the presentation,
   correlating symptoms, age, history, and vitals.
2. Build a ranked differential diagnosis with supporting evidence and reasoning.
3. Identify the single most likely (top) diagnosis.
4. Suggest plausible ICD-10-CM codes for the top diagnosis, each with a short
   rationale. Use real ICD-10-CM code formats. If unsure of an exact code,
   give the closest category code and say so in the rationale.
5. Review the AUTOMATED VITALS FLAGS provided to you. Treat them as verified
   objective facts. Confirm them, add clinical interpretation, and add any
   additional risk signals you infer. Do not contradict an objective flag.
6. Recommend concrete next steps (tests, monitoring, escalation).
7. Write a concise clinical summary a clinician could read in 15 seconds.
8. State an overall confidence level and ALWAYS include a safety disclaimer
   that this is decision support, not a diagnosis, and requires clinician review.

Be calibrated: express uncertainty honestly and never fabricate certainty."""


def build_user_message(case: PatientCase, auto_flags: List[str]) -> str:
    """Assemble the user message: the case + the rule-based flags."""
    case_json = case.model_dump_json(indent=2)
    flags_block = "\n".join(f"- {f}" for f in auto_flags) if auto_flags else "- None detected by rule-based check."
    return (
        "PATIENT CASE (JSON):\n"
        f"{case_json}\n\n"
        "AUTOMATED VITALS FLAGS (rule-based, age-adjusted, treat as verified):\n"
        f"{flags_block}\n\n"
        "Produce your full structured clinical assessment now."
    )


def analyze_case(client, case: PatientCase, model: str) -> ClinicalAssessment:
    """Run one full agent pass on a patient case and return structured output."""
    # Step 1: deterministic, auditable vitals check (runs before the model)
    auto_flags = rule_based_vitals_flags(case)

    # Step 2: LLM reasoning, coerced into our Pydantic schema by instructor.
    # Note: Anthropic uses a dedicated `system` parameter (not a system message),
    # and requires `max_tokens`. instructor's Mode.TOOLS enforces the schema.
    assessment = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": build_user_message(case, auto_flags)},
        ],
        response_model=ClinicalAssessment,   # <-- Pydantic schema enforcement
        temperature=0.2,                      # low temp for clinical consistency
        max_retries=2,                        # re-ask if validation fails
    )
    return assessment


# ===========================================================================
# 5. PRETTY PRINTER  — clean console output for a live / video demo
# ===========================================================================
def _rule(char="=", width=76):
    print(char * width)


def print_assessment(case: PatientCase, assessment: ClinicalAssessment):
    _rule()
    sex_abbr = "M" if case.sex.lower().startswith("m") else "F"
    print(f" CASE {case.case_id}  |  {case.age}{sex_abbr}  |  {case.chief_complaint}")
    _rule()

    print("\n PATIENT SNAPSHOT")
    print(f"   Symptoms : {', '.join(case.symptoms)}")
    print(f"   History  : {case.history}")
    v = case.vitals
    print(f"   Vitals   : HR {v.heart_rate}  BP {v.systolic_bp}/{v.diastolic_bp}  "
          f"RR {v.respiratory_rate}  Temp {v.temperature_f}F  SpO2 {v.spo2}%")

    print("\n CHAIN-OF-THOUGHT REASONING")
    for s in assessment.reasoning_steps:
        print(f"   {s.step_number}. {s.thought}")

    print("\n DIFFERENTIAL DIAGNOSIS")
    for d in assessment.differential_diagnosis:
        print(f"   [{d.likelihood.upper():8}] {d.condition}")
        print(f"              evidence: {', '.join(d.supporting_evidence)}")

    print(f"\n TOP DIAGNOSIS : {assessment.top_diagnosis}   (confidence: {assessment.confidence})")

    print("\n SUGGESTED ICD-10 CODES  (Payment / coding)")
    for c in assessment.suggested_icd10_codes:
        print(f"   {c.code:8} {c.description}")
        print(f"            -> {c.rationale}")

    print("\n RISK FLAGS / ANOMALIES")
    if assessment.risk_flags:
        for f in assessment.risk_flags:
            print(f"   ({f.severity.upper():8}) {f.signal} — {f.detail}")
            print(f"              action: {f.recommended_action}")
    else:
        print("   None.")

    print("\n RECOMMENDED NEXT STEPS")
    for step in assessment.recommended_next_steps:
        print(f"   - {step}")

    print("\n CLINICAL SUMMARY")
    print(f"   {assessment.clinical_summary}")

    print(f"\n  {assessment.disclaimer}")
    _rule()
    print()


# ===========================================================================
# 6. HARDCODED EXAMPLE CASES  — diverse on purpose (cardiac / pediatric / geriatric)
#    Each one shows the age-specific anomaly detection doing real work.
# ===========================================================================
EXAMPLE_CASES = [
    # Case 1: Adult — classic possible acute coronary syndrome
    PatientCase(
        case_id="001",
        age=58, sex="Male",
        chief_complaint="Crushing chest pain for 40 minutes",
        symptoms=["substernal chest pressure", "pain radiating to left arm",
                  "shortness of breath", "diaphoresis", "nausea"],
        history="Hypertension, type 2 diabetes, 30 pack-year smoking history.",
        vitals=Vitals(heart_rate=112, systolic_bp=158, diastolic_bp=96,
                      respiratory_rate=22, temperature_f=98.6, spo2=94),
    ),
    # Case 2: Pediatric — fever + tachypnea; age-specific ranges matter a lot
    PatientCase(
        case_id="002",
        age=4, sex="Female",
        chief_complaint="High fever and fast breathing for 2 days",
        symptoms=["fever", "lethargy", "rapid breathing", "poor appetite",
                  "decreased urine output"],
        history="No chronic conditions. Up to date on vaccinations.",
        vitals=Vitals(heart_rate=158, systolic_bp=88, diastolic_bp=55,
                      respiratory_rate=38, temperature_f=103.1, spo2=93),
    ),
    # Case 3: Geriatric — acute confusion; UTI/delirium pattern, subtle vitals
    PatientCase(
        case_id="003",
        age=82, sex="Female",
        chief_complaint="New confusion and agitation since this morning",
        symptoms=["acute confusion", "agitation", "urinary frequency",
                  "mild lower abdominal discomfort", "reduced oral intake"],
        history="Mild dementia, osteoporosis, recurrent urinary tract infections.",
        vitals=Vitals(heart_rate=104, systolic_bp=108, diastolic_bp=64,
                      respiratory_rate=20, temperature_f=100.6, spo2=95),
    ),
]


# ===========================================================================
# 7. ENTRYPOINT
# ===========================================================================
def parse_args(argv):
    """Tiny arg parser: optional case number and optional --model."""
    model = DEFAULT_MODEL
    case_index = None
    i = 0
    while i < len(argv):
        if argv[i] == "--model" and i + 1 < len(argv):
            model = argv[i + 1]
            i += 2
        elif argv[i].isdigit():
            case_index = int(argv[i])
            i += 1
        else:
            i += 1
    return case_index, model


def main():
    case_index, model = parse_args(sys.argv[1:])

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: set ANTHROPIC_API_KEY first")
        sys.exit(1)

    # Patch the Anthropic client with instructor so response_model works.
    client = instructor.from_anthropic(Anthropic(api_key=api_key))

    # Choose which cases to run
    if case_index is not None:
        if not (1 <= case_index <= len(EXAMPLE_CASES)):
            print(f"Case must be 1..{len(EXAMPLE_CASES)}")
            sys.exit(1)
        cases = [EXAMPLE_CASES[case_index - 1]]
    else:
        cases = EXAMPLE_CASES

    print(f"\nRunning Agentic Clinical Decision Support POC  (model: {model})\n")
    for case in cases:
        assessment = analyze_case(client, case, model)
        print_assessment(case, assessment)


if __name__ == "__main__":
    main()
