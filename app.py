import os
from typing import List, Dict

import streamlit as st
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Clinical Documentation Assistant", page_icon="🩺", layout="wide")

# All records below are fictional and created solely for this demonstration.
ENCOUNTERS: Dict[str, Dict[str, object]] = {
    "ENC-001 — Diabetes follow-up": {
        "patient": "Synthetic patient A",
        "visit": "Outpatient follow-up",
        "chunks": [
            "Chief concern: Follow-up for type 2 diabetes. Patient reports increased thirst over the last two weeks.",
            "History: Type 2 diabetes is listed in the problem list. Current medication list includes metformin. Dose and adherence were not recorded in this note.",
            "Vitals: Blood pressure 128/82 mmHg, pulse 78 beats per minute, temperature 36.8 C. No height, weight, or BMI documented.",
            "Assessment: Type 2 diabetes mellitus. The note does not include a current HbA1c value or a documented assessment of glycemic control.",
            "Plan: Continue current management and obtain laboratory testing. Specific follow-up interval and medication changes are not documented."
        ]
    },
    "ENC-002 — Cough consultation": {
        "patient": "Synthetic patient B",
        "visit": "Primary care consultation",
        "chunks": [
            "Chief concern: Cough for five days, described as dry and worse at night.",
            "History: Patient reports no known sick contacts. Smoking history, relevant exposure history, and complete associated symptoms are not documented.",
            "Vitals: Temperature 37.2 C, pulse 88 beats per minute, oxygen saturation 97 percent. Respiratory rate and blood pressure are not recorded.",
            "Examination: Lungs documented as clear to auscultation. Work of breathing and other respiratory examination details are not described.",
            "Assessment: Acute cough. The note does not document a differential diagnosis or clinical reasoning.",
            "Plan: Supportive care discussed. Return precautions and a specific follow-up timeframe are not documented."
        ]
    },
    "ENC-003 — Knee pain visit": {
        "patient": "Synthetic patient C",
        "visit": "Musculoskeletal consultation",
        "chunks": [
            "Chief concern: Right knee pain for approximately three weeks after increasing running distance.",
            "History: Pain is worse with stairs. Severity score, swelling, locking, instability, prior injury, and functional limitations are not documented.",
            "Vitals: Blood pressure 122/78 mmHg and pulse 72 beats per minute. Other vital signs are not recorded.",
            "Examination: Mild tenderness around the anterior right knee. Range of motion, gait, stability testing, and comparison with the left knee are not documented.",
            "Assessment: Right knee pain, possibly related to overuse. The supporting clinical reasoning is brief.",
            "Plan: Activity modification recommended. Specific exercise guidance, medication details, imaging considerations, and follow-up interval are not documented."
        ]
    }
}


def get_api_key() -> str:
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        try:
            key = st.secrets.get("OPENAI_API_KEY", "")
        except Exception:
            key = ""
    return key


def retrieve(question: str, chunks: List[str], top_k: int = 4):
    corpus = chunks + [question]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(corpus)
    scores = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
    ranked = scores.argsort()[::-1][: min(top_k, len(chunks))]
    return [(chunks[i], float(scores[i])) for i in ranked if scores[i] > 0]


def main():
    st.title("🩺 Clinical Documentation Assistant")
    st.caption("A retrieval-augmented generation (RAG) prototype using synthetic encounters")

    st.warning(
        "Educational demo only. Records are fictional. Do not enter real patient information. "
        "Outputs are not medical advice, coding advice, or a substitute for qualified clinical review."
    )

    with st.sidebar:
        st.header("Encounter")
        selected = st.selectbox("Choose a synthetic encounter", list(ENCOUNTERS.keys()))
        st.divider()
        st.markdown("**Demo workflow**")
        st.markdown("1. Select an encounter\n2. Ask a documentation question\n3. Review retrieved evidence\n4. Verify all suggestions")

    encounter = ENCOUNTERS[selected]
    chunks = encounter["chunks"]

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Encounter overview")
        st.write(f"**Patient:** {encounter['patient']}")
        st.write(f"**Visit type:** {encounter['visit']}")
        st.markdown("**Available chart excerpts**")
        for i, chunk in enumerate(chunks, start=1):
            with st.expander(f"Chart excerpt {i}"):
                st.write(chunk)

    with right:
        st.subheader("Ask the assistant")
        question = st.text_area(
            "Documentation review question",
            value="Identify potentially missing documentation and explain what should be verified.",
            height=120,
        )
        run = st.button("Review documentation", type="primary", use_container_width=True)

        if run:
            if not question.strip():
                st.error("Please enter a question.")
                return

            key = get_api_key()
            if not key:
                st.error("API key not found. Set OPENAI_API_KEY in your environment or Streamlit Secrets.")
                st.info("For local use, set the OPENAI_API_KEY environment variable before starting Streamlit.")
                return

            retrieved = retrieve(question, chunks)
            if not retrieved:
                st.warning("No sufficiently relevant excerpts were retrieved. Try a more specific question.")
                return

            evidence = "\n".join(f"- {text}" for text, _ in retrieved)
            prompt = f"""You are assisting with a fictional clinical documentation quality review.
Use ONLY the encounter excerpts below. Do not invent facts, diagnoses, medications, codes, or results.
Distinguish clearly between:
1. Facts explicitly documented
2. Potential documentation gaps (say 'not documented in the supplied excerpts', not that something did not happen)
3. Questions a qualified clinician should verify

Do not make treatment recommendations or assign billing codes. Be concise and evidence-grounded.
If the evidence is insufficient, say so.

ENCOUNTER EXCERPTS:
{evidence}

USER QUESTION:
{question}
"""
            try:
                client = OpenAI(api_key=key)
                with st.spinner("Reviewing retrieved evidence..."):
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        temperature=0.2,
                        messages=[
                            {"role": "system", "content": "You are a cautious clinical documentation review assistant for a synthetic educational demo."},
                            {"role": "user", "content": prompt},
                        ],
                    )
                st.markdown("### Assistant review")
                st.markdown(response.choices[0].message.content or "No response was returned.")
                st.markdown("### Retrieved evidence")
                for text, score in retrieved:
                    st.markdown(f"- {text}")
                    st.caption(f"TF-IDF similarity: {score:.3f}")
            except Exception as exc:
                st.error(f"The API request failed: {exc}")
                st.caption("Check that your API key is valid and that API billing/usage is enabled.")

    st.divider()
    st.caption("Prototype architecture: synthetic notes → TF-IDF retrieval → LLM response → human verification.")


if __name__ == "__main__":
    main()
