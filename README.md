# Clinical Documentation Assistant — RAG Demo

A small Streamlit demonstration of retrieval-augmented generation (RAG) for clinical documentation review. It uses **fictional, synthetic patient encounters only**.

## Features
- Select one of three synthetic clinical encounters.
- Retrieve relevant chart excerpts using TF-IDF similarity.
- Ask a documentation question grounded in the retrieved excerpts.
- Generate a concise review highlighting documented facts, possible gaps, and items requiring clinician verification.
- Display the evidence snippets used to answer.

> **Safety:** This is an educational prototype, not a medical device or coding authority. It must not be used for real patient care, diagnosis, billing, or autonomous clinical decisions. All outputs require qualified human review. Do not enter real patient information.

## Run locally

1. Create and activate a virtual environment (recommended).
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set your API key as an environment variable:

   **macOS / Linux**
   ```bash
   export OPENAI_API_KEY="your-api-key"
   ```

   **Windows PowerShell**
   ```powershell
   $env:OPENAI_API_KEY="your-api-key"
   ```

4. Start the app:

   ```bash
   streamlit run app.py
   ```

The app uses the OpenAI API with `gpt-4o-mini`. API usage may incur charges.

## Streamlit Community Cloud

When deploying, add `OPENAI_API_KEY` under **App settings → Secrets**. Never commit API keys or paste them into source files.

## Architecture

1. **Synthetic encounter data** — fictional notes stored in the app.
2. **Retrieval** — TF-IDF vectorization and cosine similarity select relevant excerpts.
3. **Generation** — the LLM receives only the selected excerpts and the user's question.
4. **Human review** — the UI clearly labels gaps and uncertainty for clinician verification.

## Project structure

- `app.py` — Streamlit interface, retrieval, and LLM call.
- `requirements.txt` — Python dependencies.
- `.gitignore` — excludes secrets, local environments, and caches.

## Example questions
- What documentation supports the assessment?
- What important details are missing from the encounter?
- Summarize the available evidence for the plan.

## License
MIT
