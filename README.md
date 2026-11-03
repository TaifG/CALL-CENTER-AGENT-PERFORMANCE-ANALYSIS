# Call Center Agent Performance Analysis

A standalone, CPU-based pipeline for processing simulated banking calls through speaker-separated transcription, Gemini-powered analysis, evidence validation, and a Gradio review interface.

## Project Overview

This project processes raw audio from simulated call center conversations, separates agent and caller channels, and uses Gemini to extract structured summaries, customer sentiment, and agent steps while strictly validating claims against transcript evidence.

## How to Run

### 1. Run in Google Colab
* Open `CALL_CENTER_AGENT_PERFORMANCE_ANALYSIS.ipynb`.
* Store your API key in Colab Secrets as `GEMINI_API_KEY`.
* Run the cells sequentially to download a 16-call sample, run the analysis, and launch the Gradio interface.

### 2. Run Locally (Python 3.12 or 3.13)
```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python run.py prepare
.venv/bin/python run.py asr
# Set GEMINI_API_KEY in your environment variables
.venv/bin/python run.py analyze --limit 16 --model gemini-3.5-flash-lite
.venv/bin/python app.py
```

## Key Challenges & Fixes

* **API & GPU Limits:** Handled by using a lightweight CPU model (`base.en` with int8) and content-based request caching to ensure smooth execution.
* **Audio Decoding Issues:** Resolved by decoding WAV files directly via `SoundFile` and resampling with `SciPy` to bypass system library incompatibilities.
* **Incomplete Reference Data:** Handled by excluding incomplete human reference channels from WER calculations and documenting the limitation transparently.

## Repository Structure

* **`src/`:** Core logic (data acquisition, ASR transcription, and Gemini analysis).
* **`research/`:** Research surveys and data licensing details.
* **`tests/`:** Automated regression and safety safeguards.
* **`app.py`:** Read-only Gradio UI for browsing results and audio playback.
* **`run.py`:** Pipeline execution script.

## License & Attribution

Source data and derived validation files are attributed to Gridspace and the Gridspace-Stanford Harper Valley dataset under **CC BY 4.0**: [https://creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/).

* **Raw audio:** Should be downloaded from the source rather than committed directly.
* **Original license:** Details are available in `research/DATA_LICENSE.txt`.
