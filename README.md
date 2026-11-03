# CALL CENTER AGENT PERFORMANCE ANALYSIS

Analyze public simulated banking calls from raw audio through speaker-separated transcription, summaries, customer sentiment and review. This is a standalone CPU project. It never imports an earlier project or searches for its caches.

## Run in Colab

Open `CALL_CENTER_AGENT_PERFORMANCE_ANALYSIS.ipynb`. Run sections in order. The notebook contains the project files, creates a separate Python environment and downloads a fresh 16-call sample. Store your key in Colab Secrets as `GEMINI_API_KEY` and enable notebook access. A one-call live pilot precedes the full batch. The interface only reads saved results.

## Run locally (Python 3.12 or 3.13)

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python run.py prepare
.venv/bin/python run.py asr
# Set GEMINI_API_KEY in your environment, never in a committed file.
.venv/bin/python run.py analyze --limit 1 --model gemini-3.5-flash-lite
.venv/bin/python run.py analyze --limit 16 --model gemini-3.5-flash-lite
.venv/bin/python app.py
```

On Windows use `.venv\Scripts\python`. An Internet connection is required for initial data/model downloads and API inference. Model availability and free quotas depend on your account. The displayed model name is configurable and is not a guarantee of access. Requests use the `generateContent` REST endpoint with `responseMimeType` and `responseJsonSchema`.

## Dataset and selection

Source: https://github.com/cricketclub/gridspace-stanford-harper-valley

Commit: `0bd721e877c4a85d8c13ff837e68661ea6200a98`.

The pinned repository contains 1,446 calls. These are English **simulated human conversations**, not private bank calls or generated audio. Each has separate agent and caller WAV files. Human and existing machine transcripts are available. Some human annotations are incomplete. The data is not translated. Existing human and machine transcripts are pre-transcribed references; this pipeline generates its own transcripts from audio. Simulated refers to human role-play, not text-to-speech synthesis.

We randomly order metadata with seed 42, then take two single-task calls from each of eight scenarios. Selection depends on metadata only, not model accuracy. One per scenario is designated development and one evaluation. These small subsets are not statistically representative and speaker-disjointness is not guaranteed. No model is trained. Do not tune on evaluation results and then describe that set as untouched.

HVB fits this assessment because it has English call-center audio, known roles, reference transcripts and a clear CC BY 4.0 license. SpokenWOZ is a strong larger alternative; MINDS-14 lacks complete two-party calls; CSDS offers summarization ideas but is Chinese text. `research/Research_Survey.xlsx` compares sources and papers.

## Processing logic

1. Download from the pinned source. Verify Git blob hashes and record SHA-256 hashes.
2. Decode WAV as float32 with SoundFile. Reject empty, nonfinite or nonmono audio. Resample to 16 kHz with SciPy.
3. Run pinned `base.en` via faster-whisper, CPU int8, beam 3, VAD, no previous-text conditioning. Decode **whole speaker channels**, not tiny annotation cuts. Passing a waveform avoids the PyAV file-opening interface involved in the previous `metadata_errors` exception.
4. Keep labeled ASR segments within each speaker channel. Roles come from dataset file paths, not inferred diarization. Cross-speaker chronology is explicitly unverified; do not render channel order as a synchronized conversation.
5. Send only ASR text, role labels and turn IDs to Gemini. Reference words, timing annotations, scenario labels and provided transcripts are excluded from model input. One structured response supplies tasks, summaries, sentiment and limitations.
6. Validate the schema and each citation. Require agent evidence for employee action and resolution, customer evidence for issue and sentiment. Mixed-role context is allowed when agent confirmation is present. Withhold invalid fields individually and preserve original output.
7. Save JSON, JSONL and CSV. Show citations, audio and review reasons in Gradio. Human reviewers check interpretation and fill the separate review sheet.

### Outcome definitions

| Field | Meaning |
|---|---|
| agent_step | What the employee explicitly completed, initiated, failed to do, or left unknown |
| customer_goal | Whether the wider customer goal is completed, pending, failed or unknown |
| resolution | Whether the requested service is resolved, partially resolved, pending, not resolved or unknown |
| next_step | Explicitly stated instruction and responsible party; empty when unstated |
| observed_handoff | Transfer stated in the conversation |
| recommendation | System suggestion, always `executed: false` |

Scheduling a bill payment establishes scheduling, not payment execution. Booking an appointment can satisfy a booking request. Sending a reset link can complete the employee's requested responsibility without proving the password changed. A generic offer does not prove initiation. Unknown is not failure. Proposed conflicting amounts trigger verification, not automatic re-execution. Human specialist review is recommended for an explicit capability limit; the application does not actually transfer a customer.

Sentiment describes expressed customer attitude. It is not CSAT, a personnel score or a measure of whether the bank event was good. The schema supports positive, neutral, negative, mixed and unknown, with reasons and customer evidence.

## Results and validation

The final Colab run reported 16 of 16 calls analyzed, `complete: true` and `error: null`. The table below records the metrics reported from that run. These are execution results, not a human accuracy assessment.

| Subset | New ASR WER | Provided ASR WER | Eligible channels | Reference words |
|---|---:|---:|---:|---:|
| Overall | 19.13% | 8.54% | 31 | 1,709 |
| Development | 17.11% | 5.38% | 16 | 836 |
| Evaluation | 21.08% | 11.57% | 15 | 873 |

All 32 audio channels were transcribed. One channel has an incomplete human reference and is excluded from WER, without substituting machine text. The provided transcription baseline performed better on this sample. Some individual ASR channels exceeded 50% WER and warrant listening checks.

WER = (substitutions + deletions + insertions) / reference words. Both texts use lowercase and punctuation normalization. Numeric words and digits are not unified. WER is not end-to-end system accuracy and can exceed 100% due to insertions.

A separate local development run produced 19.31% overall WER. Files named `local_*` are retained as evidence of that run and must not be presented as the final Colab results. `results/reported_colab_run.json` records the figures reported from Colab; it is not a replacement for the full generated output files. Copy the actual Colab `results` folder into the submission so another reviewer can inspect every call.

Thirteen synthetic regression tests passed locally, covering citations, speaker roles, resampling, caching, request limits and preservation of manual review entries. The interface also returned HTTP 200 locally. These tests verify software behavior, not LLM factual accuracy. The final Colab API batch completed successfully, but independent manual accuracy has not been measured.

For manual evaluation, fill `results/manual_review.csv` after listening to the audio. Enter 0 or 1, reviewer name and notes, then run `evaluate_review.py`. Empty cells remain unscored. Report the reviewed sample size with any accuracy figure.

## Challenges and fixes

| Challenge | Effect | Response and remaining limit |
|---|---|---|
| Colab GPU access unavailable due to usage limits | Restricted model size and experimentation | CPU int8 `base.en`, two inference threads and a balanced 16-call sample. A larger model was not evaluated in this implementation. |
| Limited free API quota and earlier interrupted or rejected requests | Slowed iteration and restricted repeated trials | One-call pilot, paced requests, a local budget and content-based caching. The final 16-call batch completed without a reported API error. |
| NumPy and other package conflicts during earlier development | Import failures in the shared runtime | Install project dependencies in a separate virtual environment. |
| `av.open(..., metadata_errors=...)` incompatibility | Audio decoding failed before ASR | Read WAV with SoundFile, resample with SciPy and pass a float32 waveform to faster-whisper. |
| Fragmented transcription in earlier trials | Lost context and unreliable speech output | Decode each full speaker channel with VAD. This does not remove all ASR errors. |
| Missing human annotations | Incomplete ground truth for WER | Exclude incomplete-reference channels from WER and record the limitation. |
| Cross-speaker timing not verified | Corrections and conversational order can be ambiguous | Keep within-channel order and disclose the limitation instead of inventing a synchronized conversation. |
| Audio did not play through the Colab Gradio interface | Prevented listening from the interface | Section 8.1 embeds WAV bytes directly in the notebook. The exact interface playback cause has not been confirmed. |

Resource limits constrained the available experiments. They do not by themselves explain every transcription or interpretation error. Audio quality, model capacity, missing annotations and evidence ambiguity also affect results. No causal contribution is measured for each factor.

## Reproducibility and quota controls

- Dedicated environment avoids changing the Colab kernel's NumPy/Torch installation. Torch, transformers and librosa are unnecessary.
- `requirements.txt` is the portable dependency specification. `requirements-tested-py312.txt` records the tested environment. `results/environment.json` records your runtime.
- Dataset commit and ASR revision are pinned. Content-based caches include audio hash and ASR settings, or prompt/model/schema/input.
- Successful API results resume from cache. Maximum 20 local requests, paced 15 seconds apart. No automatic retry or silent model substitution. A failed request stays recorded; diagnose it before deliberately retrying.
- A fresh server response may vary despite temperature zero. Saved raw outputs support replay and audit, not deterministic regenerated LLM text.
- The notebook download cell creates a submission ZIP excluding keys, raw audio, virtual environment and caches.

## Diagrams

See [Architecture and sequence](docs/ARCHITECTURE.md) for the full processing flow, cache lookup and validation stages.

## Structure

`src/data.py`: Dataset acquisition and audit. `src/speech.py`: Transcriber and WER. `src/analysis.py`: typed contract, GeminiClient and EvidenceReview. `run.py`: CallCenterPipeline. `app.py`: read-only Gradio UI. `tests/`: synthetic safeguards. `research/`: survey. `docs/`: slides, architecture and recording plan. `results/`: actual fresh ASR validation artifacts and generated analysis.

## Limits and research use

Speech errors propagate into analysis; exact quotes can still be semantically misleading. Deterministic checks establish citation presence and role consistency, not entailment. Channel timing is not synchronized, which can obscure corrections. A small simulated sample cannot establish production readiness or employee rankings. No backend confirms actual bank operations.

FECT motivates claim-level evaluation; no FECT judge is implemented. WhisperX motivates attention to VAD/timing; this code does not implement forced alignment. CSDS motivates role-specific summaries. Google Quality AI motivates explicit assessment criteria; its service is not deployed. Links and limitations are in the survey.

## License and submission

Source data and included transcript-derived validation files are attributed to Gridspace and the Gridspace-Stanford Harper Valley dataset under **CC BY 4.0**: https://creativecommons.org/licenses/by/4.0/ . Changes include resampling, ASR transcription, normalization and analysis. The original data license is in `research/DATA_LICENSE.txt`. Download raw audio from source rather than committing it. This repository uses no private recordings.

Before submission: include the actual Colab results, review samples if possible, record the 10-minute presentation, insert the recording link in `docs/RECORDING.md`, and upload to GitHub. The recording and GitHub publication are not completed by this package. The supplied slides describe the earlier local run (19.31% WER). Update the results and validation slides with the final Colab figures above before recording. No recording is included.

## Repository checks

Keep API keys in Colab Secrets or environment variables. Do not commit `.env`, credentials, raw recordings, the virtual environment or request caches. Only public licensed sample data is used. CC BY 4.0 permits sharing and adaptation with attribution, a license link and an indication of changes; no additional restrictions should be imposed on the licensed data. Preserve `research/DATA_LICENSE.txt` when redistributing derived transcripts.

The notebook export excludes runtime folders and caches. Before publishing, inspect the repository and notebook outputs for credentials and include the notebook, actual generated results, research workbook, README, dependency files, diagrams, slides and recording link.
