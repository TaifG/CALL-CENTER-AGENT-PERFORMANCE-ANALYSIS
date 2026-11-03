"""Read-only interface: browsing results never consumes API quota."""

import json
from pathlib import Path
import gradio as gr

ROOT = Path(__file__).parent


def load_results():
    file = ROOT / "results/analysis.json"
    if not file.exists():
        return []
    return json.loads(file.read_text())


def build_app():
    rows = load_results()
    by_id = {r["call_id"]: r for r in rows}

    def show(cid):
        r = by_id.get(cid)
        if not r:
            return "Run the analysis stage first.", {}, "", None, None
        text = "\n\n".join(
            f"**{t['turn_id']} — {t['speaker']}**: {t['text']}" for t in r["transcript"]
        )
        audio = [
            ROOT / "data_source/data/audio" / role / (cid + ".wav")
            for role in ["agent", "caller"]
        ]
        details = {
            k: r[k]
            for k in [
                "sentiment",
                "tasks",
                "recommendations",
                "needs_review",
                "review_reasons",
                "field_checks",
                "notes",
            ]
        }
        return (
            r["summary"],
            details,
            text,
            *[str(p) if p.exists() else None for p in audio],
        )

    with gr.Blocks(title="Evidence-based call review") as demo:
        gr.Markdown(
            "# Call review\nASR → structured claims → evidence checks → human review.\nSpeaker channels are known; cross-speaker chronology is **not verified**. This tool does not execute transfers or customer actions."
        )
        choice = gr.Dropdown(
            list(by_id), label="Call ID", value=next(iter(by_id), None)
        )
        summary = gr.Markdown()
        detail = gr.JSON(label="Validated fields, citations and review reasons")
        with gr.Row():
            agent = gr.Audio(label="Agent channel")
            customer = gr.Audio(label="Customer channel")
        transcript = gr.Markdown()
        choice.change(show, choice, [summary, detail, transcript, agent, customer])
        demo.load(show, choice, [summary, detail, transcript, agent, customer])
    return demo


if __name__ == "__main__":
    build_app().launch(server_name="127.0.0.1", server_port=7860)
