"""Run each stage explicitly. No import-time downloads or API requests."""

import argparse, csv, os, sys, platform, importlib.metadata
from pathlib import Path
from src.data import Dataset, save, read
from src.speech import Transcriber, metric_summary
from src.analysis import GeminiClient, EvidenceReview


class CallCenterPipeline:
    def __init__(self, root):
        self.root = Path(root)
        self.output = self.root / "results"
        self.output.mkdir(parents=True, exist_ok=True)
        self.dataset = Dataset(self.root / "data_source")

    def prepare(self):
        manifest = self.dataset.prepare()
        save(self.output / "audit.json", self.dataset.audit(manifest["selection"]))
        print(
            "Prepared", len(manifest["selection"]), "calls; source checksums verified."
        )

    def transcribe(self):
        manifest = self.dataset.prepare()
        asr = Transcriber(self.root / "cache/asr")
        calls = []
        metrics = []
        for item in manifest["selection"]:
            call, rows = asr.call(self.dataset, item)
            calls.append(call)
            metrics.extend(rows)
            save(self.output / "asr_calls.json", calls)
            save(self.output / "wer_channels.json", metrics)
        report = {
            "overall": metric_summary(metrics),
            "by_split": {
                s: metric_summary([r for r in metrics if r["split"] == s])
                for s in ["development", "evaluation"]
            },
        }
        save(self.output / "wer_summary.json", report)
        print(report)

    def analyze(self, limit, model):
        calls = read(self.output / "asr_calls.json")[:limit]
        client = GeminiClient(
            os.getenv("GEMINI_API_KEY", ""), self.root / "cache/api", model=model
        )
        reviewer = EvidenceReview()
        results = []
        error = None
        # Save each completed call before moving to the next request.
        for call in calls:
            print("ANALYZE", call["call_id"], flush=True)
            try:
                raw, provenance = client.analyze(call)
                result = reviewer.check(call, raw)
                result["api_provenance"] = provenance
                results.append(result)
                save(self.output / "analysis.json", results)
            except Exception as exc:
                error = str(exc)
                print(error)
                break
        status = {
            "requested": len(calls),
            "completed": len(results),
            "complete": len(results) == len(calls),
            "error": error,
            "human_accuracy": "Not measured; independent review required.",
        }
        save(self.output / "api_status.json", status)
        if results:
            self.export(results)
        print(status)
        if error:
            raise SystemExit(2)

    def export(self, results):
        (self.output / "analysis.jsonl").write_text(
            "\n".join(__import__("json").dumps(r, ensure_ascii=False) for r in results)
            + "\n"
        )
        with (self.output / "calls.csv").open("w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "call_id",
                    "summary",
                    "sentiment",
                    "needs_review",
                    "review_reasons",
                ],
            )
            w.writeheader()
            for r in results:
                w.writerow(
                    {
                        k: (r["sentiment"]["label"] if k == "sentiment" else r[k])
                        for k in w.fieldnames
                    }
                )
        # Never overwrite reviewer annotations when resuming.
        manual = self.output / "manual_review.csv"
        # Preserve human annotations when the pilot expands to a full batch.
        columns = [
            "call_id",
            "summary_correct_0_1",
            "outcome_correct_0_1",
            "sentiment_correct_0_1",
            "evidence_correct_0_1",
            "reviewer",
            "notes",
        ]
        existing = (
            list(csv.DictReader(manual.read_text().splitlines()))
            if manual.exists()
            else []
        )
        known = {r["call_id"] for r in existing}
        for result in results:
            if result["call_id"] not in known:
                existing.append(
                    {k: result["call_id"] if k == "call_id" else "" for k in columns}
                )
        with manual.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(existing)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["prepare", "asr", "analyze"])
    p.add_argument("--root", default=str(Path(__file__).parent))
    p.add_argument("--limit", type=int, default=1)
    p.add_argument("--model", default="gemini-3.5-flash-lite")
    a = p.parse_args()
    flow = CallCenterPipeline(a.root)
    versions = {
        name: importlib.metadata.version(name)
        for name in [
            "numpy",
            "scipy",
            "soundfile",
            "faster-whisper",
            "ctranslate2",
            "pydantic",
            "jiwer",
        ]
    }
    save(
        flow.output / "environment.json",
        {"python": sys.version, "platform": platform.platform(), "packages": versions},
    )
    if a.stage == "prepare":
        flow.prepare()
    elif a.stage == "asr":
        flow.transcribe()
    else:
        flow.analyze(a.limit, a.model)


if __name__ == "__main__":
    main()
