"""Synthetic boundary cases. These test safeguards, not model accuracy."""

import unittest, tempfile, json, io, copy
from pathlib import Path
from unittest.mock import patch
import numpy as np
import soundfile as sf
from src.analysis import EvidenceReview, GeminiClient, Analysis
from src.speech import audio_array


def fixture():
    call = {
        "call_id": "synthetic",
        "source": "synthetic_test",
        "chronology": "channel_order_only",
        "turns": [
            {
                "turn_id": "C001",
                "speaker": "Customer",
                "text": "Please book an appointment on Thursday at one thirty.",
            },
            {
                "turn_id": "A001",
                "speaker": "Agent",
                "text": "That is done. Your appointment is booked.",
            },
        ],
    }
    c = {
        "turn_id": "C001",
        "quote": "Please book an appointment on Thursday at one thirty.",
    }
    a = {"turn_id": "A001", "quote": "Your appointment is booked."}
    empty = {"text": "", "evidence": []}
    raw = {
        "tasks": [
            {
                "issue": {"text": "Book an appointment", "evidence": [c]},
                "agent_action": {"text": "Booked the appointment", "evidence": [a]},
                "agent_step": {
                    "value": "completed",
                    "reason": "Explicit confirmation",
                    "evidence": [a],
                },
                "customer_goal": {
                    "value": "completed",
                    "reason": "Booking confirmed",
                    "evidence": [c, a],
                },
                "resolution": "resolved",
                "resolution_reason": "Requested booking confirmed",
                "resolution_evidence": [c, a],
                "next_step": {"owner": "Unknown", "instruction": empty},
                "discrepancy": empty,
                "capability_limit": empty,
                "observed_handoff": empty,
            }
        ],
        "sentiment": {"label": "neutral", "reason": "Routine request", "evidence": [c]},
        "limitations": [],
    }
    return call, copy.deepcopy(raw)


class LogicTests(unittest.TestCase):
    def test_mixed_speaker_contextual_confirmation(self):
        c, r = fixture()
        out = EvidenceReview().check(c, r)
        self.assertEqual(out["tasks"][0]["resolution"], "resolved")
        self.assertFalse(out["needs_review"])

    def test_customer_alone_cannot_prove_agent_completion(self):
        c, r = fixture()
        r["tasks"][0]["resolution_evidence"] = r["tasks"][0]["issue"]["evidence"]
        out = EvidenceReview().check(c, r)
        self.assertEqual(out["tasks"][0]["resolution"], "unknown")
        self.assertEqual(out["sentiment"]["label"], "neutral")

    def test_invented_quote_masked_without_erasing_other_fields(self):
        c, r = fixture()
        r["tasks"][0]["agent_action"]["evidence"][0]["quote"] = "Paid 100 dollars"
        out = EvidenceReview().check(c, r)
        self.assertEqual(out["tasks"][0]["agent_action"]["text"], "")
        self.assertTrue(out["tasks"][0]["issue"]["text"])
        self.assertEqual(out["original_model_output"], r)

    def test_request_is_not_pending_goal(self):
        c, r = fixture()
        r["tasks"][0]["customer_goal"] = {
            "value": "pending",
            "reason": "Requested",
            "evidence": r["tasks"][0]["issue"]["evidence"],
        }
        self.assertEqual(
            EvidenceReview().check(c, r)["tasks"][0]["customer_goal"]["value"],
            "unknown",
        )

    def test_sentiment_requires_customer(self):
        c, r = fixture()
        r["sentiment"]["evidence"] = r["tasks"][0]["agent_action"]["evidence"]
        self.assertEqual(EvidenceReview().check(c, r)["sentiment"]["label"], "unknown")

    def test_handoff_recommendation_not_observed(self):
        c, r = fixture()
        c["turns"].append(
            {
                "turn_id": "A002",
                "speaker": "Agent",
                "text": "I cannot access this account.",
            }
        )
        r["tasks"][0]["capability_limit"] = {
            "text": "No account access",
            "evidence": [{"turn_id": "A002", "quote": "I cannot access this account."}],
        }
        out = EvidenceReview().check(c, r)
        self.assertFalse(out["recommendations"][0]["executed"])
        self.assertEqual(out["tasks"][0]["observed_handoff"]["text"], "")

    def test_discrepancy_needs_both_sides(self):
        c, r = fixture()
        r["tasks"][0]["discrepancy"] = {
            "text": "Wrong time",
            "evidence": r["tasks"][0]["issue"]["evidence"],
        }
        out = EvidenceReview().check(c, r)
        self.assertEqual(out["tasks"][0]["discrepancy"]["text"], "")
        self.assertTrue(out["needs_review"])

    def test_schema_rejects_extra_fields(self):
        c, r = fixture()
        r["secret_guess"] = "123"
        with self.assertRaises(ValueError):
            Analysis.model_validate(r)

    def test_audio_resampled_with_correct_duration(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "test.wav"
            sf.write(p, np.zeros(8000, dtype=np.float32), 8000)
            wave = audio_array(p)
            self.assertEqual(len(wave), 16000)
            self.assertEqual(wave.dtype, np.float32)

    def test_api_success_cached_and_reference_not_sent(self):
        c, r = fixture()
        c["reference"] = "NEVER SEND ME"
        response = {
            "candidates": [
                {
                    "finishReason": "STOP",
                    "content": {"parts": [{"text": json.dumps(r)}]},
                }
            ]
        }
        with tempfile.TemporaryDirectory() as d:
            client = GeminiClient("test-secret", d, interval=0)
            with patch(
                "urllib.request.urlopen", return_value=io.StringIO(json.dumps(response))
            ) as http:
                client.analyze(c)
                _, provenance = client.analyze(c)
                self.assertEqual(http.call_count, 1)
                self.assertTrue(provenance["cache_hit"])
                self.assertNotIn("NEVER SEND ME", http.call_args.args[0].data.decode())

    def test_api_failure_does_not_retry(self):
        c, r = fixture()
        with tempfile.TemporaryDirectory() as d:
            client = GeminiClient("test-secret", d, interval=0)
            with patch(
                "urllib.request.urlopen", side_effect=TimeoutError("timeout")
            ) as http:
                for _ in range(2):
                    with self.assertRaises(RuntimeError):
                        client.analyze(c)
                self.assertEqual(http.call_count, 1)

    def test_api_budget_zero_sends_nothing(self):
        c, r = fixture()
        with tempfile.TemporaryDirectory() as d:
            with patch("urllib.request.urlopen") as http:
                with self.assertRaises(RuntimeError):
                    GeminiClient("key", d, budget=0).analyze(c)
                http.assert_not_called()


if __name__ == "__main__":
    unittest.main()


class ExportTests(unittest.TestCase):
    def test_pilot_review_preserved_when_batch_expands(self):
        import csv
        from run import CallCenterPipeline

        c, r = fixture()
        first = EvidenceReview().check(c, r)
        with tempfile.TemporaryDirectory() as d:
            p = CallCenterPipeline(d)
            p.export([first])
            file = p.output / "manual_review.csv"
            rows = list(csv.DictReader(file.read_text().splitlines()))
            rows[0]["reviewer"] = "tester"
            rows[0]["summary_correct_0_1"] = "1"
            with file.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            second = copy.deepcopy(first)
            second["call_id"] = "second"
            p.export([first, second])
            rows = list(csv.DictReader(file.read_text().splitlines()))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["reviewer"], "tester")
