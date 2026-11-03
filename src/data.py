"""Pinned public data acquisition; never reads an older project or its caches."""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib, json, urllib.request, re, random

COMMIT = "0bd721e877c4a85d8c13ff837e68661ea6200a98"
BASE = f"https://raw.githubusercontent.com/cricketclub/gridspace-stanford-harper-valley/{COMMIT}/"


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    tmp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(text):
    return re.sub(
        r"\s+", " ", re.sub(r"\[[^\]]*\]|<[^>]*>", " ", str(text or ""))
    ).strip()


def fetch(relative, destination, expected_blob=None):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        data = destination.read_bytes()
    else:
        with urllib.request.urlopen(BASE + relative, timeout=90) as response:
            data = response.read()
        destination.write_bytes(data)
    if expected_blob:
        got = hashlib.sha1(
            b"blob " + str(len(data)).encode() + b"\0" + data
        ).hexdigest()
        if got != expected_blob:
            raise ValueError("Source checksum mismatch: " + relative)
    return destination


class Dataset:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def prepare(self):
        # Verify saved source files before reusing this sample.
        manifest = self.root / "selection.json"
        if manifest.exists():
            existing = read(manifest)
            if existing["commit"] != COMMIT:
                raise ValueError("Dataset commit changed")
            for relative, record in existing["files"].items():
                file = self.root / relative
                if not file.exists() or sha(file) != record["sha256"]:
                    raise ValueError("Missing or changed source file: " + relative)
            return existing
        tree_path = self.root / "source_tree.json"
        if tree_path.exists():
            tree = read(tree_path)
        else:
            url = f"https://api.github.com/repos/cricketclub/gridspace-stanford-harper-valley/git/trees/{COMMIT}?recursive=1"
            with urllib.request.urlopen(url, timeout=90) as r:
                tree = json.load(r)
            save(tree_path, tree)
        if tree.get("truncated"):
            raise ValueError("Incomplete repository tree")
        blobs = {r["path"]: r["sha"] for r in tree["tree"] if r["type"] == "blob"}
        ids = sorted(
            Path(p).stem
            for p in blobs
            if p.startswith("data/metadata/") and p.endswith(".json")
        )
        random.Random(42).shuffle(ids)
        groups = {}
        inspected = 0
        # Sample by scenario using metadata only, not transcript quality or model results.
        for start in range(0, len(ids), 32):
            batch = ids[start : start + 32]

            def meta(cid):
                relative = f"data/metadata/{cid}.json"
                return cid, read(fetch(relative, self.root / relative, blobs[relative]))

            with ThreadPoolExecutor(max_workers=8) as pool:
                rows = list(pool.map(meta, batch))
            for cid, m in rows:
                inspected += 1
                tasks = m.get("tasks", [])
                if len(tasks) == 1:
                    scenario = tasks[0].get("task_type", "unknown")
                    if (
                        scenario != "unknown"
                        and len(groups.setdefault(scenario, [])) < 2
                    ):
                        groups[scenario].append(cid)
            if len(groups) == 8 and all(len(g) == 2 for g in groups.values()):
                break
        if len(groups) != 8 or any(len(g) != 2 for g in groups.values()):
            raise ValueError("Could not form scenario-balanced sample")
        selected = [
            {
                "call_id": cid,
                "scenario": scenario,
                "split": "development" if i == 0 else "evaluation",
            }
            for scenario, g in sorted(groups.items())
            for i, cid in enumerate(g)
        ]
        # Download both speaker channels and the reference transcript.
        jobs = []
        for item in selected:
            cid = item["call_id"]
            for rel in [
                f"data/transcript/{cid}.json",
                f"data/audio/agent/{cid}.wav",
                f"data/audio/caller/{cid}.wav",
            ]:
                jobs.append(rel)
        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(lambda rel: fetch(rel, self.root / rel, blobs[rel]), jobs))
        fetch("LICENSE", self.root / "LICENSE", blobs.get("LICENSE"))
        result = {
            "commit": COMMIT,
            "population_calls": len(ids),
            "metadata_inspected": inspected,
            "seed": 42,
            "selection": selected,
            "sampling": "first two per scenario in seeded random metadata order; first development, second evaluation",
            "files": {
                rel: {"git_blob": blobs[rel], "sha256": sha(self.root / rel)}
                for rel in jobs
            },
        }
        save(manifest, result)
        return result

    def audio(self, cid, role):
        return self.root / "data/audio" / role / (cid + ".wav")

    def rows(self, cid):
        return read(self.root / "data/transcript" / (cid + ".json"))

    def references(self, cid, role):
        rows = sorted(
            [r for r in self.rows(cid) if r["speaker_role"] == role],
            key=lambda r: r.get("offset_ms", 0),
        )
        complete = all(str(r.get("human_transcript") or "").strip() for r in rows)
        return {
            "human": " ".join(clean(r.get("human_transcript")) for r in rows),
            "provided_asr": " ".join(clean(r.get("transcript")) for r in rows),
            "human_complete": complete,
        }

    def audit(self, selection):
        import soundfile as sf

        items = []
        for call in selection:
            for role in ["agent", "caller"]:
                audio, rate = sf.read(
                    self.audio(call["call_id"], role), dtype="float32"
                )
                ref = self.references(call["call_id"], role)
                items.append(
                    {
                        "call_id": call["call_id"],
                        "role": role,
                        "sample_rate": rate,
                        "channels": 1 if audio.ndim == 1 else audio.shape[1],
                        "seconds": len(audio) / rate,
                        "samples": len(audio),
                        "human_complete": ref["human_complete"],
                        "sha256": sha(self.audio(call["call_id"], role)),
                    }
                )
        return items
