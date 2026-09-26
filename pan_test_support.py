"""Small independent fixtures for offline PAN contract tests."""
from pathlib import Path
from types import SimpleNamespace
import dental_pipeline as dp


class Runner:
    def __init__(self, responses=None):
        self.responses = responses or {}
        self.requests = []

    def settings(self):
        return {"model": "offline-fixture", "runtime_provenance": {"revision": "fixture"}}

    def ask(self, image, question):
        self.requests.append((str(image), question))
        task = next(k for k, t in dp.TASKS.items() if t["questions"][0] == question)
        value = self.responses.get(task, "No\nNo finding detected.")
        if isinstance(value, Exception):
            raise value
        return {"text": value, "finish_reason": "stop", "truncated": False} if isinstance(value, str) else value


def result(tmp_path, responses=None):
    image = Path(tmp_path) / "image.png"
    image.write_bytes(b"fixture image bytes")
    value = dp.analyze_image(Runner(responses), image)
    value["image_id"] = "image"
    return value


class Client:
    def __init__(self, replies=None):
        self.requests = []
        self.replies = list(replies or [])
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **request):
        self.requests.append(request)
        reply = self.replies.pop(0) if self.replies else "No"
        if isinstance(reply, Exception):
            raise reply
        text, finish = reply if isinstance(reply, tuple) else (reply, "stop")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=finish)],
                               usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2))


def report(structured):
    return {"language": "English", "sections": [
        {"category": c["key"], "findings": [
            {"finding": f["finding"], "status": f["status"], "regions": f["regions"]}
            for f in structured["findings"] if f["category"] == c["key"]]}
        for c in structured["categories"]], "impression": list(structured["summary"]["present"])}
