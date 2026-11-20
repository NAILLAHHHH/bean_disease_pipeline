"""
Flood-of-requests simulation for the Bean Leaf Disease API.

Interactive UI:
    locust -f locust/locustfile.py --host http://localhost:8080

Headless run (what run_load_tests.sh uses):
    locust -f locust/locustfile.py --host http://localhost:8080 \
           --headless -u 100 -r 20 -t 60s --csv locust/results/c1
"""
import random
from pathlib import Path

from locust import HttpUser, between, task

TEST_DIR = Path(__file__).resolve().parents[1] / "data" / "test"
IMAGES = [p.read_bytes() for p in sorted(TEST_DIR.rglob("*.jpg"))[::4]]  # ~65 real leaf photos


class FarmerUser(HttpUser):
    """A user of the app: mostly sends leaf photos, sometimes checks health."""
    wait_time = between(0.5, 1.5)

    @task(10)
    def predict(self):
        img = random.choice(IMAGES)
        with self.client.post("/predict", files={"file": ("leaf.jpg", img, "image/jpeg")},
                              name="/predict", catch_response=True) as r:
            if r.status_code != 200 or "label" not in r.text:
                r.failure(f"bad response {r.status_code}")

    @task(1)
    def health(self):
        self.client.get("/health", name="/health")
