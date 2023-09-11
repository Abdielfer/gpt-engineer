import hashlib

from typing import List

import steps
from db import DBs
from domain import Step
from learning import Learning, extract_learning


def send_learning(learning: Learning):
    import rudderstack.analytics as rudder_analytics

    rudder_analytics.write_key = "2Re4kqwL61GDp7S8ewe6K5dbogG"
    rudder_analytics.dataPlaneUrl = "https://gptengineerezm.dataplane.rudderstack.com"

    rudder_analytics.track(
        user_id=learning.session,
        event="learning",
        properties=learning.to_dict(),  # type: ignore
    )


def collect_learnings(model: str, temperature: float, steps: List[Step], dbs: DBs):
    learnings = extract_learning(
        model, temperature, steps, dbs, steps_file_hash=steps_file_hash()
    )
    send_learning(learnings)


def steps_file_hash():
    with open(steps.__file__, "r") as f:
        content = f.read()
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
