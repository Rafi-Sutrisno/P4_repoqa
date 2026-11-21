import json

import pytest

from repoqa_ctx import tasks as T


@pytest.fixture(scope="session")
def mini_dataset(tmp_path_factory):
    """First repo with needles of every language (6 x 10 = 60 tasks).

    Written as ASCII JSON so the original code can read it with any default encoding.
    """
    full = T.load_dataset()
    mini = {lang: [next(r for r in repos if "needles" in r)] for lang, repos in full.items()}
    path = tmp_path_factory.mktemp("data") / "mini.json"
    path.write_text(json.dumps(mini), encoding="ascii")
    return mini, str(path)


@pytest.fixture(scope="session")
def task_cache(tmp_path_factory):
    """Task cache shared by both tests so contexts are built only once."""
    return str(tmp_path_factory.mktemp("task_cache"))
