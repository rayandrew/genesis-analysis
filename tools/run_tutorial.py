"""Run the Python blocks of a tutorial and check their output.

Each ```python block runs in one shared namespace, as the reader runs it in one
Python session. The ```text block right after it is the output the page
promises. A block that raises is checked by the last line of the error, since
the traceback paths differ on each machine.

    uv run python tools/run_tutorial.py tutorials/01-count-the-runs.md
    uv run python tools/run_tutorial.py --record tutorials/01-count-the-runs.md

--record writes the real output into the text blocks instead of checking it.
"""

import argparse
import contextlib
import io
import os
import re
import sys
import time
import traceback
from pathlib import Path

FENCE = re.compile(r"^```(\w*)\n(.*?)^```\n", re.DOTALL | re.MULTILINE)


def blocks(page):
    return [
        (m.group(1), m.group(2), m.start(2), m.end(2)) for m in FENCE.finditer(page)
    ]


def run(code, namespace):
    out = io.StringIO()
    error = None
    with contextlib.redirect_stdout(out):
        try:
            exec(compile(code, "<stdin>", "exec"), namespace)  # noqa: S102  # runs the page's own code
        except Exception as e:  # noqa: BLE001  # the page records any error a block raises
            error = e
    text = out.getvalue()
    if error is not None:
        lines = traceback.format_exception_only(type(error), error)
        text += "Traceback (most recent call last):\n  ...\n" + "".join(lines)
    return text, error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", action="store_true")
    parser.add_argument("pages", nargs="+")
    args = parser.parse_args()
    os.environ.setdefault("DFTRACER_UTILS_LOG_LEVEL", "warn")
    failed = 0
    for name in args.pages:
        path = Path(name)
        page = path.read_text()
        found = blocks(page)
        namespace = {"__name__": "__main__"}
        edits = []
        start = time.time()
        for i, (lang, code, _, _) in enumerate(found):
            if lang != "python":
                continue
            got, error = run(code, namespace)
            nxt = found[i + 1] if i + 1 < len(found) else None
            if nxt is None or nxt[0] != "text":
                if error is not None or got:
                    print(f"{path}: block {i} has output but no text block after it")
                    failed += 1
                continue
            if args.record:
                edits.append((nxt[2], nxt[3], got))
                continue
            want = nxt[1]
            ok = (
                want.splitlines()[-1:] == got.splitlines()[-1:]
                if error
                else want == got
            )
            if not ok:
                failed += 1
                print(
                    f"{path}: block {i} output differs\n--- want\n{want}--- got\n{got}"
                )
        for begin, end, text in reversed(edits):
            page = page[:begin] + text + page[end:]
        if args.record:
            path.write_text(page)
        print(f"{path}: {time.time() - start:.1f} s")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
