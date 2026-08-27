"""Manuscript stale-number audit (FINAL_POLISH_BRIEF section 3).

Two rule sets over every .tex source of the paper and supplement:

FORBIDDEN -- patterns that can only be a stale copy of a superseded
headline value (pre-campaign actuation medians, the old retention
lead, the old derived ratio, dropped phrasing). Any hit fails.

REQUIRED -- current headline values that must appear at least once
somewhere in the sources (they are allowed to appear many times; the
point is that the current value exists and the greps above prove the
stale one does not).

Exit 0 on pass; non-zero with a file:line report on failure.
Run via `make audit` (also chained at the end of `make paper`).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1] / "manuscript" / "paper"
SOURCES = sorted(PAPER.glob("sections/*.tex")) + [
    PAPER / "main.tex", PAPER / "supplement.tex"]

# Stale values and dropped phrasing. Comments give the replacement.
FORBIDDEN = [
    (r"3\.3--4\.3", "old 6-cycle actuation range; now 3.0--4.2 s"),
    (r"3\.3\\,s", "old rebalance median; now 2.95 s"),
    (r"4\.3\\,s", "old rate-effect median; now 4.2 s"),
    (r"1\.3--4\.5", "old 6-cycle rebalance range; IQRs now"),
    (r"3--4\\,s", "imprecise actuation range; now 3.0--4.2 s"),
    (r"3\.8\\,s", "old actuation midpoint; superseded"),
    (r"45\$?\\times\$?", "old derived ratio; now 'more than 40x'"),
    (r"45 times", "old derived ratio; now 'more than 40x'"),
    (r"lead at least 174", "old pooled retention lead; now >=176 s"),
    (r"\\ge\$?174\\,s", "old pooled retention lead; now >=176 s"),
    (r"median \$?\\ge\$?174", "old pooled retention lead; now >=176 s"),
    (r"token-free", "dropped phrasing (polish item 5)"),
    (r"token cost", "dropped phrasing (polish item 5)"),
    (r"six scale-up cycles", "old cycle count; now 15 pooled"),
    (r"\bP4\b", "internal obligation label; the PDF says Lemma 1/2"),
    (r"\bP5\b", "internal obligation label; the PDF says Lemma 1/2"),
    (r"(?i)draft manuscript", "submission build carries no draft mark"),
    (r"Manuscript draft", "submission build carries no draft mark"),
]

# Current headline values; each must appear in at least one source.
REQUIRED = [
    r"0\.8--4\.7",            # drift relative error
    r"0\.039",                # held-out forecast W1
    r"0\.347",                # persistence W1
    r"41 (pooled |forecastable )?bursts",
    r"12/12",                 # onset recall
    r"3 false alarms",
    r"29 no-onset",
    r"11\\,ms",               # deployment-config latency
    r"23 (campaign )?crossings",
    r"seven (conditions|cells)|seven-cell",
    r"\\ge\$?176\\,s|\\ge\$176",   # pooled retention lead
    r"2\.95",                 # rebalance median, 15 cycles
    r"4\.2",                  # rate-effect median, 15 cycles
    r"0\.104",                # 384-partition forecast W1
    r"0\.297",                # 384-partition persistence W1
    r"-0\.49",                # convergence slope vs N
    r"-0\.83",                # convergence slope vs B
    r"0\.87\$?--\$?1\.47",    # measured spectral decay exponents
    r"M = 48|M=48|M\$?{=}\$?48",   # genie continuations
]


def main() -> int:
    failures: list[str] = []
    texts = {p: p.read_text() for p in SOURCES}

    for pattern, why in FORBIDDEN:
        rx = re.compile(pattern)
        for path, text in texts.items():
            for i, line in enumerate(text.splitlines(), 1):
                if rx.search(line):
                    failures.append(
                        f"STALE {path.relative_to(PAPER)}:{i} "
                        f"matches /{pattern}/ ({why})")

    for pattern in REQUIRED:
        rx = re.compile(pattern)
        if not any(rx.search(t) for t in texts.values()):
            failures.append(f"MISSING current headline /{pattern}/ "
                            f"in every source")

    if failures:
        print("number audit FAILED:")
        for f in failures:
            print("  " + f)
        return 1
    print(f"number audit passed: {len(FORBIDDEN)} stale patterns absent, "
          f"{len(REQUIRED)} current headlines present "
          f"({len(SOURCES)} sources)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
