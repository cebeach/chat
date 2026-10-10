"""The pure logic of /duel: two sides write the same story and challenge each other.

No terminal and no network (see docs/duel.md). This module holds the fixed phase
sequence and who takes part in each phase, the issue ledger (its transition table is
the only way a status changes), the prompt builders, the lenient parsers for what the
models answer, the option precedence for the two sides, the /duel argument parser and
the atomic record file. chat.py runs the phases and the prompts around them.

Words used in the same sense everywhere: a *turn* is one model generation by one
conversation, a *phase* is a step in which every participant takes one turn, and a
*duel* is the fixed sequence run once.
"""

import enum
import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from conversation import Conversation

SIDES = ("a", "b")
JUDGE = "judge"  # the neutral conversation that runs JUDGE and MERGE

# Stored in place of an empty answer in CROSS, DEFEND and REPLY, because an assistant
# message with empty content is dropped from later prompts together with its user message.
NO_ANSWER = "(no answer)"

REWRITE_MARK = "---REWRITE---"


class Phase(enum.Enum):
    DRAFT = "draft"
    CROSS = "cross-examine"
    DEFEND = "defend"
    REPLY = "reply"
    REVISE = "revise"
    JUDGE = "judge"
    MERGE = "merge"


# The sequence a duel runs once. MERGE is not part of it: it runs only on request.
SEQUENCE = (Phase.DRAFT, Phase.CROSS, Phase.DEFEND, Phase.REPLY, Phase.REVISE, Phase.JUDGE)

# Phases whose empty answer is useless, so the phase ends abnormally; in the others the
# missing-answer rules apply.
EMPTY_IS_ABNORMAL = frozenset({Phase.DRAFT, Phase.REVISE, Phase.JUDGE, Phase.MERGE})

# Causes of an abnormal end, as stored in the record ("abnormal:<PHASE>:<cause>").
CAUSES = ("interrupted", "context", "cut_off", "server_error", "empty_answer", "format")

DEFAULT_AXES = ["plot", "character", "dialogue", "pacing", "voice"]
DEFAULT_STANCE_A = (
    "You are plot-driven. You care most about structure, cause and effect, and stakes: "
    "every scene should change something and lead to the next."
)
DEFAULT_STANCE_B = (
    "You are character-driven. You care most about motivation, voice and interior life: "
    "events matter for what they do to the people in them."
)

DEFAULT_DRAFT_LENGTH = "about 500 to 800 words"


def other(side):
    """The opposing side."""
    return "b" if side == "a" else "a"


def side_label(side):
    return "Judge" if side == JUDGE else f"Side {side.upper()}"


# ---------------------------------------------------------------------------
# The issue ledger

RAISED = "raised"
ACCEPTED = "accepted"
REBUTTED = "rebutted"
MAINTAINED = "maintained"
WITHDRAWN = "withdrawn"
ADDRESSED = "addressed"
DECLINED = "declined"

STANDING = frozenset({ACCEPTED, MAINTAINED})
TERMINAL = frozenset({ADDRESSED, DECLINED, WITHDRAWN})

# The only legal transitions made by the phases ...
_PHASE_MOVES = {
    (RAISED, ACCEPTED),
    (RAISED, REBUTTED),
    (REBUTTED, MAINTAINED),
    (REBUTTED, WITHDRAWN),
    (ACCEPTED, ADDRESSED),
    (ACCEPTED, DECLINED),
    (MAINTAINED, ADDRESSED),
    (MAINTAINED, DECLINED),
}
# ... and by the user at a pause.
_ACCEPT_FROM = frozenset({RAISED, REBUTTED, MAINTAINED, WITHDRAWN})
_DROP_FROM = frozenset({RAISED, ACCEPTED, REBUTTED, MAINTAINED})


class IllegalTransition(ValueError):
    """A status change the lifecycle does not allow."""


class Refused(ValueError):
    """A user override or steer that is no longer possible; the message says why."""


@dataclass
class Issue:
    n: int
    against: str  # the side whose draft it is about
    by: str  # the side that raised it
    axis: str
    quote: str = ""
    objection: str = ""
    alternative: str = ""
    status: str = RAISED
    by_user: bool = False
    unparsed: bool = False
    axis_known: bool = True
    rebuttal: str = ""  # the defender's reason, from DEFEND
    declined_reason: str = ""  # from REVISE

    def to_dict(self):
        return {
            "n": self.n,
            "against": self.against,
            "by": self.by,
            "axis": self.axis,
            "axis_known": self.axis_known,
            "quote": self.quote,
            "objection": self.objection,
            "alternative": self.alternative,
            "status": self.status,
            "by_user": self.by_user,
            "unparsed": self.unparsed,
            "rebuttal": self.rebuttal,
            "declined_reason": self.declined_reason,
        }


@dataclass
class Ledger:
    """Every issue of a duel, plus the steers queued per side.

    The statuses change only through the methods here, which raise on anything the
    lifecycle table does not allow. `closed` is set once REVISE has run or been
    skipped: overrides and steers are refused after that.
    """

    issues: list = field(default_factory=list)
    steers: dict = field(default_factory=lambda: {"a": [], "b": []})
    closed: bool = False

    def get(self, n):
        for issue in self.issues:
            if issue.n == n:
                return issue
        raise KeyError(n)

    def with_status(self, status, against=None, by=None):
        return [
            i
            for i in self.issues
            if i.status == status and (against is None or i.against == against) and (by is None or i.by == by)
        ]

    def standing(self, against=None):
        return [i for i in self.issues if i.status in STANDING and (against is None or i.against == against)]

    # --- phase transitions
    def raise_issue(self, by, axis, quote="", objection="", alternative="", axis_known=True, unparsed=False):
        issue = Issue(
            n=len(self.issues) + 1,
            against=other(by),
            by=by,
            axis=axis,
            quote=quote,
            objection=objection,
            alternative=alternative,
            axis_known=axis_known,
            unparsed=unparsed,
        )
        self.issues.append(issue)
        return issue

    def _move(self, issue, to, allowed):
        if (issue.status, to) not in allowed:
            raise IllegalTransition(f"issue {issue.n}: {issue.status} -> {to} is not allowed")
        issue.status = to

    def accept(self, n, unparsed=False):
        issue = self.get(n)
        self._move(issue, ACCEPTED, _PHASE_MOVES)
        issue.unparsed = issue.unparsed or unparsed

    def rebut(self, n, reason):
        issue = self.get(n)
        self._move(issue, REBUTTED, _PHASE_MOVES)
        issue.rebuttal = reason

    def maintain(self, n, unparsed=False):
        issue = self.get(n)
        self._move(issue, MAINTAINED, _PHASE_MOVES)
        issue.unparsed = issue.unparsed or unparsed

    def withdraw(self, n):
        self._move(self.get(n), WITHDRAWN, _PHASE_MOVES)

    def address(self, n):
        self._move(self.get(n), ADDRESSED, _PHASE_MOVES)

    def decline(self, n, reason):
        issue = self.get(n)
        self._move(issue, DECLINED, _PHASE_MOVES)
        issue.declined_reason = reason

    # --- user overrides and steers (only at a pause, only until REVISE has run or been skipped)
    def _check_open(self, what):
        if self.closed:
            raise Refused(f"Too late to {what}: the revision has already run or been skipped.")

    def override_accept(self, n):
        self._check_open("accept an issue")
        issue = self._existing(n)
        if issue.status not in _ACCEPT_FROM:
            raise Refused(f"Issue {n} is {issue.status}; it cannot be accepted.")
        issue.status = ACCEPTED
        issue.by_user = True

    def override_drop(self, n):
        self._check_open("drop an issue")
        issue = self._existing(n)
        if issue.status not in _DROP_FROM:
            raise Refused(f"Issue {n} is {issue.status}; it cannot be dropped.")
        issue.status = WITHDRAWN
        issue.by_user = True

    def _existing(self, n):
        try:
            return self.get(n)
        except KeyError:
            raise Refused(f"There is no issue {n}.") from None

    def add_steer(self, target, note):
        """Queue `note` for side `target` ('a', 'b' or 'both')."""
        self._check_open("steer")
        for side in SIDES if target == "both" else (target,):
            self.steers[side].append(note)

    def close(self):
        self.closed = True

    def to_dict(self):
        return {
            "issues": [i.to_dict() for i in self.issues],
            "steers": {k: list(v) for k, v in self.steers.items()},
        }


# ---------------------------------------------------------------------------
# Who takes part in a phase


def participants(phase, ledger):
    """The conversations that take a turn in `phase`, computed from the ledger as it is now."""
    if phase in (Phase.DRAFT, Phase.CROSS):
        return list(SIDES)
    if phase == Phase.DEFEND:
        return [s for s in SIDES if ledger.with_status(RAISED, against=s)]
    if phase == Phase.REPLY:
        return [s for s in SIDES if ledger.with_status(REBUTTED, by=s)]
    if phase == Phase.REVISE:
        return [s for s in SIDES if ledger.standing(against=s) or ledger.steers[s]]
    return [JUDGE]


def next_phase(after, ledger):
    """(phase, participants, skipped) for the phase that runs after `after`, or None at the end.

    `after` is the last phase of SEQUENCE that finished or was skipped (None at the start).
    `skipped` lists the phases passed over because nobody takes part in them.
    """
    index = 0 if after is None else SEQUENCE.index(after) + 1
    skipped = []
    for phase in SEQUENCE[index:]:
        who = participants(phase, ledger)
        if who:
            return phase, who, skipped
        skipped.append(phase)
    return None


def phase_plan(ledger, after=None):
    """The phases that would run from here, with their participants, if the ledger did not change.

    Later phases depend on answers not yet given, so this is exact only for the next
    phase; chat.py asks again after every phase.
    """
    plan = []
    cursor = after
    while (step := next_phase(cursor, ledger)) is not None:
        phase, who, _ = step
        plan.append((phase, who))
        cursor = phase
    return plan


def skip_note(skipped):
    """The one-line note shown when the phases in `skipped` are passed over, or '' for none."""
    if not skipped:
        return ""
    reasons = {
        Phase.DEFEND: "No issues raised",
        Phase.REPLY: "No rebuttals to answer",
        Phase.REVISE: "Nothing to revise",
    }
    names = [p.value for p in skipped]
    listed = names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"
    return f"{reasons.get(skipped[0], 'Nobody takes part')}; skipping {listed}."


def new_conversation(system_prompt, notes=""):
    """A fresh conversation for a side or the judge, built here and not through chat.Conversation."""
    conversation = Conversation(system_prompt)
    conversation.project_notes = notes
    return conversation


def usable_text(phase, answer):
    """The story text an answer holds, or '' when there is none: the DECLINE lines of REVISE do not count."""
    return (parse_revise(answer)[0] if phase == Phase.REVISE else answer).strip()


# ---------------------------------------------------------------------------
# Prompts


def side_system(base, stance):
    """A side's system prompt: the main conversation's, then the stance."""
    return f"{base}\n\n{stance}" if base else stance


JUDGE_SYSTEM = (
    "You are a fair, careful editor. You compare two versions of a story written for the same brief "
    "and say which serves the brief better, and why."
)


def draft_prompt(brief):
    return (
        f"Brief:\n{brief}\n\n"
        "Write the story the brief asks for. Use the length the brief gives; if it gives none, "
        f"write {DEFAULT_DRAFT_LENGTH}. Reply with the story text only, no preface and no commentary."
    )


def cross_prompt(brief, opponent_text, axes):
    return (
        f"Brief:\n{brief}\n\n"
        f"Another writer's draft for the same brief:\n\n{opponent_text}\n\n"
        "Find the real problems in that draft. For each, write exactly one line:\n"
        'ISSUE | axis | "exact quote from the draft" | what is wrong | what you would do instead\n'
        f"axis is one of: {', '.join(axes)}.\n"
        "Raise only issues that matter; if the draft has none, reply with the single line NO ISSUES."
    )


def _issue_block(issue, with_rebuttal=False):
    lines = [
        f"Issue {issue.n} ({issue.axis}): {issue.objection}",
        f'  Quote: "{issue.quote}"' if issue.quote else None,
        f"  Suggested: {issue.alternative}" if issue.alternative else None,
        f"  Your rebuttal: {issue.rebuttal}" if with_rebuttal else None,
    ]
    return "\n".join(line for line in lines if line)


def defend_prompt(brief, issues):
    blocks = "\n\n".join(_issue_block(i) for i in issues)
    return (
        f"Brief:\n{brief}\n\n"
        f"Another writer raised these issues against your draft:\n\n{blocks}\n\n"
        "Answer every issue with one line each:\n"
        "ACCEPT n\n"
        "REBUT n: your reason\n"
        "Accept what is right; rebut only what is wrong or against the brief."
    )


def reply_prompt(brief, issues):
    blocks = "\n\n".join(f"{_issue_block(i)}\n  Their rebuttal: {i.rebuttal}" for i in issues)
    return (
        f"Brief:\n{brief}\n\n"
        f"The writer rebutted these issues you raised:\n\n{blocks}\n\n"
        "Answer every issue with one line each:\n"
        "MAINTAIN n\n"
        "WITHDRAW n\n"
        "Withdraw an issue if the rebuttal is right."
    )


def word_count(text):
    return len(text.split())


def revise_prompt(brief, own_text, issues, steers):
    parts = [f"Brief:\n{brief}", f"Your current text:\n\n{own_text}"]
    if issues:
        parts.append("Issues to fix:\n\n" + "\n\n".join(_issue_block(i) for i in issues))
    if steers:
        parts.append("Notes from the editor:\n" + "\n".join(f"- {s}" for s in steers))
    parts.append(
        "Rewrite the text, fixing the issues and following the notes. Keep it within about 10% of "
        f"the current length ({word_count(own_text)} words) unless a note says otherwise. "
        "Reply with the full revised text. If you decline an issue, add after the text one line "
        "per declined issue: DECLINE n: your reason"
    )
    return "\n\n".join(parts)


def judge_prompt(brief, texts):
    """`texts` is the pair of story texts in the order shown as Text 1 and Text 2."""
    return (
        f"Brief:\n{brief}\n\n"
        f"Text 1:\n\n{texts[0]}\n\nText 2:\n\n{texts[1]}\n\n"
        "Say which text serves the brief better and why, in a few short paragraphs. "
        "End with one final line: RECOMMEND: Text 1, RECOMMEND: Text 2 or RECOMMEND: NONE."
    )


def merge_prompt(brief, base, donor):
    return (
        f"Brief:\n{brief}\n\n"
        f"Base text:\n\n{base}\n\nOther text:\n\n{donor}\n\n"
        "List the strengths of the other text that the base text lacks. Then write a line "
        f"containing only {REWRITE_MARK} and, after it, the base text rewritten to take those "
        "strengths you select. Keep the base text's length and voice."
    )


def judge_order(rng):
    """The sides in the order shown as Text 1, Text 2: a random permutation from `rng`."""
    order = list(SIDES)
    rng.shuffle(order)
    return tuple(order)


def order_note(order):
    return ", ".join(f"Text {i} = {side_label(s)}" for i, s in enumerate(order, 1))


# ---------------------------------------------------------------------------
# Parsers (they work on the answer part only; thinking is never passed in)

_LINE_PREFIX = r"^[\s>*#\-\d.)]*"
_ISSUE_LINE = re.compile(_LINE_PREFIX + r"ISSUE\b\s*[|:]?\s*(.*)$", re.I)
_NO_ISSUES = re.compile(r"^\W*NO\s+ISSUES\b", re.I | re.M)
_DEFEND_LINE = re.compile(_LINE_PREFIX + r"(ACCEPT|REBUT)\b\s*#?(\d+)\s*[:\-]?\s*(.*)$", re.I)
_REPLY_LINE = re.compile(_LINE_PREFIX + r"(MAINTAIN|WITHDRAW)\b\s*#?(\d+)", re.I)
_DECLINE_LINE = re.compile(_LINE_PREFIX + r"DECLINE\b\s*#?(\d+)\s*[:\-]?\s*(.*)$", re.I)
_RECOMMEND = re.compile(r"^\W*RECOMMEND\s*:\s*(?:TEXT\s*([12])|NONE)\b", re.I)


def _unquote(text):
    text = text.strip()
    return text[1:-1].strip() if len(text) >= 2 and text[0] in "\"'“" and text[-1] in "\"'”" else text


def parse_cross(text, axes):
    """The issues in a CROSS answer as dicts, or [] for NO ISSUES.

    An answer that is neither (and has no ISSUE line) becomes one `general` issue marked
    unparsed, so a small model that ignores the format still has its objection heard.
    An axis outside `axes` is kept but flagged (axis_known False).
    """
    found = []
    for line in text.splitlines():
        m = _ISSUE_LINE.match(line)
        if not m:
            continue
        parts = [p.strip() for p in m.group(1).split("|")]
        parts += [""] * (4 - len(parts))
        axis, quote, objection = parts[0].lower(), _unquote(parts[1]), parts[2]
        alternative = " | ".join(p for p in parts[3:] if p)
        if not (axis or quote or objection):
            continue
        found.append(
            {
                "axis": axis or "general",
                "quote": quote,
                "objection": objection,
                "alternative": alternative,
                "axis_known": axis in axes,
                "unparsed": False,
            }
        )
    if found:
        return found
    body = text.strip()
    if not body or body == NO_ANSWER or _NO_ISSUES.search(body):
        return []
    return [
        {
            "axis": "general",
            "quote": "",
            "objection": body[:600],
            "alternative": "",
            "axis_known": True,
            "unparsed": True,
        }
    ]


def parse_defend(text, ids):
    """{n: ("accept", "", unparsed) | ("rebut", reason, unparsed)} for every n in `ids`.

    An issue the answer does not mention is accepted (unparsed): a missing answer lets it stand.
    """
    answers = {}
    for line in text.splitlines():
        m = _DEFEND_LINE.match(line)
        if m and int(m.group(2)) in ids:
            n = int(m.group(2))
            if m.group(1).upper() == "ACCEPT":
                answers.setdefault(n, ("accept", "", False))
            else:
                answers.setdefault(n, ("rebut", m.group(3).strip(), False))
    return {n: answers.get(n, ("accept", "", True)) for n in ids}


def parse_reply(text, ids):
    """{n: ("maintain" | "withdraw", unparsed)}; an unmentioned issue is maintained (unparsed)."""
    answers = {}
    for line in text.splitlines():
        m = _REPLY_LINE.match(line)
        if m and int(m.group(2)) in ids:
            answers.setdefault(int(m.group(2)), (m.group(1).lower(), False))
    return {n: answers.get(n, ("maintain", True)) for n in ids}


def parse_revise(text):
    """(story, {n: reason}): trailing `DECLINE n: reason` lines are stripped from the text."""
    lines = text.rstrip().splitlines()
    declines = {}
    while lines:
        if not lines[-1].strip():
            lines.pop()
            continue
        m = _DECLINE_LINE.match(lines[-1])
        if not m:
            break
        declines.setdefault(int(m.group(1)), m.group(2).strip())
        lines.pop()
    return "\n".join(lines).strip(), declines


def parse_judge(text, order):
    """(reasons, recommended side or None). Text numbers are mapped back through `order`.

    A missing, unparseable or NONE recommendation gives None; the reasons are still returned.
    """
    lines = text.rstrip().splitlines()
    side = None
    for i in range(len(lines) - 1, -1, -1):
        m = _RECOMMEND.match(lines[i])
        if m:
            side = order[int(m.group(1)) - 1] if m.group(1) else None
            del lines[i]
            break
    return "\n".join(lines).strip(), side


def parse_merge(text):
    """(strengths, rewritten text), or None when the REWRITE line or the text is missing."""
    strengths, mark, rewrite = text.partition(REWRITE_MARK)
    if not mark or not rewrite.strip():
        return None
    return strengths.strip(), rewrite.strip()


# ---------------------------------------------------------------------------
# Applying a finished phase's answers to the ledger


def apply_cross(ledger, side, answer, axes):
    """`side` cross-examined the opponent's draft; returns the issues it raised."""
    return [ledger.raise_issue(side, **found) for found in parse_cross(answer, axes)]


def apply_defend(ledger, side, answer):
    raised = ledger.with_status(RAISED, against=side)
    for n, (verdict, reason, unparsed) in parse_defend(answer, [i.n for i in raised]).items():
        if verdict == "accept":
            ledger.accept(n, unparsed=unparsed)
        else:
            ledger.rebut(n, reason)


def apply_reply(ledger, side, answer):
    rebutted = ledger.with_status(REBUTTED, by=side)
    for n, (verdict, unparsed) in parse_reply(answer, [i.n for i in rebutted]).items():
        if verdict == "maintain":
            ledger.maintain(n, unparsed=unparsed)
        else:
            ledger.withdraw(n)


def apply_revise(ledger, side, answer):
    """Returns the revised story; each standing issue against `side` becomes addressed or declined."""
    story, declines = parse_revise(answer)
    for issue in ledger.standing(against=side):
        if issue.n in declines:
            ledger.decline(issue.n, declines[issue.n])
        else:
            ledger.address(issue.n)
    return story


# ---------------------------------------------------------------------------
# Options per side

_RANDOM_SEEDS = (None, -1, 0xFFFFFFFF)


def side_options(options, cfg, server_defaults):
    """{"a": opts, "b": opts}: the request options of each side.

    Precedence: the duel setting (`temperature_x`, `seed_x` in `cfg`), then the `/set`
    override in `options`, then the server's default. The server's "random" seed does not
    count as a seed. When both sides end up with the same fixed seed, B uses that + 1.
    """
    out = {}
    effective = {}
    for side in SIDES:
        opts = dict(options)
        if cfg.get(f"temperature_{side}") is not None:
            opts["temperature"] = cfg[f"temperature_{side}"]
        if cfg.get(f"seed_{side}") is not None:
            opts["seed"] = cfg[f"seed_{side}"]
        seed = opts.get("seed")
        if seed in _RANDOM_SEEDS:
            seed = (server_defaults or {}).get("seed")
        effective[side] = None if seed in _RANDOM_SEEDS else seed
        out[side] = opts
    if effective["a"] is not None and effective["a"] == effective["b"]:
        out["b"]["seed"] = (effective["a"] + 1) & 0xFFFFFFFF
    return out


# ---------------------------------------------------------------------------
# The /duel command line


@dataclass
class Command:
    kind: str  # "usage", "ledger", "set", "run" or "error"
    brief: str = ""
    auto: bool = False
    key: str = ""
    value: str = ""
    error: str = ""


def parse_args(args):
    """Parse what follows `/duel`.

    The first word is a subcommand only when it is exactly `set` or `ledger`. Flags
    (`--auto`) precede the brief, and `--` ends flag and subcommand parsing.
    """
    text = args.strip()
    if not text:
        return Command("usage")
    word, _, rest = text.partition(" ")
    rest = rest.strip()
    if word == "ledger":
        return Command("ledger")
    if word == "set":
        key, _, value = rest.partition(" ")
        if not key:
            return Command("error", error="Usage: /duel set <key> <value>")
        return Command("set", key=key, value=value.strip())
    auto = False
    while text.startswith("--"):
        word, _, rest = text.partition(" ")
        if word == "--":
            text = rest.strip()
            break
        if word == "--auto":
            auto = True
            text = rest.strip()
            continue
        return Command("error", error=f"Unknown flag {word}. Use -- before a brief that starts with --.")
    if not text:
        return Command("error", error="A duel needs a brief.")
    return Command("run", brief=text, auto=auto)


# ---------------------------------------------------------------------------
# The record file


def write_record(directory, record, started):
    """Write `record` as duel-YYYYMMDD-HHMMSS.json (+ -2, -3, ... if taken) under `directory`.

    The file is written atomically, so an interruption never leaves half a record.
    `started` is a datetime. Returns the path.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"duel-{started:%Y%m%d-%H%M%S}"
    path = directory / f"{stem}.json"
    n = 2
    while path.exists():
        path = directory / f"{stem}-{n}.json"
        n += 1
    rewrite_record(path, record)
    return path


def rewrite_record(path, record):
    """Replace the record at `path` atomically."""
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".duel-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
