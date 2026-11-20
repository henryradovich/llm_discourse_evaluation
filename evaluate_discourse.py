#!/usr/bin/env python3
"""Annotate CSV/JSONL podcast text using Llama and/or Qwen on a CURC GPU.

Gold labels may be supplied in a separate JSONL file using --gold-file.
Supports flat gold_label and nested labels.dominant_mode annotation exports.
Secondary discourse labels and emotion annotations are ignored.
System and user prompts are read from one .txt file; see prompt_example.txt.
If the [USER] section omits {text}, the target transcript text is appended automatically.
"""
import argparse
import csv
import gc
import hashlib
import json
import re
import os
import tempfile
from html.parser import HTMLParser
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

LABELS = ("Argumentation", "Narration", "Description", "Exposition", "Ad Read", "Unclear")
MODELS = {"llama": "meta-llama/Llama-3.1-8B-Instruct", "qwen": "Qwen/Qwen3-8B"}
PLACEHOLDERS = ("{text}", "{id}", "{speaker}", "{context_before}", "{context_after}", "{context}")


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, help="CSV or JSONL text records (gold_label optional)")
    p.add_argument("--prompt-file", type=Path, help="Text file containing [SYSTEM] and [USER] sections (not required when reparsing)")
    p.add_argument("--gold-file", type=Path, help="Separate JSONL file with IDs and human gold labels")
    p.add_argument("--models", nargs="+", default=["llama", "qwen"], help="llama qwen or Hugging Face model IDs")
    p.add_argument("--output-dir", type=Path, default=Path("results"))
    p.add_argument("--text-col", default="text")
    p.add_argument("--gold-col", default="gold_label",
                   help="Gold-label field or dotted path (default: gold_label; supports labels, labels.dominant_mode, and automatic nested fallback)")
    p.add_argument("--id-col", default="id", help="ID field in the transcript/text input")
    p.add_argument("--gold-id-col", default=None,
                   help="Gold JSONL ID field (auto-detects instance_id if --id-col is absent)")
    p.add_argument("--speaker-col", default="speaker")
    p.add_argument("--before-col", default="context_before")
    p.add_argument("--after-col", default="context_after")
    p.add_argument("--context-col", default="context",
                   help="JSONL field containing the HTML transcript with a <mark>ed target (default: context)")
    p.add_argument("--context-window", type=int, default=3,
                   help="Maximum surrounding utterances on EACH side of target; 0 means all (default: 3)")
    p.add_argument("--auto-include-context", action="store_true",
                   help="Insert context into [USER] automatically if the prompt has no context placeholders")
    p.add_argument("--preview-prompts", action="store_true",
                   help="Print first three rendered user prompts and exit, without loading any model or GPU")
    p.add_argument("--limit", type=int, default=0, help="Test only first N rows; 0 means all")
    p.add_argument("--max-input-tokens", type=int, default=16384)
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--dtype", choices=("bf16", "fp16", "auto"), default="bf16")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--reparse-only", action="store_true",
                   help="Re-extract labels from saved JSONL raw_response values and rebuild CSV/metrics without GPU inference")
    p.add_argument("--repair-csv", type=Path,
                   help="Repair an existing predictions CSV directly; no models, input JSONL, or prompt required")
    p.add_argument("--repair-output", type=Path,
                   help="Where to save repaired CSV (default: SOURCE_reparsed.csv)")
    return p.parse_args()


def canonical(value):
    if value is None:
        return None
    key = re.sub(r"[\s_-]+", "", str(value).strip().lower())
    aliases = {re.sub(r"[\s_-]+", "", x.lower()): x for x in LABELS}
    aliases.update({"argumentative": "Argumentation", "narrative": "Narration",
                    "descriptive": "Description", "expository": "Exposition",
                    "advertisement": "Ad Read", "uncodable": "Unclear"})
    return aliases.get(key)


def read_jsonl(path):
    records = []
    with path.open(encoding="utf-8-sig") as f:
        for number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {number}: {exc}") from exc
            if not isinstance(rec, dict):
                raise ValueError(f"{path} line {number} must be a JSON object")
            records.append(rec)
    return records


def get_field(record, path):
    """Read a direct key or dotted path, e.g. labels.dominant_mode."""
    value = record
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


class _TranscriptHTMLParser(HTMLParser):
    """Convert annotation-tool HTML into ordered (speaker, text, marked) units.

    The input format uses one <div> per speaker turn, one <p> per utterance,
    and a <mark> around the target. The top-level <p> naming the episode is
    metadata, not dialogue. Nothing from this HTML is executed or displayed.
    """

    _speaker_re = re.compile(r"^Speaker\s+([\w-]+)(?:\s*[·•]\s*this turn)?$", re.I)

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.utterances = []
        self.depth = 0
        self.speaker = None
        self.in_paragraph = False
        self.parts = []
        self.marked = False

    def handle_starttag(self, tag, attrs):
        if tag == "div":
            if self.depth == 0:
                self.speaker = None
            self.depth += 1
        elif tag == "p":
            self.in_paragraph = True
            self.parts = []
            self.marked = False
        elif tag == "mark" and self.in_paragraph:
            self.marked = True

    def handle_data(self, data):
        if self.in_paragraph:
            self.parts.append(data)

    def handle_endtag(self, tag):
        if tag == "p" and self.in_paragraph:
            value = re.sub(r"\s+", " ", "".join(self.parts)).strip()
            self.in_paragraph = False
            if not value:
                return
            speaker_match = self._speaker_re.fullmatch(value)
            if speaker_match:
                self.speaker = "Speaker " + speaker_match.group(1)
            elif self.depth > 0:
                self.utterances.append((self.speaker or "Unknown speaker", value, self.marked))
        elif tag == "div":
            self.depth = max(0, self.depth - 1)


def parse_html_context(raw_html, target, identifier, window):
    """Return (before, after) without the highlighted target utterance.

    Speaker changes are preserved. Prefer the single <mark> in the supplied
    HTML; if there is no highlight, allow an unambiguous exact-text match.
    Do not guess if the target cannot be located in the HTML.
    """
    parser = _TranscriptHTMLParser()
    parser.feed(raw_html)
    parser.close()
    units = parser.utterances
    if not units:
        raise ValueError(f"Text id={identifier!r}: HTML context contains no transcript utterances")

    def normalize(text):
        return re.sub(r"\s+", " ", text).strip().casefold()

    highlighted = [i for i, (_, _, marked) in enumerate(units) if marked]
    if highlighted:
        exact = [i for i in highlighted if normalize(units[i][1]) == normalize(target)]
        if len(exact) == 1:
            target_index = exact[0]
        elif len(highlighted) == 1:
            raise ValueError(
                f"Text id={identifier!r}: highlighted HTML text does not match the 'text' field; "
                f"highlight={units[highlighted[0]][1]!r}, text={target!r}"
            )
        else:
            raise ValueError(f"Text id={identifier!r}: multiple highlighted utterances match ambiguously")
    else:
        exact = [i for i, (_, text, _) in enumerate(units) if normalize(text) == normalize(target)]
        if len(exact) != 1:
            raise ValueError(f"Text id={identifier!r}: no unique highlighted or exact-matching "
                             f"target in HTML context ({len(exact)} matches)")
        target_index = exact[0]

    before_units = units[:target_index]
    after_units = units[target_index + 1:]
    if window:
        before_units = before_units[-window:]
        after_units = after_units[:window]

    def format_units(selected):
        return "\n".join(f"{speaker}: {text}" for speaker, text, _ in selected)

    return format_units(before_units), format_units(after_units)


def combine_context(before, after):
    return ("CONTEXT BEFORE (not the target):\n" + (before or "(none)")
            + "\n\nCONTEXT AFTER (not the target):\n" + (after or "(none)"))


def extract_gold_label(record, gold_col, identifier):
    """Extract ONLY the dominant discourse mode from a gold record.

    Handles both the original flat format:
        {"gold_label": "Narration"}
    and the nested human-annotation format:
        {"labels": {"dominant_mode": {"Narration": "Narration"},
                    "secondary_mode": {...}, "emotion": {...}}}

    Rejects ambiguous or invalid dominant_mode values instead of arbitrarily
    selecting a label. Secondary labels, emotion, spans and links are ignored.
    """
    raw = get_field(record, gold_col)
    chosen_col = gold_col
    if raw is None and gold_col == "gold_label":
        chosen_col = "labels.dominant_mode"
        raw = get_field(record, chosen_col)

    # When --gold-col labels is supplied, the selected object contains ALL
    # annotation dimensions, not just the dominant discourse mode. Descend
    # into dominant_mode before checking that the selected label is unique.
    # This ignores secondary_mode, emotion, spans, links, etc.
    if isinstance(raw, dict) and "dominant_mode" in raw:
        raw = raw["dominant_mode"]
        chosen_col += ".dominant_mode"

    if raw is None or raw == "":
        raise ValueError(f"Gold id={identifier!r} has no value at {gold_col!r} "
                         "or fallback 'labels.dominant_mode'")

    if isinstance(raw, dict):
        if len(raw) != 1:
            raise ValueError(f"Gold id={identifier!r}: {chosen_col!r} must contain "
                             f"exactly one dominant discourse label; got {raw!r}")
        key, value = next(iter(raw.items()))
        from_key = canonical(key)
        from_value = canonical(value)
        if from_key is not None and from_value is not None and from_key != from_value:
            raise ValueError(f"Gold id={identifier!r}: conflicting dominant mode "
                             f"key/value: {raw!r}")
        # Some annotation exports have a label as the key and an empty/boolean
        # selection value, so accept the key when the value is not a label.
        label = from_value or from_key
    elif isinstance(raw, str):
        label = canonical(raw)
    else:
        raise ValueError(f"Gold id={identifier!r}: unsupported dominant_mode "
                         f"structure {raw!r}; expected string or one-item mapping")

    if label is None:
        raise ValueError(f"Gold id={identifier!r}: unknown dominant discourse "
                         f"label in {chosen_col!r}: {raw!r}")
    return label


def load_rows(a):
    context_window = getattr(a, "context_window", 3)
    context_col = getattr(a, "context_col", "context")
    if context_window < 0:
        raise ValueError("--context-window must be >= 0 (0 means all available context)")
    if a.input.suffix.lower() == ".csv":
        with a.input.open(encoding="utf-8-sig", newline="") as f:
            data = list(csv.DictReader(f))
    elif a.input.suffix.lower() in (".jsonl", ".ndjson"):
        data = read_jsonl(a.input)
    else:
        raise ValueError("--input must be .csv, .jsonl, or .ndjson")

    if not data:
        raise ValueError("Input dataset is empty")

    rows = []
    for index, rec in enumerate(data[:a.limit] if a.limit else data, 1):
        text_value = rec.get(a.text_col)
        if not isinstance(text_value, str) or not text_value.strip():
            raise ValueError(f"Row {index} must have a nonempty string in {a.text_col!r}")
        # When --gold-file is provided, ignore any labels embedded in the text data.
        gold_value = None if a.gold_file else rec.get(a.gold_col)
        if gold_value is None or str(gold_value).strip() == "":
            label = None
        else:
            label = canonical(gold_value)
            if label is None:
                raise ValueError(f"Invalid gold label at row {index}: {gold_value!r}")
        record_id = rec.get(a.id_col)
        if record_id is None or str(record_id).strip() == "":
            record_id = f"row_{index}"
        before = str(rec.get(a.before_col) or "")
        after = str(rec.get(a.after_col) or "")
        context = rec.get(context_col)
        if isinstance(context, str) and re.search(r"<p(?:\s|>)", context, re.I):
            # The annotation export stores surrounding turns as HTML, with
            # <mark> around the target. Convert it into plain transcript text.
            html_before, html_after = parse_html_context(
                context, text_value, str(record_id), context_window
            )
            before = before or html_before
            after = after or html_after
            context_text = combine_context(before, after)
        elif isinstance(context, str) and context.strip():
            # Support a plain-text context field, too, without HTML markup.
            context_text = context.strip()
        else:
            context_text = combine_context(before, after)

        rows.append({
            "id": str(record_id), "text": text_value.strip(), "gold": label,
            "speaker": str(rec.get(a.speaker_col) or ""),
            "context_before": before, "context_after": after,
            "context": context_text,
        })
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Text IDs must be unique. Specify --id-col for your dataset.")

    if a.gold_file:
        # The two files are joined on ID, never on line number or text.
        if not all(rec.get(a.id_col) is not None and str(rec[a.id_col]).strip()
                   for rec in (data[:a.limit] if a.limit else data)):
            raise ValueError("All text records need explicit IDs when using --gold-file")
        if a.gold_file.suffix.lower() not in (".jsonl", ".ndjson"):
            raise ValueError("--gold-file must be .jsonl or .ndjson")
        gold_by_id = {}
        gold_data = read_jsonl(a.gold_file)
        if not gold_data:
            raise ValueError(f"Gold file is empty: {a.gold_file}")
        if a.gold_id_col:
            gold_id_col = a.gold_id_col
        elif a.id_col in gold_data[0]:
            gold_id_col = a.id_col
        elif "instance_id" in gold_data[0]:
            gold_id_col = "instance_id"
        else:
            raise ValueError(f"Could not identify gold record ID field. "
                             f"Use --gold-id-col. Available keys: {list(gold_data[0])}")
        print(f"Reading gold IDs from {gold_id_col!r}; "
              "using only dominant discourse labels.", flush=True)
        for number, rec in enumerate(gold_data, 1):
            gold_id = rec.get(gold_id_col)
            if gold_id is None or not str(gold_id).strip():
                raise ValueError(f"Gold row {number} is missing {gold_id_col!r}")
            gold_id = str(gold_id)
            if gold_id in gold_by_id:
                raise ValueError(f"Duplicate gold ID: {gold_id!r}")
            gold_by_id[gold_id] = extract_gold_label(rec, a.gold_col, gold_id)
        text_ids = {r["id"] for r in rows}
        matched = 0
        for row in rows:
            row["gold"] = gold_by_id.get(row["id"])
            matched += row["gold"] is not None
        print(f"Matched {matched}/{len(rows)} text records to gold labels by ID.", flush=True)
        if matched == 0:
            raise ValueError("No IDs matched between --input and --gold-file")
        missing = len(rows) - matched
        unused = len(set(gold_by_id) - text_ids)
        if missing:
            print(f"Warning: {missing} text records have no gold label; they will still be annotated.", flush=True)
        if unused:
            print(f"Note: {unused} gold records are not in the selected text subset.", flush=True)
    return rows


def load_prompt(path, auto_include_context=False):
    """Read both prompt messages from a single .txt file.

    The user's JSONL transcript is loaded separately through --input.
    The [USER] prompt may explicitly contain {text}, but is not required to:
    when missing, the target text is appended to the user message by default.
    An empty [USER] block is also supported.
    """
    raw = path.read_text(encoding="utf-8-sig")
    parts = re.split(r"(?im)^\s*\[(SYSTEM|USER)\]\s*$", raw)
    if len(parts) != 5 or parts[0].strip() or parts[1].upper() != "SYSTEM" or parts[3].upper() != "USER":
        raise ValueError("Prompt file must contain [SYSTEM] followed by [USER] on their own lines")
    system, user = parts[2].strip(), parts[4].strip()
    if not system:
        raise ValueError("The [SYSTEM] prompt section must be nonempty")
    if auto_include_context and not any(
        placeholder in system or placeholder in user
        for placeholder in ("{context}", "{context_before}", "{context_after}")
    ):
        # Opt-in compatibility with prompt files originally designed for
        # target-only evaluation. Insert context without editing prompt.txt.
        if "{text}" in user:
            user = user.replace("{text}",
                "CONTEXT BEFORE (do not classify):\n{context_before}\n\n"
                "TARGET SENTENCE (classify this only):\n{text}\n\n"
                "CONTEXT AFTER (do not classify):\n{context_after}")
        else:
            user = (user + "\n\n" if user else "") + (
                "CONTEXT BEFORE (do not classify):\n{context_before}\n\n"
                "TARGET SENTENCE (classify this only):\n{text}\n\n"
                "CONTEXT AFTER (do not classify):\n{context_after}"
            )
        print("Note: appending parsed transcript context to the user prompt.", flush=True)
    if "{text}" not in user:
        # Never run classification without the target text, even when the
        # template contains only instructions or has an empty [USER] block.
        print("Note: [USER] has no {text} placeholder; appending target text automatically.",
              flush=True)
        user = (user + "\n\n" if user else "") + (
            "TARGET SENTENCE (classify this text only):\n"
            "{text}\n\n"
            "Return only the primary discourse mode."
        )
    # Fingerprint the *effective* messages. This prevents resuming output from
    # an earlier run whose old prompt did not have the auto-appended target.
    prompt_hash = hashlib.sha256(
        json.dumps({"system": system, "user": user}, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return system, user, prompt_hash


def render(template, row):
    # Literal replacement allows JSON examples and other {braces} in prompt text.
    for name in PLACEHOLDERS:
        template = template.replace(name, row.get(name[1:-1], ""))
    return template


# Capture a discourse label only when the model presents it as the ANSWER,
# rather than when a category happens to be mentioned in an explanation.
_LABEL_RE = r"(?:Argumentation|Narration|Description|Exposition|Ad[\s_-]*Read|Unclear)"
_ANSWER_RE = re.compile(rf"(?P<label>{_LABEL_RE})(?![A-Za-z])", re.I)
_KEYED_ANSWER_RE = re.compile(
    rf"^\s*(?:[-*]\s*)?(?:\*\*)?"
    rf"(?:final\s+answer|(?:primary|dominant)\s+(?:discourse\s+)?(?:mode|label)|"
    rf"(?:discourse\s+)?(?:mode|label)|prediction|classification|answer)"
    rf"(?:\*\*)?\s*[:=–-]\s*[\s\*`\"']*(?P<label>{_LABEL_RE})\b",
    re.I,
)
_PHRASAL_ANSWER_RES = (
    re.compile(
        rf"\b(?:I\s+(?:would|will|might)\s+|I'd\s+|I\s+)?"
        rf"(?:label|classify|tag)\b[^\n]{{0,240}}?\b(?:as)\s*:?\s*[\s\*`\"']*"
        rf"(?P<label>{_LABEL_RE})\b", re.I),
    re.compile(
        rf"\b(?:the\s+)?(?:primary|dominant)\s+(?:discourse\s+)?(?:mode|label)\b"
        rf"[\s\S]{{0,260}}?\b(?:is|would\s+be)\s*:?\s*[\s\*`\"']*"
        rf"(?P<label>{_LABEL_RE})\b", re.I),
    re.compile(
        rf"\b(?:I\s+(?:would|will)\s+say|my\s+(?:answer|classification)\s+is|"
        rf"the\s+(?:correct|best)\s+label\s+is)\s*:?\s*[\s\*`\"']*"
        rf"(?P<label>{_LABEL_RE})\b", re.I),
    re.compile(
        rf"\b(?:the\s+)?(?:text|sentence|segment|utterance)\s+(?:is|should\s+be)\s+"
        rf"(?:labeled|classified|tagged)\s+(?:as|:)\s*[\s\*`\"']*"
        rf"(?P<label>{_LABEL_RE})\b", re.I),
)


def extract_prediction(response):
    """Return (label, method) for explicit model answers, or (None, 'unparsed').

    Explanatory responses often mention several category names; matching the
    first category word ANYWHERE in the text would silently create false labels.
    This parser accepts JSON, keyed answers, a standalone label, and phrases
    explicitly stating the model's chosen classification.
    """
    if not isinstance(response, str) or not response.strip():
        return None, "unparsed"

    text = re.sub(r"<think>.*?</think>", "", response, flags=re.I | re.S).strip()
    # Do not mistake an unclosed Qwen thinking block for the final answer.
    if re.search(r"<think>", text, re.I):
        text = re.sub(r"(?is)^.*?</think>", "", text).strip() if re.search(r"</think>", text, re.I) else ""
    if not text:
        return None, "unparsed"

    # Structured response, including common Markdown JSON fences.
    json_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I).strip()
    try:
        obj = json.loads(json_text)
    except (json.JSONDecodeError, ValueError):
        obj = None
    if isinstance(obj, str):
        label = canonical(obj)
        if label:
            return label, "json_string"
    elif isinstance(obj, dict):
        for key in ("primary_mode", "dominant_mode", "label", "prediction", "discourse_mode", "answer"):
            candidate = obj.get(key)
            if isinstance(candidate, dict) and len(candidate) == 1:
                k, v = next(iter(candidate.items()))
                candidate = v if canonical(v) else k
            label = canonical(candidate)
            if label:
                return label, "json_object"

    # Explicit answer markers take precedence over mentions in reasoning.
    for line in text.splitlines():
        match = _KEYED_ANSWER_RE.match(line)
        if match:
            return canonical(match.group("label")), "keyed_answer"

    # Leading standalone label, e.g. "Argumentation. This sentence ...".
    first = next((line.strip() for line in text.splitlines() if line.strip()), "")
    normalized = re.sub(r"^\s*(?:[-*•]\s*)?", "", first)
    normalized = normalized.lstrip(" `*\"'")
    match = _ANSWER_RE.match(normalized)
    if match and (not normalized[match.end():] or normalized[match.end()] in " .,;:!?)\"'`*—–\t"):
        return canonical(match.group("label")), "leading_label"

    # Free-form explanation that explicitly states the classification.
    # Limit to the opening text to avoid class names mentioned only later in rationale.
    head = text[:800]
    for pattern in _PHRASAL_ANSWER_RES:
        match = pattern.search(head)
        if match:
            return canonical(match.group("label")), "phrased_answer"

    # "I would label this as:\n\n- Narration" and similar responses.
    # Accept only a complete standalone label, not arbitrary explanatory text.
    for line in text.splitlines()[:12]:
        stripped = re.sub(r"^\s*(?:[-*•]\s*)?", "", line).strip(" \t`*\"'.,:;!")
        label = canonical(stripped)
        if label:
            return label, "standalone_label"

    return None, "unparsed"


def parse_label(response):
    """Return the model's explicit primary discourse label, if identifiable."""
    return extract_prediction(response)[0]


def reparse_saved_predictions(output):
    """Reparse existing checkpoint responses WITHOUT re-running either model.

    Updates saved JSONL predictions atomically, so report() can regenerate the
    predictions CSV and metrics from the repaired values.
    """
    if not output.exists():
        raise FileNotFoundError(f"No saved predictions at {output}; cannot use --reparse-only")
    records = []
    changed = 0
    unresolved = 0
    with output.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict) or "id" not in obj:
                raise ValueError(f"Invalid prediction record at {output}:{line_number}")
            if "raw_response" in obj:
                label, method = extract_prediction(obj["raw_response"])
                if obj.get("prediction") != label:
                    changed += 1
                obj["prediction"] = label
                obj["parse_method"] = method
                obj["quality_warning"] = response_warning(obj["raw_response"], obj.get("text"))
            else:
                obj["parse_method"] = "no_raw_response"
                label = obj.get("prediction")
            if label is None:
                unresolved += 1
            records.append(obj)
    # Write to a temporary file in the same directory and atomically replace.
    fd, name = tempfile.mkstemp(prefix=output.name + ".", suffix=".tmp", dir=str(output.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for obj in records:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        os.replace(name, output)
    finally:
        if os.path.exists(name):
            os.unlink(name)
    print(f"Reparsed {len(records)} saved responses for {output.name}: "
          f"{changed} predictions corrected; {unresolved} still unresolved.", flush=True)


def response_warning(raw_response, target_text):
    """Flag obvious off-target responses for researcher review, without changing the parsed label."""
    if not isinstance(raw_response, str):
        return None
    if re.search(r"\b(?:here(?:'s| is)|consider)\s+(?:a|this)\s+sentence\s*:", raw_response, re.I):
        # An invented example instead of answering for the provided target.
        if not target_text or target_text.strip().casefold() not in raw_response.casefold():
            return "possible_off_target_example"
    return None


def repair_csv_file(source, destination=None):
    """Recover labels in a prior CSV without loading models or needing gold-file inputs."""
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination is None:
        destination = source.with_name(source.stem + "_reparsed.csv")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "raw_response" not in reader.fieldnames:
            raise ValueError("CSV must include a raw_response column")
        fieldnames = list(reader.fieldnames)
        for field in ("parse_method", "quality_warning"):
            if field not in fieldnames:
                fieldnames.append(field)
        records = list(reader)
    corrected = 0
    unresolved = 0
    flagged = 0
    for rec in records:
        label, method = extract_prediction(rec["raw_response"])
        new_label = label or ""
        if rec.get("prediction", "").strip() != new_label:
            corrected += 1
        rec["prediction"] = new_label
        rec["parse_method"] = method
        rec["quality_warning"] = response_warning(rec["raw_response"], rec.get("text")) or ""
        unresolved += not bool(label)
        flagged += bool(rec["quality_warning"])
    with destination.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"Repaired CSV: {destination} ({len(records)} rows, {corrected} corrected, "
          f"{unresolved} unresolved, {flagged} warnings)", flush=True)
    return destination


def infer(model_id, rows, prompts, output, a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU detected. Request a CURC interactive GPU session.")
    print(f"\nLoading {model_id} on {torch.cuda.get_device_name(0)}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=a.local_files_only)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "auto": "auto"}[a.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, device_map="auto", low_cpu_mem_usage=True,
        local_files_only=a.local_files_only)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    completed = {}
    if output.exists():
        with output.open(encoding="utf-8") as f:
            completed = {r["id"]: r for line in f if line.strip() for r in [json.loads(line)]}
    with output.open("a", encoding="utf-8") as f:
        for i, row in enumerate(rows, 1):
            if row["id"] in completed:
                continue
            messages = [{"role": "system", "content": render(prompts[0], row)},
                        {"role": "user", "content": render(prompts[1], row)}]
            kw = {"tokenize": True, "add_generation_prompt": True, "return_tensors": "pt"}
            if "qwen3" in model_id.lower():
                kw["enable_thinking"] = False
            token_ids = tokenizer.apply_chat_template(messages, **kw)
            if token_ids.shape[-1] > a.max_input_tokens:
                raise ValueError(f"Prompt for id={row['id']} has {token_ids.shape[-1]} tokens; "
                                 "shorten the prompt or increase --max-input-tokens")
            token_ids = token_ids.to(model.device)
            with torch.inference_mode():
                generated = model.generate(input_ids=token_ids, do_sample=False,
                                           max_new_tokens=a.max_new_tokens,
                                           pad_token_id=tokenizer.pad_token_id)
            raw = tokenizer.decode(generated[0, token_ids.shape[-1]:], skip_special_tokens=True).strip()
            prediction, parse_method = extract_prediction(raw)
            f.write(json.dumps({"id": row["id"], "text": row["text"],
                                "prediction": prediction, "raw_response": raw,
                                "parse_method": parse_method,
                                "quality_warning": response_warning(raw, row["text"])}, ensure_ascii=False) + "\n")
            f.flush()
            if i == 1 or i % 20 == 0 or i == len(rows):
                print(f"{i}/{len(rows)}: {prediction or 'INVALID'}", flush=True)
            del token_ids, generated
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


def report(output, rows):
    import pandas as pd

    with output.open(encoding="utf-8") as f:
        records = {r["id"]: r for line in f if line.strip() for r in [json.loads(line)]}
    # Always attach gold from the current independent gold file, not a stale checkpoint.
    items = [{**records[row["id"]], "gold": row["gold"]}
             for row in rows if row["id"] in records]
    if not items:
        return

    prefix = output.with_suffix("")
    pd.DataFrame(items).to_csv(prefix.with_name(prefix.name + "_predictions.csv"), index=False)
    print(f"Saved {len(items)} predictions to {output}")

    # Gold labels are OPTIONAL. Unlabeled rows still receive model predictions.
    labeled = [r for r in items if r.get("gold") is not None]
    if not labeled:
        print("No gold labels supplied; skipped evaluation metrics.")
        return

    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

    gold = [r["gold"] for r in labeled]
    predicted = [r["prediction"] or "INVALID" for r in labeled]
    labels = list(LABELS) + (["INVALID"] if "INVALID" in predicted else [])
    metrics = {
        "n_total": len(items), "n_labeled": len(labeled),
        "n_unlabeled": len(items) - len(labeled), "invalid": predicted.count("INVALID"),
        "accuracy": float(accuracy_score(gold, predicted)),
        "macro_f1": float(f1_score(gold, predicted, labels=LABELS,
                                   average="macro", zero_division=0)),
        "per_class": classification_report(gold, predicted, labels=LABELS,
                                           output_dict=True, zero_division=0),
    }
    prefix.with_name(prefix.name + "_metrics.json").write_text(json.dumps(metrics, indent=2))

    cm = confusion_matrix(
        gold,
        predicted,
        labels=labels
    )
    cm_df = pd.DataFrame(
        cm,
        index=labels,
        columns=labels
    )
    cm_df.index.name = "Human Gold / LLM Predicted"
    cm_df.to_csv(prefix.with_name(prefix.name + "_confusion.csv"))
    # pd.DataFrame(confusion_matrix(gold, predicted, labels=labels), index=labels,
    #              columns=labels).to_csv(prefix.with_name(prefix.name + "_confusion.csv"))
    print(
        f"Gold-labeled: {len(labeled)}/{len(items)}  "
        f"accuracy={metrics['accuracy']:.4f}  "
        f"macro-F1={metrics['macro_f1']:.4f}"
    )

    fig, ax = plt.subplots(figsize=(10, 8))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels
    )

    display.plot(
        ax=ax,
        xticks_rotation=45,
        values_format="d",
        colorbar=False
    )

    ax.set_title("LLM Discourse Classification Confusion Matrix")
    ax.set_xlabel("LLM Predicted Mode")
    ax.set_ylabel("Human Gold Mode")

    fig.tight_layout()

    confusion_png = prefix.with_name(
        prefix.name + "_confusion.png"
    )

    fig.savefig(confusion_png, dpi=200)
    plt.close(fig)


def main():
    a = arguments()
    if a.repair_csv:
        repair_csv_file(a.repair_csv, a.repair_output)
        return
    if not a.input:
        raise ValueError("--input is required unless --repair-csv is used")
    if not a.reparse_only and not a.prompt_file:
        raise ValueError("--prompt-file is required when running model inference")
    rows = load_rows(a)
    if a.reparse_only:
        system = user = None
        prompt_hash = None
    else:
        system, user, prompt_hash = load_prompt(
            a.prompt_file, auto_include_context=a.auto_include_context
        )
    if a.preview_prompts:
        for row in rows[:3]:
            print(f"\n=== Target {row['id']} (speaker: {row['speaker']}) ===", flush=True)
            print(render(user, row), flush=True)
        print("\nPreview complete; no model loaded.", flush=True)
        return
    a.output_dir.mkdir(parents=True, exist_ok=True)
    # Gold labels are deliberately excluded from the inference fingerprint:
    # updating gold must not require rerunning models.
    inference_rows = [{k: v for k, v in row.items() if k != "gold"} for row in rows]
    data_hash = hashlib.sha256(json.dumps(inference_rows, sort_keys=True).encode()).hexdigest()
    for model_name in a.models:
        model_id = MODELS.get(model_name, model_name)
        slug = re.sub(r"[^\w.-]+", "_", model_id)
        output = a.output_dir / f"{slug}.jsonl"
        metadata_file = a.output_dir / f"{slug}_run.json"
        config = {"model": model_id, "prompt_hash": prompt_hash, "data_hash": data_hash,
                  "max_new_tokens": a.max_new_tokens, "max_input_tokens": a.max_input_tokens,
                  "dtype": a.dtype}
        if a.reparse_only:
            # Offline re-scoring should not load GPUs, models, or require exact
            # reproduction of the old prompt/model metadata.
            reparse_saved_predictions(output)
        else:
            if output.exists():
                if not metadata_file.exists() or json.loads(metadata_file.read_text()) != config:
                    raise ValueError(f"Existing predictions for {model_id} use different settings. "
                                     "Choose a new --output-dir to prevent mixing runs.")
            else:
                metadata_file.write_text(json.dumps(config, indent=2))
            # Repair previously completed rows so old null predictions don't
            # survive when resuming an interrupted run.
            if output.exists():
                reparse_saved_predictions(output)
            infer(model_id, rows, (system, user), output, a)
        report(output, rows)
    print("\nDone. See predictions, metrics, and confusion CSVs in", a.output_dir)


if __name__ == "__main__":
    main()
