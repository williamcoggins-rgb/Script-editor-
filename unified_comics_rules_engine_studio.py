# unified_comics_rules_engine_studio.py
"""
Unified Comics Writing Rules Engine + Studio Mode + REWRITE ENGINE

Modes:
- Critic Mode (default): evaluate an input spec JSON.
- Studio Mode (--generate): generate N rule-aligned comic idea specs from a seed.
- Rewrite Mode (--rewrite): analyze spec, then ACTUALLY rewrite weak content.

Auto-fix modes (structural only):
  --apply-fixes none|safe|suggest

Rewrite Mode (NEW):
  --rewrite [--rewrite-passes N]
  Chains: critic -> targeted rewrite of flagged content -> re-evaluate.
  Rewrites dialogue, hooks, stakes, art directions — produces real text, not placeholders.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum, IntEnum
from typing import Any, Callable, Dict, List, Optional, Tuple
import argparse
import json
import math
import random
import re
import sys
import textwrap


# ---------------------------------------------------------------------
# PRIORITY (higher number = higher priority)
# ---------------------------------------------------------------------


class Priority(IntEnum):
    P0_HARD_CONSTRAINTS = 100
    P1_STORY_FUNCTION = 80
    P2_COMICS_SPECIFICITY = 60
    P3_DRAMA_AND_PACING = 40
    P4_TRANSITIONS = 30
    P5_STRUCTURE_SHAPE = 20
    P6_MEDIUM_CONSTRAINTS = 10
    P7_REWRITE_LOOP = 0


class Level(IntEnum):
    PASS = 0
    NOTE = 1
    WARN = 2
    FAIL = 3


@dataclass
class RuleResult:
    ok: bool
    rule_id: str
    priority: Priority
    level: Level
    message: str
    location: Optional[str] = None
    fixes: List[str] = field(default_factory=list)
    fix_functions: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Rule:
    rule_id: str
    priority: Priority
    description: str
    check: Callable[[Dict[str, Any]], List[RuleResult]]


def make_result(
    *,
    rule_id: str,
    priority: Priority,
    level: Level,
    message: str,
    location: Optional[str] = None,
    fixes: Optional[List[str]] = None,
    fix_functions: Optional[List[str]] = None,
    evidence: Optional[Dict[str, Any]] = None,
) -> RuleResult:
    return RuleResult(
        ok=(level != Level.FAIL),
        rule_id=rule_id,
        priority=priority,
        level=level,
        message=message,
        location=location,
        fixes=list(fixes or []),
        fix_functions=list(fix_functions or []),
        evidence=dict(evidence or {}),
    )


class RulesEngine:
    def __init__(self) -> None:
        self.rules: List[Rule] = []

    def register(self, rule: Rule) -> None:
        self.rules.append(rule)
        self.rules.sort(key=lambda r: int(r.priority), reverse=True)

    def evaluate(
        self,
        spec: Dict[str, Any],
        *,
        stop_on_first_failing_priority: bool = True,
        stop_threshold: Priority = Priority.P0_HARD_CONSTRAINTS,
    ) -> List[RuleResult]:
        results: List[RuleResult] = []
        current_bucket: Optional[Priority] = None
        bucket_failed = False

        for rule in self.rules:
            if current_bucket is None:
                current_bucket = rule.priority
            elif rule.priority != current_bucket:
                if stop_on_first_failing_priority and bucket_failed and current_bucket >= stop_threshold:
                    break
                current_bucket = rule.priority
                bucket_failed = False

            try:
                res = rule.check(spec)
                if isinstance(res, list):
                    results.extend(res)
                    if any(r.level == Level.FAIL for r in res):
                        bucket_failed = True
                else:
                    results.append(
                        make_result(
                            rule_id=f"{rule.rule_id}.BAD_RETURN",
                            priority=rule.priority,
                            level=Level.FAIL,
                            message="Rule returned non-list; expected list[RuleResult].",
                            evidence={"returned_type": str(type(res))},
                        )
                    )
                    bucket_failed = True
            except Exception as e:
                results.append(
                    make_result(
                        rule_id=f"{rule.rule_id}.EXCEPTION",
                        priority=rule.priority,
                        level=Level.FAIL,
                        message=f"Rule crashed: {e}",
                        fixes=["Fix the spec shape for this rule or harden the rule against missing fields."],
                        evidence={"exception": repr(e)},
                    )
                )
                bucket_failed = True

        results.sort(key=lambda rr: (-int(rr.level), -int(rr.priority)))
        return results


# ---------------------------------------------------------------------
# Shared helpers / indexing
# ---------------------------------------------------------------------

_WORD_RE = re.compile(r"\b[\w'']+\b", flags=re.UNICODE)


def word_count(text: str) -> int:
    return len(_WORD_RE.findall(text or ""))


def safe_list(x: Any) -> List[Any]:
    return x if isinstance(x, list) else []


def get_medium_target(spec: Dict[str, Any]) -> str:
    return (spec.get("medium_target") or "comics").strip().lower()


def get_pages(spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    return safe_list(spec.get("pages"))


class SpecIndex:
    def __init__(self, spec: Dict[str, Any]) -> None:
        self.spec = spec
        self.pages: List[Dict[str, Any]] = get_pages(spec)
        self.page_no_to_index: Dict[int, int] = {}
        for idx, p in enumerate(self.pages):
            page_no = p.get("page_no")
            if isinstance(page_no, int) and page_no not in self.page_no_to_index:
                self.page_no_to_index[page_no] = idx

    def ensure_scenes(self) -> List[Dict[str, Any]]:
        scenes = self.spec.get("scenes")
        if isinstance(scenes, list) and scenes:
            return scenes
        if isinstance(scenes, list):
            return scenes

        inferred: List[Dict[str, Any]] = []
        for idx, p in enumerate(self.pages):
            page_no = p.get("page_no", idx + 1)
            inferred.append(
                {
                    "scene_id": f"PAGE_{page_no}",
                    "purpose": "",
                    "entry_hook": "",
                    "exit_hook": "",
                    "next_scene_id": None,
                    "page_no": page_no,
                }
            )
        self.spec["scenes"] = inferred
        return inferred

    def scenes(self) -> List[Dict[str, Any]]:
        scenes = self.spec.get("scenes")
        if isinstance(scenes, list):
            return scenes
        return self.ensure_scenes()


def get_scenes(spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    return SpecIndex(spec).scenes()


def panel_text_entries(panel: Dict[str, Any]) -> List[Dict[str, str]]:
    txt = panel.get("text", [])
    if isinstance(txt, list) and txt and isinstance(txt[0], dict):
        return [t for t in txt if isinstance(t, dict)]
    if isinstance(txt, list) and txt and isinstance(txt[0], str):
        return [{"type": "balloon", "value": s} for s in txt if isinstance(s, str)]
    return []


def estimate_panel_seconds(panel: Dict[str, Any]) -> float:
    base = 1.2
    entries = panel_text_entries(panel)
    total_words = sum(word_count(e.get("value", "")) for e in entries)
    read_seconds = total_words / 3.0
    balloon_count = len(entries)
    switching = max(0, balloon_count - 2) * 0.35
    if total_words == 0:
        if (panel.get("silent_intent") or "").strip():
            return base + 0.8
        return base
    return base + read_seconds + switching


def split_sentences(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    return re.split(r"(?<=[\.\!\?…])\s+", text)


def truncate_words(text: str, n: int) -> str:
    words = _WORD_RE.findall(text or "")
    if len(words) <= n:
        return text.strip()
    return " ".join(words[:n]).rstrip() + "..."


def parse_balloon_ptr(ptr: str) -> Tuple[str, int, int, int]:
    m = re.match(r"pi(\d+)-pn(\d+)-b(\d+)$", ptr)
    if m:
        return ("pi", int(m.group(1)) - 1, int(m.group(2)) - 1, int(m.group(3)) - 1)
    m = re.match(r"p(\d+)-pn(\d+)-b(\d+)$", ptr)
    if m:
        return ("pno", int(m.group(1)), int(m.group(2)) - 1, int(m.group(3)) - 1)
    return ("bad", -1, -1, -1)


# ---------------------------------------------------------------------
# Fix functions (structural mutators — placeholders only)
# ---------------------------------------------------------------------


class FixCapability(str, Enum):
    SAFE = "safe"
    UNSAFE = "unsafe"


FixFn = Callable[..., None]


def fix_add_entry_hook(spec: Dict[str, Any], scene_id: str, hook: str) -> None:
    idx = SpecIndex(spec)
    for s in idx.ensure_scenes():
        if s.get("scene_id") == scene_id:
            s["entry_hook"] = hook
            return


def fix_add_exit_hook(spec: Dict[str, Any], scene_id: str, hook: str) -> None:
    idx = SpecIndex(spec)
    for s in idx.ensure_scenes():
        if s.get("scene_id") == scene_id:
            s["exit_hook"] = hook
            return


def fix_overlap_dialogue_transition(spec: Dict[str, Any], from_scene: str, to_scene: str, bridging_line: str) -> None:
    fix_add_exit_hook(spec, from_scene, bridging_line)
    fix_add_entry_hook(spec, to_scene, bridging_line)


def fix_reduce_balloon_load(panel: Dict[str, Any], target_total_words: int = 35) -> None:
    entries = panel_text_entries(panel)
    joined = " ".join(e.get("value", "") for e in entries)
    words = _WORD_RE.findall(joined)
    if len(words) > target_total_words:
        joined = " ".join(words[:target_total_words]).rstrip() + "..."
    panel["text"] = [{"type": "balloon", "value": joined}]


def fix_add_scene_outcome(spec: Dict[str, Any], scene_id: str, outcome: str) -> None:
    idx = SpecIndex(spec)
    for s in idx.ensure_scenes():
        if s.get("scene_id") == scene_id:
            s["outcome"] = outcome
            return


def fix_tag_page_turn(spec: Dict[str, Any], page_no: int, label: str = "") -> None:
    for p in get_pages(spec):
        if p.get("page_no") == page_no:
            p["page_turn_reveal"] = True
            if label:
                p["reveal_label"] = label
            return


def fix_set_central_conflict(spec: Dict[str, Any], conflict: str, conflict_type: str = "") -> None:
    spec["central_conflict"] = conflict
    if conflict_type:
        spec["central_conflict_type"] = conflict_type


def fix_add_character_role(spec: Dict[str, Any], role: str, name: str) -> None:
    chars = safe_list(spec.get("characters"))
    for c in chars:
        if (c.get("role") or "").lower() == role.lower():
            c["name"] = name
            spec["characters"] = chars
            return
    chars.append({"role": role, "name": name})
    spec["characters"] = chars


def fix_set_protagonist_need(spec: Dict[str, Any], need: str) -> None:
    spec["protagonist_need"] = need


FIX_REGISTRY: Dict[str, Tuple[FixCapability, FixFn]] = {
    "fix_add_entry_hook": (FixCapability.SAFE, fix_add_entry_hook),
    "fix_add_exit_hook": (FixCapability.SAFE, fix_add_exit_hook),
    "fix_add_scene_outcome": (FixCapability.SAFE, fix_add_scene_outcome),
    "fix_tag_page_turn": (FixCapability.SAFE, fix_tag_page_turn),
    "fix_set_central_conflict": (FixCapability.SAFE, fix_set_central_conflict),
    "fix_add_character_role": (FixCapability.SAFE, fix_add_character_role),
    "fix_set_protagonist_need": (FixCapability.SAFE, fix_set_protagonist_need),
    "fix_overlap_dialogue_transition": (FixCapability.SAFE, fix_overlap_dialogue_transition),
    "fix_reduce_balloon_load": (FixCapability.UNSAFE, fix_reduce_balloon_load),
}

SAFE_FIX_FUNCS = {name for name, (cap, _) in FIX_REGISTRY.items() if cap == FixCapability.SAFE}


# ---------------------------------------------------------------------
# SUGGESTION GENERATORS (original)
# ---------------------------------------------------------------------


def suggest_entry_hook_options(spec: Dict[str, Any], scene: Dict[str, Any]) -> List[str]:
    who = (spec.get("protagonist") or "Protagonist")
    goal = (spec.get("protagonist_goal") or "a clear goal")
    prem = (spec.get("premise") or "").strip()
    purpose = (scene.get("purpose") or "purpose beat").lower()
    return [
        f"[Where/When] -- {who} steps into change: a new {purpose}.",
        f"After the break: {who} chases {goal}, but something's off.",
        f"Meanwhile, {who} finds the cost of {goal} just went up.",
        f"[Location card] -- the moment *after* the status quo cracks.",
        f"A simple task turns sharp: {who} is forced to choose.",
        f"Orientation: we're here because {prem or 'the situation shifted'}.",
    ]


def suggest_exit_hook_options(spec: Dict[str, Any], scene: Dict[str, Any]) -> List[str]:
    who = (spec.get("protagonist") or "Protagonist")
    stakes = (spec.get("stakes") or "the cost rises")
    nxt = (scene.get("next_scene_id") or "next scene")
    return [
        f"A threat lands -- move or lose: {stakes}.",
        f"{who} decides, knowing the price isn't paid yet.",
        "A question with teeth: what happens if they're wrong?",
        f"The door opens on trouble; {nxt} can't be avoided.",
        "The clock starts; now every beat hurts.",
        "A reveal flips the board; the only way out is through.",
    ]


def suggest_balloon_options(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    sents = split_sentences(text)
    words = _WORD_RE.findall(text)
    mid = max(6, min(len(words) - 6, math.floor(len(words) / 2))) if len(words) > 12 else max(1, len(words) // 2)
    o1 = f"BALLOON 1: {truncate_words(text, mid)} | BALLOON 2: {truncate_words(' '.join(words[mid:]), 18)}"
    if len(sents) >= 2:
        o2 = f"BALLOON 1: {sents[0]} | BALLOON 2: {' '.join(sents[1:])}"
    else:
        o2 = f"BALLOON 1: {truncate_words(text, 14)} | BALLOON 2: {truncate_words(' '.join(words[14:]), 16)}"
    o3 = f"BALLOON: {truncate_words(text, 14)} | CAPTION: {truncate_words(' '.join(words[14:]), 18)}"
    o4 = f"BALLOON 1: {truncate_words(text, 12)} | BALLOON 2: {truncate_words(' '.join(words[12:]), 12)}"
    o5 = f"BALLOON 1: {truncate_words(text, 8)} | BALLOON 2: {truncate_words(' '.join(words[8:]), 10)}"
    o6 = f"BALLOON: {truncate_words(text, 10)} | NOTE: move remaining info to earlier/later beat."
    return [o1, o2, o3, o4, o5, o6]


def suggest_stakes_options(spec: Dict[str, Any]) -> List[str]:
    goal = (spec.get("protagonist_goal") or "the goal")
    stakes = (spec.get("stakes") or "").strip()
    who = (spec.get("protagonist") or "The protagonist")
    base = [
        f"If {who.lower()} fails to {goal}, someone gets hurt -- name who & how.",
        f"If {goal} fails, **freedom** is lost (spell out jail/job/blacklist).",
        f"If {goal} fails, **reputation** burns (what doors close?).",
        f"If {goal} fails, **relationship** snaps (who walks?).",
        f"If {goal} fails, **time** runs out (what deadline?).",
        f"If {goal} fails, **identity** crumbles (what belief dies?).",
    ]
    if stakes:
        base.insert(0, f"Refine: If {who} fails to {goal}, then {stakes}. Make the loss concrete.")
    return base


def ensure_suggestions_bucket(spec: Dict[str, Any]) -> Dict[str, Any]:
    if "suggestions" not in spec or not isinstance(spec["suggestions"], dict):
        spec["suggestions"] = {}
    return spec["suggestions"]



# ---------------------------------------------------------------------
# RULES (all rule check functions)
# ---------------------------------------------------------------------


def rule_spec_schema_shape(spec: Dict[str, Any]) -> List[RuleResult]:
    problems: List[str] = []
    if not isinstance(spec, dict):
        return [
            make_result(
                rule_id="P0.SPEC_SCHEMA_SHAPE", priority=Priority.P0_HARD_CONSTRAINTS,
                level=Level.FAIL, message="Spec must be a JSON object (dict).",
                evidence={"received_type": str(type(spec))},
            )
        ]

    pages = spec.get("pages")
    if pages is None:
        problems.append("Missing required key: pages (list).")
    elif not isinstance(pages, list):
        problems.append(f"pages must be a list; got {type(pages)}.")
    else:
        for pi, page in enumerate(pages, start=1):
            if not isinstance(page, dict):
                problems.append(f"pages[{pi}] must be an object; got {type(page)}.")
                continue
            page_no = page.get("page_no")
            if page_no is not None and not isinstance(page_no, int):
                problems.append(f"pages[{pi}].page_no must be int; got {type(page_no)}.")
            panels = page.get("panels")
            if panels is None:
                problems.append(f"pages[{pi}] missing panels (list).")
                continue
            if not isinstance(panels, list):
                problems.append(f"pages[{pi}].panels must be a list; got {type(panels)}.")
                continue
            for pj, panel in enumerate(panels, start=1):
                if not isinstance(panel, dict):
                    problems.append(f"pages[{pi}].panels[{pj}] must be an object; got {type(panel)}.")
                    continue
                has_any_content = (
                    bool((panel.get("art") or "").strip())
                    or bool(safe_list(panel.get("beats")))
                    or bool(safe_list(panel.get("text")))
                    or bool(safe_list(panel.get("dialogue")))
                )
                if not has_any_content:
                    problems.append(f"pages[{pi}].panels[{pj}] has no content (art/beats/text/dialogue).")

    if problems:
        return [make_result(
            rule_id="P0.SPEC_SCHEMA_SHAPE", priority=Priority.P0_HARD_CONSTRAINTS,
            level=Level.FAIL, message="Spec schema/shape problems detected.",
            fixes=["Ensure spec is a JSON object with pages: [ {page_no:int, panels:[{art:str, text:list}]} ]."],
            evidence={"problems": problems[:50]},
        )]
    return [make_result(
        rule_id="P0.SPEC_SCHEMA_SHAPE", priority=Priority.P0_HARD_CONSTRAINTS,
        level=Level.PASS, message="Spec schema shape looks sane.",
    )]


def rule_script_min_fields(spec: Dict[str, Any]) -> List[RuleResult]:
    pages = get_pages(spec)
    if not pages:
        return [make_result(
            rule_id="P0.SCRIPT_MIN_FIELDS", priority=Priority.P0_HARD_CONSTRAINTS,
            level=Level.FAIL, message="Script missing pages.",
            fixes=["Provide pages: [{page_no, page_type, panels:[{art, text:[...]}]}]."],
        )]
    problems: List[str] = []
    for i, page in enumerate(pages, start=1):
        panels = safe_list(page.get("panels"))
        if not panels:
            problems.append(f"Page {page.get('page_no', i)}: no panels.")
            continue
        for j, panel in enumerate(panels, start=1):
            if not (panel.get("art") or "").strip():
                problems.append(f"Page {page.get('page_no', i)}, Panel {j}: missing art direction.")
            txt = panel.get("text", [])
            if txt is not None and not isinstance(txt, list):
                problems.append(f"Page {page.get('page_no', i)}, Panel {j}: text must be a list.")
    if problems:
        return [make_result(
            rule_id="P0.SCRIPT_MIN_FIELDS", priority=Priority.P0_HARD_CONSTRAINTS,
            level=Level.FAIL, message="Script minimum field problems detected.",
            fixes=["Add art direction to every panel.", "Store dialogue/captions/sfx as a list."],
            evidence={"problems": problems[:80]},
        )]
    return [make_result(rule_id="P0.SCRIPT_MIN_FIELDS", priority=Priority.P0_HARD_CONSTRAINTS, level=Level.PASS, message="Script minimum fields pass.")]


def rule_hybrid_language(spec: Dict[str, Any]) -> List[RuleResult]:
    pages = get_pages(spec)
    if not pages:
        return [make_result(rule_id="P0.HYBRID_LANGUAGE", priority=Priority.P0_HARD_CONSTRAINTS, level=Level.FAIL, message="No pages provided.")]
    fails: List[str] = []
    for i, page in enumerate(pages, start=1):
        for j, panel in enumerate(safe_list(page.get("panels")), start=1):
            art = (panel.get("art") or "").strip()
            entries = panel_text_entries(panel)
            has_text = len(entries) > 0
            silent_intent = (panel.get("silent_intent") or "").strip()
            if not art and has_text:
                fails.append(f"Page {page.get('page_no', i)}, Panel {j}: text but no art.")
            if art and (not has_text) and (not silent_intent):
                fails.append(f"Page {page.get('page_no', i)}, Panel {j}: art but no text/silent_intent.")
    if fails:
        return [make_result(
            rule_id="P0.HYBRID_LANGUAGE", priority=Priority.P0_HARD_CONSTRAINTS,
            level=Level.FAIL, message="Word/image integration failures.",
            fixes=["Add silent_intent for art-only panels; add art for text-only panels."],
            evidence={"failures": fails[:80]},
        )]
    return [make_result(rule_id="P0.HYBRID_LANGUAGE", priority=Priority.P0_HARD_CONSTRAINTS, level=Level.PASS, message="Word/image integration passes.")]


def rule_page_panel_sanity(spec: Dict[str, Any]) -> List[RuleResult]:
    pages = get_pages(spec)
    if not pages:
        return [make_result(rule_id="P0.PAGE_SANITY", priority=Priority.P0_HARD_CONSTRAINTS, level=Level.FAIL, message="No pages provided.")]
    problems: List[str] = []
    for i, page in enumerate(pages, start=1):
        panels = safe_list(page.get("panels"))
        page_type = (page.get("page_type") or "normal").strip().lower()
        count = len(panels)
        if page_type in {"splash", "full_page_shot"} and count > 3:
            problems.append(f"Page {page.get('page_no', i)}: {page_type} too many panels ({count}).")
        if page_type == "normal" and count == 0:
            problems.append(f"Page {page.get('page_no', i)}: normal page has zero panels.")
        if count > 9:
            problems.append(f"Page {page.get('page_no', i)}: too many panels ({count}).")
    if problems:
        return [make_result(rule_id="P0.PAGE_SANITY", priority=Priority.P0_HARD_CONSTRAINTS, level=Level.FAIL, message="Page structure problems.", evidence={"problems": problems[:80]})]
    return [make_result(rule_id="P0.PAGE_SANITY", priority=Priority.P0_HARD_CONSTRAINTS, level=Level.PASS, message="Page structure passes.")]


def rule_story_definition(spec: Dict[str, Any]) -> List[RuleResult]:
    premise = (spec.get("premise") or "").strip()
    protagonist = (spec.get("protagonist") or "").strip()
    change = (spec.get("character_change") or "").strip()
    proposition = (spec.get("proposition") or "").strip()
    target_emotion = (spec.get("target_emotion") or "").strip()
    missing = []
    if not premise:
        missing.append("premise")
    if not protagonist:
        missing.append("protagonist")
    if not (change or proposition or target_emotion):
        missing.append("one of: character_change | proposition | target_emotion")
    if missing:
        return [make_result(rule_id="P1.STORY_DEFINITION", priority=Priority.P1_STORY_FUNCTION, level=Level.FAIL, message=f"Missing: {', '.join(missing)}.")]
    return [make_result(rule_id="P1.STORY_DEFINITION", priority=Priority.P1_STORY_FUNCTION, level=Level.PASS, message="Story definition satisfied.")]


def rule_structure_has_consequences(spec: Dict[str, Any]) -> List[RuleResult]:
    beats = safe_list(spec.get("beats"))
    if len(beats) < 4:
        return [make_result(rule_id="P1.STORY_STRUCTURE_MIN", priority=Priority.P1_STORY_FUNCTION, level=Level.FAIL, message="Need at least 4 beats.")]
    hollow = [b for b in beats if not (b.get("change") or b.get("new_problem") or b.get("cost"))]
    if hollow:
        return [make_result(rule_id="P1.STORY_STRUCTURE_CONSEQUENCE", priority=Priority.P1_STORY_FUNCTION, level=Level.FAIL, message="Some beats lack change/cost/new_problem.", evidence={"hollow_beats_count": len(hollow)})]
    return [make_result(rule_id="P1.STORY_STRUCTURE_CONSEQUENCE", priority=Priority.P1_STORY_FUNCTION, level=Level.PASS, message="Beats carry consequences.")]


def rule_creating_drama(spec: Dict[str, Any]) -> List[RuleResult]:
    goal = (spec.get("protagonist_goal") or "").strip()
    obstacles = safe_list(spec.get("obstacles"))
    stakes = (spec.get("stakes") or "").strip()
    missing = []
    if not goal:
        missing.append("protagonist_goal")
    if len(obstacles) < 2:
        missing.append(">=2 obstacles")
    if not stakes:
        missing.append("stakes")
    if missing:
        return [make_result(rule_id="P2.CREATING_DRAMA", priority=Priority.P2_COMICS_SPECIFICITY, level=Level.FAIL, message=f"Missing: {', '.join(missing)}.")]
    return [make_result(rule_id="P2.CREATING_DRAMA", priority=Priority.P2_COMICS_SPECIFICITY, level=Level.PASS, message="Drama fundamentals present.")]


def rule_characterization_choice(spec: Dict[str, Any]) -> List[RuleResult]:
    moments = safe_list(spec.get("character_reveals"))
    if len(moments) < 1:
        return [make_result(rule_id="P2.CHARACTERIZATION_CHOICE", priority=Priority.P2_COMICS_SPECIFICITY, level=Level.FAIL, message="Need >=1 character reveal moment.")]
    weak = [m for m in moments if not (m.get("pressure") and m.get("choice") and m.get("cost"))]
    if weak:
        return [make_result(rule_id="P2.CHARACTERIZATION_CHOICE", priority=Priority.P2_COMICS_SPECIFICITY, level=Level.FAIL, message="Some reveals missing pressure/choice/cost.", evidence={"weak_reveals": len(weak)})]
    return [make_result(rule_id="P2.CHARACTERIZATION_CHOICE", priority=Priority.P2_COMICS_SPECIFICITY, level=Level.PASS, message="Character reveals are choice-driven.")]


def rule_sloane_absorption_hooks(spec: Dict[str, Any]) -> List[RuleResult]:
    idx = SpecIndex(spec)
    scenes = idx.scenes()
    if not scenes:
        return [make_result(rule_id="SLOANE.SCENES_RECOMMENDED", priority=Priority.P6_MEDIUM_CONSTRAINTS, level=Level.NOTE, message="No scenes found.")]
    results: List[RuleResult] = []
    for s in scenes:
        sid = s.get("scene_id") or "UNKNOWN"
        if not (s.get("purpose") or "").strip():
            results.append(make_result(rule_id="SLOANE.PURPOSE_REQUIRED", priority=Priority.P1_STORY_FUNCTION, level=Level.FAIL, message="Scene purpose missing.", location=f"scene:{sid}"))
        if not (s.get("entry_hook") or "").strip():
            results.append(make_result(rule_id="SLOANE.ENTRY_HOOK", priority=Priority.P4_TRANSITIONS, level=Level.WARN, message="Entry hook missing.", location=f"scene:{sid}", fix_functions=["fix_add_entry_hook"]))
        if not (s.get("exit_hook") or "").strip():
            results.append(make_result(rule_id="SLOANE.EXIT_PROPULSION", priority=Priority.P4_TRANSITIONS, level=Level.WARN, message="Exit hook missing.", location=f"scene:{sid}", fix_functions=["fix_add_exit_hook"]))
    return results or [make_result(rule_id="SLOANE.ABSORPTION_HOOKS", priority=Priority.P4_TRANSITIONS, level=Level.PASS, message="Scene hooks ok.")]


def rule_scene_outcome_direction(spec: Dict[str, Any]) -> List[RuleResult]:
    scenes = get_scenes(spec)
    missing = [s.get("scene_id") for s in scenes if not (s.get("outcome") or "").strip()]
    if missing:
        return [make_result(rule_id="SCENE.OUTCOME_DIRECTION", priority=Priority.P4_TRANSITIONS, level=Level.WARN, message="Scene outcomes missing.", evidence={"scenes": missing[:50]}, fix_functions=["fix_add_scene_outcome"])]
    return [make_result(rule_id="SCENE.OUTCOME_DIRECTION", priority=Priority.P4_TRANSITIONS, level=Level.PASS, message="Scene outcomes present.")]


def rule_comics_grid_page_turns(spec: Dict[str, Any]) -> List[RuleResult]:
    pages = get_pages(spec)
    if len(pages) < 3:
        return []
    flagged = [p for p in pages if p.get("page_turn_reveal")]
    if not flagged:
        return [make_result(rule_id="GRID.PAGE_TURN_REVEALS", priority=Priority.P4_TRANSITIONS, level=Level.WARN, message="No page-turn reveal tagged.", fix_functions=["fix_tag_page_turn"])]
    return [make_result(rule_id="GRID.PAGE_TURN_REVEALS", priority=Priority.P4_TRANSITIONS, level=Level.PASS, message="Page-turn reveal tagged.")]


def rule_conflict_four_levels(spec: Dict[str, Any]) -> List[RuleResult]:
    scenes = get_scenes(spec)
    obstacles = safe_list(spec.get("obstacles"))
    internal_ok = bool((spec.get("character_change") or spec.get("internal_conflict") or spec.get("target_emotion")))
    central_ok = bool((spec.get("central_conflict") or spec.get("central_conflict_type")))
    micro_ok = all(bool(s.get("conflict")) for s in scenes) if scenes else False
    macro_ok = len(obstacles) >= 2
    missing = []
    if not central_ok:
        missing.append("central conflict")
    if not macro_ok:
        missing.append("macro obstacles")
    if not micro_ok:
        missing.append("micro conflicts")
    if not internal_ok:
        missing.append("internal conflict")
    if missing:
        return [make_result(rule_id="CONFLICT.FOUR_LEVELS", priority=Priority.P2_COMICS_SPECIFICITY, level=Level.WARN, message="Conflict coverage incomplete.", evidence={"missing": missing}, fix_functions=["fix_set_central_conflict"])]
    return [make_result(rule_id="CONFLICT.FOUR_LEVELS", priority=Priority.P2_COMICS_SPECIFICITY, level=Level.PASS, message="Conflict coverage ok.")]


def rule_dcosta_inciting_incident_timing(spec: Dict[str, Any]) -> List[RuleResult]:
    beats = safe_list(spec.get("beats"))
    inciting_indexes = []
    for idx, b in enumerate(beats):
        name = (b.get("name") or "").lower()
        if b.get("new_problem") or ("inciting" in name) or ("disturbance" in name):
            inciting_indexes.append(idx)
    if not any(i <= 1 for i in inciting_indexes):
        return [make_result(rule_id="DCOSTA.INCITING_TIMING", priority=Priority.P1_STORY_FUNCTION, level=Level.FAIL, message="Inciting incident not clearly early.", evidence={"inciting_indexes": inciting_indexes})]
    return [make_result(rule_id="DCOSTA.INCITING_TIMING", priority=Priority.P1_STORY_FUNCTION, level=Level.PASS, message="Inciting incident early enough.")]


_STAKE_KEYWORDS = {"life", "death", "freedom", "jail", "reputation", "family", "job", "mission", "city", "world", "identity", "time", "deadline"}


def rule_dcosta_stakes_specificity(spec: Dict[str, Any]) -> List[RuleResult]:
    s = (spec.get("stakes") or "").strip()
    wc = word_count(s)
    hits = {w for w in _STAKE_KEYWORDS if re.search(rf"\b{re.escape(w)}\b", s.lower())}
    if wc < 8 or len(hits) == 0:
        level = Level.FAIL if wc < 8 else Level.WARN
        return [make_result(rule_id="DCOSTA.STAKES_SPECIFICITY", priority=Priority.P2_COMICS_SPECIFICITY, level=level, message="Stakes too vague.", evidence={"word_count": wc, "keyword_hits": sorted(hits)})]
    return [make_result(rule_id="DCOSTA.STAKES_SPECIFICITY", priority=Priority.P2_COMICS_SPECIFICITY, level=Level.PASS, message="Stakes specific enough.")]


def rule_myers_family_roles(spec: Dict[str, Any]) -> List[RuleResult]:
    chars = safe_list(spec.get("characters"))
    roles_present = {(c.get("role") or "").lower() for c in chars}
    coverage = len(roles_present.intersection({"protagonist", "nemesis", "mentor", "attractor", "trickster"}))
    if coverage < 3:
        return [make_result(rule_id="MYERS.FAMILY_ROLES", priority=Priority.P1_STORY_FUNCTION, level=Level.WARN, message="Too few family roles.", evidence={"roles_present": sorted(roles_present)}, fix_functions=["fix_add_character_role"])]
    return [make_result(rule_id="MYERS.FAMILY_ROLES", priority=Priority.P1_STORY_FUNCTION, level=Level.PASS, message="Family roles ok.")]


def rule_myers_unity_arc(spec: Dict[str, Any]) -> List[RuleResult]:
    need = (spec.get("protagonist_need") or spec.get("internal_conflict") or "").strip()
    if not need:
        return [make_result(rule_id="MYERS.UNITY_ARC", priority=Priority.P1_STORY_FUNCTION, level=Level.WARN, message="No protagonist_need / unity arc.", fix_functions=["fix_set_protagonist_need"])]
    return [make_result(rule_id="MYERS.UNITY_ARC", priority=Priority.P1_STORY_FUNCTION, level=Level.PASS, message="Unity arc hint present.")]


# PRIEST PACK

_FAKE_DIALECT_RE = re.compile(r"\b(ah|muh|mah)\b", flags=re.IGNORECASE)


def rule_priest_dialogue_cleanliness(spec: Dict[str, Any]) -> List[RuleResult]:
    pages = get_pages(spec)
    violations: List[str] = []
    for i, page in enumerate(pages, 1):
        for j, panel in enumerate(safe_list(page.get("panels")), 1):
            for e in panel_text_entries(panel):
                val = (e.get("value") or "")
                if _FAKE_DIALECT_RE.search(val):
                    violations.append(f"page:{page.get('page_no', i)} panel:{j}")
    if violations:
        return [RuleResult(False, "PRIEST.DIALOGUE_CLEANLINESS", Priority.P0_HARD_CONSTRAINTS, Level.FAIL,
            "Fake dialect crutches detected.", fixes=["Rewrite dialect using rhythm/syntax, not misspelling."],
            evidence={"violations": violations[:40]})]
    return [RuleResult(True, "PRIEST.DIALOGUE_CLEANLINESS", Priority.P0_HARD_CONSTRAINTS, Level.PASS, "Dialogue cleanliness OK.")]


def rule_priest_copy_heavy(spec: Dict[str, Any]) -> List[RuleResult]:
    pages = get_pages(spec)
    heavy: List[str] = []
    for i, page in enumerate(pages, 1):
        for j, panel in enumerate(safe_list(page.get("panels")), 1):
            sec = estimate_panel_seconds(panel)
            if sec > 12.0:
                heavy.append(f"page:{page.get('page_no', i)} panel:{j} ({round(sec,2)}s)")
    if heavy:
        return [RuleResult(True, "PRIEST.COPY_HEAVY", Priority.P0_HARD_CONSTRAINTS, Level.WARN,
            "Copy-heavy panels detected.", fixes=["Split panel or trim balloons."],
            fix_functions=["fix_reduce_balloon_load"], evidence={"panels": heavy[:30]})]
    return [RuleResult(True, "PRIEST.COPY_HEAVY", Priority.P0_HARD_CONSTRAINTS, Level.PASS, "Copy density acceptable.")]


def rule_priest_layout_tricks(spec: Dict[str, Any]) -> List[RuleResult]:
    pages = get_pages(spec)
    bad_pages: List[Any] = []
    for i, page in enumerate(pages, 1):
        if (page.get("layout") or "").strip().lower() in {"trick", "gimmick", "pretentious"}:
            bad_pages.append(page.get("page_no", i))
    if bad_pages:
        return [RuleResult(True, "PRIEST.LAYOUT_TRICKS", Priority.P0_HARD_CONSTRAINTS, Level.WARN,
            "Trick/gimmick layouts flagged.", evidence={"pages": bad_pages})]
    return [RuleResult(True, "PRIEST.LAYOUT_TRICKS", Priority.P0_HARD_CONSTRAINTS, Level.PASS, "Layouts clean.")]


def rule_priest_interior_splash(spec: Dict[str, Any]) -> List[RuleResult]:
    pages = get_pages(spec)
    if len(pages) < 3:
        return []
    interior = []
    for idx, page in enumerate(pages, 1):
        pt = (page.get("page_type") or "normal").strip().lower()
        if pt == "splash" and idx not in {1, len(pages)}:
            interior.append(page.get("page_no", idx))
    if interior:
        return [RuleResult(True, "PRIEST.INTERIOR_SPLASH", Priority.P2_COMICS_SPECIFICITY, Level.WARN,
            "Interior splash pages flagged.", evidence={"pages": interior})]
    return [RuleResult(True, "PRIEST.INTERIOR_SPLASH", Priority.P2_COMICS_SPECIFICITY, Level.PASS, "Splash placement OK.")]


def rule_priest_mindless_violence(spec: Dict[str, Any]) -> List[RuleResult]:
    beats = safe_list(spec.get("beats"))
    flagged = []
    for b in beats:
        text = (b.get("name", "") + " " + b.get("change", "") + " " + b.get("new_problem", "")).lower()
        if any(k in text for k in ["fight", "punch", "shoot", "kill", "battle", "attack"]):
            if not (b.get("cost") or b.get("change") or b.get("new_problem")):
                flagged.append(b)
    if flagged:
        return [RuleResult(False, "PRIEST.MINDLESS_VIOLENCE", Priority.P1_STORY_FUNCTION, Level.FAIL,
            "Violence without narrative consequence.", evidence={"beats": flagged[:10]})]
    return [RuleResult(True, "PRIEST.MINDLESS_VIOLENCE", Priority.P1_STORY_FUNCTION, Level.PASS, "Violence has consequence (or none detected).")]


def rule_priest_world_representation(spec: Dict[str, Any]) -> List[RuleResult]:
    rep = (spec.get("representation_notes") or "").strip()
    if not rep:
        return [RuleResult(True, "PRIEST.REPRESENTATION_NOTES", Priority.P1_STORY_FUNCTION, Level.NOTE,
            "Add representation_notes to avoid a flattened world.")]
    return [RuleResult(True, "PRIEST.REPRESENTATION_NOTES", Priority.P1_STORY_FUNCTION, Level.PASS, "Representation notes present.")]


# STAN LEE PACK

def rule_stan_onboarding_clarity(spec: Dict[str, Any]) -> List[RuleResult]:
    premise = (spec.get("premise") or "").strip()
    stakes = (spec.get("stakes") or "").strip()
    if len(premise.split()) < 10 or not stakes:
        return [RuleResult(True, "STAN.ONBOARDING_CLARITY", Priority.P1_STORY_FUNCTION, Level.WARN,
            "Onboarding clarity is thin.")]
    return [RuleResult(True, "STAN.ONBOARDING_CLARITY", Priority.P1_STORY_FUNCTION, Level.PASS, "Onboarding clarity OK.")]


def rule_stan_conflict_per_scene(spec: Dict[str, Any]) -> List[RuleResult]:
    scenes = get_scenes(spec)
    if not scenes:
        return []
    missing = [s.get("scene_id") or "UNKNOWN" for s in scenes if not (s.get("conflict") or "").strip()]
    if missing:
        return [RuleResult(True, "STAN.CONFLICT_PER_SCENE", Priority.P1_STORY_FUNCTION, Level.WARN,
            "Some scenes lack explicit conflict.", evidence={"scenes": missing})]
    return [RuleResult(True, "STAN.CONFLICT_PER_SCENE", Priority.P1_STORY_FUNCTION, Level.PASS, "Conflict present per scene.")]


def rule_stan_character_driven_beats(spec: Dict[str, Any]) -> List[RuleResult]:
    beats = safe_list(spec.get("beats"))
    if not beats:
        return []
    weak = [b for b in beats if not (b.get("character") or b.get("decision") or b.get("want"))]
    if weak:
        return [RuleResult(True, "STAN.CHARACTER_DRIVEN_BEATS", Priority.P1_STORY_FUNCTION, Level.WARN,
            "Some beats not tied to character want/decision.", evidence={"count": len(weak)})]
    return [RuleResult(True, "STAN.CHARACTER_DRIVEN_BEATS", Priority.P1_STORY_FUNCTION, Level.PASS, "Beats are character-driven.")]


# MOORE PACK

def rule_moore_panel_time_stoppers(spec: Dict[str, Any]) -> List[RuleResult]:
    pages = get_pages(spec)
    if not pages:
        return []
    stoppers = []
    for i, page in enumerate(pages, 1):
        for j, panel in enumerate(safe_list(page.get("panels")), 1):
            sec = estimate_panel_seconds(panel)
            if sec > 10.0 and not (panel.get("contemplative") is True):
                stoppers.append({"page": page.get("page_no", i), "panel": j, "seconds": round(sec, 2)})
    if stoppers:
        return [RuleResult(True, "MOORE.PANEL_TIME_STOPPERS", Priority.P3_DRAMA_AND_PACING, Level.WARN,
            "Overlong panels detected.", fix_functions=["fix_reduce_balloon_load"], evidence={"stoppers": stoppers[:25]})]
    return [RuleResult(True, "MOORE.PANEL_TIME_STOPPERS", Priority.P3_DRAMA_AND_PACING, Level.PASS, "No panel-time stoppers.")]


def rule_moore_transition_glue(spec: Dict[str, Any]) -> List[RuleResult]:
    scenes = get_scenes(spec)
    if not scenes:
        return []
    by_id = {s.get("scene_id"): s for s in scenes if s.get("scene_id")}
    missing = []
    for s in scenes:
        sid = s.get("scene_id")
        nxt_id = s.get("next_scene_id")
        if not sid or not nxt_id:
            continue
        nxt = by_id.get(nxt_id)
        if not nxt:
            continue
        if not (s.get("exit_hook") or "").strip() and not (nxt.get("entry_hook") or "").strip():
            missing.append(f"{sid}->{nxt_id}")
    if missing:
        return [RuleResult(False, "MOORE.TRANSITION_GLUE", Priority.P4_TRANSITIONS, Level.FAIL,
            "Scene transitions lack glue.", fix_functions=["fix_overlap_dialogue_transition"],
            evidence={"transitions": missing[:30]})]
    return [RuleResult(True, "MOORE.TRANSITION_GLUE", Priority.P4_TRANSITIONS, Level.PASS, "Transitions have glue.")]


# MEDIUM + REWRITE LOOP

def rule_medium_constraints(spec: Dict[str, Any]) -> List[RuleResult]:
    target = get_medium_target(spec)
    results: List[RuleResult] = []
    results.append(RuleResult(True, "COMICS.ALLOTTED_READING_TIME", Priority.P6_MEDIUM_CONSTRAINTS, Level.NOTE,
        "Match balloon load to intended rhythm."))
    if target == "film_adaptation":
        results.append(RuleResult(True, "ADAPT.LAYOUT_TRANSLATION", Priority.P6_MEDIUM_CONSTRAINTS, Level.WARN,
            "Film adaptation: page-turn reveals need re-authoring for screen."))
        results.append(RuleResult(True, "ADAPT.SILENCE_TO_SOUND", Priority.P6_MEDIUM_CONSTRAINTS, Level.WARN,
            "Film adaptation: comics silence is pacing; film sound can over-explain."))
    return results


def rule_rewrite_loop(spec: Dict[str, Any]) -> List[RuleResult]:
    return [RuleResult(True, "PROCESS.REWRITE_LOOP", Priority.P7_REWRITE_LOOP, Level.NOTE,
        "Plan at least one rewrite pass focused on reader absorption and clarity.")]


# =====================================================================
# PANEL ARCHITECTURE ENGINE
# =====================================================================
#
# Panels are not containers for dialogue. They are units of narrative
# physics. You are bending time, focus, and emotional gravity.
#
# Every page must answer three questions:
#   1. What is the emotional objective?
#   2. What is the tempo?
#   3. Where is the page-turn payoff?
#
# If those aren't clear, the page is structural filler.
#
# Based on O'Neil's craft framework:
#   - Panel count = time compression or expansion
#   - Panel size = emotional weight
#   - Page turns = weaponized information
#   - Layout = eye direction engineering
#   - Shot composition = controlled perception
# =====================================================================


class PanelTransition(str, Enum):
    """Scott McCloud's panel transition taxonomy.

    Each transition type controls how the reader's brain fills the gutter
    (the space between panels). The choice is not decorative — it determines
    how much cognitive work the reader does and how time is perceived.
    """
    MOMENT_TO_MOMENT = "moment_to_moment"       # Minimal change, slows time (blinking, turning head)
    ACTION_TO_ACTION = "action_to_action"        # Single subject, clear progression (punch → impact)
    SUBJECT_TO_SUBJECT = "subject_to_subject"    # Cuts between subjects in same scene (speaker A → B)
    SCENE_TO_SCENE = "scene_to_scene"            # Jump in time/space (requires reader inference)
    ASPECT_TO_ASPECT = "aspect_to_aspect"        # Wandering eye, mood/atmosphere (clock, rain, face)
    NON_SEQUITUR = "non_sequitur"                # No logical relationship (rare, experimental)


class PanelWeight(str, Enum):
    """How much visual real estate a panel commands on the page.

    A larger panel implies importance. A small panel implies transitional
    motion or a minor beat. If everything is the same size, nothing is
    emphasized — and that's a choice too (steady rhythm, metronomic pacing).
    """
    QUARTER = "quarter"       # Small — transitional, quick beat, reaction shot
    THIRD = "third"           # Standard — steady rhythm, workhorse panel
    HALF = "half"             # Prominent — key moment, important dialogue
    TWO_THIRDS = "two_thirds" # Dominant — major reveal, fight climax
    FULL_WIDTH = "full_width" # Banner — establishing shot, dramatic beat
    SPLASH = "splash"         # Full page — monumental, only if stakes justify it
    DOUBLE_SPLASH = "double_splash"  # Two-page spread — nuclear option


class Tempo(str, Enum):
    """Page-level pacing control. Panel count IS tempo.

    More panels on a page slow perceived time (reader lingers).
    Fewer panels accelerate it (reader moves faster, feels urgency).
    """
    GLACIAL = "glacial"         # 7-9 panels: hyper-detailed, moment-by-moment breakdown
    STEADY = "steady"           # 5-6 panels: controlled, standard comics rhythm
    BRISK = "brisk"             # 3-4 panels: accelerating, dramatic compression
    EXPLOSIVE = "explosive"     # 1-2 panels: monumental, splash territory
    DECOMPRESSED = "decompressed"  # 3-4 large panels: slow, emotional, breathing room


class ShotType(str, Enum):
    """Camera framing — not just what is shown, but HOW it is shown.

    Each shot carries emotional meaning independent of content.
    When the emotional meaning depends on framing, you specify it.
    """
    ESTABLISHING = "establishing"    # Wide: where are we? Scale, context, geography
    WIDE = "wide"                    # Full figures, environment visible — scale or isolation
    MEDIUM = "medium"                # Waist up — conversation, neutral, workhorse
    MEDIUM_CLOSE = "medium_close"    # Chest up — engaged, personal
    CLOSE = "close"                  # Face — intimacy or intensity
    EXTREME_CLOSE = "extreme_close"  # Detail — eyes, hands, object — maximum intensity
    OVER_SHOULDER = "over_shoulder"  # Confrontation, tension, power dynamic visible
    LOW_ANGLE = "low_angle"          # Subject looks powerful, dominant, threatening
    HIGH_ANGLE = "high_angle"        # Subject looks vulnerable, small, trapped
    BIRDS_EYE = "birds_eye"          # God's view — shows the whole board, strategic
    WORMS_EYE = "worms_eye"          # Ground level — dramatic, imposing
    DUTCH_ANGLE = "dutch_angle"      # Tilted — disorientation, unease, wrongness
    INSERT = "insert"                # Object detail — the gun, the letter, the clock


class PageLayout(str, Enum):
    """How panels are arranged on the page — cognitive choreography.

    Readers in Western comics read left-to-right, top-to-bottom.
    Layout must guide that flow without confusion. If a reader pauses
    to decode panel order, you have broken immersion.
    """
    GRID_REGULAR = "grid_regular"          # Even grid (2x3, 3x3) — steady, metronomic
    GRID_VARIED = "grid_varied"            # Grid with varied sizes — emphasis within order
    VERTICAL_STACK = "vertical_stack"      # Panels stacked top-to-bottom — descent, falling
    HORIZONTAL_BANDS = "horizontal_bands"  # Wide panels stacked — panoramic, cinematic
    L_SHAPE = "l_shape"                    # One large + smaller panels — anchor + beats
    T_SHAPE = "t_shape"                    # Top row + lower sections — establish then detail
    DIAGONAL = "diagonal"                  # Panels on a diagonal — motion, dynamism
    OVERLAPPING = "overlapping"            # Panels overlap — chaos, simultaneity, urgency
    SPLASH_WITH_INSETS = "splash_insets"   # Full page image + small inset panels
    FREE_FORM = "free_form"               # Panels break the grid — use sparingly


class EmotionalObjective(str, Enum):
    """What the page is doing to the reader's emotional state.

    This is the 'why' of the page. Without a clear objective,
    the page is structural filler.
    """
    ORIENT = "orient"           # Establish location, situation, stakes
    ESCALATE = "escalate"       # Ratchet tension upward
    BREATHE = "breathe"         # Give reader emotional space after intensity
    REVEAL = "reveal"           # Deliver new information that changes everything
    CONFRONT = "confront"       # Characters clash — the scene's engine
    DECIDE = "decide"           # Character makes a choice under pressure
    DEVASTATE = "devastate"     # The cost lands — emotional gut punch
    PROPEL = "propel"           # Pure momentum — move to the next beat fast
    REFLECT = "reflect"         # Quiet beat — character processes what happened
    CLIMAX = "climax"           # The culmination — maximum intensity


# ── Shot Sequence Patterns ────────────────────────────────────────────
# How shots progress within a page to create specific emotional effects.
# These are not random — each pattern is an engineered tension curve.

SHOT_SEQUENCES = {
    "tension_build": [
        ShotType.WIDE, ShotType.MEDIUM, ShotType.MEDIUM_CLOSE,
        ShotType.CLOSE, ShotType.EXTREME_CLOSE,
    ],
    "tension_release": [
        ShotType.EXTREME_CLOSE, ShotType.CLOSE, ShotType.MEDIUM, ShotType.WIDE,
    ],
    "conversation_standard": [
        ShotType.MEDIUM, ShotType.OVER_SHOULDER, ShotType.CLOSE,
        ShotType.OVER_SHOULDER, ShotType.MEDIUM,
    ],
    "conversation_escalating": [
        ShotType.MEDIUM, ShotType.MEDIUM_CLOSE, ShotType.CLOSE,
        ShotType.CLOSE, ShotType.EXTREME_CLOSE,
    ],
    "action_sequence": [
        ShotType.WIDE, ShotType.MEDIUM, ShotType.CLOSE,
        ShotType.INSERT, ShotType.WIDE,
    ],
    "establishing_sequence": [
        ShotType.BIRDS_EYE, ShotType.WIDE, ShotType.MEDIUM,
        ShotType.CLOSE,
    ],
    "reveal_sequence": [
        ShotType.CLOSE, ShotType.CLOSE, ShotType.MEDIUM,
        ShotType.WIDE,  # pull back to show the full picture
    ],
    "horror_dread": [
        ShotType.WIDE, ShotType.MEDIUM, ShotType.CLOSE,
        ShotType.INSERT, ShotType.EXTREME_CLOSE,
    ],
    "power_shift": [
        ShotType.LOW_ANGLE, ShotType.MEDIUM, ShotType.HIGH_ANGLE,
    ],
    "disorientation": [
        ShotType.DUTCH_ANGLE, ShotType.EXTREME_CLOSE,
        ShotType.BIRDS_EYE, ShotType.CLOSE,
    ],
    "aftermath": [
        ShotType.WIDE, ShotType.MEDIUM, ShotType.CLOSE,
        ShotType.WIDE,  # pull out, show the damage
    ],
    "intimacy": [
        ShotType.MEDIUM, ShotType.MEDIUM_CLOSE, ShotType.CLOSE,
        ShotType.CLOSE,
    ],
}


# ── Tempo Rules ───────────────────────────────────────────────────────
# Maps emotional objectives to recommended tempos and panel counts.

TEMPO_MAP: Dict[str, Tuple[Tempo, int, int]] = {
    # objective -> (tempo, min_panels, max_panels)
    "orient":     (Tempo.STEADY, 4, 6),
    "escalate":   (Tempo.BRISK, 3, 5),
    "breathe":    (Tempo.DECOMPRESSED, 2, 4),
    "reveal":     (Tempo.EXPLOSIVE, 1, 3),
    "confront":   (Tempo.STEADY, 4, 6),
    "decide":     (Tempo.BRISK, 3, 4),
    "devastate":  (Tempo.EXPLOSIVE, 1, 2),
    "propel":     (Tempo.BRISK, 4, 6),
    "reflect":    (Tempo.DECOMPRESSED, 2, 4),
    "climax":     (Tempo.EXPLOSIVE, 1, 3),
}

# ── Layout Recommendations ────────────────────────────────────────────
# Maps tempo + panel count to recommended page layouts.

LAYOUT_MAP: Dict[Tuple[Tempo, int], List[PageLayout]] = {
    (Tempo.GLACIAL, 7):    [PageLayout.GRID_REGULAR],
    (Tempo.GLACIAL, 8):    [PageLayout.GRID_REGULAR],
    (Tempo.GLACIAL, 9):    [PageLayout.GRID_REGULAR],
    (Tempo.STEADY, 5):     [PageLayout.GRID_VARIED, PageLayout.T_SHAPE],
    (Tempo.STEADY, 6):     [PageLayout.GRID_REGULAR, PageLayout.GRID_VARIED],
    (Tempo.BRISK, 3):      [PageLayout.HORIZONTAL_BANDS, PageLayout.VERTICAL_STACK],
    (Tempo.BRISK, 4):      [PageLayout.L_SHAPE, PageLayout.GRID_VARIED],
    (Tempo.BRISK, 5):      [PageLayout.GRID_VARIED, PageLayout.T_SHAPE],
    (Tempo.EXPLOSIVE, 1):  [PageLayout.SPLASH_WITH_INSETS],
    (Tempo.EXPLOSIVE, 2):  [PageLayout.HORIZONTAL_BANDS, PageLayout.L_SHAPE],
    (Tempo.DECOMPRESSED, 2): [PageLayout.HORIZONTAL_BANDS],
    (Tempo.DECOMPRESSED, 3): [PageLayout.VERTICAL_STACK, PageLayout.HORIZONTAL_BANDS],
    (Tempo.DECOMPRESSED, 4): [PageLayout.GRID_VARIED, PageLayout.L_SHAPE],
}


# ── Panel Weight Distribution ─────────────────────────────────────────
# Given N panels on a page, how to distribute visual weight.

def distribute_panel_weights(
    panel_count: int,
    objective: str,
    has_key_moment: bool = False,
) -> List[PanelWeight]:
    """Distribute visual weight across panels on a page.

    The key principle: a larger panel implies importance.
    If everything is the same size, nothing is emphasized.
    """
    if panel_count == 1:
        return [PanelWeight.SPLASH]
    if panel_count == 2:
        if has_key_moment:
            return [PanelWeight.THIRD, PanelWeight.TWO_THIRDS]
        return [PanelWeight.HALF, PanelWeight.HALF]

    weights: List[PanelWeight] = []

    if objective in ("reveal", "devastate", "climax"):
        # Key moment gets the biggest panel, others are smaller
        for i in range(panel_count):
            if i == panel_count - 1:  # Last panel = the punch
                weights.append(PanelWeight.HALF if panel_count <= 4 else PanelWeight.TWO_THIRDS)
            else:
                weights.append(PanelWeight.QUARTER if panel_count > 4 else PanelWeight.THIRD)
    elif objective in ("orient", "confront"):
        # First panel establishes, rest are even
        weights.append(PanelWeight.HALF if panel_count <= 4 else PanelWeight.FULL_WIDTH)
        for _ in range(panel_count - 1):
            weights.append(PanelWeight.THIRD if panel_count <= 5 else PanelWeight.QUARTER)
    elif objective in ("breathe", "reflect"):
        # Even distribution, generous sizing
        for _ in range(panel_count):
            weights.append(PanelWeight.HALF if panel_count <= 3 else PanelWeight.THIRD)
    elif objective in ("escalate", "propel"):
        # Progressive compression: panels get smaller = acceleration
        for i in range(panel_count):
            ratio = i / max(1, panel_count - 1)
            if ratio < 0.3:
                weights.append(PanelWeight.HALF)
            elif ratio < 0.7:
                weights.append(PanelWeight.THIRD)
            else:
                weights.append(PanelWeight.QUARTER)
    else:
        # Default: standard grid
        for _ in range(panel_count):
            weights.append(PanelWeight.THIRD)

    return weights


# ── Transition Selection ──────────────────────────────────────────────

def select_panel_transition(
    prev_panel: Optional[Dict[str, Any]],
    curr_panel: Dict[str, Any],
    page_objective: str,
) -> PanelTransition:
    """Determine the transition type between two consecutive panels.

    The transition type controls what happens in the gutter — the space
    between panels where the reader's brain fills in the gap.
    """
    if prev_panel is None:
        return PanelTransition.SCENE_TO_SCENE  # First panel on page

    prev_art = (prev_panel.get("art") or "").lower()
    curr_art = (curr_panel.get("art") or "").lower()

    # Check for scene/location change
    prev_loc = prev_panel.get("location") or ""
    curr_loc = curr_panel.get("location") or ""
    if prev_loc and curr_loc and prev_loc != curr_loc:
        return PanelTransition.SCENE_TO_SCENE

    # Check for subject change (different speakers or focus characters)
    prev_chars = set(safe_list(prev_panel.get("characters", [])))
    curr_chars = set(safe_list(curr_panel.get("characters", [])))
    if prev_chars and curr_chars and not prev_chars.intersection(curr_chars):
        return PanelTransition.SUBJECT_TO_SUBJECT

    # Mood/atmosphere pages
    if page_objective in ("breathe", "reflect"):
        silent_curr = (curr_panel.get("silent_intent") or "").strip()
        if silent_curr:
            return PanelTransition.ASPECT_TO_ASPECT

    # Action scenes
    if page_objective in ("escalate", "propel", "climax"):
        return PanelTransition.ACTION_TO_ACTION

    # Slow, intimate scenes
    if page_objective in ("devastate", "decide"):
        return PanelTransition.MOMENT_TO_MOMENT

    # Conversation default
    return PanelTransition.SUBJECT_TO_SUBJECT


# ── Shot Sequence Selection ───────────────────────────────────────────

def select_shot_sequence(
    objective: str,
    panel_count: int,
    scene_context: Optional[Dict[str, Any]] = None,
) -> List[ShotType]:
    """Pick the right shot progression for a page's emotional objective.

    Shot sequencing is how you engineer tension. Wide → medium → close
    ratchets focus. Close → wide releases it. The sequence is the
    emotional arc of the page itself.
    """
    purpose = ""
    if scene_context:
        purpose = (scene_context.get("purpose") or "").lower()

    # Map objective to sequence name
    seq_name = "tension_build"  # default
    if objective == "orient":
        seq_name = "establishing_sequence"
    elif objective == "escalate":
        seq_name = "tension_build" if "confront" not in purpose else "conversation_escalating"
    elif objective == "breathe":
        seq_name = "tension_release"
    elif objective == "reveal":
        seq_name = "reveal_sequence"
    elif objective == "confront":
        seq_name = "conversation_escalating" if panel_count >= 4 else "power_shift"
    elif objective == "decide":
        seq_name = "intimacy"
    elif objective == "devastate":
        seq_name = "aftermath"
    elif objective == "propel":
        seq_name = "action_sequence"
    elif objective == "reflect":
        seq_name = "tension_release"
    elif objective == "climax":
        seq_name = "tension_build"

    sequence = list(SHOT_SEQUENCES.get(seq_name, SHOT_SEQUENCES["tension_build"]))

    # Adjust sequence length to match panel count
    if len(sequence) > panel_count:
        # Take evenly spaced samples
        step = len(sequence) / panel_count
        sequence = [sequence[min(len(sequence) - 1, int(i * step))] for i in range(panel_count)]
    elif len(sequence) < panel_count:
        # Extend by repeating the last shot type with variation
        while len(sequence) < panel_count:
            sequence.append(sequence[-1])

    return sequence


# ── Page Architecture ─────────────────────────────────────────────────

@dataclass
class PanelArchitecture:
    """Complete architecture for a single panel within a page."""
    panel_index: int
    shot: ShotType
    weight: PanelWeight
    transition_in: PanelTransition
    is_key_moment: bool = False
    silent: bool = False
    notes: str = ""


@dataclass
class PageArchitecture:
    """Complete architecture for a page — the blueprint before content.

    This answers the three questions every page must answer:
    1. What is the emotional objective?
    2. What is the tempo?
    3. Where is the page-turn payoff?
    """
    page_no: int
    objective: EmotionalObjective
    tempo: Tempo
    panel_count: int
    layout: PageLayout
    panels: List[PanelArchitecture]
    page_turn_payoff: str = ""         # What the reader gets for turning to this page
    page_exit_tension: str = ""        # What pulls the reader to turn to the next page
    is_right_page: bool = False        # Right-hand pages = cliffhanger position
    is_left_page: bool = False         # Left-hand pages = reveal position


def architect_page(
    page_no: int,
    objective: str,
    scene_context: Optional[Dict[str, Any]] = None,
    has_key_moment: bool = False,
    forced_panel_count: Optional[int] = None,
    rng: Optional[random.Random] = None,
) -> PageArchitecture:
    """Build the full architecture for a page from its emotional objective.

    This is the core function. It takes *what the page needs to do*
    and produces *how the page should be structured* — panel count,
    shot sequence, weight distribution, layout, and transitions.
    """
    rng = rng or random.Random()
    obj_key = objective if objective in TEMPO_MAP else "orient"

    tempo, min_panels, max_panels = TEMPO_MAP[obj_key]
    if forced_panel_count is not None:
        panel_count = max(1, min(9, forced_panel_count))
    else:
        panel_count = rng.randint(min_panels, max_panels)

    # Select layout
    layout_options = LAYOUT_MAP.get((tempo, panel_count))
    if not layout_options:
        # Fallback: find closest match
        layout_options = [PageLayout.GRID_VARIED]
        for (t, pc), layouts in LAYOUT_MAP.items():
            if t == tempo and abs(pc - panel_count) <= 1:
                layout_options = layouts
                break
    layout = rng.choice(layout_options)

    # Build shot sequence
    shots = select_shot_sequence(obj_key, panel_count, scene_context)

    # Build weight distribution
    weights = distribute_panel_weights(panel_count, obj_key, has_key_moment)

    # Assemble panel architectures
    panels: List[PanelArchitecture] = []
    for i in range(panel_count):
        is_key = has_key_moment and (i == panel_count - 1)
        transition = PanelTransition.SCENE_TO_SCENE if i == 0 else PanelTransition.ACTION_TO_ACTION
        panels.append(PanelArchitecture(
            panel_index=i,
            shot=shots[i] if i < len(shots) else ShotType.MEDIUM,
            weight=weights[i] if i < len(weights) else PanelWeight.THIRD,
            transition_in=transition,
            is_key_moment=is_key,
        ))

    # Page position (odd = right page, even = left page in Western comics)
    is_right = (page_no % 2 == 1)
    is_left = not is_right

    try:
        obj_enum = EmotionalObjective(obj_key)
    except ValueError:
        obj_enum = EmotionalObjective.ORIENT

    return PageArchitecture(
        page_no=page_no,
        objective=obj_enum,
        tempo=tempo,
        panel_count=panel_count,
        layout=layout,
        panels=panels,
        is_right_page=is_right,
        is_left_page=is_left,
    )


# ── Issue-Level Architecture ─────────────────────────────────────────

def architect_issue(
    spec: Dict[str, Any],
    rng: Optional[random.Random] = None,
) -> Dict[str, Any]:
    """Build a Panel Architecture Matrix for an entire issue.

    Maps issue-by-issue panel density, splash frequency ceilings,
    and page-turn strike points so the arc escalates mathematically
    rather than intuitively.
    """
    rng = rng or random.Random(42)
    pages = get_pages(spec)
    scenes = get_scenes(spec)
    beats = safe_list(spec.get("beats"))
    total_pages = len(pages)

    if total_pages == 0:
        return {"pages": [], "metrics": {}}

    # Build scene index by page number
    scene_by_page: Dict[int, Dict[str, Any]] = {}
    for s in scenes:
        pno = s.get("page_no")
        if isinstance(pno, int):
            scene_by_page[pno] = s

    # Determine emotional arc across pages
    page_objectives: List[str] = []
    for i, page in enumerate(pages):
        page_no = page.get("page_no", i + 1)
        scene = scene_by_page.get(page_no, {})
        purpose = (scene.get("purpose") or "").lower()

        # Map scene purpose to emotional objective
        position = i / max(1, total_pages - 1)  # 0.0 to 1.0

        if page.get("page_type", "normal") in ("splash", "full_page_shot"):
            obj = "reveal" if position > 0.3 else "orient"
        elif "disturbance" in purpose or "inciting" in purpose:
            obj = "escalate"
        elif "escalation" in purpose:
            obj = "escalate"
        elif "reversal" in purpose or "reveal" in purpose:
            obj = "reveal"
        elif "decision" in purpose or "cost" in purpose:
            obj = "decide"
        elif "payoff" in purpose or "climax" in purpose:
            obj = "climax"
        elif "resolution" in purpose:
            obj = "breathe"
        elif position < 0.1:
            obj = "orient"
        elif position < 0.25:
            obj = "escalate"
        elif position < 0.4:
            obj = "confront"
        elif position < 0.55:
            obj = "escalate"
        elif position < 0.7:
            obj = "decide"
        elif position < 0.85:
            obj = "climax"
        elif position < 0.95:
            obj = "devastate"
        else:
            obj = "reflect"

        # Override: page-turn reveals should be on right-hand pages
        if page.get("page_turn_reveal"):
            obj = "reveal"

        page_objectives.append(obj)

    # Build architectures
    page_architectures: List[Dict[str, Any]] = []
    total_panels = 0
    splash_count = 0
    tempo_distribution: Dict[str, int] = {}

    for i, page in enumerate(pages):
        page_no = page.get("page_no", i + 1)
        obj = page_objectives[i]
        scene = scene_by_page.get(page_no, {})

        # Check if page has a key dramatic moment
        has_key = bool(page.get("page_turn_reveal")) or obj in ("reveal", "climax", "devastate")

        # Use existing panel count if present, otherwise let the engine decide
        existing_panels = safe_list(page.get("panels"))
        forced_count = len(existing_panels) if existing_panels else None

        arch = architect_page(
            page_no=page_no,
            objective=obj,
            scene_context=scene,
            has_key_moment=has_key,
            forced_panel_count=forced_count,
            rng=rng,
        )

        # Page-turn strategy
        if arch.is_right_page:
            arch.page_exit_tension = "RIGHT PAGE: End on tension — cliffhanger position."
        if arch.is_left_page:
            arch.page_turn_payoff = "LEFT PAGE: Reveal position — deliver the detonation."

        # Track metrics
        total_panels += arch.panel_count
        if arch.panel_count <= 2:
            splash_count += 1
        tempo_distribution[arch.tempo.value] = tempo_distribution.get(arch.tempo.value, 0) + 1

        page_architectures.append({
            "page_no": arch.page_no,
            "objective": arch.objective.value,
            "tempo": arch.tempo.value,
            "panel_count": arch.panel_count,
            "layout": arch.layout.value,
            "is_right_page": arch.is_right_page,
            "is_left_page": arch.is_left_page,
            "page_turn_payoff": arch.page_turn_payoff,
            "page_exit_tension": arch.page_exit_tension,
            "panels": [
                {
                    "index": p.panel_index,
                    "shot": p.shot.value,
                    "weight": p.weight.value,
                    "transition_in": p.transition_in.value,
                    "is_key_moment": p.is_key_moment,
                }
                for p in arch.panels
            ],
        })

    avg_panels = total_panels / max(1, total_pages)
    splash_ratio = splash_count / max(1, total_pages)

    metrics = {
        "total_pages": total_pages,
        "total_panels": total_panels,
        "avg_panels_per_page": round(avg_panels, 2),
        "splash_or_minimal_pages": splash_count,
        "splash_ratio": round(splash_ratio, 3),
        "splash_budget_ok": splash_ratio <= 0.15,  # Max ~15% splash pages
        "tempo_distribution": tempo_distribution,
        "objectives": {obj: page_objectives.count(obj) for obj in set(page_objectives)},
    }

    return {
        "pages": page_architectures,
        "metrics": metrics,
    }


# ── Panel Architecture Rules ─────────────────────────────────────────
# These rules evaluate whether the paneling architecture is sound.


def rule_panel_tempo_coherence(spec: Dict[str, Any]) -> List[RuleResult]:
    """Check that panel counts match the emotional objectives of pages.

    Six → Five → Four → Three: readers subconsciously feel acceleration.
    If panel count doesn't correlate with scene intensity, the pacing
    is fighting the content.
    """
    pages = get_pages(spec)
    if len(pages) < 3:
        return []

    problems: List[str] = []
    scenes = get_scenes(spec)
    scene_by_page: Dict[int, Dict[str, Any]] = {}
    for s in scenes:
        pno = s.get("page_no")
        if isinstance(pno, int):
            scene_by_page[pno] = s

    for i, page in enumerate(pages):
        page_no = page.get("page_no", i + 1)
        panels = safe_list(page.get("panels"))
        count = len(panels)
        page_type = (page.get("page_type") or "normal").lower()
        scene = scene_by_page.get(page_no, {})
        purpose = (scene.get("purpose") or "").lower()

        # Splash pages with too many panels defeat their purpose
        if page_type in ("splash", "full_page_shot") and count > 3:
            problems.append(
                f"Page {page_no}: splash page has {count} panels — splash means monumental, not crowded."
            )

        # Climactic scenes should not be 7+ panel grids
        if any(k in purpose for k in ("climax", "payoff", "reveal", "reversal")):
            if count > 5:
                problems.append(
                    f"Page {page_no}: {purpose} scene has {count} panels — "
                    f"compress to 3-4 for impact (fewer panels = more dramatic weight)."
                )

        # Quiet/reflective scenes should not be dense grids
        if any(k in purpose for k in ("resolution", "breath", "aftermath")):
            if count > 5:
                problems.append(
                    f"Page {page_no}: reflective scene has {count} panels — "
                    f"decompress to 2-4 (give the reader space to feel)."
                )

    if problems:
        return [make_result(
            rule_id="PANEL.TEMPO_COHERENCE",
            priority=Priority.P3_DRAMA_AND_PACING,
            level=Level.WARN,
            message="Panel counts fight their pages' emotional objectives.",
            evidence={"problems": problems[:20]},
        )]
    return [make_result(
        rule_id="PANEL.TEMPO_COHERENCE",
        priority=Priority.P3_DRAMA_AND_PACING,
        level=Level.PASS,
        message="Panel tempo coherent with page objectives.",
    )]


def rule_panel_shot_variety(spec: Dict[str, Any]) -> List[RuleResult]:
    """Check that shot types vary across pages — monotone framing kills energy.

    If every panel is a medium shot, nothing is emphasized. Shot variety
    is how you control what the reader focuses on.
    """
    pages = get_pages(spec)
    if not pages:
        return []

    monotone_pages: List[str] = []
    for i, page in enumerate(pages):
        page_no = page.get("page_no", i + 1)
        panels = safe_list(page.get("panels"))
        if len(panels) < 3:
            continue

        shots = []
        for panel in panels:
            art = (panel.get("art") or "").lower()
            shot = (panel.get("shot") or "").lower()
            combined = art + " " + shot
            # Detect shot type from art direction text
            if any(k in combined for k in ("close-up", "close up", "closeup", "extreme close")):
                shots.append("close")
            elif any(k in combined for k in ("wide", "establishing", "bird")):
                shots.append("wide")
            elif any(k in combined for k in ("over-the-shoulder", "over shoulder", "ots")):
                shots.append("ots")
            elif any(k in combined for k in ("medium",)):
                shots.append("medium")
            elif any(k in combined for k in ("low angle", "worm")):
                shots.append("low")
            elif any(k in combined for k in ("high angle",)):
                shots.append("high")
            else:
                shots.append("unspecified")

        unique_shots = set(shots)
        if len(unique_shots) == 1 and len(panels) >= 3:
            monotone_pages.append(
                f"Page {page_no}: all {len(panels)} panels use '{shots[0]}' framing."
            )

    if monotone_pages:
        return [make_result(
            rule_id="PANEL.SHOT_VARIETY",
            priority=Priority.P2_COMICS_SPECIFICITY,
            level=Level.WARN,
            message="Monotone shot selection detected — vary framing to control reader focus.",
            evidence={"pages": monotone_pages[:15]},
        )]
    return [make_result(
        rule_id="PANEL.SHOT_VARIETY",
        priority=Priority.P2_COMICS_SPECIFICITY,
        level=Level.PASS,
        message="Shot variety acceptable.",
    )]


def rule_panel_weight_distribution(spec: Dict[str, Any]) -> List[RuleResult]:
    """Check that panel sizing serves the narrative — not everything is equal.

    If all panels are the same size on every page, nothing gets emphasis.
    Key moments need bigger panels. Transitions need smaller ones.
    """
    pages = get_pages(spec)
    if not pages:
        return []

    problems: List[str] = []
    for i, page in enumerate(pages):
        page_no = page.get("page_no", i + 1)
        panels = safe_list(page.get("panels"))
        if len(panels) < 3:
            continue

        # Check if any panel has weight/size annotations
        weights = [p.get("weight") or p.get("size") or "" for p in panels]
        has_weights = any(w for w in weights)

        # If page has a page_turn_reveal but no panel is marked as key/large
        if page.get("page_turn_reveal") and not has_weights:
            problems.append(
                f"Page {page_no}: tagged as page-turn reveal but no panel is sized for emphasis. "
                f"The reveal panel should be larger than surrounding panels."
            )

    if problems:
        return [make_result(
            rule_id="PANEL.WEIGHT_DISTRIBUTION",
            priority=Priority.P3_DRAMA_AND_PACING,
            level=Level.WARN,
            message="Panel weight/sizing not specified for key pages.",
            evidence={"problems": problems[:15]},
        )]
    return [make_result(
        rule_id="PANEL.WEIGHT_DISTRIBUTION",
        priority=Priority.P3_DRAMA_AND_PACING,
        level=Level.PASS,
        message="Panel weight distribution acceptable.",
    )]


def rule_page_turn_positioning(spec: Dict[str, Any]) -> List[RuleResult]:
    """Check that page-turn reveals land on the correct physical page.

    The most powerful reveal in comics is at the turn of a page.
    Right-hand page cliffhangers. Left-hand page reveals.
    If a reveal happens mid-page, it leaks energy.
    If it happens after a page turn, it detonates.
    """
    pages = get_pages(spec)
    if len(pages) < 4:
        return []

    misplaced: List[str] = []
    for i, page in enumerate(pages):
        page_no = page.get("page_no", i + 1)
        if not page.get("page_turn_reveal"):
            continue

        # In Western comics: odd pages are right-hand, even are left-hand
        # The REVEAL should land on a left-hand (even) page — what you see after turning
        # The CLIFFHANGER should be on a right-hand (odd) page — what pushes you to turn
        is_even = (page_no % 2 == 0)
        if not is_even:
            misplaced.append(
                f"Page {page_no}: reveal tagged on right-hand (odd) page. "
                f"Reveals detonate on LEFT-hand (even) pages — "
                f"the reader turns and BAM. Move the reveal to page {page_no + 1} "
                f"and use page {page_no} as the cliffhanger lead-in."
            )

    if misplaced:
        return [make_result(
            rule_id="PANEL.PAGE_TURN_POSITION",
            priority=Priority.P4_TRANSITIONS,
            level=Level.WARN,
            message="Page-turn reveals mispositioned — reveals should land on left-hand (even) pages.",
            evidence={"misplaced": misplaced[:10]},
        )]
    return [make_result(
        rule_id="PANEL.PAGE_TURN_POSITION",
        priority=Priority.P4_TRANSITIONS,
        level=Level.PASS,
        message="Page-turn reveal positioning correct.",
    )]


def rule_splash_budget(spec: Dict[str, Any]) -> List[RuleResult]:
    """Enforce splash page discipline — splashes must earn their space.

    A full-page splash should only be used when the narrative justifies it.
    If used casually, it becomes noise. If it doesn't shift stakes, reveal
    scale, or reframe understanding, it's indulgence.

    Budget: max ~15% of pages should be splash/minimal (1-2 panels).
    """
    pages = get_pages(spec)
    if len(pages) < 5:
        return []

    splash_pages: List[int] = []
    for i, page in enumerate(pages):
        page_no = page.get("page_no", i + 1)
        panels = safe_list(page.get("panels"))
        page_type = (page.get("page_type") or "normal").lower()
        if page_type in ("splash", "full_page_shot") or len(panels) <= 2:
            splash_pages.append(page_no)

    ratio = len(splash_pages) / len(pages)
    if ratio > 0.20:
        return [make_result(
            rule_id="PANEL.SPLASH_BUDGET",
            priority=Priority.P2_COMICS_SPECIFICITY,
            level=Level.WARN,
            message=f"Splash/minimal pages at {round(ratio*100)}% ({len(splash_pages)}/{len(pages)}) — "
                    f"above 15-20% ceiling. Each splash must shift stakes, reveal scale, or reframe understanding.",
            evidence={"splash_pages": splash_pages, "ratio": round(ratio, 3)},
        )]
    return [make_result(
        rule_id="PANEL.SPLASH_BUDGET",
        priority=Priority.P2_COMICS_SPECIFICITY,
        level=Level.PASS,
        message=f"Splash budget OK ({len(splash_pages)}/{len(pages)}, {round(ratio*100)}%).",
    )]


def rule_panel_transition_awareness(spec: Dict[str, Any]) -> List[RuleResult]:
    """Check that transitions between panels are intentional, not accidental.

    Every panel-to-panel cut is a choice: moment-to-moment slows time,
    action-to-action moves plot, subject-to-subject shifts focus,
    scene-to-scene jumps context. If the spec has no transition
    annotations, the writer isn't thinking about gutters.
    """
    pages = get_pages(spec)
    if not pages:
        return []

    total_panels = 0
    annotated_transitions = 0
    for page in pages:
        panels = safe_list(page.get("panels"))
        total_panels += len(panels)
        for panel in panels:
            if panel.get("transition") or panel.get("transition_in"):
                annotated_transitions += 1

    if total_panels > 10 and annotated_transitions == 0:
        return [make_result(
            rule_id="PANEL.TRANSITION_AWARENESS",
            priority=Priority.P5_STRUCTURE_SHAPE,
            level=Level.NOTE,
            message=f"No panel transition types annotated across {total_panels} panels. "
                    f"Consider specifying transitions (moment_to_moment, action_to_action, "
                    f"subject_to_subject, scene_to_scene, aspect_to_aspect) to control "
                    f"how the reader's brain fills the gutters between panels.",
        )]
    return [make_result(
        rule_id="PANEL.TRANSITION_AWARENESS",
        priority=Priority.P5_STRUCTURE_SHAPE,
        level=Level.PASS,
        message="Panel transition annotations present.",
    )]


def rule_page_emotional_objective(spec: Dict[str, Any]) -> List[RuleResult]:
    """Every page must answer: what is the emotional objective?

    If a page has no clear objective, it's structural filler. Pages without
    an objective field or a scene purpose are drift — they exist without
    justifying their existence.
    """
    pages = get_pages(spec)
    if not pages:
        return []

    scenes = get_scenes(spec)
    scene_by_page: Dict[int, Dict[str, Any]] = {}
    for s in scenes:
        pno = s.get("page_no")
        if isinstance(pno, int):
            scene_by_page[pno] = s

    aimless: List[int] = []
    for i, page in enumerate(pages):
        page_no = page.get("page_no", i + 1)
        has_objective = bool(page.get("objective") or page.get("emotional_objective"))
        scene = scene_by_page.get(page_no, {})
        has_purpose = bool((scene.get("purpose") or "").strip())

        if not has_objective and not has_purpose:
            aimless.append(page_no)

    if len(aimless) > len(pages) * 0.3:
        return [make_result(
            rule_id="PANEL.PAGE_OBJECTIVE",
            priority=Priority.P3_DRAMA_AND_PACING,
            level=Level.WARN,
            message=f"{len(aimless)} of {len(pages)} pages have no emotional objective or scene purpose. "
                    f"Every page must answer: what is this page doing to the reader?",
            evidence={"aimless_pages": aimless[:20]},
        )]
    return [make_result(
        rule_id="PANEL.PAGE_OBJECTIVE",
        priority=Priority.P3_DRAMA_AND_PACING,
        level=Level.PASS,
        message="Pages have emotional objectives or scene purposes.",
    )]


def rule_panel_progressive_tempo(spec: Dict[str, Any]) -> List[RuleResult]:
    """Check for progressive tempo shifts across the issue.

    The duel pattern from O'Neil: Six -> Five -> Four -> Three.
    Panel count should progressively decrease toward climactic moments.
    If panel density is flat across an entire issue, pacing never
    accelerates — and that means the story never builds.
    """
    pages = get_pages(spec)
    if len(pages) < 6:
        return []

    panel_counts = []
    for page in pages:
        panels = safe_list(page.get("panels"))
        panel_counts.append(len(panels))

    if not panel_counts:
        return []

    # Check if panel density is completely flat (no variation)
    unique_counts = set(panel_counts)
    if len(unique_counts) == 1:
        return [make_result(
            rule_id="PANEL.PROGRESSIVE_TEMPO",
            priority=Priority.P3_DRAMA_AND_PACING,
            level=Level.WARN,
            message=f"Every page has exactly {panel_counts[0]} panels — flat pacing. "
                    f"Vary panel count to control time: more panels = slower tempo, "
                    f"fewer panels = faster/more dramatic. "
                    f"Progressive reduction (6->5->4->3) toward climax creates acceleration.",
            evidence={"panel_counts": panel_counts},
        )]

    # Check if the last third of the issue has any acceleration
    third = max(1, len(panel_counts) // 3)
    first_third_avg = sum(panel_counts[:third]) / max(1, third)
    last_third_avg = sum(panel_counts[-third:]) / max(1, third)

    # If the last third is denser than the first, tempo is backwards
    if last_third_avg > first_third_avg + 1.0:
        return [make_result(
            rule_id="PANEL.PROGRESSIVE_TEMPO",
            priority=Priority.P3_DRAMA_AND_PACING,
            level=Level.NOTE,
            message=f"Panel density increases toward the end "
                    f"(first third avg: {round(first_third_avg, 1)}, last third avg: {round(last_third_avg, 1)}). "
                    f"This slows the climax. Consider reducing panels in the final act for dramatic compression.",
            evidence={"first_third_avg": round(first_third_avg, 1), "last_third_avg": round(last_third_avg, 1)},
        )]

    return [make_result(
        rule_id="PANEL.PROGRESSIVE_TEMPO",
        priority=Priority.P3_DRAMA_AND_PACING,
        level=Level.PASS,
        message="Progressive tempo shifts present.",
    )]


# ── Panel Architecture Rewriter Integration ───────────────────────────

def apply_panel_architecture(spec: Dict[str, Any], rng: Optional[random.Random] = None) -> List[Dict[str, str]]:
    """Apply the panel architecture engine to a spec, enriching pages
    with objective, tempo, layout, shot, weight, and transition data.

    Returns a list of actions taken (for the rewrite action log).
    """
    rng = rng or random.Random(42)
    architecture = architect_issue(spec, rng=rng)
    actions: List[Dict[str, str]] = []

    pages = get_pages(spec)
    arch_by_page: Dict[int, Dict[str, Any]] = {}
    for pa in architecture.get("pages", []):
        arch_by_page[pa["page_no"]] = pa

    for i, page in enumerate(pages):
        page_no = page.get("page_no", i + 1)
        arch = arch_by_page.get(page_no)
        if not arch:
            continue

        # Enrich page with architecture data
        if not page.get("objective"):
            page["objective"] = arch["objective"]
            actions.append({
                "location": f"page:{page_no}", "field": "objective",
                "old_value": "(none)", "new_value": arch["objective"],
                "reason": "Page lacked emotional objective; assigned from narrative position.",
            })

        if not page.get("tempo"):
            page["tempo"] = arch["tempo"]

        if not page.get("layout"):
            page["layout"] = arch["layout"]
            actions.append({
                "location": f"page:{page_no}", "field": "layout",
                "old_value": "(none)", "new_value": arch["layout"],
                "reason": f"Layout assigned based on tempo ({arch['tempo']}) and panel count ({arch['panel_count']}).",
            })

        # Enrich panels with shot/weight/transition data
        panels = safe_list(page.get("panels"))
        for j, panel in enumerate(panels):
            if j < len(arch["panels"]):
                pa = arch["panels"][j]
                if not panel.get("shot"):
                    panel["shot"] = pa["shot"]
                if not panel.get("weight"):
                    panel["weight"] = pa["weight"]
                if not panel.get("transition_in"):
                    panel["transition_in"] = pa["transition_in"]
                if pa.get("is_key_moment"):
                    panel["is_key_moment"] = True

        # Page-turn annotations
        if arch.get("page_exit_tension") and not page.get("page_exit_tension"):
            page["page_exit_tension"] = arch["page_exit_tension"]
        if arch.get("page_turn_payoff") and not page.get("page_turn_payoff"):
            page["page_turn_payoff"] = arch["page_turn_payoff"]

    # Store metrics on the spec
    spec["panel_architecture_metrics"] = architecture.get("metrics", {})

    return actions


# =====================================================================
# REWRITE ENGINE (NEW) — Actually rewrites content, not just placeholders
# =====================================================================
#
# The key difference from auto-fix:
#   auto-fix inserts "[TODO entry hook for S1]"
#   rewrite produces "CAPTION: Rain hammers the fire escape. Mara's hands shake."
#
# This is template-driven (no LLM needed). Templates are parameterized by
# the spec's own story data (protagonist, goal, stakes, tone, setting, etc.)
# so the output sounds like YOUR story, not generic filler.
# =====================================================================


class RewriteAction:
    """One atomic rewrite: what changed, where, why."""

    def __init__(self, location: str, field: str, old_value: str, new_value: str, reason: str):
        self.location = location
        self.field = field
        self.old_value = old_value
        self.new_value = new_value
        self.reason = reason

    def to_dict(self) -> Dict[str, str]:
        return {
            "location": self.location,
            "field": self.field,
            "old_value": self.old_value[:120],
            "new_value": self.new_value[:200],
            "reason": self.reason,
        }


class ScriptRewriter:
    """
    Rule-driven rewriter. Takes a spec + rule results, rewrites weak spots.

    Design principles:
    - Never invent plot — only sharpen, compress, and restructure what's there.
    - Use the spec's own vocabulary (character names, setting, tone) in rewrites.
    - Dialogue rewrites preserve speaker intent; trim fat, add rhythm.
    - Every rewrite is logged as a RewriteAction so the author can review/reject.
    """

    def __init__(self, spec: Dict[str, Any], results: List[RuleResult], rng: Optional[random.Random] = None):
        self.spec = spec
        self.results = results
        self.rng = rng or random.Random(42)
        self.actions: List[RewriteAction] = []

        # Pull story context once for all templates
        self.who = (spec.get("protagonist") or "the protagonist").strip()
        self.goal = (spec.get("protagonist_goal") or "their goal").strip()
        self.stakes = (spec.get("stakes") or "").strip()
        self.setting = (spec.get("setting") or "the city").strip()
        self.tone = (spec.get("tone") or "tense").strip()
        self.nemesis = ""
        for c in safe_list(spec.get("characters")):
            if (c.get("role") or "").lower() == "nemesis":
                self.nemesis = (c.get("name") or "the antagonist").strip()
                break
        if not self.nemesis:
            self.nemesis = "the opposition"
        self.flaw = ""
        flaws = safe_list(spec.get("flaws"))
        if flaws:
            self.flaw = str(flaws[0]).strip()
        self.need = (spec.get("protagonist_need") or "").strip()
        self.premise = (spec.get("premise") or "").strip()
        self.change = (spec.get("character_change") or "").strip()
        self.genre = (spec.get("genre") or "").strip()

    def _pick(self, options: List[str]) -> str:
        return options[self.rng.randrange(0, len(options))]

    def _failed_rules(self) -> set:
        return {r.rule_id for r in self.results if r.level in (Level.FAIL, Level.WARN)}

    def run_all(self) -> List[RewriteAction]:
        """Execute all rewrite passes. Returns list of actions taken."""
        failed = self._failed_rules()

        # Order matters: structural fixes first, then content, then polish
        self._rewrite_missing_art_directions(failed)
        self._rewrite_silent_intents(failed)
        self._rewrite_entry_hooks(failed)
        self._rewrite_exit_hooks(failed)
        self._rewrite_scene_outcomes(failed)
        self._rewrite_scene_conflicts(failed)
        self._rewrite_vague_stakes(failed)
        self._rewrite_copy_heavy_panels(failed)
        self._rewrite_dialect_crutches(failed)
        self._rewrite_hollow_beats(failed)
        self._rewrite_page_turn_reveals(failed)
        self._ensure_central_conflict(failed)
        self._ensure_protagonist_need(failed)
        self._tighten_dialogue()

        # Panel architecture: apply objective, tempo, layout, shot, weight,
        # transition data to every page and panel in the spec.
        self._apply_panel_architecture(failed)

        return self.actions

    # -----------------------------------------------------------------
    # REWRITE: Panel Architecture (objective, tempo, layout, shot, weight)
    # -----------------------------------------------------------------

    def _apply_panel_architecture(self, failed: set) -> None:
        """Apply the panel architecture engine to enrich pages with
        structural paneling data: objectives, tempo, layout, shots,
        weights, and transitions.

        This runs regardless of which rules failed — it's additive
        enrichment, not a fix for a specific problem.
        """
        arch_actions = apply_panel_architecture(self.spec, rng=self.rng)
        for act in arch_actions:
            self.actions.append(RewriteAction(
                location=act["location"],
                field=act["field"],
                old_value=act["old_value"],
                new_value=act["new_value"],
                reason=act["reason"],
            ))

    # -----------------------------------------------------------------
    # REWRITE: Missing art directions
    # -----------------------------------------------------------------

    def _rewrite_missing_art_directions(self, failed: set) -> None:
        if "P0.SCRIPT_MIN_FIELDS" not in failed and "P0.HYBRID_LANGUAGE" not in failed:
            return

        pages = get_pages(self.spec)
        scenes = get_scenes(self.spec)
        scene_by_page: Dict[int, Dict[str, Any]] = {}
        for s in scenes:
            pno = s.get("page_no")
            if isinstance(pno, int):
                scene_by_page[pno] = s

        for i, page in enumerate(pages):
            page_no = page.get("page_no", i + 1)
            scene = scene_by_page.get(page_no, {})
            purpose = (scene.get("purpose") or "story beat").lower()

            for j, panel in enumerate(safe_list(page.get("panels"))):
                art = (panel.get("art") or "").strip()
                if not art:
                    entries = panel_text_entries(panel)
                    text_hint = ""
                    if entries:
                        first_text = (entries[0].get("value") or "").strip()
                        text_hint = truncate_words(first_text, 6)

                    new_art = self._generate_art_direction(page_no, j + 1, purpose, text_hint)
                    panel["art"] = new_art
                    self.actions.append(RewriteAction(
                        location=f"page:{page_no} panel:{j+1}",
                        field="art",
                        old_value="(empty)",
                        new_value=new_art,
                        reason="Panel had no art direction; generated from scene context.",
                    ))

    def _generate_art_direction(self, page_no: int, panel_no: int, purpose: str, text_hint: str) -> str:
        # Build a specific, drawable art direction from story context
        angle_options = ["MEDIUM SHOT", "CLOSE-UP", "WIDE SHOT", "OVER-THE-SHOULDER", "LOW ANGLE", "HIGH ANGLE"]
        angle = self._pick(angle_options)

        if page_no == 1 and panel_no == 1:
            return f"{angle} -- {self.setting}. {self.who} mid-action, body language reads {self.tone}. Establish location and mood."
        if "disturbance" in purpose or "inciting" in purpose:
            return f"{angle} -- {self.who} reacts to the disruption. Expression shifts from composure to alarm. {self.setting} visible in background."
        if "escalation" in purpose:
            return f"{angle} -- Tension visible: {self.who} cornered or pressed. {self.nemesis}'s influence visible (shadow, symbol, or agent). Environment feels hostile."
        if "decision" in purpose or "cost" in purpose:
            return f"{angle} -- {self.who} at a crossroads. Two paths visible (literal or metaphorical). Face shows the weight of the choice."
        if "payoff" in purpose or "resolution" in purpose:
            return f"{angle} -- {self.who} commits to the hard choice. Action is decisive. The cost is visible on their face or body."

        # Generic but specific
        if text_hint:
            return f"{angle} -- {self.who} in {self.setting}. Beat: {purpose}. Visual must support: \"{text_hint}\". Keep background active."
        return f"{angle} -- {self.who} in {self.setting}. Beat: {purpose}. Show character state through posture and environment."

    # -----------------------------------------------------------------
    # REWRITE: Silent intent for art-only panels
    # -----------------------------------------------------------------

    def _rewrite_silent_intents(self, failed: set) -> None:
        if "P0.HYBRID_LANGUAGE" not in failed:
            return
        pages = get_pages(self.spec)
        for i, page in enumerate(pages):
            page_no = page.get("page_no", i + 1)
            for j, panel in enumerate(safe_list(page.get("panels"))):
                art = (panel.get("art") or "").strip()
                entries = panel_text_entries(panel)
                silent = (panel.get("silent_intent") or "").strip()
                if art and not entries and not silent:
                    new_intent = f"Silent beat: {self.who}'s body language carries the emotion. The image does the work the words can't."
                    panel["silent_intent"] = new_intent
                    self.actions.append(RewriteAction(
                        location=f"page:{page_no} panel:{j+1}", field="silent_intent",
                        old_value="(empty)", new_value=new_intent,
                        reason="Art-only panel needed silent_intent for letterer/artist clarity.",
                    ))

    # -----------------------------------------------------------------
    # REWRITE: Entry hooks (actual content, not [TODO])
    # -----------------------------------------------------------------

    def _rewrite_entry_hooks(self, failed: set) -> None:
        if "SLOANE.ENTRY_HOOK" not in failed:
            return
        idx = SpecIndex(self.spec)
        scenes = idx.ensure_scenes()
        for si, s in enumerate(scenes):
            if (s.get("entry_hook") or "").strip():
                continue
            sid = s.get("scene_id") or f"S{si+1}"
            purpose = (s.get("purpose") or "beat").lower()
            conflict = (s.get("conflict") or "").strip()

            hooks = [
                f"CAPTION: {self.setting}. The air changed three minutes ago and {self.who} can feel it.",
                f"CAPTION: {self.who} steps into the room knowing one thing: {self.goal} has a new price.",
                f"CAPTION: After what just happened, {self.who} has no choice but to move forward.",
                f"CAPTION: The {purpose} hits before {self.who} is ready. It always does.",
                f"WIDE SHOT: {self.setting}. {self.who} enters frame -- posture says everything the dialogue won't.",
            ]
            if conflict:
                hooks.append(f"CAPTION: {conflict} -- and {self.who} walked right into it.")

            chosen = self._pick(hooks)
            s["entry_hook"] = chosen
            self.actions.append(RewriteAction(
                location=f"scene:{sid}", field="entry_hook",
                old_value="(empty)", new_value=chosen,
                reason=f"Scene lacked entry hook; generated from purpose '{purpose}' and story context.",
            ))

    # -----------------------------------------------------------------
    # REWRITE: Exit hooks (propulsion into next scene)
    # -----------------------------------------------------------------

    def _rewrite_exit_hooks(self, failed: set) -> None:
        if "SLOANE.EXIT_PROPULSION" not in failed:
            return
        idx = SpecIndex(self.spec)
        scenes = idx.ensure_scenes()
        for si, s in enumerate(scenes):
            if (s.get("exit_hook") or "").strip():
                continue
            sid = s.get("scene_id") or f"S{si+1}"
            nxt = s.get("next_scene_id") or ""

            hooks = [
                f"LAST PANEL: {self.who}'s face -- a decision made, and no going back. CUT TO: {nxt or 'next beat'}.",
                f"CAPTION: The clock just started. {self.stakes or 'Everything is on the line.'}",
                f"DIALOGUE: {self.who}: \"That's not a choice. That's a trap.\" -- SMASH CUT.",
                f"VISUAL: A door opens on something worse. {self.who} steps through anyway.",
                f"CAPTION: {self.who} realizes too late: {self.nemesis} was already ahead.",
                f"BEAT: The cost lands. {self.who} absorbs it. Moves. The next scene inherits the damage.",
            ]
            chosen = self._pick(hooks)
            s["exit_hook"] = chosen
            self.actions.append(RewriteAction(
                location=f"scene:{sid}", field="exit_hook",
                old_value="(empty)", new_value=chosen,
                reason="Scene lacked exit hook / propulsion into next scene.",
            ))

    # -----------------------------------------------------------------
    # REWRITE: Scene outcomes
    # -----------------------------------------------------------------

    def _rewrite_scene_outcomes(self, failed: set) -> None:
        if "SCENE.OUTCOME_DIRECTION" not in failed:
            return
        scenes = get_scenes(self.spec)
        for si, s in enumerate(scenes):
            if (s.get("outcome") or "").strip():
                continue
            sid = s.get("scene_id") or f"S{si+1}"
            purpose = (s.get("purpose") or "").lower()

            outcomes = [
                f"Value shift: {self.who}'s options narrow. Trust is damaged. The cost went up.",
                f"Power shift: {self.nemesis} gained ground. {self.who} lost a resource or ally.",
                f"Revelation: something {self.who} believed was wrong. The plan must change.",
                f"Commitment: {self.who} chose a path. No reversal possible. Stakes crystallized.",
                f"Escalation: what started as {purpose} became a crisis. Next scene inherits the pressure.",
            ]
            chosen = self._pick(outcomes)
            s["outcome"] = chosen
            self.actions.append(RewriteAction(
                location=f"scene:{sid}", field="outcome",
                old_value="(empty)", new_value=chosen,
                reason="Scene lacked outcome direction.",
            ))

    # -----------------------------------------------------------------
    # REWRITE: Scene-level conflicts
    # -----------------------------------------------------------------

    def _rewrite_scene_conflicts(self, failed: set) -> None:
        if "STAN.CONFLICT_PER_SCENE" not in failed:
            return
        scenes = get_scenes(self.spec)
        obstacles = safe_list(self.spec.get("obstacles"))
        for si, s in enumerate(scenes):
            if (s.get("conflict") or "").strip():
                continue
            sid = s.get("scene_id") or f"S{si+1}"

            if obstacles:
                obs = obstacles[si % len(obstacles)]
                conflict = f"{self.who} vs {obs} -- {self.nemesis} benefits if {self.who} stalls."
            else:
                conflict = f"{self.who} needs {self.goal} but the scene's own logic resists: time, trust, or terrain."
            s["conflict"] = conflict
            self.actions.append(RewriteAction(
                location=f"scene:{sid}", field="conflict",
                old_value="(empty)", new_value=conflict,
                reason="Scene had no explicit conflict.",
            ))

    # -----------------------------------------------------------------
    # REWRITE: Vague stakes -> concrete stakes
    # -----------------------------------------------------------------

    def _rewrite_vague_stakes(self, failed: set) -> None:
        if "DCOSTA.STAKES_SPECIFICITY" not in failed:
            return
        old_stakes = (self.spec.get("stakes") or "").strip()
        if word_count(old_stakes) >= 8:
            # Stakes exist but lack keywords -- sharpen them
            new_stakes = (
                f"If {self.who} fails to {self.goal}, "
                f"then {self.nemesis} controls the outcome: "
                f"freedom is lost, reputation is destroyed, and the deadline expires "
                f"before anyone can undo the damage. The family pays the cost."
            )
        else:
            # Stakes are too thin -- write them from scratch
            new_stakes = (
                f"If {self.who} fails to {self.goal}, "
                f"{self.nemesis} wins by default. "
                f"Someone {self.who} loves loses their freedom. "
                f"The city's power shifts permanently. "
                f"The deadline is real: miss it and the job is gone, the identity is rewritten, "
                f"and the mission dies with {self.who}'s reputation."
            )
        self.spec["stakes"] = new_stakes
        self.actions.append(RewriteAction(
            location="spec", field="stakes",
            old_value=old_stakes or "(empty)",
            new_value=new_stakes,
            reason="Stakes were too vague or too short; rewritten with concrete loss categories.",
        ))

    # -----------------------------------------------------------------
    # REWRITE: Copy-heavy panels -> split or trim
    # -----------------------------------------------------------------

    def _rewrite_copy_heavy_panels(self, failed: set) -> None:
        if "PRIEST.COPY_HEAVY" not in failed and "MOORE.PANEL_TIME_STOPPERS" not in failed:
            return
        pages = get_pages(self.spec)
        for i, page in enumerate(pages):
            page_no = page.get("page_no", i + 1)
            for j, panel in enumerate(safe_list(page.get("panels"))):
                sec = estimate_panel_seconds(panel)
                if sec <= 10.0:
                    continue
                entries = panel_text_entries(panel)
                if not entries:
                    continue

                # Strategy: keep first 25 words in balloon, move overflow to caption
                all_text = " ".join(e.get("value", "") for e in entries)
                words = _WORD_RE.findall(all_text)
                if len(words) <= 25:
                    continue

                balloon_text = " ".join(words[:20])
                overflow_text = " ".join(words[20:35])

                new_text: List[Dict[str, str]] = [
                    {"type": "balloon", "value": balloon_text + " --"},
                    {"type": "caption", "value": overflow_text},
                ]
                old_joined = all_text[:100]
                panel["text"] = new_text

                self.actions.append(RewriteAction(
                    location=f"page:{page_no} panel:{j+1}", field="text",
                    old_value=old_joined,
                    new_value=f"BALLOON: {balloon_text[:60]}... | CAPTION: {overflow_text[:60]}...",
                    reason=f"Panel was {round(sec,1)}s read time; split into balloon + caption to restore pacing.",
                ))

    # -----------------------------------------------------------------
    # REWRITE: Dialect crutches -> clean dialogue
    # -----------------------------------------------------------------

    def _rewrite_dialect_crutches(self, failed: set) -> None:
        if "PRIEST.DIALOGUE_CLEANLINESS" not in failed:
            return
        pages = get_pages(self.spec)
        for i, page in enumerate(pages):
            page_no = page.get("page_no", i + 1)
            for j, panel in enumerate(safe_list(page.get("panels"))):
                entries = panel_text_entries(panel)
                changed = False
                for e in entries:
                    val = (e.get("value") or "")
                    if _FAKE_DIALECT_RE.search(val):
                        old_val = val
                        # Replace phonetic crutches with clean equivalents
                        new_val = re.sub(r'\bmah\b', 'my', val, flags=re.IGNORECASE)
                        new_val = re.sub(r'\bmuh\b', 'my', new_val, flags=re.IGNORECASE)
                        new_val = re.sub(r'\bah\b', 'I', new_val, flags=re.IGNORECASE)
                        e["value"] = new_val
                        changed = True
                        self.actions.append(RewriteAction(
                            location=f"page:{page_no} panel:{j+1}", field="dialogue",
                            old_value=old_val[:80], new_value=new_val[:80],
                            reason="Replaced phonetic dialect crutches with clean dialogue (rhythm > misspelling).",
                        ))
                if changed:
                    # Rebuild text list from entries
                    panel["text"] = entries

    # -----------------------------------------------------------------
    # REWRITE: Hollow beats -> add consequence
    # -----------------------------------------------------------------

    def _rewrite_hollow_beats(self, failed: set) -> None:
        if "P1.STORY_STRUCTURE_CONSEQUENCE" not in failed:
            return
        beats = safe_list(self.spec.get("beats"))
        for bi, b in enumerate(beats):
            has_consequence = b.get("change") or b.get("new_problem") or b.get("cost")
            if has_consequence:
                continue

            name = (b.get("name") or f"Beat {bi+1}").strip()
            cost_options = [
                f"{self.who} loses a resource they can't replace.",
                f"An ally's trust cracks. The relationship shifts.",
                f"The plan worked -- but the price was a piece of {self.who}'s self-image.",
                f"{self.nemesis} anticipated this. The trap was the success itself.",
                f"Time ran out on something else while {self.who} was focused here.",
            ]
            change_options = [
                f"{self.who}'s status shifts: what was safe is now dangerous.",
                "The power dynamic flips. Who had leverage lost it.",
                f"{self.who} learned something that makes the old plan impossible.",
                "A door closed permanently. The story can't go backward.",
            ]
            new_problem_options = [
                f"The solution created a new enemy: someone who was neutral now opposes {self.who}.",
                f"{self.nemesis} adapts. The next obstacle is harder because of what {self.who} just did.",
                "The victory came with a witness. Now someone knows too much.",
            ]

            b["cost"] = self._pick(cost_options)
            b["change"] = self._pick(change_options)
            if not b.get("new_problem"):
                b["new_problem"] = self._pick(new_problem_options)

            self.actions.append(RewriteAction(
                location=f"beat:{bi+1} ({name})", field="cost/change/new_problem",
                old_value="(hollow -- no consequence)",
                new_value=f"cost: {b['cost'][:60]} | change: {b['change'][:60]}",
                reason="Beat lacked narrative consequence; added cost + change + new_problem.",
            ))

    # -----------------------------------------------------------------
    # REWRITE: Page-turn reveals
    # -----------------------------------------------------------------

    def _rewrite_page_turn_reveals(self, failed: set) -> None:
        if "GRID.PAGE_TURN_REVEALS" not in failed:
            return
        pages = get_pages(self.spec)
        if len(pages) < 3:
            return
        # Find the best page for a turn reveal (middle-ish, where a scene shifts)
        scenes = get_scenes(self.spec)
        best_page_no = None
        for s in scenes:
            purpose = (s.get("purpose") or "").lower()
            if any(k in purpose for k in ["reversal", "escalation", "decision", "reveal"]):
                pno = s.get("page_no")
                if isinstance(pno, int) and 1 < pno < len(pages):
                    best_page_no = pno
                    break
        if not best_page_no:
            mid = max(0, (len(pages) // 2) - 1)
            best_page_no = pages[mid].get("page_no", mid + 1)

        for p in pages:
            if p.get("page_no") == best_page_no:
                p["page_turn_reveal"] = True
                label = f"REVEAL: The truth about {self.nemesis}'s plan lands. {self.who} sees the real cost."
                p["reveal_label"] = label
                self.actions.append(RewriteAction(
                    location=f"page:{best_page_no}", field="page_turn_reveal",
                    old_value="(none tagged)", new_value=label,
                    reason="No page-turn reveal was tagged; placed at the story's pivot point.",
                ))
                break

    # -----------------------------------------------------------------
    # REWRITE: Central conflict
    # -----------------------------------------------------------------

    def _ensure_central_conflict(self, failed: set) -> None:
        if "CONFLICT.FOUR_LEVELS" not in failed:
            return
        if self.spec.get("central_conflict"):
            return
        conflict = (
            f"{self.who} vs {self.nemesis}: a fight over who controls "
            f"the truth, the resources, and the outcome of {self.goal}. "
            f"The deeper conflict is internal: {self.who}'s {self.flaw or 'worst habit'} "
            f"is the weapon {self.nemesis} exploits."
        )
        self.spec["central_conflict"] = conflict
        self.spec["central_conflict_type"] = "external + internal"
        self.actions.append(RewriteAction(
            location="spec", field="central_conflict",
            old_value="(empty)", new_value=conflict[:120],
            reason="Central conflict was missing; generated from character dynamics.",
        ))

    # -----------------------------------------------------------------
    # REWRITE: Protagonist need
    # -----------------------------------------------------------------

    def _ensure_protagonist_need(self, failed: set) -> None:
        if "MYERS.UNITY_ARC" not in failed:
            return
        if self.spec.get("protagonist_need"):
            return
        if self.flaw:
            need = f"Learn to overcome {self.flaw} by choosing vulnerability over control."
        else:
            need = "Accept that strength requires trust, not isolation."
        self.spec["protagonist_need"] = need
        self.actions.append(RewriteAction(
            location="spec", field="protagonist_need",
            old_value="(empty)", new_value=need,
            reason="Protagonist psychological need was missing; derived from character flaw.",
        ))

    # -----------------------------------------------------------------
    # POLISH: Tighten all dialogue (trim filler words)
    # -----------------------------------------------------------------

    _FILLER_PATTERNS = [
        (re.compile(r'\b(well,?\s+)', re.IGNORECASE), ''),
        (re.compile(r'\b(you know,?\s+)', re.IGNORECASE), ''),
        (re.compile(r'\b(I mean,?\s+)', re.IGNORECASE), ''),
        (re.compile(r'\b(basically,?\s+)', re.IGNORECASE), ''),
        (re.compile(r'\b(actually,?\s+)', re.IGNORECASE), ''),
        (re.compile(r'\b(just\s+)', re.IGNORECASE), ''),
        (re.compile(r'\b(really\s+)', re.IGNORECASE), ''),
        (re.compile(r'\b(very\s+)', re.IGNORECASE), ''),
        (re.compile(r'\b(quite\s+)', re.IGNORECASE), ''),
        (re.compile(r'\b(sort of\s+)', re.IGNORECASE), ''),
        (re.compile(r'\b(kind of\s+)', re.IGNORECASE), ''),
        (re.compile(r'\b(in order to\s+)', re.IGNORECASE), 'to '),
        (re.compile(r'\b(due to the fact that\s+)', re.IGNORECASE), 'because '),
        (re.compile(r'\b(at this point in time\s+)', re.IGNORECASE), 'now '),
        (re.compile(r'\b(it is important to note that\s+)', re.IGNORECASE), ''),
    ]

    def _tighten_dialogue(self) -> None:
        """Remove filler words from ALL dialogue to keep it punchy for comics."""
        pages = get_pages(self.spec)
        for i, page in enumerate(pages):
            page_no = page.get("page_no", i + 1)
            for j, panel in enumerate(safe_list(page.get("panels"))):
                entries = panel_text_entries(panel)
                for e in entries:
                    val = (e.get("value") or "").strip()
                    if not val:
                        continue
                    new_val = val
                    for pattern, replacement in self._FILLER_PATTERNS:
                        new_val = pattern.sub(replacement, new_val)
                    # Clean up double spaces
                    new_val = re.sub(r'\s{2,}', ' ', new_val).strip()
                    # Capitalize first letter after cleanup
                    if new_val and new_val[0].islower():
                        new_val = new_val[0].upper() + new_val[1:]

                    if new_val != val and len(new_val) < len(val):
                        e["value"] = new_val
                        self.actions.append(RewriteAction(
                            location=f"page:{page_no} panel:{j+1}", field="dialogue_tighten",
                            old_value=val[:60], new_value=new_val[:60],
                            reason="Trimmed filler words for comics-grade dialogue density.",
                        ))
                # Update panel text
                if entries:
                    panel["text"] = entries



# ---------------------------------------------------------------------
# BUILD ENGINE
# ---------------------------------------------------------------------


def build_unified_comics_engine() -> RulesEngine:
    eng = RulesEngine()

    # P0 - Hard constraints
    eng.register(Rule("P0.SPEC_SCHEMA_SHAPE", Priority.P0_HARD_CONSTRAINTS, "Schema sanity.", rule_spec_schema_shape))
    eng.register(Rule("P0.SCRIPT_MIN_FIELDS", Priority.P0_HARD_CONSTRAINTS, "Drawable/letterable.", rule_script_min_fields))
    eng.register(Rule("P0.HYBRID_LANGUAGE", Priority.P0_HARD_CONSTRAINTS, "Words + pictures integrated.", rule_hybrid_language))
    eng.register(Rule("P0.PAGE_SANITY", Priority.P0_HARD_CONSTRAINTS, "Readable pages.", rule_page_panel_sanity))
    eng.register(Rule("PRIEST.DIALOGUE_CLEANLINESS", Priority.P0_HARD_CONSTRAINTS, "No fake dialect crutches.", rule_priest_dialogue_cleanliness))
    eng.register(Rule("PRIEST.COPY_HEAVY", Priority.P0_HARD_CONSTRAINTS, "Avoid copy-heavy panels.", rule_priest_copy_heavy))
    eng.register(Rule("PRIEST.LAYOUT_TRICKS", Priority.P0_HARD_CONSTRAINTS, "Avoid trick layouts.", rule_priest_layout_tricks))

    # P1 - Story function
    eng.register(Rule("P1.STORY_DEFINITION", Priority.P1_STORY_FUNCTION, "Premise + protagonist + change.", rule_story_definition))
    eng.register(Rule("P1.STORY_STRUCTURE_CONSEQUENCE", Priority.P1_STORY_FUNCTION, "Beats carry consequence.", rule_structure_has_consequences))
    eng.register(Rule("DCOSTA.INCITING_TIMING", Priority.P1_STORY_FUNCTION, "Inciting lands early.", rule_dcosta_inciting_incident_timing))
    eng.register(Rule("MYERS.FAMILY_ROLES", Priority.P1_STORY_FUNCTION, "Character role coverage.", rule_myers_family_roles))
    eng.register(Rule("MYERS.UNITY_ARC", Priority.P1_STORY_FUNCTION, "Psych need stated.", rule_myers_unity_arc))
    eng.register(Rule("PRIEST.MINDLESS_VIOLENCE", Priority.P1_STORY_FUNCTION, "Violence must have consequence.", rule_priest_mindless_violence))
    eng.register(Rule("PRIEST.REPRESENTATION_NOTES", Priority.P1_STORY_FUNCTION, "Representation notes.", rule_priest_world_representation))
    eng.register(Rule("STAN.ONBOARDING_CLARITY", Priority.P1_STORY_FUNCTION, "Orient new readers.", rule_stan_onboarding_clarity))
    eng.register(Rule("STAN.CONFLICT_PER_SCENE", Priority.P1_STORY_FUNCTION, "Conflict per scene.", rule_stan_conflict_per_scene))
    eng.register(Rule("STAN.CHARACTER_DRIVEN_BEATS", Priority.P1_STORY_FUNCTION, "Beats tied to character.", rule_stan_character_driven_beats))

    # P2 - Comics specificity
    eng.register(Rule("P2.CREATING_DRAMA", Priority.P2_COMICS_SPECIFICITY, "Goal/obstacles/stakes.", rule_creating_drama))
    eng.register(Rule("P2.CHARACTERIZATION_CHOICE", Priority.P2_COMICS_SPECIFICITY, "Pressure/choice/cost.", rule_characterization_choice))
    eng.register(Rule("DCOSTA.STAKES_SPECIFICITY", Priority.P2_COMICS_SPECIFICITY, "Stakes concrete.", rule_dcosta_stakes_specificity))
    eng.register(Rule("CONFLICT.FOUR_LEVELS", Priority.P2_COMICS_SPECIFICITY, "Conflict coverage.", rule_conflict_four_levels))
    eng.register(Rule("PRIEST.INTERIOR_SPLASH", Priority.P2_COMICS_SPECIFICITY, "Interior splash check.", rule_priest_interior_splash))
    eng.register(Rule("PANEL.SHOT_VARIETY", Priority.P2_COMICS_SPECIFICITY, "Shot variety across pages.", rule_panel_shot_variety))
    eng.register(Rule("PANEL.SPLASH_BUDGET", Priority.P2_COMICS_SPECIFICITY, "Splash page discipline.", rule_splash_budget))

    # P3 - Pacing (Panel Architecture)
    eng.register(Rule("MOORE.PANEL_TIME_STOPPERS", Priority.P3_DRAMA_AND_PACING, "Flag overlong panels.", rule_moore_panel_time_stoppers))
    eng.register(Rule("PANEL.TEMPO_COHERENCE", Priority.P3_DRAMA_AND_PACING, "Panel counts match page objectives.", rule_panel_tempo_coherence))
    eng.register(Rule("PANEL.WEIGHT_DISTRIBUTION", Priority.P3_DRAMA_AND_PACING, "Panel sizing serves narrative.", rule_panel_weight_distribution))
    eng.register(Rule("PANEL.PAGE_OBJECTIVE", Priority.P3_DRAMA_AND_PACING, "Every page has emotional objective.", rule_page_emotional_objective))
    eng.register(Rule("PANEL.PROGRESSIVE_TEMPO", Priority.P3_DRAMA_AND_PACING, "Tempo shifts across issue.", rule_panel_progressive_tempo))

    # P4 - Transitions
    eng.register(Rule("SLOANE.ABSORPTION_HOOKS", Priority.P4_TRANSITIONS, "Scene hooks.", rule_sloane_absorption_hooks))
    eng.register(Rule("SCENE.OUTCOME_DIRECTION", Priority.P4_TRANSITIONS, "Scene outcomes.", rule_scene_outcome_direction))
    eng.register(Rule("MOORE.TRANSITION_GLUE", Priority.P4_TRANSITIONS, "Transition glue.", rule_moore_transition_glue))
    eng.register(Rule("GRID.PAGE_TURN_REVEALS", Priority.P4_TRANSITIONS, "Page-turn reveals.", rule_comics_grid_page_turns))
    eng.register(Rule("PANEL.PAGE_TURN_POSITION", Priority.P4_TRANSITIONS, "Reveals on left-hand pages.", rule_page_turn_positioning))

    # P5 - Structure shape
    eng.register(Rule("PANEL.TRANSITION_AWARENESS", Priority.P5_STRUCTURE_SHAPE, "Panel transition annotations.", rule_panel_transition_awareness))

    # P6 + P7 - Advisory
    eng.register(Rule("MEDIUM.CONSTRAINTS", Priority.P6_MEDIUM_CONSTRAINTS, "Medium reminders.", rule_medium_constraints))
    eng.register(Rule("PROCESS.REWRITE_LOOP", Priority.P7_REWRITE_LOOP, "Rewrite pass reminder.", rule_rewrite_loop))

    return eng


# ---------------------------------------------------------------------
# AUTO-FIX (safe placeholders — original behavior, preserved)
# ---------------------------------------------------------------------


def apply_safe_autofixes(spec: Dict[str, Any], results: List[RuleResult]) -> None:
    idx = SpecIndex(spec)
    idx.ensure_scenes()

    for r in results:
        if r.rule_id == "SLOANE.ENTRY_HOOK" and r.location and r.location.startswith("scene:"):
            sid = r.location.split("scene:")[-1]
            fix_add_entry_hook(spec, sid, f"[TODO entry hook for {sid}]")
        if r.rule_id == "SLOANE.EXIT_PROPULSION" and r.location and r.location.startswith("scene:"):
            sid = r.location.split("scene:")[-1]
            fix_add_exit_hook(spec, sid, f"[TODO exit hook for {sid}]")

    for r in results:
        if r.rule_id == "SCENE.OUTCOME_DIRECTION":
            for sid in r.evidence.get("scenes", []):
                if sid:
                    fix_add_scene_outcome(spec, sid, f"[TODO outcome for {sid}]")

    if any(r.rule_id == "GRID.PAGE_TURN_REVEALS" and r.level == Level.WARN for r in results):
        pages = get_pages(spec)
        if pages:
            mid_idx = max(0, (len(pages) // 2) - 1)
            page_no = pages[mid_idx].get("page_no", mid_idx + 1)
            if isinstance(page_no, int):
                fix_tag_page_turn(spec, page_no, "auto-suggested")

    for r in results:
        if r.rule_id == "CONFLICT.FOUR_LEVELS":
            missing = " ".join(r.evidence.get("missing", []))
            if "central conflict" in missing and not spec.get("central_conflict"):
                fix_set_central_conflict(spec, "[TBD central conflict]", "")

    if any(r.rule_id == "MYERS.FAMILY_ROLES" and r.level == Level.WARN for r in results):
        for role in ["protagonist", "nemesis", "mentor"]:
            fix_add_character_role(spec, role, f"[TBD {role}]")

    if any(r.rule_id == "MYERS.UNITY_ARC" and r.level == Level.WARN for r in results):
        if not spec.get("protagonist_need"):
            fix_set_protagonist_need(spec, "[TBD psychological need]")


def collect_suggestions(spec: Dict[str, Any], results: List[RuleResult]) -> Dict[str, List[str]]:
    suggestions = ensure_suggestions_bucket(spec)
    idx = SpecIndex(spec)
    scenes = idx.scenes()
    scene_by_id = {s.get("scene_id"): s for s in scenes if s.get("scene_id")}

    for r in results:
        if r.rule_id == "SLOANE.ENTRY_HOOK" and r.location and r.location.startswith("scene:"):
            sid = r.location.split("scene:")[-1]
            suggestions[f"scene:{sid}:entry_hook"] = suggest_entry_hook_options(spec, scene_by_id.get(sid, {}))
        if r.rule_id == "SLOANE.EXIT_PROPULSION" and r.location and r.location.startswith("scene:"):
            sid = r.location.split("scene:")[-1]
            suggestions[f"scene:{sid}:exit_hook"] = suggest_exit_hook_options(spec, scene_by_id.get(sid, {}))

    if any(r.rule_id in {"DCOSTA.STAKES_SPECIFICITY", "P2.CREATING_DRAMA"} for r in results):
        suggestions["stakes:options"] = suggest_stakes_options(spec)

    return suggestions


# ---------------------------------------------------------------------
# Summary / gate
# ---------------------------------------------------------------------


def _summarize(results: List[RuleResult]) -> Dict[str, Any]:
    counts = {"FAIL": 0, "WARN": 0, "NOTE": 0, "PASS": 0}
    by_priority: Dict[str, Dict[str, int]] = {}
    for r in results:
        lvl = r.level.name
        counts[lvl] += 1
        by_priority.setdefault(r.priority.name, {"FAIL": 0, "WARN": 0, "NOTE": 0, "PASS": 0})
        by_priority[r.priority.name][lvl] += 1
    status = "green"
    if counts["FAIL"] > 0:
        status = "red"
    elif counts["WARN"] > 0:
        status = "yellow"
    return {"status": status, "counts": counts, "by_priority": by_priority}


def _priority_from_str(name: str) -> Priority:
    name = name.strip().upper()
    mapping = {
        "P0": Priority.P0_HARD_CONSTRAINTS,
        "P1": Priority.P1_STORY_FUNCTION,
        "P2": Priority.P2_COMICS_SPECIFICITY,
        "P3": Priority.P3_DRAMA_AND_PACING,
        "P4": Priority.P4_TRANSITIONS,
        "P5": Priority.P5_STRUCTURE_SHAPE,
        "P6": Priority.P6_MEDIUM_CONSTRAINTS,
        "P7": Priority.P7_REWRITE_LOOP,
    }
    if name in mapping:
        return mapping[name]
    try:
        return Priority[name]
    except KeyError as e:
        raise ValueError(f"Unknown priority '{name}'. Use P0..P7 or enum name.") from e


def gate_results(results: List[RuleResult], min_priority: Priority) -> Tuple[bool, Optional[str]]:
    failures = [r for r in results if r.level == Level.FAIL and r.priority >= min_priority]
    if not failures:
        return True, None
    lines = []
    for f in failures:
        loc = f" @ {f.location}" if f.location else ""
        lines.append(f"[{f.priority.name}] {f.rule_id}{loc}: {f.message}")
    return False, "RulesEngine gate failed:\n" + "\n".join(lines)



# ---------------------------------------------------------------------
# STUDIO MODE (Idea generation) — preserved from original
# ---------------------------------------------------------------------


def _seed_get(seed: Dict[str, Any], key: str, default: Any) -> Any:
    v = seed.get(key, default)
    return default if v is None else v


def _slug(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-") or "idea"


def _pick(rng: random.Random, xs: List[str]) -> str:
    return xs[rng.randrange(0, len(xs))]


def _unique_obstacles(rng: random.Random, pool: List[str], n: int) -> List[str]:
    pool = list(dict.fromkeys(pool))
    rng.shuffle(pool)
    return pool[: max(2, min(n, len(pool)))]


def build_idea_spec(seed: Dict[str, Any], idea_index: int, rng: random.Random) -> Dict[str, Any]:
    genre = str(_seed_get(seed, "genre", "superhero thriller")).strip()
    tone = str(_seed_get(seed, "tone", "tense, propulsive")).strip()
    page_count = int(_seed_get(seed, "page_count", 22))
    hook = str(_seed_get(seed, "hook", "a stolen secret")).strip()
    setting = str(_seed_get(seed, "setting", "a city where lies demonstrate truth")).strip()

    prot_names = ["Mara", "Kade", "Inez", "Sol", "Juno", "Rafi", "Nyx", "Orion", "Vale", "Sable"]
    nem_names = ["Vesper", "Crown", "Gallows", "Silk", "Archon", "Morrow", "Heliot", "Cinder", "Kestrel", "Proxy"]
    mentor_names = ["Dr. Quill", "Aunt Sera", "Captain Roe", "Brother Ash", "Ms. Rook", "Old Finch", "Agent Lark"]
    jobs = ["courier", "public defender", "street medic", "systems auditor", "museum guard", "tabloid reporter", "union organizer"]
    flaws = ["control issues", "avoidance", "ruthless pragmatism", "fear of intimacy", "pride", "impulsiveness", "self-erasure"]
    needs = [
        "learn to trust help",
        "accept accountability without self-destruction",
        "choose mercy without losing strength",
        "stop confusing control with safety",
        "tell the truth even when it costs love",
        "protect others without martyring the self",
    ]
    goals = [
        "deliver proof before it disappears",
        "stop a public execution staged as justice",
        "recover a stolen memory map",
        "expose the architect behind the coups",
        "save a hostage exchange from collapse",
        "prevent a ritual that rewrites identities",
    ]
    stakes_templates = [
        "If they fail to {goal}, freedom is lost and the city spirals into sanctioned violence under a deadline.",
        "If they fail to {goal}, a family member is erased from public records and time runs out before dawn.",
        "If they fail to {goal}, reputation is destroyed, a mission collapses, and prison becomes inevitable.",
        "If they fail to {goal}, identity is rewritten and the community fractures into factions within hours.",
    ]
    obstacle_pool = [
        "a legal trap that makes the truth inadmissible",
        "a rival who benefits from chaos",
        "a mentor's secret compromise",
        "a resource deadline (power, blood, money, oxygen)",
        "a crowd that turns into a weapon",
        "an internal relapse into the old flaw",
        "an ally who lies to protect you",
        "a surveillance net that punishes movement",
        "a fake peace offer designed to stall you",
        "a second villain with a different agenda",
    ]

    protagonist = _pick(rng, prot_names)
    nemesis = _pick(rng, nem_names)
    mentor = _pick(rng, mentor_names)
    job = _pick(rng, jobs)
    flaw = _pick(rng, flaws)
    need = _pick(rng, needs)
    goal = _pick(rng, goals)

    premise = (
        f"In {setting}, a {job} named {protagonist} runs into {hook} and must {goal}, "
        f"but {nemesis} weaponizes the system to force {protagonist} back into their worst habit."
    )
    target_emotion = f"{tone} with escalating dread and a clean cathartic release."
    protagonist_goal = goal
    protagonist_motivation = f"Protect someone vulnerable and prove they can be more than {flaw}."
    stakes = _pick(rng, stakes_templates).format(goal=protagonist_goal)

    central_conflict = f"{protagonist} vs {nemesis}: control of the truth that decides who is punished and who is protected."
    internal_conflict = f"{protagonist} believes control prevents loss; the story forces them to {need}."
    character_change = f"{protagonist} shifts from {flaw} toward {need} under escalating cost."

    obstacles = _unique_obstacles(rng, obstacle_pool, n=3)

    character_reveals = [
        {
            "pressure": f"{nemesis} offers a shortcut that rewards {flaw}.",
            "choice": f"take the shortcut or accept a slower, riskier path that requires {need}",
            "cost": "lose time and protection; risk someone's safety",
        }
    ]

    subplot_budget = 1 if page_count <= 22 else 2
    subplots = [{"name": "An ally's loyalty test", "purpose": "forces a moral cost"}][:subplot_budget]

    beats = [
        {"name": "Inciting Disturbance", "new_problem": f"{hook} lands in {protagonist}'s hands and {nemesis} notices.",
         "change": "The status quo breaks in public.", "cost": "A deadline starts and allies become liabilities.", "character": protagonist},
        {"name": "Complication", "new_problem": f"A trap makes {protagonist_goal} illegal or impossible by normal means.",
         "change": "The world's rules clamp down.", "cost": "A relationship fractures and resources shrink.", "character": protagonist},
        {"name": "Escalation", "new_problem": f"{nemesis} flips an ally and turns the crowd into a weapon.",
         "change": "The plan becomes a chase with visible consequences.", "cost": "A public loss damages reputation and freedom.", "character": nemesis},
        {"name": "Payoff / Knockout", "new_problem": "Final confrontation at the worst possible moment.",
         "change": f"{protagonist} chooses {need} instead of {flaw}, winning at a personal cost.",
         "cost": "A scar remains; the victory changes future choices.", "character": protagonist},
    ]

    min_pages = max(3, min(6, page_count))
    pages: List[Dict[str, Any]] = []
    for pno in range(1, min_pages + 1):
        panels: List[Dict[str, Any]] = []
        for pn in range(1, 5):
            art = f"{genre} tone. {setting}. Beat hint: {beats[min(3, pno - 1)]['name']}."
            panel: Dict[str, Any] = {"art": art}
            if pno == 1 and pn == 1:
                panel["text"] = [{"type": "caption", "value": f"{protagonist} -- {job}. {setting}."}]
            elif pn == 4:
                panel["text"] = [{"type": "caption", "value": f"Pressure rises: {obstacles[min(len(obstacles)-1, pno-1)]}."}]
            else:
                panel["silent_intent"] = f"Visual beat: {protagonist} advances toward {protagonist_goal} with new cost."
                panel["text"] = []
            panels.append(panel)
        pages.append({"page_no": pno, "page_type": "normal", "panels": panels})

    mid_page_no = pages[max(0, (len(pages) // 2) - 1)]["page_no"]
    pages[mid_page_no - 1]["page_turn_reveal"] = True
    pages[mid_page_no - 1]["reveal_label"] = "The hidden truth changes the plan."

    scenes: List[Dict[str, Any]] = []
    for i, p in enumerate(pages):
        sid = f"S{i+1}"
        nxt = f"S{i+2}" if i + 1 < len(pages) else None
        purpose = ["Disturbance", "Escalation", "Reversal", "Decision", "Cost", "Payoff"][min(i, 5)]
        scenes.append({
            "scene_id": sid, "purpose": purpose,
            "entry_hook": f"{setting}. A change hits: {hook}.",
            "exit_hook": f"Propulsion: {protagonist} commits to {protagonist_goal} as {obstacles[min(len(obstacles)-1, i)]} tightens.",
            "next_scene_id": nxt,
            "conflict": f"{protagonist} vs {nemesis} through {obstacles[min(len(obstacles)-1, i)]}.",
            "outcome": f"Value shift: OPTIONS NARROW; COST UP; TRUST SHIFT on page {p['page_no']}.",
            "page_no": p["page_no"],
        })

    characters = [
        {"role": "protagonist", "name": protagonist},
        {"role": "nemesis", "name": nemesis},
        {"role": "mentor", "name": mentor},
        {"role": "attractor", "name": "An ally who tempts the flaw"},
    ]

    return {
        "title": f"{_slug(genre)}-{idea_index+1}: {protagonist} vs {nemesis}",
        "medium_target": "comics", "genre": genre, "tone": tone, "page_count": page_count,
        "setting": setting, "hook": hook, "premise": premise,
        "protagonist": protagonist, "protagonist_goal": protagonist_goal,
        "protagonist_motivation": protagonist_motivation, "stakes": stakes,
        "central_conflict": central_conflict, "central_conflict_type": seed.get("central_conflict_type", ""),
        "internal_conflict": internal_conflict, "character_change": character_change,
        "target_emotion": target_emotion, "protagonist_need": need, "flaws": [flaw],
        "backstory": f"{protagonist} learned that {flaw} feels like safety after a past loss tied to {setting}.",
        "obstacles": obstacles, "subplots": subplots, "beats": beats,
        "character_reveals": character_reveals, "characters": characters,
        "scenes": scenes, "pages": pages,
    }


def generate_specs(
    engine: RulesEngine, seed: Dict[str, Any], n: int,
    gate_min_priority: Priority, max_attempts: int,
    apply_fixes_mode: str, include_results: bool,
) -> List[Dict[str, Any]]:
    rng_seed = _seed_get(seed, "rng_seed", 1337)
    rng = random.Random(int(rng_seed))
    out: List[Dict[str, Any]] = []

    for idea_i in range(n):
        best_payload: Optional[Dict[str, Any]] = None
        best_score = -10_000

        for attempt in range(max_attempts):
            spec = build_idea_spec(seed, idea_index=idea_i * max_attempts + attempt, rng=rng)
            results = engine.evaluate(spec, stop_on_first_failing_priority=True)

            if apply_fixes_mode in {"safe", "suggest"}:
                apply_safe_autofixes(spec, results)
                if apply_fixes_mode == "suggest":
                    collect_suggestions(spec, results)
                results = engine.evaluate(spec, stop_on_first_failing_priority=True)

            summary = _summarize(results)
            ok, gate_message = gate_results(results, gate_min_priority)

            score = 0
            for r in results:
                if r.level == Level.FAIL:
                    score -= 300 + int(r.priority)
                elif r.level == Level.WARN:
                    score -= 20 + (int(r.priority) // 10)
                elif r.level == Level.NOTE:
                    score -= 2
                else:
                    score += 1

            payload: Dict[str, Any] = {
                "summary": summary,
                "gate": {"min_priority": gate_min_priority.name, "ok": ok, "message": gate_message},
                "spec": spec,
            }
            if include_results:
                payload["results"] = [asdict(r) for r in results]

            if ok:
                out.append(payload)
                break

            if score > best_score:
                best_score = score
                best_payload = payload
        else:
            out.append(best_payload or {
                "summary": {"status": "red"},
                "gate": {"min_priority": gate_min_priority.name, "ok": False, "message": "No attempts."},
                "spec": {},
            })

    return out



# ---------------------------------------------------------------------
# CLI (updated with --rewrite mode)
# ---------------------------------------------------------------------


def run_rewrite(spec: Dict[str, Any], engine: RulesEngine, passes: int = 2, full_report: bool = False) -> Dict[str, Any]:
    """
    The main rewrite pipeline:
    1. Evaluate the spec against all rules
    2. Run the ScriptRewriter to fix flagged issues with real content
    3. Re-evaluate to confirm improvements
    4. Repeat for N passes (diminishing returns after 2)
    """
    all_actions: List[Dict[str, str]] = []

    for pass_num in range(1, passes + 1):
        results = engine.evaluate(spec, stop_on_first_failing_priority=(not full_report))

        rewriter = ScriptRewriter(spec, results, rng=random.Random(42 + pass_num))
        actions = rewriter.run_all()
        all_actions.extend([a.to_dict() for a in actions])

        if not actions:
            break  # Nothing left to fix

    # Final evaluation after all rewrites
    final_results = engine.evaluate(spec, stop_on_first_failing_priority=(not full_report))
    final_summary = _summarize(final_results)

    # Generate panel architecture matrix for the final spec
    panel_architecture = architect_issue(spec)

    return {
        "rewrite_passes_completed": pass_num,
        "total_rewrites": len(all_actions),
        "actions": all_actions,
        "final_summary": final_summary,
        "final_results": [asdict(r) for r in final_results],
        "panel_architecture": panel_architecture,
    }


def main_cli(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Comics Script Editor: Critic + Rewriter + Studio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            MODES:
              Critic (default):  Evaluate a script spec JSON.
                python editor.py --in script.json --out report.json

              Rewrite (NEW):     Analyze AND fix the script with real content.
                python editor.py --in script.json --rewrite --out-spec fixed.json --out report.json

              Studio:            Generate new comic ideas from a seed.
                python editor.py --generate --n 5 --seed '{"genre":"noir","hook":"memory theft"}'

            REWRITE vs AUTO-FIX:
              --apply-fixes safe   -> inserts [TODO] placeholders (structural only)
              --rewrite            -> actually rewrites dialogue, hooks, stakes, art directions
        """),
    )

    # Input/output
    ap.add_argument("--in", dest="infile", default="-", help="Input JSON spec. '-' for stdin.")
    ap.add_argument("--out", dest="outfile", default="-", help="Output JSON report. '-' for stdout.")
    ap.add_argument("--out-spec", dest="outspec", default=None, help="Write the rewritten/fixed spec here.")

    # Mode switches
    ap.add_argument("--gate", default="P1", help="Min priority gate for FAILs (P0..P7). Default: P1.")
    ap.add_argument("--apply-fixes", dest="apply_fixes", choices=["none", "safe", "suggest"], default="none",
                     help="Structural auto-fix mode (placeholders only).")
    ap.add_argument("--rewrite", action="store_true",
                     help="REWRITE MODE: analyze + rewrite weak content with real text.")
    ap.add_argument("--rewrite-passes", type=int, default=2,
                     help="Number of rewrite passes (default: 2). Diminishing returns after 3.")
    ap.add_argument("--full-report", action="store_true",
                     help="Evaluate all priorities even if P0/P1 fails.")

    # Studio mode
    ap.add_argument("--generate", action="store_true", help="Studio Mode: generate idea specs.")
    ap.add_argument("--n", type=int, default=5, help="Number of ideas to generate.")
    ap.add_argument("--max-attempts", type=int, default=6, help="Max attempts per idea slot.")
    ap.add_argument("--seed", type=str, default="{}",
                     help='Seed JSON, e.g. \'{"genre":"noir sci-fi","hook":"memory theft"}\'')
    ap.add_argument("--include-results", action="store_true", help="Include full rule results in output.")

    args = ap.parse_args(argv)
    engine = build_unified_comics_engine()

    try:
        min_pri = _priority_from_str(args.gate)
    except Exception as e:
        _write_json(args.outfile, {"error": f"Invalid --gate: {e}"})
        return 2

    # --- STUDIO MODE ---
    if args.generate:
        try:
            seed = json.loads(args.seed)
            if not isinstance(seed, dict):
                raise ValueError("seed must be a JSON object")
        except Exception as e:
            _write_json(args.outfile, {"error": f"Invalid --seed JSON: {e}"})
            return 2

        generated = generate_specs(
            engine=engine, seed=seed, n=max(1, args.n),
            gate_min_priority=min_pri, max_attempts=max(1, args.max_attempts),
            apply_fixes_mode=args.apply_fixes, include_results=args.include_results,
        )
        out_payload = {
            "mode": "studio", "seed": seed,
            "gate": {"min_priority": min_pri.name},
            "generated_specs": generated,
            "fix_registry": {name: cap.value for name, (cap, _) in FIX_REGISTRY.items()},
        }
        _write_json(args.outfile, out_payload)
        return 0 if all(item["gate"]["ok"] for item in generated) else 2

    # --- CRITIC or REWRITE MODE ---
    spec = _read_spec(args.infile)

    if args.rewrite:
        # REWRITE MODE: the new hotness
        rewrite_report = run_rewrite(
            spec, engine,
            passes=max(1, min(5, args.rewrite_passes)),
            full_report=args.full_report,
        )

        ok, gate_message = gate_results(
            [RuleResult(**{k: (Priority[v] if k == 'priority' else Level[v] if k == 'level' else v)
                          for k, v in r.items()})
             for r in rewrite_report["final_results"]],
            min_pri,
        ) if False else gate_results(
            # Re-evaluate cleanly
            engine.evaluate(spec, stop_on_first_failing_priority=(not args.full_report)),
            min_pri,
        )

        final_results = engine.evaluate(spec, stop_on_first_failing_priority=(not args.full_report))
        summary = _summarize(final_results)

        out_payload = {
            "mode": "rewrite",
            "rewrite_passes": rewrite_report["rewrite_passes_completed"],
            "total_rewrites": rewrite_report["total_rewrites"],
            "rewrite_actions": rewrite_report["actions"],
            "summary": summary,
            "results": [asdict(r) for r in final_results],
            "gate": {"min_priority": min_pri.name, "ok": ok, "message": gate_message},
        }
        _write_json(args.outfile, out_payload)

        if args.outspec:
            with open(args.outspec, "w", encoding="utf-8") as f:
                json.dump(spec, f, indent=2, ensure_ascii=False)
            print(f"Rewritten spec saved to: {args.outspec}", file=sys.stderr)

        # Print human-readable summary to stderr
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"REWRITE COMPLETE", file=sys.stderr)
        print(f"  Passes: {rewrite_report['rewrite_passes_completed']}", file=sys.stderr)
        print(f"  Rewrites applied: {rewrite_report['total_rewrites']}", file=sys.stderr)
        print(f"  Status: {summary['status'].upper()}", file=sys.stderr)
        print(f"  FAILs: {summary['counts']['FAIL']}  WARNs: {summary['counts']['WARN']}  PASSes: {summary['counts']['PASS']}", file=sys.stderr)
        if rewrite_report["actions"]:
            print(f"\n  Key changes:", file=sys.stderr)
            for act in rewrite_report["actions"][:10]:
                print(f"    [{act['location']}] {act['field']}: {act['reason'][:70]}", file=sys.stderr)
            if len(rewrite_report["actions"]) > 10:
                print(f"    ... and {len(rewrite_report['actions']) - 10} more", file=sys.stderr)
        print(f"{'='*60}\n", file=sys.stderr)

        return 0 if ok else 2

    # --- CRITIC MODE (original) ---
    results = engine.evaluate(spec, stop_on_first_failing_priority=(not args.full_report))

    if args.apply_fixes in {"safe", "suggest"}:
        apply_safe_autofixes(spec, results)
        if args.apply_fixes == "suggest":
            collect_suggestions(spec, results)
        results = engine.evaluate(spec, stop_on_first_failing_priority=(not args.full_report))

    summary = _summarize(results)
    ok, gate_message = gate_results(results, min_pri)

    out_payload = {
        "mode": "critic",
        "summary": summary,
        "results": [asdict(r) for r in results],
        "gate": {"min_priority": min_pri.name, "ok": ok, "message": gate_message},
        "suggestions": spec.get("suggestions", {}) if args.apply_fixes == "suggest" else {},
        "fix_registry": {name: cap.value for name, (cap, _) in FIX_REGISTRY.items()},
    }
    _write_json(args.outfile, out_payload)

    if args.outspec:
        with open(args.outspec, "w", encoding="utf-8") as f:
            json.dump(spec, f, indent=2, ensure_ascii=False)

    return 0 if ok else 2


def _read_spec(infile: str) -> Dict[str, Any]:
    if infile == "-":
        data = json.load(sys.stdin)
    else:
        with open(infile, "r", encoding="utf-8") as f:
            data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Input spec must be a JSON object.")
    return data


def _write_json(outfile: str, payload: Dict[str, Any]) -> None:
    if outfile == "-":
        json.dump(payload, sys.stdout, indent=2, ensure_ascii=False)
        print()
        return
    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    raise SystemExit(main_cli())
