"""
Vercel serverless function: Comics Script Editor API

POST /api/analyze
Body JSON:
  {
    "mode": "critic" | "rewrite" | "studio",
    "spec": { ... },              // required for critic/rewrite
    "seed": { ... },              // required for studio
    "gate": "P1",                 // optional, default P1
    "apply_fixes": "none",        // optional: none|safe|suggest
    "rewrite_passes": 2,          // optional, default 2
    "n": 5,                       // studio: number of ideas
    "full_report": false           // optional
  }

Returns JSON with results, rewrite actions, or generated specs.
"""

from http.server import BaseHTTPRequestHandler
import json
import sys
import os
import random

# Add parent dir so we can import the engine module.
# On Vercel, __file__ is inside /var/task/api/ and the engine is at /var/task/
# Locally, it's at ./api/ and the engine is at ./
_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from unified_comics_rules_engine_studio import (
    build_unified_comics_engine,
    apply_safe_autofixes,
    collect_suggestions,
    ScriptRewriter,
    run_rewrite,
    generate_specs,
    gate_results,
    _summarize,
    _priority_from_str,
    FIX_REGISTRY,
)
from dataclasses import asdict


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            req = json.loads(body)
        except Exception as e:
            self._respond(400, {"error": f"Invalid JSON body: {e}"})
            return

        mode = req.get("mode", "critic")
        gate_str = req.get("gate", "P1")
        full_report = bool(req.get("full_report", False))

        try:
            min_pri = _priority_from_str(gate_str)
        except Exception as e:
            self._respond(400, {"error": f"Invalid gate: {e}"})
            return

        engine = build_unified_comics_engine()

        if mode == "studio":
            result = self._handle_studio(req, engine, min_pri)
        elif mode == "rewrite":
            result = self._handle_rewrite(req, engine, min_pri, full_report)
        elif mode == "critic":
            result = self._handle_critic(req, engine, min_pri, full_report)
        else:
            self._respond(400, {"error": f"Unknown mode: {mode}. Use critic, rewrite, or studio."})
            return

        self._respond(200, result)

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors_headers()
        self.end_headers()

    def _handle_critic(self, req, engine, min_pri, full_report):
        spec = req.get("spec")
        if not isinstance(spec, dict):
            return {"error": "spec must be a JSON object"}

        apply_fixes = req.get("apply_fixes", "none")
        results = engine.evaluate(spec, stop_on_first_failing_priority=(not full_report))

        if apply_fixes in ("safe", "suggest"):
            apply_safe_autofixes(spec, results)
            if apply_fixes == "suggest":
                collect_suggestions(spec, results)
            results = engine.evaluate(spec, stop_on_first_failing_priority=(not full_report))

        summary = _summarize(results)
        ok, gate_message = gate_results(results, min_pri)

        return {
            "mode": "critic",
            "summary": summary,
            "results": [asdict(r) for r in results],
            "gate": {"min_priority": min_pri.name, "ok": ok, "message": gate_message},
            "suggestions": spec.get("suggestions", {}) if apply_fixes == "suggest" else {},
            "spec": spec if apply_fixes != "none" else None,
        }

    def _handle_rewrite(self, req, engine, min_pri, full_report):
        spec = req.get("spec")
        if not isinstance(spec, dict):
            return {"error": "spec must be a JSON object"}

        passes = min(5, max(1, int(req.get("rewrite_passes", 2))))
        rewrite_report = run_rewrite(spec, engine, passes=passes, full_report=full_report)

        final_results = engine.evaluate(spec, stop_on_first_failing_priority=(not full_report))
        summary = _summarize(final_results)
        ok, gate_message = gate_results(final_results, min_pri)

        return {
            "mode": "rewrite",
            "rewrite_passes": rewrite_report["rewrite_passes_completed"],
            "total_rewrites": rewrite_report["total_rewrites"],
            "rewrite_actions": rewrite_report["actions"],
            "summary": summary,
            "results": [asdict(r) for r in final_results],
            "gate": {"min_priority": min_pri.name, "ok": ok, "message": gate_message},
            "spec": spec,
        }

    def _handle_studio(self, req, engine, min_pri):
        seed = req.get("seed", {})
        if not isinstance(seed, dict):
            return {"error": "seed must be a JSON object"}

        n = min(20, max(1, int(req.get("n", 5))))
        apply_fixes = req.get("apply_fixes", "safe")
        max_attempts = min(10, max(1, int(req.get("max_attempts", 6))))

        generated = generate_specs(
            engine=engine, seed=seed, n=n,
            gate_min_priority=min_pri,
            max_attempts=max_attempts,
            apply_fixes_mode=apply_fixes,
            include_results=bool(req.get("include_results", False)),
        )

        return {
            "mode": "studio",
            "seed": seed,
            "gate": {"min_priority": min_pri.name},
            "generated_specs": generated,
        }

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _respond(self, status, data):
        self.send_response(status)
        self._cors_headers()
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8"))
