"""
Reporting Layer — renders AnalysisResult and DiffResult into human-readable
text or machine-readable JSON.

This module contains ONLY formatting logic. It imports from engine for the
data types but has no analysis logic of its own.

Output formats
--------------
- text  : terminal-friendly, grouped by severity then category
- json  : full machine-readable output (delegated to AnalysisResult.to_dict)

Reporter is a pure-static class to keep it easily testable and mockable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from .passes.base import Finding, FindingCategory, Severity

if TYPE_CHECKING:
    from .engine import AnalysisResult, DiffResult

# ANSI colour codes (disabled automatically on non-TTY via should_colour())
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_RED    = "\033[31m"
_YELLOW = "\033[33m"
_CYAN   = "\033[36m"
_GREEN  = "\033[32m"
_WHITE  = "\033[37m"
_DIM    = "\033[2m"

_SEV_COLOUR = {
    Severity.CRITICAL : _RED + _BOLD,
    Severity.HIGH     : _RED,
    Severity.MEDIUM   : _YELLOW,
    Severity.LOW      : _CYAN,
    Severity.INFO     : _DIM,
}

_CAT_LABEL = {
    FindingCategory.RESOURCE_CONTENTION : "RESOURCE CONTENTION",
    FindingCategory.FRAGILITY           : "FRAGILITY / UNDEFINED BEHAVIOUR",
    FindingCategory.PERFORMANCE         : "PERFORMANCE INEFFICIENCY",
    FindingCategory.CORRECTNESS         : "CORRECTNESS",
}

SEPARATOR_MAJOR = "=" * 72
SEPARATOR_MINOR = "-" * 72


def _sev_tag(sev: Severity, colour: bool = False) -> str:
    label = sev.value.upper().ljust(8)
    if colour:
        return f"{_SEV_COLOUR[sev]}[{label}]{_RESET}"
    return f"[{label}]"


class Reporter:

    @staticmethod
    def text_report(result: "AnalysisResult", colour: bool = False) -> str:
        lines: List[str] = []

        # ── Header ────────────────────────────────────────────────────
        lines.append(SEPARATOR_MAJOR)
        lines.append("  HAL BEHAVIOR ANALYZER - Analysis Report")
        lines.append(SEPARATOR_MAJOR)
        if result.source_path:
            lines.append(f"  Trace   : {result.source_path}")
        if result.arch_hint:
            lines.append(f"  Arch    : {result.arch_hint}")
        lines.append(f"  Nodes   : {len(result.trace.nodes)}")
        lines.append(f"  Resources: {len(result.trace.resources)}")
        lines.append("")

        # ── Summary ───────────────────────────────────────────────────
        findings = result.findings
        if not findings:
            lines.append("  No issues detected.")
            lines.append(SEPARATOR_MAJOR)
            return "\n".join(lines)

        sev_counts = {s: 0 for s in Severity}
        for f in findings:
            sev_counts[f.severity] += 1

        lines.append("  Summary:")
        for sev in reversed(list(Severity)):
            count = sev_counts[sev]
            if count == 0:
                continue
            tag = _sev_tag(sev, colour)
            lines.append(f"    {tag}  {count} finding(s)")
        lines.append("")
        lines.append(SEPARATOR_MAJOR)

        # ── Findings grouped by category ──────────────────────────────
        categories = list(FindingCategory)
        for cat in categories:
            cat_findings = [f for f in findings if f.category == cat]
            if not cat_findings:
                continue

            lines.append("")
            lines.append(f"  >>  {_CAT_LABEL[cat]}")
            lines.append(SEPARATOR_MINOR)

            for finding in cat_findings:
                Reporter._render_finding(lines, finding, colour)
                lines.append("")

        lines.append(SEPARATOR_MAJOR)
        return "\n".join(lines)

    @staticmethod
    def _render_finding(lines: List[str], f: Finding, colour: bool) -> None:
        tag = _sev_tag(f.severity, colour)
        lines.append(f"  {tag}  {f.title}")
        lines.append(f"         Pass: {f.pass_name}")
        if f.related_nodes:
            lines.append(f"         Nodes: {f.related_nodes}")
        # Word-wrap explanation at 70 chars
        lines.append("")
        for para in f.explanation.split("\n"):
            words = para.split()
            row = "         "
            for word in words:
                if len(row) + len(word) + 1 > 78:
                    lines.append(row)
                    row = "         " + word + " "
                else:
                    row += word + " "
            if row.strip():
                lines.append(row)
        if f.suggestion:
            lines.append("")
            lines.append("         -> Suggestion:")
            for para in f.suggestion.split("\n"):
                words = para.split()
                row = "           "
                for word in words:
                    if len(row) + len(word) + 1 > 78:
                        lines.append(row)
                        row = "           " + word + " "
                    else:
                        row += word + " "
                if row.strip():
                    lines.append(row)

    @staticmethod
    def diff_report(diff: "DiffResult", colour: bool = False) -> str:
        lines: List[str] = []
        lines.append(SEPARATOR_MAJOR)
        lines.append("  HAL BEHAVIOR ANALYZER - Diff Report")
        lines.append(SEPARATOR_MAJOR)
        lines.append(f"  Baseline  : {diff.baseline.source_path or '(in-memory)'}")
        lines.append(f"  Candidate : {diff.candidate.source_path or '(in-memory)'}")
        lines.append("")
        lines.append(f"  Regressions (new findings) : {len(diff.new_findings)}")
        lines.append(f"  Improvements (fixed)       : {len(diff.fixed_findings)}")
        lines.append(SEPARATOR_MAJOR)

        if diff.new_findings:
            lines.append("")
            lines.append("  >>  REGRESSIONS (findings in candidate, not in baseline)")
            lines.append(SEPARATOR_MINOR)
            for f in diff.new_findings:
                Reporter._render_finding(lines, f, colour)
                lines.append("")

        if diff.fixed_findings:
            lines.append("")
            lines.append("  >>  IMPROVEMENTS (findings resolved in candidate)")
            lines.append(SEPARATOR_MINOR)
            for f in diff.fixed_findings:
                Reporter._render_finding(lines, f, colour)
                lines.append("")

        if not diff.new_findings and not diff.fixed_findings:
            lines.append("")
            lines.append("  No differences detected between baseline and candidate.")

        lines.append(SEPARATOR_MAJOR)
        return "\n".join(lines)
