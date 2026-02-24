"""
Analysis Engine — the public API surface for the HAL Behavior Analyzer.

This module wires together the parser, passes, and reporter into a single
clean entry point. CLI and GUI both call only this module.

Usage
-----
    from hal_analyzer.core.engine import AnalysisEngine, AnalysisResult

    engine = AnalysisEngine()
    result = engine.analyze_file("trace.json", arch_hint="ampere")
    print(result.to_text_report())
    print(result.to_json())

    # Diff two traces
    diff = engine.diff_files("old.json", "new.json")

Extending
---------
To add a new analysis pass:
    1. Create the pass module (see passes/base.py for the contract).
    2. Import and add it to DEFAULT_PASSES below.
    No other changes required.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Type

from .ir import IRTrace
from .parser import HALParser
from .passes import (
    AnalysisPass,
    BlockingCpuHotpathPass,
    Finding,
    FindingCategory,
    FragilityPass,
    PerformancePass,
    ResourceContentionPass,
    Severity,
)
from .reporter import Reporter

# ──────────────────────────────────────────────────────────────────────────────
# Default pass registry — add new passes here
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_PASSES: List[Type[AnalysisPass]] = [
    ResourceContentionPass,
    FragilityPass,
    PerformancePass,
    BlockingCpuHotpathPass,
]


@dataclass
class AnalysisResult:
    """
    Complete output of one analysis run.

    Attributes
    ----------
    trace       : The parsed and lifted IR (read-only after construction).
    findings    : All findings from all passes, in emission order.
    arch_hint   : Architecture the trace was analysed against (if any).
    source_path : File path of the input trace (if applicable).
    """
    trace       : IRTrace
    findings    : List[Finding]     = field(default_factory=list)
    arch_hint   : Optional[str]     = None
    source_path : Optional[str]     = None

    # ------------------------------------------------------------------
    # Slicing helpers
    # ------------------------------------------------------------------

    def by_severity(self, min_severity: Severity = Severity.INFO) -> List[Finding]:
        order = list(Severity)
        threshold = order.index(min_severity)
        return [f for f in self.findings if order.index(f.severity) >= threshold]

    def by_category(self, category: FindingCategory) -> List[Finding]:
        return [f for f in self.findings if f.category == category]

    def critical_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.CRITICAL)

    def high_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.HIGH)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_path": self.source_path,
            "arch_hint": self.arch_hint,
            "summary": {
                "total_findings": len(self.findings),
                "critical": self.critical_count(),
                "high": self.high_count(),
                "medium": sum(1 for f in self.findings if f.severity == Severity.MEDIUM),
                "low": sum(1 for f in self.findings if f.severity == Severity.LOW),
                "info": sum(1 for f in self.findings if f.severity == Severity.INFO),
            },
            "findings": [f.to_dict() for f in self.findings],
            "trace": self.trace.to_dict(),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_text_report(self) -> str:
        return Reporter.text_report(self)


@dataclass
class DiffResult:
    """
    Side-by-side comparison of two analysis results.

    Attributes
    ----------
    baseline    : First (older/reference) result.
    candidate   : Second (newer/candidate) result.
    new_findings    : Findings in candidate not present in baseline (regressions).
    fixed_findings  : Findings in baseline not present in candidate (improvements).
    """
    baseline        : AnalysisResult
    candidate       : AnalysisResult
    new_findings    : List[Finding]     = field(default_factory=list)
    fixed_findings  : List[Finding]     = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline": self.baseline.source_path,
            "candidate": self.candidate.source_path,
            "regressions": len(self.new_findings),
            "improvements": len(self.fixed_findings),
            "new_findings": [f.to_dict() for f in self.new_findings],
            "fixed_findings": [f.to_dict() for f in self.fixed_findings],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_text_report(self) -> str:
        return Reporter.diff_report(self)


# ──────────────────────────────────────────────────────────────────────────────
# Engine
# ──────────────────────────────────────────────────────────────────────────────

class AnalysisEngine:
    """
    Orchestrates parsing → pass execution → result assembly.

    The engine is stateless; each analyze_* call is independent.

    Parameters
    ----------
    passes  : Override the default pass list. Useful for testing or
              when you want to run a subset of passes.
    parser  : Override the default HALParser. Useful for custom backends.
    """

    def __init__(
        self,
        passes: Optional[List[AnalysisPass]] = None,
        parser: Optional[HALParser] = None,
    ) -> None:
        self._passes: List[AnalysisPass] = (
            passes if passes is not None
            else [cls() for cls in DEFAULT_PASSES]
        )
        self._parser: HALParser = parser or HALParser()

    # ------------------------------------------------------------------
    # Public analysis API
    # ------------------------------------------------------------------

    def analyze_file(
        self,
        path: str | Path,
        arch_hint: Optional[str] = None,
    ) -> AnalysisResult:
        """Parse a JSON trace file and run all passes."""
        trace = self._parser.parse_file(path)
        if arch_hint:
            trace.source_arch = arch_hint
        return self._run_passes(trace, source_path=str(path))

    def analyze_dict(
        self,
        data: Dict[str, Any],
        arch_hint: Optional[str] = None,
        source_path: Optional[str] = None,
    ) -> AnalysisResult:
        """Parse a trace from an in-memory dict and run all passes."""
        trace = self._parser.parse_dict(data)
        if arch_hint:
            trace.source_arch = arch_hint
        return self._run_passes(trace, source_path=source_path)

    def analyze_json(
        self,
        text: str,
        arch_hint: Optional[str] = None,
    ) -> AnalysisResult:
        """Parse a trace from a JSON string and run all passes."""
        trace = self._parser.parse_json(text)
        if arch_hint:
            trace.source_arch = arch_hint
        return self._run_passes(trace)

    # ------------------------------------------------------------------
    # Diff / comparison API
    # ------------------------------------------------------------------

    def diff_files(
        self,
        baseline_path: str | Path,
        candidate_path: str | Path,
        arch_hint: Optional[str] = None,
    ) -> DiffResult:
        """Compare two trace files and highlight regressions/improvements."""
        baseline  = self.analyze_file(baseline_path, arch_hint=arch_hint)
        candidate = self.analyze_file(candidate_path, arch_hint=arch_hint)
        return self._diff(baseline, candidate)

    def diff_results(
        self,
        baseline: AnalysisResult,
        candidate: AnalysisResult,
    ) -> DiffResult:
        """Compare two already-computed results."""
        return self._diff(baseline, candidate)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_passes(
        self,
        trace: IRTrace,
        source_path: Optional[str] = None,
    ) -> AnalysisResult:
        all_findings: List[Finding] = []
        for p in self._passes:
            try:
                findings = p.run(trace)
                all_findings.extend(findings)
            except Exception as exc:
                # A failing pass must not crash the entire analysis run.
                import logging
                logging.getLogger(__name__).error(
                    "Pass '%s' raised an unexpected error: %s", p.name, exc,
                    exc_info=True,
                )
        return AnalysisResult(
            trace=trace,
            findings=all_findings,
            arch_hint=trace.source_arch,
            source_path=source_path,
        )

    @staticmethod
    def _diff(baseline: AnalysisResult, candidate: AnalysisResult) -> DiffResult:
        """
        Compute the symmetric difference of findings.

        Two findings are considered "the same" if they share the same
        (pass_name, category, title) — a heuristic that works well for
        structural changes without needing stable finding IDs.
        """
        def _key(f: Finding) -> tuple:
            return (f.pass_name, f.category, f.title)

        baseline_keys  = {_key(f) for f in baseline.findings}
        candidate_keys = {_key(f) for f in candidate.findings}

        new_findings   = [f for f in candidate.findings if _key(f) not in baseline_keys]
        fixed_findings = [f for f in baseline.findings  if _key(f) not in candidate_keys]

        return DiffResult(
            baseline=baseline,
            candidate=candidate,
            new_findings=new_findings,
            fixed_findings=fixed_findings,
        )
