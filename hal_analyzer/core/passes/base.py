"""
Base classes for all analysis passes.

Design contract
---------------
- Passes are stateless after construction (all state lives in Finding lists).
- A pass receives an IRTrace and emits a list of Finding objects.
- Passes MUST NOT modify the IRTrace or any of its nodes/resources.
- Each pass is independently runnable and composable.

Adding a new pass
-----------------
1. Create a new module under hal_analyzer/core/passes/
2. Subclass AnalysisPass
3. Implement run(trace) → List[Finding]
4. Register in hal_analyzer/core/passes/__init__.py
5. Add to the default pass list in hal_analyzer/core/engine.py
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from ..ir import IRNode, IRTrace


class Severity(Enum):
    INFO     = "info"       # informational, no action required
    LOW      = "low"        # minor issue, worth noting
    MEDIUM   = "medium"     # moderate issue, should be addressed
    HIGH     = "high"       # significant issue, likely to cause problems
    CRITICAL = "critical"   # definite correctness or safety violation

    def __lt__(self, other: "Severity") -> bool:
        order = [Severity.INFO, Severity.LOW, Severity.MEDIUM,
                 Severity.HIGH, Severity.CRITICAL]
        return order.index(self) < order.index(other)


class FindingCategory(Enum):
    RESOURCE_CONTENTION = "resource_contention"
    FRAGILITY           = "fragility"
    PERFORMANCE         = "performance"
    CORRECTNESS         = "correctness"   # reserved for future passes


@dataclass
class Finding:
    """
    One diagnostic finding emitted by an analysis pass.

    Attributes
    ----------
    pass_name       : Name of the pass that produced this finding.
    category        : Broad category for grouping/filtering.
    severity        : How serious the issue is.
    title           : Short one-line description (like a compiler warning title).
    explanation     : Detailed human-readable description with context.
    related_nodes   : IR node indices directly associated with this finding.
    suggestion      : Optional remediation advice.
    metadata        : Arbitrary key/value for machine consumption.
    """
    pass_name       : str
    category        : FindingCategory
    severity        : Severity
    title           : str
    explanation     : str
    related_nodes   : List[int]         = field(default_factory=list)
    suggestion      : Optional[str]     = None
    metadata        : Dict[str, Any]    = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pass": self.pass_name,
            "category": self.category.value,
            "severity": self.severity.value,
            "title": self.title,
            "explanation": self.explanation,
            "related_nodes": self.related_nodes,
            "suggestion": self.suggestion,
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def __repr__(self) -> str:
        nodes = f"nodes={self.related_nodes}" if self.related_nodes else ""
        return (
            f"[{self.severity.value.upper()}] {self.pass_name}: {self.title}"
            + (f" ({nodes})" if nodes else "")
        )


class AnalysisPass(ABC):
    """
    Abstract base for all analysis passes.

    Subclasses implement `run(trace)` which returns a list of Findings.
    The pass name is taken from the class name by default.
    """

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def run(self, trace: IRTrace) -> List[Finding]:
        """
        Analyse the trace and return findings.

        Parameters
        ----------
        trace : IRTrace — the fully constructed IR (read-only)

        Returns
        -------
        List[Finding] — may be empty if no issues are detected.
        """
        ...

    # ------------------------------------------------------------------
    # Shared helper utilities available to all passes
    # ------------------------------------------------------------------

    @staticmethod
    def _node_label(node: IRNode) -> str:
        """Human-readable label for a node, used in explanations."""
        return f"[{node.index}] {node.op}"

    @staticmethod
    def _nodes_label(nodes: List[IRNode]) -> str:
        return ", ".join(f"[{n.index}] {n.op}" for n in nodes)
