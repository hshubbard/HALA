"""
Blocking CPU Hotpath Pass (PE-06)

Detects CPU-visible resource accesses that block immediately after a GPU
queue submission. This creates a CPU stall inside the GPU hot path,
preventing the driver from pipelining the next frame's work.

Check
-----
PE-06  Blocking CPU access in GPU hot path
       Within 3 IRNodes after a SUBMIT_QUEUE, a blocking operation
       touches a CPU_VISIBLE resource. The CPU stalls before the GPU
       has had any opportunity to make forward progress.

       Examples of this pattern:
         - MAP_BUFFER (blocking) on a CPU_VISIBLE staging buffer
           immediately after submitting a compute or render pass.
         - READBACK (blocking) on a result buffer within the same
           dispatch/present window as the preceding submit.

       This differs from PE-04 (blocking readback in hot path) in that
       PE-04 targets blocking query/readback ops between two submits;
       PE-06 targets any blocking op on a CPU_VISIBLE resource within
       the 3-node window immediately following a submit, regardless of
       whether another submit follows.
"""

from __future__ import annotations

from typing import List

from ..ir import BlockingKind, IRNode, IRTrace, MemoryDomain, OpCategory
from .base import AnalysisPass, Finding, FindingCategory, Severity

# How many nodes after a SUBMIT_QUEUE to inspect for blocking CPU access
HOTPATH_WINDOW = 3


class BlockingCpuHotpathPass(AnalysisPass):

    def run(self, trace: IRTrace) -> List[Finding]:
        findings: List[Finding] = []

        for node in trace.nodes:
            if node.category != OpCategory.SUBMIT:
                continue

            # Inspect the next HOTPATH_WINDOW nodes
            window_end = min(node.index + HOTPATH_WINDOW, len(trace.nodes) - 1)
            for candidate in trace.nodes_in_range(node.index + 1, window_end):
                if not self._is_blocking_cpu_access(candidate):
                    continue

                explanation = (
                    f"Node {self._node_label(candidate)} performs a blocking "
                    f"operation on a CPU_VISIBLE resource only "
                    f"{candidate.index - node.index} node(s) after "
                    f"SUBMIT_QUEUE at node {self._node_label(node)}. "
                    f"The CPU blocks before the GPU has had any opportunity "
                    f"to make forward progress on the submitted work. "
                    f"This stall occurs inside the GPU hot path and prevents "
                    f"the driver from pipelining the next frame or dispatch. "
                    f"On tiled-memory architectures (ARM Mali, Apple GPU) this "
                    f"pattern can also evict in-flight tile buffers, causing "
                    f"a full pipeline flush."
                )
                findings.append(Finding(
                    pass_name=self.name,
                    category=FindingCategory.PERFORMANCE,
                    severity=Severity.HIGH,
                    title="Blocking CPU access in GPU hot path",
                    explanation=explanation,
                    related_nodes=[node.index, candidate.index],
                    suggestion=(
                        "Avoid blocking CPU access immediately after GPU submission. "
                        "Consider async mapping or staging buffers. "
                        "If a readback is required, use latency-hidden readback: "
                        "read frame N-2 results while submitting frame N."
                    ),
                    metadata={
                        "submit_node": node.index,
                        "offending_node": candidate.index,
                        "gap_ops": candidate.index - node.index,
                    },
                ))

        return findings

    @staticmethod
    def _is_blocking_cpu_access(node: IRNode) -> bool:
        """
        Return True if this node both:
          1. Blocks the calling thread (BLOCKING or inferred blocking), and
          2. Touches at least one CPU_VISIBLE resource.
        """
        if node.blocking != BlockingKind.BLOCKING:
            return False

        for res in node.all_resources:
            if res.cpu_visible or res.memory == MemoryDomain.CPU_VISIBLE:
                return True

        # Also flag ops whose category implies CPU involvement on any resource,
        # even if memory domain is UNKNOWN (MAP is always a CPU operation)
        if node.category in (OpCategory.MAP, OpCategory.QUERY):
            return True

        return False
