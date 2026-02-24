"""
Performance Inefficiency Analysis Pass

Detects patterns that are correct but carry measurable runtime cost that
could be reduced with better HAL usage.

Checks
------
PE-01  Redundant synchronisation
       Two consecutive blocking waits (or a barrier immediately after a
       blocking wait) with no intervening GPU work — extra sync overhead
       with no ordering benefit.

PE-02  Excessive map/unmap churn
       A buffer is mapped and unmapped more than MAP_CHURN_THRESHOLD times.
       Each map/unmap cycle may trigger cache flushes or coherence operations
       on unified-memory architectures.

PE-03  Small repeated allocations
       Many small allocations (< SMALL_ALLOC_THRESHOLD bytes) are created
       without a pooling/suballocation strategy. GPU allocators typically have
       a minimum page granularity; small allocations waste memory and increase
       allocator lock contention.

PE-04  Synchronous readback in hot path
       READBACK or blocking QUERY operations between SUBMIT and the next
       frame's SUBMIT suggest CPU stalling to read GPU results every frame.

PE-05  Staging buffer left mapped
       A buffer in CPU_VISIBLE memory is mapped but never unmapped — keeping
       a persistent CPU mapping across queue submissions. While legal on
       some HALs (Vulkan allows it), it prevents the driver from decompressing
       or migrating the resource.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from ..ir import BlockingKind, IRNode, IRTrace, MemoryDomain, OpCategory
from .base import AnalysisPass, Finding, FindingCategory, Severity

# Tunable thresholds
MAP_CHURN_THRESHOLD     = 5      # map/unmap cycles for one resource → PE-02
SMALL_ALLOC_THRESHOLD   = 256 * 1024  # 256 KiB → PE-03
SMALL_ALLOC_MIN_COUNT   = 4      # need at least this many small allocs
REDUNDANT_SYNC_WINDOW   = 2      # ops between sync nodes to call them "consecutive"


class PerformancePass(AnalysisPass):

    def run(self, trace: IRTrace) -> List[Finding]:
        findings: List[Finding] = []
        findings.extend(self._check_redundant_sync(trace))
        findings.extend(self._check_map_churn(trace))
        findings.extend(self._check_small_allocations(trace))
        findings.extend(self._check_readback_in_hot_path(trace))
        findings.extend(self._check_persistent_cpu_map(trace))
        return findings

    # ------------------------------------------------------------------
    # PE-01: Redundant synchronisation
    # ------------------------------------------------------------------

    def _check_redundant_sync(self, trace: IRTrace) -> List[Finding]:
        """
        Detect back-to-back blocking sync ops with no GPU work in between.
        """
        findings = []
        sync_nodes = trace.sync_nodes()

        i = 0
        while i < len(sync_nodes) - 1:
            a = sync_nodes[i]
            b = sync_nodes[i + 1]

            # Are they close together?
            gap = b.index - a.index
            if gap > REDUNDANT_SYNC_WINDOW:
                i += 1
                continue

            # Any GPU-side work between them?
            between = trace.nodes_in_range(a.index + 1, b.index - 1)
            gpu_work = [
                n for n in between
                if n.category in (
                    OpCategory.DRAW, OpCategory.COMPUTE, OpCategory.TRANSFER,
                    OpCategory.SUBMIT
                )
            ]
            if gpu_work:
                i += 1
                continue

            # Both blocking?
            both_blocking = (
                a.blocking == BlockingKind.BLOCKING
                and b.blocking == BlockingKind.BLOCKING
            )
            explanation = (
                f"Nodes {self._node_label(a)} and {self._node_label(b)} are both "
                f"{'blocking ' if both_blocking else ''}synchronisation operations "
                f"with no GPU work in between (gap: {gap} ops). "
                f"The second sync is redundant: once the first wait completes, "
                f"the GPU is already idle for the relevant scope. "
                f"Redundant sync ops add CPU overhead (syscall latency) and "
                f"interfere with GPU scheduling."
            )
            findings.append(Finding(
                pass_name=self.name,
                category=FindingCategory.PERFORMANCE,
                severity=Severity.MEDIUM,
                title=f"Redundant sync: {a.op} → {b.op} with no GPU work between",
                explanation=explanation,
                related_nodes=[a.index, b.index],
                suggestion=(
                    "Remove the redundant sync operation. "
                    "If both syncs guard different resources, consider merging them "
                    "into a single, broader fence signal."
                ),
                metadata={"first_sync": a.index, "second_sync": b.index, "gap": gap},
            ))
            i += 2  # skip both nodes; they're a pair

        return findings

    # ------------------------------------------------------------------
    # PE-02: Excessive map/unmap churn
    # ------------------------------------------------------------------

    def _check_map_churn(self, trace: IRTrace) -> List[Finding]:
        findings = []
        for name, lt in trace.lifetimes.items():
            count = len(lt.map_intervals)
            if count < MAP_CHURN_THRESHOLD:
                continue

            related_nodes = []
            for start, end in lt.map_intervals[:6]:
                related_nodes.append(start)
                if end is not None:
                    related_nodes.append(end)

            explanation = (
                f"Resource '{name}' is mapped and unmapped {count} time(s) during its lifetime "
                f"(nodes {lt.alloc_index}–{lt.free_index or 'end'}). "
                f"Each map/unmap cycle may invalidate CPU caches, trigger "
                f"coherence traffic on unified-memory GPUs, and requires "
                f"driver bookkeeping. On some drivers (pre-Adreno 7xx, older PowerVR), "
                f"repeated map/unmap of the same buffer causes measurable stalls."
            )
            findings.append(Finding(
                pass_name=self.name,
                category=FindingCategory.PERFORMANCE,
                severity=Severity.MEDIUM if count < 10 else Severity.HIGH,
                title=f"Excessive map/unmap churn on '{name}' ({count} cycles)",
                explanation=explanation,
                related_nodes=related_nodes,
                suggestion=(
                    "Keep the buffer persistently mapped if the HAL permits it "
                    "(Vulkan allows persistent mapping of HOST_VISIBLE memory). "
                    "Use fence-guarded writes rather than mapping on every update. "
                    "For read-only CPU data, use a staging buffer and batch uploads."
                ),
                metadata={"resource": name, "map_count": count},
            ))
        return findings

    # ------------------------------------------------------------------
    # PE-03: Small repeated allocations
    # ------------------------------------------------------------------

    def _check_small_allocations(self, trace: IRTrace) -> List[Finding]:
        small_allocs: List[IRNode] = []
        for node in trace.nodes:
            if node.category != OpCategory.ALLOC:
                continue
            # Find the resource this alloc creates
            for res in node.resources_write:
                if 0 < res.size_bytes < SMALL_ALLOC_THRESHOLD:
                    small_allocs.append(node)
                    break

        if len(small_allocs) < SMALL_ALLOC_MIN_COUNT:
            return []

        total_bytes = sum(
            res.size_bytes
            for n in small_allocs
            for res in n.resources_write
            if res.size_bytes > 0
        )
        explanation = (
            f"Found {len(small_allocs)} small allocations (each < "
            f"{SMALL_ALLOC_THRESHOLD // 1024} KiB), totalling ~{total_bytes // 1024} KiB. "
            f"GPU memory allocators operate at page granularity (often 64 KiB–2 MiB). "
            f"Each small allocation wastes allocator internal fragmentation and "
            f"adds allocator lock contention. On D3D12 and Vulkan, each allocation "
            f"may consume an OS-level handle (limited to 65536 on Windows)."
        )
        return [Finding(
            pass_name=self.name,
            category=FindingCategory.PERFORMANCE,
            severity=Severity.MEDIUM,
            title=f"{len(small_allocs)} small GPU allocations detected — consider pooling",
            explanation=explanation,
            related_nodes=[n.index for n in small_allocs[:8]],
            suggestion=(
                "Use a suballocator: allocate one large slab and hand out "
                "sub-regions from it. Libraries like VMA (Vulkan Memory Allocator) "
                "or D3D12MA handle this automatically."
            ),
            metadata={
                "small_alloc_count": len(small_allocs),
                "total_bytes": total_bytes,
            },
        )]

    # ------------------------------------------------------------------
    # PE-04: Synchronous readback in hot path
    # ------------------------------------------------------------------

    def _check_readback_in_hot_path(self, trace: IRTrace) -> List[Finding]:
        """
        Detect READBACK or blocking QUERY between two SUBMIT_QUEUE calls,
        indicating a CPU stall to read GPU results on every iteration.
        """
        findings = []
        submit_indices = [n.index for n in trace.nodes if n.category == OpCategory.SUBMIT]

        for i in range(len(submit_indices) - 1):
            start = submit_indices[i]
            end   = submit_indices[i + 1]

            readbacks = [
                n for n in trace.nodes_in_range(start + 1, end - 1)
                if n.category == OpCategory.QUERY and n.blocking == BlockingKind.BLOCKING
            ]
            if not readbacks:
                continue

            explanation = (
                f"Between SUBMIT at node {start} and the next SUBMIT at node {end}, "
                f"{len(readbacks)} blocking readback/query operation(s) were detected: "
                f"{self._nodes_label(readbacks)}. "
                f"Blocking readbacks force the CPU to wait for GPU results every frame, "
                f"creating a hard sync point that eliminates pipelining benefits. "
                f"This pattern is common in naive benchmarking code but is very costly "
                f"in production rendering loops."
            )
            findings.append(Finding(
                pass_name=self.name,
                category=FindingCategory.PERFORMANCE,
                severity=Severity.HIGH,
                title="Blocking readback between submissions (GPU→CPU stall every frame)",
                explanation=explanation,
                related_nodes=[start] + [n.index for n in readbacks] + [end],
                suggestion=(
                    "Use latency-hidden readback: read frame N-2 results while "
                    "submitting frame N. Use timestamp query pools with deferred "
                    "CPU readback rather than synchronous queries."
                ),
                metadata={"readback_count": len(readbacks)},
            ))
        return findings

    # ------------------------------------------------------------------
    # PE-05: Staging buffer left persistently mapped across submissions
    # ------------------------------------------------------------------

    def _check_persistent_cpu_map(self, trace: IRTrace) -> List[Finding]:
        """
        Detect CPU_VISIBLE resources that are mapped but never unmapped across
        a queue submission — could interfere with driver resource management.
        """
        findings = []

        for name, lt in trace.lifetimes.items():
            if lt.resource.memory != MemoryDomain.CPU_VISIBLE:
                continue
            for map_start, map_end in lt.map_intervals:
                if map_end is not None:
                    continue   # properly unmapped
                # Is there a SUBMIT between map_start and end of trace?
                submits = [
                    n for n in trace.nodes_in_range(
                        map_start + 1, len(trace.nodes) - 1
                    )
                    if n.category == OpCategory.SUBMIT
                ]
                if not submits:
                    continue
                explanation = (
                    f"Resource '{name}' (CPU_VISIBLE) is mapped at node {map_start} "
                    f"and remains mapped across {len(submits)} queue submission(s) "
                    f"without being unmapped. While Vulkan allows persistent mapping "
                    f"of HOST_VISIBLE memory, some drivers (pre-Qualcomm Adreno 6xx, "
                    f"older ARM Mali) may defer resource migration or decompression "
                    f"while a buffer is persistently mapped, leading to cache "
                    f"performance degradation or bandwidth inflation."
                )
                findings.append(Finding(
                    pass_name=self.name,
                    category=FindingCategory.PERFORMANCE,
                    severity=Severity.LOW,
                    title=f"Staging buffer '{name}' persistently mapped across submission(s)",
                    explanation=explanation,
                    related_nodes=[map_start] + [n.index for n in submits[:3]],
                    suggestion=(
                        "If persistent mapping is intentional, document it and ensure "
                        "coherence domains are correct (use HOST_COHERENT or explicit flushes). "
                        "Otherwise, unmap before submission and remap after the GPU is done."
                    ),
                    metadata={"resource": name, "map_start": map_start},
                ))
        return findings
