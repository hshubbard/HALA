"""
Resource Contention Analysis Pass

Detects patterns that indicate exclusive, long-lived resource ownership
and serialised pipeline structures that prevent effective parallelism.

Checks
------
RC-01  Long-lived exclusive resource ownership
       A resource is live for more than LONG_LIFETIME_THRESHOLD operations
       and is never released before new work is submitted.

RC-02  Serialised submission pattern
       Queue submissions are always immediately followed by a blocking
       fence/semaphore wait, creating a CPU↔GPU ping-pong pattern.

RC-03  Resource allocation without timely release
       Resources allocated but never freed within the trace (leaked or
       kept alive across frame boundaries without justification).

RC-04  Concurrent queue starvation
       All work is submitted to a single queue even though multiple
       queues are referenced, suggesting missed parallelism opportunity.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from ..ir import IRTrace, IRNode, OpCategory, BlockingKind, ResourceLifetime
from .base import AnalysisPass, Finding, FindingCategory, Severity

# Heuristic thresholds — tunable without touching logic
LONG_LIFETIME_THRESHOLD = 20     # ops between alloc and free → RC-01
SYNC_PROXIMITY_THRESHOLD = 3     # ops after SUBMIT before WAIT → RC-02
LARGE_ALLOC_BYTES = 64 * 1024 * 1024   # 64 MiB — flag large leaked allocs


class ResourceContentionPass(AnalysisPass):

    def run(self, trace: IRTrace) -> List[Finding]:
        findings: List[Finding] = []
        findings.extend(self._check_long_lived_resources(trace))
        findings.extend(self._check_serialised_submissions(trace))
        findings.extend(self._check_leaked_resources(trace))
        findings.extend(self._check_queue_starvation(trace))
        return findings

    # ------------------------------------------------------------------
    # RC-01: Long-lived exclusive resource ownership
    # ------------------------------------------------------------------

    def _check_long_lived_resources(self, trace: IRTrace) -> List[Finding]:
        findings = []
        for name, lt in trace.lifetimes.items():
            if lt.free_index is None:
                continue   # handle in RC-03
            duration = lt.free_index - lt.alloc_index
            if duration < LONG_LIFETIME_THRESHOLD:
                continue

            # Check whether any submit occurs during the lifetime
            submits_in_range = [
                n for n in trace.nodes_in_range(lt.alloc_index, lt.free_index)
                if n.category == OpCategory.SUBMIT
            ]
            if not submits_in_range:
                continue   # no submission during lifetime — less interesting

            explanation = (
                f"Resource '{name}' ({lt.resource.kind.value}) is held exclusively "
                f"for {duration} operations (nodes {lt.alloc_index}–{lt.free_index}). "
                f"During this period, {len(submits_in_range)} queue submission(s) occur. "
                f"Long-lived exclusive ownership prevents other consumers from accessing "
                f"this resource and may serialise the pipeline."
            )
            findings.append(Finding(
                pass_name=self.name,
                category=FindingCategory.RESOURCE_CONTENTION,
                severity=Severity.MEDIUM if duration < 50 else Severity.HIGH,
                title=f"Long-lived exclusive ownership of '{name}'",
                explanation=explanation,
                related_nodes=[lt.alloc_index] + [n.index for n in submits_in_range[:3]],
                suggestion=(
                    "Consider splitting the resource into smaller, shorter-lived "
                    "allocations, or use a ring buffer / pool to allow concurrent access."
                ),
                metadata={"resource": name, "lifetime_ops": duration},
            ))
        return findings

    # ------------------------------------------------------------------
    # RC-02: Serialised submission pattern (CPU↔GPU ping-pong)
    # ------------------------------------------------------------------

    def _check_serialised_submissions(self, trace: IRTrace) -> List[Finding]:
        """
        Detect: SUBMIT_QUEUE immediately followed (within threshold) by
        a blocking WAIT_FENCE or WAIT_SEMAPHORE — classic CPU stall pattern.
        """
        findings = []
        submit_nodes = [n for n in trace.nodes if n.category == OpCategory.SUBMIT]

        serial_pairs: List[Tuple[IRNode, IRNode]] = []
        for submit in submit_nodes:
            window_end = min(submit.index + SYNC_PROXIMITY_THRESHOLD, len(trace.nodes) - 1)
            for candidate in trace.nodes_in_range(submit.index + 1, window_end):
                if (
                    candidate.category == OpCategory.SYNC
                    and candidate.blocking == BlockingKind.BLOCKING
                ):
                    serial_pairs.append((submit, candidate))
                    break

        if len(serial_pairs) < 2:
            # A single serialised pair may be intentional; flag repeated pattern
            return findings

        explanation = (
            f"Detected {len(serial_pairs)} consecutive SUBMIT→WAIT pairs within "
            f"{SYNC_PROXIMITY_THRESHOLD} operations of each other. This pattern "
            f"forces the CPU to stall after every submission, effectively serialising "
            f"CPU and GPU work. Modern HAL APIs are designed for pipelining — the CPU "
            f"should be preparing the next frame while the GPU executes the previous."
        )
        related = []
        for s, w in serial_pairs[:4]:
            related += [s.index, w.index]

        findings.append(Finding(
            pass_name=self.name,
            category=FindingCategory.RESOURCE_CONTENTION,
            severity=Severity.HIGH,
            title="Serialised SUBMIT→WAIT pattern prevents CPU/GPU pipelining",
            explanation=explanation,
            related_nodes=related,
            suggestion=(
                "Use a multi-buffering strategy (double/triple buffering) and only "
                "wait for frame N-2 while preparing frame N. "
                "Consider using timeline semaphores for fine-grained synchronisation."
            ),
            metadata={"serialised_pair_count": len(serial_pairs)},
        ))
        return findings

    # ------------------------------------------------------------------
    # RC-03: Resource allocation without timely release (leaks)
    # ------------------------------------------------------------------

    def _check_leaked_resources(self, trace: IRTrace) -> List[Finding]:
        findings = []
        for name, lt in trace.lifetimes.items():
            if not lt.is_live_at_end:
                continue

            size = lt.resource.size_bytes
            severity = Severity.LOW
            if size >= LARGE_ALLOC_BYTES:
                severity = Severity.HIGH
            elif size > 0:
                severity = Severity.MEDIUM

            explanation = (
                f"Resource '{name}' ({lt.resource.kind.value}) allocated at node "
                f"{lt.alloc_index} is never freed within this trace. "
            )
            if size >= LARGE_ALLOC_BYTES:
                explanation += (
                    f"At {size // (1024*1024)} MiB, this is a large allocation. "
                    f"If this resource crosses frame boundaries, ensure lifetime "
                    f"management is handled externally."
                )
            else:
                explanation += (
                    "If this trace represents a complete frame or operation unit, "
                    "this may indicate a resource leak."
                )

            findings.append(Finding(
                pass_name=self.name,
                category=FindingCategory.RESOURCE_CONTENTION,
                severity=severity,
                title=f"Resource '{name}' never freed in trace",
                explanation=explanation,
                related_nodes=[lt.alloc_index],
                suggestion=(
                    "Ensure every allocation has a matching free within the same "
                    "logical unit. Use RAII wrappers or explicit lifetime tracking."
                ),
                metadata={"resource": name, "size_bytes": size},
            ))
        return findings

    # ------------------------------------------------------------------
    # RC-04: Concurrent queue starvation
    # ------------------------------------------------------------------

    def _check_queue_starvation(self, trace: IRTrace) -> List[Finding]:
        """
        If multiple distinct queues are referenced but all submits go to
        one queue, there may be missed async parallelism.
        """
        findings = []
        all_queues: Set[str] = set()
        submit_queues: Dict[str, int] = {}

        for node in trace.nodes:
            if node.queue:
                all_queues.add(node.queue)
            if node.category == OpCategory.SUBMIT and node.queue:
                submit_queues[node.queue] = submit_queues.get(node.queue, 0) + 1

        if len(all_queues) <= 1 or len(submit_queues) <= 1:
            return findings  # only one queue in use — not starvation

        total_submits = sum(submit_queues.values())
        dominant = max(submit_queues, key=submit_queues.__getitem__)
        dominant_pct = 100 * submit_queues[dominant] / total_submits

        if dominant_pct < 85:
            return findings  # reasonably balanced

        idle_queues = all_queues - set(submit_queues.keys())
        explanation = (
            f"{dominant_pct:.0f}% of all submissions ({submit_queues[dominant]}/{total_submits}) "
            f"target the '{dominant}' queue. "
            f"The following queues are referenced but receive no submissions: "
            f"{', '.join(idle_queues) or '(none)'}. "
            f"Async compute and transfer queues can overlap with graphics work on most modern GPUs."
        )
        findings.append(Finding(
            pass_name=self.name,
            category=FindingCategory.RESOURCE_CONTENTION,
            severity=Severity.MEDIUM,
            title=f"Queue starvation: {dominant_pct:.0f}% of work on '{dominant}' queue",
            explanation=explanation,
            related_nodes=[],
            suggestion=(
                "Evaluate moving compute dispatches to an async compute queue "
                "and DMA transfers to a dedicated transfer queue to exploit "
                "hardware parallelism (available on GCN/RDNA, Ampere+, etc.)."
            ),
            metadata={
                "dominant_queue": dominant,
                "dominant_pct": dominant_pct,
                "queue_distribution": submit_queues,
            },
        ))
        return findings
