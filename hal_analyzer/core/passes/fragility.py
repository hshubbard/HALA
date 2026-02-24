"""
Fragility / Undefined Behaviour Analysis Pass

Detects patterns that are correct on one platform or GPU vendor but may
silently misbehave on another — the category of bugs that only manifest
during porting, driver updates, or hardware generation changes.

Checks
------
FR-01  Implicit ordering assumption after SUBMIT
       Work submitted to a queue is assumed complete on the CPU side without
       a corresponding fence/semaphore wait. Any subsequent CPU read or free
       of a resource written by that work is a data race on most HALs.

FR-02  CPU access to GPU_LOCAL memory
       Mapping a buffer declared as GPU_LOCAL (device-local) is undefined
       behaviour on discrete GPUs. On integrated GPUs it may work via cache
       coherence but is non-portable and bypasses the HAL contract.

FR-03  Missing post-submit barrier before resource reuse
       A resource is written by a GPU op, then immediately read by another
       GPU op without an intervening PIPELINE_BARRIER or MEMORY_BARRIER.
       Vendor drivers may or may not insert implicit barriers.

FR-04  Vendor-specific architecture hints in trace
       The trace declares architecture-specific hints (e.g. arch=["ampere"]).
       Code paths gated on hardware generation are fragile and require
       explicit porting work for future hardware.

FR-05  Infinite timeout on synchronisation primitive
       Using WAIT_FENCE / WAIT_SEMAPHORE with timeout=infinite creates an
       unrecoverable hang if the signal never arrives (e.g. GPU TDR, driver
       crash). This is especially fragile in production code.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from ..ir import (
    BlockingKind, IRNode, IRTrace, MemoryDomain, OpCategory, ResourceKind
)
from .base import AnalysisPass, Finding, FindingCategory, Severity


class FragilityPass(AnalysisPass):

    def run(self, trace: IRTrace) -> List[Finding]:
        findings: List[Finding] = []
        findings.extend(self._check_implicit_ordering(trace))
        findings.extend(self._check_cpu_access_gpu_local(trace))
        findings.extend(self._check_missing_barriers(trace))
        findings.extend(self._check_arch_specific_hints(trace))
        findings.extend(self._check_infinite_timeout(trace))
        return findings

    # ------------------------------------------------------------------
    # FR-01: Implicit ordering assumption after SUBMIT
    # ------------------------------------------------------------------

    def _check_implicit_ordering(self, trace: IRTrace) -> List[Finding]:
        """
        Detect: SUBMIT_QUEUE followed by a CPU-side FREE or MAP of a resource
        that was written by the submitted work, without an intervening fence wait.
        """
        findings = []

        for i, node in enumerate(trace.nodes):
            if node.category != OpCategory.SUBMIT:
                continue

            # Resources written before/at this submit (approximated by resources
            # written by all nodes between last sync and this submit)
            last_sync_idx = 0
            for j in range(i - 1, -1, -1):
                if trace.nodes[j].sync_point:
                    last_sync_idx = j + 1
                    break

            gpu_written: Set[str] = set()
            for k in range(last_sync_idx, i + 1):
                for r in trace.nodes[k].resources_write:
                    gpu_written.add(r.name)

            # Scan forward: has there been a blocking wait before we hit a
            # CPU-side op touching one of those resources?
            waited = False
            for j in range(i + 1, len(trace.nodes)):
                after = trace.nodes[j]
                if after.sync_point and after.blocking == BlockingKind.BLOCKING:
                    waited = True
                    break
                if after.category in (OpCategory.FREE, OpCategory.MAP):
                    for r in after.all_resources:
                        if r.name in gpu_written and not waited:
                            explanation = (
                                f"Node {self._node_label(after)} accesses resource '{r.name}' "
                                f"on the CPU after it was written by GPU work submitted at "
                                f"node {self._node_label(node)}, but no blocking fence/semaphore "
                                f"wait was observed between submission and this access. "
                                f"On discrete GPUs this is a data race: the GPU may still be "
                                f"writing to the resource when the CPU reads or frees it."
                            )
                            findings.append(Finding(
                                pass_name=self.name,
                                category=FindingCategory.FRAGILITY,
                                severity=Severity.CRITICAL,
                                title=f"Implicit ordering: CPU access to '{r.name}' after GPU submit without wait",
                                explanation=explanation,
                                related_nodes=[node.index, after.index],
                                suggestion=(
                                    "Insert a WAIT_FENCE or WAIT_SEMAPHORE between the "
                                    "SUBMIT_QUEUE and any subsequent CPU-side access to GPU-written resources."
                                ),
                                metadata={"resource": r.name, "submit_node": node.index},
                            ))
        return findings

    # ------------------------------------------------------------------
    # FR-02: CPU access to GPU_LOCAL memory
    # ------------------------------------------------------------------

    def _check_cpu_access_gpu_local(self, trace: IRTrace) -> List[Finding]:
        """
        Detect MAP operations on resources whose memory domain is GPU_LOCAL.
        This is UB on discrete GPUs and vendor-specific behaviour on others.
        """
        findings = []
        for node in trace.nodes:
            if node.category != OpCategory.MAP:
                continue
            for res in node.all_resources:
                if res.memory == MemoryDomain.GPU_LOCAL:
                    explanation = (
                        f"Node {self._node_label(node)} attempts to map resource '{res.name}' "
                        f"which is allocated in GPU_LOCAL (device-local) memory. "
                        f"On discrete GPUs (NVIDIA, AMD) device-local memory is not "
                        f"directly accessible by the CPU unless ReBAR / SAM is enabled. "
                        f"On integrated GPUs this may work but relies on coherence "
                        f"guarantees that are not part of the standard HAL contract. "
                        f"This code is non-portable and may crash or produce incorrect "
                        f"results on different hardware generations."
                    )
                    findings.append(Finding(
                        pass_name=self.name,
                        category=FindingCategory.FRAGILITY,
                        severity=Severity.HIGH,
                        title=f"CPU MAP of GPU_LOCAL resource '{res.name}'",
                        explanation=explanation,
                        related_nodes=[node.index],
                        suggestion=(
                            "Use a staging buffer in CPU_VISIBLE memory for CPU↔GPU transfers. "
                            "Do not map GPU_LOCAL resources directly."
                        ),
                        metadata={"resource": res.name, "memory": res.memory.value},
                    ))
        return findings

    # ------------------------------------------------------------------
    # FR-03: Missing barrier before resource reuse
    # ------------------------------------------------------------------

    def _check_missing_barriers(self, trace: IRTrace) -> List[Finding]:
        """
        Detect: a resource is written by a GPU op and then read by another GPU op
        without an intervening barrier — relies on implicit driver ordering.
        """
        findings = []

        # Track: resource name → last write node
        last_write: Dict[str, IRNode] = {}

        for node in trace.nodes:
            # Reset tracking at explicit barriers
            if node.category in (OpCategory.BARRIER,) and node.sync_point:
                last_write.clear()
                continue

            # Check reads against pending writes
            for res in node.resources_read:
                prev_write = last_write.get(res.name)
                if prev_write is None:
                    continue
                # Both nodes should be GPU-side (not MAP/FREE/CPU ops)
                if (
                    node.category not in (OpCategory.FREE, OpCategory.MAP, OpCategory.QUERY)
                    and prev_write.category not in (OpCategory.ALLOC, OpCategory.MAP)
                ):
                    gap = node.index - prev_write.index
                    # Only flag if no barrier was detected in the range
                    barrier_found = any(
                        n.category == OpCategory.BARRIER
                        for n in trace.nodes_in_range(prev_write.index + 1, node.index - 1)
                    )
                    if not barrier_found and gap > 0:
                        explanation = (
                            f"Resource '{res.name}' written at node {self._node_label(prev_write)} "
                            f"is read at node {self._node_label(node)} without an intervening "
                            f"PIPELINE_BARRIER or MEMORY_BARRIER. Some drivers insert implicit "
                            f"barriers for render targets, but this is vendor-specific. "
                            f"On Vulkan, Metal, and D3D12, explicit barriers are required for "
                            f"correctness. This pattern may work on one vendor and silently "
                            f"produce incorrect results on another."
                        )
                        findings.append(Finding(
                            pass_name=self.name,
                            category=FindingCategory.FRAGILITY,
                            severity=Severity.HIGH,
                            title=f"Missing barrier: '{res.name}' written then read without synchronisation",
                            explanation=explanation,
                            related_nodes=[prev_write.index, node.index],
                            suggestion=(
                                "Insert an explicit PIPELINE_BARRIER / MEMORY_BARRIER with "
                                "appropriate src/dst stage and access masks between the write "
                                "and subsequent read."
                            ),
                            metadata={"resource": res.name, "gap_ops": gap},
                        ))

            # Update write tracking
            for res in node.resources_write:
                if node.category not in (OpCategory.FREE,):
                    last_write[res.name] = node

        return findings

    # ------------------------------------------------------------------
    # FR-04: Vendor-specific / architecture-gated code paths
    # ------------------------------------------------------------------

    def _check_arch_specific_hints(self, trace: IRTrace) -> List[Finding]:
        findings = []
        for node in trace.nodes:
            if not node.arch_hints:
                continue
            explanation = (
                f"Node {self._node_label(node)} carries architecture-specific hints: "
                f"{', '.join(sorted(node.arch_hints))}. "
                f"Code paths that are gated on a specific GPU architecture or vendor "
                f"require explicit porting effort for each new hardware generation. "
                f"They are also a source of hard-to-reproduce bugs when the "
                f"architecture detection logic is incorrect."
            )
            findings.append(Finding(
                pass_name=self.name,
                category=FindingCategory.FRAGILITY,
                severity=Severity.MEDIUM,
                title=f"Architecture-specific code at node [{node.index}] {node.op}",
                explanation=explanation,
                related_nodes=[node.index],
                suggestion=(
                    "Prefer capability queries (e.g. 'does this device support tiled "
                    "resources?') over architecture name checks. "
                    "If arch-specific paths are unavoidable, document the fallback "
                    "path and test it on every supported architecture."
                ),
                metadata={"arch_hints": list(node.arch_hints)},
            ))
        return findings

    # ------------------------------------------------------------------
    # FR-05: Infinite timeout on synchronisation
    # ------------------------------------------------------------------

    def _check_infinite_timeout(self, trace: IRTrace) -> List[Finding]:
        findings = []
        for node in trace.nodes:
            if node.timeout_ms != -1:
                continue
            if node.category != OpCategory.SYNC:
                continue
            explanation = (
                f"Node {self._node_label(node)} waits with an infinite timeout. "
                f"If the GPU job never completes — due to a TDR (Timeout Detection "
                f"and Recovery), driver crash, or logic error — this wait will "
                f"block the calling thread forever. "
                f"In production code this manifests as a hung process with no "
                f"diagnostic information."
            )
            findings.append(Finding(
                pass_name=self.name,
                category=FindingCategory.FRAGILITY,
                severity=Severity.MEDIUM,
                title=f"Infinite timeout on sync primitive at node [{node.index}] {node.op}",
                explanation=explanation,
                related_nodes=[node.index],
                suggestion=(
                    "Use a finite timeout (e.g. 5 seconds) and implement error recovery "
                    "logic (device lost handling, retry, graceful shutdown). "
                    "Reserve infinite timeouts only for offline / validation tools."
                ),
                metadata={"timeout_ms": node.timeout_ms},
            ))
        return findings
