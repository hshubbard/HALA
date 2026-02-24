"""
HAL Call Parser — converts raw input traces into the canonical IR.

Architecture
------------
The parser is structured as a two-stage pipeline:

  Stage 1 — Normaliser
      Converts each raw dict into an IRNode with well-typed fields.
      Op-specific rules live in _OP_TABLE; unknown ops fall back gracefully.

  Stage 2 — Lifetime Builder
      Walks the normalised node list to compute ResourceLifetime records
      and infer explicit dependency edges.

Adding a new HAL backend
------------------------
Create a subclass of BaseParser (or add entries to _OP_TABLE) and override
`_pre_process(raw_calls)` to translate vendor-specific field names into the
canonical schema before Stage 1 runs.

Example: a Vulkan backend would rename "vkAllocateMemory" → "ALLOC_BUFFER"
and map Vulkan queue families to canonical queue names.

Canonical schema fields (raw dict keys)
----------------------------------------
  op          : str   — operation name (will be uppercased and normalised)
  size        : int   — bytes (for alloc ops)
  memory      : str   — MemoryDomain name
  cpu_visible : bool  — whether CPU can access
  queue       : str   — target queue name
  timeout     : int|str — ms, or "infinite" / -1
  resource    : str   — explicit resource name
  reads       : list[str] — resource names read
  writes      : list[str] — resource names written
  arch        : list[str] — architecture hints
  blocking    : str   — "blocking" / "non_blocking" / "conditional"
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .ir import (
    BlockingKind,
    Dependency,
    IRNode,
    IRTrace,
    MemoryDomain,
    OpCategory,
    Resource,
    ResourceKind,
    ResourceLifetime,
)

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Op normalisation table
# ──────────────────────────────────────────────────────────────────────────────
# Maps canonical op names → (OpCategory, BlockingKind, sync_point, resource_kind)
# This is the single place where op semantics are declared.

_OP_TABLE: Dict[str, tuple] = {
    # name                     category                  blocking                  sync?   res_kind
    "ALLOC_BUFFER":         (OpCategory.ALLOC,      BlockingKind.NON_BLOCKING, False, ResourceKind.BUFFER),
    "FREE_BUFFER":          (OpCategory.FREE,       BlockingKind.NON_BLOCKING, False, ResourceKind.BUFFER),
    "ALLOC_TEXTURE":        (OpCategory.ALLOC,      BlockingKind.NON_BLOCKING, False, ResourceKind.TEXTURE),
    "FREE_TEXTURE":         (OpCategory.FREE,       BlockingKind.NON_BLOCKING, False, ResourceKind.TEXTURE),
    "MAP_BUFFER":           (OpCategory.MAP,        BlockingKind.NON_BLOCKING, False, ResourceKind.BUFFER),
    "UNMAP_BUFFER":         (OpCategory.UNMAP,      BlockingKind.NON_BLOCKING, False, ResourceKind.BUFFER),
    "MAP_TEXTURE":          (OpCategory.MAP,        BlockingKind.NON_BLOCKING, False, ResourceKind.TEXTURE),
    "UNMAP_TEXTURE":        (OpCategory.UNMAP,      BlockingKind.NON_BLOCKING, False, ResourceKind.TEXTURE),
    "CREATE_FENCE":         (OpCategory.ALLOC,      BlockingKind.NON_BLOCKING, False, ResourceKind.FENCE),
    "DESTROY_FENCE":        (OpCategory.FREE,       BlockingKind.NON_BLOCKING, False, ResourceKind.FENCE),
    "SIGNAL_FENCE":         (OpCategory.SYNC,       BlockingKind.NON_BLOCKING, True,  ResourceKind.FENCE),
    "WAIT_FENCE":           (OpCategory.SYNC,       BlockingKind.BLOCKING,     True,  ResourceKind.FENCE),
    "CREATE_SEMAPHORE":     (OpCategory.ALLOC,      BlockingKind.NON_BLOCKING, False, ResourceKind.SEMAPHORE),
    "DESTROY_SEMAPHORE":    (OpCategory.FREE,       BlockingKind.NON_BLOCKING, False, ResourceKind.SEMAPHORE),
    "SIGNAL_SEMAPHORE":     (OpCategory.SYNC,       BlockingKind.NON_BLOCKING, True,  ResourceKind.SEMAPHORE),
    "WAIT_SEMAPHORE":       (OpCategory.SYNC,       BlockingKind.BLOCKING,     True,  ResourceKind.SEMAPHORE),
    "SUBMIT_QUEUE":         (OpCategory.SUBMIT,     BlockingKind.NON_BLOCKING, False, ResourceKind.QUEUE),
    "FLUSH_QUEUE":          (OpCategory.SYNC,       BlockingKind.BLOCKING,     True,  ResourceKind.QUEUE),
    "PIPELINE_BARRIER":     (OpCategory.BARRIER,    BlockingKind.NON_BLOCKING, True,  ResourceKind.UNKNOWN),
    "MEMORY_BARRIER":       (OpCategory.BARRIER,    BlockingKind.NON_BLOCKING, True,  ResourceKind.UNKNOWN),
    "COPY_BUFFER":          (OpCategory.TRANSFER,   BlockingKind.NON_BLOCKING, False, ResourceKind.BUFFER),
    "COPY_TEXTURE":         (OpCategory.TRANSFER,   BlockingKind.NON_BLOCKING, False, ResourceKind.TEXTURE),
    "BLIT":                 (OpCategory.TRANSFER,   BlockingKind.NON_BLOCKING, False, ResourceKind.UNKNOWN),
    "DRAW":                 (OpCategory.DRAW,       BlockingKind.NON_BLOCKING, False, ResourceKind.UNKNOWN),
    "DRAW_INDEXED":         (OpCategory.DRAW,       BlockingKind.NON_BLOCKING, False, ResourceKind.UNKNOWN),
    "DISPATCH":             (OpCategory.COMPUTE,    BlockingKind.NON_BLOCKING, False, ResourceKind.UNKNOWN),
    "QUERY_TIMESTAMP":      (OpCategory.QUERY,      BlockingKind.NON_BLOCKING, False, ResourceKind.UNKNOWN),
    "READBACK":             (OpCategory.QUERY,      BlockingKind.BLOCKING,     False, ResourceKind.UNKNOWN),
    "DEVICE_WAIT_IDLE":     (OpCategory.SYNC,       BlockingKind.BLOCKING,     True,  ResourceKind.UNKNOWN),
}

# Aliases — vendor names that map to canonical ops
_OP_ALIASES: Dict[str, str] = {
    # Vulkan-ish
    "vkAllocateMemory":         "ALLOC_BUFFER",
    "vkFreeMemory":             "FREE_BUFFER",
    "vkMapMemory":              "MAP_BUFFER",
    "vkUnmapMemory":            "UNMAP_BUFFER",
    "vkQueueSubmit":            "SUBMIT_QUEUE",
    "vkQueueWaitIdle":          "FLUSH_QUEUE",
    "vkDeviceWaitIdle":         "DEVICE_WAIT_IDLE",
    "vkCreateFence":            "CREATE_FENCE",
    "vkDestroyFence":           "DESTROY_FENCE",
    "vkWaitForFences":          "WAIT_FENCE",
    "vkSignalFence":            "SIGNAL_FENCE",
    "vkCmdPipelineBarrier":     "PIPELINE_BARRIER",
    # D3D12-ish
    "ID3D12Resource::Map":          "MAP_BUFFER",
    "ID3D12Resource::Unmap":        "UNMAP_BUFFER",
    "ID3D12CommandQueue::Signal":   "SIGNAL_FENCE",
    "ID3D12Fence::SetEventOnCompletion": "WAIT_FENCE",
    # CUDA-ish
    "cuMemAlloc":               "ALLOC_BUFFER",
    "cuMemFree":                "FREE_BUFFER",
    "cuLaunchKernel":           "DISPATCH",
    "cuStreamSynchronize":      "FLUSH_QUEUE",
    "cuEventRecord":            "SIGNAL_FENCE",
    "cuEventSynchronize":       "WAIT_FENCE",
}


def _normalise_op(raw_op: str) -> str:
    """Return the canonical UPPER_SNAKE_CASE op name."""
    canonical = _OP_ALIASES.get(raw_op)
    if canonical:
        return canonical
    upper = raw_op.upper().replace(" ", "_").replace("-", "_")
    return upper


def _parse_timeout(raw: Any) -> Optional[int]:
    """Convert timeout field to int milliseconds, -1 for infinite."""
    if raw is None:
        return None
    if isinstance(raw, str):
        if raw.lower() in ("infinite", "inf", "-1"):
            return -1
        try:
            return int(raw)
        except ValueError:
            return None
    return int(raw)


def _parse_memory_domain(raw: Any) -> MemoryDomain:
    if raw is None:
        return MemoryDomain.UNKNOWN
    mapping = {
        "gpu_local": MemoryDomain.GPU_LOCAL,
        "cpu_visible": MemoryDomain.CPU_VISIBLE,
        "shared": MemoryDomain.SHARED,
    }
    return mapping.get(str(raw).lower(), MemoryDomain.UNKNOWN)


def _parse_blocking(raw: Any, default: BlockingKind) -> BlockingKind:
    if raw is None:
        return default
    mapping = {
        "blocking": BlockingKind.BLOCKING,
        "non_blocking": BlockingKind.NON_BLOCKING,
        "conditional": BlockingKind.CONDITIONAL,
    }
    return mapping.get(str(raw).lower(), default)


# ──────────────────────────────────────────────────────────────────────────────
# Resource registry — shared across one parse run
# ──────────────────────────────────────────────────────────────────────────────

class _ResourceRegistry:
    """
    Tracks resource instances across a parse run.

    Resources are named either explicitly (field "resource") or auto-generated
    from the op type and a monotonic counter (e.g. "buffer_0", "fence_1").
    """

    def __init__(self) -> None:
        self._resources: Dict[str, Resource] = {}
        self._counters: Dict[str, int] = {}

    def get_or_create(
        self,
        name: Optional[str],
        kind: ResourceKind,
        *,
        size_bytes: int = 0,
        cpu_visible: bool = False,
        memory: MemoryDomain = MemoryDomain.UNKNOWN,
    ) -> Resource:
        if name and name in self._resources:
            return self._resources[name]

        if not name:
            prefix = kind.value
            idx = self._counters.get(prefix, 0)
            self._counters[prefix] = idx + 1
            name = f"{prefix}_{idx}"

        res = Resource(
            name=name,
            kind=kind,
            memory=memory,
            size_bytes=size_bytes,
            cpu_visible=cpu_visible,
        )
        self._resources[name] = res
        return res

    def get(self, name: str) -> Optional[Resource]:
        return self._resources.get(name)

    @property
    def all(self) -> Dict[str, Resource]:
        return dict(self._resources)


# ──────────────────────────────────────────────────────────────────────────────
# Stage 1 — Normaliser
# ──────────────────────────────────────────────────────────────────────────────

class _Normaliser:
    """Converts one raw call dict into an IRNode."""

    def __init__(self, registry: _ResourceRegistry) -> None:
        self._reg = registry

    def normalise(self, index: int, raw: Dict[str, Any]) -> IRNode:
        op = _normalise_op(raw.get("op", "UNKNOWN"))
        table_entry = _OP_TABLE.get(op)

        if table_entry:
            category, default_blocking, sync_point, res_kind = table_entry
        else:
            category = OpCategory.OTHER
            default_blocking = BlockingKind.UNKNOWN
            sync_point = False
            res_kind = ResourceKind.UNKNOWN

        blocking = _parse_blocking(raw.get("blocking"), default_blocking)
        timeout_ms = _parse_timeout(raw.get("timeout"))

        # Override blocking if timeout is infinite
        if timeout_ms == -1 and blocking == BlockingKind.CONDITIONAL:
            blocking = BlockingKind.BLOCKING

        memory = _parse_memory_domain(raw.get("memory"))
        size_bytes = int(raw.get("size", 0))
        cpu_vis = bool(raw.get("cpu_visible", False))

        reads: List[Resource] = []
        writes: List[Resource] = []

        # Explicit resource references
        for rname in raw.get("reads", []):
            r = self._reg.get_or_create(rname, ResourceKind.UNKNOWN)
            reads.append(r)

        for rname in raw.get("writes", []):
            r = self._reg.get_or_create(rname, ResourceKind.UNKNOWN)
            writes.append(r)

        # Primary resource implied by the op
        explicit_name: Optional[str] = raw.get("resource")
        if res_kind != ResourceKind.UNKNOWN or explicit_name:
            primary = self._reg.get_or_create(
                explicit_name,
                res_kind,
                size_bytes=size_bytes,
                cpu_visible=cpu_vis,
                memory=memory,
            )
            # ALLOC / MAP / SIGNAL — resource is produced (write)
            if category in (OpCategory.ALLOC, OpCategory.MAP, OpCategory.SYNC):
                if primary not in writes:
                    writes.insert(0, primary)
            # FREE / UNMAP / WAIT — resource is consumed (read)
            elif category in (OpCategory.FREE, OpCategory.UNMAP):
                if primary not in reads:
                    reads.insert(0, primary)
            else:
                # For submits, barriers, etc., treat as both read+write
                if primary not in reads:
                    reads.insert(0, primary)

        arch_hints: set = set(raw.get("arch", []))
        queue: Optional[str] = raw.get("queue")

        node = IRNode(
            index=index,
            op=op,
            category=category,
            blocking=blocking,
            resources_read=reads,
            resources_write=writes,
            sync_point=sync_point,
            timeout_ms=timeout_ms,
            queue=queue,
            arch_hints=arch_hints,
            raw=raw,
        )
        return node


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2 — Lifetime Builder
# ──────────────────────────────────────────────────────────────────────────────

class _LifetimeBuilder:
    """
    Walks the node list to produce ResourceLifetime records and infer
    ordering dependencies between sync primitives.
    """

    def build(
        self, nodes: List[IRNode], resources: Dict[str, Resource]
    ) -> tuple[Dict[str, ResourceLifetime], List[Dependency]]:
        lifetimes: Dict[str, ResourceLifetime] = {}
        deps: List[Dependency] = []

        # Map resource name → last SIGNAL node index (for fence tracking)
        last_signal: Dict[str, int] = {}

        for node in nodes:
            # Track alloc / free transitions
            for res in node.resources_write:
                if node.category == OpCategory.ALLOC:
                    if res.name not in lifetimes:
                        lifetimes[res.name] = ResourceLifetime(
                            resource=res,
                            alloc_index=node.index,
                        )

            for res in node.resources_read:
                if node.category == OpCategory.FREE:
                    if res.name in lifetimes:
                        lifetimes[res.name].free_index = node.index

            # Track map / unmap intervals
            for res in node.all_resources:
                if node.category == OpCategory.MAP:
                    lt = lifetimes.get(res.name)
                    if lt is None:
                        # resource mapped without prior alloc — create implicit lt
                        lt = ResourceLifetime(resource=res, alloc_index=node.index)
                        lifetimes[res.name] = lt
                    lt.map_intervals.append((node.index, None))

                elif node.category == OpCategory.UNMAP:
                    lt = lifetimes.get(res.name)
                    if lt and lt.map_intervals and lt.map_intervals[-1][1] is None:
                        start = lt.map_intervals[-1][0]
                        lt.map_intervals[-1] = (start, node.index)

            # Track sync edges: SIGNAL → WAIT creates an ordering dep
            if node.category == OpCategory.SYNC:
                for res in node.all_resources:
                    if node.op in ("SIGNAL_FENCE", "SIGNAL_SEMAPHORE"):
                        last_signal[res.name] = node.index
                    elif node.op in ("WAIT_FENCE", "WAIT_SEMAPHORE"):
                        signal_idx = last_signal.get(res.name)
                        if signal_idx is not None:
                            deps.append(Dependency(
                                from_index=signal_idx,
                                to_index=node.index,
                                reason=f"fence/semaphore signal→wait on '{res.name}'",
                                is_explicit=True,
                            ))

        # Any resource without a free_index is live at trace end
        return lifetimes, deps


# ──────────────────────────────────────────────────────────────────────────────
# Public parser API
# ──────────────────────────────────────────────────────────────────────────────

class HALParser:
    """
    Parses a raw trace dict (or JSON file) into a fully constructed IRTrace.

    Usage
    -----
    >>> trace = HALParser().parse_dict({"calls": [...], "arch": "ampere"})
    >>> trace = HALParser().parse_file("trace.json")

    Extending for a new HAL backend
    --------------------------------
    Subclass HALParser and override `_pre_process(raw_calls)` to translate
    backend-specific field names into the canonical schema before parsing.
    """

    def _pre_process(self, raw_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Hook for backend-specific normalisation.
        Default: identity — canonical schema is already assumed.
        Override in subclasses for vendor-specific mappings.
        """
        return raw_calls

    def parse_dict(self, data: Dict[str, Any]) -> IRTrace:
        """Parse a trace from an already-loaded dict."""
        raw_calls: List[Dict[str, Any]] = data.get("calls", [])
        source_arch: Optional[str] = data.get("arch")
        trace_meta: Dict[str, Any] = data.get("metadata", {})

        raw_calls = self._pre_process(raw_calls)

        registry = _ResourceRegistry()
        normaliser = _Normaliser(registry)

        nodes: List[IRNode] = []
        for idx, raw in enumerate(raw_calls):
            try:
                node = normaliser.normalise(idx, raw)
                nodes.append(node)
            except Exception as exc:
                log.warning("Skipping call #%d due to parse error: %s", idx, exc)

        lifetime_builder = _LifetimeBuilder()
        lifetimes, deps = lifetime_builder.build(nodes, registry.all)

        return IRTrace(
            nodes=nodes,
            resources=registry.all,
            lifetimes=lifetimes,
            dependencies=deps,
            source_arch=source_arch,
            metadata=trace_meta,
        )

    def parse_json(self, text: str) -> IRTrace:
        """Parse a trace from a JSON string."""
        data = json.loads(text)
        return self.parse_dict(data)

    def parse_file(self, path: str | Path) -> IRTrace:
        """Parse a trace from a JSON file on disk."""
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return self.parse_dict(data)
