"""
Intermediate Representation (IR) for HAL call sequences.

Design goals:
  - Source-HAL-agnostic (not tied to Vulkan, D3D12, CUDA, etc.)
  - Immutable after construction — passes read, never mutate
  - Rich enough to express ordering, lifetimes, and sync semantics
  - Serialisable to/from JSON for cross-tool exchange

Terminology
-----------
Node        : one HAL operation in program order
Resource    : any named, stateful object (buffer, texture, queue, fence…)
Lifetime    : closed interval [alloc_index, free_index] of a resource
Dependency  : explicit ordering edge between two nodes (e.g. fence wait)
SyncPoint   : a node that acts as a synchronisation barrier
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Enumerations
# ──────────────────────────────────────────────────────────────────────────────

class ResourceKind(Enum):
    BUFFER      = "buffer"
    TEXTURE     = "texture"
    QUEUE       = "queue"
    FENCE       = "fence"
    SEMAPHORE   = "semaphore"
    PIPELINE    = "pipeline"
    DESCRIPTOR  = "descriptor"
    UNKNOWN     = "unknown"


class MemoryDomain(Enum):
    GPU_LOCAL    = "GPU_LOCAL"    # device-local, not CPU-visible
    CPU_VISIBLE  = "CPU_VISIBLE"  # host-accessible (pinned / BAR)
    SHARED       = "SHARED"       # zero-copy shared memory
    UNKNOWN      = "UNKNOWN"


class BlockingKind(Enum):
    NON_BLOCKING = "non_blocking"   # fire-and-forget / async
    BLOCKING     = "blocking"       # blocks calling thread
    CONDITIONAL  = "conditional"    # may block (e.g. timeout != 0)
    UNKNOWN      = "unknown"


class OpCategory(Enum):
    ALLOC        = "alloc"          # resource creation / allocation
    FREE         = "free"           # resource destruction / release
    MAP          = "map"            # CPU mapping of a GPU resource
    UNMAP        = "unmap"          # release CPU mapping
    SUBMIT       = "submit"         # work submission to a queue
    SYNC         = "sync"           # explicit synchronisation primitive
    TRANSFER     = "transfer"       # copy / blit between resources
    DRAW         = "draw"           # rendering command
    COMPUTE      = "compute"        # compute dispatch
    BARRIER      = "barrier"        # pipeline / memory barrier
    QUERY        = "query"          # readback / timestamp / occlusion
    OTHER        = "other"


# ──────────────────────────────────────────────────────────────────────────────
# Core data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Resource:
    """
    Represents a named, stateful HAL object tracked across its lifetime.

    Attributes
    ----------
    name        : Unique identifier for this resource instance.
    kind        : What class of object this is (buffer, fence, etc.).
    memory      : Which memory domain (only meaningful for buffers/textures).
    size_bytes  : Allocation size, 0 if not applicable or unknown.
    cpu_visible : True if the CPU can read/write this resource directly.
    metadata    : Pass-through dict for source-specific attributes.
    """
    name        : str
    kind        : ResourceKind      = ResourceKind.UNKNOWN
    memory      : MemoryDomain      = MemoryDomain.UNKNOWN
    size_bytes  : int               = 0
    cpu_visible : bool              = False
    metadata    : Dict[str, Any]    = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Resource) and self.name == other.name


@dataclass
class Dependency:
    """
    An ordering or data dependency between two IR nodes.

    Attributes
    ----------
    from_index  : Index (in IRTrace.nodes) of the producer/predecessor.
    to_index    : Index of the consumer/successor.
    reason      : Human-readable description (e.g. "fence signal→wait").
    is_explicit : True if the source trace declared this dependency;
                  False if it was inferred by a pass.
    """
    from_index  : int
    to_index    : int
    reason      : str   = ""
    is_explicit : bool  = True


@dataclass
class IRNode:
    """
    One HAL operation, normalised into the canonical IR.

    Attributes
    ----------
    index           : Position in the owning IRTrace.nodes list.
    op              : Normalised operation name (UPPER_SNAKE_CASE).
    category        : Broad category for pattern matching.
    blocking        : Threading / CPU-block semantics.
    resources_read  : Resources consumed/read by this op.
    resources_write : Resources produced/written/acquired by this op.
    sync_point      : True if this node is a synchronisation barrier.
    timeout_ms      : For sync ops: timeout in ms, -1 = infinite.
    queue           : Queue name this op targets (if applicable).
    arch_hints      : Set of architecture names this op is specific to.
    raw             : Original dict from the source trace (for debugging).
    metadata        : Arbitrary key/value from the parser or passes.
    """
    index           : int
    op              : str
    category        : OpCategory            = OpCategory.OTHER
    blocking        : BlockingKind          = BlockingKind.UNKNOWN
    resources_read  : List[Resource]        = field(default_factory=list)
    resources_write : List[Resource]        = field(default_factory=list)
    sync_point      : bool                  = False
    timeout_ms      : Optional[int]         = None   # -1 = infinite
    queue           : Optional[str]         = None
    arch_hints      : Set[str]              = field(default_factory=set)
    raw             : Dict[str, Any]        = field(default_factory=dict)
    metadata        : Dict[str, Any]        = field(default_factory=dict)

    @property
    def all_resources(self) -> List[Resource]:
        """Union of read and write resource sets."""
        seen: Set[str] = set()
        result = []
        for r in self.resources_read + self.resources_write:
            if r.name not in seen:
                seen.add(r.name)
                result.append(r)
        return result


@dataclass
class ResourceLifetime:
    """
    The live range of a resource within a trace.

    Attributes
    ----------
    resource        : The resource being tracked.
    alloc_index     : IRNode index where the resource was allocated/created.
    free_index      : IRNode index where it was freed, or None if still live.
    map_intervals   : List of (map_index, unmap_index) pairs; unmap may be None.
    exclusive_owner : Whether this resource is exclusively owned for its lifetime.
    """
    resource        : Resource
    alloc_index     : int
    free_index      : Optional[int]             = None
    map_intervals   : List[Tuple[int, Optional[int]]] = field(default_factory=list)
    exclusive_owner : bool                      = True

    @property
    def duration(self) -> Optional[int]:
        """Number of nodes between alloc and free (None if still live)."""
        if self.free_index is None:
            return None
        return self.free_index - self.alloc_index

    @property
    def is_live_at_end(self) -> bool:
        return self.free_index is None


@dataclass
class IRTrace:
    """
    Complete IR for one analysis run — the unit consumed by every pass.

    Attributes
    ----------
    nodes           : Ordered list of IR nodes.
    resources       : All resources encountered, keyed by name.
    lifetimes       : Computed resource lifetime records.
    dependencies    : Explicit and inferred ordering edges.
    source_arch     : Architecture hint from the trace source (optional).
    metadata        : Arbitrary trace-level annotations.
    """
    nodes           : List[IRNode]                      = field(default_factory=list)
    resources       : Dict[str, Resource]               = field(default_factory=dict)
    lifetimes       : Dict[str, ResourceLifetime]       = field(default_factory=dict)
    dependencies    : List[Dependency]                  = field(default_factory=list)
    source_arch     : Optional[str]                     = None
    metadata        : Dict[str, Any]                    = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience queries used by analysis passes
    # ------------------------------------------------------------------

    def nodes_touching_resource(self, resource_name: str) -> List[IRNode]:
        """Return all nodes that read or write the named resource."""
        return [
            n for n in self.nodes
            if any(r.name == resource_name for r in n.all_resources)
        ]

    def nodes_in_range(self, start: int, end: int) -> List[IRNode]:
        """Return nodes whose index is in [start, end] inclusive."""
        return [n for n in self.nodes if start <= n.index <= end]

    def sync_nodes(self) -> List[IRNode]:
        """Return all nodes classified as synchronisation points."""
        return [n for n in self.nodes if n.sync_point]

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the trace to a JSON-friendly dict (for storage/diff)."""
        def _res(r: Resource) -> Dict[str, Any]:
            return {
                "name": r.name,
                "kind": r.kind.value,
                "memory": r.memory.value,
                "size_bytes": r.size_bytes,
                "cpu_visible": r.cpu_visible,
                "metadata": r.metadata,
            }

        def _node(n: IRNode) -> Dict[str, Any]:
            return {
                "index": n.index,
                "op": n.op,
                "category": n.category.value,
                "blocking": n.blocking.value,
                "resources_read": [r.name for r in n.resources_read],
                "resources_write": [r.name for r in n.resources_write],
                "sync_point": n.sync_point,
                "timeout_ms": n.timeout_ms,
                "queue": n.queue,
                "arch_hints": list(n.arch_hints),
                "metadata": n.metadata,
            }

        def _lifetime(lt: ResourceLifetime) -> Dict[str, Any]:
            return {
                "resource": lt.resource.name,
                "alloc_index": lt.alloc_index,
                "free_index": lt.free_index,
                "map_intervals": lt.map_intervals,
                "exclusive_owner": lt.exclusive_owner,
            }

        return {
            "source_arch": self.source_arch,
            "metadata": self.metadata,
            "resources": {k: _res(v) for k, v in self.resources.items()},
            "nodes": [_node(n) for n in self.nodes],
            "lifetimes": {k: _lifetime(v) for k, v in self.lifetimes.items()},
            "dependencies": [
                {
                    "from": d.from_index,
                    "to": d.to_index,
                    "reason": d.reason,
                    "explicit": d.is_explicit,
                }
                for d in self.dependencies
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
