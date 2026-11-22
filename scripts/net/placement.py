"""把 SNN 的 spike 流翻译成 NoC 模拟器能接收的消息。

什么是"适配层"
--------------
模型输出的是「时间步 t，神经元 #57 发了 spike」，而 ``noc_sim`` 需要的是
「时间步 t，从核 8 出发，沿路径 [...] 送到核 41」。两者的坐标系不同：
前者按**神经元**编号，后者按**核**编号，还带路由路径。本模块做三件事：

  1. 映射 (placement)：神经元编号 -> 核编号
  2. 聚合 (aggregation)：一个 spike 触发的是**目标核集合**，不是目标神经元集合
  3. 翻译 (translation)：拼出 ``{"t", "src", "header", "path"}``

第 2 步是关键。不聚合的话一个 spike 要发 352 条消息（每个下游神经元一条）；
聚合后只发「目标核数」条。差的倍数是每核平均神经元数。

placement 的选取
----------------
采用**顺序映射**：神经元按全局编号连续分块到核，

    core(i) = floor(i * K / N)

这是 Loihi 的 Nx SDK 与 SpiNNaker 编译器的默认行为（按 population 内的 id
顺序填核，填满一个换下一个），不是本文提出的算法。它的性质是确定、可复现、
不依赖连接图，代价是**不做任何通信优化**，因此给出的通信量是上界。
"""

from __future__ import annotations

from dataclasses import dataclass, field


# 三个模型都是「隐层1 + 隐层2 + 输出」，宽度由各自 config 决定，
# 这里只关心层边界，不关心具体宽度。
@dataclass
class LayerSpec:
    """一层的全局神经元编号范围 [lo, hi)。"""
    name: str
    lo: int
    hi: int

    @property
    def width(self) -> int:
        return self.hi - self.lo


def layers_for(hidden: int, out: int, names=("hidden1", "hidden2", "out")):
    """按 (隐层1, 隐层2, 输出) 的顺序给出三层编号区间。"""
    n1, n2, n3 = names
    return [
        LayerSpec(n1, 0, hidden),
        LayerSpec(n2, hidden, 2 * hidden),
        LayerSpec(n3, 2 * hidden, 2 * hidden + out),
    ]


class Placement:
    """顺序映射：core(i) = floor(i * n_cores / n_total)。"""

    def __init__(self, layers: list[LayerSpec], n_cores: int = 64):
        if not layers:
            raise ValueError("layers must not be empty")
        self.layers = layers
        self.n_cores = n_cores
        self.n_total = layers[-1].hi
        if self.n_total <= 0:
            raise ValueError("total neuron count must be positive")

    def core_of(self, i: int) -> int:
        return (i * self.n_cores) // self.n_total

    def cores_of_layer(self, layer: LayerSpec | str) -> list[int]:
        """该层占用的核（升序、去重）。"""
        if isinstance(layer, str):
            layer = self.by_name(layer)
        return sorted({self.core_of(i) for i in range(layer.lo, layer.hi)})

    def by_name(self, name: str) -> LayerSpec:
        for L in self.layers:
            if L.name == name:
                return L
        raise KeyError(f"unknown layer {name!r}, have {[L.name for L in self.layers]}")

    def describe(self) -> str:
        lines = [f"placement: {self.n_total} neurons -> {self.n_cores} cores "
                 f"(sequential, {self.n_total / self.n_cores:.1f} neurons/core)"]
        for L in self.layers:
            cores = self.cores_of_layer(L)
            lines.append(f"  {L.name:8s} idx [{L.lo:>4d}, {L.hi:>4d})  width {L.width:>4d}  "
                         f"cores {cores[0]:>2d}-{cores[-1]:>2d} ({len(cores)} cores)")
        return "\n".join(lines)


def fanout_targets(placement: Placement, src_layer: str, dst_layers, src_core: int) -> list[int]:
    """一个 spike 要发往的核：下游各层所占核的并集，去掉源核自己。"""
    targets = set()
    for d in dst_layers:
        targets.update(placement.cores_of_layer(d))
    targets.discard(src_core)
    return sorted(targets)


def spikes_to_messages(spikes_at_t: dict, t: float, placement: Placement,
                       edges, topo, path_table, mode="dual_path",
                       build_messages=None):
    """把某一个时间步的 spike 变成 NoC 消息。

    参数
    ----
    spikes_at_t : dict[str, list[int]]
        层名 -> 该步发放 spike 的**层内**神经元下标列表。
    t : float
        时间步（作为消息注入时刻）。
    edges : list[tuple[str, list[str]]]
        [(源层名, [下游层名...])]，描述前馈与 recurrent。
    topo, path_table : 见 topology.Mesh。
    build_messages : multicast.build_messages，延迟导入以避免循环依赖。

    返回
    ----
    list[dict] : 可直接交给 ``noc_sim.simulate_network``。
    """
    if build_messages is None:
        from .multicast import build_messages as _bm
        build_messages = _bm

    out = []
    for src_layer, dst_layers in edges:
        idxs = spikes_at_t.get(src_layer)
        if not idxs:
            continue
        layer = placement.by_name(src_layer)
        base = layer.lo
        for local_i in idxs:
            src_core = placement.core_of(base + local_i)
            targets = fanout_targets(placement, src_layer, dst_layers, src_core)
            if not targets:
                continue
            out.extend(build_messages(t, src_core, targets, topo,
                                      mode=mode, path_table=path_table))
    return out
