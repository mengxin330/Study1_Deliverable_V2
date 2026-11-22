"""Study 1 的片上网通信模型：mesh 拓扑与顺序映射。

交付包只包含本研究所需要的两个模块：
  - topology  : mesh 的最短路径表（hop 数）
  - placement : 神经元 -> 核的顺序映射与 fanout 目标核

原仓库里还有 noc_sim（排队模拟器）与 multicast（多播构造），
本研究的 flit_hops 是静态计数、不需要排队模拟，故未纳入。
"""

from .topology import Mesh
from .placement import Placement, LayerSpec, layers_for, fanout_targets

__all__ = [
    "Mesh",
    "Placement",
    "LayerSpec",
    "layers_for",
    "fanout_targets",
]
