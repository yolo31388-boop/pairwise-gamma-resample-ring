# pairwise-gamma-resample-ring

## 领域

游戏、图形与多媒体

## 主题

Gamma 正确重采样与振铃

## 说明

线性空间插值、核截断与alpha预乘。

初始状态下 `imaging/gamma_resample_ring.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
