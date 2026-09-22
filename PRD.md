# PRD: 天垓100 BI-V100 推理引擎竞赛

## 目标
首位通过全部功能测试+效果测试+性能基准的参赛者获得基础奖。

## 竞赛门槛
- 50+ 功能测试用例全部通过
- 效果偏差 ≤±4%
- 性能门槛 Token 吞吐加权值 ≥8000
- Output TPS 权重占 83%（decode kernel 优化投入产出比最高）

AllReduce 大概占 10ms。剩下的 36ms 是 Python dispatch。1400 次 PyTorch 函数调用 × 25 微秒。

这台机器有没有 NVLink 改变不了 Python 每次调用花 25 微秒的事实。NVIDIA 上用 CUDA Graph 一次性录制所有 kernel launch，replay 时零 Python 开销。但 BI-V100 CUDA 10.2 对 Graph 支持有限。

最大的问题是 太多小 kernel 走 Python dispatch。减少 launch 次数比优化任何单个 kernel 都有效。