# 当前默认checkpoint CPU复载

阶段：smoke；状态：passed_cpu_tolerance。

通过 2/2；prediction逐位一致 0。固定容差atol=1e-06、rtol=1e-05；每次均报告两种判据。

仅CPU推理，未训练；原始日志、checkpoint、数组、历史审计均不写回。原输入逐SHA在前后复核。INPUT_TENSOR_IDENTITY记录原DF与独立CSV重建的四张量、dtype、shape和边界。

CPU跨Torch版本复载不能代替20/20 GPU复载或GPU逐位一致。新X哈希没有训练时GPU张量参考，不能追溯补造缺失的原始张量记录。

结果、原/复载float64指标、失败与数值漂移详见CPU_RELOAD_SUMMARY.json及runs目录。
