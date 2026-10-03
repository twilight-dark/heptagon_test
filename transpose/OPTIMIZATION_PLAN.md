# MatrixTranspose CUDA 优化计划

日期：2026-10-03。目标平台：NVIDIA L40 48 GB。当前阶段仅分析并制定计划，不修改 `src/trans.cu`。

## 1. 参考实现与约束

已读取 `src/main.cpp` 的 `ref_MatrixTranspose`、`src/trans.cu`、`include/trans.h`、`Makefile` 和 `data/cases.txt`。

- 接口：`void MatrixTranspose(int N, const double *A, double *B)`；A、B 均为 GPU 显存指针，行主序、double 类型，当前测试使用两个独立分配的缓冲区。
- 有效矩阵尺寸按正整数且为 32 的倍数处理；不需要支持原地转置。入口可对 `N <= 0` 直接返回，避免无效 launch。
- 不能只实现普通转置：先令 `B[j*N+i] = A[i*N+j]`，再对 B 每行的每个独立 32 元素分组做 8 轮循环相邻加法。
- 每轮开始保存分组首元素，随后按下标递增更新；右邻元素尚未更新，最后一个元素使用保存的旧首元素。因此一轮严格等价于

  `x_next[l] = x[l] + x[(l+1) % 32]`。

  环绕发生在每个 32 元素分组内部，不能跨分组或跨行。8 轮之间有依赖，轮内可并行。
- 保留 double 和逐轮加法次序，避免改写为二项式卷积或降精度导致舍入差异。数学上结果也可表示为循环移位算子 `(I+S)^8 x`，但该式仅用于理解语义。
- 现有 CUDA 实现使用 `<<<1,1>>>` 串行执行全部工作，存在足够的并行优化空间。
- 禁止 cuBLAS、Thrust 等高性能计算库；使用 CUDA Runtime、自定义 kernel 和 CUDA 原语即可。

## 2. 首选方案：转置与 8 轮计算融合为一个 kernel

1. 使用 32×32 元素 tile，初始线程块为 `dim3(32,8)`，网格为 `dim3(N/32,N/32)`。所有 tile 完整，无需元素级边界判断。后续比较 block y 取 4、8、16 的表现。
2. 将输入合并读取到共享内存 `double tile[32][33]`。令 `tx=threadIdx.x`、`ty=threadIdx.y`，循环 `r=ty; r<32; r+=blockDim.y`：

   `tile[r][tx] = A[(blockIdx.y*32+r)*N + blockIdx.x*32+tx]`。

3. 执行一次所有线程都参与的 `__syncthreads()`，然后按转置后的索引读取共享内存：`v = tile[tx][r]`。每个 warp 的 32 个 lane 此时恰好对应一个输出行内的完整 32 元素分组。
4. 每个 lane 用 double 寄存器保存 v，展开 8 轮：

   ```cpp
   double neighbor = __shfl_sync(0xffffffffu, v, (tx + 1) & 31, 32);
   v = v + neighbor;
   ```

   全部 32 个 lane 均执行每轮 shuffle；先交换旧值，再更新本 lane。不能直接用不处理环绕的 `__shfl_down_sync`。warp 间没有该阶段的数据依赖，不需要每轮 block 屏障。
5. 合并写回：

   `B[(blockIdx.x*32+r)*N + blockIdx.y*32+tx] = v`。

6. 地址乘法使用 `size_t` 或足够宽的整数；接口维持不变。在 A、B 不重叠的现有契约下使用 `__restrict__`。包装函数只提交 kernel，不分配显存、不搬运主机数据、不自行同步设备。

该方案每元素仅有一次全局读、一次全局写，逻辑数据流量为 `16*N*N` 字节。8192 阶时 A、B 合计 1 GiB。此值用于计算有效带宽，并不等于缓存影响下的实际 DRAM 流量。8 轮 FP64 加法与 double shuffle 也可能限制吞吐，不能预先断言性能只受显存带宽限制。

共享内存初始用量为 `32*33*8 = 8448` 字节/块。padding 用于改善转置访问模式；double 跨越两个 32 位 bank，不能直接套用 float 的无冲突结论，需结合生成指令和 profiler 实测。CUDA 官方文档说明了 shared memory bank 模型以及支持 double 的同步 shuffle：[CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-c-programming-guide/index.html)。

## 3. 有限、按测量结果推进的调优

按顺序尝试，每次只改变少量因素，并保留正确且端到端性能最好的版本：

1. **线程块与寄存器**：比较 `(32,4)`、`(32,8)`、`(32,16)`，检查寄存器数、occupancy、local memory spill。比较原 Makefile 的 `-maxrregcount=128` 与编译器默认分配，避免盲目限制寄存器。
2. **指令级并行**：比较逐个输出分组完成全部 8 轮，与每线程持有多个分组并交错推进各轮，衡量隐藏依赖延迟的收益和寄存器开销。
3. **共享内存布局**：若 bank 冲突明显，比较不同 padding 或拆分 double 高低 32 位的布局；保留按位重组，不能数值降精度。
4. **tile 与访问模式**：仅在首版瓶颈明确后尝试每块处理多个 32×32 tile、向量化访存或 warp 寄存器转置。更大 tile 必须覆盖 N 只保证为 32 倍数的尾块，并保持每 32 元素独立环绕。
5. **高级流水线**：只有多 tile 复用和访存等待显著时，才评估异步复制及双缓冲；为单 tile 增加复杂流水线未必有收益。
6. **尺寸分派**：只有稳定测量证明有收益才增加少量按 N 分派的配置；所有路径都实际读取输入并完成计算，不缓存结果或识别测试内容。

目标构建使用项目已有 `-arch=sm_89`。Ada 对应计算能力 8.9；资源与调优依据参考 [NVIDIA Ada Tuning Guide](https://docs.nvidia.com/cuda/archive/12.9.1/pdf/Ada_Tuning_Guide.pdf)。保留现有编译选项作为基准；正确性异常时检查 `--use_fast_math` 的影响，不主动启用改变 FP64 加法结合次序的变换。

## 4. 正确性验证

1. 运行原有全部用例：512、544、1024、1056、1088、4096、6144、8192。每例必须满足原判据 `max_abs_error < 1e-3`，记录实际误差，目标是保持参考加法顺序下的逐元素一致性。
2. 追加 32、64、96 等小尺寸，以及非 64/128 倍数但为 32 倍数的尺寸，验证 tile 坐标与尾部分支。
3. 用零矩阵、全 1 矩阵、行列编码矩阵、随机正负数、单点脉冲验证。全 1 输出应全为 256；在分组下标 0、31、32 附近放置脉冲，重点验证环绕方向和分组隔离。
4. 补充重复调用、不同输入与预填充输出缓冲区，确保不读取旧 B。独立校验显式拒绝意外 NaN/Inf，避免原测试的 `diff > max_error` 漏掉 NaN。
5. 在验证驱动中检查 launch 和同步错误，使用 Compute Sanitizer 的 memcheck、racecheck、synccheck 检查越界、共享内存竞争与同步错误。验证工具不进入正式计时路径。

## 5. 性能验证与验收

- 先记录原单线程实现，再测融合基线及候选配置；测试在 L40 上执行，记录 GPU、驱动、CUDA、编译参数和运行环境。
- 原测试 `run_trans` 预热一次，然后用主机时钟测一次函数调用及 `cudaDeviceSynchronize()`；小矩阵可能受 launch、同步及测量噪声影响。保留该指标用于实际评分，并重复独立运行观察波动。
- 另用 CUDA events 在同一 stream 中预热并计时多次 kernel，报告每次平均值以及多批次中位数/离散程度。输出 kernel 时间、有效带宽和原测试端到端时间，明确二者口径差异。
- 重复使用相同矩阵可能命中 L2；区分缓存内工作集与大工作集，不能将小矩阵有效带宽当作 DRAM 带宽。
- 用 Nsight Compute 抽样分析小、中、大尺寸：内存吞吐、shared bank 冲突、寄存器与 spill、活跃 warp、FP64 和 shuffle 指令开销。profiling 时间不作为最终 benchmark 时间。
- 原评分为 small（N<600）目标 200×、medium（600≤N<2000）450×、large（N≥2000）700×，均相对 CPU 参考。将其作为评分目标，不承诺未经实测的加速比。
- 验收：所有正确性用例通过，工具检查无访问/同步错误，L40 多次测量性能稳定；每个用例报告 CPU 参考时间、CUDA 时间、加速比、误差和得分。

## 6. 实施顺序、交付物与当前限制

1. 完成并保留本计划。
2. 在 `src/trans.cu` 实现融合基线，保持 `trans.h` 的接口及原评测逻辑。
3. 运行正确性与工具检查，再记录基线性能。
4. 在 L40 按第 3 节有限调优，基于数据确定最终配置。
5. 交付优化后的 `trans.cu`、验证记录与性能对比报告；必要的额外验证驱动独立保存。

当前环境检查显示 GPU 为 NVIDIA GeForce RTX 3060 Laptop GPU（6144 MiB，计算能力 8.6），并非 L40；`command -v nvcc` 未返回路径，仅能说明当前 PATH 找不到 nvcc。此次尚未编译、运行或测量算子。后续可检查本机 CUDA 安装位置或转至目标环境；在本机进行功能验证需使用兼容架构单独构建，最终性能结论必须来自 L40，不能用本机数据替代。
