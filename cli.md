# FDS 项目说明与实验规范

更新时间：2026-09-29

本文档是本项目当前唯一的实验说明。它区分三件事：官方论文实现、当前开发分支、当前已经启动的 10k 实验。除非特别注明，文中的方法名、命令和文件路径均以当前代码为准。

## 1. 项目边界

本项目使用 JiT-B/16 在 ImageNet-256 上进行 flow-matching ODE 采样。当前代码已经整理到一个目录：

    /home/zjiaak/SSD/projects/fds
    ├── 官方基础代码、当前方法开发代码和实验文档
    ├── cli.md
    ├── evaluators/evaluate_all.py
    ├── scripts/prepare_shared_noise.py
    └── src/main_jit.py

    /home/zjiaak/fds_paper_source_79c0106
    └── 官方提交的干净源码快照，只用于原论文 Heun/FDS 复现

不要把当前目录中的后加方法称为原论文 FDS，也不要用当前目录中的后加方法覆盖 fds_paper_source_79c0106 的官方源码。

## 2. 官方来源与源码版本

    论文：https://arxiv.org/abs/2604.04646
    官方仓库：https://github.com/yeonwoo378/flow-divergence-sampler
    官方提交：79c01067e613613367d411265996bf119ea94ce4

官方干净快照：

    /home/zjiaak/fds_paper_source_79c0106

官方论文方法只有：

    heun
    heun_ours

heun 是 vanilla Heun；heun_ours 是官方 FDS。directional、Lipschitz、consistency、GD、randdir 和 accept-all 都是本项目后加的研究方法，不能放入论文 primary Heun/FDS 结论。

## 3. 代码架构与数据流

### 3.1 运行入口

正式生成入口：

    /home/zjiaak/SSD/projects/fds/src/main_jit.py

入口负责解析 CLI、初始化分布式环境、创建 DenoiserCustom、加载 checkpoint，然后调用 engine_jit.evaluate()。

生成与保存位于：

    /home/zjiaak/SSD/projects/fds/src/engine_jit.py

evaluate() 的职责是：

1. 读取固定初始噪声 bank。
2. 按全局 sample id 构造 labels 和 batch。
3. 调用 model.generate()。
4. 将输出保存为五位数字编号的 PNG。
5. 保存 config.json 和局部指标 trace。
6. 正式运行使用 --skip_online_metrics；评测由独立 evaluator 完成。

采样器实现位于：

    /home/zjiaak/SSD/projects/fds/src/denoiser.py

其中：

- Denoiser._euler_step() 是 Euler 单步。
- Denoiser._heun_step() 是完整 Heun 单步。
- DenoiserCustom._experimental_heun_step() 负责后加方法。
- JVP/VJP 局部指标实现位于 DenoiserCustom。
- GD 路径在 FP32 下启用高阶 autograd。
- model_jit.py 在 FDS_LOCAL_METRIC=1 或 FDS_GD_EAGER=1 时使用 eager/math-attention 路径。

### 3.2 一次采样的状态更新

采样从 t=0 到 t=1，共 num_sampling_steps 个时间区间。普通 Heun 为：

    v_t        = v(z_t, t)
    z_euler    = z_t + (t_next - t) * v_t
    v_next     = v(z_euler, t_next)
    z_{t_next} = z_t + (t_next - t) * (v_t + v_next) / 2

生成循环的最后一个区间沿当前代码使用 Euler 收尾。所有后加方法都在同一个 Heun trajectory 上做局部更新；active 区间默认是 t <= stop_t。

## 4. 文件接口

### 4.1 Checkpoint

当前统一使用：

    /home/zjiaak/HDD/fds/checkpoints/jit-b-16-official/checkpoint-last.pth

当前已核对 SHA256：

    4ebcf24698748548d13bef1b4c3b26c72c6ec2bc633002b3f558697920cb2695

采样使用 checkpoint 中的 model_ema1。

### 4.2 固定初始噪声

当前 10k 实验使用按全局 sample id 排列的共享 noise bank：

    /home/zjiaak/HDD/fds/templates/imagenet256_shared_10k_noise_seed0.npy

接口约定：

    shape: (10000, 3, 256, 256)
    dtype: float32
    index: global sample id
    generation seed: 0, CPU torch.Generator

engine_jit.evaluate() 用 initial_noise[start_idx:end_idx] 读取 batch，再传给 generate(noise=batch_noise, sample_ids=sample_ids)。因此这些 10k 方法不依赖 GPU 数量或 batch size 来重建初始噪声。

重新生成 noise bank 的工具：

    /home/zjiaak/SSD/projects/fds/scripts/prepare_shared_noise.py

### 4.3 生成输出

给定 --output_dir=/path/to/run 和方法 M，代码生成：

    /path/to/run/
    └── M-steps50-JiT-B-16-cfg3.0/
        ├── images/
        │   ├── 00000.png
        │   ├── 00001.png
        │   └── ...
        ├── config.json
        ├── local_metric_trace_rank000.pt
        └── local_metric_summary_rank000.json

最后两个文件仅由写入 local metric trace 的方法生成。images/ 中的文件名是全局 sample id；完整 10k run 应包含 10000 张图片，编号为 00000.png 到 09999.png。

### 4.4 运行日志

当前 tmux 任务的外层目录：

    /home/zjiaak/HDD/fds/runs/10k_local_metrics_20260928/<method>/run.log

当前代码没有自动写统一的 manifest.json 或 status.txt；不要假设它们已经存在。完成状态必须结合图片数量和 run.log 判断。

## 5. 共同实验配置

当前 10k 开发分支实验的共同配置：

| 参数 | 当前值 |
|---|---:|
| model | JiT-B/16 |
| img_size | 256 |
| class_num | 1000 |
| class_idx | -1 |
| num_images | 10000 |
| num_sampling_steps | 50 |
| cfg | 3.0 |
| interval_min | 0.1 |
| interval_max | 1.0 |
| iter | 1 |
| num_delta | 1，方法有明确覆盖时以方法定义为准 |
| perturb_scale | 0.01 |
| perturb_schedule | cosine |
| iter_schedule | linear |
| stop_t | 0.5 |
| seed | 0 |
| seed_delta | 42 |
| seed_eps | 1234 |
| gen_bsz | 4 per process |
| world_size | 1 per method |
| online metrics | disabled: --skip_online_metrics |
| local metric backend | FDS_LOCAL_METRIC=1 |
| GD backend | FDS_GD_EAGER=1 |
| checkpoint | jit-b-16-official |
| initial noise | shared 10k noise bank |

当前每个方法独占一张 GPU，用 torchrun --nproc_per_node=1。这是一组本项目的 10k 快速比较配置，不是论文规定的硬件设置。

### 5.1 10k 对齐规则

不同方法之间的 10k paired comparison 使用同一个 checkpoint、noise bank、global sample id、class label policy、num_sampling_steps、CFG 参数、seed_delta、seed_eps 和 gen_bsz。

当前 noise bank 由 sample_id 直接索引，所以改变 GPU 数量或 batch size 不会改变初始 z。但改变以下内容仍会破坏可比性：

- noise bank 文件。
- sample id 顺序。
- checkpoint。
- class_idx 或 class_num。
- 方法之外的共同采样参数。
- 随机方向的 seed 或生成顺序。

当 class_idx=-1 且 num_images=10000 时，代码要求 num_images % class_num == 0，每个类连续 10 张：

    00000--00009 -> class 0
    00010--00019 -> class 1
    ...
    09990--09999 -> class 999

当前 10k 是独立生成的 10k，并不是从旧 50k 目录抽取的 10k 子集。旧文档中关于 00000, 00005, ..., 49995 的规则只适用于历史 50k 子集方案，不适用于当前 run。

## 6. 方法矩阵与理论定义

下面的 v 指当前时间区间使用的完整 Heun velocity predictor；D=3*256*256。所有 perturbation 都使用 Gaussian-shell 尺度：方向的典型范数约为 sqrt(D)，实际半径由 perturb_scale * cosine_schedule(t) 控制。

### 6.1 官方方法

#### heun

普通 Heun，最后一个区间 Euler 收尾。不做 refinement。

#### heun_ours

官方 FDS divergence candidate refinement：

1. 对 base 点和一个 Gaussian candidate 使用 Hutchinson estimator 估计 velocity divergence。
2. 当 candidate divergence 更低且 base divergence 满足官方阈值 base_divergence >= -1/(1-t) 时接受 candidate。
3. 对选中的状态执行 Heun 更新。
4. refinement 只在 t <= stop_t 的实现区间发生。

这是论文官方 FDS，不要另造 heun_mc_fds 作为同义方法。

### 6.2 MC candidate 方法

MC 方法都先从当前 z 生成 candidate，再比较 base/candidate，最后只对选中的状态执行完整 Heun。num_delta=1 表示一个扰动 candidate 加上 base reference，不是多个 candidate。

#### heun_consist_mc

对 base 和 candidate 分别做：

    z -> Heun(t, t_next) -> Heun(t_next, t)

比较 round-trip error，candidate error 更小时接受。正向和逆向都使用 Heun。

#### heun_mc_directional

令 u_v = v / ||v||，评分为 u_v^T J_v u_v。candidate 分数更低时接受。使用 batch JVP，不显式构造 Jacobian。

#### heun_mc_acceleration

评分为 ||v_next - v|| / (|dt| * ||v|| + eps)。v_next 在完整 Heun next state 上计算，不是简单 Euler preview。candidate 分数更低时接受。

#### heun_mc_spectral

估计对称 Jacobian 部分 S=(J+J^T)/2 的最大特征值，使用 JVP/VJP power iteration。首个有效 time point 使用 5 次，之后每个 time point 使用 1 次 warm-start iteration。candidate 分数更低时接受。

这不是完整 Jacobian spectral norm；完整 spectral norm 对应下面的 Lipschitz 方法。

#### heun_mc_lipschitz

估计真正的 Jacobian spectral norm：

    a = J_v u
    b = J_v^T a
    u = b / ||b||
    L = ||a||

首个有效 time point 为 5 次，后续为 1 次 warm-start。每个 sample 维护自己的方向。candidate 的 L 更低时接受。

#### heun_mc_accept_all

抽取一个 Gaussian candidate，不计算局部指标，始终接受 candidate，然后执行 Heun。它是 MC acceptance ablation。

#### heun_randdir

每个 batch/sample 在采样开始时生成一个 Gaussian direction，在 active timestep 内固定复用；方向按 sample 归一化到典型范数 sqrt(D)。它不做 candidate metric selection，是 fixed-direction ablation。

### 6.3 Lipschitz 方法

#### heun_lip_min

这是逐点 Lipschitz MC 方法，不是三候选有限差分方法：

1. 在 base 点和一个 Gaussian candidate 点分别用 JVP/VJP power iteration 估计 ||J_v||_2。
2. 选 Lipschitz 分数更低的一侧。
3. 若 candidate 更低则移动到 candidate，否则保留 base。
4. 对选中的状态执行 Heun。

它与 heun_mc_lipschitz 使用相同的 Lipschitz 估计，但方法语义单独命名为 lip_min。

#### heun_lip_min_3candidates

每个 active timestep 抽取 3 个 Gaussian candidates。对每个 candidate 计算：

    ||v(candidate) - v(base)|| / ||candidate - base||

选择分数最小的 candidate。base 只作为 reference，不是可选 candidate，因此该方法始终选择一个扰动 candidate。此方法不求导，不使用 JVP/VJP power iteration。

### 6.4 GD 方法

GD 方法不是 MC candidate selection；它直接对一个点的 score 求输入梯度，然后沿负梯度方向移动，再执行 Heun。所有 GD 更新都使用统一半径：

    z_new = z - 0.01 * cosine_schedule(t) * sqrt(D) * g / ||g||

#### heun_fds_gd

对完整 Heun velocity 做 Hutchinson divergence 估计，再对 divergence 对 z 求梯度：

    div(z) = eps^T J_v(z) eps / D
    g      = grad_z div(z)
    z_new  = normalized_negative_gradient_step(g)

这是 divergence 的二阶输入梯度路径。当前实现使用 FP32/eager 高阶 autograd。

#### heun_consistency_gd

定义完整 Heun forward/backward round-trip error：

    z_fwd  = Heun(z, t, t_next)
    z_back = Heun(z_fwd, t_next, t)
    E(z)   = mean((z_back - z)^2)
    g      = grad_z E(z)
    z_new  = normalized_negative_gradient_step(g)

逆向明确使用 Heun，不使用 Euler 替代。该方法需要保留完整 forward/backward 计算图，因此显存和速度开销高于 MC 方法。

#### heun_lip_min_gd

1. 先用 Lipschitz JVP/VJP power iteration 得到方向 u。
2. 固定 u，定义 L(z)=||J_v(z)u||。
3. 对 L 求 grad_z。
4. 沿负梯度做统一归一化更新。

该方法包含二阶导路径，方向 warm-start 但不对 power iteration 本身反向传播。它不是 MC candidate scoring，也不使用 num_delta。

### 6.5 未纳入当前批次的方法

当前代码没有纳入本轮正式 10k 的方法：

    heun_gd
    heun_marginal
    heun_marginal_gd
    heun_lip_max
    heun_mc_reuse_all
    heun_fds_gd_nograd

其中 heun_fds_gd_nograd 是旧实验分支，不能与当前 heun_fds_gd 混称。

## 7. 运行命令模板

### 7.1 当前 10k 单卡模板

    cd /home/zjiaak/SSD/projects/fds

    CUDA_VISIBLE_DEVICES=0 \
    FDS_LOCAL_METRIC=1 \
    FDS_GD_EAGER=1 \
    FDS_DISABLE_TORCH_COMPILE=1 \
    PYTHONUNBUFFERED=1 \
    torchrun --standalone --nnodes=1 --nproc_per_node=1 \
      src/main_jit.py \
      --model JiT-B/16 \
      --img_size 256 \
      --resume /home/zjiaak/HDD/fds/checkpoints/jit-b-16-official \
      --output_dir /home/zjiaak/HDD/fds/runs/10k_local_metrics_20260928/M \
      --evaluate_gen \
      --sampling_method M \
      --num_sampling_steps 50 \
      --cfg 3.0 \
      --interval_min 0.1 \
      --interval_max 1.0 \
      --iter 1 \
      --perturb_scale 0.01 \
      --perturb_schedule cosine \
      --iter_schedule linear \
      --stop_t 0.5 \
      --num_images 10000 \
      --gen_bsz 4 \
      --seed 0 \
      --seed_delta 42 \
      --seed_eps 1234 \
      --num_delta 1 \
      --class_num 1000 \
      --class_idx -1 \
      --initial_noise_path /home/zjiaak/HDD/fds/templates/imagenet256_shared_10k_noise_seed0.npy \
      --skip_online_metrics

将 M 替换为方法名，将 CUDA_VISIBLE_DEVICES 和 --output_dir 改成未占用的 GPU/目录。heun 和 heun_ours 已经跑过，不要用此模板重复启动。

### 7.2 tmux 后台模板

    tmux new-session -d -s fds_10k_M \
      "cd /home/zjiaak/SSD/projects/fds && \
       CUDA_VISIBLE_DEVICES=GPU \
       FDS_LOCAL_METRIC=1 FDS_GD_EAGER=1 FDS_DISABLE_TORCH_COMPILE=1 \
       PYTHONUNBUFFERED=1 \
       torchrun --standalone --nnodes=1 --nproc_per_node=1 \
         src/main_jit.py ... --sampling_method M ... \
         > /home/zjiaak/HDD/fds/runs/10k_local_metrics_20260928/M/run.log 2>&1"

检查：

    tmux ls
    nvidia-smi
    tail -f /home/zjiaak/HDD/fds/runs/10k_local_metrics_20260928/M/run.log

## 8. 局部指标 trace 接口

使用 LOCAL_METRIC_METHODS 的方法会在每个 rank 保存：

    local_metric_trace_rankXXX.pt
    local_metric_summary_rankXXX.json

raw trace 字段：

    sample_ids
    t
    dt
    base_metric
    candidate_metric
    selected_metric
    accepted
    power_iterations
    metric

metric tensor 的约定形状为 [time, batch]。selected_metric 是最终选择的 base/candidate 分数。trajectory 汇总包括：

    trajectory_mean     = mean_t(metric_t)
    trajectory_max      = max_t(metric_t)
    trajectory_integral = sum_t(metric_t * abs(dt))

对 heun_mc_spectral、heun_mc_lipschitz 和 heun_lip_min，应看到：

    [5, 1, 1, ...]

如果每个时间点都是 5，或后续时间点重新随机初始化方向，应将 run 标记为无效。

heun_fds_gd、heun_consistency_gd 和 heun_lip_min_3candidates 当前不写 MC local metric trace；其正确性由日志、图片数量和 smoke test 验证。

## 9. 官方论文复现轨道

论文 primary 与当前 10k 开发分支分开管理。官方源码快照：

    /home/zjiaak/fds_paper_source_79c0106

论文轨道共同参数：

    model=JiT-B/16
    img_size=256
    num_sampling_steps=50
    cfg=3.0
    interval_min=0.1
    interval_max=1.0
    num_images=50000
    class_num=1000
    class_idx=-1
    iter=1
    num_delta=1
    perturb_scale=0.01
    perturb_schedule=cosine
    stop_t=0.5
    seed=0
    seed_delta=42
    seed_eps=1234

论文 primary 只比较：

    heun
    heun_ours

官方 notebook demo 的 cfg=1.5、interval_min=0.0 是独立 demo 配置，不与上述 50k 主实验混用。论文运行时必须记录源码 commit、checkpoint SHA256、GPU 拓扑、每卡 batch、完整 config 和图片数量。

当前 10k 开发分支使用 world_size=1、gen_bsz=4 和固定 noise bank，不能写成论文 50k 的硬件复现。

## 10. 当前 10k 实验状态

运行根目录：

    /home/zjiaak/HDD/fds/runs/10k_local_metrics_20260928

已正式完成的 10k 方法：

    heun_consist_mc
    heun_randdir
    heun_mc_directional
    heun_mc_acceleration
    heun_mc_spectral
    heun_mc_lipschitz

本轮追加并已启动的 10k 方法：

    heun_mc_accept_all
    heun_lip_min_gd
    heun_lip_min
    heun_lip_min_3candidates
    heun_fds_gd
    heun_consistency_gd

heun 和 heun_ours 已经在更早运行中完成，本轮没有重跑。

正式任务完成与否不能根据 tmux session 存在判断，必须检查：

    find /home/zjiaak/HDD/fds/runs/10k_local_metrics_20260928/M \
      -path '*/images/*.png' -type f | wc -l
    tail -20 /home/zjiaak/HDD/fds/runs/10k_local_metrics_20260928/M/run.log

完成标准：图片数为 10000，日志出现 Total elapsed time，且没有后续 traceback。在线 FID/IS 被 --skip_online_metrics 跳过，不影响生成成功；七项指标由独立 evaluator 计算。

## 11. 独立七项评测

评测入口：

    /home/zjiaak/SSD/projects/fds/evaluators/evaluate_all.py

它计算：

    FID
    IS
    KID
    Precision
    Recall
    Density
    Coverage

默认资产：

    FID stats:
    /home/zjiaak/HDD/fds/fid_stats/jit_in256_stats.npz

    reference NPZ:
    /home/zjiaak/HDD/fds/VIRTUAL_imagenet256_labeled.npz

调用格式：

    conda run -n fds python \
      /home/zjiaak/SSD/projects/fds/evaluators/evaluate_all.py \
      --run lip_min=/home/zjiaak/HDD/fds/runs/10k_local_metrics_20260928/heun_lip_min/heun_lip_min-steps50-JiT-B-16-cfg3.0/images \
      --run lip_min_3candidates=/home/zjiaak/HDD/fds/runs/10k_local_metrics_20260928/heun_lip_min_3candidates/heun_lip_min_3candidates-steps50-JiT-B-16-cfg3.0/images \
      --output-dir /home/zjiaak/HDD/fds/runs/metrics_10k

可以重复添加 --run NAME=IMAGE_DIRECTORY。评测器会递归扫描图片、不抽样、不截断，并将结果写入：

    /home/zjiaak/HDD/fds/runs/metrics_10k/results/NAME.json

特征缓存位于 output directory 下的 cache/。评测器拒绝覆盖同名结果文件。PRDC 参数当前固定为 k=3，特征为 inception-v3-compat 的 2048 维，距离按 block size 512 计算。

评测前必须确认每个输入目录完整为 10000 张；不完整目录只能标记为 INCOMPLETE，不得进入比较表。

## 12. 不变量与常见错误

1. heun_ours 才是官方 FDS；heun_fds_gd 是后加 GD，不是论文 FDS。
2. heun_consist_mc 的逆向是 Heun；heun_consistency_gd 的逆向也是 Heun，不能写成 Euler。
3. heun_lip_min 是逐点 Lipschitz JVP/VJP MC；heun_lip_min_3candidates 是 3 个有限差分 candidate，二者不是同一个方法。
4. num_delta=1 是一个扰动 candidate 加 base reference；论文实现没有“抽多个 candidate”的含义。
5. heun_mc_accept_all 是每次都接受 MC candidate 的消融；heun_randdir 是固定随机方向的消融，二者不同。
6. GD 方法可能需要 FP32 高阶 autograd，gen_bsz=4 是当前已验证的显存设置，不要随意改大。
7. 生成阶段使用 --skip_online_metrics；在线 torch-fidelity 的旧二元输入错误不能误判为采样失败。
8. 不要仅凭 tmux session、GPU 显存或单个进度条判断完成；以图片数量和最终日志为准。
9. 10k 当前是固定 noise bank 独立生成，不是 50k 子集；两种实验语义不要混写。
