#!/usr/bin/env bash
# FastWAM LIBERO / LIBERO-Plus 推理一键脚本
#
# 用法：先 conda activate fastwam && cd 到项目根目录，再
#   bash run_libero_eval.sh
#
# 只改下面【参数块】即可，无需每次手打一堆 export。
# MODE="plus"   → LIBERO-Plus 鲁棒性推理（NUM_TRIALS 默认 1，注入 plus 环境变量，走批量调度）
# MODE="master" → 原版 LIBERO 推理  （NUM_TRIALS 默认 50，注入 master 环境变量，走原版单任务调度）
#
# 结果按 MODE 自动分流到子目录：
#   plus  → evaluate_results/libero_plus/$RUN_ID/
#   master→ evaluate_results/libero/$RUN_ID/

# ==================== 参数块（按需修改）====================
MODE="plus"                                       # "plus" | "master"
GPUS="5,6,7"                                          # 用哪些卡，逗号分隔，如 "0,1,2"
TASK_LIST="task_lists/libero_plus_all.txt"         # 评测任务清单
                                                  #   plus : task_lists/libero_<suite>.txt
                                                  #   master: task_lists/master_all.txt 或 master_<suite>.txt
#TASK_LIST="task_lists/master_all.txt"
# checkpoint 二选一：
#   release: checkpoints/fastwam_release/libero_uncond_2cam224.pt
#   自训  : 你的训练输出路径/xxx.pt
CKPT="/data/WuKefei/FastWAM/runs/libero_uncond_2cam224_1e-4/2026-06-28_16-38-27/checkpoints/weights/step_028000.pt"

# dataset_stats：
#   release 版 stats 文件名带前缀，必须显式指定
#   自训 ckpt 若同目录有 dataset_stats.json，可留空 ""
#STATS="checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json"
STATS="/data/WuKefei/FastWAM/data/vae_latents_cache/dataset_stats.json"

# —— 以下一般不用改 ——
CONFIG="libero_uncond_2cam224_1e-4"               # configs/task/ 下的配置名（不带 .yaml）
MAX_TASKS_PER_GPU=4                               # 每卡并发任务数（80G 卡跑 2 个够用）
NUM_TRIALS=""                                     # 留空：plus→1 / master→50；填数字则强制覆盖

# 用哪个 conda env 跑 worker。下游调度脚本的 tmux pane 会 `conda activate $CONDA_ENV`。
# 默认 fastwam（与原版一致）；若建了独立 env（见 LIBERO_PLUS_SETUP_LOG.md 第十节），
# 改成你的 env 名，或在 shell 里 `export CONDA_ENV=fastwam_plus` 覆盖。
#CONDA_ENV="liberoplus"
CONDA_ENV="fastwam"

# —— 仅 plus 批量模式相关 ——
WORKERS_PER_GPU=""                                # 留空：默认 = MAX_TASKS_PER_GPU；填数字则每卡常驻 worker 数
SAVE_VIDEO="true"                                # plus 是否保存 rollout 视频（false 只留 json）
MAX_VIDEOS_PER_WORKER="100"                        # 每个 worker 最多保存多少个 task 的视频（防止磁盘爆炸）
# ==========================================================

set -euo pipefail

# 根据 MODE 决定环境变量开关、默认 NUM_TRIALS、结果子目录、调度脚本
if [[ "${MODE}" == "plus" ]]; then
  NUM_TRIALS="${NUM_TRIALS:-1}"
  RESULTS_SUBDIR="libero_plus"
  export USE_LIBERO_PLUS=1
  # LIBERO-Plus 用 LIBERO-plus 的 benchmark(bddl 文件名带 _language_1 后缀)
  export PYTHONPATH="${ROOT_DIR:-$(pwd)}/LIBERO-plus:${PYTHONPATH:-}"
  LIBERO_PKG_DIR="${ROOT_DIR:-$(pwd)}/LIBERO-plus/libero/libero"
elif [[ "${MODE}" == "master" ]]; then
  NUM_TRIALS="${NUM_TRIALS:-50}"
  RESULTS_SUBDIR="libero"
  export USE_LIBERO_MASTER=1
  # 原版 LIBERO 用 LIBERO-master 的 benchmark(bddl 文件名无后缀)
  export PYTHONPATH="${ROOT_DIR:-$(pwd)}/LIBERO-master:${PYTHONPATH:-}"
  LIBERO_PKG_DIR="${ROOT_DIR:-$(pwd)}/LIBERO-master/libero/libero"
else
  echo "Error: MODE 必须是 'plus' 或 'master'，当前: '${MODE}'" >&2
  exit 1
fi

# 切换 libero 的 bddl/init 路径配置(根据 MODE 指向不同目录)
LIBERO_CFG_DIR="${HOME}/.libero"
mkdir -p "${LIBERO_CFG_DIR}"
cat > "${LIBERO_CFG_DIR}/config.yaml" << EOF
assets: ${LIBERO_PKG_DIR}/assets
bddl_files: ${LIBERO_PKG_DIR}/bddl_files
benchmark_root: ${LIBERO_PKG_DIR}
datasets: ${ROOT_DIR:-$(pwd)}/data/libero_mujoco3.3.2
init_states: ${LIBERO_PKG_DIR}/init_files
EOF
echo "[config] libero config.yaml -> ${LIBERO_PKG_DIR} (MODE=${MODE})"

# 文件存在性校验（早失败早提示，免得跑到一半才发现路径错）
[[ -f "${TASK_LIST}" ]] || { echo "Error: TASK_LIST 不存在: ${TASK_LIST}" >&2; exit 1; }
[[ -f "${CKPT}"       ]] || { echo "Error: CKPT 不存在: ${CKPT}" >&2; exit 1; }
if [[ -n "${STATS}" ]]; then
  [[ -f "${STATS}" ]] || { echo "Error: STATS 不存在: ${STATS}" >&2; exit 1; }
fi

export CONFIG
export CUDA_VISIBLE_DEVICES="${GPUS}"
export MAX_TASKS_PER_GPU
export NUM_TRIALS
export CKPT
export RESULTS_SUBDIR
export CONDA_ENV
# 生成 RUN_ID；两个调度脚本都读它（都有默认值，但显式统一便于日志对应）
export RUN_ID="${RUN_ID:-eval_$(date +%Y%m%d_%H%M%S)}"
export ROOT_DIR="${ROOT_DIR:-$(pwd)}"
# 用 OUTPUT_DIR 环境变量覆盖调度脚本的默认值，确保按 MODE 分流到子目录
export OUTPUT_DIR="${ROOT_DIR}/evaluate_results/${RESULTS_SUBDIR}/${RUN_ID}"

if [[ -n "${STATS}" ]]; then
  export EXTRA_ARGS="EVALUATION.dataset_stats_path=${STATS}"
else
  export EXTRA_ARGS=""
fi

# plus 批量模式的视频/worker 旋钮（master 调度脚本不读这些，export 无害）
export SAVE_VIDEO
export MAX_VIDEOS_PER_WORKER
# WORKERS_PER_GPU 留空时让调度脚本自取默认（=MAX_TASKS_PER_GPU）
[[ -n "${WORKERS_PER_GPU}" ]] && export WORKERS_PER_GPU || true

echo "==================== LIBERO 推理 ===================="
echo "  MODE          : ${MODE}"
echo "  GPUS          : ${GPUS}"
echo "  TASK_LIST     : ${TASK_LIST} ($(wc -l < "${TASK_LIST}") 任务)"
echo "  NUM_TRIALS    : ${NUM_TRIALS}"
echo "  MAX_TASKS_PER_GPU: ${MAX_TASKS_PER_GPU}"
echo "  CKPT          : ${CKPT}"
echo "  STATS         : ${STATS:-<自动查找>}"
echo "  CONFIG        : ${CONFIG}"
echo "  CONDA_ENV     : ${CONDA_ENV}"
echo "  RESULTS_SUBDIR: ${RESULTS_SUBDIR}"
echo "  OUTPUT_DIR    : ${OUTPUT_DIR}"
if [[ "${MODE}" == "plus" ]]; then
  echo "  WORKERS_PER_GPU: ${WORKERS_PER_GPU:-<默认=MAX_TASKS_PER_GPU>}"
  echo "  SAVE_VIDEO    : ${SAVE_VIDEO}"
  echo "  MAX_VIDEOS_PER_WORKER: ${MAX_VIDEOS_PER_WORKER}"
fi
echo "====================================================="

if [[ "${MODE}" == "plus" ]]; then
  bash experiments/libero/run_libero_plus_batch.sh "${TASK_LIST}"
else
  bash experiments/libero/run_libero_parallel_test.sh "${TASK_LIST}"
fi
