#!/usr/bin/env bash
# FastWAM 训练一键脚本(tmux 后台运行,防终端关闭)
#
# 用法:先 conda activate fastwam && cd 到项目根目录,再
#   bash run_train.sh
#
# 只改下面【参数块】即可。GPU 数由 GPUS 隐式决定(len of GPUS);
# batch_size / gradient_accumulation_steps / learning_rate 等通过 hydra override 传入。
# 训练在 tmux session "fastwam_train" 内运行,关掉终端不影响。
#
# kill 训练:
#   tmux kill-session -t fastwam_train        # 杀整个训练 session
#   tmux attach -t fastwam_train              # 重新查看训练输出
#   tmux list-sessions                        # 列出所有 session
#恢复训练：RESUME="runs/libero_uncond_2cam224_1e-4/2026-.../checkpoints/state/step_001000"
#bash run_train.sh


# ==================== 参数块(按需修改)====================
GPUS="5,6,7"                                      # 用哪些卡,逗号分隔,如 "0,1" / "0,1,2,3"
TASK="libero_uncond_2cam224_1e-4"                 # configs/task/ 下的配置名(不带 .yaml)

# 训练超参(override task config 的默认值)
BATCH_SIZE="32"                                     # per-GPU micro-batch,留空用 task 默认(16);如 "8"
GRAD_ACCUM=""                                     # 梯度累积步数,留空用默认(1);如 "4"
NUM_WORKERS=""                                  # 每卡 dataloader worker 数,留空用 task 默认(8);如 "16"
LR=""                                             # 学习率,留空用默认(1e-4);如 "5e-5"
NUM_EPOCHS=""                                     # epoch 数,留空用默认(10);如 "5"
MAX_STEPS=""                                      # 最大步数,留空=按epoch算;如 "1000" 用于smoke test
#RESUME="runs/libero_uncond_2cam224_1e-4/2026-06-27_11-35-24/checkpoints/state/step_002000"                                         # 恢复训练的 state 目录,留空=从头;如 "runs/.../checkpoints/state/step_001000"
RESUME=""
# backbone 开关(wan22=5B 默认 / wan21=1.3B 快速实验)
BACKBONE=""                                       # 留空=用 yaml 里的 backbone 值;如 "wan21" 强制覆盖

# ZeRO stage: 1 或 2(对应 scripts/train_zero1.sh / train_zero2.sh)
ZERO_STAGE="1"

# tmux session 名(同时只能跑一个同名训练;换名可并行多个)
SESSION_NAME="fastwam_train"
# ==========================================================

set -euo pipefail

# 清掉可能被其他脚本(run_libero_eval.sh)污染的环境变量,避免 output_dir 用到旧 run_id
unset RUN_ID 2>/dev/null || true

# 解析 GPU 数
[[ -z "${GPUS}" ]] && { echo "Error: GPUS 不能为空" >&2; exit 1; }
NUM_GPUS=$(echo "${GPUS}" | tr ',' '\n' | wc -l)
export CUDA_VISIBLE_DEVICES="${GPUS}"

[[ -f "configs/task/${TASK}.yaml" ]] || { echo "Error: task 配置不存在: configs/task/${TASK}.yaml" >&2; exit 1; }

# 拼 hydra overrides
OVERRIDES=("task=${TASK}")
[[ -n "${BATCH_SIZE}"  ]] && OVERRIDES+=("batch_size=${BATCH_SIZE}")
[[ -n "${GRAD_ACCUM}"  ]] && OVERRIDES+=("gradient_accumulation_steps=${GRAD_ACCUM}")
[[ -n "${NUM_WORKERS}" ]] && OVERRIDES+=("num_workers=${NUM_WORKERS}")
[[ -n "${LR}"          ]] && OVERRIDES+=("learning_rate=${LR}")
[[ -n "${NUM_EPOCHS}"  ]] && OVERRIDES+=("num_epochs=${NUM_EPOCHS}")
[[ -n "${MAX_STEPS}"   ]] && OVERRIDES+=("max_steps=${MAX_STEPS}")
[[ -n "${RESUME}"      ]] && OVERRIDES+=("resume=${RESUME}")
[[ -n "${BACKBONE}"    ]] && OVERRIDES+=("model.backbone=${BACKBONE}")

# 选训练脚本
TRAIN_SCRIPT="scripts/train_zero${ZERO_STAGE}.sh"
[[ -f "${TRAIN_SCRIPT}" ]] || { echo "Error: ZERO_STAGE=${ZERO_STAGE} 对应脚本不存在: ${TRAIN_SCRIPT}" >&2; exit 1; }

# 若同名 session 已存在,先提示
if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "Warning: tmux session '${SESSION_NAME}' 已存在。"
  echo "  查看输出: tmux attach -t ${SESSION_NAME}"
  echo "  杀掉重跑: tmux kill-session -t ${SESSION_NAME} && bash run_train.sh"
  exit 1
fi

# 日志文件
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="runs/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_${RUN_ID}.log"

echo "==================== FastWAM 训练 ===================="
echo "  GPUS          : ${GPUS} (NUM_GPUS=${NUM_GPUS})"
echo "  TASK          : ${TASK}"
echo "  BATCH_SIZE    : ${BATCH_SIZE:-<task默认>}"
echo "  GRAD_ACCUM    : ${GRAD_ACCUM:-<task默认>}"
echo "  NUM_WORKERS   : ${NUM_WORKERS:-<task默认>}"
echo "  LR            : ${LR:-<task默认>}"
echo "  NUM_EPOCHS    : ${NUM_EPOCHS:-<task默认>}"
echo "  MAX_STEPS     : ${MAX_STEPS:-<按epoch算>}"
echo "  RESUME        : ${RESUME:-<从头>}"
echo "  BACKBONE      : ${BACKBONE:-<yaml默认>}"
echo "  ZERO_STAGE    : ${ZERO_STAGE}"
echo "  SESSION_NAME  : ${SESSION_NAME}"
echo "  LOG_FILE      : ${LOG_FILE}"
echo "  OVERRIDES     : ${OVERRIDES[*]}"
echo "======================================================="

# 在 tmux 里启动训练(显式 export CUDA_VISIBLE_DEVICES,避免 tmux server 缓存旧环境变量;
# 显式 unset RUN_ID 防止被 eval 脚本污染)
tmux new-session -d -s "${SESSION_NAME}" "export CUDA_VISIBLE_DEVICES=${GPUS}; unset RUN_ID; bash ${TRAIN_SCRIPT} ${NUM_GPUS} ${OVERRIDES[*]} 2>&1 | tee ${LOG_FILE}; echo; echo '[训练结束] 按任意键关闭'; read -n 1"

echo
echo "训练已在 tmux session '${SESSION_NAME}' 内启动(后台运行)。"
echo "  查看输出 : tmux attach -t ${SESSION_NAME}"
echo "  实时日志 : tail -f ${LOG_FILE}"
echo "  退出 tmux(不杀训练): Ctrl+B 然后按 D"
echo "  杀掉训练 : tmux kill-session -t ${SESSION_NAME}"
