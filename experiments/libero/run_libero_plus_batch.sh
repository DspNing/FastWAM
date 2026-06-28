#!/bin/bash
# LIBERO-Plus batched evaluation scheduler.
#
# Unlike run_libero_parallel_test.sh (one short-lived process per task_id,
# model reloaded every time), this scheduler launches one LONG-LIVED worker
# per (GPU, slot). Each worker loads the model ONCE and rolls out a static
# chunk of task_ids (eval_libero_batch.py), so the ~135s model-load tax is
# paid once per worker instead of once per task.
#
# This is what makes LIBERO-Plus tractable: 2402 spatial tasks with
# num_trials=1 would otherwise reload the model 2402 times.
#
# The original LIBERO path (run_libero_parallel_test.sh + eval_libero_single.py)
# is untouched. This script is Plus-only and is selected by run_libero_eval.sh
# when MODE=plus.
#
# Resume: workers skip any task whose gpu*_task{task_id}_results.json already
# exists, so re-running the same command after a crash/interrupt picks up where
# it left off. No queue state is needed.
#
# Completion is detected by counting result files (same contract as the
# single-task scheduler), so summarize_results.py works unchanged.

run_libero_plus_batch() {
    local task_list_file=$1
    echo "[plus-batch] task_file: $task_list_file"

    require_non_empty() {
        local var_name="$1"
        local var_val="${!var_name}"
        if [ -z "$var_val" ]; then
            echo "Error: required variable $var_name is not set"
            exit 1
        fi
    }

    # Basic configuration
    ROOT_DIR=${ROOT_DIR:-"$(pwd)"}
    export ROOT_DIR
    RUN_ID=${RUN_ID:-"eval_$(date +%Y%m%d_%H%M%S)"}
    export RUN_ID
    # RESULTS_SUBDIR is injected by run_libero_eval.sh ("libero_plus" for plus,
    # "libero" for master). Default to libero_plus when run standalone.
    RESULTS_SUBDIR=${RESULTS_SUBDIR:-"libero_plus"}
    OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT_DIR/evaluate_results/$RESULTS_SUBDIR/$RUN_ID"}
    export OUTPUT_DIR
    EXP_NAME=${EXP_NAME:-""}
    export EXP_NAME
    SESSION_NAME="libero_plus_batch"

    echo "[plus-batch] EXP_NAME: $EXP_NAME"
    mkdir -p "$OUTPUT_DIR"
    echo "[plus-batch] Results will be saved to: $OUTPUT_DIR"

    # Copy task_list_file into OUTPUT_DIR
    cp "$task_list_file" "$OUTPUT_DIR/"
    task_list_file="$OUTPUT_DIR/$(basename $task_list_file)"
    echo "[plus-batch] Task list file copied to: $task_list_file"

    # GPU configuration (same parsing as run_libero_parallel_test.sh)
    if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
        require_non_empty "NUM_GPUS"
        AVAILABLE_GPUS=$(seq 0 $((NUM_GPUS-1)) | tr '\n' ',' | sed 's/,$//')
    else
        AVAILABLE_GPUS=$CUDA_VISIBLE_DEVICES
        NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
    fi
    export NUM_GPUS
    IFS=',' read -r -a GPU_ARRAY <<< "$AVAILABLE_GPUS"
    echo "[plus-batch] NUM_GPUS: $NUM_GPUS, AVAILABLE_GPUS: $AVAILABLE_GPUS"

    require_non_empty "MAX_TASKS_PER_GPU"
    require_non_empty "NUM_TRIALS"

    # Workers per GPU (configurable). Total workers = NUM_GPUS * WORKERS_PER_GPU.
    # Default to MAX_TASKS_PER_GPU so a 2-GPU / MAX_TASKS_PER_GPU=2 run spawns
    # 4 long-lived workers and cuts model loads to 4 instead of thousands.
    WORKERS_PER_GPU=${WORKERS_PER_GPU:-$MAX_TASKS_PER_GPU}
    NUM_WORKERS=$((NUM_GPUS * WORKERS_PER_GPU))
    echo "[plus-batch] WORKERS_PER_GPU: $WORKERS_PER_GPU -> total workers: $NUM_WORKERS"

    # Video knobs (see eval_libero_batch.py). Plus default: no videos.
    SAVE_VIDEO=${SAVE_VIDEO:-false}
    MAX_VIDEOS_PER_WORKER=${MAX_VIDEOS_PER_WORKER:-50}
    export SAVE_VIDEO MAX_VIDEOS_PER_WORKER

    # tmux grid: one pane per worker + no spare panes.
    TMUX_GRID_ROWS=${TMUX_GRID_ROWS:-1}
    TMUX_GRID_COLS=${TMUX_GRID_COLS:-$((NUM_WORKERS + 1))}
    GRID_ROWS=$TMUX_GRID_ROWS
    GRID_COLS=$TMUX_GRID_COLS
    MAX_PANES=$((GRID_ROWS * GRID_COLS - 1))
    if [ "$MAX_PANES" -lt "$NUM_WORKERS" ]; then
        echo "Error: tmux grid too small (MAX_PANES=$MAX_PANES < NUM_WORKERS=$NUM_WORKERS). Increase TMUX_GRID_COLS/TMUX_GRID_ROWS."
        exit 1
    fi

    TASK_LOG_DIR="$OUTPUT_DIR/task_logs"
    CHUNK_DIR="$OUTPUT_DIR/chunks"
    mkdir -p "$TASK_LOG_DIR" "$CHUNK_DIR"

    # Checkpoint and config (same normalization as the single-task scheduler)
    CKPT=${CKPT:-""}
    export CKPT
    CONFIG=${CONFIG:-""}
    require_non_empty "CKPT"
    require_non_empty "CONFIG"
    CONFIG="${CONFIG#configs/}"
    CONFIG="${CONFIG#task/}"
    CONFIG="${CONFIG%.yaml}"
    export CONFIG

    echo "[plus-batch] CKPT: $CKPT"
    echo "[plus-batch] CONFIG: $CONFIG"
    echo "[plus-batch] NUM_TRIALS: $NUM_TRIALS"
    echo "[plus-batch] SAVE_VIDEO: $SAVE_VIDEO  MAX_VIDEOS_PER_WORKER: $MAX_VIDEOS_PER_WORKER"

    local total_tasks=$(wc -l < "$task_list_file")
    echo "[plus-batch] Total tasks: $total_tasks"

    # ---- Static chunking: split task_list into NUM_WORKERS chunks ----
    # Round-robin (interleaved) assignment: task at line N goes to worker
    # (N-1) % NUM_WORKERS. This spreads hard/easy tasks across all workers so
    # no worker gets stuck on a contiguous block of hard tasks (e.g. a long
    # run of low-success goal/10 tasks that run to max_steps). Contiguous
    # slicing used to let one worker idle on a 600s/task block while others
    # flew through 20s spatial tasks -- round-robin removes that imbalance.
    #
    # Resume-skip is unaffected: it keys on per-task result files, not on
    # chunk assignment, so already-done tasks are skipped regardless of how
    # the chunks were sliced.
    local chunk_files=()
    local wid=0
    while [ $wid -lt $NUM_WORKERS ]; do
        chunk_files+=("$CHUNK_DIR/chunk_worker${wid}.txt")
        : > "$CHUNK_DIR/chunk_worker${wid}.txt"   # truncate in case of re-run
        wid=$((wid + 1))
    done
    local linenum=0
    while IFS= read -r line; do
        [ -z "$line" ] && continue
        linenum=$((linenum + 1))
        wid=$(( (linenum - 1) % NUM_WORKERS ))
        echo "$line" >> "${chunk_files[$wid]}"
    done < "$task_list_file"
    for wid in "${!chunk_files[@]}"; do
        echo "[plus-batch] worker $wid -> $(wc -l < "${chunk_files[$wid]}") tasks (${chunk_files[$wid]})"
    done

    # ---- tmux session ----
    if tmux has-session -t $SESSION_NAME 2>/dev/null; then
        tmux kill-session -t $SESSION_NAME
        echo "[plus-batch] Deleted existing session '$SESSION_NAME'"
    fi
    tmux new-session -d -s $SESSION_NAME

    create_grid_layout() {
        local window=$1
        if [ $window -gt 0 ]; then
            if ! tmux list-windows -t $SESSION_NAME | grep -q "^$window:"; then
                tmux new-window -t $SESSION_NAME:$window
            fi
        fi
        local pane_count=$(tmux list-panes -t $SESSION_NAME:$window | wc -l)
        for ((i=pane_count; i<GRID_ROWS*GRID_COLS-1; i++)); do
            tmux split-window -t $SESSION_NAME:$window
            tmux select-layout -t $SESSION_NAME:$window tiled
        done
    }
    create_grid_layout 0

    ensure_pane_exists() {
        local window_id=$1
        local pane_id=$2
        if [ $window_id -gt 0 ]; then
            if ! tmux list-windows -t $SESSION_NAME | grep -q "^$window_id:" 2>/dev/null; then
                tmux new-window -t $SESSION_NAME:$window_id 2>/dev/null
                create_grid_layout $window_id
            fi
        fi
        if [ $pane_id -eq 0 ] && [ $window_id -gt 0 ]; then
            create_grid_layout $window_id
        fi
    }

    # ---- Launch one long-lived worker per pane ----
    echo "[plus-batch] Launching $NUM_WORKERS workers..."
    local next_pane=0
    for wid in "${!chunk_files[@]}"; do
        local chunk_file="${chunk_files[$wid]}"
        # Map worker index -> (real GPU id, slot). Round-robin GPUs so workers
        # spread across cards; within a GPU, slots 0..WORKERS_PER_GPU-1.
        local gpu_idx=$((wid % NUM_GPUS))
        local real_gpu_id=${GPU_ARRAY[$gpu_idx]}

        local window_id=$((next_pane / MAX_PANES))
        local pane_id=$((next_pane % MAX_PANES))
        local pane_info="$window_id.$pane_id"
        ensure_pane_exists $window_id $pane_id
        next_pane=$((next_pane + 1))

        local log_file="$TASK_LOG_DIR/worker${wid}_gpu${real_gpu_id}.log"
        echo "[plus-batch] Launching worker $wid on GPU$real_gpu_id (pane $pane_info), chunk=$chunk_file, log=$log_file"

        tmux select-pane -t $SESSION_NAME:$pane_info 2>/dev/null
        tmux send-keys -t $SESSION_NAME:$pane_info "clear" C-m 2>/dev/null
        # CONDA_ENV lets a user point panes at a different env (e.g. a cloned
        # fastwam_plus) without editing this script. Defaults to "fastwam" so
        # the original usage is unchanged.
        CONDA_ENV="${CONDA_ENV:-fastwam}"
        tmux send-keys -t $SESSION_NAME:$pane_info "source ~/.bashrc && conda activate $CONDA_ENV && cd $ROOT_DIR && \
            LOG_FILE='$log_file' && export EXP_NAME=$EXP_NAME && \
            export PYTHONPATH=$ROOT_DIR/LIBERO-plus && \
            export LIBERO_CONFIG_PATH=$HOME/.libero_plus && \
            export MAGICK_HOME=\$CONDA_PREFIX && \
            export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH && \
            CUDA_VISIBLE_DEVICES=$real_gpu_id python experiments/libero/eval_libero_batch.py \
            task=$CONFIG ckpt=$CKPT \
            EVALUATION.num_trials=$NUM_TRIALS \
            EVALUATION.output_dir=$OUTPUT_DIR \
            +EVALUATION.chunk_file=$chunk_file \
            +EVALUATION.worker_id=$wid \
            +EVALUATION.save_video=$SAVE_VIDEO \
            +EVALUATION.max_videos_per_worker=$MAX_VIDEOS_PER_WORKER \
            gpu_id=$real_gpu_id $EXTRA_ARGS > \"\$LOG_FILE\" 2>&1; \
            echo \"[worker $wid] exited rc=\$?\"" C-m 2>/dev/null
        sleep 0.5
    done

    # ---- Wait for completion by counting result files ----
    # Same contract as run_libero_parallel_test.sh: a task is done when its
    # gpu*_task{task_id}_results.json exists. We count across all suites.
    local monitoring_interval=${MONITORING_INTERVAL:-15}
    local status_interval=${STATUS_INTERVAL:-60}
    local last_status_time=0

    echo "[plus-batch] Workers launched. Waiting for completion (total=$total_tasks)..."
    while true; do
        current_time=$(date +%s)
        local total_completed=$(find "$OUTPUT_DIR" -type f -name "gpu*_task*_results.json" | wc -l)
        if [ "$total_completed" -ge "$total_tasks" ]; then
            echo "[plus-batch] All $total_tasks tasks complete!"
            break
        fi

        # Detect total worker death (all panes exited) while tasks remain:
        # if no eval_libero_batch.py process is alive and we are still short,
        # something went wrong -- report and stop to avoid an infinite loop.
        local alive_workers=$(pgrep -fc "eval_libero_batch.py" 2>/dev/null || true)
        alive_workers=${alive_workers:-0}
        if [ "$alive_workers" -eq 0 ] && [ "$total_completed" -lt "$total_tasks" ]; then
            echo "[plus-batch] WARNING: no live workers but $total_completed/$total_tasks done."
            echo "[plus-batch] Some workers crashed. Re-run this command to resume unfinished tasks (already-done ones are skipped)."
            echo "[plus-batch] Incomplete? Check $TASK_LOG_DIR and re-run."
            break
        fi

        if [ $((current_time - last_status_time)) -ge $status_interval ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Plus-batch Status ==="
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed: $total_completed / $total_tasks"
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Live workers: $alive_workers / $NUM_WORKERS"
            for wid in "${!chunk_files[@]}"; do
                local cdone=$(find "$OUTPUT_DIR" -type f -name "gpu${wid}_task*_results.json" | wc -l)
                local csize=$(wc -l < "${chunk_files[$wid]}")
                echo "[$(date '+%Y-%m-%d %H:%M:%S')]   worker $wid: $cdone / $csize done"
            done
            echo "[$(date '+%Y-%m-%d %H:%M:%S']) =================="
            last_status_time=$current_time
        fi

        sleep $monitoring_interval
    done

    # ---- Summarize ----
    echo "[plus-batch] Generating evaluation report..."
    python experiments/libero/summarize_results.py --output_dir="$OUTPUT_DIR"
    echo "[plus-batch] Done. Results: $OUTPUT_DIR"
}


# Entrypoint
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    if [ $# -lt 1 ]; then
        echo "Error: task file path is required"
        echo "Usage: $0 <task_file>"
        exit 1
    fi
    test_file="$1"
    run_libero_plus_batch "$test_file"
    exit $?
fi