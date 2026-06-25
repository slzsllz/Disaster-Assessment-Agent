#!/bin/bash

# ================================================================
# Batch Evaluation Script for All Models
# ================================================================
# Evaluates 188 questions in batches of 10 (configurable)
# Organizes logs by model/batch structure
# ================================================================

WORK_DIR="./Earth-Agent"

# Configuration
BATCH_SIZE=40
TOTAL_QUESTIONS=188
MAX_PARALLEL_JOBS=5 # Maximum number of jobs to run in parallel (to avoid cluster limits)

# Create logs directory if it doesn't exist
mkdir -p "${WORK_DIR}/logs"

# Model configurations
declare -A MODELS=(
    ["deepseek"]="langchain_deepseek_enhanced.py training_free_results_gpt5/earth_agent_practice_gpt5_enhanced_config.json 19"
)

# All 188 evaluation question IDs
ALL_QUESTIONS=(
    '1' '5' '6' '7' '9' '13' '14' '15' '16' '18' '20' '22' '24' '25' '26'
    '27' '29' '31' '32' '34' '35' '36' '37' '38' '39' '41' '42' '44' '46'
    '47' '48' '49' '50' '51' '52' '54' '56' '57' '58' '59' '60' '61' '62'
    '63' '64' '65' '66' '67' '68' '69' '70' '71' '72' '73' '74' '75' '76'
    '77' '78' '79' '81' '82' '83' '84' '85' '86' '87' '88' '89' '90' '91'
    '92' '93' '94' '95' '96' '97' '98' '99' '100' '101' '103' '104' '105'
    '106' '108' '111' '113' '115' '117' '118' '120' '122' '123' '124' '126'
    '127' '128' '129' '130' '131' '132' '133' '134' '135' '139' '140' '141'
    '142' '143' '144' '145' '146' '147' '148' '150' '151' '152' '153' '154'
    '156' '157' '158' '159' '160' '162' '163' '164' '165' '166' '167' '168'
    '169' '170' '171' '172' '174' '175' '177' '178' '179' '180' '181' '182'
    '183' '184' '185' '188' '189' '192' '193' '194' '196' '197' '198' '199'
    '200' '201' '204' '205' '206' '207' '211' '212' '213' '214' '216' '217'
    '219' '220' '221' '222' '226' '227' '230' '232' '234' '235' '237' '238'
    '239' '240' '241' '243' '245' '246' '247' '248'
)

# Function to submit a batch job
submit_batch() {
    local model_name=$1
    local script_path=$2
    local config_path=$3
    local num_exp=$4
    local batch_num=$5
    local start_idx=$6
    local end_idx=$7

    # Extract question IDs for this batch
    local batch_questions=("${ALL_QUESTIONS[@]:$start_idx:$BATCH_SIZE}")
    local question_range="${batch_questions[0]}-${batch_questions[-1]}"

    # Convert array to space-separated string for passing to Python
    local question_ids_str="${batch_questions[*]}"

    # Create batch log directory under model name
    local batch_dir="${WORK_DIR}/evaluate_langchain/${model_name}_AP_enhanced_${num_exp}exp/batch_$(printf '%03d' $batch_num)"

    # Run job locally
    echo "  Batch $batch_num: Questions $question_range (indices $start_idx-$end_idx)"
    local log_file="${WORK_DIR}/logs/${model_name}_batch_$(printf '%03d' $batch_num)_$(date +%Y%m%d_%H%M%S).log"

    # Run locally in background
    (
        cd ${WORK_DIR}
        python ${script_path} \
            --enhanced_config ${config_path} \
            --batch_dir ${batch_dir} \
            --question_ids ${question_ids_str} \
            > "${log_file}" 2>&1
    ) &

    echo "    Log: ${log_file}"
    echo ""
}

# Main execution
echo "========================================"
echo "Batch Evaluation Started"
echo "========================================"
echo "Total questions: ${TOTAL_QUESTIONS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Max parallel jobs: ${MAX_PARALLEL_JOBS}"
echo "Number of batches per model: $((($TOTAL_QUESTIONS + $BATCH_SIZE - 1) / $BATCH_SIZE))"
echo ""

# Process each model
for model_name in "${!MODELS[@]}"; do
    IFS=' ' read -r script_path config_path num_exp <<< "${MODELS[$model_name]}"

    echo "========================================"
    echo "Model: ${model_name} (${num_exp} experiences)"
    echo "========================================"

    # Calculate number of batches
    num_batches=$(( ($TOTAL_QUESTIONS + $BATCH_SIZE - 1) / $BATCH_SIZE ))

    # Submit batch jobs with concurrency control
    batch_num=1
    start_idx=0
    job_count=0

    while [ $start_idx -lt $TOTAL_QUESTIONS ]; do
        end_idx=$(( $start_idx + $BATCH_SIZE - 1 ))
        if [ $end_idx -ge $TOTAL_QUESTIONS ]; then
            end_idx=$(( $TOTAL_QUESTIONS - 1 ))
        fi

        submit_batch "$model_name" "$script_path" "$config_path" "$num_exp" "$batch_num" "$start_idx" "$end_idx"

        job_count=$(( $job_count + 1 ))

        # Wait if we've reached max parallel jobs
        if [ $job_count -ge $MAX_PARALLEL_JOBS ]; then
            echo "  Reached max parallel jobs ($MAX_PARALLEL_JOBS), waiting for current batch to complete..."
            wait
            job_count=0
            echo "  Continuing with next batch..."
            echo ""
        fi

        batch_num=$(( $batch_num + 1 ))
        start_idx=$(( $start_idx + $BATCH_SIZE ))
    done

    # Wait for remaining jobs of this model
    wait

    echo ""
done

echo "========================================"
echo "All batch jobs started!"
echo "========================================"
echo ""
echo "Monitor progress:"
echo "  tail -f logs/*.log"
echo ""
echo "After all batches complete, merge results:"
echo "  bash merge_batch_results.sh"
echo "========================================"

# Wait for all jobs
wait

echo ""
echo "All batch evaluations completed!"
echo "Run 'bash merge_batch_results.sh' to merge results."
