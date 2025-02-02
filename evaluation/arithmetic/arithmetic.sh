#!/bin/bash

MODEL_FILE="all_models.txt"
PYTHON_CMD="lm_eval --model hf --model_args pretrained=PLACEHOLDER,dtype=float16,parallelize=True,trust_remote_code=True --num_fewshot 2 --tasks bigbench_arithmetic_multiple_choice\
 --device cuda --batch_size 1 --output_path eval_out/arithmetic --log_samples"

while IFS= read -r model_name
do
    # Check if the model name is not empty
    if [ -n "$model_name" ]; then
        # Replace PLACEHOLDER with the actual model name
        cmd="${PYTHON_CMD//PLACEHOLDER/$model_name}"
        
        echo "Running command: $cmd"
        # Execute the modified command
        eval "$cmd"
    fi
done < "$MODEL_FILE"
