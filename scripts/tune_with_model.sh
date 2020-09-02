# Tune a model using AutoTVM and the ranking model.
# Usage: ./tune_with_mode.sh <rank model dir> <target> <measure top N> model_list.txt
# One model per line in model_list.txt
# Example:
# MobileNet1.0
# ResNet50_v1

RANK_MODEL_DIR=$1
TARGET=$2
MEASURE_TOP_N=$3
MODEL_LIST_FILE=$4

while IFS= read -r line; do
    python3 app/main.py --list-net ${RANK_MODEL_DIR} \
        --target "${TARGET}" \
        --measure-top-n ${MEASURE_TOP_N} \
        --graph \
        --gcv $line
    mv tune.log ${line}.json
    mv graph.log ${line}_graph.json
    mv graph_tuner.log ${line}_graph_tuner.log
    echo "Finished $line"
done < ${MODEL_LIST_FILE}

