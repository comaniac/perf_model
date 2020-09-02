# Tune a model using AutoTVM and the ranking model.
# Usage: ./tune_with_mode.sh <rank model dir> <target> model_list.txt
# One model per line in model_list.txt
# Example:
# MobileNet1.0
# ResNet50_v1

RANK_MODEL_DIR=$1
TARGET=$2
MODEL_LIST_FILE=$3

while IFS= read -r line; do
    python3 app/main.py --list-net ${RANK_MODEL_DIR} \
        --target "${TARGET}" \
        --measure-top-n 32 \
        --gcv $line
    mv tune.log ${line}.json
    echo "Finished $line"
done < ${MODEL_LIST_FILE}

