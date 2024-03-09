export PYTHONPATH=./
python src/training/pipelines/train.py \
--dataset_name cola \
--model_name FacebookAI/roberta-base \
--output_dir cola-classification-model
