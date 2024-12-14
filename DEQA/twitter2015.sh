#!/bin/bash


python preprocess.py \
--release_or_debug debug \
--processor_name CombineProcessor \
--dataset_name twitter2015

python train.py \
--seed 42 \
--batch_size 1 \
--num_train_epochs 10 \
--learning_rate 1e-5 \
--weight_decay 0.01 \
--gradient_accumulation_steps 8 \
--warmup_ratio 0.1 \
--model_name Twitter2015MATEDecisionAddSentence_max_length_Fixed_padding_target_DeBERTaModel \
--lr_scheduler_type linear \
--Is_save_model_history True \
--early_stopping_patience 3 \
--early_stopping_threshold 0.01 \
--metric_for_best_model f1 \
--greater_is_better True \
--Is_not_train False \
--save_total_limit 3 \
--Is_save_model True \
--Is_save_optimizer False

python test.py \
--model_name Twitter2015MATEDecisionAddSentence_max_length_Fixed_padding_target_DeBERTaModel

python train.py \
--seed 2 \
--batch_size 4 \
--num_train_epochs 10 \
--learning_rate 1e-5 \
--weight_decay 0.01 \
--gradient_accumulation_steps 2 \
--warmup_ratio 0.1 \
--model_name Twitter2015MASCDeBERTa_large_target_Model \
--lr_scheduler_type linear \
--Is_save_model_history True \
--early_stopping_patience 3 \
--early_stopping_threshold 0.01 \
--metric_for_best_model f1 \
--greater_is_better True \
--Is_not_train False \
--save_total_limit 3 \
--Is_save_model True \
--Is_save_optimizer False

python train.py \
--seed 21 \
--batch_size 2 \
--num_train_epochs 10 \
--learning_rate 1e-5 \
--weight_decay 0.01 \
--gradient_accumulation_steps 4 \
--warmup_ratio 0.1 \
--model_name Twitter2015MASCDescriptionDeBERTa_large_target_Model \
--lr_scheduler_type linear \
--Is_save_model_history True \
--early_stopping_patience 3 \
--early_stopping_threshold 0.01 \
--metric_for_best_model f1 \
--greater_is_better True \
--Is_not_train False \
--save_total_limit 3 \
--Is_save_model True \
--Is_save_optimizer False

python train.py \
--seed 29 \
--batch_size 4 \
--num_train_epochs 16 \
--learning_rate 5e-6 \
--weight_decay 0.01 \
--gradient_accumulation_steps 2 \
--warmup_ratio 0.1 \
--model_name Twitter2015MASCCLIP_large_336_target_DeBERTaModel \
--lr_scheduler_type linear \
--Is_save_model_history True \
--early_stopping_patience 3 \
--early_stopping_threshold 0.01 \
--metric_for_best_model f1 \
--greater_is_better True \
--Is_not_train False \
--save_total_limit 3 \
--Is_save_model True \
--Is_save_optimizer False

python train.py \
--seed 41 \
--batch_size 1 \
--num_train_epochs 8 \
--learning_rate 0.001 \
--weight_decay 0.01 \
--gradient_accumulation_steps 2 \
--warmup_ratio 0.1 \
--model_name Twitter2015MASCDecisionModel \
--lr_scheduler_type linear \
--Is_save_model_history False \
--early_stopping_patience 3 \
--early_stopping_threshold 0.01 \
--metric_for_best_model f1 \
--greater_is_better True \
--Is_not_train True \
--save_total_limit 1 \
--Is_save_model True \
--Is_save_optimizer False

python test.py \
--model_name Twitter2015MASCDecisionModel

cp "./pre_processed/Twitter2015MASCDecisionDataset/Twitter2015MASCDecisionDataset.json" "./Macro-F1"
cp "./trained_model/Twitter2015MASCDecisionModel/predictions.json" "./Macro-F1"
cd Macro-F1 || exit
python generate_gold.py
python evaluation.py
cd ..

cp "./Macro-F1/gold.json" "./evaluation/"
cp "./trained_model/Twitter2015MATEDecisionAddSentence_max_length_Fixed_padding_target_DeBERTaModel/predictions.json" "./evaluation/MATE.json"
cp "./evaluation/MATE.json" "./evaluation/MATE - 1.json"
sed -i 's/0]/-1]/g' "./evaluation/MATE - 1.json"
cp "./Macro-F1/predictions.json" "./evaluation/MASC.json"
cd evaluation || exit
python replace.py
python F1.py
cd ..