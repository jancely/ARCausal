export CUDA_VISIBLE_DEVICES=0

model_name=ARCausal
el=1
dm=256
lr=0.0001

python -u run.py \
  --is_training 1 \
  --root_path ./data/ETT/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_96 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers $el \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model $dm \
  --d_ff $dm \
  --itr 1 \
  --learning_rate $lr

python -u run.py \
  --is_training 1 \
  --root_path ./data/ETT/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_192 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers $el \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model $dm \
  --d_ff $dm \
  --itr 1 \
  --learning_rate $lr

python -u run.py \
  --is_training 1 \
  --root_path ./data/ETT/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_336 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers $el \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model $dm \
  --d_ff $dm \
  --itr 1 \
  --learning_rate $lr

python -u run.py \
  --is_training 1 \
  --root_path ./data/ETT/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_720 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers $el \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model $dm \
  --d_ff $dm \
  --itr 1 \
  --learning_rate $lr

