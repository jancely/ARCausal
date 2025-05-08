export CUDA_VISIBLE_DEVICES=1

model_name=ARCausal

el=2
dm=128
lr=0.0001


python -u run.py \
  --is_training 1 \
  --root_path ./data/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers $el \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --d_model $dm \
  --d_ff $dm \
  --itr 1 \
  --learning_rate $lr \
  #--train_epochs 1

python -u run.py \
  --is_training 1 \
  --root_path ./data/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 1 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --d_model $dm \
  --d_ff $dm \
  --itr 1 \
  --learning_rate $lr \
  #--train_epochs 2

python -u run.py \
  --is_training 1 \
  --root_path ./data/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers $el \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --itr 1 \
  --d_model $dm \
  --d_ff $dm \
  --learning_rate $lr \

python -u run.py \
  --is_training 1 \
  --root_path ./data/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers $el \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --d_model $dm \
  --d_ff $dm \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 2

