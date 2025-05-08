export CUDA_VISIBLE_DEVICES=0

model_name=ARCausal
el=3
lr=0.0002
dm=512


python -u run.py \
  --is_training 1 \
  --root_path .data/weather/ \
  --data_path 2020.csv \
  --model_id weather_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers $el \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model $dm\
  --d_ff $dm\
  --itr 1 \
  --learning_rate $lr


python -u run.py \
  --is_training 1 \
  --root_path .data/weather/ \
  --data_path 2020.csv \
  --model_id weather_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers $el \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model $dm \
  --d_ff $dm \
  --itr 1 \
  --learning_rate $lr


python -u run.py \
  --is_training 1 \
  --root_path .data/weather/ \
  --data_path 2020.csv \
  --model_id weather_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers $el \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model $dm \
  --d_ff $dm \
  --itr 1 \
  --learning_rate $lr

python -u run.py \
  --is_training 1 \
  --root_path .data/weather/ \
  --data_path 2020.csv \
  --model_id weather_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers $el \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model $dm \
  --d_ff $dm \
  --itr 1 \
  --learning_rate $lr

