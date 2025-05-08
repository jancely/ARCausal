export CUDA_VISIBLE_DEVICES=1

model_name=ARCausal
el=4
dm=512
lr=0.001

python -u run.py \
  --is_training 1 \
  --root_path ./data/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers $el \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model $dm \
  --d_ff $dm \
  --batch_size 16 \
  --learning_rate $lr \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./data/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers $el \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model $dm \
  --d_ff $dm \
  --batch_size 16 \
  --learning_rate $lr \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./data/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers $el \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model $dm \
  --d_ff $dm \
  --batch_size 16 \
  --learning_rate $lr \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./data/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers $el \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model $dm \
  --d_ff $dm \
  --batch_size 16 \
  --learning_rate $lr\
  --itr 1

