export CUDA_VISIBLE_DEVICES=1

model_name=ARCausal
el=2
dm=512
lr=0.0001

python -u run.py \
  --is_training 1 \
  --root_path ./data/flight/ \
  --data_path Flight.csv \
  --model_id Flight_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers $el \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model $dm \
  --d_ff $dm \
  --itr 1 \
  --use_norm 1 \
  --learning_rate $lr
#dm128

python -u run.py \
  --is_training 1 \
  --root_path ./data/flight/ \
  --data_path Flight.csv \
  --model_id Flight_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers $el \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model $dm \
  --d_ff $dm \
  --itr 1 \
  --use_norm 1 \
  --learning_rate $lr

python -u run.py \
  --is_training 1 \
  --root_path ./data/flight/ \
  --data_path Flight.csv \
  --model_id Flight_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers $el \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model $dm \
  --d_ff $dm \
  --itr 1  \
  --use_norm 1  \
  --learning_rate $lr

python -u run.py \
  --is_training 1 \
  --root_path ./data/flight/ \
  --data_path Flight.csv \
  --model_id Flight_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers $el \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model $dm \
  --d_ff $dm \
  --itr 1  \
  --use_norm 1 \
  --learning_rate $lr


