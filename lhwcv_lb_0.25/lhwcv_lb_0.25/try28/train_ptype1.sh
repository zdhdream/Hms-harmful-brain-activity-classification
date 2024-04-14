export PYTHONPATH=../

##### raw ########
CUDA_VISIBLE_DEVICES=1 python train_v2.py \
   --dtype 'raw' --aug 0 --add_reg 0 \
   --backbone 'b2' \
   --l_reverse_prob 0.0 \
   --g_reverse_prob 0.0\
   --secs 50 --batch_size 24 --seed 42

CUDA_VISIBLE_DEVICES=1 python train_v2.py \
   --dtype 'raw' --aug 0 --add_reg 0 \
   --backbone 'b2' \
   --l_reverse_prob 0.0 \
   --g_reverse_prob 0.0\
   --secs 40 --batch_size 24 --seed 42

CUDA_VISIBLE_DEVICES=1 python train_v2.py \
   --dtype 'raw' --aug 0 --add_reg 0 \
   --backbone 'b2' \
   --l_reverse_prob 0.0 \
   --g_reverse_prob 0.0\
   --secs 30 --batch_size 24 --seed 42


CUDA_VISIBLE_DEVICES=1 python train_v2.py \
   --dtype 'raw' --aug 0 --add_reg 0 \
   --backbone 'b1' \
   --l_reverse_prob 0.0 \
   --g_reverse_prob 0.0\
   --secs 50 --batch_size 24 --seed 43

CUDA_VISIBLE_DEVICES=1 python train_v2.py \
   --dtype 'raw' --aug 0 --add_reg 0 \
   --backbone 'b1' \
   --l_reverse_prob 0.0 \
   --g_reverse_prob 0.0\
   --secs 40 --batch_size 24 --seed 43

CUDA_VISIBLE_DEVICES=1 python train_v2.py \
   --dtype 'raw' --aug 0 --add_reg 0 \
   --backbone 'b1' \
   --l_reverse_prob 0.0 \
   --g_reverse_prob 0.0\
   --secs 30  --batch_size 24 --seed 43

##### hybrid
CUDA_VISIBLE_DEVICES=1 python train_v2.py \
   --dtype 'hybrid'  --model_type 'hybrid2' --aug 1 --add_reg 0 \
   --backbone 'b2'  --secs 50 --batch_size 20 --seed 300

CUDA_VISIBLE_DEVICES=1 python train_v2.py \
   --dtype 'hybrid'  --model_type 'hybrid2' --aug 1 --add_reg 0 \
   --backbone 'b2'  --secs 40 --batch_size 20 --seed 301

CUDA_VISIBLE_DEVICES=1 python train_v2.py \
   --dtype 'hybrid'  --model_type 'hybrid2' --aug 1 --add_reg 0 \
   --backbone 'b2'  --secs 30 --batch_size 20 --seed 302
######
CUDA_VISIBLE_DEVICES=1 python train_v2.py \
   --dtype 'hybrid'  --model_type 'hybrid2' --aug 1 --add_reg 0 \
   --backbone 'b1'  --secs 50 --l_reverse_prob 0.2 --batch_size 24 --seed 400

CUDA_VISIBLE_DEVICES=1 python train_v2.py \
   --dtype 'hybrid'  --model_type 'hybrid2' --aug 1 --add_reg 0 \
   --backbone 'b1'  --secs 40 --l_reverse_prob 0.2 --batch_size 24 --seed 401

CUDA_VISIBLE_DEVICES=1 python train_v2.py \
   --dtype 'hybrid'  --model_type 'hybrid2' --aug 1 --add_reg 0 \
   --backbone 'b1'  --secs 30 --l_reverse_prob 0.2 --batch_size 24 --seed 402


##### cwt
#CUDA_VISIBLE_DEVICES=1 python train_v2.py \
#   --dtype 'raw' --model_type 'cwt' --aug 0 --add_reg 0 \
#   --backbone 'b2' \
#   --l_reverse_prob 0.0 \
#   --g_reverse_prob 0.0\
#   --secs 50 --seed 666 \
#   --batch_size 32 \
#   --t1 10 \
#   --t2 5
#
#CUDA_VISIBLE_DEVICES=1 python train_v2.py \
#   --dtype 'raw' --model_type 'cwt' --aug 0 --add_reg 0 \
#   --backbone 'b2' \
#   --l_reverse_prob 0.0 \
#   --g_reverse_prob 0.0\
#   --secs 40 --seed 666 \
#   --batch_size 32 \
#   --t1 10 \
#   --t2 5
#
#CUDA_VISIBLE_DEVICES=1 python train_v2.py \
#   --dtype 'raw' --model_type 'cwt' --aug 0 --add_reg 0 \
#   --backbone 'b2' \
#   --l_reverse_prob 0.0 \
#   --g_reverse_prob 0.0\
#   --secs 30 --seed 666 \
#   --batch_size 32 \
#   --t1 10 \
#   --t2 5
#
#CUDA_VISIBLE_DEVICES=1 python train_v2.py \
#   --dtype 'raw' --model_type 'cwt' --aug 0 --add_reg 0 \
#   --backbone 'b1' \
#   --l_reverse_prob 0.0 \
#   --g_reverse_prob 0.0\
#   --secs 50 --seed 888 \
#   --batch_size 32 \
#   --t1 10 \
#   --t2 5
#
#CUDA_VISIBLE_DEVICES=1 python train_v2.py \
#   --dtype 'raw' --model_type 'cwt' --aug 0 --add_reg 0 \
#   --backbone 'b1' \
#   --l_reverse_prob 0.0 \
#   --g_reverse_prob 0.0\
#   --secs 40 --seed 888 \
#   --batch_size 32 \
#   --t1 10 \
#   --t2 5
#
#CUDA_VISIBLE_DEVICES=1 python train_v2.py \
#   --dtype 'raw' --model_type 'cwt' --aug 0 --add_reg 0 \
#   --backbone 'b1' \
#   --l_reverse_prob 0.0 \
#   --g_reverse_prob 0.0\
#   --secs 30 --seed 888 \
#   --batch_size 32 \
#   --t1 10 \
#   --t2 5
