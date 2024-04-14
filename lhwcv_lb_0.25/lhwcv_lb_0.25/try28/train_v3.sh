export PYTHONPATH=../


CUDA_VISIBLE_DEVICES=0 python train_v3.py \
   --dtype 'raw' --model_type 'wavenet' --aug 0 --add_reg 0 \
   --backbone 'wavenet' \
   --l_reverse_prob 0.0 \
   --g_reverse_prob 0.0\
   --secs 50 --seed 42 \
   --batch_size 28 \
   --t1 10 \
   --t2 10

CUDA_VISIBLE_DEVICES=0 python train_v3.py \
   --dtype 'raw' --model_type 'wavenet' --aug 0 --add_reg 0 \
   --backbone 'wavenet' \
   --l_reverse_prob 0.0 \
   --g_reverse_prob 0.0\
   --secs 40 --seed 42 \
   --batch_size 32 \
   --t1 10 \
   --t2 5


CUDA_VISIBLE_DEVICES=0 python train_v3.py \
   --dtype 'raw' --model_type 'wavenet' --aug 0 --add_reg 0 \
   --backbone 'wavenet' \
   --l_reverse_prob 0.0 \
   --g_reverse_prob 0.0\
   --secs 30 --seed 42 \
   --batch_size 32 \
   --t1 10 \
   --t2 5


CUDA_VISIBLE_DEVICES=0 python train_v3.py \
   --dtype 'raw' --model_type 'cwt' --aug 0 --add_reg 0 \
   --backbone 'b2' \
   --l_reverse_prob 0.0 \
   --g_reverse_prob 0.0\
   --secs 50 --seed 666 \
   --batch_size 32 \
   --t1 10 \
   --t2 10

CUDA_VISIBLE_DEVICES=0 python train_v3.py \
   --dtype 'raw' --model_type 'cwt' --aug 0 --add_reg 0 \
   --backbone 'b2' \
   --l_reverse_prob 0.0 \
   --g_reverse_prob 0.0\
   --secs 40 --seed 666 \
   --batch_size 32 \
   --t1 10 \
   --t2 5

CUDA_VISIBLE_DEVICES=0 python train_v3.py \
   --dtype 'raw' --model_type 'cwt' --aug 0 --add_reg 0 \
   --backbone 'b2' \
   --l_reverse_prob 0.0 \
   --g_reverse_prob 0.0\
   --secs 30 --seed 666 \
   --batch_size 32 \
   --t1 10 \
   --t2 5

CUDA_VISIBLE_DEVICES=0 python train_v3.py \
   --dtype 'raw' --model_type 'cwt' --aug 0 --add_reg 0 \
   --backbone 'b1' \
   --l_reverse_prob 0.0 \
   --g_reverse_prob 0.0\
   --secs 50 --seed 888 \
   --batch_size 32 \
   --t1 10 \
   --t2 10

CUDA_VISIBLE_DEVICES=0 python train_v3.py \
   --dtype 'raw' --model_type 'cwt' --aug 0 --add_reg 0 \
   --backbone 'b1' \
   --l_reverse_prob 0.0 \
   --g_reverse_prob 0.0\
   --secs 40 --seed 888 \
   --batch_size 32 \
   --t1 10 \
   --t2 5

CUDA_VISIBLE_DEVICES=0 python train_v3.py \
   --dtype 'raw' --model_type 'cwt' --aug 0 --add_reg 0 \
   --backbone 'b1' \
   --l_reverse_prob 0.0 \
   --g_reverse_prob 0.0\
   --secs 30 --seed 888 \
   --batch_size 32 \
   --t1 10 \
   --t2 10


#CUDA_VISIBLE_DEVICES=0 python train_v3.py \
#   --dtype 'raw' --model_type 'raw_spec' --aug 0 --add_reg 0 \
#   --backbone 'b2' \
#   --l_reverse_prob 0.0 \
#   --g_reverse_prob 0.0\
#   --secs 50 --seed 42 \
#   --batch_size 32 \
#   --t1 10 \
#   --t2 3
#
#
#CUDA_VISIBLE_DEVICES=0 python train_v3.py \
#   --dtype 'raw' --model_type 'raw_spec' --aug 0 --add_reg 0 \
#   --backbone 'b2' \
#   --l_reverse_prob 0.0 \
#   --g_reverse_prob 0.0\
#   --secs 50 --seed 42 \
#   --batch_size 32 \
#   --t1 15 \
#   --t2 3
#
#CUDA_VISIBLE_DEVICES=0 python train_v3.py \
#   --dtype 'raw' --model_type 'raw_spec' --aug 0 --add_reg 0 \
#   --backbone 'b2' \
#   --l_reverse_prob 0.0 \
#   --g_reverse_prob 0.0\
#   --secs 50 --seed 42 \
#   --batch_size 32 \
#   --t1 15 \
#   --t2 8