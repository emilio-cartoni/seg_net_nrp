#!/bin/bash

# declare -a pred_loss_array=('1' '0')
# declare -a n_layers_array=('4' '3')
# declare -a rnn_type_array=('hgru' 'conv' 'lstm')
# declare -a axon_delay_array=('1' '0')

# for pred_loss in ${pred_loss_array[@]}; do
#     for n_layers in ${n_layers_array[@]}; do
#         for rnn_type in ${rnn_type_array[@]}; do
#             for axon_delay in ${axon_delay_array[@]}; do

#                 python main.py --n_layers=$n_layers \
#                                --rnn_type=$rnn_type \
#                                --axon_delay=$axon_delay \
#                                --pred_loss=$pred_loss \
#                                --load_model=1

#             done
#         done
#     done
# done

python main.py --n_layers=4 --rnn_type=hgru --axon_delay=1 --pred_loss=1 --load_model=1
python main.py --n_layers=4 --rnn_type=hgru --axon_delay=0 --pred_loss=1 --load_model=1
python main.py --n_layers=4 --rnn_type=lstm --axon_delay=1 --pred_loss=1 --load_model=1
python main.py --n_layers=4 --rnn_type=conv --axon_delay=1 --pred_loss=1 --load_model=1
python main.py --n_layers=4 --rnn_type=conv --axon_delay=0 --pred_loss=0 --load_model=1

python main.py --n_layers=3 --rnn_type=hgru --axon_delay=1 --pred_loss=1 --load_model=1
python main.py --n_layers=3 --rnn_type=hgru --axon_delay=0 --pred_loss=1 --load_model=1
python main.py --n_layers=3 --rnn_type=conv --axon_delay=1 --pred_loss=1 --load_model=1
python main.py --n_layers=3 --rnn_type=lstm --axon_delay=1 --pred_loss=1 --load_model=1
python main.py --n_layers=3 --rnn_type=conv --axon_delay=0 --pred_loss=0 --load_model=1
