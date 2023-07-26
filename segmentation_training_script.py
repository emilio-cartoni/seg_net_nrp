# Import libraries
import os
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Lambda
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from data_utils_seg import SequenceGenerator
from networks.predirep_seg import PredNet

# Load training data
DATA_DIR = "/media/ibrahim/DataIbrahim/data/rl_dataset/processed_data/"  # where data files are stored as .hkl
train_file = os.path.join(DATA_DIR, 'X_train.hkl')
train_sources = os.path.join(DATA_DIR, 'sources_train.hkl')
val_file = os.path.join(DATA_DIR, 'X_val.hkl')
val_sources = os.path.join(DATA_DIR, 'sources_val.hkl')

# Define training parameters
nb_epoch = 100
batch_size = 4
samples_per_epoch = 500
N_seq_val = 100  # number of sequences to use for validation

# Create data generators
train_generator = SequenceGenerator(train_file, train_sources, 10, batch_size=batch_size, shuffle=True)
val_generator = SequenceGenerator(val_file, val_sources, 10, batch_size=batch_size, N_seq=N_seq_val)

# Model parameters
n_channels, im_height, im_width = (3, 120, 160)
input_shape = (n_channels, im_height, im_width) if K.image_data_format() == 'channels_first' else (im_height, im_width, n_channels)
A_filt_sizes = (3, 3, 3)
Ahat_filt_sizes = (3, 3, 3, 3)
R_filt_sizes = (3, 3, 3, 3)

# nt, R filters, Ahat filters, non-linearity, run
training_parameters = [[10, (32, 64, 128, 256), (3, 32, 64, 128), 'relu', 0]]

for t_p in training_parameters:
    nt = t_p[0]  # number of time steps used for sequences in training
    R_stack_sizes = t_p[1]
    stack_sizes = t_p[2]
    lstm_activation = t_p[3]
    run = t_p[4]

    WEIGHTS_DIR = './weights'
    if not os.path.exists(WEIGHTS_DIR): os.mkdir(WEIGHTS_DIR)
    if not os.path.exists(WEIGHTS_DIR + '/{}'.format(run)): os.mkdir(WEIGHTS_DIR + '/{}'.format(run))

    weights_file = WEIGHTS_DIR + '/{}/weights.hdf5'.format(run)  # where weights will be saved
    json_file = WEIGHTS_DIR + '/{}/model.json'.format(run)  # where weights will be saved

    # Create model
    prednet = PredNet(stack_sizes, R_stack_sizes,
                      A_filt_sizes, Ahat_filt_sizes, R_filt_sizes,
                      output_mode='segmentation', LSTM_activation=lstm_activation, return_sequences=True)

    inputs = Input(shape=(nt,) + input_shape)
    outputs = prednet(inputs)  # output will be (batch_size, nt, img_shape)
    outputs = Lambda(lambda x: x[:, 1:])(outputs)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss="focal_loss", optimizer='adam')

    # Define learning schedule and set up callbacks
    callbacks = list()
    lr_schedule = lambda epoch: 0.0001 if epoch < 50 else 0.00001
    callbacks.append(LearningRateScheduler(lr_schedule))
    callbacks.append(ModelCheckpoint(filepath=weights_file, monitor='val_loss', save_best_only=True))

    # Training
    history = model.fit_generator(train_generator, samples_per_epoch / batch_size, nb_epoch, callbacks=callbacks, verbose=2,
                    validation_data=val_generator, validation_steps=N_seq_val / batch_size)

    # Save model
    json_string = model.to_json()
    with open(json_file, "w") as f:
        f.write(json_string)

    K.clear_session()
