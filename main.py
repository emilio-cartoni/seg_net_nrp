# Code strongly inspired by https://www.youtube.com/watch?v=G3pOvrKkFuk
import torch
import pytorch_lightning as pl
import os
from absl import app, flags, logging
from src.model import PredNet
from src.loss_fn import loss_fn
from src.utils import plot_recons, select_scheduler
from src.dataset_fn_handover import handover_dl


flags.DEFINE_boolean('debug', False, 'Debug mode')
flags.DEFINE_boolean('load_model', False, 'Load model last checkpoint (if available)')
flags.DEFINE_integer('batch_size', 4, 'Batch size')
flags.DEFINE_integer('num_epochs', 50, 'Number of epochs to train')
flags.DEFINE_integer('num_workers', 4, 'Number of workers for dataloader')
flags.DEFINE_integer('num_classes', 5, 'Number of classes to segment')
flags.DEFINE_integer('num_frames', 10, 'Number of frames in each sequence')
flags.DEFINE_integer('num_versions', 1, 'Number of versions to train for each model')
flags.DEFINE_string('logs_dir', 'logs', 'Path to logs directory (e.g., for checkpoints)')
flags.DEFINE_string('data_dir', '/mnt/d/DL/datasets/nrp/handover', 'Data directory')
flags.DEFINE_string('scheduler_type', 'onecycle', 'Scheduler used to update learning rate')
flags.DEFINE_float('lr', 5e-4, 'Learning rate')
flags.DEFINE_integer('n_layers', 4, 'Number of layers in the model')
flags.DEFINE_string('rnn_type', 'hgru', 'Type of the recurrent cells used')
flags.DEFINE_boolean('axon_delay', True, 'Whether to use axonal delays or not')
flags.DEFINE_boolean('pred_loss', False, 'Whether to minimize prediction error')
FLAGS = flags.FLAGS


class PLPredNet(pl.LightningModule):
    ''' Python lightning version of PredNet '''
    def __init__(self, device):
        ''' Initialize the model
            Dataloaders also defined here to compute num_batches
        
        Args:
        -----
        device: torch.device
            Device to use for the model ('cpu', 'cuda')
        n_layers: int
            Number of layers in the model
        rnn_type: str
            Type of recurrent cell used ('conv', 'lstm', 'hgru')
        axon_delay: bool
            Whether to include axonal delays or not
        pred_loss: bool
            Whether or not to minimize prediction error

        '''
        super().__init__()
        channels = compute_fair_number_of_channels()
        if FLAGS.pred_loss:
            seg_layers = tuple(range(len(channels))[1:])
        else:
            seg_layers = (0,)
        self.model = PredNet(device=device,
                             num_classes=FLAGS.num_classes,
                             rnn_type=FLAGS.rnn_type,
                             axon_delay=FLAGS.axon_delay,
                             pred_loss=FLAGS.pred_loss,
                             seg_layers=seg_layers,
                             bu_channels=(3,) + channels[:-1],
                             td_channels=channels)
        self.train_dl = handover_dl(mode='train',
                                    data_dir=FLAGS.data_dir,
                                    batch_size=FLAGS.batch_size,
                                    num_frames=FLAGS.num_frames,
                                    num_classes=FLAGS.num_classes,
                                    shuffle=True,
                                    drop_last=True,
                                    num_workers=FLAGS.num_workers)
        self.val_dl = handover_dl(mode='valid',
                                  data_dir=FLAGS.data_dir,
                                  batch_size=FLAGS.batch_size,
                                  num_frames=FLAGS.num_frames,
                                  num_classes=FLAGS.num_classes,
                                  shuffle=False,
                                  drop_last=True,
                                  num_workers=FLAGS.num_workers)
        self.num_training_batches = len(self.train_dl)
        self.loss_fn = loss_fn

    def prepare_data(self):
        # TODO: this could include the generation of the h5 files
        pass
    
    def forward(self, input_sequence):
        ''' Forward pass of the PredNet model
        
        Args:
        -----
        input_sequence: torch.Tensor of shape (b, c, h, w, n_frames)
            Image input sequence
        
        Returns:
        --------
        E_seq: list of torch.Tensor of shape (b, c, h*, w*)
            where h* and w* get smaller as the layer gets deeper
            Sequence of n_frames prediction error signals
        P_seq: list of torch.Tensor of shape (b, c, h*, w*)
            Sequence of n_frames image predictions
        S_seq: list of torch.Tensor of shape (b, c, h*, w*)
            Sequence of n_frames segmentation predictions

        '''
        E_seq, P_seq, S_seq = [], [], []
        for t in range(FLAGS.num_frames):
            input_image = input_sequence[..., t]
            E, P, S = self.model(input_image, t)
            E_seq.append(E); P_seq.append(P); S_seq.append(S)
        return E_seq, P_seq, S_seq
            
    def training_step(self, batch, batch_idx):
        ''' Training step of the model
        
        Args:
        -----
        batch: dict
            Dictionary containing the batch data
        batch_idx: int
            Index of the batch being processed

        Returns:
        --------
        dict:
            Dictionary containing the loss

        '''
        E_seq, _, S_seq = self.forward(batch['samples'])
        S_seq_true = batch['labels']
        loss = self.loss_fn(E_seq,
                            S_seq,
                            S_seq_true,
                            pred_flag=FLAGS.pred_loss)
        if FLAGS.scheduler_type == 'onecycle' or batch_idx == 0:
            self.lr_schedulers().step()
        self.log('train_loss', loss.cpu().detach())
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        ''' Validation step of the model
            
        Args:
        -----
        batch: dict
            Dictionary containing the batch data
        batch_idx: int
            Index of the batch being processed

        Returns:
        --------
        dict:
            Dictionary containing the loss and prediction

        '''
        E_seq, P_seq, S_seq = self.forward(batch['samples'])
        P_seq_true, S_seq_true = batch['samples'], batch['labels']
        loss = self.loss_fn(E_seq, S_seq, S_seq_true, val_flag=True)
        self.log('valid_loss', loss.cpu())
        sequences = {'P_seq': torch.stack(P_seq, dim=-1),
                     'P_seq_true': P_seq_true,
                     'S_seq': torch.stack(S_seq, dim=-1),
                     'S_seq_true': S_seq_true}
        return {'loss': loss, 'sequences': sequences}
    
    def validation_epoch_end(self, outputs):
        ''' Instructions run at the end of every validation epoch
            
        Args:
        -----
        outputs: list of dict
            List of dictionaries containing the loss and prediction

        '''
        sequences = outputs[0]['sequences']
        writer = self.logger.experiment
        writer.add_video(tag='valid_reconstruction',
                         vid_tensor=plot_recons(**sequences),
                         global_step=self.global_step,
                         fps=12)

    def train_dataloader(self):
        ''' Training dataloader '''
        return self.train_dl

    def val_dataloader(self):
        ''' Validation dataloader '''
        return self.val_dl
    
    def configure_optimizers(self):
        ''' Define the optimizer and  the learning rate scheduler '''
        optimizer = torch.optim.Adam(self.parameters(), lr=FLAGS.lr)
        scheduler = select_scheduler(optimizer,
                                     scheduler_type=FLAGS.scheduler_type,
                                     lr=FLAGS.lr,
                                     num_epochs=FLAGS.num_epochs,
                                     num_batches=self.num_training_batches)
        return [optimizer], [scheduler]
    

def compute_fair_number_of_channels():
    ''' Define the layers of the PredNet
        so that any combination has the same number of parameters
    
    Returns:
    --------
    channels: tuple of int
        Tuple of channels for each layer of PredNet

    '''
    if FLAGS.pred_loss:
        scale = {'hgru': 64, 'conv': 74, 'lstm': 28}[FLAGS.rnn_type]
        channels = tuple([scale * 2 ** i for i in range(FLAGS.n_layers)])
    else:
        scale = {'hgru': 68, 'conv': 96, 'lstm': 30}[FLAGS.rnn_type]
        channels = tuple([scale * 2 ** i for i in range(FLAGS.n_layers)])
    return channels


def generate_ckpt_path(model_name, model_version):
    ''' Generate the path to a potentially existing checkpoint.
        Note that, in case a ckpt is found but load_model=False,
        earlier checkpoints are added until the lasest one is
        reached, point from which the latest one is overwritten.

    Args:
    -----
    model_name: str
        Name of the model
    model_version: int
        Version of the model

    Returns:
    --------
    ckpt_path: str or None
        Path to checkpoint file (None if no checkpoint found)

    '''
    if FLAGS.load_model:
        try:
            ckpt_dir = os.path.join(FLAGS.logs_dir,
                                    model_name,
                                    f'version_{model_version}',
                                    'checkpoints')
            ckpt_path = os.path.join(ckpt_dir,
                                     os.listdir(ckpt_dir)[-1])
        except FileNotFoundError:
            os.makedirs(FLAGS.logs_dir, exist_ok=True)
            ckpt_path = None
    else:
        os.makedirs(FLAGS.logs_dir, exist_ok=True)
        ckpt_path = None
    return ckpt_path


def train_one_net(model_version):
    ''' Train one PredNet model using the pytorch-lightning framework
        
    Args:
    -----
    model_version: str
        Version of the model, to index different runs
        
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = f'Prednet_L-{FLAGS.n_layers}_R-{FLAGS.rnn_type}'\
               + f'_A-{FLAGS.axon_delay}_P-{FLAGS.pred_loss}'
    model = PLPredNet(device)
    trainer = pl.Trainer(default_root_dir=FLAGS.logs_dir,
                         gpus=(1 if device=='cuda' else 0),
                         max_epochs=FLAGS.num_epochs,
                         fast_dev_run=FLAGS.debug,
                         log_every_n_steps=5,
                         logger=pl.loggers.TensorBoardLogger(
                             save_dir=FLAGS.logs_dir,
                             name=model_name,
                             version=model_version))
    ckpt_path = generate_ckpt_path(model_name, model_version)
    trainer.fit(model, ckpt_path=ckpt_path)


def main(_):
    ''' Run the training procedure for a PredNet model '''
    pl.seed_everything(4, workers=True)
    for model_version in range(FLAGS.num_versions):
        torch.cuda.empty_cache()  # make sure available memory is freed (?)
        train_one_net(model_version)


if __name__ == '__main__':
    app.run(main)
