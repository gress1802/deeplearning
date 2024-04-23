import torch
from torch.optim import AdamW
from torch.utils.data import Subset
from torchvision.transforms import functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
import os
import pandas as pd
import time
import platform
from IPython.display import display, HTML


import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

def train_model(model, loss_fn, optimizer=None, scheduler=None, epochs=5, train_loader=None,
                val_loader=None, metrics=None, data_module=None,
                save_model_filename=None, load_model_filename=None,
                update_pct_interval=5, max_epochs_display=5, resume_training=False,
                pause_before_train = 5
               ):
    """
    Trains a PyTorch model, with options to load a pre-trained model and
    resume training from a checkpoint.

    Parameters:
    - model: The PyTorch model to be trained.
    - loss_fn: Loss function.
    - optimizer: Optimizer for training.
    - epochs: Number of epochs to train.
    - train_loader: DataLoader for training data.
    - val_loader: DataLoader for validation data.
    - metrics: Metrics to calculate during training.
    - data_module: Data module for handling data loading.
    - save_model_filename: Path to save the best model.
    - load_model_filename: Path to a pre-trained model checkpoint.
    - update_pct_interval: Update display percentage interval.
    - max_epochs_display: How many last epochs to display metrics for.
    - resume_training: Whether to resume training from the checkpoint.
    - pause_before_train: number of seconds to wait to view initial messages.  Default 5.
    """
    
    # Initialize the wrapped model
    pl_model = GenericModel(model, loss_fn, optimizer, metrics, scheduler)
    
    # Load model state from checkpoint if not resuming entire training
    if load_model_filename and not resume_training:
        checkpoint = torch.load(load_model_filename, map_location=lambda storage, loc: storage)
        model_state_dict = checkpoint['state_dict']
        # Adjust for the 'model.' prefix used by PyTorch Lightning
        adjusted_model_state_dict = {k.replace('model.', ''): v for k, v in model_state_dict.items()}
        model.load_state_dict(adjusted_model_state_dict)
    
    # Setup callbacks
    print_progress_cb = PrintProgressMetricsCallback(update_percent=update_pct_interval,
                                                     max_epochs_display=max_epochs_display,
                                                     pause_before_train = pause_before_train)
    callbacks = [print_progress_cb]
    
    # Conditionally add a ModelCheckpoint callback
    if save_model_filename:
        monitor = "val_loss" if val_loader is not None else "train_loss"
        checkpoint_callback = ModelCheckpoint(monitor=monitor, dirpath=".",
                                              filename=save_model_filename, save_top_k=1, mode="min")
        callbacks.append(checkpoint_callback)

    # Initialize the Trainer
    trainer = Trainer(max_epochs=epochs, 
                      callbacks=callbacks, 
                      enable_progress_bar=False,
                      enable_checkpointing=True if save_model_filename else False,
                      num_sanity_val_steps=0,
                     )

    # Determine the checkpoint path for resuming training
    ckpt_path = None
    if resume_training and load_model_filename:
        ckpt_path = load_model_filename

    # Fit the model
    if data_module:
        trainer.fit(pl_model, datamodule=data_module, ckpt_path=ckpt_path)
    else:
        trainer.fit(pl_model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_path)

    # Optionally, return metrics and the trained model
    metrics_df = print_progress_cb.metrics_df

    model = pl_model.model

    return metrics_df


# def train_model(base_model, loss_fn, optimizer=None, epochs=5, train_loader=None,
#                 val_loader=None, metrics=None, data_module=None,
#                 save_model_filename=None, load_model_filename=None,
#                 update_pct_interval=5, max_epochs_display=5, resume_training=False):
#     """
#     Trains a PyTorch model, with the option to load a pre-trained model and
#     resume training from a checkpoint.

#     Parameters
#     ----------
#     base_model : torch.nn.Module
#         The model to train.
#     loss_fn : Callable
#         The loss function.
#     optimizer : torch.optim.Optimizer, optional
#         The optimizer for training.
#     epochs : int, optional
#         The number of epochs for training.
#     train_loader : DataLoader, optional
#         DataLoader for the training data.
#     val_loader : DataLoader, optional
#         DataLoader for the validation data.
#     metrics : dict, optional
#         Metrics to calculate during training.
#     data_module : LightningDataModule, optional
#         Data module for handling data loading.
#     save_model_filename : str, optional
#         Path to save the best model.
#     load_model_filename : str, optional
#         Path to a pre-trained model checkpoint.
#     update_pct_interval : int, optional
#         Update display percentage interval.
#     max_epochs_display : int, optional
#         How many last epochs to display metrics for.
#     resume_training : bool, optional
#         Whether to resume training from the checkpoint.
#     """

#     model = GenericModel(base_model, loss_fn, optimizer, metrics)
    
#     if load_model_filename and not resume_training:
#         # Load only the model's parameters from the checkpoint
#         checkpoint = torch.load(load_model_filename, map_location=lambda storage, loc: storage)
#         model.load_state_dict(checkpoint['state_dict'])
    
#     callbacks = [PrintProgressMetricsCallback(update_percent=update_pct_interval,
#                                               max_epochs_display=max_epochs_display)]
    
#     if save_model_filename:
#         monitor = "val_loss" if val_loader is not None else "train_loss"
#         checkpoint_callback = ModelCheckpoint(
#             monitor=monitor,
#             dirpath=".",
#             filename=save_model_filename,
#             save_top_k=1,
#             mode="min"
#         )
#         callbacks.append(checkpoint_callback)

#     trainer = Trainer(
#         max_epochs=epochs,
#         callbacks=callbacks,
#         enable_progress_bar=False,
#         enable_checkpointing=True if save_model_filename else False,
#         num_sanity_val_steps=0,
#     )

#     # Determine the checkpoint path for resuming training, if applicable
#     ckpt_path = load_model_filename if resume_training else None

#     if data_module is not None:
#         trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_path)
#     else:
#         trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)

#     metrics_df = callbacks[0].metrics_df if metrics is not None else pd.DataFrame()

#     return model, metrics_df

class GenericModel(pl.LightningModule):
    def __init__(self, model, loss_fn, optimizer, metrics, scheduler=None):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        # If optimizer is None, initialize AdamW with model parameters and a default lr of 0.001
        self.optimizer = AdamW(self.model.parameters(), lr=0.001) if optimizer is None else optimizer
        self.scheduler = scheduler
        self.metrics = metrics if metrics is not None else {}
        # Dynamically add metrics as modules
        for metric_name, metric in self.metrics.items():
            self.add_module(metric_name, metric)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        self.log('train_loss', loss.item(), on_step=False, on_epoch=True, prog_bar=False)
        if self.metrics:
            for name, metric in self.metrics.items():
                self.log(f'train_{name}', metric(logits, y).item(), on_step=False, on_epoch=True, prog_bar=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        self.log('val_loss', loss.item(), on_epoch=True, prog_bar=False)
        if self.metrics:
            for name, metric in self.metrics.items():
                self.log(f'val_{name}', metric(logits, y).item(), on_epoch=True, prog_bar=False)

    def configure_optimizers(self):
        if self.scheduler:
            return {'optimizer': self.optimizer, 'lr_scheduler': {'scheduler': self.scheduler, 'interval': 'step'}}
        else:
            return self.optimizer


class PrintProgressMetricsCallback(Callback):

    def __init__(self, update_percent = 10, max_epochs_display = 10, pause_before_train=5):
        super().__init__()
        self.metrics_df = pd.DataFrame()
        self.training_step_total = 0
        self.validation_step_total = 0
        self.validation_elapsed_time = 0 #in case there is no validation step
        self.validation_percent_complete = 0
        self.update_percent = update_percent
        self.max_epochs_display = max_epochs_display
        self.pause_before_train = pause_before_train

    def on_train_start(self, trainer, pl_module):
        # Add a pause at the start of training (default 5 seconds)
        print(f'\n Training starts in {self.pause_before_train} seconds ...')
        time.sleep(self.pause_before_train)

    def _update_display(self, trainer, pl_module):
        clear_output() #this is our own function to work in multiple evirons
        epoch_str = f'Epoch {trainer.current_epoch + 1}/{trainer.max_epochs}'
        training_percent_str = f'Training {self.training_percent_complete:0.2f}% complete'
        display_str = f'{epoch_str}, {training_percent_str}'

        # Check if validation step total is greater than 0 to determine if validation is being performed
        if self.validation_step_total > 0:
            validation_percent_str = f'Validation {self.validation_percent_complete:0.2f}% complete'
            display_str += f', {validation_percent_str}'

        lr = trainer.optimizers[0].param_groups[0]['lr']
        display_str += f' lr = {lr:0.3e}'
        
        print(display_str)
        if not self.metrics_df.empty:
            display_dataframe(self.metrics_df.tail(self.max_epochs_display))

    def on_train_epoch_start(self, trainer, pl_module):
        self.training_start_time = time.time()
        self.training_elapsed_time = 0
        self.training_step_total = len(trainer.train_dataloader)
        self.training_step_counter = 0
        self.training_percent_complete = 0
        if self.validation_step_total > 0:
            self.validation_percent_complete = 0

    def on_validation_epoch_start(self, trainer, pl_module):
        self.validation_start_time = time.time()
        self.validation_elapsed_time = 0
        if trainer.num_val_batches:
            self.validation_step_total = sum(trainer.num_val_batches)
        else:
            self.validation_step_total = 0
        self.validation_step_counter = 0
        self.validation_percent_complete = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.training_step_counter += 1
        self.training_percent_complete = (self.training_step_counter / self.training_step_total) * 100
        if self.training_percent_complete % self.update_percent <= (1 / self.training_step_total) * 100:
            self._update_display(trainer, pl_module)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.validation_step_counter += 1
        self.validation_percent_complete = (self.validation_step_counter / self.validation_step_total) * 100
        if self.validation_percent_complete % self.update_percent <= (1 / self.validation_step_total) * 100:
            self._update_display(trainer, pl_module)

    def on_validation_epoch_end(self, trainer, pl_module):
        self.validation_elapsed_time += time.time() - self.validation_start_time

    def on_train_epoch_end(self, trainer, pl_module):
        new_row = pd.DataFrame({'Epoch':[trainer.current_epoch + 1]})
        self.epoch_metrics = {key: [value.item()] for key, value in sorted(trainer.logged_metrics.items())}
        for col,data in self.epoch_metrics.items():
            new_row[col] = data
        self.training_elapsed_time += time.time() - self.training_start_time
        new_row['Time'] = [self.training_elapsed_time+self.validation_elapsed_time]
        lr = trainer.optimizers[0].param_groups[0]['lr']
        new_row['LR'] =[lr]
        self.metrics_df = pd.concat([self.metrics_df, new_row],ignore_index=True)
        self._update_display(trainer, pl_module)

def in_notebook():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' in get_ipython().config:
            return True
    except Exception:
        return False

# Function to clear the output in the notebook or console
def clear_output():
    if in_notebook():
        from IPython.display import clear_output as clear
        clear(wait=True)
    else:
        os_name = platform.system()
        if os_name == 'Windows':
            os.system('cls')
        else:
            os.system('clear')

# Keep track of whether the dataframe has been displayed for the first time
first_display = True

def display_dataframe(df):
    print(df.to_string(index=False))

class PrintMetricsCallback(Callback):
    def __init__(self, print_epoch=1):
        """
        Args:
            print_epoch (int): Frequency of epochs to print metrics. Default is 1, meaning print every epoch.
        """
        super().__init__()
        self.print_epoch = print_epoch

    # Lightning 2.2 executes on_train_epoch_end AFTER on_valid_epoch_end (go figure ...)

    def on_train_epoch_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch + 1  # Adjust for human-readable epoch numbering
        if current_epoch % self.print_epoch == 0:
            logs = trainer.logged_metrics
            current_lr = trainer.optimizers[0].param_groups[0]['lr']
            metrics_string = ", ".join([f"{key}: {value:0.4f}" for key, value in logs.items()])
            print(f"Epoch {current_epoch} Metrics: {metrics_string}, LR: {current_lr:10e}")

    def on_fit_start(self, trainer, pl_module):
        print(f'Beginning training for at most {trainer.max_epochs} epochs')

    def on_fit_end(self, trainer, pl_module):
        print(f'End.  Trained for {trainer.current_epoch} epochs.')

def denormalize(tensor, mean, std):
    """
    Denormalizes a tensor given the mean and std used for normalization.
    This function supports tensors for single-channel and multi-channel images.
    
    - tensor: Input tensor with shape (C, H, W) or (N, C, H, W).
    - mean: The mean used for normalization (per channel).
    - std: The standard deviation used for normalization (per channel).
    
    Both mean and std arguments should be sequences (e.g., lists or tuples) or scalars,
    with the length equal to the number of channels in the tensor.
    """
    if not isinstance(mean, (list, tuple)):
        mean = [mean]
    if not isinstance(std, (list, tuple)):
        std = [std]
    
    if len(tensor.shape) == 3:  # Single image (C, H, W)
        mean = torch.tensor(mean, dtype=tensor.dtype).view(-1, 1, 1)
        std = torch.tensor(std, dtype=tensor.dtype).view(-1, 1, 1)
    elif len(tensor.shape) == 4:  # Batch of images (N, C, H, W)
        mean = torch.tensor(mean, dtype=tensor.dtype).view(1, -1, 1, 1)
        std = torch.tensor(std, dtype=tensor.dtype).view(1, -1, 1, 1)
    
    return tensor * std + mean

def sample_dataset(dataset, num_samples=1000, random_state=42):
    # for reproducibility
    np.random.seed(random_state)

    # Determine the number of classes
    num_classes = len(dataset.classes)

    # Initialize a list to store indices for each class
    indices_per_class = {class_idx: [] for class_idx in range(num_classes)}

    # Go through the dataset and store indices of each class
    for idx, (_, class_idx) in enumerate(dataset):
        indices_per_class[class_idx].append(idx)

    # Sample indices for each class
    sampled_indices = []
    for class_idx, indices in indices_per_class.items():
        if len(indices) > num_samples:
            sampled_indices.extend(np.random.choice(indices, num_samples, replace=False))
        else:
            sampled_indices.extend(indices)

    sampled_dataset = Subset(dataset, sampled_indices)

    return sampled_dataset

def center_crop_and_resize(image_path, output_size):
    """
    Crop an image to the largest possible center square and then resize it to a specified square size.
    
    Parameters:
    - image_path (str): The file path of the image to be processed.
    - output_size (int): The width and height of the output image in pixels. The output image will be a square,
      so only one dimension is needed.
    
    Returns:
    - Image: A PIL Image object of the cropped and resized image.
    
    This function first calculates the largest square that can be cropped from the center of the original image.
    It then crops the image to this square and resizes the cropped image to the specified dimensions.
    """
    
    # Open the image
    image = Image.open(image_path)
    
    # Calculate the dimensions for a center square crop
    width, height = image.size
    crop_size = min(width, height)
    
    left = (width - crop_size) / 2
    top = (height - crop_size) / 2
    right = (width + crop_size) / 2
    bottom = (height + crop_size) / 2
    
    # Crop the center of the image
    image_cropped = image.crop((left, top, right, bottom))
    
    # Resize the cropped image
    image_resized = image_cropped.resize((output_size, output_size), Image.Resampling.LANCZOS)
    
    return image_resized



class Visualizer():
    def __init__(self, dataset, image_extractor=None, label_extractor=None):
        self.dataset = list(dataset)
        self.image_extractor = image_extractor
        self.label_extractor = label_extractor
        
    def visualize(self, rows=4, cols=4, figsize=(12, 12), sample=False):
        # Visualizer
        fig=plt.figure(figsize=figsize)
        for i in range(1, rows * cols + 1):
            if sample:
                data = random.choice(self.dataset)
            else:
                data = self.dataset[i-1]
            if self.image_extractor:
                img = self.image_extractor(data)
            else:
                img = data
                
            # An image can be a string (filepath), numpy array, PIL Image
            ax = fig.add_subplot(rows, cols, i)
            #ax.set_yticklabels([])
            #ax.set_xticklabels([])
            ax.set_axis_off()
            
            if self.label_extractor:
                label = self.label_extractor(data)
                ax.set_title(label)
                
            if type(img) == Image:
                img = np.array(img)
            if type(img) == torch.Tensor:
                img = img.numpy()
            plt.imshow(img)
        plt.show()

def process_experiment_logs(logs_path, exper_name):
    """
    Processes the latest version of the experiment logs, groups metrics by epoch,
    and adds an experiment name column.  
    
    Parameters:
    - logs_path: Path to the directory containing CSV logs.  
    - legend_name: Name of the experiment for plotting legend.

    Returns:
    - A pandas DataFrame with metrics grouped by epoch and an added 'exp_name' column.
    """
    log_dir = os.path.join(logs_path, "model")
    versions = [d for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))]
    latest_version = sorted(versions, key=lambda x: int(x.split('_')[-1]))[-1]
    latest_log_path = os.path.join(log_dir, latest_version, "metrics.csv")

    # Read the CSV into a DataFrame
    df = pd.read_csv(latest_log_path)

    # Drop the 'step' column if it exists
    if 'step' in df.columns:
        df = df.drop(columns=['step'])

    # Group by 'epoch' and aggregate the metrics
    df_grouped = df.groupby('epoch', as_index=False).mean()

    # Add the 'exp_name' column
    df_grouped['exp_name'] = exper_name

    return df_grouped

class Cutout(object):
    """Randomly mask out one or more patches from an image with a given probability.
    
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
        p (float): Probability that the Cutout is applied.
    """
    
    def __init__(self, n_holes, length, p=1.0):
        assert 0.0 <= p <= 1.0, 'p should be in range [0, 1]'
        self.n_holes = n_holes
        self.length = length
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W) to be masked.
            
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it if applied.
        """
        # Apply Cutout with probability p
        if np.random.rand() > self.p:
            return img
        
        h = img.size(1)
        w = img.size(2)
        
        mask = np.ones((h, w), np.float32)
        
        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            
            mask[y1: y2, x1: x2] = 0.
        
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        
        return img

