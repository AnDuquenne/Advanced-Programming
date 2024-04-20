"""
Transformer model for stock forecasting trainer
"""

import numpy as np
import torch

import matplotlib.pyplot as plt

import datetime

from tqdm import tqdm

import wandb

from utils.utils import mkdir_save_model, log_gradient_norms


class Trainer:
    def __init__(self, train_loader, test_loader, model, optimizer, criterion, scheduler, device, n_epoch,
                 save_path_loss, save_path_weights, model_name, debug, save_path, wandb_=False):

        self.wandb_ = wandb_

        if self.wandb_:
            # Initialize wandb
            wandb.init(project="stock_forecasting", entity="anduquenne")

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device

        self.n_epoch = n_epoch

        self.save_path_loss = save_path_loss
        self.save_path_weights = save_path_weights

        self.name = model_name
        self.debug = debug

        timestamp = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        self.save_path = save_path + "/" + timestamp

    def train(self):

        mkdir_save_model(self.save_path)

        # Initialize the loss history
        epoch_train_loss = np.zeros((self.n_epoch, 1))
        epoch_test_loss = np.zeros((self.n_epoch, 1))

        # Initialize the lr history
        epoch_lr = np.zeros((self.n_epoch, 1))

        for epoch in tqdm(range(self.n_epoch)):

            tmp_train_loss = np.zeros((len(self.train_loader), 1))
            tmp_test_loss = np.zeros((len(self.test_loader), 1))

            for idx, (signal_encoder, signal_decoder, target) in enumerate(self.train_loader):

                self.model.train()

                signal_encoder = signal_encoder.float()
                signal_decoder = signal_decoder.float()
                target = target.float()

                # Move the data to the device
                signal_encoder = signal_encoder.to(self.device)
                signal_decoder = signal_decoder.to(self.device)
                target = target.to(self.device)

                # Reset grad
                self.optimizer.zero_grad()
                # Make predictions
                preds = self.model(signal_encoder, signal_decoder).to(self.device)

                loss = self.criterion(preds, target)

                if self.debug:
                    print("signal_encoder", signal_encoder.size(), signal_encoder[0, :, :].cpu())
                    print("signal_decoder", signal_decoder.size(), signal_decoder[0, :, :].cpu())
                    print("preds", preds.size(), preds[0, :, :].cpu())
                    print("targets", target.size(), target[0, :, :].cpu())

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.7)
                self.optimizer.step()

                tmp_train_loss[idx] = np.mean(loss.cpu().detach().item())

                if idx == len(self.train_loader) - 1:
                    with torch.no_grad():
                        self.model.eval()
                        for idx_test, (signal_encoder_test, signal_decoder_test,
                                       target_test) in enumerate(self.test_loader):

                            signal_encoder_test = signal_encoder_test.float()
                            signal_decoder_test = signal_decoder_test.float()
                            target_test = target_test.float()

                            signal_encoder_test = signal_encoder_test.to(self.device)
                            signal_decoder_test = signal_decoder_test.to(self.device)
                            target_test = target_test.to(self.device)

                            preds_test = self.model(signal_encoder_test, signal_decoder_test)

                            if self.debug:
                                print("Training testing part")
                                print("signal", signal_encoder_test[0, :, :].cpu())
                                print("signal_decoder", signal_decoder_test[0, :, :].cpu())
                                print("preds", preds_test[0, :, :].cpu())
                                print("targets", target_test[0, :, :].cpu())

                            loss_test = self.criterion(preds_test, target_test)
                            tmp_test_loss[idx_test] = np.mean(loss_test.cpu().detach().item())

                if self.wandb_:
                    total_norm = log_gradient_norms(self.model)
                    wandb.log({"Gradient Norm": total_norm}, step=epoch)

            epoch_train_loss[epoch] = np.mean(tmp_train_loss)
            epoch_test_loss[epoch] = np.mean(tmp_test_loss)
            epoch_lr[epoch] = self.optimizer.param_groups[0]['lr']

            # Save the model every 10 epochs (preventive measure)
            if epoch % 10 == 0 and epoch != 0:
                # Save the model
                torch.save(self.model.state_dict(), self.save_path + "/weights/" + "weights.pt")

                # save the loss
                np.save(self.save_path + "/loss/" + "train_loss.npy", epoch_train_loss)
                np.save(self.save_path + "/loss/" + "test_loss.npy", epoch_test_loss)

            if self.wandb_ and epoch >= 5:
                wandb.log({"train_loss": np.mean(tmp_train_loss)}, step=epoch)
                wandb.log({"test_loss": np.mean(tmp_test_loss)}, step=epoch)
                wandb.log({"lr": self.optimizer.param_groups[0]['lr']}, step=epoch)

            self.scheduler.step()

        if self.wandb_:
            wandb.finish()

    def evaluate(self, signal_encoder_batch, signal_decoder_batch, signal_target_batch):
        """
        The function will take a batch of size n, of n sequences in temporal order, and will return the predictions of
        the n-1 next periods.

        In other words, taking the first value of each batch: in -> [t0, t1, ..., t-size_batch]     (signal)
                                                              out -> [t1, t2, ..., t-size_batch+1]  (predictions)

        Then plots [t1, ..., t-size_batch] for in and out
        :param data:
        :return:
        """

        # self.model.eval()
        #
        with torch.no_grad():

            signal_encoder_batch = signal_encoder_batch.float().to(self.device)
            signal_decoder_batch = signal_decoder_batch.float().to(self.device)
            signal_target_batch = signal_target_batch.float().to(self.device)

            preds = self.model(signal_encoder_batch, signal_decoder_batch)

            print("preds", preds.size())
            print("signal_encoder", signal_encoder_batch.size())
            print("signal_decoder", signal_decoder_batch.size())

            targets = signal_target_batch[1:, -1, :]
            preds = preds[:-1, -1, :]

        plt.plot(targets.cpu().detach().numpy(), label="targets")
        plt.plot(preds.cpu().detach().numpy(), label="predictions")
        plt.legend()
        plt.show()

