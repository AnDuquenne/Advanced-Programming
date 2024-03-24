import numpy as np
import torch

from tqdm import tqdm

import wandb

class Trainer:
    def __init__(self, train_loader, test_loader, model, optimizer, criterion, device, n_epoch,
                 save_path_loss, save_path_weights, model_name):

        # Initialize wandb
        wandb.init(project="stock_forecasting", entity="anduquenne")

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model

        self.optimizer = optimizer
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.95)
        self.criterion = criterion
        self.device = device

        self.n_epoch = n_epoch

        self.save_path_loss = save_path_loss
        self.save_path_weights = save_path_weights

        self.name = model_name

    def train(self):

        # Initialize the loss history
        epoch_train_loss = np.zeros((self.n_epoch, 1))
        epoch_test_loss = np.zeros((self.n_epoch, 1))

        for epoch in tqdm(range(self.n_epoch)):

            tmp_train_loss = np.zeros((len(self.train_loader), 1))
            tmp_test_loss = np.zeros((len(self.test_loader), 1))

            for idx, (signal, target) in enumerate(self.train_loader):

                self.model.train()

                # We want [window, batch_size, features]
                signal = signal.permute(2, 0, 1)
                target = target.permute(2, 0, 1)

                signal = signal.float()
                target = target.float()

                # Move the data to the device
                signal = signal.to(self.device)
                target = target.to(self.device)

                # Reset grad
                self.optimizer.zero_grad()
                # Make predictions
                preds = self.model(signal.to(self.device))

                loss = self.criterion(preds, target)
                loss.backward()
                self.optimizer.step()

                tmp_train_loss[idx] = np.mean(loss.cpu().detach().item())

                if idx == len(self.train_loader) - 1:
                    with torch.no_grad():
                        self.model.eval()
                        for idx_test, (signal_test, target_test) in enumerate(self.test_loader):

                            signal_test = signal_test.permute(2, 0, 1)
                            target_test = target_test.permute(2, 0, 1)

                            signal_test = signal_test.float()
                            target_test = target_test.float()

                            signal_test = signal_test.to(self.device)
                            target_test = target_test.to(self.device)

                            preds_test = self.model(signal_test.to(self.device))

                            print("signal", signal_test[:, 0, :].cpu())
                            print("preds", preds_test[:, 0, :].cpu())
                            print("targets", target_test[:, 0, :].cpu())

                            loss_test = self.criterion(preds_test, target_test)
                            tmp_test_loss[idx_test] = np.mean(loss_test.cpu().detach().item())

            epoch_train_loss[epoch] = np.mean(tmp_train_loss)
            epoch_test_loss[epoch] = np.mean(tmp_test_loss)

            wandb.log({"train_loss": np.mean(tmp_train_loss)}, step=epoch)
            wandb.log({"test_loss": np.mean(tmp_test_loss)}, step=epoch)

            self.scheduler.step()

        wandb.finish()

        # Save the model
        torch.save(self.model.state_dict(), self.save_path_weights + self.name + ".pt")

        # save the loss
        np.save(self.save_path_loss + self.name + "_train_loss.npy", epoch_train_loss)
        np.save(self.save_path_loss + self.name + "_test_loss.npy", epoch_test_loss)
