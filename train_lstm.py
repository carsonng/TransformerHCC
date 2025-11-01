import torch
from torch.utils.data import DataLoader
from readdata.readdata_internal import Classification
import torch.nn as nn
from models.LSTM import ConvLSTMNet
import numpy as np
from tools.tools import calculate_metrics
import os
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import trange

matplotlib.use('Agg')


class CrossValidator:
    def __init__(self, project_name, model_class, device='cuda:0'):
        self.project_name = project_name
        self.model_class = model_class
        self.device = device
        self.result_folder = os.path.join('result', project_name)
        self.weights_folder = os.path.join('weights', project_name)
        os.makedirs(self.result_folder, exist_ok=True)
        os.makedirs(self.weights_folder, exist_ok=True)

        self.EPOCH = 500
        self.patience = 25
        self.lr = 1e-4
        self.wd = 0

    def create_model_and_optimizer(self):
        model = self.model_class().to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.wd)
        criterion = nn.CrossEntropyLoss().to(self.device)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, patience=10, factor=0.5, mode='max'
        )
        return model, optimizer, criterion, lr_scheduler

    def train_epoch(self, model, train_loader, optimizer, criterion):
        model.train()
        per_epoch_loss = 0
        train_num_correct = 0

        with torch.enable_grad():
            for voxel1, voxel2, voxel3, voxel4, voxel5, label in train_loader:
                voxel1 = voxel1.to(self.device)
                voxel2 = voxel2.to(self.device)
                voxel3 = voxel3.to(self.device)
                voxel4 = voxel4.to(self.device)
                voxel5 = voxel5.to(self.device)
                label = label.to(self.device)

                logits = model(voxel1, voxel2, voxel3, voxel4, voxel5)
                loss = criterion(logits, label)
                per_epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pred = logits.argmax(dim=1)
                train_num_correct += torch.eq(pred, label).sum().float().item()

        loss_train = per_epoch_loss / len(train_loader)
        acc = train_num_correct / len(train_loader.dataset)
        return loss_train, acc

    def validate_epoch(self, model, val_loader, criterion):
        model.eval()
        val_true = []
        val_pred = []
        val_prob = []
        per_epoch_loss = 0.
        val_num_correct = 0

        with torch.no_grad():
            for voxel1, voxel2, voxel3, voxel4, voxel5, label, _ in val_loader:
                voxel1 = voxel1.to(self.device)
                voxel2 = voxel2.to(self.device)
                voxel3 = voxel3.to(self.device)
                voxel4 = voxel4.to(self.device)
                voxel5 = voxel5.to(self.device)
                label = label.to(self.device)

                logits = model(voxel1, voxel2, voxel3, voxel4, voxel5)
                loss = criterion(logits, label)
                per_epoch_loss += loss.item()

                pred = logits.argmax(dim=1)
                probabilities = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()

                val_true.extend(label.cpu().numpy())
                val_pred.extend(pred.cpu().numpy())
                val_prob.extend(probabilities)
                val_num_correct += torch.eq(pred, label).sum().float().item()

        loss_val = per_epoch_loss / len(val_loader)
        acc = val_num_correct / len(val_loader.dataset)
        sensitivity, specificity, auc, cm = calculate_metrics(val_true, val_pred, val_prob)

        return loss_val, acc, sensitivity, specificity, auc, cm

    def plot_metrics(self, train_acc, val_acc, train_loss, val_loss,
                     val_sensitivity, val_specificity, fold):
        plt.figure()
        plt.plot(train_acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.title('Accuracy over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(f'{self.result_folder}/accuracy_{fold}.png')
        plt.close()

        plt.figure()
        plt.plot(train_loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.title('Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'{self.result_folder}/loss_{fold}.png')
        plt.close()

        plt.figure()
        plt.plot(val_sensitivity, label='Sensitivity')
        plt.plot(val_specificity, label='Specificity')
        plt.title('Sensitivity and Specificity')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.legend()
        plt.savefig(f'{self.result_folder}/metrics_{fold}.png')
        plt.close()

    def run_fold(self, fold):
        print(f'Cross validation fold {fold + 1}')

        val_dataset = Classification(mode='test')
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, pin_memory=True)
        train_dataset = Classification(mode='train')
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)

        model, optimizer, criterion, lr_scheduler = self.create_model_and_optimizer()

        train_acc, train_loss = [], []
        val_acc, val_loss = [], []
        val_sensitivity, val_specificity = [], []

        trigger = 0
        best_val_loss = 10000.
        best_val_acc = 0.
        save_epoch = 0
        save_acc = 0
        save_auc = 0
        best_sensitivity = 0
        best_specificity = 0

        for epoch in trange(self.EPOCH):
            loss_train, acc_train = self.train_epoch(model, train_loader, optimizer, criterion)
            train_loss.append(loss_train)
            train_acc.append(acc_train)

            loss_val, acc_val, sensitivity, specificity, auc_val, cm = self.validate_epoch(
                model, val_loader, criterion
            )

            val_acc.append(acc_val)
            val_loss.append(loss_val)
            val_sensitivity.append(sensitivity)
            val_specificity.append(specificity)

            lr_now = optimizer.param_groups[0]['lr']
            print(f"train epoch: {epoch + 1}\t loss: {loss_train:.4f}\t acc: {acc_train:.4f} lr:{lr_now:.4e}")
            print(f"val epoch: {epoch + 1} acc: {acc_val:.4f} loss: {loss_val:.4f} auc:{auc_val:.4f} "
                  f"sensitivity:{sensitivity:.4f} specificity:{specificity:.4f}")

            if acc_val - best_val_acc > 0:
                save_epoch = epoch
                save_acc = acc_val
                save_auc = auc_val
                best_val_loss = loss_val
                best_val_acc = acc_val
                best_sensitivity = sensitivity
                best_specificity = specificity
                trigger = 0

                print(f'best val acc:{acc_val:.4f} val loss:{loss_val:.4f} best val auc:{auc_val:.4f}')

                torch.save(model.state_dict(), f'{self.weights_folder}/{fold}.pth')
                print('best weight saved')

                plt.figure()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.ylabel('true label')
                plt.xlabel('predicted label')
                plt.savefig(f'{self.result_folder}/confusion_matrix_{fold}.png')
                plt.close()
            else:
                trigger += 1

            lr_scheduler.step(acc_val)

            self.plot_metrics(train_acc, val_acc, train_loss, val_loss,
                              val_sensitivity, val_specificity, fold)

            if trigger > self.patience:
                print('****************early stopping****************')
                break

        output_str = (
            f'{self.project_name}\n'
            f'lr {self.lr}\n'
            f'cross validation {fold + 1}\n'
            f'best acc:{save_acc:.4f}\n'
            f'best auc:{save_auc:.4f}\n'
            f'best val loss: {best_val_loss:.4f}\n'
            f'best sensitivity: {best_sensitivity}\n'
            f'best specificity: {best_specificity}\n'
        )

        with open(f'{self.project_name}_output.txt', 'a') as file:
            file.write(output_str)

        return save_acc, save_auc, best_val_loss

    def run_cross_validation(self, n_folds=10):
        mean_acc = []
        mean_auc = []
        mean_val_loss = []

        for fold in range(n_folds):
            acc, auc, val_loss = self.run_fold(fold)
            mean_acc.append(acc)
            mean_auc.append(auc)
            mean_val_loss.append(val_loss)

        result_str = (
            f'mean acc {np.mean(mean_acc)}\n'
            f'mean auc {np.mean(mean_auc)}\n'
            f'mean val loss {np.mean(mean_val_loss)}\n'
        )

        with open(f'{self.project_name}_output.txt', 'a') as file:
            file.write(result_str)


def main():
    project_name = 'LSTM'
    validator = CrossValidator(project_name, ConvLSTMNet)
    validator.run_cross_validation(n_folds=10)


if __name__ == "__main__":
    main()




