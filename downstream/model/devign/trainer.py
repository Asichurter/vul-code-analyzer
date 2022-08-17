import copy
from sys import stderr

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from tqdm import tqdm

from downstream.model.devign.devign_utils import debug
from downstream.model.devign.devign_global_flag import global_cuda_device
from utils.file import load_json, dump_json


def evaluate_loss(model, loss_function, num_batches, data_iter, cuda_device=False):
    model.eval()
    with torch.no_grad():
        _loss = []
        all_predictions, all_targets = [], []
        for _ in range(num_batches):
            graph, targets = data_iter()
            targets = targets.cuda(cuda_device)
            graph.cuda(cuda_device)
            predictions = model(graph, cuda=True)
            batch_loss = loss_function(predictions, targets)
            _loss.append(batch_loss.detach().cpu().item())
            predictions = predictions.detach().cpu()
            if predictions.ndim == 2:
                all_predictions.extend(np.argmax(predictions.numpy(), axis=-1).tolist())
            else:
                all_predictions.extend(
                    predictions.ge(torch.ones(size=predictions.size()).fill_(0.5)).to(
                        dtype=torch.int32).numpy().tolist()
                )
            all_targets.extend(targets.detach().cpu().numpy().tolist())
        model.train()
        return np.mean(_loss).item(), f1_score(all_targets, all_predictions) * 100
    pass


def evaluate_metrics(model, loss_function, num_batches, data_iter, cuda_device):
    model.eval()
    with torch.no_grad():
        _loss = []
        all_predictions, all_targets = [], []
        for _ in range(num_batches):
            graph, targets = data_iter()
            targets = targets.cuda(cuda_device)
            graph.cuda(cuda_device)
            predictions = model(graph, cuda=True)
            batch_loss = loss_function(predictions, targets)
            _loss.append(batch_loss.detach().cpu().item())
            predictions = predictions.detach().cpu()
            if predictions.ndim == 2:
                all_predictions.extend(np.argmax(predictions.numpy(), axis=-1).tolist())
            else:
                all_predictions.extend(
                    predictions.ge(torch.ones(size=predictions.size()).fill_(0.5)).to(
                        dtype=torch.int32).numpy().tolist()
                )
            all_targets.extend(targets.detach().cpu().numpy().tolist())
        model.train()
        return accuracy_score(all_targets, all_predictions) * 100, \
               precision_score(all_targets, all_predictions) * 100, \
               recall_score(all_targets, all_predictions) * 100, \
               f1_score(all_targets, all_predictions) * 100
    pass


def train(model, dataset, max_steps, dev_every, loss_function, optimizer, save_path, cuda_device, log_every=50, max_patience=5, do_val=True, dump_key='default'):
    debug('Start Training')
    train_losses = []
    best_model = None
    patience_counter = 0
    best_f1 = 0
    try:
        train_stat_preds = []
        train_stat_labels = []
        for step_count in range(max_steps):
            model.train()
            model.zero_grad()
            graph, targets = dataset.get_next_train_batch()
            targets = targets.cuda(cuda_device)
            # Fix cuda bug
            graph.cuda(cuda_device)
            predictions = model(graph, cuda=True)
            batch_loss = loss_function(predictions, targets)

            predict_labels = (predictions > 0.5).long().detach().cpu().tolist()
            labels = targets.detach().cpu().tolist()
            train_stat_preds.extend(predict_labels)
            train_stat_labels.extend(labels)
            if log_every is not None and (step_count % log_every == log_every - 1):
                debug('Step %d\t\tTrain Loss %10.3f Train F1 %.2f' % (step_count, batch_loss.detach().cpu().item(), f1_score(train_stat_labels, train_stat_preds) * 100))
                train_stat_preds.clear()
                train_stat_labels.clear()
            train_losses.append(batch_loss.detach().cpu().item())
            batch_loss.backward()
            optimizer.step()
            if step_count % dev_every == (dev_every - 1):
                if do_val:
                    valid_loss, valid_f1 = evaluate_loss(model, loss_function, dataset.initialize_valid_batch(),
                                                         dataset.get_next_valid_batch, cuda_device)
                    if valid_f1 > best_f1:
                        patience_counter = 0
                        best_f1 = valid_f1
                        best_model = copy.deepcopy(model.state_dict())
                        _save_file = open(save_path + '-model.bin', 'wb')
                        torch.save(model.state_dict(), _save_file)
                        _save_file.close()
                    else:
                        patience_counter += 1
                    debug('Step %d\t\tTrain Loss %10.3f\tTrain f1 %5.2f\tValid Loss%10.3f\tf1: %5.2f\tPatience %d' % (
                        step_count, np.mean(train_losses).item(),
                        f1_score(train_stat_labels, train_stat_preds) * 100, valid_loss, valid_f1, patience_counter))
                    debug('=' * 100)
                else:
                    debug('Step %d\t\tTrain Loss %10.3f\tTrain f1 %5.2f' % (
                        step_count, np.mean(train_losses).item(),
                        f1_score(train_stat_labels, train_stat_preds) * 100))
                    debug('=' * 100)
                train_losses = []
                train_stat_preds.clear()
                train_stat_labels.clear()
                if patience_counter == max_patience:
                    break
    except KeyboardInterrupt:
        debug('Training Interrupted by user!')

    if best_model is not None:
        model.load_state_dict(best_model)
    _save_file = open(save_path + '-model.bin', 'wb')
    torch.save(model.state_dict(), _save_file)
    _save_file.close()
    acc, pr, rc, f1 = evaluate_metrics(model, loss_function, dataset.initialize_test_batch(),
                                       dataset.get_next_test_batch, cuda_device)
    result_str = '%s\tTest Accuracy: %0.2f\tPrecision: %0.2f\tRecall: %0.2f\tF1: %0.2f (val f1=%5.2f)' % (save_path, acc, pr, rc, f1, best_f1)
    debug(result_str)
    debug('=' * 100)

    result_file_path = '/data1/zhijietang/temp/devign_results.json'
    results = load_json(result_file_path)
    if dump_key not in results:
        results[dump_key] = [result_str]
    else:
        results[dump_key].append(result_str)
    dump_json(results, result_file_path)
