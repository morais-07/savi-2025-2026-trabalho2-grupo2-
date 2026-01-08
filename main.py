#!/usr/bin/env python3
# shebang line for linux / mac

import glob
import os
from random import randint
import shutil
import signal
# from matplotlib import pyplot as plt
from matplotlib import pyplot as plt
import numpy as np
import argparse

import torch
# NOVO (Aponta para o teu ficheiro novo):
from dataset_new import Dataset

from torchvision import transforms
from model import ModelBetterCNN, ModelFullyconnected, ModelConvNet, ModelConvNet3
from trainer import Trainer
from datetime import datetime


def sigintHandler(signum, frame):
    print('SIGINT received. Exiting gracefully.')
    exit(0)


def main():

    # ------------------------------------
    # Setup argparse
    # ------------------------------------
    parser = argparse.ArgumentParser()

    # MUDANÇA 1: Usamos './data'. 
    # Isto vai criar: /home/rafael-morais/Desktop/Trabalho_2/Tarefa_1/data
    parser.add_argument('-df', '--dataset_folder', type=str,
                        default='./data',
                        help='Path where the dataset will be downloaded')

    # MUDANÇA 2: 100% dos dados para a Tarefa 1
    parser.add_argument('-pe', '--percentage_examples', type=float, default=1.0,
                        help='Percentage of examples to use for training and testing')

    parser.add_argument('-ne', '--num_epochs', type=int, default=10,
                        help='Number of epochs for training')

    parser.add_argument('-bs', '--batch_size', type=int, default=64,
                        help='Batch size for training and testing.')

    # MUDANÇA 3: Usamos './experiments'.
    # Isto vai criar: /home/rafael-morais/Desktop/Trabalho_2/Tarefa_1/experiments
    parser.add_argument('-ep', '--experiment_path', type=str,
                        default='./experiments',
                        help='Path to save experiment results.')

    parser.add_argument('-rt', '--resume_training', action='store_true',
                        help='Resume training from last checkpoint if available.')

    args = vars(parser.parse_args())
    print(args)

    # ------------------------------------
    # Register the sigtinthandler
    # ------------------------------------
    signal.signal(signal.SIGINT, sigintHandler)

    # ------------------------------------
    # Create the experiment
    # ------------------------------------

    # experiment_name = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    # experiment_name = datetime.today().strftime('%Y-%m-%d %H')  # same experiment every hour
    # experiment_name = datetime.today().strftime('%Y-%m-%d %H')  # same experiment every hour
    # args['experiment_full_name'] = os.path.join(
    #     args['experiment_path'], experiment_name)
    args['experiment_full_name'] = args['experiment_path']

    print('Starting experiment: ' + args['experiment_full_name'])

    # if os.path.exists(args['experiment_full_name']):
    #     shutil.rmtree(args['experiment_full_name'])
    #     print('Experiment folder already exists. Deleting to start fresh.')

    os.makedirs(args['experiment_full_name'], exist_ok=True)

    # ------------------------------------
    # Create datasets
    # ------------------------------------
    train_dataset = Dataset(args, is_train=True)
    test_dataset = Dataset(args, is_train=False)

    # ------------------------------------
    # Create the model
    # ------------------------------------
    # model = ModelFullyconnected()
    # model = ModelConvNet()
    # model = ModelConvNet3()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModelBetterCNN().to(device)
    # DEPOIS TRANSFORMAR EM ARGUMENTO PARA ESCOLHER O MODELO
    # DEPOIS TRANSFORMAR EM ARGUMENTO PARA ESCOLHER O MODELO
    # DEPOIS TRANSFORMAR EM ARGUMENTO PARA ESCOLHER O MODELO
    # DEPOIS TRANSFORMAR EM ARGUMENTO PARA ESCOLHER O MODELO
    # DEPOIS TRANSFORMAR EM ARGUMENTO PARA ESCOLHER O MODELO

# ------------------------------------
    # Start training
    # ------------------------------------
    trainer = Trainer(args, train_dataset, test_dataset, model)

    # --- APAGUEI O BLOCO QUE DAVA ERRO AQUI ---
    # Aquelas linhas do __getitem__ e model.forward manuais 
    # não são precisas para a Tarefa 1.
    
    trainer.train()     # Inicia o treino (loops e épocas)
    trainer.evaluate()  # Faz a avaliação final (Matriz de Confusão e F1)


if __name__ == '__main__':
    main()
