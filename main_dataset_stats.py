import glob
import os
import zipfile
import numpy as np
import requests
import torch
from colorama import init as colorama_init
from colorama import Fore, Style
from PIL import Image
from torchvision import transforms
import argparse
from dataset_new import Dataset
import random

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
from collections import Counter

def analyze_dataset(dataset_path):
    images_dir = os.path.join(dataset_path, 'train', 'images')
    labels_file = os.path.join(dataset_path, 'train', 'labels.txt')
    
    # Listas para guardar estatísticas
    all_labels = []
    digits_per_image = []
    all_sizes = []
    
    # Lista para visualização (mosaico)
    visualization_data = []

    with open(labels_file, 'r') as f:
        lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 6: continue # Ignora linhas vazias ou mal formadas
            
            img_name = parts[0]
            
            # Se a segunda parte for a label (e não o número de dígitos), 
            # as coordenadas começam logo a seguir.
            # Vamos verificar se a Versão A ou C:
            if len(parts) == 6: # FORMATO VERSÃO A: img label x y w h
                num_digits = 1
                cursor = 1
            else: # FORMATO VERSÃO C: img num_digits label x y w h ...
                num_digits = int(parts[1])
                cursor = 2
            
            digits_per_image.append(num_digits)
            img_bboxes = []
            for _ in range(num_digits):
                try:
                    label = int(parts[cursor])
                    # Tenta ler os 4 valores. O erro acontecia aqui.
                    x = int(parts[cursor+1])
                    y = int(parts[cursor+2])
                    w = int(parts[cursor+3])
                    h = int(parts[cursor+4])
                
                    all_labels.append(label)
                    all_sizes.append((w, h))
                    img_bboxes.append({'label': label, 'bbox': (x, y, w, h)})
                    cursor += 5
                except IndexError:
                    print(f"Erro na linha da imagem {img_name}.Formato inesperado.")
                    break
            
            # Guardamos as primeiras 9 imagens para o mosaico
            if len(visualization_data) < 9:
                visualization_data.append((img_name, img_bboxes))

    # --- PARTE 1: VISUALIZAÇÃO (MOSAICO 3x3) ---
    fig, axs = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle(f'Visualização: {dataset_path}', fontsize=16)
    
    for i, (img_name, bboxes) in enumerate(visualization_data):
        ax = axs[i//3, i%3]
        img_path = os.path.join(images_dir, img_name)
        img = Image.open(img_path)
        ax.imshow(img, cmap='gray')
        
        for item in bboxes:
            label = item['label']
            x, y, w, h = item['bbox']
            # Desenhar o retângulo (x, y, largura, altura)
            rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x, y-2, str(label), color='red', fontsize=10, fontweight='bold')
        
        ax.set_title(img_name)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

    # --- PARTE 2: ESTATÍSTICAS ---
    print(f"\n--- Estatísticas para {dataset_path} ---")
    
    # 1. Distribuição de Classes (Quantos 0s, 1s, etc)
    class_counts = Counter(all_labels)
    plt.figure(figsize=(10, 4))
    plt.bar(class_counts.keys(), class_counts.values())
    plt.title("Distribuição de Classes (0-9)")
    plt.xlabel("Dígito")
    plt.ylabel("Frequência")
    plt.xticks(range(10))
    plt.show()

    # 2. Histograma de dígitos por imagem
    plt.figure(figsize=(10, 4))
    plt.hist(digits_per_image, bins=range(min(digits_per_image), max(digits_per_image) + 2), align='left', rwidth=0.8)
    plt.title("Histograma: Número de dígitos por imagem")
    plt.xlabel("Quantidade de dígitos")
    plt.ylabel("Número de imagens")
    plt.show()

    # 3. Tamanho médio
    avg_w = np.mean([s[0] for s in all_sizes])
    avg_h = np.mean([s[1] for s in all_sizes])
    print(f"Tamanho médio dos dígitos: {avg_w:.2f}x{avg_h:.2f} pixels")

# Chamada da função para a tua Versão A ou C
analyze_dataset('Dataset_Cenas_Versão_A')
analyze_dataset('Dataset_Cenas_Versão_D')