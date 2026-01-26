import torch

try:
    torch.backends.nnpack.enabled = False
except AttributeError:
    pass

import torch.nn.functional as F
from torchvision import transforms
from torchvision.ops import nms 
from PIL import Image, ImageDraw
import os
import glob
from tqdm import tqdm
from model import ModelBetterCNN 
import numpy as np
import matplotlib.pyplot as plt

def load_trained_model(checkpoint_path, device):

    model = ModelBetterCNN().to(device) # Importa o modelo betterCNN
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint não encontrado: {checkpoint_path}")
    print(f"[INFO] A carregar modelo de: {checkpoint_path}")
    
    # Carregar pesos
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif isinstance(checkpoint, dict): 
            # Tenta carregar direto se o dict forem só os pesos
            model.load_state_dict(checkpoint, strict=False)
        else:
            # Caso seja o modelo inteiro salvo (menos comum hoje em dia)
            model = checkpoint
            
    except Exception as e:
        print(f"[ERRO] Falha ao carregar pesos: {e}")
        exit()

    model.eval()
    return model

def calculate_entropy(probs):
    # Calcula a entropia da distribuição de probabilidades (Incerteza).
    # Entropia = - Σ p(x) * log(p(x))
    # Se o modelo estiver confuso, a entropia será alta.
    log_probs = torch.log(probs + 1e-10) 
    entropy = -torch.sum(probs * log_probs, dim=1) # Entropia por amostra
    return entropy.item()

def sliding_window(image, step_size, window_size):
    w, h = image.size
    # O '+1' garante que chegamos à borda
    for y in range(0, h - window_size[1] + 1, step_size):
        for x in range(0, w - window_size[0] + 1, step_size):
            window = image.crop((x, y, x + window_size[0], y + window_size[1]))
            yield (x, y, window)

def calculate_iou(box1, box2):
    """Calcula se duas caixas se sobrepõem (Intersection over Union)"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

def load_ground_truth(labels_path):
    """Lê o ficheiro labels.txt para sabermos onde estão os dígitos reais"""
    gt_data = {}
    if not os.path.exists(labels_path):
        return {}
    with open(labels_path, 'r') as lines:
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 2: continue
            filename = parts[0]
            boxes = []
            cursor = 2
            # Ajusta o cursor dependendo se tem numero de digitos ou nao (Versao A vs D)
            # Assumindo formato Versao D: [Nome] [N] [Label X Y W H]...
            num_digits = int(parts[1])
            for _ in range(num_digits):
                try:
                    lbl, x, y, w, h = map(int, parts[cursor:cursor+5])
                    boxes.append([x, y, x+w, y+h, lbl]) # [x1, y1, x2, y2, label]
                    cursor += 5
                except: break
            gt_data[filename] = boxes
    return gt_data

def visualize_grid(images_list, titles_list):
    """Cria o mosaico 3x3"""
    if not images_list: return
    # Limita a 9 imagens
    images_list = images_list[:9]
    titles_list = titles_list[:9]
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        if i < len(images_list):
            ax.imshow(images_list[i])
            ax.set_title(titles_list[i], fontsize=8)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def main():

    # ------------------
    # CONFIGURAÇÕES
    # ------------------
    CHECKPOINT_PATH = './experiments/checkpoint.pkl'
    TEST_IMAGES_DIR = './Dataset_Cenas_Versão_D/test/images/'
    OUTPUT_DIR = './results_sliding_window'
    WINDOW_SIZE = (32, 32) # Tamanho da janela deslizante (em pixels)
    STEP_SIZE = 4 # Passo da janela deslizante (em pixels)
    CONFIDENCE_THRESHOLD = 0.98 # Confiança mínima (Probabilidade)
    ENTROPY_THRESHOLD = 0.1 # Entropia máxima (Se > 0.1, a rede está confusa/fundo)
    IOU_THRESHOLD = 0.1 # Sobreposição máxima permitida (NMS)
    PIXEL_THRESHOLD = 70 # Se a janela tiver menos que X intensidade de pixel, ignora (é fundo preto)
    
    # Carregar Ground Truth
    LABELS_FILE = './Dataset_Cenas_Versão_D/test/labels.txt'
    ground_truth = load_ground_truth(LABELS_FILE)

    # Contadores para estatísticas
    total_tp = 0
    total_fp = 0
    total_gt = 0
    grid_images = []
    grid_titles = []


    # ------------------
    # Início do Processo
    # ------------------

    print("\n--- Início do Processo de Deteção com Sliding Window ---\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"A usar dispositivo: {device}")
    
    # Verificar se a pasta de imagens de teste existe
    if not os.path.exists(TEST_IMAGES_DIR):
        print(f"[ERRO] Diretoria de imagens não encontrada: {TEST_IMAGES_DIR}")
        return

    # Criar pasta de output se não existir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Carregar modelo treinado (BetterCNN)
    model = load_trained_model(CHECKPOINT_PATH, device)
    
    # Transformação (Igual ao treino da Tarefa 1)
    transform = transforms.Compose([
        transforms.Resize((28, 28)), # Garante 28x28
        transforms.ToTensor(), # Converte para Tensor
        transforms.Normalize((0.1307,), (0.3081,)) # Normalização MNIST
    ])

    image_files = glob.glob(os.path.join(TEST_IMAGES_DIR, "*.png")) # Lista de imagens .png na pasta de teste
    
    # Verificar se há imagens para processar
    if len(image_files) == 0:
        print("[ERRO] Nenhuma imagem .png encontrada na pasta de teste.")
        return

    print(f"A processar {len(image_files)} imagens...")

    # ------------------
    # Processamento das Imagens
    # ------------------

    # Loop sobre cada imagem de teste
    for img_path in image_files:

        all_boxes = [] # Coordenadas das caixas detetadas
        all_scores = [] # Scores das detecções
        all_labels = [] # Labels das detecções
        final_preds = [] # Deteções finais após NMS

        original_img = Image.open(img_path).convert('L') # Converter para grayscale (MNIST é grayscale)
    
        draw_img = original_img.convert('RGB') # Converte para RGB para desenhar caixas
        draw = ImageDraw.Draw(draw_img) # Objeto para desenhar - caixas
        filename = os.path.basename(img_path) # Nome do ficheiro da imagem

        # Cálculo do total de janelas para a barra de progresso
        total_windows_x = (original_img.size[0] - WINDOW_SIZE[0]) // STEP_SIZE + 1
        total_windows_y = (original_img.size[1] - WINDOW_SIZE[1]) // STEP_SIZE + 1
        total_windows = total_windows_x * total_windows_y

        with tqdm(total=total_windows, desc=f"Scan: {filename}", leave=False) as pbar: # Barra de progresso para cada imagem

            for (x, y, window) in sliding_window(original_img, STEP_SIZE, WINDOW_SIZE): # Gerar janelas deslizantes
                
                # -----------------
                # Ignorar janelas quase pretas
                # -----------------
                window_tensor_check = transforms.ToTensor()(window) # Converter para tensor para análise
                if window_tensor_check.sum() < (PIXEL_THRESHOLD / 255.0): # Normalizado entre 0 e 1, se for quase preto não processa
                    pbar.update(1) # Atualiza a barra de progresso
                    continue
                # ----------------- 
                # Classificação da janela
                # -----------------
                input_tensor = transform(window).unsqueeze(0).to(device) # Transformar e adicionar dimensão batch (1, C, H, W)

                with torch.no_grad(): # Desliga o cálculo dos gradientes
                    output = model(input_tensor) # Logits de saída do modelo (números brutos)
                    probs = F.softmax(output, dim=1) # Converter para probabilidades com Softmax (0 a 1)

                    conf, pred_class = torch.max(probs, 1) # Confiança e classe prevista
                    conf = conf.item() # Converter para float
                    pred_class = pred_class.item() # Converter para int
                    
                    entropy = calculate_entropy(probs) # Calcular entropia (incerteza)
                
                #-----------------
                # Verificar limiares de confiança e entropia
                #-----------------
                if conf >= CONFIDENCE_THRESHOLD and entropy < ENTROPY_THRESHOLD:
                    all_boxes.append([x, y, x + WINDOW_SIZE[0], y + WINDOW_SIZE[1]]) # [x1, y1, x2, y2]
                    all_scores.append(conf) # Score da deteção
                    all_labels.append(pred_class) # Label da deteção

                pbar.update(1) # Atualiza a barra de progresso

        # ------------------
        # NON-MAXIMUM SUPPRESSION (NMS)
        # -----------------

        # Basicamente, o NMS remove caixas que se sobrepõem demasiado, mantendo apenas a mais confiável.
        # Isto ajuda a reduzir múltiplas deteções do mesmo objeto, tornando os resultados melhores.

        if len(all_boxes) > 0: # Se houver deteções e houver mais que uma caixa para processar
            boxes_tensor = torch.tensor(all_boxes, dtype=torch.float32) # Converter para tensor
            scores_tensor = torch.tensor(all_scores, dtype=torch.float32) # Converter para tensor
            keep_indices = nms(boxes_tensor, scores_tensor, IOU_THRESHOLD) # Aplicar NMS
            tqdm.write(f" -> {filename}: {len(keep_indices)} deteções após NMS.")

            # ------------------
            # Desenhar caixas finais após NMS
            # ------------------

            for idx in keep_indices: # Iterar sobre as caixas mantidas após NMS
                idx = idx.item() # Converter para int
                box = all_boxes[idx] # Caixa [x1, y1, x2, y2]
                label = all_labels[idx] # Label da caixa
                score = all_scores[idx] # Score da caixa
                final_preds.append((all_boxes[idx], all_labels[idx])) # Guardar para avaliação
                
                x1, y1, x2, y2 = box # Coordenadas da caixa
                
                draw.rectangle([x1, y1, x2, y2], outline="red", width=1) # Desenhar a caixa
            else:
                pass # Nenhuma caixa para desenhar

        # ------------------
        # Avaliação das Deteções
        # ------------------

        if filename in ground_truth:
            gt_boxes = ground_truth[filename]
            total_gt += len(gt_boxes)
            
            for (p_box, p_label) in final_preds:
                match = False
                for g_box in gt_boxes:
                    # g_box[:4] são as coords, g_box[4] é a label real
                    if p_label == g_box[4] and calculate_iou(p_box, g_box[:4]) > 0.5:
                        match = True
                        break
                if match: total_tp += 1
                else: total_fp += 1

        # Guardar imagem individual
        save_path = os.path.join(OUTPUT_DIR, f"res_{filename}")
        draw_img.save(save_path)
        
        # [NOVO] Adicionar à lista para a grelha
        grid_images.append(np.array(draw_img))
        
        # Título da imagem na grelha
        title_text = f"{filename}\nPred: {len(final_preds)}"
        if filename in ground_truth:
            title_text += f" | Real: {len(ground_truth[filename])}"
        grid_titles.append(title_text)


    else:
        tqdm.write(f" -> {filename}: Nenhuma deteção encontrada.") # Nenhuma caixa para desenhar

        save_path = os.path.join(OUTPUT_DIR, f"res_{filename}") # Caminho para salvar a imagem com deteções
        draw_img.save(save_path) # Salvar a imagem com as caixas desenhadas

    print("\n--- Processo Concluído ---")
    print("\n--- PERFORMANCE ---")
    if total_gt > 0:
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / total_gt
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"Total Dígitos Reais (GT): {total_gt}")
        print(f"Total Deteções Feitas:    {total_tp + total_fp}")
        print(f"Acertos (True Pos):       {total_tp}")
        print(f"Erros (False Pos):        {total_fp}")
        print("-" * 20)
        print(f"PRECISÃO: {precision:.2%} (Confiança)")
        print(f"RECALL:   {recall:.2%} (Cobertura)")
        print(f"F1-SCORE: {f1:.2f}")
    else:
        print("Não foi possível calcular métricas (labels.txt vazio ou não encontrado).")
    # [NOVO] MOSTRAR GRELHA
    visualize_grid(grid_images, grid_titles)
    print(f"Resultados salvos em: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()