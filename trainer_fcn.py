import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

class TrainerFCN:
    def __init__(self, model, train_dataset, test_dataset, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = model.to(self.device)
        
        self.train_loader = DataLoader(
            train_dataset, batch_size=args['batch_size'], shuffle=True
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=args['batch_size'], shuffle=False
        )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.get('lr', 0.001))
        
        #Alocar pesos: fundo-0.1 , dígitos-1.0
        weights = torch.ones(11)
        weights[10] = 0.1  #faz com que a rede foque nos dígitos
        self.criterion = nn.CrossEntropyLoss(weight=weights.to(self.device)) #erro da rede, dá mais importância a erros onde deviam estar dígitos

        self.criterion_reg = nn.MSELoss() #erro entre as coordenadas previstas e as reais

    def train_epoch(self):
        self.model.train()   #avisa a rede que o treino começou
        running_loss = 0.0
        
        for images, targets_cls, targets_reg in tqdm(self.train_loader, desc="Training FCN"):
            images, targets_cls, targets_reg = images.to(self.device), targets_cls.to(self.device), targets_reg.to(self.device)
            
            self.optimizer.zero_grad()  #limpa erros do ciclo anterior
            
            outputs_cls, outputs_reg = self.model(images)
            
            #Se o target_map do dataset for maior que o output da rede redimensionamos o output
            if outputs_cls.shape[-2:] != targets_cls.shape[-2:]:
                outputs_cls = torch.nn.functional.interpolate(outputs_cls, size=targets_cls.shape[-2:], mode='bilinear', align_corners=False)
                outputs_reg = torch.nn.functional.interpolate(outputs_reg, size=targets_reg.shape[-2:], mode='bilinear', align_corners=False)
            #erro de classificação
            loss_cls = self.criterion(outputs_cls, targets_cls) #calcula o erro entre o output que a rede previu com o target preparado no dataset 
            
            #erro de regressão (Coordenadas)
            #só calculamos erro de regressão onde REALMENTE existe um dígito.
            #criamos uma máscara onde a classe é < 10 (ou seja, não é fundo).
            mask = (targets_cls < 10).unsqueeze(1).expand_as(outputs_reg).float()
            # Multiplicamos o output e o target pela máscara para ignorar o fundo na regressão
            loss_reg = self.criterion_reg(outputs_reg * mask, targets_reg * mask)
            
            loss = loss_cls + (loss_reg * 5.0)
            #multiplica-se a loss_reg por um peso para ambas terem importancia igual

            loss.backward()  #indica como cada peso deve ser ajustado para que no próximo ciclo o erro seja menor
            self.optimizer.step() #atualização dos pesos
            
            running_loss += loss.item()  #erro acumulado 
            
        return running_loss / len(self.train_loader) #erro médio, objetivo: este diminuir em cada epoch

    #testar a rede, perceber se aprendeu a detetar dígitos ou só decorou 
    def evaluate(self):
        self.model.eval()
        test_loss = 0.0
        correct_pixels = 0
        total_pixels = 0
        
        with torch.no_grad():  #não é necessário calcular gradientes para mais tarde corrigir, vamos apensas testar e não treinar
            for images, targets_cls, targets_reg in self.test_loader:
                images, targets_cls, targets_reg = images.to(self.device), targets_cls.to(self.device), targets_reg.to(self.device)
                outputs_cls, outputs_reg = self.model(images)
                
                #redimensionamento
                if outputs_cls.shape[-2:] != targets_cls.shape[-2:]:
                    outputs_cls = torch.nn.functional.interpolate(outputs_cls, size=targets_cls.shape[-2:], mode='bilinear', align_corners=False)
                    outputs_reg = torch.nn.functional.interpolate(outputs_reg, size=targets_reg.shape[-2:], mode='bilinear', align_corners=False)

                loss_cls = self.criterion(outputs_cls, targets_cls)  #calcular erro entre targets criados no dataset e outputs da FCN
                mask = (targets_cls < 10).unsqueeze(1).expand_as(outputs_reg).float()
                loss_reg =  self.criterion_reg(outputs_reg * mask, targets_reg * mask)

                test_loss += (loss_cls + (loss_reg*5.0)).item()    #acumular erro
                
                
                _, predicted = torch.max(outputs_cls, 1) #qual camada tem o valor mais alto em cada pixel
                #torch.max devolve o valor do score e a posição e classe, apenas nos interessa a ultima
                correct_pixels += (predicted == targets_cls).sum().item() #comparação do mapa predicted com o target
                #é criado um mapa de verdadeiros e falsos (True=1 e False=0), são somados os pixeis corretos 
                total_pixels += targets_cls.nelement() #acumulação do nº total de pixeis analizados 
                
        return test_loss / len(self.test_loader), correct_pixels / total_pixels #return do erro médio total e da precisão no acerto de pixeis 

    def save_checkpoint(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Modelo guardado em {path}")