'''
This part is used to train the speaker model and evaluate the performances
'''

import torch, sys, os, tqdm, numpy, soundfile, time, pickle
import torch.nn as nn
from tools import *
from loss import AAMsoftmax
from model import ECAPA_TDNN
import os
import time
import numpy as np
import random, glob
from scipy import signal
from torch.nn import TransformerEncoder, TransformerEncoderLayer

        
class ECAPAModel(nn.Module):
    def __init__(self, lr, lr_decay, C, n_class, m, s, test_step, **kwargs):
        super(ECAPAModel, self).__init__()

        self.speaker_encoder = ECAPA_TDNN(C=C).cuda()
        self.speaker_encoder = nn.DataParallel(self.speaker_encoder)
        self.speaker_loss = AAMsoftmax(n_class=n_class, m=m, s=s).cuda()
        self.optim = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9, weight_decay=1.0e-4)
        # self.optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=2e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=test_step, gamma=lr_decay)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optim, T_0=40, T_mult=2, eta_min=1e-7)
      
        print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f" % (
                sum(param.numel() for param in self.speaker_encoder.parameters()) / 1024 / 1024))


    def train_network(self, epoch, loader):
        self.train()
        self.scheduler.step(epoch - 1)
        total_loss = 0
        total_correct = 0
        total_samples = 0
        lr = self.optim.param_groups[0]['lr']
        
        for num, (datares, labels) in enumerate(loader, start=1):
            self.zero_grad()
            # compute output
           
            labels = torch.LongTensor(labels).cuda()
            
            feature = self.speaker_encoder.forward(datares.cuda(), aug=True)

            nloss, correct = self.speaker_loss(feature,labels)
                                   
            nloss.backward()
            self.optim.step()
            total_loss += nloss.item()
            total_samples += labels.size(0)
            
            # ========== 计算准确率 ==========
            total_correct += correct.item()
            # ===============================
            total_steps = len(loader)
            if num % 10 == 0 or num == total_steps:
                accuracy = len(labels) * total_correct / total_samples
                current_time = time.strftime("%m-%d %H:%M:%S")
                training_progress = 100.0 * num / total_steps
                average_loss = total_loss / num
                sys.stderr.write(f"{current_time} [{epoch:2d}] Lr: {lr:.6f}, Training: {training_progress:.2f}%, "
                                f"Loss: {average_loss:.5f}, Acc: {accuracy:.2f}%\r")
                if (num % 10 == 0 or num == total_steps) and num % 50 == 0:
                    sys.stderr.flush()

        return total_loss / num, lr, accuracy


    def eval_network(self, eval_list, eval_path):
        self.eval()
        files = []
        embeddings = {}
        lines = open(eval_list).read().splitlines()
        for line in lines:
            files.append(line.split()[1])
            files.append(line.split()[2])
        setfiles = list(set(files))
        setfiles.sort()

        for idx, file in tqdm.tqdm(enumerate(setfiles), total=len(setfiles)):

            audio, _ = soundfile.read(os.path.join(eval_path, file))
            
            max_length = 9000 * 160 + 240
            if audio.shape[0] > max_length:
               audio = audio[0:32240]
               
            # Full utterance
            data_1 = torch.FloatTensor(numpy.stack([audio], axis=0)).cuda()

            # Spliited utterance matrix
            max_audio = 300 * 160 + 240
            if audio.shape[0] <= max_audio:
                shortage = max_audio - audio.shape[0]
                audio = numpy.pad(audio, (0, shortage), 'wrap')
            feats = []
            startframe = numpy.linspace(0, audio.shape[0] - max_audio, num=5)
            for asf in startframe:
                feats.append(audio[int(asf):int(asf) + max_audio])
            feats = numpy.stack(feats, axis=0).astype(float)
            data_2 = torch.FloatTensor(feats).cuda()
            # Speaker embeddings
            with torch.no_grad():
                embedding_1 = self.speaker_encoder.forward(x = data_1.cuda(), aug = False)
                embedding_1 = F.normalize(embedding_1, p=2, dim=-1)
                embedding_2 = self.speaker_encoder.forward(x = data_2.cuda(), aug = False)
                embedding_2 = F.normalize(embedding_2, p=2, dim=-1)
            embeddings[file] = [embedding_1, embedding_2]

        scores, labels = [], []

        for line in lines:
            embedding_11, embedding_12 = embeddings[line.split()[1]]
            embedding_21, embedding_22 = embeddings[line.split()[2]]
            # Compute the scores
            score_1 = torch.mean(torch.matmul(embedding_11, embedding_21.T))  # higher is positive
            score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))
            score = (score_1 + score_2) / 2
            score = score.detach().cpu().numpy()
            scores.append(score)
            labels.append(int(line.split()[0]))

        # Coumpute EER and minDCF
        EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
        fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
        minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.01, 1, 1)

        return EER, minDCF


    def save_parameters(self, path):
        torch.save(self.state_dict(), path)


    def load_parameters(self, path):
        self_state = self.state_dict()
        loaded_state = torch.load(path)
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("module.", "")
                if name not in self_state:
                    print("%s is not in the model." % origname)
                    continue
            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s" % (
                    origname, self_state[name].size(), loaded_state[origname].size()))
                continue
            self_state[name].copy_(param)


    def accuracy(output, target, topk=(1,)):
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        return res