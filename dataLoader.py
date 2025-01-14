import os
import glob
import random
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset
from scipy import signal


class TrainDataset(Dataset):
    def __init__(self, train_list, train_path, musan_path, rir_path, num_frames, **kwargs):
        self.train_path = train_path
        self.num_frames = num_frames
        self.length = self.num_frames * 160 + 240

        # 加载数据和标签
        self.data_list = []
        self.data_label = []

        # 构建说话人字典
        with open(train_list, 'r') as f:
            lines = f.readlines()
        speakers = [line.strip().split()[0] for line in lines]
        speaker_dict = {speaker: idx for idx, speaker in enumerate(sorted(set(speakers)))}

        for line in lines:
            speaker_label = speaker_dict[line.strip().split()[0]]
            file_name = os.path.join(self.train_path, line.strip().split()[1])
            self.data_label.append(speaker_label)
            self.data_list.append(file_name)

        # 预加载噪声和 RIR 文件列表
        self.noise_types = ['noise', 'speech', 'music']
        self.noise_snr = {'noise': [0, 15], 'speech': [13, 20], 'music': [5, 15]}
        self.num_noise = {'noise': [1, 1], 'speech': [3, 8], 'music': [1, 1]}
        self.noise_list = {noise_type: [] for noise_type in self.noise_types}

        augment_files = glob.glob(os.path.join(musan_path, '*/*/*.wav'))
        for file in augment_files:
            category = os.path.basename(os.path.dirname(os.path.dirname(file)))
            if category in self.noise_list:
                self.noise_list[category].append(file)

        self.rir_files = glob.glob(os.path.join(rir_path, '*/*/*.wav'))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        # 读取音频文件
        audio_path = self.data_list[index]
        audio, sr = sf.read(audio_path)
        audio = np.float32(audio)

        # 裁剪或填充音频
        audio = self._process_audio(audio)

        # 数据增强
        # 以 60% 的概率执行代码
        if random.random() < 0.0:
            audio_aug = self._augment_audio(audio)
            audio_aug = torch.FloatTensor(audio_aug)
        else:
            audio_aug = torch.FloatTensor(audio)
        label = self.data_label[index]

        return audio_aug, label

    def _process_audio(self, audio):
        """裁剪或填充音频到固定长度。"""
        if len(audio) <= self.length:
            shortage = self.length - len(audio)
            audio = np.pad(audio, (0, shortage), mode='wrap')
        else:
            start_frame = np.random.randint(0, len(audio) - self.length)
            audio = audio[start_frame:start_frame + self.length]
        return audio

    def _augment_audio(self, audio):
        """随机选择一种数据增强方式。"""
        aug_type = random.randint(1, 5)
        if aug_type == 1:
            return self._add_reverb(audio)
        elif aug_type in [2, 3, 4]:
            noise_type = self.noise_types[aug_type - 2]
            return self._add_noise(audio, noise_type)
        elif aug_type == 5:
            audio_aug = self._add_noise(audio, 'speech')
            return self._add_noise(audio_aug, 'music')
        else:
            return audio

    def _add_reverb(self, audio):
        """添加混响效果。"""
        rir_file = random.choice(self.rir_files)
        rir, sr = sf.read(rir_file)
        rir = np.float32(rir)
        rir = rir / np.sqrt(np.sum(rir ** 2))
        audio_aug = signal.convolve(audio, rir, mode='full')[:self.length]
        return audio_aug

    def _add_noise(self, audio, noise_type):
        """添加噪声。"""
        clean_db = 10 * np.log10(np.mean(audio ** 2) + 1e-4)
        num_noise = self.num_noise[noise_type]
        num = random.randint(num_noise[0], num_noise[1])
        noise_files = random.sample(self.noise_list[noise_type], num)
        noises = []

        for noise_file in noise_files:
            noise_audio, sr = sf.read(noise_file)
            noise_audio = np.float32(noise_audio)
            noise_audio = self._process_audio(noise_audio)

            noise_db = 10 * np.log10(np.mean(noise_audio ** 2) + 1e-4)
            snr = random.uniform(*self.noise_snr[noise_type])
            noise_audio = np.sqrt(10 ** ((clean_db - noise_db - snr) / 10)) * noise_audio
            noises.append(noise_audio)

        noise = np.sum(noises, axis=0)
        audio_aug = audio + noise
        return audio_aug