import numpy as np
import math
import torch.distributed as dist
import torch
import torch.nn as nn
import torch.utils.data as data
import os
import cv2 as cv
import random 
import soundfile as sf 
import librosa
import sys 

from tools import  audioread  

import pdb 

EPS = np.finfo(float).eps
MAX_INT16 = np.iinfo(np.int16).max
np.random.seed(0)
random.seed(0)




class dataset(data.Dataset):
    def __init__(self,
                speaker_dict,
                mix_lst_path,
                visual_direc,
                mixture_direc,
                batch_size,
                partition='val',
                sampling_rate=16000,
                mix_no=2):

        self.minibatch =[]
        self.visual_direc = visual_direc
        self.mixture_direc = mixture_direc
        self.sampling_rate = sampling_rate
        self.partition = partition
        self.C=mix_no
        self.fps = 25
        self.batch_size = batch_size
        self.speaker_id=speaker_dict

        self.normMean = 0.4161
        self.normStd = 0.1688


        mix_lst=open(mix_lst_path).read().splitlines()
        mix_lst=list(filter(lambda x: x.split(',')[0]==partition, mix_lst))


        sorted_mix_lst = sorted(mix_lst, key=lambda data: float(data.split(',')[-1]), reverse=True)
        if self.partition=='train':
            random.shuffle(sorted_mix_lst)
        start = 0
        while True:
            end = min(len(sorted_mix_lst), start + self.batch_size)
            self.minibatch.append(sorted_mix_lst[start:end])
            if end == len(sorted_mix_lst):
                break
            start = end

    def __getitem__(self, index):
        batch_lst = self.minibatch[index]

        min_length = np.inf
        for _ in range(len(batch_lst)):
            if float(batch_lst[_].split(',')[-7])<min_length:
                min_length= float(batch_lst[_].split(',')[-7])
        
        mixtures=[]
        audios=[]
        visuals=[]
        speakers = []


        for line in batch_lst:
            path_line = line.split(',')[0:-6]
            mixture_path=self.mixture_direc+self.partition+'/mix/'+ ','.join(path_line).replace(',','_').replace('/','_')+'.wav'
            mixture,sr = audioread(mixture_path)
            if sr != self.sampling_rate:
                mixture = librosa.resample(mixture,orig_sr=sr, target_sr=self.sampling_rate) 
            
            #truncate 
            mixture=mixture[0:int(min_length*self.sampling_rate)]
            if len(mixture)<int(min_length*self.sampling_rate):
                mixture = np.pad(mixture,(0,int(min_length*self.sampling_rate)-len(mixture)))

            for c in range(self.C):
                # read target audio
                audio_path =self.mixture_direc+self.partition+'/s%d/'%(c+1)+ ','.join(path_line).replace(',','_').replace('/','_')+'.wav'

                audio,sr = audioread(audio_path)
                audio = audio[0:int(min_length*self.sampling_rate)]
                if sr != self.sampling_rate:
                    audio = librosa.resample(audio,orig_sr=sr, target_sr=self.sampling_rate) 
                if len(audio)<int(min_length*self.sampling_rate):
                    audio = np.pad(audio,(0,int(min_length*self.sampling_rate)-len(audio)))
                audios.append(audio)


                # read video 
                mask_start = int(line.split(',')[c*3+10])
                mask_length = int(line.split(',')[c*3+11])
                mask_type = int(line.split(',')[c*3 +12]) #0:full_mask 1: occluded 2: low resolution 
                visual_path=self.visual_direc + line.split(',')[1+c*4]+'/'+line.split(',')[2+c*4]+'/'+line.split(',')[3+c*4]+'.mp4'
                captureObj = cv.VideoCapture(visual_path)
                roiSequence = []
                roiSize = 112
                start=0

                ratio= mask_length/(min_length*25)
                while (captureObj.isOpened()):
                    ret, frame = captureObj.read()
                    if ret == True:
                        grayed = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                        grayed = grayed/255
                        grayed = cv.resize(grayed, (roiSize*2,roiSize*2))
                        roi = grayed[int(roiSize-(roiSize/2)):int(roiSize+(roiSize/2)), 
                                int(roiSize-(roiSize/2)):int(roiSize+(roiSize/2))]
                        if start>=mask_start and start< mask_start+mask_length:
                            if mask_type==0:
                                roi = np.zeros_like(roi)
                            elif mask_type==1:
                                noise_occlude = np.random.normal(0.5,0.5,(roiSize//2,roiSize//2))
                                noise_occlude = np.clip(noise_occlude,0,1)
                                roi[roiSize//8*3:roiSize//8*3+roiSize//2,roiSize//4:roiSize//4+roiSize//2] = 0
                                roi[roiSize//8*3:roiSize//8*3+roiSize//2,roiSize//4:roiSize//4+roiSize//2] += noise_occlude
                            elif mask_type==2:
                                if self.partition=='train':
                                    times = int(np.random.uniform(5,15))
                                else:
                                    times = 10
                                roi =  cv.resize(roi, (roiSize //times, roiSize //times))
                                roi = cv.resize(roi, (roiSize , roiSize))
                            else:
                                sys.exit('error: ',mask_type)
                        roiSequence.append(roi)
                        start += 1
                    else:
                        break
                captureObj.release()
                visual = np.asarray(roiSequence)
                visual = visual[0:int(min_length*self.fps),...]
                visual = (visual -self.normMean)/self.normStd
                if visual.shape[0]<int(min_length*self.fps):
                    # print(visual.shape,int(min_length*self.fps),visual_path)
                    visual = np.pad(visual, ((0, int(min_length*self.fps) - visual.shape[0]), (0,0), (0,0)), mode = 'edge')
                visuals.append(visual)

                #read speaker label 
                speakers.append(self.speaker_id[line.split(',')[c*4+2]])

            mixtures.append(mixture)
            mixtures.append(mixture)

        np_mixtures = np.asarray(mixtures)
        np_audios = np.asarray(audios)
        np_visuals = np.asarray(visuals)
        np_speakers = np.asarray(speakers)

        return np_mixtures,np_audios,np_visuals,np_speakers

    def __len__(self):
        if self.partition=='train':
            return len(self.minibatch)
        else:
            return len(self.minibatch)


class DistributedSampler(data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            # indices = torch.randperm(len(self.dataset), generator=g).tolist()
            ind = torch.randperm(int(len(self.dataset)/self.num_replicas), generator=g)*self.num_replicas
            indices = []
            for i in range(self.num_replicas):
                indices = indices + (ind+i).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        # indices = indices[self.rank:self.total_size:self.num_replicas]
        indices = indices[self.rank*self.num_samples:(self.rank+1)*self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

def get_dataloader(args, partition):
    datasets = dataset(
                speaker_dict =args.speaker_dict,
                mix_lst_path=args.mix_lst_path,
                visual_direc=args.visual_direc,
                mixture_direc=args.mixture_direc,
                batch_size=args.batch_size,
                partition=partition,
                mix_no=args.C,)

    sampler = DistributedSampler(
        datasets,
        num_replicas=args.world_size,
        rank=args.local_rank) if args.distributed else None

    generator = data.DataLoader(datasets,
            batch_size = 1,
            shuffle = (sampler is None),
            num_workers = args.num_workers,
            sampler=sampler,pin_memory=True)

    return sampler, generator