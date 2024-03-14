import torch
import os
from tqdm import tqdm
import numpy as np
import argparse
import random
import csv
import cv2 as cv 
import pdb 
random.seed(0)


def main(args):
    f=open(args.mixture_data_list_with_occludded,'w')
    w=csv.writer(f)
    low_quality_type = [0,1,2] #0:full mask 1: occluded 2: low resolution 

    mix_lst=open(args.mix_data_list).read().splitlines()
    for line in tqdm(mix_lst):
        
        line=line.split(',')
        min_length = float(line[-1])
        video_path_1 =args.video_data_direc + line[1]+'/'+line[2]+'/'+line[3]+'.mp4'
        captureObj = cv.VideoCapture(video_path_1)
        roiSequence_1 = []
        while (captureObj.isOpened()):
            ret, frame = captureObj.read()
            if ret == True:
                roiSequence_1.append(frame)
            else:
                break
        captureObj.release()

        video_path_2 =args.video_data_direc + line[5]+'/'+line[6]+'/'+line[7]+'.mp4'
        captureObj = cv.VideoCapture(video_path_2)
        roiSequence_2 = []
        while (captureObj.isOpened()):
            ret, frame = captureObj.read()
            if ret == True:
                roiSequence_2.append(frame)
            else:
                break
        captureObj.release()

        temp = min(len(roiSequence_2),len(roiSequence_1))
        min_len = min(temp,int(min_length*25))
        
        if 'test' in line:
            mask_ratio = round(random.uniform(0,1),2)
        else:
            mask_ratio = round(random.uniform(0,0.8),2)
        mask_len_frames = int(mask_ratio*min_len)
        mask_start_frame = random.randint(0,int(min_len-mask_len_frames))
        type_1 = random.choice(low_quality_type)
        line.append(mask_start_frame)
        line.append(mask_len_frames)
        line.append(type_1)

        if 'test' in line:
            mask_ratio = round(random.uniform(0,1),2)
        else:
            mask_ratio = round(random.uniform(0,0.8),2)
        mask_len_frames = int(mask_ratio*min_len)
        mask_start_frame = random.randint(0,int(min_len-mask_len_frames))
        type_1 = random.choice(low_quality_type)
        line.append(mask_start_frame)
        line.append(mask_len_frames)
        line.append(type_1)

        w.writerow(line)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Voxceleb2 dataset')
    parser.add_argument('--mix_data_list', default = './mixture_data_list_2mix.csv', type=str)
    parser.add_argument('--video_data_direc', default='/mntcephfs/lee_dataset/separation/voxceleb2/mp4/', type=str)
    parser.add_argument('--mixture_data_list_with_occludded', default='./mixture_data_list_2mix_with_occludded.csv', type=str)
    args = parser.parse_args()
    main(args)
