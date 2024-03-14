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
import matplotlib.pyplot as plt

import pdb 

# visual_path = '/mntcephfs/lee_dataset/separation/voxceleb2/mp4/train/id00015/0fijmz4vTVU/00002.mp4'

# captureObj = cv.VideoCapture(visual_path)
# roiSequence = []
# roiSize = 112
# start=0
# mask_type=2
# mask_start=10
# mask_length=10
# while (captureObj.isOpened()):
#     ret, frame = captureObj.read()
#     if ret == True:
#         grayed = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#         grayed = grayed/255
#         grayed = cv.resize(grayed, (roiSize*2,roiSize*2))
#         roi = grayed[int(roiSize-(roiSize/2)):int(roiSize+(roiSize/2)), 
#                 int(roiSize-(roiSize/2)):int(roiSize+(roiSize/2))]
#         if start>=mask_start and start< mask_start+mask_length:
#             if mask_type==0:
#                 roi = np.zeros_like(roi)
#             elif mask_type==1:
#                 noise_occlude = np.random.normal(0.5,0.5,(roiSize//2,roiSize//2))
#                 noise_occlude = np.clip(noise_occlude,0,1)
#                 roi[roiSize//8*3:roiSize//8*3+roiSize//2,roiSize//4:roiSize//4+roiSize//2] = 0
#                 roi[roiSize//8*3:roiSize//8*3+roiSize//2,roiSize//4:roiSize//4+roiSize//2] += noise_occlude
#             elif mask_type==2:
                
#                 times = 10
#                 roi =  cv.resize(roi, (roiSize //times, roiSize //times))
#                 roi = cv.resize(roi, (roiSize , roiSize))
#             else:
#                 sys.exit('error: ',mask_type)
#         cv.imwrite('./lip.jpg',roi*255)
#         pdb.set_trace()
#         roiSequence.append(roi)
#         start += 1
#     else:
#         break
# captureObj.release()



# x=['[0,10)','[10,20)','[20,30)','[30,40)','[40,50)','[50,60)','[60,70)','[70,80)','[80,90)','[90,100)']
# MuSE_sisnr =                      [8.91,8.31,7.77,6.73,6.69,6.11,5.55,4.03,3.52,2.88]
# MoMuSE_wo_init_wo_penalty_sisnr = [7.81,6.74,7.01,6.86,7.59,5.96,6.32,5.00,5.30,3.69]
# MoMuSE_wo_init_sisnr            = [8.32,7.64,8.01,7.06,7.86,6.54,6.95,5.16,5.75,2.10]
# MoMuSE_sisnr                    = [8.59,7.83,8.63,7.74,7.90,6.79,7.25,5.36,5.92,2.68]
# plt.plot(x,MuSE_sisnr,'darkorange',marker='s',linestyle='-',label="MuSE(baseline)")
# plt.plot(x,MoMuSE_wo_init_wo_penalty_sisnr,'darkgreen',marker='s',linestyle='-',label="MoMuSE_w/o(PI+Pe)")
# plt.plot(x,MoMuSE_wo_init_sisnr,'cornflowerblue',marker='s',linestyle='-',label="MoMuSE_w/o(Pe)")
# plt.plot(x,MoMuSE_sisnr,'brown',marker='s',linestyle='-',label="MoMuSE")
# plt.xlabel("The ratio of impaired video frames (%)")#横坐标名字
# plt.ylabel("SI-SNR (dB)")#横坐标名字
# plt.legend(loc = "lower left",prop = {'size':14})#图例
# plt.show()
# plt.savefig('./SISNR.png')
# plt.cla()
# MuSE_sdr =                      [9.41,8.85,8.27,7.34,7.27,6.73,6.15,4.73,4.21,3.68]
# MoMuSE_wo_init_wo_penalty_sdr = [8.42,7.49,7.63,7.57,8.15,6.70,7.14,5.79,6.12,4.62]
# MoMuSE_wo_init_sdr            = [8.89,8.33,8.63,7.78,8.45,7.30,7.77,5.99,6.72,3.25]
# MoMuSE_sdr                    = [9.13,8.47,9.15,8.37,8.49,7.46,7.97,6.16,6.79,3.86]
# plt.plot(x,MuSE_sdr,'darkorange',marker='s',linestyle='-',label="MuSE(baseline)")
# plt.plot(x,MoMuSE_wo_init_wo_penalty_sdr,'darkgreen',marker='s',linestyle='-',label="MoMuSE_w/o(PI+Pe)")
# plt.plot(x,MoMuSE_wo_init_sdr,'cornflowerblue',marker='s',linestyle='-',label="MoMuSE_w/o(Pe)")
# plt.plot(x,MoMuSE_sdr,'brown',marker='s',linestyle='-',label="MoMuSE")
# plt.xlabel("The ratio of impaired video frames (%)")#横坐标名字
# plt.ylabel("SDR (dB)")#横坐标名字
# plt.legend( loc = "lower left",prop = {'size':14})#图例
# plt.show()
# plt.savefig('./SDR.png')
# plt.cla()
# MuSE_pesq =                      [1.81,1.77,1.77,1.74,1.72,1.64,1.63,1.54,1.57,1.53]
# MoMuSE_wo_init_wo_penalty_pesq = [1.79,1.76,1.78,1.82,1.82,1.71,1.79,1.70,1.78,1.67]
# MoMuSE_wo_init_pesq            = [1.81,1.80,1.83,1.85,1.88,1.77,1.82,1.73,1.86,1.64]
# MoMuSE_pesq                    = [1.83,1.80,1.86,1.87,1.88,1.77,1.85,1.75,1.85,1.67]
# plt.plot(x,MuSE_pesq,'darkorange',marker='s',linestyle='-',label="MuSE(baseline)")
# plt.plot(x,MoMuSE_wo_init_wo_penalty_pesq,'darkgreen',marker='s',linestyle='-',label="MoMuSE_w/o(PI+Pe)")
# plt.plot(x,MoMuSE_wo_init_pesq,'cornflowerblue',marker='s',linestyle='-',label="MoMuSE_w/o(Pe)")
# plt.plot(x,MoMuSE_pesq,'brown',marker='s',linestyle='-',label="MoMuSE")
# plt.xlabel("The ratio of impaired video frames (%)")#横坐标名字
# plt.ylabel("PESQ")#横坐标名字
# plt.legend(loc = "lower left",prop = {'size':14})#图例
# plt.show()
# plt.savefig('./PESQ.png')
# plt.cla()
# MuSE_stoi =                      [0.83,0.82,0.81,0.78,0.79,0.77,0.75,0.72,0.70,0.68]
# MoMuSE_wo_init_wo_penalty_stoi = [0.81,0.78,0.78,0.77,0.80,0.76,0.76,0.73,0.74,0.70]
# MoMuSE_wo_init_stoi            = [0.81,0.80,0.81,0.77,0.81,0.77,0.77,0.73,0.76,0.65]
# MoMuSE_stoi                    = [0.82,0.80,0.82,0.79,0.81,0.78,0.78,0.74,0.76,0.67]
# plt.plot(x,MuSE_stoi,'darkorange',marker='s',linestyle='-',label="MuSE(baseline)")
# plt.plot(x,MoMuSE_wo_init_wo_penalty_stoi,'darkgreen',marker='s',linestyle='-',label="MoMuSE_w/o(PI+Pe)")
# plt.plot(x,MoMuSE_wo_init_stoi,'cornflowerblue',marker='s',linestyle='-',label="MoMuSE_w/o(Pe)")
# plt.plot(x,MoMuSE_stoi,'brown',marker='s',linestyle='-',label="MoMuSE")
# plt.xlabel("The ratio of impaired video frames (%)")#横坐标名字
# plt.ylabel("STOI")#横坐标名字
# plt.legend(loc = "lower left",prop = {'size':14})#图例
# plt.show()
# plt.savefig('./STOI.png')
# plt.cla()





def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2.-0.15, 1.03*height, "%.2f" % float(height))
                        

tick_label = ['[0,1)','[1,2)','[2,3)','[3,4)','[4,∞]']
x=np.arange(5)#柱状图在横坐标上的位置
width=0.3
MoMuSE=[4.20,8.41 ,8.62 ,8.89 ,8.35]
MuSE = [1.76,6.06,7.56,8.29,8.04]
a= plt.bar(x, MoMuSE,color='brown',width=width,label='MoMuSE')
b= plt.bar(x+width, MuSE,color='darkorange',width=width,label='MuSE')
autolabel(a)
autolabel(b)
plt.xticks(x+width/2,tick_label)#显示x坐标轴的标签,即tick_label,调整位置，使其落在两个直方图中间位置
plt.ylabel("SI-SNR (dB)")#横坐标名字
plt.xlabel("Impaired visual onset (second)")#横坐标名字
plt.ylim(0,10)
plt.show()
plt.legend(loc = "upper left")#图例
plt.savefig('./bar.png')


