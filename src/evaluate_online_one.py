import argparse
import torch
import os
from MuSE_online.model import muse 
from pystoi import stoi
from pesq import pesq
import sys 
import numpy as np 
import soundfile as sf 
import torch.utils.data as data
import librosa
import cv2 as cv
import tqdm
from tools import audiowrite, audioread, segmental_snr_mixer,cal_SISNR
import fast_bss_eval
import pdb 

EPS = np.finfo(float).eps
MAX_INT16 = np.iinfo(np.int16).max


def main(args):
    # Model
    
    model = muse(args.N, args.L, args.B, args.H, args.P, args.X, args.R,
                        args.C, 800,causal=False)

    model = model.cuda()
    pretrained_model = torch.load('%s/model_dict_best.pt' % args.continue_from, map_location='cpu')['model']
    state = model.state_dict()
    for key in state.keys():
        pretrain_key = key
        if pretrain_key in pretrained_model.keys():
            state[key] = pretrained_model[pretrain_key]
        elif 'module.'+pretrain_key in pretrained_model.keys():
            state[key] = pretrained_model['module.'+pretrain_key]
        else:
            print(key +' is not loaded!!') 
    model.load_state_dict(state)
    print('load from ', args.continue_from)
    video_path = args.video_path1    
    audio_path = args.audio_path1
    inf_video_path = args.video_path2
    inf_audio_path = args.audio_path2

    tgt_audio,sr = audioread(audio_path)
    intervention_audio,sr = audioread(inf_audio_path)

    tgt_audio, intervention_audio, mixture, _ = segmental_snr_mixer(tgt_audio,intervention_audio,0,min_option=True,target_level_lower=-35,target_level_upper=-5)
    if sr != args.sampling_rate:
        mixture = librosa.resample(mixture,orig_sr=sr, target_sr=args.sampling_rate) 
    
    captureObj = cv.VideoCapture(video_path)
    roiSequence = []
    roiSize = 112
    id_=0
    while (captureObj.isOpened()):
        ret, frame = captureObj.read()
        if ret == True:
            grayed = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            grayed = grayed/255
            grayed = cv.resize(grayed, (roiSize*2,roiSize*2))
            roi = grayed[int(roiSize-(roiSize/2)):int(roiSize+(roiSize/2)), int(roiSize-(roiSize/2)):int(roiSize+(roiSize/2))]
            frame[int(roiSize-(roiSize/2)):int(roiSize-(roiSize/2))+5, int(roiSize-(roiSize/2)):int(roiSize+(roiSize/2)),2]=0
            frame[int(roiSize+(roiSize/2))-5:int(roiSize+(roiSize/2)), int(roiSize-(roiSize/2)):int(roiSize+(roiSize/2)),2]=0
            frame[int(roiSize-(roiSize/2)):int(roiSize+(roiSize/2)), int(roiSize-(roiSize/2)): int(roiSize-(roiSize/2))+5,2]=0
            frame[int(roiSize-(roiSize/2)):int(roiSize+(roiSize/2)), int(roiSize+(roiSize/2))-5:int(roiSize+(roiSize/2)),2]=0
            if id_ >=75:
                roi = np.zeros_like(roi)
                frame=np.zeros_like(frame)
            roiSequence.append(roi)
            if not os.path.exists(str(args.save_dir)+'/frames/'):
                os.makedirs(str(args.save_dir)+'/frames/')
            cv.imwrite(str(args.save_dir)+'/frames/'+'image_'+str(id_)+'.png',frame)
            id_ +=1
        else:
            break
    captureObj.release()
    visual = np.asarray(roiSequence)

    K = args.sampling_rate//25
    length  = min(visual.shape[0],len(mixture)//K)
    mixture = mixture[0:length*K]
    tgt_audio = tgt_audio[0:length*K]
    visual = visual[0:length,...]

    model.eval()
    with torch.no_grad():

        #audio
        a_chunk_samples = int(args.chunk_time* args.sampling_rate)
        a_receptive_samples = int(args.recep_field* args.sampling_rate)
        #video 
        v_chunk_samples = int(args.chunk_time* 25)
        v_receptive_samples = int(args.recep_field* 25)
        #init 
        initilization_samples = min(int(args.initilization*args.sampling_rate),a_receptive_samples)

        a_mix = torch.from_numpy(mixture).cuda().float().unsqueeze(0)
        v_tgt = torch.from_numpy(visual).cuda().float().unsqueeze(0)
        a_tgt = torch.from_numpy(tgt_audio).cuda().float().unsqueeze(0) 
        estimate_source = torch.zeros_like(a_mix).cuda()

        j_start = max(0,initilization_samples//a_chunk_samples)
        spk_emb=[]

        for j in range(j_start, a_mix.shape[1]//a_chunk_samples):
            #audio 
            a_start = max(0,j*a_chunk_samples - a_receptive_samples)
            if a_start+a_receptive_samples>a_mix.shape[1]:
                a_start = a_mix.shape[1]-a_receptive_samples
            if a_chunk_samples*j > a_mix.shape[1]:
                a_duration_seg = a_chunk_samples - (a_chunk_samples*j - a_mix.shape[1])
            else:
                a_duration_seg = a_chunk_samples

            a_end = a_chunk_samples*j + a_duration_seg
            a_mix_seg = a_mix[:,a_start:a_end]

            #video
            v_start = max(0,j*v_chunk_samples - v_receptive_samples)
            if v_chunk_samples*j > v_tgt.shape[1]:
                v_duration_seg = v_chunk_samples - (v_chunk_samples*j - v_tgt.shape[1])
            else:
                v_duration_seg = v_chunk_samples
            v_end = v_chunk_samples*j + v_duration_seg

            v_seg = v_tgt[:,v_start:v_end,:,:] 

            _, est_a_tgt_seg,spk_emb,_ = model(a_mix_seg,v_seg,spk_emb) #MoMuSE
            
            # energy normilization 
            a_cache = estimate_source[:,a_start:a_end-a_duration_seg]
            a_new = est_a_tgt_seg[:,:-a_duration_seg]
            if j!=j_start:
                a_cache_power = torch.linalg.norm(a_cache, 2)**2 / a_cache.shape[1]
                a_new_power = torch.linalg.norm(a_new, 2)**2 / a_new.shape[1]
                est_a_tgt_seg = est_a_tgt_seg * torch.sqrt(a_cache_power/a_new_power)
                estimate_source[:,a_end-a_duration_seg:a_end] = est_a_tgt_seg[:,-a_duration_seg:]
            else:
                #init for normlization
                est_a_tgt_seg = est_a_tgt_seg /(torch.max(abs(est_a_tgt_seg))) *0.5
                estimate_source[:,a_start:a_end] = est_a_tgt_seg
        # pdb.set_trace()
        sisnr = cal_SISNR(a_tgt, estimate_source).item()
        sisnr_mix = cal_SISNR(a_mix, estimate_source).item()
        estimate_source = estimate_source.squeeze().cpu().numpy()
        a_mix = a_mix.squeeze().cpu().numpy()

        audiowrite(str(args.save_dir)+'/'+'est_source.wav',estimate_source)
        audiowrite(str(args.save_dir)+'/'+'mix.wav',a_mix)
        print('sisnri:',sisnr-sisnr_mix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("avConv-tasnet")

# /mntcephfs/lee_dataset/separation/voxceleb2/wav/test/id04570/sESEzSlLBlY/00351.wav
# /mntcephfs/lee_dataset/separation/voxceleb2/wav/test/id04570/0YMGn6BI9rg/00003.wav
# /mntcephfs/lee_dataset/separation/voxceleb2/wav/test/id04570/zsnG6eKzOGE/00411.wav
# /mntcephfs/lee_dataset/separation/voxceleb2/wav/test/id04570/MEvAAvUiXdY/00133.wav
# /mntcephfs/lee_dataset/separation/voxceleb2/wav/test/id04570/pU1bFXTpgPM/00332.wav
# /mntcephfs/lee_dataset/separation/voxceleb2/wav/test/id00061/jT6eew_nWz4/00183.wav
# /mntcephfs/lee_dataset/separation/voxceleb2/wav/test/id00061/jT6eew_nWz4/00203.wav
# /mntcephfs/lee_dataset/separation/voxceleb2/wav/test/id00061/cAT9aR8oFx0/00141.wav
# /mntcephfs/lee_dataset/separation/voxceleb2/wav/test/id00061/bdkqfVtDZVY/00136.wav
# /mntcephfs/lee_dataset/separation/voxceleb2/wav/test/id00061/bdkqfVtDZVY/00135.wav
# /mntcephfs/lee_dataset/separation/voxceleb2/wav/test/id00061/ljIkW4uVVQY/00230.wav
# /mntcephfs/lee_dataset/separation/voxceleb2/wav/test/id00061/ljIkW4uVVQY/00233.wav
# /mntcephfs/lee_dataset/separation/voxceleb2/wav/test/id05055/2onVoeSgouI/00029.wav
# /mntcephfs/lee_dataset/separation/voxceleb2/wav/test/id05055/2onVoeSgouI/00028.wav
# /mntcephfs/lee_dataset/separation/voxceleb2/wav/test/id04094/nZqKra_Vo2g/00381.wav
# /mntcephfs/lee_dataset/separation/voxceleb2/wav/test/id04094/wvHI__c6n2Y/00491.wav
# /mntcephfs/lee_dataset/separation/voxceleb2/wav/test/id04094/SeYATZknAgI/00185.wav
# /mntcephfs/lee_dataset/separation/voxceleb2/wav/test/id04094/WxHoMvYuoyg/00242.wav
# /mntcephfs/lee_dataset/separation/voxceleb2/wav/test/id04094/WxHoMvYuoyg/00249.wav
# /mntcephfs/lee_dataset/separation/voxceleb2/wav/test/id04094/C3YXUnkp9RU/00065.wav
# /mntcephfs/lee_dataset/separation/voxceleb2/wav/test/id04094/C3YXUnkp9RU/00068.wav
# /mntcephfs/lee_dataset/separation/voxceleb2/wav/test/id04232/tCiPy0q5588/00477.wav
# /mntcephfs/lee_dataset/separation/voxceleb2/wav/test/id07620/WATd8hqnjZE/00245.wav
    
    parser.add_argument('--video_path1', type=str, default='/mntcephfs/lee_dataset/separation/voxceleb2/mp4/test/id07620/WATd8hqnjZE/00245.mp4', help='path of vidoe data')
    parser.add_argument('--audio_path1', type=str, default='/mntcephfs/lee_dataset/separation/voxceleb2/wav/test/id07620/WATd8hqnjZE/00245.wav', help='path of vidoe data')
    parser.add_argument('--video_path2', type=str, default='/mntcephfs/lee_dataset/separation/voxceleb2/mp4/test/id04570/pU1bFXTpgPM/00332.mp4', help='path of vidoe data')
    parser.add_argument('--audio_path2', type=str, default='/mntcephfs/lee_dataset/separation/voxceleb2/wav/test/id04570/pU1bFXTpgPM/00332.wav', help='path of vidoe data')

    parser.add_argument('--continue_from', type=str, default='./logs/Online_MuSE_mask_pre_0.05penalty_finetune2024-02-21(15:57:40)/')
    

    parser.add_argument('--save_dir', default='./save_audio/', type=str,
                        help='audio_save_path')

    # Model hyperparameters
    parser.add_argument('--L', default=40, type=int,
                        help='Length of the filters in samples (80=5ms at 16kHZ)')
    parser.add_argument('--N', default=256, type=int,
                        help='Number of filters in autoencoder')
    parser.add_argument('--B', default=256, type=int,
                        help='Number of channels in bottleneck 1 × 1-conv block')
    parser.add_argument('--C', type=int, default=2,
                        help='number of speakers to mix')
    parser.add_argument('--H', default=512, type=int,
                        help='Number of channels in convolutional blocks')
    parser.add_argument('--P', default=3, type=int,
                        help='Kernel size in convolutional blocks')
    parser.add_argument('--X', default=8, type=int,
                        help='Number of convolutional blocks in each repeat')
    parser.add_argument('--R', default=4, type=int,
                        help='Number of repeats')

    #Online evaluation setting 
    parser.add_argument('--chunk_time', type=float, default=0.2, help='time length(s) of chunk_time')
    parser.add_argument('--recep_field', type=float, default=2.5, help='time length(s) of receptive filed')
    parser.add_argument('--sampling_rate', type=int, default=16000, help='sampling_rate of audio')
    parser.add_argument('--initilization', type=int, default=1.0, help='time length(s) of initilization')

    args = parser.parse_args()

    main(args)