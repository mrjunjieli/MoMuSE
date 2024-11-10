import argparse
import torch
import os
from MuSE.model import muse 
# from MoMuSE.model import muse 
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
    video_path = args.video_path2
    audio_path = args.audio_path2
    inf_video_path = args.video_path1
    inf_audio_path = args.audio_path1

    tgt_audio,sr = audioread(audio_path)
    intervention_audio,sr = audioread(inf_audio_path)
    # mixture, sr = audioread('/mntcephfs/lee_dataset/separation/voxceleb2/mixture/test/mix/test_test_id04478_GZQGZOmFU5U_00078_0_test_id01298_3j7WnbVzR4c_00040_2.1806386311827133_14.08.wav')
    tgt_audio, intervention_audio, mixture, _ = segmental_snr_mixer(tgt_audio,intervention_audio,0,min_option=True,target_level_lower=-35,target_level_upper=-5)
    # audiowrite('tgt_audio.wav',tgt_audio)
    # audiowrite('mix.wav',mixture)
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
            if id_ >=25*1:
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
    visual = (visual -0.4161)/0.1688

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

            # _, est_a_tgt_seg,spk_emb,_ = model(a_mix_seg,v_seg,spk_emb) #MoMuSE
            _,est_a_tgt_seg = model(a_mix_seg,v_seg)  # MuSE
            
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

    # ('/mntcephfs/lee_dataset/separation/voxceleb2/mixture/test/mix/test_test_id08701_61Al05HARgA_00006_0_test_id01541_C29fUBtimOE_00047_6.937530228442423_14.976.wav',) 14.125800132751465
    # ('/mntcephfs/lee_dataset/separation/voxceleb2/mixture/test/mix/test_test_id07426_gol32BfUJh4_00155_0_test_id07354_DPDPVItsdg8_00138_5.981796823638694_11.456.wav',) 17.007740020751953
    # ('/mntcephfs/lee_dataset/separation/voxceleb2/mixture/test/mix/test_test_id01333_4iv48rM2Qk8_00036_0_test_id00419_w3eVBzB4AcI_00466_-0.5055948509456325_12.352.wav',) 11.522909164428711
    # ('/mntcephfs/lee_dataset/separation/voxceleb2/mixture/test/mix/test_test_id04478_GZQGZOmFU5U_00078_0_test_id01298_3j7WnbVzR4c_00040_2.1806386311827133_14.08.wav',) 12.45551872253418
    # ('/mntcephfs/lee_dataset/separation/voxceleb2/mixture/test/mix/test_test_id01618_p_pa0JGeT3g_00177_0_test_id00926_mpqgHcoq87w_00129_0.9086165229501635_17.344.wav',) 12.798543930053711
    # ('/mntcephfs/lee_dataset/separation/voxceleb2/mixture/test/mix/test_test_id04950_SAyGsI0hsMU_00159_0_test_id07414_NWqZelGdBPA_00190_6.535908094880135_11.968.wav',) 13.537038803100586
    # ('/mntcephfs/lee_dataset/separation/voxceleb2/mixture/test/mix/test_test_id01333_Qw8-jKhzwEg_00201_0_test_id06692_9vs0zAHfI0M_00149_5.724163515874711_12.096.wav',) 17.133071899414062
    # ('/mntcephfs/lee_dataset/separation/voxceleb2/mixture/test/mix/test_test_id03030_xwcuW7hrVgg_00331_0_test_id07620_axaQeZdVOgM_00350_-5.873834020974289_14.208.wav',) 10.029067993164062
    # ('/mntcephfs/lee_dataset/separation/voxceleb2/mixture/test/mix/test_test_id01298_wGQIqoQNXxA_00426_0_test_id04570_3s7mF3vF2QQ_00033_-0.5379086945830274_14.4.wav',) 13.196396827697754
    
    parser.add_argument('--video_path1', type=str, default='/mntcephfs/lee_dataset/separation/voxceleb2/mp4/test/id01333/X8REn1clroY/00246.mp4', help='path of vidoe data')
    parser.add_argument('--audio_path1', type=str, default='/mntcephfs/lee_dataset/separation/voxceleb2/wav/test/id01333/X8REn1clroY/00246.wav', help='path of vidoe data')
    # parser.add_argument('--audio_path1', type=str, default='/mntcephfs/lee_dataset/separation/voxceleb2/mixture/test/s1/test_test_id01618_p_pa0JGeT3g_00177_0_test_id00926_mpqgHcoq87w_00129_0.9086165229501635_17.344.wav', help='path of vidoe data')
    parser.add_argument('--video_path2', type=str, default='/mntcephfs/lee_dataset/separation/voxceleb2/mp4/test/id04253/yh7acTe-vnA/00326.mp4', help='path of vidoe data')
    parser.add_argument('--audio_path2', type=str, default='/mntcephfs/lee_dataset/separation/voxceleb2/wav/test/id04253/yh7acTe-vnA/00326.wav', help='path of vidoe data')
    # parser.add_argument('--audio_path2', type=str, default='/mntcephfs/lee_dataset/separation/voxceleb2/mixture/test/s2/test_test_id01618_p_pa0JGeT3g_00177_0_test_id00926_mpqgHcoq87w_00129_0.9086165229501635_17.344.wav', help='path of vidoe data')

    # parser.add_argument('--continue_from', type=str, default='./logs/Online_MuSE_mask_pre_0.05penalty_finetune2024-02-21(15:57:40)/')
    parser.add_argument('--continue_from', type=str, default='./logs/MuSE_mask2024-02-15(19:11:19)/')
    

    parser.add_argument('--save_dir', default='./save_audio_MUSE2/', type=str,
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