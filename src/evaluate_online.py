import argparse
import torch
import os
# from MuSE.model import muse
# from MuSE_causal.model import muse 
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
from tools import audioread, audiowrite, cal_SISNR
import fast_bss_eval
import pdb 

EPS = np.finfo(float).eps
MAX_INT16 = np.iinfo(np.int16).max


class dataset(data.Dataset):
    def __init__(self,
                mix_lst_path,
                visual_direc,
                mixture_direc,
                batch_size=1,
                partition='test',
                sampling_rate=16000,
                mix_no=2):

        self.minibatch =[]
        self.visual_direc = visual_direc
        self.mixture_direc = mixture_direc
        self.sampling_rate = sampling_rate
        self.partition = partition
        self.C=mix_no
        self.fps = 25
        self.normMean = 0.4161
        self.normStd = 0.1688

        mix_csv=open(mix_lst_path).read().splitlines()
        self.mix_lst=list(filter(lambda x: x.split(',')[0]==partition, mix_csv))

    def __getitem__(self, index):
        
        line = self.mix_lst[index]
        

        #read mix 
        path_line = line.split(',')[0:-6]
        mixture_path=self.mixture_direc+self.partition+'/mix/'+ ','.join(path_line).replace(',','_').replace('/','_')+'.wav'
        mixture,sr = audioread(mixture_path)
        if sr != self.sampling_rate:
            mixture = librosa.resample(mixture,orig_sr=sr, target_sr=self.sampling_rate) 
        
        min_length = mixture.shape[0]
        min_length_tim = min_length/self.sampling_rate

        #read tgt audio 
        c=0
        audio_path =self.mixture_direc+self.partition+'/s%d/'%(c+1)+ ','.join(path_line).replace(',','_').replace('/','_')+'.wav'
        audio,sr = audioread(audio_path)
        audio = audio[0:min_length]
        if sr != self.sampling_rate:
            audio = librosa.resample(audio,orig_sr=sr, target_sr=self.sampling_rate) 
        
        if len(audio)<int(min_length):
            audio = np.pad(audio,(0,int(min_length)-len(audio)))
        
        

       # read video 
        mask_start = int(line.split(',')[c*3+10])
        mask_length = int(line.split(',')[c*3+11])
        mask_type = int(line.split(',')[c*3 +12]) #0:full_mask 1: occluded 2: low resolution 
        # mask_start= 3*25
        # mask_length = np.inf
        # mask_type = 0 

        visual_path=self.visual_direc + line.split(',')[1+c*4]+'/'+line.split(',')[2+c*4]+'/'+line.split(',')[3+c*4]+'.mp4'
        captureObj = cv.VideoCapture(visual_path)
        roiSequence = []
        roiSize = 112
        start=0
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
        visual_length = len(roiSequence)
        visual = visual[0:int(min_length_tim*self.fps),...]
        visual = (visual -self.normMean)/self.normStd
        if visual.shape[0]<int(min_length_tim*self.fps):
            visual = np.pad(visual, ((0, int(min_length_tim*self.fps) - visual.shape[0]), (0,0), (0,0)), mode = 'edge')

        return mixture, audio, visual,round(mask_length/visual_length,2), mask_type,mask_start/25

    def __len__(self):
        return len(self.mix_lst)


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
    datasets = dataset(
                mix_lst_path=args.mix_lst_path,
                visual_direc=args.visual_direc,
                mixture_direc=args.mixture_direc,
                mix_no=args.C,
                sampling_rate = args.sampling_rate)

    test_generator = data.DataLoader(datasets,
            batch_size = 1,
            shuffle = False,
            num_workers = args.num_workers)


    model.eval()
    with torch.no_grad():
        mask_start_sisnr=[[],[],[],[],[]] #0-1s 1-2s,2-3s,3-4s,4-s
        avg_sisnr = [[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]]]
        avg_sdr = [[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]]]
        avg_pesq = [[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]]]
        avg_stoi = [[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]]]


        audio_length_sisnr=[[],[],[],[],[]]#4-8s,8-12s,12-16s,16-20s,20-infs

        #audio
        a_chunk_samples = int(args.chunk_time* args.sampling_rate)
        
         = int(args.recep_field* args.sampling_rate)
        #video 
        v_chunk_samples = int(args.chunk_time* 25)
        v_receptive_samples = int(args.recep_field* 25)
        #init 
        initilization_samples = min(int(args.initilization*args.sampling_rate),a_receptive_samples)

        for i, (a_mix, a_tgt, v_tgt,mask_ratio,mask_type,mask_start) in enumerate(tqdm.tqdm(test_generator)):
            if mask_ratio==1:
                mask_ratio -=0.01
            # mask_ratio=0

            a_mix = a_mix.cuda().squeeze().float().unsqueeze(0)
            a_tgt = a_tgt.cuda().squeeze().float().unsqueeze(0)
            v_tgt = v_tgt.cuda().squeeze().float().unsqueeze(0)
            
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
                a_tgt_seg = a_tgt[:,a_start:a_end]

                #video
                v_start = max(0,j*v_chunk_samples - v_receptive_samples)
                if v_chunk_samples*j > v_tgt.shape[1]:
                    v_duration_seg = v_chunk_samples - (v_chunk_samples*j - v_tgt.shape[1])
                else:
                    v_duration_seg = v_chunk_samples
                v_end = v_chunk_samples*j + v_duration_seg

                v_seg = v_tgt[:,v_start:v_end,:,:] 

                # _,est_a_tgt_seg = model(a_mix_seg,v_seg)  # MuSE
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
            
            sisnr = cal_SISNR(a_tgt, estimate_source).item()
            avg_sisnr[int(mask_ratio*10)][int(mask_type)].append(sisnr)
            if mask_start>=4:
                mask_start=4
            mask_start_sisnr[int(mask_start)].append(sisnr)

            if v_tgt.shape[1]//25>=4 and v_tgt.shape[1]//25<8:
                audio_length_sisnr[0].append(sisnr)
            elif v_tgt.shape[1]//25>=8 and v_tgt.shape[1]//25<12:
                audio_length_sisnr[1].append(sisnr)
            elif v_tgt.shape[1]//25>=12 and v_tgt.shape[1]//25<16:
                audio_length_sisnr[2].append(sisnr)
            elif v_tgt.shape[1]//25>=16 and v_tgt.shape[1]//25<20:
                audio_length_sisnr[3].append(sisnr)
            else:
                audio_length_sisnr[4].append(sisnr)

            estimate_source = estimate_source.squeeze().cpu().numpy()
            a_tgt = a_tgt.squeeze().cpu().numpy()
            a_mix = a_mix.squeeze().cpu().numpy()
            
            sdr = fast_bss_eval.sdr(np.expand_dims(a_tgt,0),np.expand_dims(estimate_source,0))[0]
            pesq_ = pesq(16000, a_tgt, estimate_source, 'wb')
            stoi_ = stoi(a_tgt, estimate_source, 16000, extended=False)

            avg_sdr[int(mask_ratio*10)][int(mask_type)].append(sdr)
            avg_pesq[int(mask_ratio*10)][int(mask_type)].append(pesq_)
            avg_stoi[int(mask_ratio*10)][int(mask_type)].append(stoi_)

            if args.save:
                if not os.path.exists(args.save_dir):
                    os.makedirs(args.save_dir)
                audiowrite(str(args.save_dir)+'/'+'s_%d_mix.wav'%i,a_mix)
                audiowrite(str(args.save_dir)+'/'+'s_%d_tgt.wav'%i,a_tgt)
                audiowrite(str(args.save_dir)+'/'+'s_%d_est_%.2f.wav'%(i,sisnr),estimate_source)
        
        
        print('--------------')
        mask_type_dict = {0:'full_mask     ',1:'lip_occlude   ',2:'low_resolution'}
        mask_type_sisnr = {0:0,1:0,2:0}
        mask_type_sdr = {0:0,1:0,2:0}
        mask_type_pesq = {0:0,1:0,2:0}
        mask_type_stoi = {0:0,1:0,2:0}
        for j in range(10):
            sisnr_ = 0
            sdr_ = 0
            pesq_ = 0
            stoi_ = 0
            print('Mask_ratio:['+str(j*10)+'%,'+str((j+1)*10)+'%)')
            for p in range(3):
                tmp_sisnr = np.mean(avg_sisnr[j][p])
                tmp_sdr = np.mean(avg_sdr[j][p])
                tmp_pesq = np.mean(avg_pesq[j][p])
                tmp_stoi = np.mean(avg_stoi[j][p])
                print('     Mask_type:',mask_type_dict[p],'SI-SNR:',round(tmp_sisnr,2)\
                ,',SDR:',round(tmp_sdr,2),',PESQ:',round(tmp_pesq,2)\
                ,',STOI:',round(tmp_stoi,2))
                sisnr_+=tmp_sisnr
                sdr_+=tmp_sdr
                pesq_+=tmp_pesq
                stoi_+=tmp_stoi
                mask_type_sisnr[p] +=tmp_sisnr
                mask_type_sdr[p] +=tmp_sdr
                mask_type_pesq[p] +=tmp_pesq
                mask_type_stoi[p] +=tmp_stoi

            print('  AVG SI-SNR:',round(sisnr_/3,2)\
                ,',SDR:',round(sdr_/3,2),',PESQ:',round(pesq_/3,2),',STOI:',round(stoi_/3,2))
            
        print('--------------')
        for p in range(3):
            print('AVG Mask_type:',mask_type_dict[p],'SI-SNR:',round(mask_type_sisnr[p]/10,2)\
                ,',SDR:',round(mask_type_sdr[p]/10,2),',PESQ:',round(mask_type_pesq[p]/10,2)\
                ,',STOI:',round(mask_type_stoi[p]/10,2))
        print('-------------')
        for i in range(5):
            if i!=4:
                print('Mask_start_time:['+str(i)+','+str((i+1))+')',round(np.mean(mask_start_sisnr[i]),2))
            else:
                print('Mask_start_time:['+str(i)+','+'∞)',round(np.mean(mask_start_sisnr[i]),2))

        for i in range(5):
            if i!=4:
                print('Audio_length:['+str(i*4+4)+','+str(((i+1)*4+4))+')',round(np.mean(audio_length_sisnr[i]),2))
            else:
                print('Audio_length:['+str(i*4+4)+','+'∞)',round(np.mean(audio_length_sisnr[i]),2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser("avConv-tasnet")
    

    parser.add_argument('--mix_lst_path', type=str, default='../data_preparation/mixture_data_list_2mix_with_occludded.csv',
                        help='directory including train data')
    parser.add_argument('--visual_direc', type=str, default='/mntcephfs/lee_dataset/separation/voxceleb2/mp4/',
                        help='directory including test data')
    parser.add_argument('--mixture_direc', type=str, default='/mntcephfs/lee_dataset/separation/voxceleb2/mixture/',
                        help='directory of audio')
    # parser.add_argument('--continue_from', type=str, default='./logs/MuSE_mask2024-02-15(19:11:19)/')
    # 
    parser.add_argument('--continue_from', type=str, default='./logs/Online_MuSE_mask_pre_0.05penalty_finetune2024-02-21(15:57:40)/')
    
    
    parser.add_argument('--save', default=0, type=int,
                        help='whether to save audio')
    parser.add_argument('--save_dir', default='./save_audio/', type=str,
                        help='audio_save_path')

    # Training    
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers to generate minibatch')

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