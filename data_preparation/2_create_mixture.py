import os
import numpy as np 
import argparse
import tqdm
import soundfile as sf 
import librosa
from tools import audiowrite, audioread, segmental_snr_mixer  
import pdb

def main(args):
	# create mixture
	mixture_data_list = open(args.mixture_data_list).read().splitlines()
	print(len(mixture_data_list))
	sampling_rate = args.sampling_rate

	for line in tqdm.tqdm(mixture_data_list,desc = "Generating audio mixtures"):
		data = line.split(',')
		save_direc=args.mixture_audio_direc+data[0]+'/'
		if not os.path.exists(save_direc):
			os.makedirs(save_direc)
		
		mixture_save_path=save_direc+'mix/'+line.replace(',','_').replace('/','_') +'.wav'
		s1_path=save_direc+'s1/'+line.replace(',','_').replace('/','_') +'.wav'
		s2_path=save_direc+'s2/'+line.replace(',','_').replace('/','_') +'.wav'
		if os.path.exists(mixture_save_path):
			continue

		# read target audio
		tgt_audio,sr =audioread(args.audio_data_direc+data[1]+'/'+data[2]+'/'+data[3]+'.wav')
		if sr!=sampling_rate:
			tgt_audio = librosa.resample(tgt_audio,orig_sr=sr, target_sr=sampling_rate)

		# read inteference audio
		for c in range(1, args.C):
			audio_path=args.audio_data_direc+data[c*4+1]+'/'+data[c*4+2]+'/'+data[c*4+3]+'.wav'
			intervention_audio,sr = audioread(audio_path)
			if sr!=sampling_rate:
				audio = librosa.resample(audio,orig_sr=sr, target_sr=sampling_rate)
			snr = float(data[c*4+4])

			# truncate long audio with short audio in the mixture
			if 'test' not in save_direc:
				tgt_audio, intervention_audio, mix_audio, _ = segmental_snr_mixer(tgt_audio,intervention_audio,snr,length=int(args.max_length* args.sampling_rate),target_level_lower=-35,target_level_upper=-5)
			else:
				tgt_audio, intervention_audio, mix_audio, _ = segmental_snr_mixer(tgt_audio,intervention_audio,snr,min_option=True,target_level_lower=-35,target_level_upper=-5)
		audiowrite(mixture_save_path,mix_audio,sample_rate=sampling_rate)
		audiowrite(s1_path,tgt_audio,sample_rate=sampling_rate)
		audiowrite(s2_path,intervention_audio,sample_rate=sampling_rate)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Voxceleb2 dataset')
	parser.add_argument('--C', type=int)
	parser.add_argument('--audio_data_direc', type=str)
	parser.add_argument('--mixture_audio_direc', type=str)
	parser.add_argument('--mixture_data_list', type=str)
	parser.add_argument('--sampling_rate', type=int)
	parser.add_argument('--max_length', type=int)
	args = parser.parse_args()
	main(args)