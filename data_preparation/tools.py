import os
import numpy as np 
import argparse
import soundfile as sf 
import librosa

EPS = np.finfo(float).eps
np.random.seed(0)

def audiowrite(destpath, audio, sample_rate=16000, norm=False, target_level=-25, \
                clipping_threshold=0.99, clip_test=False):
    '''Function to write audio'''

    if clip_test:
        if is_clipped(audio, clipping_threshold=clipping_threshold):
            raise ValueError("Clipping detected in audiowrite()! " + \
                            destpath + " file not written to disk.")

    if norm:
        audio = normalize(audio, target_level)
        max_amp = max(abs(audio))
        if max_amp >= clipping_threshold:
            audio = audio/max_amp * (clipping_threshold-EPS)

    destpath = os.path.abspath(destpath)
    destdir = os.path.dirname(destpath)

    if not os.path.exists(destdir):
        os.makedirs(destdir)

    sf.write(destpath, audio, sample_rate)
    return

def is_clipped(audio, clipping_threshold=0.99):
    return any(abs(audio) > clipping_threshold)

def normalize(audio, target_level=-25):
    '''Normalize the signal to the target level'''
    rms = (audio ** 2).mean() ** 0.5
    scalar = 10 ** (target_level / 20) / (rms+EPS)
    audio = audio * scalar
    return audio

def normalize_segmental_rms(audio, rms, target_level=-25):
    '''Normalize the signal to the target level
    based on segmental RMS'''
    scalar = 10 ** (target_level / 20) / (rms+EPS)
    audio = audio * scalar
    return audio

def audioread(path, norm=False, start=0, stop=None, target_level=-25):
    '''Function to read audio'''
    '''taget_level dBFs'''

    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise ValueError("[{}] does not exist!".format(path))
    try:
        audio, sample_rate = sf.read(path, start=start, stop=stop)
    except RuntimeError:  # fix for sph pcm-embedded shortened v2
        print('WARNING: Audio type not supported')

    if len(audio.shape) == 1:  # mono
        if norm:
            rms = (audio ** 2).mean() ** 0.5
            scalar = 10 ** (target_level / 20) / (rms+EPS)
            audio = audio * scalar
    else:  # multi-channel
        audio = audio.T
        audio = audio.sum(axis=0)/audio.shape[0]
        if norm:
            audio = normalize(audio, target_level)
    return audio, sample_rate



def segmental_snr_mixer(clean, noise, snr,min_option=True,length=None, target_level_lower=-35,target_level_upper=-5,target_level=-25, clipping_threshold=0.99):
    '''Function to mix clean speech and noise at various segmental SNR levels'''
    if min_option and length==None:
        length = min(len(clean),len(noise))
        if len(clean)>length:
            clean = clean[0:length]
        if len(noise)>length:
            noise = noise[0:length]
    if length !=None:
        if len(clean) > length:
            clean = clean[0:length]
        if len(noise) > length:
            noise = noise[0:length]
        if len(noise) < length:
            noise = np.append(noise, np.zeros(length-len(noise)))
        if len(clean) < length:
            clean = np.append(clean, np.zeros(length-len(clean)))
    if min_option==False and length==None:
        if len(clean) > len(noise):
            noise = np.append(noise, np.zeros(len(clean)-len(noise)))
        else:
            noise = noise[0:len(clean)]
        
    clean = clean/(max(abs(clean))+EPS)
    noise = noise/(max(abs(noise))+EPS)
    rmsclean, rmsnoise = active_rms(clean=clean, noise=noise)
    
    # Set the noise level for a given SNR
    if rmsclean==0:
        noise = normalize_segmental_rms(noise, rms=rmsnoise, target_level=target_level)
        noisenewlevel=noise
    elif rmsnoise==0:
        noisenewlevel = noise
        clean = normalize_segmental_rms(clean, rms=rmsclean, target_level=target_level)
    else:
        clean = normalize_segmental_rms(clean, rms=rmsclean, target_level=target_level)
        noise = normalize_segmental_rms(noise, rms=rmsnoise, target_level=target_level)
        noisescalar = rmsclean / (10**(snr/20)) / (rmsnoise+EPS)
        noisenewlevel = noise * noisescalar

    # Mix noise and clean speech
    noisyspeech = clean + noisenewlevel
    # Randomly select RMS value between -15 dBFS and -35 dBFS and normalize noisyspeech with that value
    # There is a chance of clipping that might happen with very less probability, which is not a major issue. 
    noisy_rms_level = np.random.randint(target_level_lower, target_level_upper)
    rmsnoisy = (noisyspeech**2).mean()**0.5
    scalarnoisy = 10 ** (noisy_rms_level / 20) / (rmsnoisy+EPS)
    noisyspeech = noisyspeech * scalarnoisy
    clean = clean * scalarnoisy
    noisenewlevel = noisenewlevel * scalarnoisy
    # Final check to see if there are any amplitudes exceeding +/- 1. If so, normalize all the signals accordingly
    if is_clipped(noisyspeech):
        noisyspeech_maxamplevel = max(abs(noisyspeech))/(clipping_threshold-EPS)
        noisyspeech = noisyspeech/noisyspeech_maxamplevel
        clean = clean/noisyspeech_maxamplevel
        noisenewlevel = noisenewlevel/noisyspeech_maxamplevel
        noisy_rms_level = int(20*np.log10(scalarnoisy/noisyspeech_maxamplevel*(rmsnoisy+EPS)))

    return clean, noisenewlevel, noisyspeech, noisy_rms_level


def active_rms(clean, noise, fs=16000, energy_thresh=-50):
    '''Returns the clean and noise RMS of the noise calculated only in the active portions'''
    window_size = 100 # in ms
    window_samples = int(fs*window_size/1000)
    sample_start = 0
    noise_active_segs = []
    clean_active_segs = []

    while sample_start < len(noise):
        sample_end = min(sample_start + window_samples, len(noise))
        noise_win = noise[sample_start:sample_end]
        clean_win = clean[sample_start:sample_end]
        noise_seg_rms = 20*np.log10((noise_win**2).mean()+EPS)
        clean_seg_rms= 20*np.log10((clean_win**2).mean()+EPS)
        # Considering frames with energy
        if noise_seg_rms > energy_thresh:
            noise_active_segs = np.append(noise_active_segs, noise_win)
        if clean_seg_rms>energy_thresh:
            clean_active_segs = np.append(clean_active_segs, clean_win)
        sample_start += window_samples

    if len(noise_active_segs)!=0:
        noise_rms = (noise_active_segs**2).mean()**0.5
    else:
        noise_rms = 0
        
    if len(clean_active_segs)!=0:
        clean_rms = (clean_active_segs**2).mean()**0.5
    else:
        clean_rms = 0

    return clean_rms, noise_rms