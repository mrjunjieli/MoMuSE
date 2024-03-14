import time
from apex import amp
from apex.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch.nn as nn 
from datetime import datetime
import torch 
from tools import cal_SISNR
import copy
import numpy as np 
import random 
random.seed(0)



EPS = np.finfo(float).eps
import pdb 


class Solver(object):
    def __init__(self, train_data, validation_data, model, optimizer, args):
        self.train_data = train_data
        self.validation_data = validation_data
        self.args = args
        self.amp = amp 
        self.ae_loss = nn.CrossEntropyLoss()

        self.print = False
        if (self.args.distributed and self.args.local_rank ==0) or not self.args.distributed:
            self.print = True
            if self.args.use_tensorboard:
                self.writer = SummaryWriter('logs/%s/tensorboard/' % args.log_name)
        self.model, self.optimizer = self.amp.initialize(model, optimizer,
                                                        opt_level=args.opt_level,
                                                        patch_torch_functions=args.patch_torch_functions)


        if self.args.distributed:
            self.model = DDP(self.model)

        self._reset()

    def _reset(self):
        self.halving = False
        if self.args.continue_from:
            checkpoint = torch.load('logs/%s/model_dict_last.pt' % self.args.continue_from, map_location='cpu')
            pretrained_model = checkpoint['model']
            state = self.model.state_dict()
            for key in state.keys():
                pretrain_key = key
                if pretrain_key in pretrained_model.keys():
                    state[key] = pretrained_model[pretrain_key]
                elif 'module.'+pretrain_key in pretrained_model.keys():
                    state[key] = pretrained_model['module.'+pretrain_key]
                else:
                    print(key +' is not loaded!!') 
            
            self.model.load_state_dict(state)
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.amp.load_state_dict(checkpoint['amp'])

            self.start_epoch=checkpoint['epoch']
            self.best_val_loss = checkpoint['best_val_loss']
            self.val_no_impv = checkpoint['val_no_impv']

            if self.print: print("Resume training from epoch: {}".format(self.start_epoch))
            
        else:
            self.best_val_loss = float("inf")
            self.val_no_impv = 0
            self.start_epoch=1
            if self.print: print('Start new training')

    def train(self):
        for epoch in range(self.start_epoch, self.args.epochs+1):
            self.joint_loss_weight=epoch
            if self.args.distributed: self.args.train_sampler.set_epoch(epoch)

#             Train
            self.model.train()
            start = time.time()
            tr_loss = self._run_one_epoch(data_loader = self.train_data, state='train',epoch=epoch)
            reduced_tr_loss = self._reduce_tensor(tr_loss)

            if self.print: print('Train Summary | End of Epoch {0} | Time {1:.2f}s | Current time {2} |'
                      'Train Loss {3:.3f}| '.format(
                        epoch, time.time() - start,datetime.now(),reduced_tr_loss))

            #Validation
            self.model.eval()
            start = time.time()
            with torch.no_grad():
                val_loss = self._run_one_epoch(data_loader = self.validation_data, state='val',epoch=epoch)
                reduced_val_loss = self._reduce_tensor(val_loss)
                if self.print: print('Valid Summary | End of Epoch {0} | Time {1:.2f}s | Current time {2} |'
                        'Valid Loss {3:.3f}| '.format(
                            epoch, time.time() - start, datetime.now(),reduced_val_loss))
            
            # Check whether to adjust learning rate and early stop
            find_best_model = False
            if reduced_val_loss >= self.best_val_loss:
                self.val_no_impv += 1
                if self.val_no_impv >= 10:
                    if self.print: print("No imporvement for 10 epochs, early stopping.")
                    break
            else:
                self.val_no_impv = 0
                self.best_val_loss = reduced_val_loss
                find_best_model=True

            if self.val_no_impv == 6 :
                self.halving = True

            # Halfing the learning rate
            if self.halving:
                optim_state = self.optimizer.state_dict()
                optim_state['param_groups'][0]['lr'] = optim_state['param_groups'][0]['lr'] /2
                self.optimizer.load_state_dict(optim_state)
                if self.print: print('Learning rate adjusted to: {lr:.6f}'.format(
                    lr=optim_state['param_groups'][0]['lr']))
                self.halving = False

            if self.print:
                # Tensorboard logging
                if self.args.use_tensorboard:
                    self.writer.add_scalar('Train_loss', reduced_tr_loss, epoch)
                    self.writer.add_scalar('Validation_loss', reduced_val_loss, epoch)

                # Save model
                checkpoint = {'model': self.model.state_dict(),
                                'optimizer': self.optimizer.state_dict(),
                                'amp': self.amp.state_dict(),
                                'epoch': epoch+1,
                                'best_val_loss': self.best_val_loss,
                                'val_no_impv': self.val_no_impv}
                torch.save(checkpoint, "logs/"+ self.args.log_name+"/model_dict_last.pt")
                if find_best_model:
                    torch.save(checkpoint, "logs/"+ self.args.log_name+"/model_dict_best.pt")
                    print("Fund new best model, dict saved")
                if epoch %5 ==0:
                    torch.save(checkpoint, "logs/"+ self.args.log_name+"/model_dict_"+str(epoch)+".pt")


    def _run_one_epoch(self, data_loader, state, epoch=0):
        step=0
        total_step = len(data_loader)
        total_loss = 0

        self.accu_count = 0
        self.optimizer.zero_grad()
        for i, (a_mix,a_tgt,v_tgt,speaker,mask_type,ratio) in enumerate(data_loader):
            a_mix = a_mix.cuda().squeeze(0).float()
            a_tgt = a_tgt.cuda().squeeze(0).float()
            v_tgt = v_tgt.cuda().squeeze(0).float()
            speaker = speaker.cuda().squeeze()


            chunk = random.uniform(0.05,0.2)
            recep_field = random.uniform(1,3) 
            init = int(random.uniform(chunk,recep_field) * 16000)

            a_chunk_length = int(chunk *16000)
            v_chunk_length = a_chunk_length//(16000//25)

            a_recep_length = int(recep_field*16000)
            v_recep_length = a_recep_length//(16000//25)
            
            j = random.randint(init//a_chunk_length,a_mix.shape[1]//a_chunk_length)
            
            a_start = max(0, j*a_chunk_length-a_recep_length)
            if a_chunk_length*(j+1) >= a_mix.shape[1]:
                a_duration_seg = a_chunk_length - (a_chunk_length*(j+1) - a_mix.shape[1])
            else:
                a_duration_seg = a_chunk_length
            a_end = a_chunk_length*j+a_duration_seg
            a_mix_seg = a_mix[:,a_start:a_end]
            a_tgt_seg = a_tgt[:,a_start:a_end]

            v_start = max(0, j*v_chunk_length-v_recep_length)
            if v_chunk_length*(j+1) >= v_tgt.shape[1]:
                v_duration_seg = v_chunk_length - (v_chunk_length*(j+1) - v_tgt.shape[1])
            else:
                v_duration_seg = v_chunk_length
            v_end = v_chunk_length*j+v_duration_seg
            v_tgt_seg = v_tgt[:,v_start:v_end,:,:]

            a_mix_ref = a_mix[:,:a_end-a_duration_seg]
            v_ref = v_tgt[:,:v_end-v_duration_seg,:]
            a_tgt_ref = a_tgt[:,:a_end-a_duration_seg]
            
            # offline part 
            est_speaker_off, est_a_tgt_off,_,_ = self.model(a_mix, v_tgt,[])

            # # online part
            with torch.no_grad():
                _, _,spk_emb,_ = self.model(a_mix_ref, v_ref,[])
            est_speaker, est_a_tgt_seg,_,att_score = self.model(a_mix_seg, v_tgt_seg,spk_emb)

            
            max_snr_off = cal_SISNR(a_tgt, est_a_tgt_off)
            sisnr_loss_off = 0 - torch.mean(max_snr_off)
            
            max_snr = cal_SISNR(a_tgt_seg, est_a_tgt_seg)
            sisnr_loss = 0 - torch.mean(max_snr)

            speaker_loss_off = self.ae_loss(est_speaker_off[0], speaker) + \
                            self.ae_loss(est_speaker_off[1], speaker) + \
                            self.ae_loss(est_speaker_off[2], speaker) + \
                            self.ae_loss(est_speaker_off[3], speaker)
            speaker_loss = self.ae_loss(est_speaker[0], speaker) + \
                            self.ae_loss(est_speaker[1], speaker) + \
                            self.ae_loss(est_speaker[2], speaker) + \
                            self.ae_loss(est_speaker[3], speaker)

            loss_off = sisnr_loss_off + 0.1* speaker_loss_off
            loss_on =  sisnr_loss     + 0.1* speaker_loss
            
            # penalty part
            l1loss = nn.L1Loss()
            penalty_loss = l1loss(att_score[0], torch.zeros_like(att_score[0]))+\
                            l1loss(att_score[1], torch.zeros_like(att_score[1]))+\
                            l1loss(att_score[2], torch.zeros_like(att_score[2]))+\
                            l1loss(att_score[3], torch.zeros_like(att_score[3]))
            
            
            if state =='train':
                loss = 0.3*loss_off +0.7*loss_on +0.05*penalty_loss
                self.accu_count += 1
                step+=1
                total_loss+=loss.data

                if self.args.accu_grad:
                    loss = loss/(self.args.effec_batch_size / self.args.batch_size)
                    with self.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.amp.master_params(self.optimizer),
                                                       self.args.max_norm)
                    if self.accu_count == (self.args.effec_batch_size / self.args.batch_size):
                        sself.optimizer.step()
                        self.optimizer.zero_grad()
                        self.accu_count = 0
                else:
                    with self.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                            scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_norm, norm_type=2)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                if i%1000==0:
                    print('step:{}/{} avg loss:{:.3f}'.format(step, total_step,total_loss / step))
            else: 
                step+=1
                loss = sisnr_loss 
                total_loss+=loss.data

                if i%300==0:
                    print('step:{}/{} avg loss:{:.3f}'.format(step, total_step,total_loss / step))

        return total_loss / (i+1)

    def _reduce_tensor(self, tensor):
        if not self.args.distributed: return tensor
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= self.args.world_size
        return rt

