import torch
import torch.optim as optim

import time
import random
import os
import sys

from config import *
from volleyball import *
from dataset import *
from gcn_model import *
from base_model import *
from utils import *
from utils_vis import Bar
import random

import pdb

        

def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()
            
def adjust_lr(optimizer, new_lr):
    print('change learning rate:',new_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def train_net(cfg):
    """
    training gcn net
    """
    os.environ['CUDA_VISIBLE_DEVICES']=cfg.device_list
    
    # Show config parameters
    cfg.init_config()
    show_config(cfg)
    
    # Reading dataset
    training_set,validation_set=return_dataset(cfg)
    
    params = {
        'batch_size': cfg.batch_size,
        'shuffle': True,
        'num_workers': 2
    }
    training_loader=data.DataLoader(training_set,**params)
    
    params['batch_size']=cfg.test_batch_size
    validation_loader=data.DataLoader(validation_set,**params)
    
    # Set random seed
    np.random.seed(cfg.train_random_seed)
    torch.manual_seed(cfg.train_random_seed)
    random.seed(cfg.train_random_seed)

    # Set data position
    if cfg.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # Build model and optimizer
    # basenet_list={'volleyball':Basenet_volleyball}
    # gcnnet_list={'volleyball':GCNnet_volleyball}
    
    GCNnet=GCNnet_volleyball
    model=GCNnet(cfg)
    # Load backbone
    model.loadmodel(cfg.stage1_model_path)
    if cfg.resume:
        print("Resume from %s"%cfg.stage2_model_path)
        checkpoint=torch.load(cfg.stage2_model_path)
        pdb.set_trace()
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            name = k[7:] # remove 'module.' of dataparallel
            new_state_dict[name]=v
        model.load_state_dict(new_state_dict)
        # model.load_state_dict(checkpoint['state_dict'])
    
    if cfg.use_multi_gpu:
        model=nn.DataParallel(model)

    model=model.to(device=device)
    
    model.train()
    model.apply(set_bn_eval)

    D2 = Discriminator2(1024)
    D2 = D2.to(device=device)
    D2_solver = optim.Adam(D2.parameters(), lr=cfg.train_learning_rate, weight_decay=cfg.weight_decay)
    optimizer=optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=cfg.train_learning_rate,weight_decay=cfg.weight_decay)

    # train_list={'volleyball':train_volleyball}
    # test_list={'volleyball':test_volleyball}
    # train=train_list[cfg.dataset_name]
    # test=test_list[cfg.dataset_name]
    
    if cfg.test_before_train:
        ratios=[0.1,0.4,0.7]
        for i in ratios:
            print(i)
            test_info=test_volleyball_ratio(validation_loader, model, device, 0, cfg, i)
            print(test_info)
        return

    # Training iteration
    best_result={'epoch':0, 'activities_acc':0}
    start_epoch=1
    for epoch in range(start_epoch, start_epoch+cfg.max_epoch):
        
        if epoch in cfg.lr_plan:
            adjust_lr(optimizer, cfg.lr_plan[epoch])
        
        # One epoch of forward and backward
        train_info=train_volleyball(training_loader, model, device, optimizer, D2, D2_solver, epoch, cfg)
        show_epoch_info('Train', cfg.log_path, train_info)

        # Test
        if epoch % cfg.test_interval_epoch == 0:
            test_info=test_volleyball(validation_loader, model, device, epoch, cfg)
            show_epoch_info('Test', cfg.log_path, test_info)
            
            if test_info['activities_acc']>best_result['activities_acc']:
                best_result=test_info
            print_log(cfg.log_path, 
                      'Best group activity accuracy: %.2f%% at epoch #%d.'%(best_result['activities_acc'], best_result['epoch']))
            

            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            filepath=cfg.result_path+'/epoch%d_%.2f%%.pth'%(epoch,test_info['activities_acc'])
            torch.save(state, filepath)
            print('model saved to:',filepath)   

    
   
def train_volleyball(data_loader, model, device, optimizer, D2, D2_solver, epoch, cfg):
    
    activities_meter=AverageMeter()
    loss_meter=AverageMeter()
    rec_loss_meter=AverageMeter()
    reg_loss_meter=AverageMeter()
    epoch_timer=Timer()

    ratios_set=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    bar = Bar('Processing', max=len(data_loader))

    D_interval = 20


    for i_batch, batch_data in enumerate(data_loader):
        zeros_label = (torch.zeros(batch_data[0].size(0)) + torch.rand(batch_data[0].size(0)) * 0.3).to(device)  # partial videos
        ones_label = (torch.ones(batch_data[0].size(0)) + (torch.rand(batch_data[0].size(0)) - 0.5) * 0.2).to(device)  # full videos
        ratios=random.sample(ratios_set,1)
        model.train()
        model.apply(set_bn_eval)
    
        # prepare batch data
        batch_size=batch_data[0].shape[0]
        num_frames=batch_data[0].shape[1]

        activities_in=batch_data[3].reshape((batch_size,num_frames))
        activities_in=activities_in[:,0].reshape((batch_size,)).to(device=device)

        traj_rel = batch_data[4].clone()
        traj_rel[:,1:,:,:] = batch_data[4][:,1:,:,:] - batch_data[4][:,:-1,:,:]

        stage_video = batch_data[0][:,::4][:,1:].to(device)
        stage_bbox = batch_data[1][:,::4][:,1:].to(device)
        stage_center = batch_data[4][:,::4][:,1:].to(device)

        for i in ratios:
            num_frames=int(i*20)
            video = batch_data[0][:, :num_frames][:, ::2].to(device=device)
            bbox = batch_data[1][:, :num_frames][:, ::2].to(device=device)
            traj_rel = traj_rel[:, :num_frames].to(device=device)
        
            activities_scores,seq_gen,seq_gt,seq_gen_abs=model((video,bbox), (stage_video,stage_bbox), \
                traj_rel, stage_center)

            D2_fake_score = D2(seq_gen)
            D2_real_score = D2(seq_gt)
            D_loss_real = F.binary_cross_entropy_with_logits(D2_real_score[:,0], ones_label)
            D_loss_fake = F.binary_cross_entropy_with_logits(D2_fake_score[:,0], zeros_label)
            G_loss_fake = F.binary_cross_entropy_with_logits(D2_fake_score[:,0], ones_label)

            reconstruct_loss=F.mse_loss(seq_gen,seq_gt)
            reg_loss = F.mse_loss(seq_gen_abs, stage_center)


            # Predict activities
            activities_loss=F.cross_entropy(activities_scores,activities_in)
            activities_labels=torch.argmax(activities_scores,dim=1)  
            activities_correct=torch.sum(torch.eq(activities_labels.int(), activities_in.int()).float())

            # Optimize 
            G_loss = activities_loss + reconstruct_loss*0.00001 + reg_loss*0.001 + G_loss_fake
            # Optim
            optimizer.zero_grad()
            G_loss.backward(retain_graph=True)
            optimizer.step()

            if (i_batch%D_interval==0):
                D_loss = D_loss_fake + D_loss_real
                D2_solver.zero_grad()    
                D_loss.backward(retain_graph=True)
                D2_solver.step()  # update parameters in D1_solver
                

            # Get accuracy
            activities_accuracy=activities_correct.item()/activities_scores.shape[0] 
            activities_meter.update(activities_accuracy, activities_scores.shape[0])
    
            # Total loss
            # total_loss=activities_loss+reconstruct_loss*0.0001+reg_loss*0.001
            loss_meter.update(G_loss.item(), batch_size)
            rec_loss_meter.update(reconstruct_loss.item(), batch_size)
            reg_loss_meter.update(reg_loss.item(), batch_size)



        # plot progress
        bar.suffix  = '({batch}/{size}) | Total: {total:} | ETA: {eta:} | activities_accuracy: {activities_accuracy: .4f} | activities_loss: {activities_loss: .4f} | rec_loss_meter: {rec_loss_meter: .4f} | reg_loss_meter: {reg_loss_meter: .4f}'.format(
                    batch=i_batch + 1,
                    size=len(data_loader),
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    activities_accuracy=activities_meter.avg,
                    activities_loss=loss_meter.avg,
                    rec_loss_meter=rec_loss_meter.avg,
                    reg_loss_meter=reg_loss_meter.avg,
                    )
        bar.next()
    bar.finish()
    
    train_info={
        'time':epoch_timer.timeit(),
        'epoch':epoch,
        'loss':loss_meter.avg,
        'activities_acc':activities_meter.avg*100,
    }
    
    return train_info

def test_volleyball_ratio(data_loader, model, device, epoch, cfg, ratio):
    model.eval()

    activities_meter=AverageMeter()
    loss_meter=AverageMeter()

    epoch_timer=Timer()

    with torch.no_grad():
        for batch_data_test in data_loader:
            # prepare batch data
            batch_size=batch_data_test[0].shape[0]
            num_frames=batch_data_test[0].shape[1]

            activities_in=batch_data_test[3].reshape((batch_size,num_frames))
            activities_in=activities_in[:,0].reshape((batch_size,)).to(device)
            
            traj_rel = batch_data_test[4].clone()
            traj_rel[:,1:,:,:] = batch_data_test[4][:,1:,:,:] - batch_data_test[4][:,:-1,:,:]

            num_frames=int(ratio*20)

            video = batch_data_test[0][:, :num_frames][:, ::2].to(device=device)
            bbox = batch_data_test[1][:, :num_frames][:, ::2].to(device=device)
            traj_rel = traj_rel[:, :num_frames].to(device=device)
        
            activities_scores,seq_gen,_,seq_gen_abs=model((video,bbox),traj_rel=traj_rel,test_stage=True)

            # Predict activities
            activities_loss=F.cross_entropy(activities_scores,activities_in)
            activities_labels=torch.argmax(activities_scores,dim=1)
            activities_correct=torch.sum(torch.eq(activities_labels.int(),activities_in.int()).float())

            # Get accuracy
            activities_accuracy=activities_correct.item()/activities_scores.shape[0]
            activities_meter.update(activities_accuracy, activities_scores.shape[0])

            # Total loss
            total_loss=activities_loss
            loss_meter.update(total_loss.item(), batch_size)
    test_info={
        'time':epoch_timer.timeit(),
        'epoch':epoch,
        'loss':loss_meter.avg,
        'activities_acc':activities_meter.avg*100,
    }
    #
    return test_info


        
    
def test_volleyball(data_loader, model, device, epoch, cfg):
    model.eval()
    
    activities_meter=AverageMeter()
    loss_meter=AverageMeter()
    
    epoch_timer=Timer()
    bar = Bar('Processing', max=len(data_loader)) 

    # ratios=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    ratios=[0.1,0.3,0.5,0.7,0.9]
    with torch.no_grad():
        for i_batch, batch_data_test in enumerate(data_loader):
            # prepare batch data
            batch_size=batch_data_test[0].shape[0]
            num_frames=batch_data_test[0].shape[1]

            activities_in=batch_data_test[3].reshape((batch_size,num_frames))
            activities_in=activities_in[:,0].reshape((batch_size,)).to(device)
    
            stage_video = batch_data_test[0][:,::4][:,1:].to(device)
            stage_bbox = batch_data_test[1][:,::4][:,1:].to(device)
        
            traj_rel = batch_data_test[4].clone()
            traj_rel[:,1:,:,:] = batch_data_test[4][:,1:,:,:] - batch_data_test[4][:,:-1,:,:]

            for i in ratios:
                num_frames=int(i*20)

                video = batch_data_test[0][:, :num_frames][:, ::2].to(device=device)
                bbox = batch_data_test[1][:, :num_frames][:, ::2].to(device=device)
                traj_rel = traj_rel[:, :num_frames].to(device=device)

                activities_scores,_,_,_=model((video,bbox), (stage_video, stage_bbox), \
                    traj_rel=traj_rel, test_stage=True)

                # Predict activities
                activities_loss=F.cross_entropy(activities_scores,activities_in)
                activities_labels=torch.argmax(activities_scores,dim=1) 
                
                activities_correct=torch.sum(torch.eq(activities_labels.int(),activities_in.int()).float())
                
                # Get accuracy
                activities_accuracy=activities_correct.item()/activities_scores.shape[0]
                activities_meter.update(activities_accuracy, activities_scores.shape[0])

                # Total loss
                total_loss=activities_loss 
                loss_meter.update(total_loss.item(), batch_size)

            # plot progress
            bar.suffix  = '({batch}/{size})  Total: {total:} | ETA: {eta:} | activities_loss: {activities_loss: .2f}'.format(
                        batch=i_batch + 1,
                        size=len(data_loader),
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        activities_loss=activities_meter.avg,
                        )
            bar.next()

        bar.finish()

    test_info={
        'time':epoch_timer.timeit(),
        'epoch':epoch,
        'loss':loss_meter.avg,
        'activities_acc':activities_meter.avg*100
    }
    
    return test_info