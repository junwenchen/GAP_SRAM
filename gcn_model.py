import torch 
import torch.nn as nn
import torch.nn.functional as F 

import numpy as np

from backbone import *
from utils import *
from roi_align.roi_align import RoIAlign      # RoIAlign module
from roi_align.roi_align import CropAndResize # crop_and_resize module
from st_gcn import st_gcn, st_gcn_decoder, st_gcn_short, ConvGraphicalDecoder, ConvGraphicalEncoder

import pdb


class Discriminator2(nn.Module):
	def __init__(self, x_dim):
		super(Discriminator2, self).__init__()
		self.fc1 = nn.Linear(x_dim, 1)
		self._initialize_weights()

	def forward(self, x):
		return self.fc1(torch.mean(x[:,-1,:,:], dim=2))

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Linear):
				n = m.weight.size(1)
				m.weight.data.normal_(0, 0.01)
				m.bias.data.zero_()

class GCN_Module(nn.Module):
    def __init__(self, cfg):
        super(GCN_Module, self).__init__()
        
        self.cfg=cfg
        
        NFR =cfg.num_features_relation
        
        NG=1
        N=cfg.num_boxes
        T=cfg.num_unrolling_stages
        NFG=cfg.num_features_gcn
        NFG_ONE=NFG
        
        self.fc_rn_theta_list=torch.nn.ModuleList([ nn.Linear(NFG,NFR) for i in range(NG) ])
        self.fc_rn_phi_list=torch.nn.ModuleList([ nn.Linear(NFG,NFR) for i in range(NG) ])
        
        self.fc_gcn_list=torch.nn.ModuleList([ nn.Linear(NFG,NFG_ONE,bias=False) for i in range(NG) ])
        self.nl_gcn_list=torch.nn.ModuleList([ nn.LayerNorm([T*N,NFG_ONE]) for i in range(NG) ])

        
    def forward(self,graph_boxes_features,boxes_in_flat):
        """
        graph_boxes_features  [B*T,N,NFG]
        """ 
        # GCN graph modeling
        # Prepare boxes similarity relation
        
        B,N,NFG=graph_boxes_features.shape
        NFR=self.cfg.num_features_relation
        NG=1
        NFG_ONE=NFG
        
        OH, OW=self.cfg.out_size
        pos_threshold=self.cfg.pos_threshold
        
        # Prepare position mask
        graph_boxes_positions=boxes_in_flat.view(B*N,4)  #B*T*N, 4
        graph_boxes_positions[:,0]=(graph_boxes_positions[:,0] + graph_boxes_positions[:,2]) / 2 
        graph_boxes_positions[:,1]=(graph_boxes_positions[:,1] + graph_boxes_positions[:,3]) / 2 
        graph_boxes_positions=graph_boxes_positions[:,:2].reshape(B,N,2)  #B,T, N, 2

        graph_boxes_distances=calc_pairwise_distance_3d(graph_boxes_positions.reshape(B,N,2), \
        graph_boxes_positions.reshape(B,N,2))  #B*T, N, N 
        position_mask=( graph_boxes_distances > (pos_threshold*OW) )
        
        relation_graph=None
        graph_boxes_features_list=[]
      
        for i in range(NG):
            graph_boxes_features_theta=self.fc_rn_theta_list[i](graph_boxes_features)  #B,N,NFR
            graph_boxes_features_phi=self.fc_rn_phi_list[i](graph_boxes_features)  #B,N,NFR

            similarity_relation_graph=torch.matmul(graph_boxes_features_theta, \
            graph_boxes_features_phi.transpose(1,2))  #B,N,N
            similarity_relation_graph=similarity_relation_graph/np.sqrt(NFR)
            similarity_relation_graph=similarity_relation_graph.reshape(-1,1)  #B*N*N, 1 
        
            # Build relation graph
            relation_graph=similarity_relation_graph
            relation_graph = relation_graph.reshape(B,N,N)
            relation_graph[position_mask]=-float('inf')
            relation_graph = torch.softmax(relation_graph,dim=2)   
        
            # Graph convolution
            one_graph_boxes_features=self.fc_gcn_list[i](torch.matmul(relation_graph, graph_boxes_features))  #B, N, NFG_ONE
            one_graph_boxes_features=F.relu(one_graph_boxes_features) 
            graph_boxes_features_list.append(one_graph_boxes_features)

        graph_boxes_features=torch.sum(torch.stack(graph_boxes_features_list),dim=0) #B, N, NFG
        
        return graph_boxes_features, relation_graph

class GCNnet_volleyball(nn.Module):
    """
    main module of GCN for the volleyball dataset
    """
    def __init__(self, cfg):
        super(GCNnet_volleyball, self).__init__()
        self.cfg=cfg
        
        T, N=self.cfg.num_frames, self.cfg.num_boxes
        D=self.cfg.emb_features
        K=self.cfg.crop_size[0]
        NFB=self.cfg.num_features_boxes
        NFR, NFG=self.cfg.num_features_relation, self.cfg.num_features_gcn
        
        
        if cfg.backbone=='inv3':
            self.backbone=MyInception_v3(transform_input=False,pretrained=True)
        elif cfg.backbone=='vgg16':
            self.backbone=MyVGG16(pretrained=True)
        elif cfg.backbone=='vgg19':
            self.backbone=MyVGG19(pretrained=False)
        else:
            assert False
        
        if not cfg.train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad=False
        
        self.roi_align=RoIAlign(*self.cfg.crop_size)
        
        self.fc_emb_1=nn.Linear(K*K*D,NFB)
        self.nl_emb_1=nn.LayerNorm([NFB])
                   
        self.dropout_global=nn.Dropout(p=self.cfg.train_dropout_prob)

        self.fc_activities=nn.Linear(NFG,self.cfg.num_activities)

        self.gcn_list= GCN_Module(cfg)
        
        self.encoder = st_gcn(cfg,in_channels=1024,out_channels=64,kernel_size=(3,1),stride=1)
        self.encoder_a = st_gcn_short(cfg,in_channels=1024,out_channels=64,kernel_size=(1,1),stride=1)
        self.decoder_a = st_gcn_decoder(cfg,in_channels=1024,out_channels=64,kernel_size=(1,1),stride=1)

        self.encoder_p = ConvGraphicalEncoder(in_channels=2, out_channels=1024, \
            kernel_size=1)

        self.decoder_p = ConvGraphicalDecoder(in_channels=1024, out_channels=2, \
            kernel_size=1)

        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(m.weight)
 
    def loadmodel(self,filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        self.fc_emb_1.load_state_dict(state['fc_emb_state_dict'])
        self.fc_activities.load_state_dict(state['fc_activities_state_dict'])
        print('Load model states from: ',filepath)
        
    def loadGCN(self,filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        self.fc_emb_1.load_state_dict(state['fc_emb_state_dict'])
        print('Load model states from: ',filepath)
                
    # def forward(self,batch_data,t,frame_indices):
    def forward(self,batch_data,stage_batch_data=None,traj_rel=None,stage_center=None,test_stage=False):
        images_in, boxes_in = batch_data

        # read config parameters
        B=images_in.shape[0]
        T=images_in.shape[1]
        H, W=self.cfg.image_size
        OH, OW=self.cfg.out_size
        N=self.cfg.num_boxes
        NFB=self.cfg.num_features_boxes
        NFR, NFG=self.cfg.num_features_relation, self.cfg.num_features_gcn
        # NG=self.cfg.num_graph
        
        D=self.cfg.emb_features
        K=self.cfg.crop_size[0]
                
        images_in_flat=torch.reshape(images_in,(B*T,3,H,W))  #B*T, 3, H, W
        boxes_in_flat=torch.reshape(boxes_in,(B*T*N,4))  #B*T*N, 4

        boxes_idx=[i * torch.ones(N, dtype=torch.int)   for i in range(B*T) ]
        boxes_idx=torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, N
        boxes_idx_flat=torch.reshape(boxes_idx,(B*T*N,))  #B*T*N,

        # Use backbone to extract features of images_in
        # Pre-precess first
        images_in_flat=prep_images(images_in_flat)   #(B*T,C=3,720,1280)
        outputs=self.backbone(images_in_flat)   #(B*T,288,87,157),(B*T,768,43,78)

        # Build  features
        assert outputs[0].shape[2:4]==torch.Size([OH,OW])
        features_multiscale=[]
        for features in outputs:
            if features.shape[2:4]!=torch.Size([OH,OW]):
                features=F.interpolate(features,size=(OH,OW),mode='bilinear',align_corners=True)
            features_multiscale.append(features)
        
        features_multiscale=torch.cat(features_multiscale,dim=1)  #(B*T,1056,87,157)

        # RoI Align
        boxes_in_flat.requires_grad=False
        boxes_idx_flat.requires_grad=False
        boxes_features=self.roi_align(features_multiscale,
                                            boxes_in_flat,
                                              boxes_idx_flat)  
 
        boxes_features=boxes_features.reshape(B,T,N,-1)  #B,T,N,26400

        # Embedding 
        boxes_features=self.fc_emb_1(boxes_features)  # B,T,N,1024
        boxes_features=self.nl_emb_1(boxes_features)  # B,T,N,1024
        boxes_features=F.relu(boxes_features) 
        boxes_in_flat = boxes_in_flat.view(B,T,N,4)

        if not test_stage:
            stage_images_in, stage_boxes_in = stage_batch_data
            TS=stage_images_in.shape[1]
            stage_images_in_flat=torch.reshape(stage_images_in,(B*TS,3,H,W))
            stage_boxes_in_flat=torch.reshape(stage_boxes_in,(B*TS*N,4))
        
            stage_boxes_idx=[i * torch.ones(N, dtype=torch.int)   for i in range(B*TS) ]
            stage_boxes_idx=torch.stack(stage_boxes_idx).to(device=stage_boxes_in.device)  # B*T, N
            stage_boxes_idx_flat=torch.reshape(stage_boxes_idx,(B*TS*N,))  #B*T*N,
        
            stage_images_in_flat=prep_images(stage_images_in_flat)   #(B*T,C=3,720,1280)
            stage_outputs=self.backbone(stage_images_in_flat)

            stage_features_multiscale=[]
            for features in stage_outputs:
                if features.shape[2:4]!=torch.Size([OH,OW]):
                    features=F.interpolate(features,size=(OH,OW),mode='bilinear',align_corners=True)
                stage_features_multiscale.append(features)

            stage_features_multiscale=torch.cat(stage_features_multiscale,dim=1)
        
            stage_boxes_features=self.roi_align(stage_features_multiscale,
                                                stage_boxes_in_flat,
                                                stage_boxes_idx_flat)  #B*Ts*N, 1056, 5, 5
    
            stage_boxes_features=stage_boxes_features.reshape(B,TS,N,-1)  #B,Ts,N,26400

            stage_boxes_features=self.fc_emb_1(stage_boxes_features)  # B,T,N,1024
            stage_boxes_features=self.nl_emb_1(stage_boxes_features)  # B,T,N,1024
            stage_boxes_features=F.relu(stage_boxes_features).view(B*TS,N,-1) 
            
            stage_boxes_in_flat = stage_boxes_in_flat.view(B*TS*N,4)

            stage_boxes_features_list, _ =self.gcn_list(stage_boxes_features, stage_boxes_in_flat)
            stage_boxes_features_list=stage_boxes_features_list.view(B,5,N,-1).permute(0,1,3,2)

        else:
            stage_boxes_features_list=[]
     
        # GCN       
        graph_boxes_features=boxes_features.reshape(B,T,N,NFG).permute(0,3,1,2) #(B,1024,T,N)
        #Encoder st_gcn
        graph_boxes_features_latent = self.encoder(graph_boxes_features,boxes_in_flat)
        latent_t = graph_boxes_features_latent  #B,T,N,NFG
        latent_t = self.dropout_global(latent_t)

        dec_t = graph_boxes_features[:,:,-1,:]
        dec_t_gen = []
        pos_t_gen = []

        pos_t = traj_rel[:,-1,:,:].permute(0,2,1)

        box_t = boxes_in_flat[:,-1,:,:]
        h = (box_t[:,:,3]-box_t[:,:,1])/2
        w = (box_t[:,:,2]-box_t[:,:,0])/2

        
        for j in range(5):
            latent_s, relation_graph_a = self.encoder_a(dec_t.unsqueeze(2), box_t.unsqueeze(1))
            latent_p = self.encoder_p(pos_t, relation_graph_a)
            dec_t = self.decoder_a(latent_t, latent_s, relation_graph_a)
            pos_t = self.decoder_p(latent_t, latent_p, relation_graph_a)
            abs_pos_t = traj_rel[:,0,:,:] + pos_t.permute(0,2,1)
            box_t = torch.stack([abs_pos_t[:,:,0]-w, abs_pos_t[:,:,1]-h, abs_pos_t[:,:,0]+w, abs_pos_t[:,:,1]+h], dim=2)
            dec_t_gen.append(dec_t)
            pos_t_gen.append(abs_pos_t)
        
        video_states=torch.mean(torch.stack(dec_t_gen),dim=0)
        
        video_states=torch.max(video_states,dim=2)
        activities_scores=self.fc_activities(video_states[0])  #B*T, acty_num
        
        return activities_scores, torch.stack(dec_t_gen, dim=1), stage_boxes_features_list, torch.stack(pos_t_gen, dim=1)
        