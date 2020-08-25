from volleyball import *
from collective import *

import pickle


def return_dataset(cfg):
    train_anns = volley_read_dataset(cfg.data_path, cfg.train_seqs)
    train_frames = volley_all_frames(train_anns)

    test_anns = volley_read_dataset(cfg.data_path, cfg.test_seqs)
    test_frames = volley_all_frames(test_anns)

    all_anns = {**train_anns, **test_anns}
    all_tracks = pickle.load(open(cfg.data_path + '/tracks_normalized.pkl', 'rb'))


    training_set=VolleyballDataset(all_anns,all_tracks,train_frames,
                                    cfg.data_path,cfg.image_size,cfg.out_size,num_before=cfg.num_before,
                                    num_after=cfg.num_after,is_training=True,is_finetune=(cfg.training_stage==1))

    validation_set=VolleyballDataset(all_anns,all_tracks,test_frames,
                                    cfg.data_path,cfg.image_size,cfg.out_size,num_before=cfg.num_before,
                                        num_after=cfg.num_after,is_training=False,is_finetune=(cfg.training_stage==1))
                                         
    
    print('Reading dataset finished...')
    print('%d train samples'%len(train_frames))
    print('%d test samples'%len(test_frames))
    
    return training_set, validation_set
    