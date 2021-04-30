def save_gnn_csv(encoder_path, decoder_path, name):
    import argparse
    import random

    import torch
    import torch.nn as nn

    from utils.Model.layers.Encoder import Encoder
    from utils.Model.layers.Decoder import Decoder
    from utils.Model.utils.test import test

    from utils.Model.utils.data.dataset import SantanderDataset
    from utils.Model.utils.data.dataloader import SantanderDataloader

    # import setting_param
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath("__file__"))
    sys.path.append( str(current_dir) + '/../../' )
    from utils.setting_param import Model_InputDir, Model_main_worker, Model_main_batchSize, Model_main_state_dim, Model_main_output_dim, Model_main_n_steps, Model_main_init_L, Model_main_niter, Model_main_lr

    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=Model_main_worker)
    parser.add_argument('--batchSize', type=int, default=Model_main_batchSize, help='input batch size')
    parser.add_argument('--state_dim', type=int, default=Model_main_state_dim, help='GGNN hidden state size')
    parser.add_argument('--output_dim', type=int, default=Model_main_output_dim, help='Model output state size')
    parser.add_argument('--n_steps', type=int, default=Model_main_n_steps, help='propogation steps number of GGNN')
    parser.add_argument('--init_L', type=int, default=Model_main_init_L, help='number of observation time step')
    parser.add_argument('--niter', type=int, default=Model_main_niter, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=Model_main_lr, help='learning rate')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--verbal', action='store_true', help='print training info or not')
    parser.add_argument('--manualSeed', type=int, help='manual seed')

    opt = parser.parse_args()
    print(opt)

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    opt.dataroot = Model_InputDir
    opt.L = opt.init_L
    opt.H = 8

    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)

    test_dataset = SantanderDataset(8500, opt.dataroot, opt.L, False, False)
    test_dataloader = SantanderDataloader(test_dataset, batch_size=1, \
                                     shuffle=False, num_workers=opt.batchSize)

    opt.annotation_dim = 1
    opt.n_edge_types = test_dataset.n_edge_types
    opt.n_node = test_dataset.n_node

    encoder = Encoder(opt, kernel_size=2, n_blocks=1, state_dim_bottleneck=2, annotation_dim_bottleneck=1)
    decoder = Decoder(opt, hidden_state=40)
    encoder.double()
    decoder.double()
    print(encoder)
    print(decoder)
    #net.double()
    #print(net)

    criterion = nn.L1Loss()

    if opt.cuda:
        #net.cuda()
        encoder.cuda()
        decoder.cuda()
        criterion.cuda()

    encoder.load_state_dict(torch.load(encoder_path))
    decoder.load_state_dict(torch.load(decoder_path))

    test_loss, abs_target, abs_output = test(test_dataloader, encoder, decoder, criterion, opt, name)

    import pickle
    from utils.Model import util
    _ = pickle.load(open("./data/data.pickle","rb"))
    cluster_sizes = _["cluster_sizes"]
    cluster2id = _["cluster2id"]
    locations = _["locations"]
    timestamps = _["timestamps"]
    raw_data = _["raw_data"]
    cluster_data = _["cluster_data"]
    onehot_data = _["onehot_data"]
    id2cluster = [None for i in range(len(locations))]
    for cluster,ids in cluster2id.items():
        for pid in ids:
            id2cluster[pid] = cluster

    train_start, train_end = 8500, -15000
    valid_start, valid_end = -15000,-13000
    test_start, test_end  = -13000,-12000

    in_size = 48
    out_size = 8
    interval = 15

    tests = util.data_iterator(cluster_data[test_start:test_end],timestamps[test_start:test_end],in_size,out_size=out_size,slide=1)
    tests_onehot = util.data_iterator(onehot_data[test_start:test_end],timestamps[test_start:test_end],in_size,out_size=out_size,slide=1)
    test_data_time = timestamps[test_start:test_end]

    import numpy as np
    ts = tests[1] * cluster_sizes
    ts = np.sum(ts[::out_size],axis=2)

    target = abs_target[57:]
    target = target[::8].sum(axis=1)

    output = abs_output[57:]
    output = output[::8].sum(axis=1)

    import matplotlib.pyplot as plt
    import matplotlib.pyplot as plt
    from pandas.plotting import register_matplotlib_converters
    register_matplotlib_converters()

    pr = output
    plt.rcParams["font.family"] =  'Times New Roman'
    plt.rcParams["font.size"] =  14

    plt.figure(figsize=(16,4))

    plt.ylabel("Number of available parking lots")
    plt.xlabel("Date")
    colorlist =  ["r", "g", "b", "c", "m", "y","orange","pink","purple"]
    for term, tmp in enumerate(zip(ts,pr)):
        gx = test_data_time[term*out_size:term*out_size+out_size]
        ts_i,pr_i = tmp
        color = colorlist[term%len(colorlist)]
        plt.plot(gx,323-np.round(ts_i),":",color=color)
        plt.plot(gx,323-np.round(pr_i),"-",color=color)
    plt.plot([],[],":",label="Measured value",color="black") #凡例用
    plt.plot([],[],"-",label="Predicted value",color="black")
    plt.legend()
    plt.savefig('results/pdf/' + name + '.pdf',bbox_inches='tight')