def fit(period, training_idx):
    import argparse
    import random

    import torch
    import torch.nn as nn
    import torch.optim as optim

    #from model import STGGNN
    from utils.Model.layers.Encoder import Encoder
    from utils.Model.layers.Decoder import Decoder
    from utils.Model.utils.train import train
    from utils.Model.utils.valid import valid
    from utils.Model.utils.test import test
    #from utils.pytorchtools import EarlyStopping
    from utils.Model.utils.pytorchtools_seq2seq import EarlyStopping

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

    train_dataset = SantanderDataset(training_idx, opt.dataroot, opt.L, True, False)
    train_dataloader = SantanderDataloader(train_dataset, batch_size=opt.batchSize, \
                                      shuffle=True, num_workers=2, drop_last=True)

    valid_dataset = SantanderDataset(training_idx, opt.dataroot, opt.L, False, True)
    valid_dataloader = SantanderDataloader(valid_dataset, batch_size=opt.batchSize, \
                                      shuffle=True, num_workers=2, drop_last=True)

    opt.annotation_dim = 1
    opt.n_edge_types = train_dataset.n_edge_types
    opt.n_node = train_dataset.n_node

    #net = STGGNN(opt, kernel_size=2, n_blocks=1, state_dim_bottleneck=3, annotation_dim_bottleneck=1)
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

    #optimizer = optim.Adam(net.parameters(), lr=opt.lr)
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=opt.lr)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=opt.lr)
    early_stopping = EarlyStopping(patience=3, verbose=True)

    #encoder.load_state_dict(torch.load('encoder_checkpoint.pt'))
    #decoder.load_state_dict(torch.load('decoder_checkpoint.pt'))

    for epoch in range(0, opt.niter):
        train(epoch, train_dataloader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, opt)
        valid_loss = valid(valid_dataloader, encoder, decoder, criterion, opt)
        early_stopping(valid_loss, encoder, decoder, period)
        if early_stopping.early_stop:
            print("Early stopping")
            break
