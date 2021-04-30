import torch
from torch.autograd import Variable
import numpy as np
import pickle
import pandas as pd

def test(dataloader, encoder, decoder, criterion, opt, name):
    _ = pickle.load(open(opt.dataroot.replace('forGNN/learning_data', '') + "data.pickle", "rb"))
    cluster_sizes = _["cluster_sizes"]

    test_loss = 0
    mae = np.zeros((1000, 27, 8))
    abs_output = np.zeros((1000, 27, 8))
    abs_target = np.zeros((1000, 27, 8))

    encoder.eval()
    decoder.eval()
    for i, (sample_idx, annotation, adj_matrix, label_attribute) in enumerate(dataloader, 0):
        padding = torch.zeros(opt.batchSize, opt.n_node, opt.L, opt.state_dim - opt.annotation_dim).double()
        init_input = torch.cat((annotation, padding), 3)

        if opt.cuda:
            adj_matrix      = adj_matrix.cuda()
            annotation      = annotation.cuda()
            init_input      = init_input.cuda()
            label_attribute = label_attribute.cuda()

        adj_matrix      = Variable(adj_matrix)
        annotation      = Variable(annotation)
        init_input      = Variable(init_input)
        label_attribute = Variable(label_attribute)

        encoder_output = encoder(init_input, annotation, adj_matrix) # (batch_size, n_node, state_dim)
        encoder_output = encoder_output.unsqueeze(0) # lstmのstateへの入力のnum_layers * num_directionsの次元を拡張 (1, batch_size, n_node, state_dim)
        # encoder_output = encoder_output.view(1, opt.batchSize * opt.n_node, opt.state_dim) # (1, batch_size * n_node, state_dim)
        encoder_output = encoder_output.view(1, opt.batchSize * opt.n_node, 40)  # (1, batch_size * n_node, state_dim)
        decoder_hidden = (encoder_output, encoder_output)

        decoder_input = annotation[:, :, -1] # decoderの最初の入力はencoderへの最後の入力
        decoder_input = torch.cat((decoder_input, label_attribute[:,:,0,1:]), dim=2) # H=0のattrを連結
        decoder_input = decoder_input.unsqueeze(2) # Hの次元を拡張
        decoder_input = torch.cat((decoder_input, encoder_output.unsqueeze(2)), dim=3) # peeky

        output = torch.zeros((opt.batchSize, opt.n_node, 1, opt.output_dim), dtype=torch.double) # dummyのzero要素
        for h in range(opt.H):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, False)
            decoder_output = decoder_output.view(opt.batchSize, opt.n_node, 1, opt.output_dim)
            output = torch.cat((output, decoder_output), dim=2) # 結果を保持
            if h+1<opt.H:
                decoder_input = decoder_output.view(opt.batchSize, opt.n_node, opt.output_dim)
                decoder_input = torch.cat((decoder_input, label_attribute[:, :, h+1, 1:]), dim=2)  # H=h+1, h+1>0のattrを連結
                decoder_input = decoder_input.unsqueeze(2) # Hの次元を拡張
                decoder_input = torch.cat((decoder_input, encoder_output.unsqueeze(2)), dim=3) # peeky

        output = output[:,:,1:,:] # dummyのzero要素を除去
        output = output.view(opt.batchSize, opt.n_node, opt.H)
        target = label_attribute[:,:,:,0]

        for C in range(27):
            for H in range(8):
                abs_output[i][C][H] = cluster_sizes[C] * output[0, C, H]
                abs_target[i][C][H] = cluster_sizes[C] * target[0, C, H]
                loss = criterion(cluster_sizes[C] * output[0, C, H], cluster_sizes[C] * target[0, C, H]).item()
                mae[i][C][H] = loss

        loss = criterion(output, target).item()
        test_loss += loss
        print(i, output.shape, target.shape, loss)

    test_loss /= (len(dataloader.dataset) / opt.batchSize)

    print('Test set: Average loss: {:.4f}'.format(test_loss))

    print(np.abs(abs_output.sum(axis=1) - abs_target.sum(axis=1)).mean(axis=0))

    mae_all = np.abs(abs_output.sum(axis=1) - abs_target.sum(axis=1)).mean(axis=0)
    result_dic = {'15 min': [mae_all[0]], '30 min': [mae_all[1]], '45 min': [mae_all[2]], '60 min': [mae_all[3]], '75 min': [mae_all[4]],
                  '90 min': [mae_all[5]], '105 min': [mae_all[6]], '120 min': [mae_all[7]]}
    df = pd.DataFrame(result_dic)
    df.to_csv('results/csv/' + name + '_v1.csv')

    mae = mae.mean(axis=0).transpose((1,0))
    result_dic = {'15 min':mae[0], '30 min':mae[1], '45 min':mae[2], '60 min':mae[3], '75 min':mae[4], '90 min':mae[5], '105 min':mae[6], '120 min':mae[7]}
    df = pd.DataFrame(result_dic)
    df.to_csv('results/csv/' + name + '_v2.csv')

    return test_loss, abs_target, abs_output