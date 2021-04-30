import torch
from torch.autograd import Variable

#def train(epoch, dataloader, net, criterion, optimizer, opt):
def train(epoch, dataloader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, opt):
    encoder.train()
    decoder.train()
    for i, (sample_idx, annotation, adj_matrix, label_attribute) in enumerate(dataloader, 0):
        encoder.zero_grad()
        decoder.zero_grad()
        padding = torch.zeros(opt.batchSize, opt.n_node, opt.L, opt.state_dim - opt.annotation_dim).double()
        #padding = torch.zeros(opt.batchSize, opt.n_node, opt.L, 0).double() パディングしないパターン
        init_input = torch.cat((annotation, padding), 3)
        #annotation = annotation[:,:,:,0].reshape(1, 27, 48, 1) 空き台数だけ使いたいパターン。（もともと他の特徴量が無いので気にしなくていい）
        """
        軽いやつ
        adj_matrix      :(4, 2, 3, 888)         :(batch_size, 2[in, out], 3[value + row + col], max_nnz_am)
        annotation      :(4, 100, 3, 101)       :(batch_size, n_node, L, annotation_dim)
        init_input      :(4, 100, 3, 101)       :(batch_size, n_node, L, state_dim )
        label_edge      :(4, 3, 777)            :(batch_size, 3[value + row + col], max_nnz_label_edge)
        # label_attribute :(4, 100, 101)          :(batch_size, n_node, label_attribute_dim)
        label_attribute :(4, 100, 8, 101)          :(batch_size, n_node, H, label_attribute_dim)
        label_lost      :(4, 80)                :(batch_size, n_existing_node)
        label_return    :(4, 80)                :(batch_size, n_existing_node)
        
        本番
        adj_matrix      :(4, 2, 3, 6466561)     :(batch_size, 2[in, out], 3[value + row + col], max_nnz_am)
        annotation      :(4, 4359930, 3, 101)   :(batch_size, n_node, L, annotation_dim)
        init_input      :(4, 4359930, 3, 101)   :(batch_size, n_node, L, state_dim)
        label_edge      :(4, 3, 2067514)        :(batch_size, 3[value + row + col], max_nnz_label_edge)
        label_attribute :(4, 4359930, 101)      :(batch_size, n_node, label_attribute_dim)
        label_lost      :(4, 2118551)           :(batch_size, n_existing_node)
        label_return    :(4, 2118551)           :(batch_size, n_existing_node)
        """
        if opt.cuda:
            adj_matrix      = adj_matrix.cuda()
            annotation      = annotation.cuda()
            init_input      = init_input.cuda()
            label_attribute = label_attribute.cuda()

        adj_matrix      = Variable(adj_matrix)
        annotation      = Variable(annotation)
        init_input      = Variable(init_input)
        label_attribute = Variable(label_attribute)

        #output = net(init_input, annotation, adj_matrix)
        encoder_output = encoder(init_input, annotation, adj_matrix) # (batch_size, n_node, state_dim)
        encoder_output = encoder_output.unsqueeze(0) # lstmの入力へのnum_layers * num_directionsの次元を拡張 (1, batch_size, n_node, state_dim)
        #encoder_output = encoder_output.view(1, opt.batchSize * opt.n_node, opt.state_dim) # (1, batch_size * n_node, state_dim)
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
        """
        decoder_input = label_attribute.clone()
        decoder_input[:,:,1:,0] = label_attribute[:,:,0:opt.H-1,0]  # 1ステップずらす
        decoder_input[:,:,0,0] = annotation[:,:,-1,0] # annotationの最後の入力をdecoderの最初の入力にする

        output, _ = decoder(decoder_input, decoder_hidden, True)
        output = output.view(opt.batchSize, opt.n_node, opt.H)        
        target = label_attribute[:,:,:,0]
        """

        loss = criterion(output, target)
        #loss = criterion(output, target) + torch.mean(torch.abs((output[:,:,-1]-output[:,:,0]) - (target[:,:,-1]-target[:,:,0]))) / 10

        loss.backward()
        #optimizer.step()
        encoder_optimizer.step()
        decoder_optimizer.step()

        #if i % int(len(dataloader) / 10 + 1) == 0 or opt.verbal:
        if i % int(len(dataloader) / 1000 + 1) == 0 and opt.verbal:
            print('[%d/%d][%d/%d] Loss: %.4f' % (epoch, opt.niter, i, len(dataloader), loss.item()))
            #torch.save(encoder.state_dict(), 'encoder_checkpoint_tmp.pt')
            #torch.save(decoder.state_dict(), 'decoder_checkpoint_tmp.pt')