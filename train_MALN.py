import numpy as np, argparse, time, pickle, random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dataloader_multimodal import IEMOCAPMULTIDataset
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, precision_recall_fscore_support
from transformer_model import MaskedNLLLoss
from MALN_model_fix import MALN

seed = 100

def seed_everything(seed=seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid*size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])


def get_IEMOCAPMULTIDataset_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = IEMOCAPMULTIDataset('train')
    validset = IEMOCAPMULTIDataset('train')
    testset = IEMOCAPMULTIDataset('test')
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader


def train_or_eval_graph_model(model, loss_function, dataloader, epoch, cuda, optimizer=None, train=False):
    losses, preds, labels, masks = [], [], [], []
    scores, vids = [], []

    ei, et, en, el = torch.empty(0).type(torch.LongTensor), torch.empty(0).type(torch.LongTensor), torch.empty(0), []
    emotion_feature_original = torch.empty(0).type(torch.FloatTensor)
    private_text_original = torch.empty(0).type(torch.FloatTensor)
    private_visual_original = torch.empty(0).type(torch.FloatTensor)
    private_audio_original = torch.empty(0).type(torch.FloatTensor)
    common_feature_original = torch.empty(0).type(torch.FloatTensor)

    #if torch.cuda.is_available():
    if cuda:
        ei, et, en = ei.cuda(), et.cuda(), en.cuda()
        emotion_feature_original = emotion_feature_original.cuda()
        private_text_original = private_text_original.cuda()
        private_visual_original = private_visual_original.cuda()
        private_audio_original = private_audio_original.cuda()
        common_feature_original = common_feature_original.cuda()
        common_text_original = common_feature_original.cuda()
        common_visual_original = common_feature_original.cuda()
        common_audio_original = common_feature_original.cuda()

    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()

    seed_everything()


    for data in dataloader:
        if train:
            optimizer.zero_grad()

        # Read the dialog features
        r1, r2, r3, r4, \
        x1, x2, x3, x4, x5, x6, \
        o1, o2, o3, \
        = [d.cuda() for d in data[:13]] if cuda else data[:13]

        acouf, visuf, qmask, umask, label = [d.cuda() for d in data[13:-1]] if cuda else data[13:-1]
        
        roberta1 = r1
        roberta2 = r2
        roberta3 = r3
        roberta4 = r4
        roberta = (roberta1 + roberta2 + roberta3 + roberta4)/4
        # Average the extracted text features roberta1 ~ roberta4

        lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]

        if train:
            # Take the generated features and predictions of the model
            all_loss, emotion_feature_0, log_prob, log_prob_audio, log_prob_text, log_prob_visual, common_pred, DistanceCenter, private_text_0, private_visual_0, private_audio_0, common_feature_0, common_text_0, common_visual_0, common_audio_0 = model(train, label, roberta, visuf, acouf, x1, x2, x3, x4, x5, x6, o1, o2, o3, qmask, umask) 

            # Define the eigenvectors, which can be saved to a file later
            emotion_feature = emotion_feature_0.view(-1, emotion_feature_0.size()[-1])
            private_text = private_text_0.view(-1, private_text_0.size()[-1])
            private_audio = private_audio_0.view(-1, private_audio_0.size()[-1])
            private_visual = private_visual_0.view(-1, private_visual_0.size()[-1])
            common_feature = common_feature_0.view(-1, common_feature_0.size()[-1])
            
            common_text = common_text_0.view(-1, common_text_0.size()[-1])
            common_visual = common_visual_0.view(-1, common_visual_0.size()[-1])
            common_audio = common_audio_0.view(-1, common_audio_0.size()[-1])

            lp_ = log_prob.transpose(0, 1).contiguous().view(-1, log_prob.size()[2]) # batch*seq_len, n_classes
            lp_audio = log_prob_audio.transpose(0, 1).contiguous().view(-1, log_prob_audio.size()[2]) # batch*seq_len, n_classes
            lp_text = log_prob_text.transpose(0, 1).contiguous().view(-1, log_prob_text.size()[2]) # batch*seq_len, n_classes
            lp_visual = log_prob_visual.transpose(0, 1).contiguous().view(-1, log_prob_visual.size()[2]) # batch*seq_len, n_classes
            
            common_lp_ = common_pred.transpose(0, 1).contiguous().view(-1, common_pred.size()[2]) # batch*seq_len, n_classes
            labels_ = label.view(-1) # batch*seq_len

            # Define the classification loss
            loss_cat = loss_function(lp_audio, labels_, umask) + loss_function(lp_text, labels_, umask) +loss_function(lp_visual, labels_, umask)

            loss = 0.1 * loss_cat + loss_function(lp_, labels_, umask) + all_loss 
            # + 0.1*loss_function(common_lp_, labels_, umask) + DistanceCenter
 
        if not train:
            emotion_feature_0, log_prob, log_prob_audio, log_prob_text, log_prob_visual, DistanceCenter, private_text_0, private_visual_0, private_audio_0, common_feature_0, common_text_0, common_visual_0, common_audio_0 = model(train, label, roberta, visuf, acouf, x1, x2, x3, x4, x5, x6, o1, o2, o3, qmask, umask)
            emotion_feature = emotion_feature_0.view(-1, emotion_feature_0.size()[-1])
            private_text = private_text_0.view(-1, private_text_0.size()[-1])
            private_audio = private_audio_0.view(-1, private_audio_0.size()[-1])
            private_visual = private_visual_0.view(-1, private_visual_0.size()[-1])
            common_feature = common_feature_0.view(-1, common_feature_0.size()[-1])
            
            common_text = common_text_0.view(-1, common_text_0.size()[-1])
            common_visual = common_visual_0.view(-1, common_visual_0.size()[-1])
            common_audio = common_audio_0.view(-1, common_audio_0.size()[-1])
            
            lp_ = log_prob.transpose(0, 1).contiguous().view(-1, log_prob.size()[2]) # batch*seq_len, n_classes
            
            lp_audio = log_prob_audio.transpose(0, 1).contiguous().view(-1, log_prob_audio.size()[2]) # batch*seq_len, n_classes
            lp_text = log_prob_text.transpose(0, 1).contiguous().view(-1, log_prob_text.size()[2]) # batch*seq_len, n_classes
            lp_visual = log_prob_visual.transpose(0, 1).contiguous().view(-1, log_prob_visual.size()[2]) # batch*seq_len, n_classes
            
            labels_ = label.view(-1) # batch*seq_len
            loss_cat = 0
            loss = 0.1 * loss_cat + loss_function(lp_, labels_, umask) # + DistanceCenter
        
        #  The feature concatenation process, we aim to generate a feature matrix, which can be used for the visualization process
        if train:
            emotion_feature_original = torch.cat([emotion_feature_original, emotion_feature], dim=0)
            private_text_original = torch.cat([private_text_original, private_text], dim=0)
            private_visual_original = torch.cat([private_visual_original, private_visual], dim=0)
            private_audio_original = torch.cat([private_audio_original, private_audio], dim=0)
            common_feature_original = torch.cat([common_feature_original, common_feature], dim=0)
            
            common_text_original = torch.cat([common_text_original, common_text], dim=0)
            common_visual_original = torch.cat([common_visual_original, common_visual], dim=0)
            common_audio_original = torch.cat([common_audio_original, common_audio], dim=0)


        if not train:
            emotion_feature_original = torch.cat([emotion_feature_original, emotion_feature], dim=0)
            private_text_original = torch.cat([private_text_original, private_text], dim=0)
            private_visual_original = torch.cat([private_visual_original, private_visual], dim=0)
            private_audio_original = torch.cat([private_audio_original, private_audio], dim=0)
            common_feature_original = torch.cat([common_feature_original, common_feature], dim=0)
            
            common_text_original = torch.cat([common_text_original, common_text], dim=0)
            common_visual_original = torch.cat([common_visual_original, common_visual], dim=0)
            common_audio_original = torch.cat([common_audio_original, common_audio], dim=0)
        
        pred_ = torch.argmax(lp_,1) # batch*seq_len
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())
        losses.append(loss.item()*masks[-1].sum())

        if train:
            loss.backward()
            if args.tensorboard:
                for param in model.named_parameters():
                    writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step()

    if preds!=[]:
        preds  = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks  = np.concatenate(masks)

    else:
        return float('nan'), float('nan'), [], [], float('nan'), [], [], [], [], []

    vids += data[-1]

    # Convert the feature format and save these feature for the visualization experiments
    ei = ei.data.cpu().numpy()
    et = et.data.cpu().numpy()
    en = en.data.cpu().numpy()

    emotion_feature_original = emotion_feature_original.data.cpu().numpy()
    
    private_text_original = private_text_original.data.cpu().numpy()
    private_visual_original = private_visual_original.data.cpu().numpy()
    private_audio_original = private_audio_original.data.cpu().numpy()
    common_feature_original = common_feature_original.data.cpu().numpy()
    common_text_original = common_text_original.data.cpu().numpy()
    common_visual_original = common_visual_original.data.cpu().numpy()
    common_audio_original = common_audio_original.data.cpu().numpy()
    # np.savetxt('./Features-tsne/emotion_feature_final.txt', emotion_feature_original)
    labels = np.array(labels)
    # np.savetxt('./Features/emotion_label.txt', labels)
    preds = np.array(preds)
    vids = np.array(vids)
    class_report = classification_report(labels, preds, digits=4)
    
    avg_loss = round(np.sum(losses)/np.sum(masks), 4)
    avg_accuracy = round(accuracy_score(labels, preds, sample_weight=masks)*100, 2)
    avg_fscore = round(f1_score(labels, preds, sample_weight=masks, average='weighted')*100, 2)

    return avg_loss, avg_accuracy, labels, preds, masks, avg_fscore, vids, ei, et, en, el, class_report, emotion_feature_original, private_text_original, private_visual_original, private_audio_original, common_feature_original, common_text_original, common_visual_original, common_audio_original


def similarity(x_feature, y_feature):
    x_y_feature_sim = []
    for i in range(x_feature.size(0)):
        x_feature_ = x_feature[i]/(np.linalg.norm(x_feature[i].cpu()))
        y_feature_sim = []
        for j in range(y_feature.size(0)):
            y_feature_ = y_feature[j]/(np.linalg.norm(y_feature[j].cpu()))
            y_feature_ = np.linalg.norm(x_feature_.cpu()-y_feature_.cpu())
            y_feature_sim.append((1-0.5*y_feature_))
        x_y_feature_sim.append(y_feature_sim)
    similarity_ = torch.from_numpy(np.array(x_y_feature_sim))
    return similarity_


if __name__ == '__main__':

    path = './saved/IEMOCAP/'

    parser = argparse.ArgumentParser()

    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--cuda', action='store_true', default=1, help='does not use GPU')
    parser.add_argument('--base-model', default='LSTM', help='base recurrent model, must be one of DialogRNN/LSTM/GRU')

    parser.add_argument('--graph-model', action='store_true', default=True, help='whether to use graph model after recurrent encoding')

    parser.add_argument('--nodal-attention', action='store_true', default=False, help='whether to use nodal attention in graph model: Equation 4,5,6 in Paper')

    parser.add_argument('--windowp', type=int, default=8, help='context window size for constructing edges in graph model for past utterances')

    parser.add_argument('--windowf', type=int, default=8, help='context window size for constructing edges in graph model for future utterances')

    parser.add_argument('--lr', type=float, default=0.0004, metavar='LR', help='learning rate')
    
    parser.add_argument('--l2', type=float, default=0.00003, metavar='L2', help='L2 regularization weight')
    
    parser.add_argument('--rec-dropout', type=float, default=0.4, metavar='rec_dropout', help='rec_dropout rate')
    
    parser.add_argument('--dropout', type=float, default=0.66, metavar='dropout', help='dropout rate')
    
    parser.add_argument('--batch-size', type=int, default=32, metavar='BS', help='batch size')
    
    parser.add_argument('--epochs', type=int, default=60, metavar='E', help='number of epochs')
    
    parser.add_argument('--class-weight', action='store_true', default=False, help='use class weights')
    
    parser.add_argument('--active-listener', action='store_true', default=False, help='active listener')
    
    parser.add_argument('--attention', default='general', help='Attention type in DialogRNN model')
    
    parser.add_argument('--tensorboard', action='store_true', default=False, help='Enables tensorboard log')

    #Add
    parser.add_argument("--heads", type=int, default=1, help="Num of heads in multi-head attention.")

    parser.add_argument("--n_speakers", type=int, default=2, help="Number of speakers.")

    parser.add_argument('--num_layers', type=int, default=1, help='Num of aggcn block.')

    parser.add_argument('--num_speaker_layers', type=int, default=2, help='Num of aggcn block.')
    parser.add_argument('--num_fusion_layers', type=int, default=1, help='Num of aggcn block.')


    parser.add_argument('--sublayer_first', type=int, default=2, help='Num of the first sublayers in dcgcn block.')

    parser.add_argument('--sublayer_second', type=int, default=4, help='Num of the second sublayers in dcgcn block.')

    parser.add_argument("--u_dim", type=int, default=100, help="utterance dimension")

    parser.add_argument("--g_dim", type=int, default=200, help="First GCN_input dimension")

    parser.add_argument("--h1_dim", type=int, default=100, help="First GCN_output dimension")

    parser.add_argument("--h2_dim", type=int, default=100, help="Second GCN_output dimension")

    parser.add_argument('--gcn_dropout', type=float, default=0.5, help='AGGCN layer dropout rate.')

    parser.add_argument("--layer_stack", type=int, default=4, help="Num of attention and GAN layers")

    parser.add_argument('--vonly', action='store_true', default=False,
                        help='use the crossmodal fusion into v (default: False)')
    parser.add_argument('--aonly', action='store_true', default=False,
                        help='use the crossmodal fusion into a (default: False)')
    parser.add_argument('--lonly', action='store_true',default=True,
                        help='use the crossmodal fusion into l (default: False)')
    parser.add_argument('--attn_dropout', type=float, default=0.1,
                        help='attention dropout')
    parser.add_argument('--attn_dropout_a', type=float, default=0.0,
                        help='attention dropout (for audio)')
    parser.add_argument('--attn_dropout_v', type=float, default=0.0,
                        help='attention dropout (for visual)')
    parser.add_argument('--relu_dropout', type=float, default=0.1,
                        help='relu dropout')
    parser.add_argument('--embed_dropout', type=float, default=0.25,
                        help='embedding dropout')
    parser.add_argument('--res_dropout', type=float, default=0.1,
                        help='residual block dropout')
    parser.add_argument('--out_dropout', type=float, default=0.0,
                        help='output layer dropout')
    parser.add_argument('--num_heads', type=int, default=5,
                        help='number of heads for the transformer network (default: 5)')
    parser.add_argument('--attn_mask', action='store_false',
                        help='use attention mask for Transformer (default: true)')
    parser.add_argument('--nlevels', type=int, default=5,
                        help='number of layers in the network (default: 5)')
    
    parser.add_argument("--fusion_dim", type=int, default=172)
    parser.add_argument("--cross_n_layers", type=int, default=1)
    parser.add_argument("--cross_n_heads", type=int, default=4)

    args = parser.parse_args()
    # print(args)

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    if args.tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter()

    n_classes  = 6
    cuda = args.cuda
    n_epochs   = args.epochs
    batch_size = args.batch_size

    # D_m = 512
    D_m = 1024
    # D_m = 100
    D_g = 150
    D_p = 150
    D_e = 100
    D_h = 100
    D_a = 100
    graph_h = 100

    model = MALN(args)

    if cuda:
        model.cuda()


    loss_weights = torch.FloatTensor([1/0.086747,
                                      1/0.144406,
                                      1/0.227883,
                                      1/0.160585,
                                      1/0.127711,
                                      1/0.252668])

    loss_function = MaskedNLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)


    train_loader, valid_loader, test_loader = get_IEMOCAPMULTIDataset_loaders(valid=0.0, batch_size=batch_size,num_workers=0)

    best_fscore, best_loss, best_label, best_pred, best_mask = None, None, None, None, None
    all_fscore, all_acc, all_loss = [], [], []

    for e in range(n_epochs):
        start_time = time.time()

        if args.graph_model:

            train_loss, train_acc, train_label, _, _, train_fscore, _, _, _, _, _, _, emotion_feature_original_train, private_text_train, private_visual_train, private_audio_train, common_feature_train, common_text_train, common_visual_train, common_audio_train = train_or_eval_graph_model(model, loss_function, train_loader, e, cuda, optimizer, True)

            # Save the generated feature matrix for the following visualization experiments
            if e == 0:
                np.savetxt('./Features-tsne/feature_train_original.txt', emotion_feature_original_train)
            if train_acc>=98:
                np.savetxt('./Features-tsne/feature_train_>98.txt', emotion_feature_original_train)
            test_loss, test_acc, test_label, test_pred, test_mask, test_fscore, _, _, _, _, _, _, emotion_feature_original, private_text, private_visual, private_audio, common_feature, common_text, common_visual, common_audio = train_or_eval_graph_model(model, loss_function, test_loader, e, cuda)
            all_fscore.append(test_fscore)
            if (e+1) % 10 == 0:
                name_string = str('./Features-tsne/epochs/emotion_feature_epoch_') + str(e+1) + str('_acc') + str(test_acc) + str('_') + str('_fscore') + str(test_fscore)+str('.txt')
                # np.savetxt(name_string, emotion_feature_original)
                # torch.save({'model_state_dict': model.state_dict()}, path + name + args.base_model + '_' + str(e) + '.pkl')

        if best_fscore == None or best_fscore < test_fscore:
            # best_fscore, best_loss, best_label, best_pred, best_mask, best_attn = test_fscore, test_loss, test_label, test_pred, test_mask, attentions
            best_fscore, best_loss, best_label, best_pred, best_mask = test_fscore, test_loss, test_label, test_pred, test_mask
            best_epoch = e+1
            best_acc = test_acc
            best_feature = emotion_feature_original
            best_string = str('./Features-tsne/best/emotion_feature_best_') + str('epoch') + str(best_epoch)+ str('_acc') + str(best_acc) + str('_') + str(best_fscore)+str('.txt')

        #  Save the features (including the private, common features,labels, models and the global features) for visualization
            
        #     np.savetxt('./Features-tsne/best/feature_train.txt', emotion_feature_original_train)
        #     np.savetxt('./Features-tsne/best/private_audio.txt', private_audio)
        #     np.savetxt('./Features-tsne/best/common_feature.txt', common_feature)
        #     np.savetxt('./Features-tsne/best/private_audio_train.txt', private_audio_train)
        #     np.savetxt('./Features-tsne/best/common_text_train.txt', common_text_train)
        # np.savetxt('./Features-tsne/best/label.txt', test_label)
        # np.savetxt('./Features-tsne/best/label_train.txt', train_label)

        if args.tensorboard:
            writer.add_scalar('test: accuracy/loss', test_acc/test_loss, e)
            writer.add_scalar('train: accuracy/loss', train_acc/train_loss, e)

        print('epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'.\
                format(e+1, train_loss, train_acc, train_fscore, test_loss, test_acc, test_fscore, round(time.time()-start_time, 2)))

    if args.tensorboard:
        writer.close()

    print('Test performance..')
    print('Fscore {} accuracy {}'.format(best_fscore,
                                         round(accuracy_score(best_label, best_pred, sample_weight=best_mask) * 100,
                                               2)))
    # print the training details
    print("best epoch: ")
    print(best_epoch)
    print(classification_report(best_label, best_pred, sample_weight=best_mask, digits=4))
    print(confusion_matrix(best_label, best_pred, sample_weight=best_mask))
