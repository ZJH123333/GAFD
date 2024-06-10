import random
import os
import time
from tqdm import tqdm
import option
from model import *
from load_ABIDE_data import *
from function import *
from metrics import evaluate
from server import *
from training_step import step

# Load option
argv = option.parse()

# Set up random seeds and run devices
torch.manual_seed(argv.seed)
np.random.seed(argv.seed)
random.seed(argv.seed)
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.manual_seed_all(argv.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    device = torch.device("cpu")

# Construct dataset
train_bold_dir = './Data/bold'
train_info_dir = './Data/info'
bold_file = os.listdir(train_bold_dir)
info_file = os.listdir(train_info_dir)
site_num = len(bold_file)
sample_num_list = [] # Record the number of samples per site
for num in range(site_num):
    data_path = train_bold_dir+'/'+bold_file[num]
    csv_path = train_info_dir+'/'+info_file[num]
    exec("dataset%s=Data_ABIDE(data_path, csv_path, k_fold=argv.k_fold)"%num)
    exec("dataset_test%s=Data_ABIDE(data_path, csv_path, k_fold=argv.k_fold)" % num)
    exec("sample_num_list.append(len(dataset%s))"%num)
    exec("dataloader%s=torch.utils.data.DataLoader(dataset%d, batch_size=argv.minibatch_size, shuffle=False)"%(num,num))
    exec("dataloader_test%s=torch.utils.data.DataLoader(dataset_test%d, batch_size=1, shuffle=False)" % (num, num))

# Collect dataset information
site_id = [str(item) for item in range(site_num)]
site_names = [item.split('.')[0] for item in bold_file]
sample_size = [str(item) for item in sample_num_list]
print('\n')
print('====================Dataset Information====================')
print('Number of Sites: ', site_num)
print('Site ID: ', ' | '.join(site_id))
print('Site Names: ', ' | '.join(site_names))
print('Sample Size: ', ' | '.join(sample_size))
print('===========================================================')
print('\n')

# Initialize parameters
# Initialize the feature encoder
extractor_init = Extractor(input_dim=argv.input_dim,
                           hidden_dim=argv.hidden_dim,
                           output_dim=argv.output_dim,
                           d_k=argv.d_k,
                           d_v=argv.d_v,
                           d_ff=argv.d_ff,
                           fc_dim=argv.fc_dim,
                           fc_hidden_dim=argv.fc_hidden_dim,
                           transformer_layer=argv.num_layers,
                           num_heads=argv.num_heads)

# Initialize the classifier
classifier_init = Classifier(input_dim=argv.fc_hidden_dim, num_class=argv.num_class)

# Initialize feature encoder and classifier parameters
init_extractor_para = [extractor_init.state_dict()] * site_num
init_classifier_para = [classifier_init.state_dict()] * site_num

# Reset feature encoder and classifier parameters
extractor_para_fix = [extractor_init.state_dict()] * site_num
classifier_para_fix = [classifier_init.state_dict()] * site_num

# Training and testing
ACCURACY = 0.0
if not argv.acc_only: PRECISION, RECALL, AUC, F1 = 0.0, 0.0, 0.0, 0.0
Time = [] # Record training time

for k in range(argv.k_fold):
    ACCURACY_k = []
    if not argv.acc_only: PRECISION_k, RECALL_K, AUC_k, F1_k = [], [], [], []
    lr = argv.lr

    for i in range(site_num):
        exec('L%s=[]' % i)

    Merge_score = None

    for iter_num in range(argv.num_iters):
        # Start training
        print('Training...')
        extractor_para_list = []
        classifier_para_list = []

        for i in range(argv.num_class):
            exec('collect_class_score%s = []' % i) # Store scores for each category
            exec('class%s_sample_num_list = []' % i) # Store sample sizes for each category

        for site in range(site_num):
            exec('dataset%s.set_fold(k,train=True)' % site)
            exec("dataloader=dataloader%s" % site)

            extractor = Extractor(input_dim=argv.input_dim,
                                  hidden_dim=argv.hidden_dim,
                                  output_dim=argv.output_dim,
                                  d_k=argv.d_k,
                                  d_v=argv.d_v,
                                  d_ff=argv.d_ff,
                                  fc_dim=argv.fc_dim,
                                  fc_hidden_dim=argv.fc_hidden_dim,
                                  transformer_layer=argv.num_layers,
                                  num_heads=argv.num_heads).to(device)
            extractor.load_state_dict(init_extractor_para[site]) # Load the initial parameters of the feature encoder for the corresponding site
            classifier = Classifier(input_dim=argv.fc_hidden_dim, num_class=argv.num_class).to(device)
            classifier.load_state_dict(init_classifier_para[site]) # Load the initial parameters of the classifier for the corresponding site

            sim_criterion = nn.CosineSimilarity() # Contrastive loss
            cls_criterion = nn.CrossEntropyLoss() # Classification loss
            kd_criterion = nn.KLDivLoss() # Knowledge distillation loss

            # Optimizer
            optimizer_e = torch.optim.Adam(extractor.parameters(), lr=lr, weight_decay=argv.weight_decay)
            optimizer_c = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=argv.weight_decay)

            start_time = time.time() # Record start time
            for epoch in range(argv.num_epochs):
                loss_accumulate = 0.0
                Label = []
                Pred = []
                Prob = []

                if epoch == argv.num_epochs - 1: # The score for each category is stored on the last local epoch
                    for i in range(argv.num_class):
                        exec('class_score%s=[]' % i)

                for _, x in enumerate(tqdm(dataloader, ncols=80, desc=f'k:{k},i:{iter_num},s:{site},e:{epoch}')):
                    t = x['timeseries'].permute(0, 2, 1)  # batch * node * time
                    H = batch_adj(t) # batch * node * node
                    if argv.fisher_z: H = fisher_z(H)
                    A = batch_adjacent_matrix(H, sparsity=argv.sparsity, self_loop=argv.self_loop).to_dense() # batch * node * node
                    if argv.add_edge_weight: A = add_edge_weight(W=batch_adj(t), A=A, n2p=argv.n2p)
                    fc = flatten(up_triu(H)[0], up_triu(H)[1])
                    label = x['label']
                    Label.extend(label.tolist())

                    # Graph augmentation
                    aug_A, aug_H = graph_augmented(batch_adj_mat=A,
                                                   batch_node_feature=H,
                                                   pe=argv.pe,
                                                   pf=argv.pf,
                                                   threshold=argv.p_threshold)
                    if argv.add_edge_weight: aug_A = add_edge_weight(W=batch_adj(t), A=aug_A, n2p=argv.n2p)
                    aug_fc = flatten(up_triu(aug_H)[0], up_triu(aug_H)[1])

                    # training step
                    org_feature, aug_feature, \
                    org_logit, aug_logit, \
                    org_score, aug_score, \
                    loss = step(extractor=extractor, classifier=classifier,
                                       sim_loss=sim_criterion, cls_loss=cls_criterion, kd_loss=kd_criterion,
                                       org_H=H, org_A=A, fc=fc,
                                       aug_H=aug_H, aug_A=aug_A, aug_fc=aug_fc,
                                       label=label,
                                       temperature=argv.temperature, alpha=argv.alpha,
                                       merged_score=Merge_score,
                                       device=device,
                                       joint=False,
                                       optimizer_e=optimizer_e,
                                       optimizer_c=optimizer_c)

                    pred = org_logit.argmax(1).tolist()
                    Pred.extend(pred)
                    prob = org_logit.softmax(1).tolist()
                    Prob.extend(prob)
                    score = (aug_logit / argv.temperature).softmax(1).tolist()

                    # The score for each category is stored on the last local epoch
                    if epoch == argv.num_epochs - 1:
                        for i in range(argv.num_class):
                            exec('Score%s = [score[ii] for ii in range(len(label)) if label[ii] == %d]' % (i, i))
                            exec('class_score%s.extend(Score%d)' % (i, i)) # class_score%s: The score of the s-th category

                    loss_accumulate += loss.detach().cpu().numpy() # Add up each batch of loss
                exec('L%s.append(loss_accumulate)' % site) # Save loss

                # Calculation the evaluation metrics
                Prob = np.array(Prob)
                evaluation_metrics = evaluate(pred=Pred, prob=Prob, label=Label, acc_only=argv.acc_only)
                print({'acc': evaluation_metrics[0],
                       'pre': evaluation_metrics[1],
                       'rec': evaluation_metrics[2],
                       'auc': evaluation_metrics[3],
                       'f1': evaluation_metrics[4]})
            end_time = time.time() # Record end time
            Time.append(end_time - start_time)

            # Average the scores for each category at each site
            for i in range(argv.num_class):
                exec('class%s_sample_num_list.append(len(class_score%d))' % (i, i)) # class%s_sample_num_list: The sample size of the s-th category
                exec('class_score%s = torch.tensor(class_score%d)' % (i, i)) # class_score%s: The score of the s-th category
                exec('class_score%s = torch.mean(class_score%d, dim = 0).tolist()' % (i, i)) # Average the scores of the s-th category
                exec('collect_class_score%s.append(class_score%d)' % (i, i)) # collect_class_score%s: The average score of the s-th category

            # Save model parameters for each site
            extractor_para_list.append(extractor.state_dict())
            classifier_para_list.append(classifier.state_dict())

            # Save the final model parameters for each fold
            if iter_num == argv.num_iters - 1:
                torch.save(extractor.state_dict(), argv.save_root_path + '/' + 'site' + str(site) + '_fold' + str(k)  + '_extractor.pth')
                torch.save(classifier.state_dict(), argv.save_root_path + '/' + 'site' + str(site) + '_fold' + str(k) + '_classifier.pth')

        # The server side aggregates scores for each site by category
        Merge_score = []
        for i in range(argv.num_class):
            # Aggregate scores by category
            exec('Merge_score%s = merge_score(collect_class_score%s, avg_weighted=argv.avg_weighted, max_weighted=argv.max_weighted, num_sample=class%s_sample_num_list)' % (i, i, i))
            # The aggregate scores of each category are loaded into Merge_score, and the final Merge_score = [aggregated scores of the first category, aggregated scores of the second category,...]
            exec('Merge_score.append(Merge_score%s)' % i)

        # Update the initial parameters for each site
        init_extractor_para = extractor_para_list
        init_classifier_para = classifier_para_list

        # Start testing
        print('Testing...')
        for site in range(site_num):
            exec('dataset_test%s.set_fold(k,train=False)' % site)
            exec("dataloader_test=dataloader_test%s" % site)

            extractor = Extractor(input_dim=argv.input_dim,
                                  hidden_dim=argv.hidden_dim,
                                  output_dim=argv.output_dim,
                                  d_k=argv.d_k,
                                  d_v=argv.d_v,
                                  d_ff=argv.d_ff,
                                  fc_dim=argv.fc_dim,
                                  fc_hidden_dim=argv.fc_hidden_dim,
                                  transformer_layer=argv.num_layers,
                                  num_heads=argv.num_heads).to(device)
            extractor.load_state_dict(init_extractor_para[site])  # Load the feature encoder parameters for the corresponding site
            classifier = Classifier(input_dim=argv.fc_hidden_dim, num_class=argv.num_class).to(device)
            classifier.load_state_dict(init_classifier_para[site])  # Load the classifier parameters for the corresponding site
            extractor.eval()
            classifier.eval()

            sim_criterion = nn.CosineSimilarity()  # Contrastive loss
            cls_criterion = nn.CrossEntropyLoss()  # Classification loss
            kd_criterion = nn.KLDivLoss()  # Knowledge distillation loss

            Label = []
            Pred = []
            Prob = []
            if iter_num == argv.num_iters - 1: # Save test sample features from the last communication
                Feature = []
            for _, x in enumerate(tqdm(dataloader_test, ncols=80, desc=f'k:{k},i:{iter_num},s:{site}')):
                t = x['timeseries'].permute(0, 2, 1)  # batch * node * time
                H = batch_adj(t)  # batch * node * node
                if argv.fisher_z: H = fisher_z(H)
                A = batch_adjacent_matrix(H, sparsity=argv.sparsity, self_loop=argv.self_loop).to_dense()  # batch * node * node
                if argv.add_edge_weight: A = add_edge_weight(W=batch_adj(t), A=A, n2p=argv.n2p)
                fc = flatten(up_triu(H)[0], up_triu(H)[1])
                label = x['label']
                Label.extend(label.tolist())
                org_feature, aug_feature, \
                org_logit, aug_logit, \
                org_score, aug_score, \
                loss = step(extractor=extractor, classifier=classifier,
                                   sim_loss=sim_criterion, cls_loss=cls_criterion, kd_loss=kd_criterion,
                                   org_H=H, org_A=A, fc=fc,
                                   label=label,
                                   temperature=argv.temperature,
                                   device=device)

                pred = org_logit.argmax(1).tolist()
                Pred.extend(pred)
                prob = org_logit.softmax(1).tolist()
                Prob.extend(prob)
                if iter_num == argv.num_iters - 1: # Save test sample features from the last communication
                    Feature.extend(org_feature.tolist())

            # Calculation the evaluation metrics
            Prob = np.array(Prob)
            evaluation_metrics = evaluate(pred=Pred, prob=Prob, label=Label, acc_only=argv.acc_only)
            print('k:', k, 'i:', iter_num, 's:', site, 'result:')
            print({'acc': evaluation_metrics[0],
                   'pre': evaluation_metrics[1],
                   'rec': evaluation_metrics[2],
                   'auc': evaluation_metrics[3],
                   'f1': evaluation_metrics[4]})


            ACCURACY_k.append(evaluation_metrics[0])
            if not argv.acc_only:
                PRECISION_k.append(evaluation_metrics[1])
                RECALL_K.append(evaluation_metrics[2])
                AUC_k.append(evaluation_metrics[3])
                F1_k.append(evaluation_metrics[4])

            # Save the final experiment result of each fold
            if iter_num == argv.num_iters - 1:
                Label = np.array(Label)
                Pred = np.array(Pred)
                Prob = np.array(Prob)
                Feature = np.array(Feature)
                np.save(argv.save_root_path + '/' + 'site' + str(site) + '_fold' + str(k) + '_Label', Label)
                np.save(argv.save_root_path + '/' + 'site' + str(site) + '_fold' + str(k) + '_Pred', Pred)
                np.save(argv.save_root_path + '/' + 'site' + str(site) + '_fold' + str(k) + '_Prob', Prob)
                np.save(argv.save_root_path + '/' + 'site' + str(site) + '_fold' + str(k) + '_Feature', Feature)

    ACCURACY_k = np.array(ACCURACY_k).reshape(argv.num_iters, site_num)
    if not argv.acc_only:
        PRECISION_k = np.array(PRECISION_k).reshape(argv.num_iters, site_num)
        RECALL_K = np.array(RECALL_K).reshape(argv.num_iters, site_num)
        AUC_k = np.array(AUC_k).reshape(argv.num_iters, site_num)
        F1_k = np.array(F1_k).reshape(argv.num_iters, site_num)

    # Save training loss per site
    for i in range(site_num):
        exec('L%s=np.array(L%d)' % (i, i))
        exec("np.save(argv.save_root_path + '/' + 'site' + str(i) + '_fold' + str(k) + '_Loss', L%s)" % i)

    # Print final evaluation metrics per fold
    print('\n', 'fold ', str(k), 'result:')
    print('acc:')
    print(ACCURACY_k)
    ACCURACY += ACCURACY_k
    if not argv.acc_only:
        print('\n', 'pre:')
        print(PRECISION_k)
        print('\n', 'rec:')
        print(RECALL_K)
        print('\n', 'auc:')
        print(AUC_k)
        print('\n', 'f1:')
        print(F1_k)
        PRECISION += PRECISION_k
        RECALL += RECALL_K
        AUC += AUC_k
        F1 += F1_k

    # Initializes the parameters of each fold
    init_extractor_para = extractor_para_fix
    init_classifier_para = classifier_para_fix

# Print final evaluation metrics
print('\n', 'final result:')
print('acc:')
print(ACCURACY / argv.k_fold)
if not argv.acc_only:
    print('\n', 'pre:')
    print(PRECISION / argv.k_fold)
    print('\n', 'rec:')
    print(RECALL / argv.k_fold)
    print('\n', 'auc:')
    print(AUC / argv.k_fold)
    print('\n', 'f1:')
    print(F1 / argv.k_fold)

# Collect training time information
total_time = round(sum(Time), 2)
Time = np.array(Time).reshape(argv.k_fold, argv.num_iters, site_num)
avg_fold_time = np.sum(np.sum(Time, axis=0), axis=0) / argv.k_fold
avg_iter_time = avg_fold_time / argv.num_iters
avg_fold_time = [str(round(item, 2)) for item in avg_fold_time]
avg_iter_time = [str(round(item, 2)) for item in avg_iter_time]
print('\n')
print('===========================================Training Time===========================================')
print('Site ID: ', ' | '.join(site_id))
print('Site Names: ', ' | '.join(site_names))
print('Total Training Time (s): ', total_time)
print('Average Training Time per Fold per Site (s): ', ' | '.join(avg_fold_time))
print('Average Training Time per Communication per Site (s): ', ' | '.join(avg_iter_time))
print('===================================================================================================')
