import torch.nn.functional as F
import torch

# Training Step
def step(extractor, classifier, # Feature encoder and classifier
         sim_loss, cls_loss, kd_loss, # Loss functions, sim_loss: Contrastive loss, cls_loss: Classification loss, kd_loss: Knowledge distillation loss
         org_H, org_A, fc, label, # org_H: Original graph feature, org_A: Original adjacency matrix, fc: Original fc, label: Labels
         aug_H=None, aug_A=None, aug_fc=None, # aug_H: Augmented graph feature, aug_A: Augmented adjacency matrix, aug_fc: Augmented fc
         temperature=1, alpha=0.7, # temperature: Distillation temperature, alpha: The proportion of classification loss to knowledge distillation loss, alpha * cls_loss + (1 - alpha) * kd_loss
         merged_score=None, device='cpu', joint=False, optimizer_e=None, optimizer_c=None): # joint: Whether to adopt joint optimization

    aug_feature, aug_logit, aug_score, loss = None, None, None, None

    # Entry model
    org_feature, org_project_feature = extractor(org_H.to(device), org_A.to(device), fc.to(device)) # Get the original graph representation
    if aug_A is not None and aug_H is not None: aug_feature, aug_project_feature = extractor(aug_H.to(device), aug_A.to(device), aug_fc.to(device)) # Get the augmented graph representation
    elif aug_A is None and aug_H is None: pass
    else: raise ValueError('aug_A and aug_H must be input or None at the same time')

    org_logit = classifier(org_feature) # Get the original graph score
    if aug_A is not None and aug_H is not None: aug_logit = classifier(aug_feature) # Get the augmented graph score
    elif aug_A is None and aug_H is None: pass
    else: raise ValueError('aug_A and aug_H must be input or None at the same time')

    org_score = F.log_softmax(org_logit / temperature, dim=1) # Get the original graph soft label
    if aug_A is not None and aug_H is not None: aug_score = (aug_logit / temperature).softmax(1) # Get the augmented graph soft label
    elif aug_A is None and aug_H is None: pass
    else: raise ValueError('aug_A and aug_H must be input or None at the same time')

    # Optimization Process
    if joint: # Joint optimization
        if optimizer_e is not None and optimizer_c is None: # If only the feature encoder is trained, the loss is a contrastive loss
            extractor.train()
            with torch.no_grad(): classifier.eval()

            loss = -0.5 * (sim_loss(org_feature, aug_project_feature.detach()).mean() + sim_loss(org_project_feature.detach(), aug_feature).mean())

            optimizer_e.zero_grad()
            loss.backward()
            optimizer_e.step()

        elif optimizer_e is not None and optimizer_c is not None: # If both the feature encoder and classifier are trained, the losses are classification loss and knowledge distillation loss
            extractor.train()
            classifier.train()

            loss_cls = 0.5 * (cls_loss(org_logit, label.to(device)) + cls_loss(aug_logit, label.to(device)))
            loss_kd_accumulate = 0.0
            if merged_score is not None: # If there are aggregated soft label vectors, the knowledge distillation loss needs to be calculated, otherwise the knowledge distillation loss is not calculated
                loss_kd_accumulate = 0.0
                for i in range(len(label)):
                    loss_kd_accumulate += kd_loss(org_score[i], merged_score[label[i]].to(device)) # Knowledge distillation for each category, respectively
                loss_kd_accumulate /= len(label)
            loss = alpha * loss_cls + (1 - alpha) * loss_kd_accumulate

            optimizer_e.zero_grad()
            optimizer_c.zero_grad()
            loss.backward()
            optimizer_e.step()
            optimizer_c.step()

        else:
            with torch.no_grad():
                extractor.eval()
                classifier.eval()


    if not joint: # Direct optimization
        if optimizer_e is not None and optimizer_c is not None:
            extractor.train()
            classifier.train()

            loss_sim = -0.5 * (sim_loss(org_feature, aug_project_feature.detach()).mean() + sim_loss(org_project_feature.detach(), aug_feature).mean())
            loss_kd_accumulate = 0.0
            loss_cls = 0.5 * (cls_loss(org_logit, label.to(device)) + cls_loss(aug_logit, label.to(device)))
            if merged_score is not None:
                loss_kd_accumulate = 0.0
                for i in range(len(label)):
                    loss_kd_accumulate += kd_loss(org_score[i], merged_score[label[i]].to(device))
                loss_kd_accumulate /= len(label)
            loss = loss_sim + alpha * loss_cls + (1 - alpha) * loss_kd_accumulate

            optimizer_e.zero_grad()
            optimizer_c.zero_grad()
            loss.backward()
            optimizer_e.step()
            optimizer_c.step()

        else:
            with torch.no_grad():
                extractor.eval()
                classifier.eval()

    return org_feature, aug_feature, org_logit, aug_logit, org_score, aug_score, loss

