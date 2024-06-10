import torch

def merge_score(score_list, avg_weighted=True, max_weighted=False, num_sample=None):
    # The function is used to merge scores for each site
    # score_list: The list of the scores stored for each site
    # avg_weighted: Whether the average pooling is weighted
    # max_weighted: Whether the max pooling is weighted
    # num_sample: Sample size for each site

    score = torch.tensor(score_list)

    if num_sample is not None:
        rate = torch.tensor([x/sum(num_sample) for x in num_sample])

    if avg_weighted and num_sample is not None:
        avg_pooling = torch.sum((score.T * rate).T, dim=0)
    if not avg_weighted or num_sample is None:
        avg_pooling = torch.mean(score, dim=0)

    if max_weighted and num_sample is not None:
        max_pooling = torch.tensor([torch.max(score, dim=0)[0][i] * rate[torch.max(score, dim=0)[1][i]] for i in range(len(num_sample) - 1)])
    if not max_weighted or num_sample is None:
        max_pooling = torch.max(score, dim=0)[0]

    final_score = (avg_pooling + max_pooling).softmax(0)
    return final_score

