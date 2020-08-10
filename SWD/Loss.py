import torch

def SWD(p1, p2):

    '''
    Paper Link : https://arxiv.org/pdf/1903.04064.pdf
    Github Link : https://github.com/apple/ml-cvpr2019-swd
    '''

    assert p1.size(0)==p2.size(0) and p1.size(1)==p2.size(1), 'p1 and p2 should be the same size.'

    if p1.size(1) > 1:
        proj = torch.randn(p1.size(1), 128)
        proj*= torch.rsqrt(torch.sum(torch.pow(proj, 2), 0, keepdim=True))
        p1 = torch.mm(p1, proj.to(p1.device))
        p2 = torch.mm(p2, proj.to(p2.device))
    
    p1 = torch.topk(p1, p1.size(0), dim=0, largest=True, sorted=True, out=None)[0]
    p2 = torch.topk(p2, p2.size(0), dim=0, largest=True, sorted=True, out=None)[0]

    sliced_wasserstein_discrepancy = torch.mean(torch.pow((p1-p2), 2))

    return sliced_wasserstein_discrepancy