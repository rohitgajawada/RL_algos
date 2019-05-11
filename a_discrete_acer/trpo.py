import torch

def TRPO(model, distribution, average_distribution, loss, threshold, g_acer, k):

	kl = -1 * (average_distribution * (distribution.log() - average_distribution.log())).sum(1).mean(0)

	k_k = (k ** 2).sum(1).mean(0)
	k_g = (k * g_acer).sum(1).mean(0)

	if k_k.item() > 0:
		trust_factor = ((k_g - threshold) / k_k).clamp(min=0).detach()
	else:
		trust_factor = torch.zeros(1).cuda()

	trust_loss_trpo = trust_factor * kl + loss

	return trust_loss_trpo
