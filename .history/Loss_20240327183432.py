   def _cal_loss(self, pred, targets, targets_mask):##轨迹掩码恢复
        self.criterion_mask = torch.nn.NLLLoss(ignore_index=0, reduction='none')
        batch_loss_list = self.criterion_mask(pred.transpose(1, 2), targets)
        batch_loss = torch.sum(batch_loss_list)
        num_active = targets_mask.sum()
        mean_loss = batch_loss / num_active  # mean loss (over samples) used for optimization
        return mean_loss, batch_loss, num_active