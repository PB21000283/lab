   def _cal_loss(self, pred, targets, targets_mask):##轨迹掩码恢复
        self.criterion_mask = torch.nn.NLLLoss(ignore_index=0, reduction='none')
        batch_loss_list = self.criterion_mask(pred.transpose(1, 2), targets)
        batch_loss = torch.sum(batch_loss_list)
        num_active = targets_mask.sum()
        mean_loss = batch_loss / num_active  # mean loss (over samples) used for optimization
        return mean_loss, batch_loss, num_active



 def _contrastive_loss(self, z1, z2, loss_type):##最大化相似度损失
        if loss_type == 'simsce':
            return self._contrastive_loss_simsce(z1, z2)
        elif loss_type == 'simclr':
            return self._contrastive_loss_simclr(z1, z2)
        elif loss_type == 'consert':
            return self._contrastive_loss_consert(z1, z2)
        else:
            raise ValueError('Error contrastive loss type {}!'.format(loss_type))

    def _contrastive_loss_simsce(self, z1, z2):
        assert z1.shape == z2.shape
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        if self.similarity == 'inner':
            similarity_matrix = torch.matmul(z1, z2.T)
        elif self.similarity == 'cosine':
            similarity_matrix = F.cosine_similarity(z1.unsqueeze(1), z2.unsqueeze(0), dim=-1)
        else:
            similarity_matrix = torch.matmul(z1, z2.T)
        similarity_matrix /= self.temperature

        labels = torch.arange(similarity_matrix.shape[0]).long().to(self.device)
        loss_res = self.criterion(similarity_matrix, labels)
        return loss_res

    def _contrastive_loss_simclr(self, z1, z2):
        """

        Args:
            z1(torch.tensor): (batch_size, d_model)
            z2(torch.tensor): (batch_size, d_model)

        Returns:

        """
        assert z1.shape == z2.shape
        batch_size, d_model = z1.shape
        features = torch.cat([z1, z2], dim=0)  # (batch_size * 2, d_model)

        labels = torch.cat([torch.arange(batch_size) for i in range(self.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)
        if self.similarity == 'inner':
            similarity_matrix = torch.matmul(features, features.T)
        elif self.similarity == 'cosine':
            similarity_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=-1)
        else:
            similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)  # [batch_size * 2, 1]

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)  # [batch_size * 2, 2N-2]

        logits = torch.cat([positives, negatives], dim=1)  # (batch_size * 2, batch_size * 2 - 1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)  # (batch_size * 2, 1)
        logits = logits / self.temperature

        loss_res = self.criterion(logits, labels)
        return loss_res

    def _contrastive_loss_consert(self, z1, z2):
        """

        Args:
            z1(torch.tensor): (batch_size, d_model)
            z2(torch.tensor): (batch_size, d_model)

        Returns:

        """
        assert z1.shape == z2.shape
        batch_size, d_model = z1.shape

        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        hidden1_large = z1
        hidden2_large = z2

        labels = torch.arange(0, batch_size).to(device=self.device)
        masks = F.one_hot(torch.arange(0, batch_size), num_classes=batch_size).to(
            device=self.device, dtype=torch.float)

        if self.similarity == 'inner':
            logits_aa = torch.matmul(z1, hidden1_large.transpose(0, 1)) / self.temperature  # shape (bsz, bsz)
            logits_bb = torch.matmul(z2, hidden2_large.transpose(0, 1)) / self.temperature  # shape (bsz, bsz)
            logits_ab = torch.matmul(z1, hidden2_large.transpose(0, 1)) / self.temperature  # shape (bsz, bsz)
            logits_ba = torch.matmul(z2, hidden1_large.transpose(0, 1)) / self.temperature  # shape (bsz, bsz)
        elif self.similarity == 'cosine':
            logits_aa = F.cosine_similarity(z1.unsqueeze(1), hidden1_large.unsqueeze(0), dim=-1) / self.temperature  # shape (bsz, bsz)
            logits_bb = F.cosine_similarity(z2.unsqueeze(1), hidden2_large.unsqueeze(0), dim=-1) / self.temperature  # shape (bsz, bsz)
            logits_ab = F.cosine_similarity(z1.unsqueeze(1), hidden2_large.unsqueeze(0), dim=-1) / self.temperature  # shape (bsz, bsz)
            logits_ba = F.cosine_similarity(z2.unsqueeze(1), hidden1_large.unsqueeze(0), dim=-1) / self.temperature  # shape (bsz, bsz)
        else:
            logits_aa = torch.matmul(z1, hidden1_large.transpose(0, 1)) / self.temperature  # shape (bsz, bsz)
            logits_bb = torch.matmul(z2, hidden2_large.transpose(0, 1)) / self.temperature  # shape (bsz, bsz)
            logits_ab = torch.matmul(z1, hidden2_large.transpose(0, 1)) / self.temperature  # shape (bsz, bsz)
            logits_ba = torch.matmul(z2, hidden1_large.transpose(0, 1)) / self.temperature  # shape (bsz, bsz)
        logits_aa = logits_aa - masks * 1e9
        logits_bb = logits_bb - masks * 1e9
        loss_a = self.criterion(torch.cat([logits_ab, logits_aa], dim=1), labels)
        loss_b = self.criterion(torch.cat([logits_ba, logits_bb], dim=1), labels)
        return loss_a + loss_b