def __getitem__(self, ind):
        traj_ind, mask, temporal_mat = super().__getitem__(ind)
        traj_ind1 = self.traj_list1[ind]  # (seq_length, feat_dim)
        traj_ind2 = self.traj_list2[ind]  # (seq_length, feat_dim)
        temporal_mat1 = self.temporal_mat_list1[ind]  # (seq_length, seq_length)
        temporal_mat2 = self.temporal_mat_list2[ind]  # (seq_length, seq_length)
        mask1 = None
        mask2 = None
        if 'mask' in self.data_argument1:
            mask1 = noise_mask(traj_ind1, self.masking_ratio, self.avg_mask_len, self.masking_mode, self.distribution,
                               self.exclude_feats, self.add_cls)  # (seq_length, feat_dim) boolean array
        if 'mask' in self.data_argument2:
            mask2 = noise_mask(traj_ind2, self.masking_ratio, self.avg_mask_len, self.masking_mode, self.distribution,
                               self.exclude_feats, self.add_cls)  # (seq_length, feat_dim) boolean array
        return traj_ind, mask, temporal_mat, \
               torch.LongTensor(traj_ind1), torch.LongTensor(traj_ind2), \
               torch.LongTensor(temporal_mat1), torch.LongTensor(temporal_mat2), \
               torch.LongTensor(mask1) if mask1 is not None else None, \
               torch.LongTensor(mask2) if mask2 is not None else None


def collate_unsuperv_contrastive_split_lm(data, max_len=None, vocab=None, add_cls=True):
    features, masks, temporal_mat, features1, features2, temporal_mat1, temporal_mat2, mask1, mask2 = zip(*data)
    data_for_mask = list(zip(features, masks, temporal_mat))
    dara_for_contra = list(zip(features1, features2, temporal_mat1, temporal_mat2, mask1, mask2))

    X1, X2, padding_masks1, padding_masks2, batch_temporal_mat1, batch_temporal_mat2 \
        = collate_unsuperv_contrastive_split(data=dara_for_contra, max_len=max_len, vocab=vocab, add_cls=add_cls)

    masked_x, targets, target_masks, padding_masks, batch_temporal_mat = collate_unsuperv_mask(
        data=data_for_mask, max_len=max_len, vocab=vocab, add_cls=add_cls)
    return X1, X2, padding_masks1, padding_masks2, batch_temporal_mat1, batch_temporal_mat2, \
           masked_x, targets, target_masks, padding_masks, batch_temporal_mat



def collate_unsuperv_contrastive_split(data, max_len=None, vocab=None, add_cls=True):
    batch_size = len(data)
    features1, features2, temporal_mat1, temporal_mat2, mask1, mask2 = zip(*data)  # list of (seq_length, feat_dim)
    X1, batch_temporal_mat1, padding_masks1 = _inner_slove_data(
        features1, temporal_mat1, batch_size, max_len, vocab, mask1)
    X2, batch_temporal_mat2, padding_masks2 = _inner_slove_data(
        features2, temporal_mat2, batch_size, max_len, vocab, mask2)
    return X1.long(), X2.long(), padding_masks1, padding_masks2, \
           batch_temporal_mat1.long(), batch_temporal_mat2.long()


def _inner_slove_data(features, temporal_mat, batch_size, max_len, vocab=None, mask=None):
    # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
    lengths = [X.shape[0] for X in features]  # original sequence length for each time series
    if max_len is None:
        max_len = max(lengths)
    X = torch.zeros(batch_size, max_len, features[0].shape[-1], dtype=torch.long)  # (batch_size, padded_length, feat_dim)
    batch_temporal_mat = torch.zeros(batch_size, max_len, max_len,
                                     dtype=torch.long)  # (batch_size, padded_length, padded_length)

    target_masks = torch.zeros_like(X, dtype=torch.bool)  # (batch_size, padded_length, feat_dim)
    for i in range(batch_size):
        end = min(lengths[i], max_len)
        X[i, :end, :] = features[i][:end, :]
        batch_temporal_mat[i, :end, :end] = temporal_mat[i][:end, :end]
        if mask[i] is not None:
            target_masks[i, :end, :] = mask[i][:end, :]

    padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16), max_len=max_len)

    target_masks = ~target_masks  # (batch_size, padded_length, feat_dim)
    target_masks = target_masks * padding_masks.unsqueeze(-1)

    if mask[0] is not None:
        X[..., 0:1].masked_fill_(target_masks[..., 0:1] == 1, vocab.mask_index)  # loc -> mask_index
        X[..., 1:].masked_fill_(target_masks[..., 1:] == 1, vocab.pad_index)  # others -> pad_index
    return X, batch_temporal_mat, padding_masks



def padding_mask(lengths, max_len=None):
    batch_size = lengths.numel()
    max_len = max_len or lengths.max_val()  # trick works because of overloading of 'or' operator for non-boolean types
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


def noise_mask(X, masking_ratio, lm=3, mode='together', distribution='random', exclude_feats=None, add_cls=True):
    if exclude_feats is not None:
        exclude_feats = set(exclude_feats)

    if distribution == 'geometric':  # stateful (Markov chain)
        if mode == 'separate':  # each variable (feature) is independent
            mask = np.ones(X.shape, dtype=bool)
            for m in range(X.shape[1]):  # feature dimension
                if exclude_feats is None or m not in exclude_feats:
                    mask[:, m] = geom_noise_mask_single(X.shape[0], lm, masking_ratio)  # time dimension
        else:  # replicate across feature dimension (mask all variables at the same positions concurrently)
            mask = np.tile(np.expand_dims(geom_noise_mask_single(X.shape[0], lm, masking_ratio), 1), X.shape[1])
    elif distribution == 'random':  # each position is independent Bernoulli with p = 1 - masking_ratio
        if mode == 'separate':
            mask = np.random.choice(np.array([True, False]), size=X.shape, replace=True,
                                    p=(1 - masking_ratio, masking_ratio))
        else:
            mask = np.tile(np.random.choice(np.array([True, False]), size=(X.shape[0], 1), replace=True,
                                            p=(1 - masking_ratio, masking_ratio)), X.shape[1])
    else:
        mask = np.ones(X.shape, dtype=bool)
    if add_cls:
        mask[0] = True  # CLS at 0, set mask=1
    return mask


def collate_unsuperv_mask(data, max_len=None, vocab=None, add_cls=True):
    batch_size = len(data)
    features, masks, temporal_mat = zip(*data)  # list of (seq_length, feat_dim)

    # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
    lengths = [X.shape[0] for X in features]  # original sequence length for each time series
    if max_len is None:
        max_len = max(lengths)
    X = torch.zeros(batch_size, max_len, features[0].shape[-1], dtype=torch.long)  # (batch_size, padded_length, feat_dim)
    batch_temporal_mat = torch.zeros(batch_size, max_len, max_len,
                                     dtype=torch.long)  # (batch_size, padded_length, padded_length)

    # masks related to objective
    target_masks = torch.zeros_like(X, dtype=torch.bool)  # (batch_size, padded_length, feat_dim)
    for i in range(batch_size):
        end = min(lengths[i], max_len)
        X[i, :end, :] = features[i][:end, :]
        target_masks[i, :end, :] = masks[i][:end, :]
        batch_temporal_mat[i, :end, :end] = temporal_mat[i][:end, :end]

    padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16), max_len=max_len)

    target_masks = ~target_masks  # (batch_size, padded_length, feat_dim)
    target_masks = target_masks * padding_masks.unsqueeze(-1)

    targets = X.clone()
    targets = targets.masked_fill_(target_masks == 0, vocab.pad_index)

    X[..., 0:1].masked_fill_(target_masks[..., 0:1] == 1, vocab.mask_index)  # loc -> mask_index
    X[..., 1:].masked_fill_(target_masks[..., 1:] == 1, vocab.pad_index)  # others -> pad_index
    return X.long(), targets.long(), target_masks, padding_masks, batch_temporal_mat.long()


def geom_noise_mask_single(L, lm, masking_ratio):
    keep_mask = np.ones(L, dtype=bool)
    p_m = 1 / lm  # probability of each masking sequence stopping. parameter of geometric distribution.
    p_u = p_m * masking_ratio / (1 - masking_ratio)  # probability of each unmasked sequence stopping. parameter of geometric distribution.
    p = [p_m, p_u]

    # Start in state 0 with masking_ratio probability
    state = int(np.random.rand() > masking_ratio)  # state 0 means masking, 1 means not masking
    for i in range(L):
        keep_mask[i] = state  # here it happens that state and masking value corresponding to state are identical
        if np.random.rand() < p[state]:
            state = 1 - state

    return keep_mask
