import os
import numpy as np
import scanpy as sc
import pandas as pd
import anndata as ad
import umap
from sclambda.model import *

'''
All dataset splitting functions are from or modified based on GEARS dataset spiltting
https://github.com/snap-stanford/GEARS
'''

def data_split(adata, 
               split_type = 'simulation',
               split_name = 'split',
               train_gene_set_size = 0.75, 
               combo_seen2_train_frac = 0.75,
               seed=0):

    np.random.seed(seed=seed)

    # Identify unique perturbations and genes
    unique_perts = [p for p in adata.obs['condition'].unique() if p != 'ctrl']

    if split_type == 'simulation':
        train, test, test_subgroup = get_simulation_split(unique_perts,
                                                          train_gene_set_size=train_gene_set_size,
                                                          combo_seen2_train_frac=combo_seen2_train_frac)
        train, val, val_subgroup = get_simulation_split(train,
                                                        0.9,
                                                        0.9)
        train = list(train)
        train.append('ctrl')
        map_dict = {x: 'train' for x in train}
        map_dict.update({x: 'val' for x in val})
        map_dict.update({x: 'test' for x in test})

    elif split_type == 'all_train':
        train = list(adata.obs['condition'].unique())
        map_dict = {x: 'train' for x in train}
        test_subgroup = {'combo_seen0': [],
                         'combo_seen1': [],
                         'combo_seen2': [],
                         'unseen_single': []}
        val_subgroup = test_subgroup.copy()

    elif split_type == 'single':
        train, test, test_subgroup = get_simulation_split_single(unique_perts,
                                                                 train_gene_set_size=train_gene_set_size)
        train, val, val_subgroup = get_simulation_split_single(train,
                                                               0.9)
        train = list(train)
        train.append('ctrl')
        map_dict = {x: 'train' for x in train}
        map_dict.update({x: 'val' for x in val})
        map_dict.update({x: 'test' for x in test})

    adata.obs[split_name] = adata.obs['condition'].map(map_dict)

    return adata, {'train': train,
                   'test_subgroup': test_subgroup, 
                   'val_subgroup': val_subgroup}

def get_simulation_split_single(unique_perts, 
                                train_gene_set_size):
    pert_train = []
    pert_test = []
    unique_pert_genes = np.unique([g for g in unique_perts if g != 'ctrl'])
    train_gene_candidates = np.random.choice(unique_pert_genes,
                            int(len(unique_pert_genes) * train_gene_set_size), replace = False)
    pert_single_train = train_gene_candidates
    unseen_single = np.setdiff1d(unique_pert_genes, train_gene_candidates)

    return pert_single_train, unseen_single, {'unseen_single': unseen_single}


def get_simulation_split(unique_perts, 
                         train_gene_set_size, 
                         combo_seen2_train_frac):
    pert_train = []
    pert_test = []
    gene_list = [p.split('+') for p in unique_perts]
    gene_list = [item for sublist in gene_list for item in sublist]
    unique_pert_genes = np.unique([g for g in gene_list if g != 'ctrl'])
    train_gene_candidates = np.random.choice(unique_pert_genes,
                            int(len(unique_pert_genes) * train_gene_set_size), replace = False)
    ood_genes = np.setdiff1d(unique_pert_genes, train_gene_candidates)
    # Sample single gene perturbations for training
    pert_single_train = np.intersect1d([(g+'+ctrl') for g in train_gene_candidates]+[('ctrl+'+g) for g in train_gene_candidates],
                        unique_perts)
    pert_train.extend(pert_single_train)
    # Combo perturbations for training with at least one seen gene
    pert_combo = [p for p in unique_perts if 'ctrl' not in p]
    # Combo sets with one of them in OOD should be in the testing set
    combo_seen1 = [x for x in pert_combo if len([t for t in x.split('+') if
                   t in train_gene_candidates]) == 1]
    pert_test.extend(combo_seen1)
    pert_combo = np.setdiff1d(pert_combo, combo_seen1)
    # Combo seen 0 in the testing set
    combo_seen0 = [x for x in pert_combo if len([t for t in x.split('+') if
                   t in (list(train_gene_candidates)+['ctrl'])]) == 0]
    pert_test.extend(combo_seen0)
    pert_combo = np.setdiff1d(pert_combo, combo_seen0)
    # Sample the combo seen 2 for training
    pert_combo_train = np.random.choice(pert_combo, int(len(pert_combo) * combo_seen2_train_frac), 
                       replace = False)
    combo_seen2 = np.setdiff1d(pert_combo, pert_combo_train).tolist()
    pert_test.extend(combo_seen2)
    pert_train.extend(pert_combo_train)
    # Unseen single in the testing set
    unseen_single = np.intersect1d([(g+'+ctrl') for g in ood_genes]+[('ctrl+'+g) for g in ood_genes],
                    unique_perts)
    pert_test.extend(unseen_single)
    
    return pert_train, pert_test, {'combo_seen0': combo_seen0,
                                   'combo_seen1': combo_seen1,
                                   'combo_seen2': combo_seen2,
                                   'unseen_single': list(unseen_single)}


def compute_umap(adata, rep=None, compute_s_umap=False):
    import umap

    reducer = umap.UMAP(n_neighbors=30,
                        n_components=2,
                        metric="correlation",
                        n_epochs=None,
                        learning_rate=1.0,
                        min_dist=0.3,
                        spread=1.0,
                        set_op_mix_ratio=1.0,
                        local_connectivity=1,
                        repulsion_strength=1,
                        negative_sample_rate=5,
                        a=None,
                        b=None,
                        random_state=1234,
                        metric_kwds=None,
                        angular_rp_forest=False,
                        verbose=True)
    if rep is None:
        X_umap = reducer.fit_transform(adata.X)
    else:
        X_umap = reducer.fit_transform(adata.obsm[rep])
    adata.obsm['X_umap'] = X_umap

    if compute_s_umap:
        embedding_s = reducer.fit_transform(adata.uns['emb_s'].values)
        adata.uns['s_umap'] = embedding_s


def gene2loc(genes, gene_name_vec, n_tgs):
    tg_loc_i = []
    if len(genes) > 1:
        for gene in genes:
            if gene in gene_name_vec:
                tg_loc_i.append(np.where(gene == gene_name_vec)[0][0])
            else:
                tg_loc_i.append(-1)
    elif genes[0] == 'ctrl':
        tg_loc_i = [-1] * n_tgs
    else:
        if genes[0] in gene_name_vec:
            tg_loc_i.append(np.where(genes[0] == gene_name_vec)[0][0])
        else:
            tg_loc_i.append(-1)
    return tg_loc_i


def adjust_tg(x_hat, x_hat_tg_ls, tg_loc, ctrl_mean_tensor):
    # adjust the predicted target gene values
    for j in range(tg_loc.shape[1]):
        x_hat_tg = x_hat_tg_ls[j]
        mask = tg_loc[:, j]>=0
        x_hat[torch.arange(x_hat.shape[0]).to(mask.device)[mask], tg_loc[:, j][mask]] = x_hat_tg[mask].reshape(-1) - ctrl_mean_tensor[tg_loc[:, j]][mask]
    return x_hat


def compute_corr_split(adata, seed=0):
    '''
    compute the replicability of perturbation effects by randomly splitting perturbed cells into two equal subsets,
    and computing the PCC between the average perturbation effects estimated from each subset.
    '''
    np.random.seed(seed)
    adata_ = adata.copy()
    ctrl_x = adata_[adata_.obs['condition'].values == 'ctrl'].X
    ctrl_mean = np.mean(ctrl_x, axis=0)
    adata_.X = adata_.X - ctrl_mean.reshape(1, -1)
    pert_ls = []
    corr_ls = []
    for i in np.unique(adata_.obs['condition'].values):
        if i != 'ctrl':
            adata_i = adata_[adata_.obs['condition'].values == i].copy()
            if adata_i.shape[0] > 1: # cannot be calculated with less then two cells
                idx = np.random.choice(adata_i.shape[0], int(adata_i.shape[0]/2), replace=False)
                delta_i_1 = np.array(np.mean(adata_i.X[idx], axis=0)).reshape(-1)
                delta_i_2 = np.array(np.mean(adata_i.X[~np.isin(np.arange(adata_i.shape[0]), idx)], axis=0)).reshape(-1)
                pert_ls.append(i)
                corr_ls.append(np.corrcoef(delta_i_1, delta_i_2)[0, 1])
    return pert_ls, corr_ls


def compute_reliability_score(adata,
                              pert_ls_unseen,
                              pert_embeddings,
                              pert_delta,
                              k=5):

    pert_ls_seen = np.setdiff1d(np.unique(adata.obs['condition'].values), (list(pert_ls_unseen) + ['ctrl']))
    pert_ls, corr_ls = compute_corr_split(adata)
    pcc_rep_df = pd.DataFrame(index=pert_ls)
    pcc_rep_df['PCC_rep'] = corr_ls
    
    from scipy.spatial.distance import cdist
    emb_df = pd.DataFrame(index=['dim_'+str(i) for i in range(len(pert_embeddings[list(pert_embeddings.keys())[0]]))])
    for i in list(pert_embeddings.keys()):
        if np.linalg.norm(pert_embeddings[i]) > 0:
            emb_df[i] = pert_embeddings[i]
    emb_df = emb_df.T

    emb_df_unseen = emb_df.loc[np.intersect1d(pert_ls_unseen, emb_df.index)]
    emb_df_seen = emb_df.loc[np.intersect1d(pert_ls_seen, emb_df.index)] # available observed perturbations

    D = cdist(emb_df_unseen.values, emb_df_seen.values, 'cosine') # cosine distance

    rs_ls = []
    rs2_ls = []

    for i in range(emb_df_unseen.shape[0]):
        distance = D[i]
        nearest_neighbor_indices = np.argsort(distance)[:k]

        delta_sum = np.zeros(adata.shape[1])
        rs2 = 0
        for n, j in enumerate(emb_df_seen.index[nearest_neighbor_indices]):
            delta_sum = delta_sum + pert_delta[j]
        for n, j in enumerate(emb_df_seen.index[nearest_neighbor_indices]):
            rs2 = rs2 + np.corrcoef(pert_delta[j], delta_sum - pert_delta[j])[0, 1]
        rs2 = rs2 / k
        rs_ls.append(np.mean((1-distance[nearest_neighbor_indices]) * pcc_rep_df.loc[emb_df_seen.index[nearest_neighbor_indices]].values))
        rs2_ls.append(rs2)

    rs_df = pd.DataFrame(index=emb_df_unseen.index)
    rs_df['rs_rep'] = rs_ls
    rs_df['rs_cons'] = rs2_ls
    return rs_df.sort_values(by='rs_rep', ascending=False)



