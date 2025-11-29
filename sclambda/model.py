import os
import time
import numpy as np
import scanpy as sc
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import copy

from sclambda.networks import *
from sclambda.utils import *

class Model(object):
    def __init__(self, 
                 adata, # anndata object already splitted
                 gene_emb, # dictionary for gene embeddings
                 split_name = 'split',
                 latent_dim = 30, hidden_dim = 512,
                 training_epochs = 200,
                 batch_size = 500,
                 lambda_MI = 200,
                 sigma_x_sq = 1,
                 eps = 0.001,
                 seed = 1234,
                 model_path = "models",
                 multi_gene = True,
                 use_tg_coord = False,
                 ctrl_size = None, # reduce # control cells used for evaluation if there are too many control cells
                 ):

        # add device
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

        self.adata = adata.copy()
        self.gene_emb = gene_emb
        self.x_dim = adata.shape[1]
        self.p_dim = gene_emb[list(gene_emb.keys())[0]].shape[0]
        self.gene_emb.update({'ctrl': np.zeros(self.p_dim)})
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.training_epochs = training_epochs
        self.batch_size = batch_size
        self.lambda_MI = lambda_MI
        self.sigma_x_sq = sigma_x_sq
        self.eps = eps
        self.model_path = model_path
        self.multi_gene = multi_gene
        self.use_tg_coord = use_tg_coord
        self.ctrl_size = ctrl_size

        # compute perturbation embeddings
        print("Computing %s-dimentisonal perturbation embeddings for %s cells..." % (self.p_dim, adata.shape[0]))
        self.n_tgs = np.max([len(i.split('+')) for i in np.unique(adata.obs['condition'].values)])
        self.pert_emb_cells = np.zeros((adata.shape[0], self.p_dim))
        self.pert_emb = {}
        self.pert_emb_cells_tg = {i: np.zeros((adata.shape[0], self.p_dim)) for i in range(self.n_tgs)}
        for i in tqdm(np.unique(adata.obs['condition'].values)):
            genes = i.split('+')
            if len(genes) > 1:
                for j in range(len(genes)):
                    self.pert_emb_cells_tg[j][adata.obs['condition'].values == i] += self.gene_emb[genes[j]].reshape(1, -1)
                    if j == 0:
                        pert_emb_p = self.gene_emb[genes[0]]
                    else:
                        pert_emb_p = pert_emb_p + self.gene_emb[genes[j]]
            else:
                pert_emb_p = self.gene_emb[genes[0]]
            self.pert_emb_cells[adata.obs['condition'].values == i] += pert_emb_p.reshape(1, -1)
            self.pert_emb[i] = pert_emb_p
        self.adata.obsm['pert_emb'] = self.pert_emb_cells

        if self.use_tg_coord:
            print("Getting coordinates of target genes...")
            self.tg_loc_cells = np.zeros((adata.shape[0], self.n_tgs))
            self.tg_loc = {}
            for i in tqdm(np.unique(adata.obs['condition'].values)):
                genes = i.split('+')
                tg_loc_i = gene2loc(genes, self.adata.var.gene_name.values, self.n_tgs)
                self.tg_loc_cells[adata.obs['condition'].values == i] += np.array(tg_loc_i).reshape(1, -1)
                self.tg_loc[i] = tg_loc_i
            self.adata.obsm['tg_loc'] = np.int64(self.tg_loc_cells)

        # control cells
        ctrl_x = adata[adata.obs['condition'].values == 'ctrl'].X
        self.ctrl_mean = np.mean(ctrl_x, axis=0)
        self.ctrl_x = torch.from_numpy(ctrl_x - self.ctrl_mean.reshape(1, -1)).float().to(self.device)
        self.adata.X = self.adata.X - self.ctrl_mean.reshape(1, -1)
        if self.ctrl_size is not None:
            # subsample a group of control cells for validation if there are too many ctrl cells
            self.ctrl_idx = np.random.choice(self.ctrl_x.shape[0], self.ctrl_size, replace=False)
            self.ctrl_idx = torch.from_numpy(self.ctrl_idx).long().to(self.device)
            self.ctrl_x = self.ctrl_x[self.ctrl_idx]

        # split datasets
        print("Spliting data...")
        self.adata_train = self.adata[self.adata.obs[split_name].values == 'train']
        self.adata_val = self.adata[self.adata.obs[split_name].values == 'val']
        self.pert_val = np.unique(self.adata_val.obs['condition'].values)

        if self.use_tg_coord:
            self.ctrl_mean_tensor = torch.from_numpy(self.ctrl_mean).view(-1).float().to(self.device)
            self.train_data = PertDataset(torch.from_numpy(self.adata_train.X).float().to(self.device), 
                                          torch.from_numpy(self.adata_train.obsm['pert_emb']).float().to(self.device),
                                          torch.from_numpy(self.adata_train.obsm['tg_loc']).long().to(self.device),
                                          {i: torch.from_numpy(self.pert_emb_cells_tg[i]).float().to(self.device) for i in list(self.pert_emb_cells_tg.keys())})
        else:
            self.train_data = PertDataset(torch.from_numpy(self.adata_train.X).float().to(self.device), 
                                          torch.from_numpy(self.adata_train.obsm['pert_emb']).float().to(self.device))
        self.train_dataloader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

        self.pert_delta = {}
        for i in np.unique(self.adata.obs['condition'].values):
            adata_i = self.adata[self.adata.obs['condition'].values == i]
            delta_i = np.mean(adata_i.X, axis=0)
            self.pert_delta[i] = delta_i

    def loss_function(self, x, x_hat, p, p_hat, mean_z, log_var_z, s, s_marginal, T):
        reconstruction_loss = 0.5 / self.sigma_x_sq * torch.mean(torch.sum((x_hat - x)**2, axis=1)) + 0.5 * torch.mean(torch.sum((p_hat - p)**2, axis=1))
        KLD_z = - 0.5 * torch.mean(torch.sum(1 + log_var_z - mean_z**2 - log_var_z.exp(), axis=1))
        MI_latent = torch.mean(T(mean_z, s.detach())) - torch.log(torch.mean(torch.exp(T(mean_z, s_marginal.detach()))))
        return reconstruction_loss + KLD_z + self.lambda_MI * MI_latent

    def loss_recon(self, x, x_hat):
        reconstruction_loss = 0.5 * torch.mean(torch.sum((x_hat - x)**2, axis=1))
        return reconstruction_loss

    def loss_MINE(self, mean_z, s, s_marginal, T):
        MI_latent = torch.mean(T(mean_z, s)) - torch.log(torch.mean(torch.exp(T(mean_z, s_marginal))))
        return - MI_latent

    def train(self, retrain=False):
        if not retrain:
            self.Net = Net(x_dim = self.x_dim, p_dim = self.p_dim, 
                           latent_dim = self.latent_dim, hidden_dim = self.hidden_dim,
                           use_tg_coord = self.use_tg_coord)
        params = list(self.Net.Encoder_x.parameters())+list(self.Net.Encoder_p.parameters())+list(self.Net.Decoder_x.parameters())+list(self.Net.Decoder_p.parameters())
        optimizer = Adam(params, lr=0.0005)
        scheduler = StepLR(optimizer, step_size=30, gamma=0.2)
        optimizer_MINE = Adam(self.Net.MINE.parameters(), lr=0.0005, weight_decay=0.0001)
        scheduler_MINE = StepLR(optimizer_MINE, step_size=30, gamma=0.2)

        corr_val_best = 0
        if retrain:
            if len(self.pert_val) > 0: # If validating
                self.Net.eval()
                corr_ls = []
                for i in self.pert_val:
                    if self.multi_gene:
                        genes = i.split('+')
                        pert_emb_p = self.gene_emb[genes[0]] + self.gene_emb[genes[1]]
                        if self.use_tg_coord:
                            p_tg_ls = [self.gene_emb[gene] for gene in genes]
                    else:
                        pert_emb_p = self.gene_emb[i]
                        if self.use_tg_coord:
                            p_tg_ls = [self.gene_emb[i]]
                    val_p = torch.from_numpy(np.tile(pert_emb_p, 
                                                     (self.ctrl_x.shape[0], 1))).float().to(self.device)
                    genes = i.split('+')
                    tg_loc_i = gene2loc(genes, self.adata.var.gene_name.values, self.n_tgs)
                    tg_loc = torch.from_numpy(np.tile(np.array(tg_loc_i), 
                                                     (self.ctrl_x.shape[0], 1))).long().to(self.device)
                    if self.use_tg_coord:
                        p_tg = [torch.from_numpy(np.tile(p_tg_i, 
                                                     (self.ctrl_x.shape[0], 1))).float().to(self.device) for p_tg_i in p_tg_ls]
                        x_hat, x_hat_tg_ls, p_hat, mean_z, log_var_z, s = self.Net(self.ctrl_x, val_p, tg_loc.long(), p_tg)
                        x_hat = adjust_tg(x_hat, x_hat_tg_ls, tg_loc, self.ctrl_mean_tensor)
                    else:
                        x_hat, p_hat, mean_z, log_var_z, s = self.Net(self.ctrl_x, val_p)
                    x_hat = np.mean(x_hat.detach().cpu().numpy(), axis=0)
                    corr = np.corrcoef(x_hat, self.pert_delta[i])[0, 1]
                    corr_ls.append(corr)

                corr_val_best = np.mean(corr_ls)
                print("Previous best validation correlation delta %.5f" % corr_val_best)
        self.Net.train()
        for epoch in tqdm(range(self.training_epochs)):
            if self.use_tg_coord:
                for x, p, tg_loc, p_tg in self.train_dataloader:
                    # adversarial training on p
                    p.requires_grad = True 
                    self.Net.eval()
                    with torch.enable_grad():
                        x_hat, x_hat_tg_ls, _, _, _, _ = self.Net(x, p, tg_loc, p_tg)
                        recon_loss = self.loss_recon(x, x_hat)
                        grads = torch.autograd.grad(recon_loss, p)[0]
                        p_ae = p + self.eps * torch.norm(p, dim=1).view(-1, 1) * torch.sign(grads.data) # generate adversarial examples

                    self.Net.train()
                    x_hat, x_hat_tg_ls, p_hat, mean_z, log_var_z, s = self.Net(x, p_ae, tg_loc, p_tg)

                    # for MINE
                    index_marginal = np.random.choice(np.arange(len(self.train_data)), size=x_hat.shape[0])
                    p_marginal = self.train_data.p[index_marginal]
                    s_marginal = self.Net.Encoder_p(p_marginal)
                    for _ in range(1):
                        optimizer_MINE.zero_grad()
                        loss = self.loss_MINE(mean_z, s, s_marginal, T=self.Net.MINE)
                        loss.backward(retain_graph=True)
                        optimizer_MINE.step()

                    optimizer.zero_grad()
                    loss = self.loss_function(x, x_hat, p, p_hat, mean_z, log_var_z, s, s_marginal, T=self.Net.MINE)
                    for j in range(tg_loc.shape[1]):
                        x_hat_tg = x_hat_tg_ls[j]
                        mask = tg_loc[:, j]>=0
                        loss = loss + 10*torch.mean((x[torch.arange(x.shape[0]).to(mask.device)[mask], tg_loc[:, j][mask]] - (x_hat_tg[mask].reshape(-1) - self.ctrl_mean_tensor[tg_loc[:, j]][mask]))**2)
                    loss.backward()
                    optimizer.step()
                scheduler.step()
                scheduler_MINE.step()
            else:
                for x, p in self.train_dataloader:
                    # adversarial training on p
                    p.requires_grad = True 
                    self.Net.eval()
                    with torch.enable_grad():
                        x_hat, _, _, _, _ = self.Net(x, p)
                        recon_loss = self.loss_recon(x, x_hat)
                        grads = torch.autograd.grad(recon_loss, p)[0]
                        p_ae = p + self.eps * torch.norm(p, dim=1).view(-1, 1) * torch.sign(grads.data) # generate adversarial examples

                    self.Net.train()
                    x_hat, p_hat, mean_z, log_var_z, s = self.Net(x, p_ae)

                    # for MINE
                    index_marginal = np.random.choice(np.arange(len(self.train_data)), size=x_hat.shape[0])
                    p_marginal = self.train_data.p[index_marginal]
                    s_marginal = self.Net.Encoder_p(p_marginal)
                    for _ in range(1):
                        optimizer_MINE.zero_grad()
                        loss = self.loss_MINE(mean_z, s, s_marginal, T=self.Net.MINE)
                        loss.backward(retain_graph=True)
                        optimizer_MINE.step()

                    optimizer.zero_grad()
                    loss = self.loss_function(x, x_hat, p, p_hat, mean_z, log_var_z, s, s_marginal, T=self.Net.MINE)
                    loss.backward()
                    optimizer.step()
                scheduler.step()
                scheduler_MINE.step()
            if (epoch+1) % 10 == 0:
                print("\tEpoch", (epoch+1), "complete!", "\t Loss: ", loss.item())
                if len(self.pert_val) > 0: # If validating
                    self.Net.eval()
                    corr_ls = []
                    for i in self.pert_val:
                        if self.multi_gene:
                            genes = i.split('+')
                            pert_emb_p = self.gene_emb[genes[0]] + self.gene_emb[genes[1]]
                            if self.use_tg_coord:
                                p_tg_ls = [self.gene_emb[gene] for gene in genes]
                        else:
                            pert_emb_p = self.gene_emb[i]
                            if self.use_tg_coord:
                                p_tg_ls = [self.gene_emb[i]]
                        val_p = torch.from_numpy(np.tile(pert_emb_p, 
                                                         (self.ctrl_x.shape[0], 1))).float().to(self.device)
                        genes = i.split('+')
                        tg_loc_i = gene2loc(genes, self.adata.var.gene_name.values, self.n_tgs)
                        tg_loc = torch.from_numpy(np.tile(np.array(tg_loc_i), 
                                                         (self.ctrl_x.shape[0], 1))).long().to(self.device)
                        if self.use_tg_coord:
                            p_tg = [torch.from_numpy(np.tile(p_tg_i, 
                                                         (self.ctrl_x.shape[0], 1))).float().to(self.device) for p_tg_i in p_tg_ls]
                            x_hat, x_hat_tg_ls, p_hat, mean_z, log_var_z, s = self.Net(self.ctrl_x, val_p, tg_loc.long(), p_tg)
                            x_hat = adjust_tg(x_hat, x_hat_tg_ls, tg_loc, self.ctrl_mean_tensor)
                        else:
                            x_hat, p_hat, mean_z, log_var_z, s = self.Net(self.ctrl_x, val_p)
                        x_hat = np.mean(x_hat.detach().cpu().numpy(), axis=0)
                        corr = np.corrcoef(x_hat, self.pert_delta[i])[0, 1]
                        corr_ls.append(corr)
                    corr_val = np.mean(corr_ls)
                    print("Validation correlation delta %.5f" % corr_val)
                    if corr_val > corr_val_best:
                        corr_val_best = corr_val
                        self.model_best = copy.deepcopy(self.Net)
                    self.Net.train()
                else:
                    if epoch == (self.training_epochs-1):
                        self.model_best = copy.deepcopy(self.Net)
        print("Finish training.")
        self.Net = self.model_best
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        state = {'Net': self.Net.state_dict()}
        torch.save(state, os.path.join(self.model_path, "ckpt.pth"))

    def load_pretrain(self):
        self.Net = Net(x_dim = self.x_dim, p_dim = self.p_dim, 
                       latent_dim = self.latent_dim, hidden_dim = self.hidden_dim,
                       use_tg_coord = self.use_tg_coord)
        self.Net.load_state_dict(torch.load(os.path.join(self.model_path, "ckpt.pth"))['Net'])

    def predict(self, 
                pert_test, # perturbation or a list of perturbations
                ctrl_exp = None, # control cell gene expression matrix in np.array for in silico perturbation; use self.ctrl_x if None
                return_type = 'mean', # return mean or cells
                ):
        if ctrl_exp is not None:
            ctrl_exp = torch.from_numpy(ctrl_exp).float().to(self.device)
        else:
            ctrl_exp = self.ctrl_x
        self.Net.eval()
        res = {} 
        if isinstance(pert_test, str):
            pert_test = [pert_test]
        for i in pert_test:
            if self.multi_gene:
                genes = i.split('+')
                pert_emb_p = self.gene_emb[genes[0]] + self.gene_emb[genes[1]]
                if self.use_tg_coord:
                    p_tg_ls = [self.gene_emb[gene] for gene in genes]
            else:
                pert_emb_p = self.gene_emb[i]
                if self.use_tg_coord:
                    p_tg_ls = [self.gene_emb[i]]
            val_p = torch.from_numpy(np.tile(pert_emb_p, 
                                     (ctrl_exp.shape[0], 1))).float().to(self.device)
            if self.use_tg_coord:
                genes = i.split('+')
                tg_loc_i = gene2loc(genes, self.adata.var.gene_name.values, self.n_tgs)
                tg_loc = torch.from_numpy(np.tile(np.array(tg_loc_i), 
                                                 (ctrl_exp.shape[0], 1))).long().to(self.device)
                p_tg = [torch.from_numpy(np.tile(p_tg_i, 
                                             (ctrl_exp.shape[0], 1))).float().to(self.device) for p_tg_i in p_tg_ls]
                x_hat, x_hat_tg_ls, p_hat, mean_z, log_var_z, s = self.Net(ctrl_exp, val_p, tg_loc, p_tg)
            else:
                x_hat, p_hat, mean_z, log_var_z, s = self.Net(ctrl_exp, val_p)
            if return_type == 'cells':
                adata_pred = ad.AnnData(X=(x_hat.detach().cpu().numpy() + self.ctrl_mean.reshape(1, -1)))
                adata_pred.obs['condition'] = i
                res[i] = adata_pred
            elif return_type == 'mean':
                x_hat = np.mean(x_hat.detach().cpu().numpy(), axis=0) + self.ctrl_mean
                res[i] = x_hat
            else:
                raise ValueError("return_type can only be 'mean' or 'cells'.")
        return res

    def generate(self, 
                 pert_test, # perturbation or a list of perturbations
                 n_cells = 10000, # number of cells to generate
                 return_type = 'mean', # return mean or cells
                 ):
        self.Net.eval()
        res = {} 
        if isinstance(pert_test, str):
            pert_test = [pert_test]
        for i in pert_test:
            if self.multi_gene:
                genes = i.split('+')
                pert_emb_p = self.gene_emb[genes[0]] + self.gene_emb[genes[1]]
                if self.use_tg_coord:
                    p_tg_ls = [self.gene_emb[gene] for gene in genes]
            else:
                pert_emb_p = self.gene_emb[i]
                if self.use_tg_coord:
                    p_tg_ls = [self.gene_emb[i]]
            val_p = torch.from_numpy(np.tile(pert_emb_p, 
                                             (n_cells, 1))).float().to(self.device)
            s = self.Net.Encoder_p(val_p)
            z = torch.randn(n_cells, self.latent_dim).to(self.device)
            if self.use_tg_coord:
                genes = i.split('+')
                tg_loc_i = gene2loc(genes, self.adata.var.gene_name.values, self.n_tgs)
                tg_loc = torch.from_numpy(np.tile(np.array(tg_loc_i), 
                                                 (n_cells, 1))).long().to(self.device)
                p_tg = [torch.from_numpy(np.tile(p_tg_i, 
                                             (n_cells, 1))).float().to(self.device) for p_tg_i in p_tg_ls]
                x_hat_tg_ls = []
                for j in range(tg_loc.shape[1]):
                    s_tg = self.Net.Encoder_p(p_tg[j])
                    if j == 0:
                        x_hat, x_hat_tg = self.Net.Decoder_x(z+s, s_tg)
                    else:
                        _, x_hat_tg = self.Net.Decoder_x(z+s, s_tg)
                    x_hat_tg_ls.append(x_hat_tg)
                x_hat = adjust_tg(x_hat, x_hat_tg_ls, tg_loc, self.ctrl_mean_tensor)
            else:
                x_hat = self.Net.Decoder_x(z+s)
            if return_type == 'cells':
                res[i] = (x_hat.detach().cpu().numpy() + self.ctrl_mean.reshape(1, -1)).astype(np.float16)
            elif return_type == 'mean':
                x_hat = np.mean(x_hat.detach().cpu().numpy(), axis=0) + self.ctrl_mean
                res[i] = x_hat
            else:
                raise ValueError("return_type can only be 'mean' or 'cells'.")
        return res

    def get_embedding(self, adata=None):
        if adata == None:
            input_adata = None
            adata = self.adata
        x = torch.from_numpy(adata.X).float().to(self.device)
        p = torch.from_numpy(adata.obsm['pert_emb']).float().to(self.device)
        if self.use_tg_coord:
            tg_loc = torch.from_numpy(self.adata.obsm['tg_loc']).long().to(self.device)
            p_tg = [torch.from_numpy(self.pert_emb_cells_tg[i]).float().to(self.device) for i in list(self.pert_emb_cells_tg.keys())]
            for i in range(x.shape[0] // 1000 + 1): # use minibatch as this is more memory consuming
                _, _, _, mean_z_batch, _, s_batch = self.Net(x[i*1000:(i+1)*1000], 
                                                             p[i*1000:(i+1)*1000], 
                                                             tg_loc[i*1000:(i+1)*1000], 
                                                             [it[i*1000:(i+1)*1000] for it in p_tg])
                if i == 0:
                    mean_z = mean_z_batch.clone()
                    s  = s_batch.clone()
                else:
                    mean_z = torch.cat((mean_z, mean_z_batch), axis=0)
                    s = torch.cat((s, s_batch), axis=0)
        else:
            x_hat, p_hat, mean_z, log_var_z, s = self.Net(x, p)
        adata.obsm['mean_z'] = mean_z.detach().cpu().numpy()
        adata.obsm['z+s'] = adata.obsm['mean_z'] + s.detach().cpu().numpy()

        emb_s = pd.DataFrame(s.detach().cpu().numpy(), index=adata.obs['condition'].values)
        emb_s = emb_s.groupby(emb_s.index, axis=0).mean()
        adata.uns['emb_s'] = emb_s
        if input_adata is None:
            self.adata = adata
        return adata

class PertDataset(Dataset):
    def __init__(self, x, p, tg_loc=None, pert_emb_cells_tg=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.x = x
        self.p = p
        self.tg_loc = tg_loc
        self.pert_emb_cells_tg = pert_emb_cells_tg

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if self.tg_loc is None:
            return self.x[idx].to(self.device), self.p[idx].to(self.device)
        else:
            return self.x[idx].to(self.device), self.p[idx].to(self.device), self.tg_loc[idx].to(self.device), [self.pert_emb_cells_tg[i][idx].to(self.device) for i in list(self.pert_emb_cells_tg.keys())]

