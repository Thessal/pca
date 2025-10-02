import pandas as pd 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MinVar(nn.Module):
    def __init__(self, cov, gross_exposure_constraint, use_latent=False, cuda=True):
        super().__init__()
        self.enable_cuda = cuda
        self.cov = cov.cuda() if cuda else cov
        num_instrument = cov.shape[0]
        self.use_latent = use_latent
        if use_latent:
            self.latent = nn.Linear(in_features=30, out_features=num_instrument, bias=False) # A
            self.fc = nn.Linear(in_features=30, out_features=1, bias=False) # w_e
        else:
            self.fc = nn.Linear(in_features=num_instrument, out_features=1, bias=False) # w_e
        self.fc.weight.data = (torch.rand_like(self.fc.weight) - 0.5) / num_instrument 
        self.gross_exp_constr = gross_exposure_constraint

        eigenvalues, eigenvectors = torch.linalg.eigh(self.cov)
        sorted_indices = torch.argsort(eigenvalues, descending=True)
        self.eigenvalues = eigenvalues[sorted_indices]
        self.eigenvectors = eigenvectors[:, sorted_indices]

    @property
    def portfolio_weight(self): # w_p = v * A * w_e
        if self.use_latent:
            weight_in_eigenvalue_space = self.latent(self.fc.weight).T
        weight_in_eigenvalue_space = self.fc.weight.T
        weight_in_instrument_space = self.eigenvectors @ weight_in_eigenvalue_space
        norm_coef = self.normalize_coeff(weight_in_instrument_space)
        return weight_in_eigenvalue_space / norm_coef, weight_in_instrument_space / norm_coef
    
    def get_weight(self):
        return self.portfolio_weight[1].cpu().detach().numpy()

    @staticmethod
    def normalize_coeff(x):
        # 1x Long
        return x.sum()
    
    def _normalize_weight(self):
        # Prevents divergence
        if self.use_latent:
            self.latent.weight.data = self.latent.weight / torch.linalg.matrix_norm(self.latent.weight, ord='fro')
        # self.fc.weight.data = self.fc.weight / torch.linalg.vector_norm(self.fc.weight, ord=2)
        self.fc.weight.data = self.fc.weight / self.normalize_coeff(self.fc.weight)

    def fit(self, n_epochs=10000, print_step=1000, learning_rate=1e-4):
        if self.enable_cuda:
            self.cuda()
        optimizer = torch.optim.Adagrad(self.parameters(), lr=learning_rate) # Momentum-less optimizer for barrier method
        for e in range(n_epochs):
            w_e, w_i = self.portfolio_weight
            gross_exposure = torch.linalg.vector_norm(w_i, ord=1)
            portfolio_variance = self.eigenvalues @ (w_e.square())
            barrier_loss = torch.relu(gross_exposure - self.gross_exp_constr)
            loss = torch.log(portfolio_variance) + barrier_loss

            loss.backward()
            optimizer.step()
            # self._normalize_weight()
            print(f'{str(e).zfill(5)}/{n_epochs}, variance={portfolio_variance.item():.2e}, gross_exposure = {gross_exposure.item():.2e}, barrier_loss={barrier_loss:.2e}, loss={loss.item():.2e}, normalization={torch.sum(self.fc.weight):.2e}     ', end='\r', flush=True)
            if print_step and e%print_step == 0 :
                print()
        print("\nTrain Finished")

class Covariance:
    def __init__(self, df_logret, lookback):
        self.df = df_logret # columns are multiindexed : sector id, instrument id
        self.lookback = lookback

    def _split_data(self, date):
        history = self.df.loc[:date].iloc[-1-self.lookback:-1]
        next_logret = df.loc[:date].iloc[-1]
        return history, next_logret

    def covar_sample(self, date):
        history, next_logret = self._split_data(date)
        cov = torch.Tensor(history.values).cuda().T.cov()
        return cov, next_logret

    def covar_thresholded(self,date,h):
        history, next_logret = self._split_data(date)
        n,p = history.shape
        threshold = h * np.sqrt(np.log(p) / n)
        cov = torch.Tensor(history.values).cuda().T.cov()
        cov = torch.maximum(cov - threshold,torch.tensor(0))
        return cov, next_logret
    
    def calculate_block_diagonal(self, gics_map):
        df_block_mask = pd.DataFrame().reindex(index=self.df.columns, columns=self.df.columns).fillna(0)
        for g in set(gics_map.values()):
            df_block_mask.loc[g,g] = 1
        return df_block_mask.values
    
    def covar_poet(self, date, sector_block_diagonal_mask):
        history, next_logret = self._split_data(date)
        cov = torch.Tensor(history.values).cuda().T.cov()
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        sorted_indices = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        rmax = 30
        c1 = 0.02 * eigenvalues[rmax-1]
        c2 = 0.5
        n,p = history.shape
        j = torch.arange(rmax).cuda()+1
        l = eigenvalues[:rmax]
        x = (l / p + j * c1 * (np.sqrt(np.log(p)/n)+np.log(p)/p)**c2 )
        rank = torch.argmin(x).cpu().item()

        dense_cov = sum( eigenvalues[i] * eigenvectors[:,[i]] @ eigenvectors[:,[i]].T for i in range(rank)).cpu().numpy()
        resid = (cov.cpu().numpy() - dense_cov) * sector_block_diagonal_mask
        return dense_cov + resid, next_logret
    
    
if __name__ == "__main__":
    # Sector group
    gics_map = pd.read_csv("GICS_code.csv", dtype=str).set_index("Stock Code")["GICS Code"].to_dict()
    gics_map = { k.zfill(6): v for k, v in gics_map.items()}

    # Price
    df = pd.read_csv("close.csv", index_col="Date", parse_dates=True, dtype=float)
    assert(df.isna().any().any()==False)
    df.columns = pd.MultiIndex.from_arrays([df.columns.map(gics_map), df.columns], names=["sector", "code"])
    df = df.sort_index(axis=1)
    df = df.T.drop_duplicates().T
    df = df.apply(np.log).diff().dropna()

    # Risk free rate
    df_cd91_yearly = pd.read_csv("cd91.csv", index_col=0, parse_dates=True)["value"] / 100
    df_rf = df_cd91_yearly / 252
    df_rf = df_rf.reindex_like(df)
    df = df.subtract(df_rf, axis=0).dropna()

    for exposure in [1, 1.5, 2.0, 2.5, 3.0]:
        for lookback, dates in [ (252 * 4, df.loc["2019":].index), (252, df.loc["2016":].index), ]:
            c = Covariance(df, lookback) 
            mask = c.calculate_block_diagonal(gics_map)
            for date in dates:
                try:
                    cov, next_ret = c.covar_poet(date, mask)
                    cov = torch.Tensor(cov).cuda()
                    portfolio = MinVar(cov, exposure, use_latent=False)
                    portfolio.fit(learning_rate=1e-4, n_epochs=2000)
                    np.save("weights_poet/"+str(lookback)+"_"+str(exposure)+"_"+date.strftime("%Y%m%d")+'_weight.npy', portfolio.get_weight())
                except KeyboardInterrupt:
                    print ('KeyboardInterrupt exception is caught')
                    raise ValueError
                except Exception as e:
                    print(e)
                    pass