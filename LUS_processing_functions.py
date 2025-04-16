import torch
import scipy.io as spio
from scipy.interpolate import UnivariateSpline
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.regression

class Lamb_modes_lookup():
    """
    Lookup class for lamb modes.
    """

    def __init__(self, filename):
        data_lookup = spio.loadmat(filename)
        nu_vec = data_lookup["nuVec"].squeeze()
        fS1S2 = data_lookup["fS1S2"].squeeze()
        fCL1 = data_lookup["fCL1"].squeeze()
        fCT1 = data_lookup["fCT1"].squeeze()
        fA2A3 = data_lookup["fA2A3"].squeeze()
        fA4A7 = data_lookup["fA4A7"].squeeze()
        fS3S6 = data_lookup["fS3S6"].squeeze()
        self.func_nu = UnivariateSpline((fCL1/fS1S2)[~np.isnan(fCL1/fS1S2)], nu_vec[~np.isnan(fCL1/fS1S2)])
        self.func_psiT_psiL = UnivariateSpline(nu_vec[~np.isnan(fCT1/fCL1)], (fCT1/fCL1)[~np.isnan(fCT1/fCL1)])
        self.func_psiA2A3_psiL = UnivariateSpline(nu_vec[~np.isnan(fA2A3/fCL1)], (fA2A3/fCL1)[~np.isnan(fA2A3/fCL1)])
        self.func_psiA4A7_psiL = UnivariateSpline(nu_vec[~np.isnan(fA4A7/fCL1)], (fA4A7/fCL1)[~np.isnan(fA4A7/fCL1)])
        self.func_psiS3S6_psiL = UnivariateSpline(nu_vec[~np.isnan(fS3S6/fCL1)], (fS3S6/fCL1)[~np.isnan(fS3S6/fCL1)])

    def lookup(self,psiS1S2,psiL):
        """
        Returns Poisson's ratio and lamb mode frequencies for given S1S2 and L resonance frequencies.

        TODO: Extension to other given frequency combinations.
        """
        nu = self.func_nu(psiL/psiS1S2)*1
        psiT = self.func_psiT_psiL(nu)*psiL
        psiA2A3 = self.func_psiA2A3_psiL(nu)*psiL
        psiA4A7 = self.func_psiA4A7_psiL(nu)*psiL
        psiS3S6 = self.func_psiS3S6_psiL(nu)*psiL
        return nu, psiT, psiA2A3, psiA4A7, psiS3S6

class LUS_estimator():
    """
    Object that implements procedures for estimating the parameters of the lamb modes.
    """
    def __init__(self, KL, KT, n0, N, device, filename_lookup, ZGV_modes = [True, False, False, False]): # ZGV_modes: S1S2, A2A3, A4A7, S3S6
        self.device = device
        self.Lookup = Lamb_modes_lookup(filename_lookup)
        self.ZGV_modes = ZGV_modes
        self.psiL = torch.tensor(0.,device=device,dtype=torch.float64)
        self.psiS1S2 = torch.tensor(0.,device=device,dtype=torch.float64)
        self.n = torch.arange(n0,N+n0,device=device,dtype=torch.float64).reshape(-1,1)
        if isinstance(KL,int):
            self.kL = torch.arange(1,KL+1,device=self.device,dtype=torch.float64)
        else:
            self.kL = torch.tensor(KL,device=self.device,dtype=torch.float64)
        if isinstance(KT,int):
            self.kT = torch.arange(1,KT+1,device=self.device,dtype=torch.float64)
        else:
            self.kT = torch.tensor(KT,device=self.device,dtype=torch.float64)
        self.r = 0.999*torch.ones(1,len(self.kL)+len(self.kT)+sum(ZGV_modes),device=device,dtype=torch.float64)
        self.inv_cov_mat = None
        
    def init_frequencies(self,x,plot=False):
        """
        Find initial estimates for the frequencies by performing a grid search. 

        Right now this only works with either L and all ZGV modes False ([False, False, False, False]), or with L and S1S2.
        For L and S1S2 True, model order selection (on harmonics of L) can be performed by providing list of models.
        """
        list_models = [
            # [1,2,3,4,5,6,7],
            # [1,2,3,4,5,6,7,8],
            # [1,2,3,4,5,6,7,8,9],
            [1,2,3,4,5,6,7,8,9,10],
            # [2,3,4,5,6,7,8,9,10,11,12,13,14],
        ]
        psiL_min = 0.004
        psiL_max = 0.006
        max_distance_psiL_psiS1S2 = 0.007
        if self.ZGV_modes[0]:
            psiL = np.linspace(psiL_min,psiL_max,200)
            psiS1S2 = np.linspace(psiL_min,psiL_max,100)
            psiL_mesh,psiS1S2_mesh = np.meshgrid(psiL,psiS1S2)
            cond = (psiL_mesh>psiS1S2_mesh) * (np.abs(psiL_mesh-psiS1S2_mesh)<max_distance_psiL_psiS1S2)#*(np.abs(psi0_mesh-psi_disturb_mesh)>0.001)#psi0_mesh!=psi_disturb_mesh#
            psiL_mesh_flat = psiL_mesh[cond]
            psiS1S2_mesh_flat = psiS1S2_mesh[cond]
            cf = np.empty((len(list_models),len(psiS1S2),len(psiL)))
            cf[:] = np.nan
            cf_max = np.zeros(len(list_models))
            for i in range(len(list_models)):
                cf[i,cond] = map_model_order(torch.tensor(x,device=self.device),psiL_mesh_flat,psiS1S2_mesh_flat,list_models[i]).cpu().numpy()
                cf_max[i] = np.nanmax(cf[i])
                # print("model ", i+1," / ", len(list_models), list_models[i])
            idx_model = np.argmax(cf_max)
            idx_cf_max = np.unravel_index(np.nanargmax(cf[idx_model]),cf[idx_model].shape)
            # psi0_hat_init = lus.parabola_interpolation(-cf.flatten()[idx_cf_max-1:idx_cf_max+2].reshape(1,-1), psi0_mesh.flatten()[idx_cf_max-1:idx_cf_max+2].reshape(-1,1)).squeeze()
            psiL_hat_init = parabola_interpolation(-cf[idx_model][idx_cf_max[0],idx_cf_max[1]-1:idx_cf_max[1]+2].reshape(1,-1), psiL_mesh[idx_cf_max[0],idx_cf_max[1]-1:idx_cf_max[1]+2].reshape(1,-1)).squeeze()
            psiS1S2_hat_init = parabola_interpolation(-cf[idx_model][idx_cf_max[0]-1:idx_cf_max[0]+2,idx_cf_max[1]].reshape(1,-1), psiS1S2_mesh[idx_cf_max[0]-1:idx_cf_max[0]+2,idx_cf_max[1]].reshape(-1,1)).squeeze()
            self.psiL = torch.tensor(psiL_hat_init,device=self.device,dtype=torch.float64)
            self.psiS1S2 = torch.tensor(psiS1S2_hat_init,device=self.device,dtype=torch.float64)
            if plot:
                plt.figure()
                plt.imshow(cf[idx_model])
        else: # TODO: model order selection (only first model considered in list_models)
            cf, psiL_grid = ANLS_pitch(x,[psiL_min,psiL_max],nfft=2**18,m=np.array(list_models[0]))
            idx_cf_max = np.argmax(cf)
            psiL_hat_init = parabola_interpolation(-cf[idx_cf_max-1:idx_cf_max+2].reshape(1,-1), psiL_grid[idx_cf_max-1:idx_cf_max+2].reshape(1,-1)).squeeze()
            self.psiL = torch.tensor(psiL_hat_init,device=self.device,dtype=torch.float64)
            if plot:
                plt.figure()
                plt.plot(psiL_grid,cf)

    def obs_mat(self,psiL,psiS1S2,r):
        """
        Returns observation matrix for the lamb modes resonances (including decay and power laws).
        """
        H = torch.zeros(len(self.n),4*len(self.kL)+2*len(self.kT)+2*sum(self.ZGV_modes),device=self.device,dtype=torch.float64)
        if self.ZGV_modes[0]:
            if self.device == torch.device(type="cpu"):
                _, psiT, psiA2A3, psiA4A7, psiS3S6 = self.Lookup.lookup(self.psiS1S2.numpy(),self.psiL.numpy())
            else:
                _, psiT, psiA2A3, psiA4A7, psiS3S6 = self.Lookup.lookup(self.psiS1S2.cpu().numpy(),self.psiL.cpu().numpy())
            psiT, psiA2A3, psiA4A7, psiS3S6 = torch.tensor(psiT,device=self.device), torch.tensor(psiA2A3,device=self.device), torch.tensor(psiA4A7,device=self.device), torch.tensor(psiS3S6,device=self.device)

        H[:,0:len(self.kL)] = self.n**(-1.)*r[:,0:len(self.kL)]**self.n*torch.cos(2*np.pi*psiL*self.n*self.kL)
        H[:,len(self.kL):2*len(self.kL)] = self.n**(-1.)*r[:,0:len(self.kL)]**self.n*torch.sin(2*np.pi*psiL*self.n*self.kL)
        H[:,2*len(self.kL):3*len(self.kL)] = self.n**(-1.5)*r[:,0:len(self.kL)]**self.n*torch.cos(2*np.pi*psiL*self.n*self.kL)
        H[:,3*len(self.kL):4*len(self.kL)] = self.n**(-1.5)*r[:,0:len(self.kL)]**self.n*torch.sin(2*np.pi*psiL*self.n*self.kL)
        if self.ZGV_modes[0]:
            H[:,4*len(self.kL):4*len(self.kL)+len(self.kT)] = self.n**(-1.5)*r[:,len(self.kL):len(self.kL)+len(self.kT)]**self.n*torch.cos(2*np.pi*psiT*self.n*self.kT)
            H[:,4*len(self.kL)+len(self.kT):4*len(self.kL)+2*len(self.kT)] = self.n**(-1.5)*r[:,len(self.kL):len(self.kL)+len(self.kT)]**self.n*torch.sin(2*np.pi*psiT*self.n*self.kT)
            H[:,4*len(self.kL)+2*len(self.kT):4*len(self.kL)+2*len(self.kT)+1] = self.n**(-0.5)*r[:,len(self.kL)+len(self.kT):len(self.kL)+len(self.kT)+1]**self.n*torch.cos(2*np.pi*psiS1S2*self.n)
            H[:,4*len(self.kL)+2*len(self.kT)+1:4*len(self.kL)+2*len(self.kT)+2] = self.n**(-0.5)*r[:,len(self.kL)+len(self.kT):len(self.kL)+len(self.kT)+1]**self.n*torch.sin(2*np.pi*psiS1S2*self.n)
        if self.ZGV_modes[1]:
            H[:,4*len(self.kL)+2*len(self.kT)+2:4*len(self.kL)+2*len(self.kT)+3] = self.n**(-0.5)*r[:,len(self.kL)+len(self.kT)+1:len(self.kL)+len(self.kT)+2]**self.n*torch.cos(2*np.pi*psiA2A3*self.n)
            H[:,4*len(self.kL)+2*len(self.kT)+3:4*len(self.kL)+2*len(self.kT)+4] = self.n**(-0.5)*r[:,len(self.kL)+len(self.kT)+1:len(self.kL)+len(self.kT)+2]**self.n*torch.sin(2*np.pi*psiA2A3*self.n)
        if self.ZGV_modes[2]:
            H[:,4*len(self.kL)+2*len(self.kT)+2+2*sum(self.ZGV_modes[1:2]):4*len(self.kL)+2*len(self.kT)+3+2*sum(self.ZGV_modes[1:2])] = self.n**(-0.5)*r[:,len(self.kL)+len(self.kT)+1+sum(self.ZGV_modes[1:2]):len(self.kL)+len(self.kT)+2+sum(self.ZGV_modes[1:2])]**self.n*torch.cos(2*np.pi*psiA4A7*self.n)
            H[:,4*len(self.kL)+2*len(self.kT)+3+2*sum(self.ZGV_modes[1:2]):4*len(self.kL)+2*len(self.kT)+4+2*sum(self.ZGV_modes[1:2])] = self.n**(-0.5)*r[:,len(self.kL)+len(self.kT)+1+sum(self.ZGV_modes[1:2]):len(self.kL)+len(self.kT)+2+sum(self.ZGV_modes[1:2])]**self.n*torch.sin(2*np.pi*psiA4A7*self.n)
        if self.ZGV_modes[3]:
            H[:,4*len(self.kL)+2*len(self.kT)+2+2*sum(self.ZGV_modes[1:3]):4*len(self.kL)+2*len(self.kT)+3+2*sum(self.ZGV_modes[1:3])] = self.n**(-0.5)*r[:,len(self.kL)+len(self.kT)+1+sum(self.ZGV_modes[1:3]):len(self.kL)+len(self.kT)+2+sum(self.ZGV_modes[1:3])]**self.n*torch.cos(2*np.pi*psiS3S6*self.n)
            H[:,4*len(self.kL)+2*len(self.kT)+3+2*sum(self.ZGV_modes[1:3]):4*len(self.kL)+2*len(self.kT)+4+2*sum(self.ZGV_modes[1:3])] = self.n**(-0.5)*r[:,len(self.kL)+len(self.kT)+1+sum(self.ZGV_modes[1:3]):len(self.kL)+len(self.kT)+2+sum(self.ZGV_modes[1:3])]**self.n*torch.sin(2*np.pi*psiS3S6*self.n)
        return H
    
    def linearized_obs_mat(self,x):
        """
        Returns linearized (w.r.t. decays and frequencies) observation matrix for the lamb modes resonances. The amplitudes are replaced by their least squares estimate (compressed least squares).
        """
        delta_psi = 1e-14
        delta_r = 1e-14
        C = torch.zeros(len(self.n),1+sum(self.ZGV_modes[0:1])+self.r.shape[1],device=self.device,dtype=torch.float64)
        for i in range(self.r.shape[1]):
            r_plus = self.r.clone()
            r_plus[0,i] += delta_r
            r_minus = self.r.clone()
            r_minus[0,i] -= delta_r
            C[:,i] = (self.est_signal(x,self.obs_mat(self.psiL,self.psiS1S2,r_plus))
                    -self.est_signal(x,self.obs_mat(self.psiL,self.psiS1S2,r_minus)))/(2*delta_r)
        C[:,self.r.shape[1]] = (self.est_signal(x,self.obs_mat(self.psiL+delta_psi,self.psiS1S2,self.r))
                            -self.est_signal(x,self.obs_mat(self.psiL-delta_psi,self.psiS1S2,self.r)))/(2*delta_psi)
        if self.ZGV_modes[0]:
            C[:,self.r.shape[1]+1] = (self.est_signal(x,self.obs_mat(self.psiL,self.psiS1S2+delta_psi,self.r))
                                -self.est_signal(x,self.obs_mat(self.psiL,self.psiS1S2-delta_psi,self.r)))/(2*delta_psi)
        return C

    def linearized_obs_mat_const_r(self,x):
        """
        Returns linearized (w.r.t. frequencies) observation matrix for the lamb modes resonances. The amplitudes are replaced by their least squares estimate (compressed least squares).
        """
        delta_psi = 1e-14
        C = torch.zeros(len(self.n),1+sum(self.ZGV_modes[0:1]),device=self.device,dtype=torch.float64)
        C[:,0] = (self.est_signal(x,self.obs_mat(self.psiL+delta_psi,self.psiS1S2,self.r))
                -self.est_signal(x,self.obs_mat(self.psiL-delta_psi,self.psiS1S2,self.r)))/(2*delta_psi)
        if self.ZGV_modes[0]:
            C[:,1] = (self.est_signal(x,self.obs_mat(self.psiL,self.psiS1S2+delta_psi,self.r))
                    -self.est_signal(x,self.obs_mat(self.psiL,self.psiS1S2-delta_psi,self.r)))/(2*delta_psi)
        return C

    def est_signal(self,x,H):
        """
        Returns least squares signal estimate for current nonlinear parameters (decays and frequencies).
        """
        return (H@self.est_lin_params(x,H)).squeeze()

    def est_lin_params(self,x,H):
        """
        Returns least squares linear parameter (amplitude) estimate for current nonlinear parameters (decays and frequencies).
        """
        if self.inv_cov_mat == None:
            HTHinv = torch.linalg.inv(torch.swapaxes(H,-1,-2)@H)
            return HTHinv@torch.swapaxes(H,-1,-2)@x.reshape(-1,1)
        else:
            HTCinvHinv = torch.linalg.inv(torch.swapaxes(H,-1,-2)@self.inv_cov_mat@H)
            return HTCinvHinv@torch.swapaxes(H,-1,-2)@(self.inv_cov_mat@x.reshape(-1,1))
    
    def est_AB(self,x):
        """
        Returns least squares linear parameter (A and B) estimate for current nonlinear parameters (decays and frequencies).
        """
        theta = self.est_lin_params(x,self.obs_mat(self.psiL,self.psiS1S2,self.r))
        A = torch.sqrt(theta[:len(self.kL)]**2+theta[len(self.kL):2*len(self.kL)]**2)
        B = torch.sqrt(theta[2*len(self.kL):3*len(self.kL)]**2+theta[3*len(self.kL):4*len(self.kL)]**2)
        return A,B
    
    def get_alpha(self):
        """
        Returns normalizing damping factors in units 1/s.
        """
        return -torch.log(self.r)/(2*torch.pi)

    def step_Gauss_Newton(self,x):
        """
        Performs a step of Gauss Newton optimization on the nonlinear parameters (decays and frequencies).
        """
        C = self.linearized_obs_mat(x)
        s_hat = self.est_signal(x,self.obs_mat(self.psiL,self.psiS1S2,self.r))
        beta = self.est_lin_params(x-s_hat,C).squeeze()
        self.r += beta[:self.r.shape[1]].reshape(self.r.shape)
        self.r[self.r>0.9999] = 0.9999
        self.r[self.r<0.9] = 0.9
        self.psiL += beta[self.r.shape[1]]
        if self.ZGV_modes[0]:
            self.psiS1S2 += beta[-1]

    def step_Gauss_Newton_const_r(self,x):
        """
        Performs a step of Gauss Newton optimization on parts of the nonlinear parameters (decays).
        """
        C = self.linearized_obs_mat(x)
        s_hat = self.est_signal(x,self.obs_mat(self.psiL,self.psiS1S2,self.r))
        beta = self.est_lin_params(x-s_hat,C).squeeze()
        self.psiL += beta[0]
        if self.ZGV_modes[0]:
            self.psiS1S2 += beta[1]


    def est_residual_cov_dft(self,x,window_size,p=15):
        """
        Estimate covariance matrix from residuals. Using the DFT method, which assumes local wide sense stationarity, due to lack of multiple realizations (to e.g., use the sample covariance matrix estimator).
        """
        s_hat = self.est_signal(x,self.obs_mat(self.psiL,self.psiS1S2,self.r))
        residual = (x-s_hat).cpu()
        inv_cov_mat = torch.zeros(len(residual),len(residual),device=self.device,dtype=torch.float64)
        self.Pxx = []
        for i in range(0,window_size*(len(residual)//window_size-1),window_size):
            inv_cov_mat[i:i+window_size,i:i+window_size], psda = est_inv_cov_dft_ar(residual[i:i+window_size],p)
            self.Pxx.append(psda)
        inv_cov_mat[window_size*(len(residual)//window_size-1):,window_size*(len(residual)//window_size-1):], psda = est_inv_cov_dft_ar(residual[window_size*(len(residual)//window_size-1):],p)
        self.Pxx.append(psda)
        self.inv_cov_mat = inv_cov_mat
        self.inv_cov_mat = self.inv_cov_mat.to(x.device)
        
def est_inv_cov_dft_ar(x,p):
    """
    Estimate covariance matrix using the DFT eigendecomposition (large and wide sense stationary x). PSD is estimated using yule-walker.
    """
    a, sigma = statsmodels.regression.linear_model.yule_walker(x.numpy(), order=p, method='mle', df=None, inv=False, demean=True)
    w = 2*np.pi*np.arange(len(x)).reshape(1,-1)/len(x)
    psda = sigma**2/(np.abs(1-np.sum(a.reshape(-1,1)*np.exp(-1j*np.arange(1,len(a)+1).reshape(-1,1)*w),axis=0))**2)
    dft_mat = torch.fft.fft(torch.eye(len(x)),norm="ortho")#
    return torch.real(torch.conj(dft_mat.T)@(dft_mat/torch.tensor(psda,dtype=torch.complex64).reshape(-1,1))), psda

    # def est_residual_cov(self,x,window_size):
    #     s_hat = self.est_signal(x,self.obs_mat(self.psiL,self.psiS1S2,self.r))
    #     residual = (x-s_hat).cpu()
    #     cov_mat = torch.zeros(len(residual),len(residual)+2*(window_size//2),device=self.device,dtype=torch.float64)
    #     for i in range(0,len(residual)-window_size+1):
    #         residual_windowed = residual[i:i+window_size]*torch.hann_window(window_size,dtype=torch.float64)
    #         cov_mat[i,i:i+window_size] = torch.tensor(np.correlate(residual_windowed,residual_windowed,mode="same"),device=self.device,dtype=torch.float64)#torch.hann_window(window_size,device=self.device,dtype=torch.float64)*
    #     for i in range(len(residual)-window_size+1,len(residual)):
    #         cov_mat[i,i:i+window_size] = cov_mat[len(residual)-window_size,len(residual)-window_size:len(residual)]
    #     self.inv_cov_mat = torch.inverse(cov_mat[:,window_size//2:-window_size//2+1])#+torch.eye(len(residual),device=self.device,dtype=torch.float64)

# def inverse_svd(y):
#     if y.dim()==0:
#         return 1/y
#     cutoff=1e-10
#     u,s,v = torch.linalg.svd(y)
#     s[s<cutoff] = 1e10
#     print(sum(s<cutoff))
#     return v@torch.diag(1/s)@u.T

def ANLS_pitch(x,limits,nfft=2**20,m=False):
    """
    Approximate nonlinear least squares pitch estimator (for a multiharmonic signal).
    """
    x_four = np.abs(np.fft.fft(x,nfft))**2
    psi_search = np.arange(limits[0],limits[1],1/nfft)
    cost_fun = np.zeros(len(psi_search))
    for i in range(len(psi_search)):
        if np.any(m)==False:
            m = np.round(1/(2*psi_search[i]))
            idcs_m = np.int32(nfft*((np.arange(1,m)+1)*psi_search[i]))
        else:
            m_temp = m[m*psi_search[i]<0.5]
            idcs_m = np.int32(nfft*(m_temp*psi_search[i]))
        cost_fun[i] = np.sum(x_four[idcs_m])
    return cost_fun,psi_search

def obs_matr_calibration(psi0,psi_disturb,N,k):
    """
    Observation matrix for a DC (constant), multiharmonic signal, and a single sinusoidal signal.
    """
    n_k = len(k)
    # dim: (psi0 and psi_disturb mesh, n, k)
    h = np.zeros((len(psi0),N,2*n_k+3))
    h[:,:,0] = 1 # DC
    n = np.arange(N).reshape(1,-1,1)
    k = np.array(k).reshape(1,1,-1)
    psi0 = psi0.reshape(-1,1,1)
    psi_disturb = psi_disturb.reshape(-1,1,1)
    h[:,:,1] = np.cos(2*np.pi*psi_disturb*n)[:,:,0]
    h[:,:,2] = np.sin(2*np.pi*psi_disturb*n)[:,:,0]
    h[:,:,3:3+n_k] = np.cos(2*np.pi*psi0*n*k)
    h[:,:,3+n_k:] = np.sin(2*np.pi*psi0*n*k)
    return h

def least_squares_cf(x,h):
    """
    Compressed least squares cost function.
    """
    xt = x.reshape(1,1,-1)
    h = torch.tensor(h,device=x.device,dtype=torch.float64)
    # ht = np.swapaxes(h,-1,-2)
    # hth_inv = np.linalg.inv(ht@h)
    # xth = xt@h
    # return (xth@hth_inv@np.swapaxes(xth,-1,-2)).squeeze()
    ht = torch.swapaxes(h,-1,-2)
    hth_inv = torch.linalg.inv(ht@h)
    xth = xt@h
    return (xth@hth_inv@torch.swapaxes(xth,-1,-2)).squeeze()

def map_model_order(x,psi0,psi_disturb,k):
    """
    MAP cost function for nonlinear parameters and model order selection.
    """
    result = []
    if isinstance(k[0], list):
        for ki in k:
            h = obs_matr_calibration(psi0,psi_disturb,len(x),ki)
            penalty = len(ki)/2*np.log(len(x))
            result.append(-len(x)/2*np.log((np.sum(x**2)-least_squares_cf(x,h)))-penalty)
        return result
    # penalty = len(k)/2*np.log(len(x))
    # h = obs_matr_calibration(psi0,psi_disturb,len(x),k)
    # return -len(x)/2*np.log(np.sum(x**2)-least_squares_cf(x,h))-penalty
    penalty = len(k)/2*torch.log(torch.tensor(len(x)))
    h = obs_matr_calibration(psi0,psi_disturb,len(x),k)
    return -len(x)/2*torch.log(torch.sum(x**2)-least_squares_cf(x,h))-penalty

def parabola_interpolation(y, grid):
    """
    Parabola interpolation to find minimum with subgrid accuracy.
    """
    
    # index of minima
    idx_min_ = np.argmin(y, axis=1, keepdims=True)

    # check index boundaries and add +-1 indices
    idx_min = np.maximum(np.ones(idx_min_.shape,dtype=np.int32), idx_min_)
    idx_min = np.minimum((y.shape[1]-2)*np.ones(idx_min_.shape,dtype=np.int32), idx_min)
    # idx_min = idx_min.int()
    idx_min = idx_min + np.arange(-1,2,1).reshape(1,-1)

    # x values
    if grid.shape[1]>1:
        x = grid[np.arange(y.shape[0]).reshape(-1,1),idx_min]
    else:
        x = grid[idx_min]
    x = x.reshape(y.shape[0],-1,1)**np.tile(np.arange(2,-1,-1).reshape(1,3),(3,1))

    # y values
    # print(torch.arange(y.shape[0]).reshape(-1,1),idx_max,y.shape[0])
    y_min = y[np.arange(y.shape[0]).reshape(-1,1),idx_min].reshape(y.shape[0],-1,1)

    # calc parabola coeffs and minima
    coeffs = np.linalg.inv(x)@y_min
    minima = -coeffs[:,1]/coeffs[:,0]/2

    # remove outliers (prabolas with negative curvature)
    minima[coeffs[:,0] < 0] = np.array(idx_min_[coeffs[:,0] < 0],dtype=minima.dtype)
    if grid.shape[1]>1:
        minima[minima > grid[:,-1].reshape(-1,1)] = grid[:,-1].reshape(-1,1)[minima > grid[:,-1].reshape(-1,1)]
        minima[minima < grid[:,0].reshape(-1,1)] = grid[:,0].reshape(-1,1)[minima < grid[:,0].reshape(-1,1)]
    else:
        minima[minima > grid[-1,0]] = grid[-1,0]
        minima[minima < grid[0,0]] = grid[0,0]

    return minima