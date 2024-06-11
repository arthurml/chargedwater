import numpy as np
import torch
import gpytorch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


class GaussianProcessModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GaussianProcessModel, self).__init__(train_x, train_y, likelihood)
        #self.mean_module = gpytorch.means.LinearMean(input_size=2)
        self.mean_module =gpytorch.means.MultitaskMean(gpytorch.means.ConstantMean(), num_tasks=2)
        self.covar_module = gpytorch.kernels.MultitaskKernel(gpytorch.kernels.RBFKernel(),num_tasks=2, rank=1)
        self.double()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

data=np.loadtxt('data.dat')
np.shape(data)

x_numpy=data[:,0:2]
y_numpy=np.zeros([200,2])
y_numpy[:,0] = data[:,2]
y_numpy[:,1] = data[:,4]
counter = 0
#for i in range(200):
#    if i==np.shape(y_numpy)[0]:
#        break
#    if y_numpy[counter,0] > 150:
#        y_numpy = np.delete(y_numpy,counter,axis=0)
#        x_numpy = np.delete(x_numpy,counter,axis=0)
#        counter = counter-1
#    counter = counter+1
#y_numpy[:,1] = data[:,4]
#%%
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)

x_train,x_teste,y_train,y_teste=train_test_split(x_numpy,y_numpy,test_size=0.3,shuffle=True,random_state=243)

scale_xtg=preprocessing.StandardScaler()
scale_xtg.fit(x_train)
x_train=scale_xtg.transform(x_train)
x_teste=scale_xtg.transform(x_teste)

scale_ytg=preprocessing.StandardScaler()
scale_ytg.fit(y_train)
y_train=scale_ytg.transform(y_train)
y_teste=scale_ytg.transform(y_teste)


X_train=torch.from_numpy(x_train).to(device)
Y_train=torch.from_numpy(y_train).to(device)

X_teste=torch.from_numpy(x_teste).to(device)
Y_teste=torch.from_numpy(y_teste).to(device)

   
model_tg = GaussianProcessModel(X_train, Y_train, gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)).to(device)

# Find optimal hyperparameters
model_tg.train()
model_tg.likelihood.train()
optimizer = torch.optim.Adam(model_tg.parameters(), lr=0.1)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(model_tg.likelihood, model_tg)


num_iterations = 5000
from tqdm import tqdm
for i in tqdm(range(num_iterations)):
    optimizer.zero_grad()
    output = model_tg(X_train)
    loss = -mll(output, Y_train)
    loss.backward()
    optimizer.step()
print(f"final loss {loss:.4f}:")


#%%

import numpy as np
from matplotlib import pyplot as plt
import scipy.integrate
import matplotlib
from matplotlib import animation
import matplotlib.colors as mcolors

params = {
	'font.size': 22,
	'text.usetex': True,
	'lines.linewidth': 3.0,
	'lines.markersize': 4,
	'errorbar.capsize': 2,
	'legend.frameon': False,
	'savefig.format': 'png',
	'savefig.dpi': 500,
	'savefig.bbox': 'tight',
	'figure.subplot.hspace': 0.1,
}
plt.rcParams.update(params)

matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})
matplotlib.rcParams["font.size"] = 22
matplotlib.rcParams["axes.labelsize"] = 24
matplotlib.rcParams["xtick.labelsize"] = 22
matplotlib.rcParams["ytick.labelsize"] = 22
matplotlib.rcParams["legend.fontsize"] = 16

model_tg.eval()
model_tg.likelihood.eval()
x=np.zeros([16,2])
from scipy.stats import qmc
l_bounds = [3.5, 0.35]
u_bounds = [5.0, 0.5]
sampler = qmc.LatinHypercube(d=2)
sample = sampler.random(n=20000)
sample_scaled = qmc.scale(sample, l_bounds, u_bounds)
epsilon_chosen_1=3.5
epsilon_chosen_2=4.2734
epsilon_chosen_3=5.0
eps_const_1 = np.zeros([2000,2])
eps_const_2 = np.zeros([2000,2])
eps_const_3 = np.zeros([2000,2])
eps_const_1[:,0] = epsilon_chosen_1
eps_const_1[:,1] = np.linspace(0.35,0.5,num=2000)
eps_const_2[:,0] = epsilon_chosen_2
eps_const_2[:,1] = np.linspace(0.35,0.5,num=2000)
eps_const_3[:,0] = epsilon_chosen_3
eps_const_3[:,1] = np.linspace(0.35,0.5,num=2000)
sample_scaled=np.column_stack([eps_const_1])
x_otim_1 = scale_xtg.transform(sample_scaled)
X_otim_1 = torch.from_numpy(x_otim_1).to(device)
sample_scaled_2=np.column_stack([eps_const_2])
x_otim_2 = scale_xtg.transform(sample_scaled_2)
X_otim_2 = torch.from_numpy(x_otim_2).to(device)
sample_scaled_3=np.column_stack([eps_const_3])
x_otim_3 = scale_xtg.transform(sample_scaled_3)
X_otim_3 = torch.from_numpy(x_otim_3).to(device)

with torch.no_grad():
    observed_otim_1 = model_tg.likelihood(model_tg(X_otim_1))
    observed_teste_1 = observed_otim_1.mean.to('cpu').numpy()
    observed_otim_2 = model_tg.likelihood(model_tg(X_otim_2))
    observed_teste_2 = observed_otim_2.mean.to('cpu').numpy()
    observed_otim_3 = model_tg.likelihood(model_tg(X_otim_3))
    observed_teste_3 = observed_otim_3.mean.to('cpu').numpy()
    
x_otim_1= scale_xtg.inverse_transform(x_otim_1)
observed_teste_1 = scale_ytg.inverse_transform(observed_teste_1)
x_otim_2= scale_xtg.inverse_transform(x_otim_2)
observed_teste_2 = scale_ytg.inverse_transform(observed_teste_2)
x_otim_3= scale_xtg.inverse_transform(x_otim_3)
observed_teste_3 = scale_ytg.inverse_transform(observed_teste_3)

#for i in range(20000):
#    if np.abs((observed_teste[i,1]-78.5))<0.5:
#        print(observed_teste[i,:])
#        print(x_otim[i,:])
#        print(i)

plt.plot(x_otim_1[:,1],observed_teste_1[:,1], label=r'$\epsilon = 3.5 $')
plt.plot(x_otim_2[:,1],observed_teste_2[:,1], label=r'$\epsilon = 4.2734 $')
plt.plot(x_otim_3[:,1],observed_teste_3[:,1], label=r'$\epsilon = 5.0 $')
plt.legend()
plt.xlabel(r'$q$')
plt.ylabel(r'Dielectric Constant')
plt.savefig('dielecxq.png')
#plt.ylabel(r'Density ($kg/m^3$)')
#plt.savefig('densxq.png')



#%%
with torch.no_grad():
    observed_pred = model_tg.likelihood(model_tg(X_train))
    observed=observed_pred.mean.to('cpu').numpy()
    observed_un=observed_pred.variance.to('cpu').numpy()
    beforehand=Y_train.to('cpu').numpy()

observed_plot=observed#.reshape(-1,1).reshape(-1)
beforehand_plot=beforehand#.reshape(-1,1).reshape(-1)
observed_un_plot=observed_un#.reshape(-1,1).reshape(-1)

np.corrcoef(observed_plot,beforehand_plot)
x[:,0]=np.arange( np.min(observed_plot)-5, np.max(observed_plot)+5,1)
x[:,1] = x[:,0]
plt.plot(scale_ytg.inverse_transform(observed_plot)[:,0],scale_ytg.inverse_transform(beforehand_plot)[:,0],'bo',markersize=5)
plt.plot(scale_ytg.inverse_transform(x),scale_ytg.inverse_transform(x),'b-.',linewidth=2)
plt.xlim((950,1120))
plt.ylim((950,1120))
plt.xlabel(r"MD prediction")
plt.ylabel(r"Trained Model")
plt.savefig('tg_density.jpg')