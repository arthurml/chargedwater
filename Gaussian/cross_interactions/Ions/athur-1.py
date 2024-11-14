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
        self.mean_module =gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

data=np.loadtxt('results.dat')
np.shape(data)

x_numpy=data[:,0:2]
y_numpy=np.zeros([95,1])

y_numpy[:,0] = -data[:,3]
#%%
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)

x_train,x_teste,y_train,y_teste=train_test_split(x_numpy,y_numpy,test_size=0.3,shuffle=True,random_state=42)

scale_xtg=preprocessing.MinMaxScaler()
scale_xtg.fit(x_train)
x_train=scale_xtg.transform(x_train)
x_teste=scale_xtg.transform(x_teste)

scale_ytg=preprocessing.MinMaxScaler()
scale_ytg.fit(y_train.reshape(-1,1))
y_train=scale_ytg.transform(y_train.reshape(-1,1)).reshape(-1)
y_teste=scale_ytg.transform(y_teste.reshape(-1,1)).reshape(-1)


X_train=torch.from_numpy(x_train).to(device)
Y_train=torch.from_numpy(y_train).to(device)

X_teste=torch.from_numpy(x_teste).to(device)
Y_teste=torch.from_numpy(y_teste).to(device)

   
model_tg = GaussianProcessModel(X_train, Y_train, gpytorch.likelihoods.GaussianLikelihood()).to(device)

# Find optimal hyperparameters
model_tg.train()
model_tg.likelihood.train()
optimizer = torch.optim.Adam(model_tg.parameters(), lr=0.01)
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
	'lines.markersize': 2,
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
matplotlib.rcParams["axes.labelsize"] = 26
matplotlib.rcParams["xtick.labelsize"] = 22
matplotlib.rcParams["ytick.labelsize"] = 22
matplotlib.rcParams["legend.fontsize"] = 26
n_size=1.3
plt.figure(figsize=(n_size*10,n_size*8))

eps_1 = 2.0
eps_2 = 4.0
eps_3 = 6.0

sig = np.linspace(0.250,0.500,num=200)

values_1 = np.zeros([200,2])
values_2 = np.zeros([200,2])
values_3 = np.zeros([200,2])

values_1[:,1] = eps_1
values_1[:,0] = sig
values_2[:,1] = eps_2
values_2[:,0] = sig
values_3[:,1] = eps_3
values_3[:,0] = sig


model_tg.eval()
model_tg.likelihood.eval()
x_1 = scale_xtg.transform(values_1)
X_1 = torch.from_numpy(x_1).to(device)
x_2 = scale_xtg.transform(values_2)
X_2 = torch.from_numpy(x_2).to(device)
x_3 = scale_xtg.transform(values_3)
X_3 = torch.from_numpy(x_3).to(device)

with torch.no_grad():
    observed_1 = model_tg.likelihood(model_tg(X_1))
    observed_1= observed_1.mean.to('cpu').numpy()
    observed_3 = model_tg.likelihood(model_tg(X_3))
    observed_3= observed_3.mean.to('cpu').numpy()
    observed_2 = model_tg.likelihood(model_tg(X_2))
    observed_2= observed_2.mean.to('cpu').numpy()

x_1 = scale_xtg.inverse_transform(x_1)
observed_1 = -scale_ytg.inverse_transform(observed_1.reshape(-1,1)).reshape(-1)
x_2 = scale_xtg.inverse_transform(x_2)
observed_2 = -scale_ytg.inverse_transform(observed_2.reshape(-1,1)).reshape(-1)
x_3 = scale_xtg.inverse_transform(x_3)
observed_3 = -scale_ytg.inverse_transform(observed_3.reshape(-1,1)).reshape(-1)


plt.plot(x_1[:,0],observed_1, label=r'$\epsilon$ = 2.0')
plt.plot(x_2[:,0],observed_2, label=r'$\epsilon$ = 4.0')
plt.plot(x_3[:,0],observed_3, label=r'$\epsilon$ = 6.0')
plt.legend()
plt.xlabel(r'$\sigma$ (nm)')
plt.ylabel(r'Hydration Free Energy (kJ/mol)')
plt.savefig('hyd_ion.jpg', dpi=500)



#%%
model_tg.eval()
model_tg.likelihood.eval()
x=np.linspace(-3,3,100)
with torch.no_grad():
    observed_pred = model_tg.likelihood(model_tg(X_train))
    observed=observed_pred.mean.to('cpu').numpy()
    observed_un=observed_pred.variance.to('cpu').numpy()
    beforehand=Y_train.to('cpu').numpy()

matplotlib.rcParams["axes.labelsize"] = 20
observed_plot=observed#.reshape(-1,1).reshape(-1)
beforehand_plot=beforehand#.reshape(-1,1).reshape(-1)
observed_un_plot=observed_un#.reshape(-1,1).reshape(-1)

np.corrcoef(observed_plot,beforehand_plot)
x=np.arange( np.min(observed_plot)-5, np.max(observed_plot)+5,1)
plt.plot(-scale_ytg.inverse_transform(observed_plot.reshape(-1,1)).reshape(-1),-scale_ytg.inverse_transform(beforehand_plot.reshape(-1,1)).reshape(-1),'bo',markersize=12)
plt.plot(scale_ytg.inverse_transform(x.reshape(-1,1)).reshape(-1),scale_ytg.inverse_transform(x.reshape(-1,1)).reshape(-1),'b-.',linewidth=5)
plt.xlim((-300,-50))
plt.ylim((-300,-50))
plt.xlabel(r"MD prediction (kJ/mol)",fontweight='bold')
plt.ylabel(r"Trained Model (kJ/mol)",fontweight='bold')
plt.savefig('tg_train.jpg')