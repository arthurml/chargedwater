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

x_numpy=data[:,0]
y_numpy=np.zeros([200,1])
y_numpy[:,0] = -data[:,1]
#%%
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)

x_train,x_teste,y_train,y_teste=train_test_split(x_numpy,y_numpy,test_size=0.3,shuffle=True,random_state=243)

scale_xtg=preprocessing.StandardScaler()
scale_xtg.fit(x_train.reshape(-1,1))
x_train=scale_xtg.transform(x_train.reshape(-1,1)).reshape(-1)
x_teste=scale_xtg.transform(x_teste.reshape(-1,1)).reshape(-1)

scale_ytg=preprocessing.StandardScaler()
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

plt.rcParams["font.size"] = 26
plt.rcParams["axes.labelsize"] = 28
plt.rcParams["xtick.labelsize"] = 26
plt.rcParams["ytick.labelsize"] = 26
plt.rcParams["legend.fontsize"] = 24
plt.rcParams["legend.framealpha"] = 0.0
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.serif'] = ['Arial']
n_size=1.3
plt.figure(figsize=(n_size*10,n_size*8))



#%%



model_tg.eval()
model_tg.likelihood.eval()
x=np.linspace(-3,3,100)
with torch.no_grad():
    observed_pred = model_tg.likelihood(model_tg(X_train))
    observed=observed_pred.mean.to('cpu').numpy()
    observed_un=observed_pred.variance.to('cpu').numpy()
    beforehand=Y_train.to('cpu').numpy()

observed_plot=observed#.reshape(-1,1).reshape(-1)
beforehand_plot=beforehand#.reshape(-1,1).reshape(-1)
observed_un_plot=observed_un#.reshape(-1,1).reshape(-1)

np.corrcoef(observed_plot,beforehand_plot)
x=np.arange( np.min(observed_plot)-5, np.max(observed_plot)+5,1)
plt.plot(scale_ytg.inverse_transform(observed_plot.reshape(-1,1)).reshape(-1),scale_ytg.inverse_transform(beforehand_plot.reshape(-1,1)).reshape(-1),'bo',markersize=12)
plt.plot(scale_ytg.inverse_transform(x.reshape(-1,1)).reshape(-1),scale_ytg.inverse_transform(x.reshape(-1,1)).reshape(-1),'b-.',linewidth=5)
plt.xlim((-15,50))
plt.ylim((-15,50))
plt.xlabel(r"MD prediction (kJ/mol)",fontweight='bold')
plt.ylabel(r"Surrogate Model (kJ/mol)",fontweight='bold')
plt.savefig('tg.jpg')