
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm

data = pd.read_csv(r'D:\Data science\MLR\50_Startups.csv')

data1=data.rename({'R&D':'RDS','Administration':'ADMS','Marketing':'MKTS'},axis=1)

plt.bar(height = data1.RDS, x = np.arange(1, 51, 1))
plt.hist(data1.RDS,) 
plt.boxplot(data1.RDS,)


plt.bar(height = data1.ADMS, x = np.arange(1, 51, 1))
plt.hist(data1.ADMS,) 
plt.boxplot(data1.ADMS,)

plt.bar(height = data1.MKTS, x = np.arange(1, 51, 1))
plt.hist(data1.MKTS,) 
plt.boxplot(data1.MKTS,)


sns.jointplot(x=data1['RDS'], y=data1['ADMS'])

from scipy import stats
import pylab
stats.probplot(data1.Profit, dist = "norm", plot = pylab)
plt.show()

sns.pairplot(data1.iloc[:, :])

data1.corr()


ml1 = smf.ols('Profit ~ RDS + ADMS + MKTS', data = data1).fit()
ml1.summary()

sm.graphics.influence_plot(ml1)

data2=data1.drop(data1.index[[49]],axis=0).reset_index(drop=True)
data2


ml_new = smf.ols('Profit ~ RDS + ADMS + MKTS', data = data2).fit()
ml_new.summary()


rsq_RDS = smf.ols('RDS ~ ADMS + MKTS', data = data2).fit().rsquared 
vif_RDS = 1/(1 - rsq_RDS)

rsq_ADMS = smf.ols('ADMS ~ RDS + MKTS', data = data2).fit().rsquared 
vif_ADMS = 1/(1 - rsq_ADMS)

rsq_MKTS = smf.ols('MKTS ~ RDS + ADMS', data = data2).fit().rsquared 
vif_MKTS = 1/(1 - rsq_MKTS)


d1 = {'Variables':['RDS', 'ADMS', 'MKTS'], 'VIF':[vif_RDS, vif_ADMS, vif_MKTS]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame


final_ml = smf.ols('Profit ~ RDS + ADMS + MKTS', data = data2).fit()
final_ml.summary()

pred = final_ml.predict(data1)


res = final_ml.resid
sm.qqplot(res)
plt.show()

stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

sns.residplot(x = pred, y = data1.Profit, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

sm.graphics.influence_plot(final_ml)

from sklearn.model_selection import train_test_split
data1_train, data1_test = train_test_split(data1, test_size = 0.2)

model_train = smf.ols('Profit ~ RDS + ADMS + MKTS', data = data2).fit()

test_pred = model_train.predict(data1_test)

test_resid = test_pred - data1_test.Profit

test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse


train_pred = model_train.predict(data1_train)

train_resid  = train_pred - data1_train.Profit

train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse





















