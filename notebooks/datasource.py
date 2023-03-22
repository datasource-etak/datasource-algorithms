import os
import requests
import json
import base64
from textwrap import wrap
from tabulate import tabulate
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Patch, Ellipse
from matplotlib import cm, rc_file_defaults
import pylab
import seaborn as sns
import graphviz
import pydot

import statsmodels as sm
from linearmodels import PooledOLS, PanelOLS, RandomEffects, FamaMacBeth
import statsmodels.api as sma
import statsmodels.formula.api as smf
import lightgbm as lgb

from matplotlib.font_manager import FontProperties

emoji_prop = FontProperties(fname = "AppleColorEmoji.ttc")

from scipy.stats import skew, kurtosis, pearsonr, gaussian_kde, t # , lognorm, norm
from scipy.spatial import distance

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LassoLarsIC
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.svm import SVR, NuSVR
from sklearn.neural_network import MLPRegressor
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
from xgboost import XGBRegressor

from sklearn.cluster import DBSCAN, KMeans, MiniBatchKMeans, AffinityPropagation
from sklearn.decomposition import PCA, KernelPCA, FastICA, FactorAnalysis, TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation as LDiA
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.manifold import MDS, TSNE, SpectralEmbedding, Isomap
from sklearn.manifold import LocallyLinearEmbedding as LLE
from sklearn.random_projection import GaussianRandomProjection

from scipy.cluster.hierarchy import dendrogram, linkage, ward, fcluster
from scipy.spatial import distance
from scipy.stats import norm


default_colors = ['#008fd5', '#fc4f30', '#e5ae38', '#6d904f', '#8b8b8b', '#810f7c']

URL = "snf-34626.ok-kno.grnetcloud.net"
class DatasourceDataClient:
  def __init__(self, username, password):
    self.__username = username
    self.__password = password
    self.__base_endpoint = "https://" + URL + ":9999/api"
    self.__headers = {"Content-Type" : "application/json"}
    self._getToken()

  def _getToken(self):
    user_login_params = {"username": self.__username, "password" : self.__password}
    user_signin_response = requests.post(url = self.__base_endpoint  + "/users/signin", json = user_login_params, headers = self.__headers)
    if user_signin_response.status_code == 200:
      token = user_signin_response.json()["access_token"]
      self.__headers["Authorization"] = "bearer " + token
      print("Successfully Authenticated")
    else:
      print("Could not authenticate. Please re-initialize object to try again")

  def showDownloadedData(self):
    response = requests.get(url = self.__base_endpoint + "/datastore/" + self.__username + "/dtable?tableName=dataset_history_store&filters=", headers = self.__headers)
    data = []
    tuplelist = response.json()
    for i in range(len(tuplelist)):
      row = tuplelist[i]["tuple"]
      rowjson = {}
      for j in range(len(row)):
        if (row[j]["key"] == "dataset_description"):
          rowjson[row[j]["key"]] = row[j]["value"]#"\r\n".join(wrap(row[j]["value"], 70))
        elif (row[j]["key"] == "associated_filter"):   
          filters = row[j]["value"]
          filters = json.loads(base64.b64decode(filters))
          rowjson["associated_filters"] = ""
          for i in range(len(filters)):
            afilter = filters[i]
            rowjson["associated_filters"] += afilter["name"] + " = " + afilter["values"][0]["description"] + " "
        elif (row[j]["key"] == "dataset_name") or (row[j]["key"] == "source_name"):
          rowjson[row[j]["key"]] = row[j]["value"]#"\r\n".join(wrap(row[j]["value"], 17))
        else:
          rowjson[row[j]["key"]] = row[j]["value"]

      data.append(rowjson)
    uuids = []
    dataset_ids = []
    if len(data) > 0:
      data = pd.DataFrame.from_records(data)
      uuids = data["uuid"].tolist()
      dataset_ids = data["dataset_id"].tolist()
      return data
    else:
      return None

  def getDatasetFromUUID(self, uuid):
    filter_parameters = "message_type:" + uuid
    params = {
	'filters' : filter_parameters
	}
    response = requests.get(self.__base_endpoint + '/datastore/' + self.__username + '/select', params=params, headers=self.__headers)
    data = response.json()
    datalist = []
    for i in range(len(data)):
      row = data[i]["tuple"]
      rowjson = {}
      payload = {}
      value_column_name = ""
      for j in range(len(row)):
        key = row[j]["key"]
        value = row[j]["value"]
        if key != "message_type" and key != "scn_slug" and key != "source_id":
          if key == "dataset_id":
            value_column_name = value
          elif key == "payload":
            payload = json.loads(value)
          else:
            rowjson[key] = value
              
      renamings = []
      for akey in rowjson.keys():
        if akey != "time" and akey != "value":
          if akey in payload:
            newkeyname = payload[akey][0]
            value = str(payload[rowjson[akey]])
            renamings.append((akey, newkeyname, value))
      
      for key, newkey, val in renamings:
        rowjson[newkeyname] = val
        del rowjson[key]
      
      rowjson[value_column_name] = rowjson["value"]
      del rowjson["value"]

      datalist.append(rowjson)
    
    return pd.DataFrame(datalist).sort_values(by=['time']).reset_index(drop=True)
  
  def getDatasetFamily(self, did):

    filter_parameters = "dataset_id:" + did
    params = {
  	'filters' : filter_parameters
  	}
    response = requests.get(self.__base_endpoint + '/datastore/' + self.__username + '/select', params=params, headers=self.__headers)
    data = response.json()
    datalist = []
    for i in range(len(data)):
      row = data[i]["tuple"]
      rowjson = {}
      payload = {}
      value_column_name = ""
      for j in range(len(row)):
        key = row[j]["key"]
        value = row[j]["value"]

        if key != "message_type" and key != "scn_slug" and key != "source_id":
          if key == "dataset_id":
            value_column_name = value
          elif key == "payload":
            payload = json.loads(value)
          else: 
            rowjson[key] = value

      renamings = []
      for akey in rowjson.keys():
        if akey != "time" and akey != "value":
          if akey in payload:
            newkeyname = payload[akey][0]
            value = str(payload[rowjson[akey]])
            renamings.append((akey, newkeyname, value))
      
      for key, newkey, val in renamings:
        rowjson[newkeyname] = val
        del rowjson[key]
      
      rowjson[value_column_name] = rowjson["value"]
      del rowjson["value"]

      datalist.append(rowjson)
    return pd.DataFrame(datalist).sort_values(by=['time']).reset_index(drop=True)

  def getMergedDatasetFromFamilies(self, families, merge_on_columns= None, types = None, renamings = None):
    total = None
    
    with ThreadPoolExecutor() as executer:
      datasets = executer.map(lambda x : self.getDatasetFamily(x), families)
    
    merge_cols = None
    for fam, data in zip(families, datasets):
      if total is None:
        total = data
        merge_cols = [x for x in list(data.columns) if x != fam] if merge_on_columns is None else merge_on_columns[fam]
      else:
        if merge_on_columns is None:
          total = total.merge(data, on = merge_cols)
        else:
          total = total.merge(data, left_on = merge_cols, right_on = merge_on_columns[fam])
    total = total if types is None else total.astype(types)
    total = total if renamings is None else total.rename(columns = renamings)
    return total 

class DatasourceAlgorithmic:
  def __init__(self, data, features, target):
    self.__data = data
    self.__features_columns = features
    self.__target_column = target
    self.__features = self.__data.loc[:, self.__features_columns]
    self.__target = self.__data[self.__target_column]
    self.__features_scaler = StandardScaler()
    self.__features_scaler.fit(self.__features)
    self.__target_scaler = StandardScaler()
    self.__target_scaler.fit(self.__target.values.reshape(-1, 1))
    self.__scaled_features = self.__features_scaler.transform(self.__features)
    self.__scaled_target = self.__target_scaler.transform(self.__target.values.reshape(-1,1)).ravel()

  def significance_stars(self, x):
    if (x <= 0.1) & (x > 0.05):
        star_string = "".join(["+"])
    else:
        star_string = "".join(["*" for critical_t in [0.001, 0.01, 0.05] if x <= critical_t])
    return star_string
 
  def getStats(self):
    return self.__data.describe()

  def kurtosis(self, scaled = True, include_target = True):
    features = self.__scaled_features if scaled else self.__features
    frame = pd.DataFrame(features, columns = self.__features_columns)
    if include_target:
      target = self.__scaled_target if scaled else self.__target
      frame[self.__target_column] = target
    return frame.kurtosis()

  def skew(self, scaled = True, include_target = True):
    features = self.__scaled_features if scaled else self.__features
    frame = pd.DataFrame(features, columns = self.__features_columns)
    if include_target:
      target = self.__scaled_target if scaled else self.__target
      frame[self.__target_column] = target
    return frame.skew()

  def corr(self, vsTarget, scaled = True, include_target = True, significantDigits = 6):
    features = self.__scaled_features if scaled else self.__features
    frame = pd.DataFrame(features, columns = self.__features_columns)
    if include_target:
      target = self.__scaled_target if scaled else self.__target
      frame[self.__target_column] = target
    rho = frame.corr()
    pval = frame.corr(method = lambda x, y: pearsonr(x, y)[1]) - np.eye(*rho.shape)
    pstars = pval.applymap(lambda x: "".join(["*" for critical_t in [0.001, 0.01, 0.05] if x <= critical_t]))
    corr = np.round_(rho, significantDigits).astype(str) + pstars
    if vsTarget:
      return corr[self.__target_column]
    else:
      return corr

  def linearRegression(self, scaled = True, include_target = True):
    features = self.__scaled_features if scaled else self.__features
    frame = pd.DataFrame(features, columns = self.__features_columns)
    if include_target:
      target = self.__scaled_target if scaled else self.__target
      frame[self.__target_column] = target
    linear = LinearRegression()
    linear.fit(features, target)
    result = {}
    result["r_squared"] = linear.score(features, target)
        
    predictions = linear.predict(features)
    result["rmse"] = np.sqrt(mean_squared_error(target, predictions))
    result["predictions"] = predictions

    params = np.append(linear.intercept_, linear.coef_)
    newX = pd.DataFrame({"Constant" : np.ones(len(features))}).join(pd.DataFrame(features).reset_index(drop=True))
    MSE = (sum((target - predictions) ** 2)) / (len(newX) - len(newX.columns))
    
    var_b = MSE*(np.linalg.inv(np.dot(newX.T,newX)).diagonal())
    sd_b = np.sqrt(var_b)
    ts_b = params / sd_b

    p_values = [2*(1 - t.cdf(np.abs(i), newX.shape[0] - newX.shape[1])) for i in ts_b]
     
    coefs = pd.DataFrame()
    coefs["Predictors"], coefs["Coefficients"], coefs["Standard Errors"], coefs["t values"], coefs["Probabilities"] = \
	[["intercept"] + list(self.__features_columns), params, sd_b, ts_b, p_values]
    coefs = coefs.set_index(["Predictors"])
    result["coefficients"] = coefs
    return result
     
  def plotPredictedVsObserved(self, regression_results, regression_name, show=False):
    with plt.style.context("fivethirtyeight"):
      fig, ax = plt.subplots(figsize = (16, 18))
    linear_title =  regression_name + " (standard-scaled values)\n"
    linear_title += "r² = {:.6f}  |  RMSE = {:.6f}\n".format(regression_results["r_squared"], regression_results["rmse"])
    ax.set_title(linear_title, fontsize = 20)
    ax.set_xlabel("\nObserved " + self.__target_column)
    ax.set_ylabel("Predicted " + self.__target_column + "\n")
    ax.plot(np.sort(self.__scaled_target), np.sort(self.__scaled_target), linewidth = 2, color = default_colors[1], zorder = 1)
    ax.scatter(self.__scaled_target, regression_results["predictions"], zorder = 2)
    fig.tight_layout()
    if show:
        fig.show()
    else:
        fig.savefig(regression_name + ".png")

  def skew(self, scaled = True, include_target = True):
    features = self.__scaled_features if scaled else self.__features
    frame = pd.DataFrame(features, columns = self.__features_columns)
    if include_target:
      target = self.__scaled_target if scaled else self.__target
      frame[self.__target_column] = target
    return frame.skew()

  def panelOLS(self, timeColumn, scaled = True, include_target = True):
    features = self.__scaled_features if scaled else self.__features
    frame = pd.DataFrame(features, columns = self.__features_columns)
    if include_target:
      target = self.__scaled_target if scaled else self.__target
      frame[self.__target_column] = target
    frame = pd.DataFrame(data = features, columns = self.__features_columns)
    frame[self.__target_column] = target
    otherColumns = [x for x in list(self.__data.columns) if not ((x in self.__features_columns) or (x == self.__target_column) or (x == timeColumn))]
    otherColumns.append(timeColumn)
    for col in otherColumns:
      frame[col] = self.__data[col].values
    frame = frame.set_index(otherColumns)
    exog = sm.tools.tools.add_constant(frame[self.__features_columns])
    fixed_entity_effects = PanelOLS(frame[self.__target_column], exog, entity_effects = True, time_effects = False)
    fee_results = fixed_entity_effects.fit(cov_type = "unadjusted")
    rmse = np.sqrt(mean_squared_error(frame[self.__target_column], fee_results.fitted_values))
    return fee_results, rmse
   
if __name__ == "__main__":
  client = DatasourceDataClient("test", "test")
  #print(client.showDownloadedData())
  #print(client.getDatasetFromUUID("23465ddf-d15d-4b69-99f4-e93df7aad853"))
  #print(client.getDatasetFamily("EN.ATM.CO2E.PC"))
  datasets = [
    "NY.GDP.PCAP.CD", 
    "BX.KLT.DINV.CD.WD", 
    "SL.TLF.TOTL.IN", 
    "SM.POP.NETM", 
    "FP.CPI.TOTL.ZG", 
    "EN.ATM.CO2E.PC"
  ]

  merge_columns = {
	"NY.GDP.PCAP.CD" : ["time", "Country"], 
	"BX.KLT.DINV.CD.WD" : ["time", "Country"],
	"SL.TLF.TOTL.IN" : ["time", "Country"], 
	"SM.POP.NETM" : ["time", "Country"], 
	"FP.CPI.TOTL.ZG" : ["time", "Country"], 
	"EN.ATM.CO2E.PC": ["time", "Country"]
  }

  renames = {
	"NY.GDP.PCAP.CD" : "gdp", 
	"BX.KLT.DINV.CD.WD" : "fdi",
	"SL.TLF.TOTL.IN" : "labor",
	"SM.POP.NETM" : "migration",
	"FP.CPI.TOTL.ZG" : "inflation", 
	"EN.ATM.CO2E.PC": "co2",
	"time" : "year"
  }
  
  types = {
	"NY.GDP.PCAP.CD" : "float32", 
	"BX.KLT.DINV.CD.WD" : "float32",
	"SL.TLF.TOTL.IN" : "float32",
	"SM.POP.NETM" : "float32",
	"FP.CPI.TOTL.ZG" : "float32", 
	"EN.ATM.CO2E.PC": "float32",
	"time" : "int32"
  }
  
  data = client.getMergedDatasetFromFamilies(datasets, merge_columns, types, renames)
  data = data[data["year"] >= 2007]
  algorithmic = DatasourceAlgorithmic(data, ["gdp", "fdi", "labor", "migration", "inflation"], "co2")
  print(algorithmic.getStats())
  print(algorithmic.kurtosis())
  print(algorithmic.skew())
  print(algorithmic.corr(True))
  print(algorithmic.corr(False))
  result = algorithmic.linearRegression()
  print(result)
  algorithmic.plotPredictedVsObserved(result, "Traditional Linear Regression")
  results, rmse = algorithmic.panelOLS("year")
  algorithmic.plotPredictedVsObserved({
		"r_squared": results.corr_squared_overall **2, 
		"rmse" : rmse, 
  		"predictions" : results.fitted_values
	}, "Fixed Entity Effects Regression")
  print(results)
  
