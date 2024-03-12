# seed value for random number generators to obtain reproducible results
RANDOM_SEED = 1

# although we standardize X and y variables on input,
# we will fit the intercept term in the models
# Expect fitted values to be close to zero
SET_FIT_INTERCEPT = True

import os
import requests
import json
import base64
from textwrap import wrap

import numpy as np
import pandas as pd

import statsmodels as sm
from linearmodels import PooledOLS, PanelOLS, RandomEffects, FamaMacBeth
import statsmodels.api as sma
import statsmodels.formula.api as smf

from scipy.stats import skew, kurtosis, pearsonr, gaussian_kde, t # , lognorm, norm
from scipy.spatial import distance

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LassoLarsIC
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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

class DatasourceAlgorithmic:
  def __init__(self, data, workflow_type, features, target=None):
    if (workflow_type == "supervised") and (target is None):
        raise Exception("Supervised Algorithms can not set without target")
    self.__data = data
    self.__train_data = data
    self.__test_data = None
    self.__features_train_columns = features
    self.__target_train_column = target
    self.__features_train = self.__data.loc[:, self.__features_train_columns]
    self.__target_train = self.__data[self.__target_train_column].values.reshape(-1,1)
    self.__features_test = None
    self.__target_test = None
    self.__features_scaler = None
    self.__target_scaler = None
    self.__poly = None
    self.isSplit = False
    self.isScaled = False
    self.__regression_models = []
    self.__metrics = []



  def significance_stars(self, x):
    if (x <= 0.1) & (x > 0.05):
        star_string = "".join(["+"])
    else:
        star_string = "".join(["*" for critical_t in [0.001, 0.01, 0.05] if x <= critical_t])
    return star_string

  def getStats(self):
    return self.__train_data.describe()

  def getInfo(self):
    return self.__train_data.info()

  def train_test_split(self, test_size = 1 / np.e):
    self.isSplit = True
    self.__train_data, self.__test_data = \
    train_test_split(self.__data, test_size = test_size, random_state = RANDOM_SEED)
    self.__features_train = self.__train_data.loc[:, self.__features_train_columns]
    self.__target_train = self.__train_data[self.__target_train_column].values.reshape(-1,1)
    self.__features_test = self.__test_data.loc[:, self.__features_train_columns]
    self.__target_test = self.__test_data[self.__target_train_column].values.reshape(-1,1)

  def scale(self, scale_type):
    if scale_type == "standard":
        self.__features_scaler = StandardScaler()
        self.__target_scaler = StandardScaler()
    elif scale_type == "minmax":
        self.__features_scaler = MinMaxScaler()
        self.__target_scaler = MinMaxScaler()
    elif scale_type == "none":
        pass
    else:
        raise NotImplemented()

    if self.__features_scaler is not None:
        self.isScaled = True
        self.__features_scaler.fit(self.__features_train)
        self.__target_scaler.fit(self.__target_train)

        self.__features_train = self.__features_scaler.transform(self.__features_train)
        self.__target_train = self.__target_scaler.transform(self.__target_train)

        if not (self.__features_test is None):
            self.__features_test = self.__features_scaler.transform(self.__features_test)
            self.__target_test = self.__target_scaler.transform(self.__target_test)

    self.__target_train = self.__target_train.ravel()
    if not (self.__features_test is None):
        self.__target_test = self.__target_test.ravel()

  def skew(self, include_target = True):
    frame = pd.DataFrame(self.__features_train, columns = self.__features_train_columns)
    if include_target:
      frame[self.__target_train_column] = self.__target_train
    return frame.skew()

  def kurtosis(self, include_target = True):
    frame = pd.DataFrame(self.__features_train, columns = self.__features_train_columns)
    if include_target:
      frame[self.__target_train_column] = self.__target_train
    return frame.kurtosis()

  def corr(self, vsTarget, include_target = True, significantDigits = 6):
    frame = pd.DataFrame(self.__features_train, columns = self.__features_train_columns)
    if include_target:
      frame[self.__target_train_column] = self.__target_train
    rho = frame.corr()
    pval = frame.corr(method = lambda x, y: pearsonr(x, y)[1]) - np.eye(*rho.shape)
    pstars = pval.applymap(lambda x: "".join(["*" for critical_t in [0.001, 0.01, 0.05] if x <= critical_t]))
    corr = np.round_(rho, significantDigits).astype(str) + pstars
    if vsTarget:
      return corr[self.__target_train_column]
    else:
      return corr

  def linearRegression(self):
    def get_features_target(for_test = False):
      if for_test and (self.__features_test is None):
        return None, None
      features = self.__features_train if not for_test else self.__features_test
      frame = pd.DataFrame(self.__features_train, columns = self.__features_train_columns)
      target = self.__target_train if not for_test else self.__target_test
      return features, target

    features, target = get_features_target(for_test=False)
    features_test, target_test = get_features_target(for_test=True)

    linear = LinearRegression()


    linear.fit(features, target)

    predictions = linear.predict(features)

    test_predictions = linear.predict(features_test) if not (features_test is None) else None

    params = np.append(linear.intercept_, linear.coef_)
    newX = pd.DataFrame({"Constant" : np.ones(len(features))}).join(pd.DataFrame(features).reset_index(drop=True))
    MSE = (sum((target - predictions) ** 2)) / (len(newX) - len(newX.columns))

    var_b = MSE*(np.linalg.inv(np.dot(newX.T,newX)).diagonal())
    sd_b = np.sqrt(var_b)
    ts_b = params / sd_b

    p_values = [2*(1 - t.cdf(np.abs(i), newX.shape[0] - newX.shape[1])) for i in ts_b]

    significance = [self.significance_stars(p) for p in p_values]
    coefs = pd.DataFrame()
    coefs["Predictors"], coefs["Coefficients"], coefs["Standard Errors"], coefs["t values"], coefs["Probabilities"], coefs["Significance"] = \
    [["intercept"] + list(self.__features_train_columns), params, sd_b, ts_b, p_values, significance]
    coefs = coefs.set_index(["Predictors"])
    #result["coefficients"] = coefs

    nonrandomness_array = 1 - np.array(p_values[1:])
    significant_lines = abs(linear.coef_) * nonrandomness_array
    modified_beta_coefficients = significant_lines / significant_lines.sum()


    feature_names = self.__features_train_columns

    importances = modified_beta_coefficients
    sorted_index = np.argsort(importances)[::-1]
    labels = np.array(feature_names)[sorted_index]

    feature_sign = pd.DataFrame()
    feature_sign["Feature Column"], feature_sign["Linear Regression"] = \
    [labels, importances[sorted_index]]

    model = DatasourceRegressionModelResult(linear, "LinearRegression", features, target, predictions, features_test, target_test, test_predictions, feature_sign)

    self.__regression_models.append(model)

    return coefs #{"train_output_log" : coefs, "model" : model}



  def panelOLS(self, timeColumn, entity_effects, time_effects):
    def get_frame_and_exog(for_test=False):
      if for_test and self.__features_test is None:
        return None, None
      features = self.__features_train if not for_test else self.__features_test
      frame = pd.DataFrame(features, columns = self.__features_train_columns)
      #if include_target:
      target = self.__target_train if not for_test else self.__target_test
      #frame[self.__target_train_column] = target
      frame = pd.DataFrame(data = features, columns = self.__features_train_columns)
      frame[self.__target_train_column] = target
      otherColumns = [x for x in list(self.__data.columns) if not ((x in self.__features_train_columns) or (x == self.__target_train_column) or (x == timeColumn))]
      otherColumns.append(timeColumn)
      for col in otherColumns:
        frame[col] = self.__train_data[col].values if not for_test else self.__test_data[col].values
      frame = frame.set_index(otherColumns)
      exog = sm.tools.tools.add_constant(frame[self.__features_train_columns])
      return frame, exog

    frame, exog = get_frame_and_exog()
    test_frame, test_exog = get_frame_and_exog(True)
    fixed_entity_effects = PanelOLS(frame[self.__target_train_column], exog, entity_effects = entity_effects, time_effects = time_effects)
    fee_results = fixed_entity_effects.fit(cov_type = "unadjusted")
    predictions = fee_results.predict(exog)
    test_predictions = fee_results.predict(test_exog)["predictions"] if not (test_exog is None) else None
    rmse = np.sqrt(mean_squared_error(frame[self.__target_train_column], fee_results.fitted_values))

    parameters = pd.DataFrame()
    parameters["Parameter Name"] = ["const"] + self.__features_train_columns
    parameters["Parameter Estimate"] = list(fee_results.params)

    parameters["Std. Err"] = list(fee_results.std_errors)
    parameters["T-stat"] = list(fee_results.tstats)
    parameters["P Value"] = list(fee_results.pvalues)
    parameters["Parameter Significance"] = [self.significance_stars(p) for p in list(fee_results.pvalues)]
    parameters["Lower CI"], parameters["Upper CI"] = list(fee_results.conf_int()["lower"]), list(fee_results.conf_int()["upper"])

    beta_sum = (np.abs(fee_results.params) * (1 - fee_results.pvalues)).sum()
    split_fee_beta_coefficients = (np.abs(fee_results.params) * (1 - fee_results.pvalues)) / beta_sum

    feature_names = ["const"] + self.__features_train_columns



    importances = np.array(split_fee_beta_coefficients)
    sorted_index = np.argsort(importances)[::-1]
    labels = np.array(feature_names)[sorted_index]

    name = "PanelOLS(entity_effects=" + str(entity_effects) + "-time_effects=" + str(time_effects) + ")"


    feature_sign = pd.DataFrame()
    feature_sign["Feature Column"], feature_sign[name] = \
    [labels, importances[sorted_index]]

    model = DatasourceRegressionModelResult(fee_results, name, exog, frame[self.__target_train_column],
                                            predictions["predictions"], test_exog, test_frame[self.__target_train_column] if test_frame is not None else None,
                                            test_predictions, feature_sign)

    self.__regression_models.append(model)

    return parameters

  def registerR2(self):
        self.__metrics.append("r2")

  def registerRMSE(self):
        self.__metrics.append("rmse")

  def registerMSE(self):
        self.__metrics.append("mse")

  def compareRegressionModelOutputs(self):
    metrics = self.__metrics
    use_test = True if self.__features_test is not None else False
    train_results = pd.DataFrame()
    test_results = pd.DataFrame()

    model_identifiers = [model.getModelType() for model in self.__regression_models]
    train_results["model"] = model_identifiers
    test_results["model"] = model_identifiers
    for metric in metrics:
      values = None
      if metric == "mse":
        values = [model.mse(use_test) for model in self.__regression_models]
      elif metric == "rmse":
        values = [model.rmse(use_test) for model in self.__regression_models]
      elif metric == "r2":
        values = [model.rmse(use_test) for model in self.__regression_models]
      else:
        raise NotImplemented()

      train_results[metric] = [value[0] for value in values]
      if use_test == True:
        test_results[metric] = [value[1] for value in values]

    return [train_results, test_results] if use_test else [train_results]

  def getSignificantsFeaturesForRegressionModels(self):
    result = None
    for model in self.__regression_models:
        df = model.feature_significance()
        result = df if result is None else pd.merge(result, df, on = "Feature Column", how = "outer")
    return result

  def getTargetsPredictionsForRegressionModels(self):
    train_result = None
    test_result = None
    for model in self.__regression_models:
        data = model.targets_vs_predictions()
        train_result = data[0] if train_result is None else pd.merge(train_result, data[0], on = "target", how = "inner")
        if self.isSplit:
          test_result = data[1] if test_result is None else pd.merge(test_result, data[1], on = "target", how = "inner")
    if self.isSplit:
        return [train_result, test_result]
    else:
        return [train_result]
    return []

  def getTrainData(self):
    frame = pd.DataFrame(data = self.__features_train, columns = self.__features_train_columns)
    frame[self.__target_train_column] = self.__target_train
    otherColumns = [x for x in list(self.__data.columns) if not ((x in self.__features_train_columns) or (x == self.__target_train_column))]
    for col in otherColumns:
      frame[col] = self.__train_data[col].values
    return frame

  def getTestData(self):
    frame = pd.DataFrame(data = self.__features_test, columns = self.__features_train_columns)
    frame[self.__target_train_column] = self.__target_test
    otherColumns = [x for x in list(self.__data.columns) if not ((x in self.__features_train_columns) or (x == self.__target_train_column))]
    for col in otherColumns:
      frame[col] = self.__test_data[col].values
    return frame

#  def getNonFeatureColumns(self):
#        [x for x in self.__train_data.columns]

class DatasourceRegressionModelResult:
  def __init__(self, model, model_type, train_features, train_targets, train_predictions, test_features, test_targets, test_predictions, feature_significance):
    self.__model = model
    self.__model_type = model_type
    self.__train_target = train_targets
    self.__train_predictions = train_predictions
    self.__test_target = test_targets
    self.__test_predictions = test_predictions
    self.__feature_significance = feature_significance

  def significance_stars(self, x):
    if (x <= 0.1) & (x > 0.05):
        star_string = "".join(["+"])
    else:
        star_string = "".join(["*" for critical_t in [0.001, 0.01, 0.05] if x <= critical_t])
    return star_string

  def getModel(self):
    return self.__model

  def getModelType(self):
    return self.__model_type

  def targets_vs_predictions(self):

    train = pd.DataFrame()

    train["target"] = self.__train_target
    train[self.__model_type] = self.__train_predictions

    if not (self.__test_target is None):
      test = pd.DataFrame()
      test["target"] = self.__test_target
      test[self.__model_type] = self.__test_predictions
      return [train.reset_index(drop=True), test.reset_index(drop=True)]

    return [train.reset_index(drop=True)]

  def mse(self, use_test=False):
    res = [mean_squared_error(self.__train_target, self.__train_predictions)]
    if use_test:
        res.append(mean_squared_error(self.__test_target, self.__test_predictions))
    return res

  def rmse(self, use_test=False):
    res = [np.sqrt(mean_squared_error(self.__train_target, self.__train_predictions))]
    if use_test:
        res.append(np.sqrt(mean_squared_error(self.__test_target, self.__test_predictions)))
    return res

  def r2(self, use_test=False):
    res = [r2_score(self.__train_target, self.__train_predictions)]
    if use_test:
        res.append(r2_score(self.__test_target, self.__test_predictions))
    return res

  def feature_significance(self):
    return self.__feature_significance
