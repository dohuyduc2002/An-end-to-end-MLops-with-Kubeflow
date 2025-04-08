import pandas as pd
import numpy as np
from optbinning import BinningProcess
from sklearn.feature_selection import SequentialFeatureSelector, RFE
from xgboost import XGBClassifier

# =============================================================================
# Preprocess Class
# =============================================================================
class Preprocess:
    """
    Handles missing value imputation and optimal binning using optbinning.
    Also provides methods to calculate Weight of Evidence (WoE), Information Value (IV),
    and to exclude features based on preset rules.
    """
    def __init__(self, X_train, X_test, y_train, categorical_cols, numerical_cols):
        """
        Parameters:
        - X_train (pd.DataFrame): Training features.
        - X_test (pd.DataFrame): Test features.
        - y_train (np.array): Training target labels.
        - categorical_cols (list): List of categorical column names.
        - numerical_cols (list): List of numerical column names.
        """
        # Store copies of original data for missing ratio and deduplication checks
        self.X_train_orig = X_train.copy()
        self.X_test_orig = X_test.copy()  # if needed later
        self.X_train = X_train.copy()
        self.X_test = X_test.copy()
        self.y_train = y_train
        self.categorical_cols = categorical_cols
        self.numerical_cols = numerical_cols
        self.X_train_processed = None
        self.X_test_processed = None
        self.binning_process = None

    def run(self):
        # Impute numerical columns with median (in-place)
        for col in self.numerical_cols:
            if col in self.X_train.columns:
                median_val = self.X_train[col].median()
                self.X_train[col].fillna(median_val, inplace=True)
                if col in self.X_test.columns:
                    self.X_test[col].fillna(median_val, inplace=True)
        # Impute categorical columns with mode or "missing"
        for col in self.categorical_cols:
            if col in self.X_train.columns:
                mode_val = self.X_train[col].mode()
                fill_val = mode_val[0] if not mode_val.empty else "missing"
                self.X_train[col].fillna(fill_val, inplace=True)
                if col in self.X_test.columns:
                    self.X_test[col].fillna(fill_val, inplace=True)
        # Fit optimal binning process on all features (order matters for subsequent IV computation)
        all_features = self.categorical_cols + self.numerical_cols
        self.binning_process = BinningProcess(variable_names=all_features,
                                              categorical_variables=self.categorical_cols)
        self.binning_process.fit(self.X_train[all_features], self.y_train)
        # Transform the data using the learned binning
        X_train_binned = self.binning_process.transform(self.X_train[all_features])
        self.X_train_processed = pd.DataFrame(X_train_binned, columns=all_features)
        if not self.X_test.empty:
            X_test_binned = self.binning_process.transform(self.X_test[all_features])
            self.X_test_processed = pd.DataFrame(X_test_binned, columns=all_features)
        else:
            self.X_test_processed = pd.DataFrame(columns=all_features)
        return self.X_train_processed, self.X_test_processed

    @staticmethod
    def compute_iv(series, y):
        """
        Compute the Information Value (IV) for a given binned series.
        Parameters:
        - series (pd.Series): Binned variable (e.g., as produced by BinningProcess).
        - y (array-like): Binary target (assumed 0/1).
        Returns:
        - iv (float): The Information Value.
        """
        df = pd.DataFrame({"bin": series, "target": y})
        total_good = (df["target"] == 0).sum()
        total_bad = (df["target"] == 1).sum()
        iv = 0
        eps = 0.5
        for val, group in df.groupby("bin"):
            good = (group["target"] == 0).sum()
            bad = (group["target"] == 1).sum()
            if good == 0:
                good = eps
            if bad == 0:
                bad = eps
            dist_good = good / total_good
            dist_bad = bad / total_bad
            woe = np.log(dist_good / dist_bad)
            iv += (dist_good - dist_bad) * woe
        return iv

    def filter_features(self):
        """
        Calculates the Information Value (IV) for each processed feature and
        excludes features based on these rules:
         - IV > 0.5 or IV < 0.02
         - Missing value rate (from original data) is above 30%
         - For categorical features: if the ratio of unique values (uniqueness) is above 40%
        
        Drops the flagged features from both the processed training and test sets.
        
        Returns:
        - iv_dict (dict): Dictionary of {feature: IV_value}.
        - features_to_exclude (list): List of feature names that were excluded.
        """
        if self.X_train_processed is None:
            raise RuntimeError("Processed data not available. Run the 'run()' method first.")
        features_to_exclude = []
        iv_dict = {}
        for col in self.X_train_processed.columns:
            iv = Preprocess.compute_iv(self.X_train_processed[col], self.y_train)
            iv_dict[col] = iv
            if iv > 0.5 or iv < 0.02:
                features_to_exclude.append(col)
                continue
            missing_ratio = self.X_train_orig[col].isnull().mean()
            if missing_ratio > 0.3:
                features_to_exclude.append(col)
                continue
        self.X_train_processed = self.X_train_processed.drop(columns=features_to_exclude)
        if self.X_test_processed is not None and not self.X_test_processed.empty:
            self.X_test_processed = self.X_test_processed.drop(columns=features_to_exclude, errors='ignore')
        return iv_dict, features_to_exclude

# =============================================================================
# FeatureSelectionWrapper Class
# =============================================================================
class FeatureSelectionWrapper:
    """
    A wrapper for feature selection using Sequential Feature Selector (SFS) 
    or Recursive Feature Elimination (RFE).
    """
    def __init__(self, estimator, method="sfs", n_features_to_select=None, **kwargs):
        """
        Parameters:
        - estimator: A scikit-learn estimator used to score features.
        - method (str): "sfs" or "rfe".
        - n_features_to_select (int or None): Number of features to select (default: half).
        - kwargs: Additional arguments for the feature selector.
        """
        self.estimator = estimator
        self.method = method.lower()
        self.n_features_to_select = n_features_to_select
        self.kwargs = kwargs
        self.selector = None
        self.selected_features = None
        self.excluded_features = None

    def fit(self, X, y):
        if self.n_features_to_select is None:
            self.n_features_to_select = X.shape[1] // 2
        if self.method == "sfs":
            self.selector = SequentialFeatureSelector(
                self.estimator,
                n_features_to_select=self.n_features_to_select,
                direction='forward',
                **self.kwargs
            )
        elif self.method == "rfe":
            self.selector = RFE(self.estimator, n_features_to_select=self.n_features_to_select, **self.kwargs)
        else:
            raise ValueError("Method must be 'sfs' or 'rfe'")
        self.selector.fit(X, y)
        mask = self.selector.get_support()
        if hasattr(X, "columns"):
            self.selected_features = list(X.columns[mask])
            self.excluded_features = list(X.columns[~mask])
        else:
            self.selected_features = None
            self.excluded_features = None
        return self

    def transform(self, X):
        return self.selector.transform(X)
    
    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

# =============================================================================
# PreprocessFeatureSelector Wrapper Class
# =============================================================================
class PreprocessFeatureSelector:
    """
    A wrapper class that runs both preprocessing and feature selection.
    
    It performs missing value imputation and optimal binning via the Preprocess class,
    optionally filters features using IV rules, then applies feature selection using
    FeatureSelectionWrapper.
    
    Parameters:
      - X_train (pd.DataFrame): Raw training features.
      - X_test (pd.DataFrame): Raw test features.
      - y_train (np.array): Training target labels.
      - categorical_cols (list): List of categorical feature names.
      - numerical_cols (list): List of numerical feature names.
      - estimator: Estimator used for feature selection (default: XGBClassifier).
      - fs_method (str): Feature selection method ("sfs" or "rfe").
      - n_features_to_select (int): Number of features to select.
      - fs_kwargs (dict): Additional keyword arguments for the feature selector.
    
    Returns:
      - selected_train (pd.DataFrame): Final processed training data.
      - selected_test (pd.DataFrame): Final processed test data.
      - selected_features (list): List of features that were retained.
      - excluded_features (list): List of features dropped during preprocessing and feature selection.
    """
    def __init__(self, X_train, X_test, y_train, categorical_cols, numerical_cols,
                 estimator=None, fs_method="sfs", n_features_to_select=None, fs_kwargs=None):
        self.X_train = X_train.copy()
        self.X_test = X_test.copy()
        self.y_train = y_train
        self.categorical_cols = categorical_cols
        self.numerical_cols = numerical_cols
        if estimator is None:
            self.estimator = XGBClassifier(use_label_encoder=False, eval_metric='auc')
        else:
            self.estimator = estimator
        self.fs_method = fs_method
        self.n_features_to_select = n_features_to_select
        self.fs_kwargs = fs_kwargs if fs_kwargs is not None else {}
        self.preprocess_obj = None
        self.fs_wrapper = None
        self.excluded_features = []  # to store features excluded at any stage

    def run(self):
        # Step 1: Preprocessing
        self.preprocess_obj = Preprocess(self.X_train, self.X_test, self.y_train,
                                          self.categorical_cols, self.numerical_cols)
        X_train_proc, X_test_proc = self.preprocess_obj.run()
        # Optionally filter features based on IV/missing ratio rules
        _, iv_excluded = self.preprocess_obj.filter_features()
        self.excluded_features.extend(iv_excluded)
        
        # Step 2: Feature Selection
        self.fs_wrapper = FeatureSelectionWrapper(self.estimator, method=self.fs_method,
                                                   n_features_to_select=self.n_features_to_select,
                                                   **self.fs_kwargs)
        self.fs_wrapper.fit(X_train_proc, self.y_train)
        selected_train = pd.DataFrame(self.fs_wrapper.transform(X_train_proc),
                                      columns=self.fs_wrapper.selected_features)
        if X_test_proc is not None and not X_test_proc.empty:
            selected_test = pd.DataFrame(self.fs_wrapper.transform(X_test_proc),
                                         columns=self.fs_wrapper.selected_features)
        else:
            selected_test = None
        self.excluded_features.extend(self.fs_wrapper.excluded_features)
        return selected_train, selected_test, self.fs_wrapper.selected_features, self.excluded_features
