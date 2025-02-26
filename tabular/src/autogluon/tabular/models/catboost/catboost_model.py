import logging
import math
import os
import pickle
import sys
import time
import psutil
import numpy as np

from autogluon.core.constants import PROBLEM_TYPES_CLASSIFICATION, MULTICLASS, SOFTCLASS
from autogluon.core.features.types import R_OBJECT
from autogluon.core.models import AbstractModel
from autogluon.core.models._utils import get_early_stopping_rounds
from autogluon.core.utils.exceptions import NotEnoughMemoryError, TimeLimitExceeded
from autogluon.core.utils import try_import_catboost, try_import_catboostdev

from autogluon.tabular.models.catboost.catboost_utils import construct_custom_catboost_metric
from autogluon.tabular.models.catboost.hyperparameters.parameters import get_param_baseline
from autogluon.tabular.models.catboost.hyperparameters.searchspaces import get_default_searchspace

logger = logging.getLogger(__name__)


# TODO: Consider having CatBoost variant that converts all categoricals to numerical as done in RFModel, was showing improved results in some problems.
class CatBoostModel(AbstractModel):
    """
    CatBoost model: https://catboost.ai/

    Hyperparameter options: https://catboost.ai/docs/concepts/python-reference_parameters-list.html
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._category_features = None

    def _set_default_params(self):
        default_params = get_param_baseline(problem_type=self.problem_type)
        for param, val in default_params.items():
            self._set_default_param_value(param, val)
        self._set_default_param_value('random_seed', 0)  # Remove randomness for reproducibility
        # Set 'allow_writing_files' to True in order to keep log files created by catboost during training (these will be saved in the directory where AutoGluon stores this model)
        self._set_default_param_value('allow_writing_files', False)  # Disables creation of catboost logging files during training by default
        if self.problem_type != SOFTCLASS:  # TODO: remove this after catboost 0.24
            self._set_default_param_value('eval_metric', construct_custom_catboost_metric(self.stopping_metric, True, not self.stopping_metric.needs_pred, self.problem_type))

    def _get_default_searchspace(self):
        return get_default_searchspace(self.problem_type, num_classes=self.num_classes)

    def _preprocess_nonadaptive(self, X, **kwargs):
        X = super()._preprocess_nonadaptive(X, **kwargs)
        if self._category_features is None:
            self._category_features = list(X.select_dtypes(include='category').columns)
        if self._category_features:
            X = X.copy()
            for category in self._category_features:
                current_categories = X[category].cat.categories
                if '__NaN__' in current_categories:
                    X[category] = X[category].fillna('__NaN__')
                else:
                    X[category] = X[category].cat.add_categories('__NaN__').fillna('__NaN__')
        return X

    # TODO: Use Pool in preprocess, optimize bagging to do Pool.split() to avoid re-computing pool for each fold! Requires stateful + y
    #  Pool is much more memory efficient, avoids copying data twice in memory
    def _fit(self,
             X,
             y,
             X_val=None,
             y_val=None,
             time_limit=None,
             num_gpus=0,
             sample_weight=None,
             sample_weight_val=None,
             **kwargs):
        try_import_catboost()
        from catboost import CatBoostClassifier, CatBoostRegressor, Pool
        ag_params = self._get_ag_params()
        params = self._get_model_params()
        if self.problem_type == SOFTCLASS:
            try_import_catboostdev()  # Need to first import catboost then catboost_dev not vice-versa.
            from catboost_dev import CatBoostClassifier, CatBoostRegressor, Pool
            from .catboost_softclass_utils import SoftclassCustomMetric, SoftclassObjective
            params['loss_function'] = SoftclassObjective.SoftLogLossObjective()
            params['eval_metric'] = SoftclassCustomMetric.SoftLogLossMetric()

        model_type = CatBoostClassifier if self.problem_type in PROBLEM_TYPES_CLASSIFICATION else CatBoostRegressor
        if isinstance(params['eval_metric'], str):
            metric_name = params['eval_metric']
        else:
            metric_name = type(params['eval_metric']).__name__
        num_rows_train = len(X)
        num_cols_train = len(X.columns)
        if self.problem_type == MULTICLASS:
            if self.num_classes is not None:
                num_classes = self.num_classes
            else:
                num_classes = 10  # Guess if not given, can do better by looking at y
        elif self.problem_type == SOFTCLASS:  # TODO: delete this elif if it's unnecessary.
            num_classes = y.shape[1]
        else:
            num_classes = 1

        # TODO: Add ignore_memory_limits param to disable NotEnoughMemoryError Exceptions
        max_memory_usage_ratio = self.params_aux['max_memory_usage_ratio']
        approx_mem_size_req = num_rows_train * num_cols_train * num_classes / 2  # TODO: Extremely crude approximation, can be vastly improved
        if approx_mem_size_req > 1e9:  # > 1 GB
            available_mem = psutil.virtual_memory().available
            ratio = approx_mem_size_req / available_mem
            if ratio > (1 * max_memory_usage_ratio):
                logger.warning('\tWarning: Not enough memory to safely train CatBoost model, roughly requires: %s GB, but only %s GB is available...' % (round(approx_mem_size_req / 1e9, 3), round(available_mem / 1e9, 3)))
                raise NotEnoughMemoryError
            elif ratio > (0.2 * max_memory_usage_ratio):
                logger.warning('\tWarning: Potentially not enough memory to safely train CatBoost model, roughly requires: %s GB, but only %s GB is available...' % (round(approx_mem_size_req / 1e9, 3), round(available_mem / 1e9, 3)))

        start_time = time.time()
        X = self.preprocess(X)
        cat_features = list(X.select_dtypes(include='category').columns)
        X = Pool(data=X, label=y, cat_features=cat_features, weight=sample_weight)

        if X_val is None:
            eval_set = None
            num_sample_iter_max = 50
            early_stopping_rounds = None
        else:
            X_val = self.preprocess(X_val)
            X_val = Pool(data=X_val, label=y_val, cat_features=cat_features, weight=sample_weight_val)
            eval_set = X_val
            modifier = min(1.0, 10000 / num_rows_train)
            num_sample_iter_max = max(round(modifier * 50), 2)
            early_stopping_rounds = ag_params.get('ag.early_stop', 'auto')
            if isinstance(early_stopping_rounds, str):
                early_stopping_rounds = self._get_early_stopping_rounds(num_rows_train=num_rows_train, strategy=early_stopping_rounds)

        if params.get('allow_writing_files', False):
            if 'train_dir' not in params:
                try:
                    # TODO: What if path is in S3?
                    os.makedirs(os.path.dirname(self.path), exist_ok=True)
                except:
                    pass
                else:
                    params['train_dir'] = self.path + 'catboost_info'

        # TODO: Add more control over these params (specifically early_stopping_rounds)
        verbosity = kwargs.get('verbosity', 2)
        if verbosity <= 1:
            verbose = False
        elif verbosity == 2:
            verbose = False
        elif verbosity == 3:
            verbose = 20
        else:
            verbose = True

        init_model = None
        init_model_tree_count = None
        init_model_best_score = None

        num_features = len(self._features)

        if num_gpus != 0:
            if 'task_type' not in params:
                params['task_type'] = 'GPU'
                logger.log(20, f'\tTraining {self.name} with GPU, note that this may negatively impact model quality compared to CPU training.')
                # TODO: Confirm if GPU is used in HPO (Probably not)
                # TODO: Adjust max_bins to 254?

        if params.get('task_type', None) == 'GPU':
            if 'colsample_bylevel' in params:
                params.pop('colsample_bylevel')
                logger.log(30, f'\t\'colsample_bylevel\' is not supported on GPU, using default value (Default = 1).')
            if 'rsm' in params:
                params.pop('rsm')
                logger.log(30, f'\t\'rsm\' is not supported on GPU, using default value (Default = 1).')

        if self.problem_type == MULTICLASS and 'rsm' not in params and 'colsample_bylevel' not in params and num_features > 1000:
            if time_limit:
                # Reduce sample iterations to avoid taking unreasonable amounts of time
                num_sample_iter_max = max(round(num_sample_iter_max/2), 2)
            # Subsample columns to speed up training
            if params.get('task_type', None) != 'GPU':  # RSM does not work on GPU
                params['colsample_bylevel'] = max(min(1.0, 1000 / num_features), 0.05)
                logger.log(30, f'\tMany features detected ({num_features}), dynamically setting \'colsample_bylevel\' to {params["colsample_bylevel"]} to speed up training (Default = 1).')
                logger.log(30, f'\tTo disable this functionality, explicitly specify \'colsample_bylevel\' in the model hyperparameters.')
            else:
                params['colsample_bylevel'] = 1.0
                logger.log(30, f'\t\'colsample_bylevel\' is not supported on GPU, using default value (Default = 1).')

        logger.log(15, f'\tCatboost model hyperparameters: {params}')

        if time_limit:
            time_left_start = time_limit - (time.time() - start_time)
            if time_left_start <= time_limit * 0.4:  # if 60% of time was spent preprocessing, likely not enough time to train model
                raise TimeLimitExceeded
            params_init = params.copy()
            num_sample_iter = min(num_sample_iter_max, params_init['iterations'])
            params_init['iterations'] = num_sample_iter
            self.model = model_type(
                **params_init,
            )
            self.model.fit(
                X,
                eval_set=eval_set,
                use_best_model=True,
                verbose=verbose,
                # early_stopping_rounds=early_stopping_rounds,
            )

            init_model_tree_count = self.model.tree_count_
            init_model_best_score = self._get_best_val_score(self.model, metric_name)

            time_left_end = time_limit - (time.time() - start_time)
            time_taken_per_iter = (time_left_start - time_left_end) / num_sample_iter
            estimated_iters_in_time = round(time_left_end / time_taken_per_iter)
            init_model = self.model

            if self.stopping_metric._optimum == init_model_best_score:
                # Done, pick init_model
                params_final = None
            else:
                params_final = params.copy()

                # TODO: This only handles memory with time_limit specified, but not with time_limit=None, handle when time_limit=None
                available_mem = psutil.virtual_memory().available
                if self.problem_type == SOFTCLASS:  # TODO: remove this once catboost-dev is no longer necessary and SOFTCLASS objectives can be pickled.
                    model_size_bytes = 1  # skip memory check
                else:
                    model_size_bytes = sys.getsizeof(pickle.dumps(self.model))

                max_memory_proportion = 0.3 * max_memory_usage_ratio
                mem_usage_per_iter = model_size_bytes / num_sample_iter
                max_memory_iters = math.floor(available_mem * max_memory_proportion / mem_usage_per_iter)
                if params.get('task_type', None) == 'GPU':
                    # Cant use init_model
                    iterations_left = params['iterations']
                else:
                    iterations_left = params['iterations'] - num_sample_iter
                params_final['iterations'] = min(iterations_left, estimated_iters_in_time)
                if params_final['iterations'] > max_memory_iters - num_sample_iter:
                    if max_memory_iters - num_sample_iter <= 500:
                        logger.warning('\tWarning: CatBoost will be early stopped due to lack of memory, increase memory to enable full quality models, max training iterations changed to %s from %s' % (max_memory_iters, params_final['iterations'] + num_sample_iter))
                    params_final['iterations'] = max_memory_iters - num_sample_iter
        else:
            params_final = params.copy()

        if params_final is not None and params_final['iterations'] > 0:
            self.model = model_type(
                **params_final,
            )

            fit_final_kwargs = dict(
                eval_set=eval_set,
                verbose=verbose,
                early_stopping_rounds=early_stopping_rounds,
            )

            # TODO: Strangely, this performs different if clone init_model is sent in than if trained for same total number of iterations. May be able to optimize catboost models further with this
            warm_start = False
            if params_final.get('task_type', None) == 'GPU':
                # Cant use init_model
                fit_final_kwargs['use_best_model'] = True
            elif init_model is not None:
                fit_final_kwargs['init_model'] = init_model
                warm_start = True
            self.model.fit(X, **fit_final_kwargs)

            if init_model is not None:
                final_model_best_score = self._get_best_val_score(self.model, metric_name)

                if self.stopping_metric._optimum == init_model_best_score:
                    # Done, pick init_model
                    self.model = init_model
                else:
                    if (init_model_best_score > self.stopping_metric._optimum) or (final_model_best_score > self.stopping_metric._optimum):
                        init_model_best_score = -init_model_best_score
                        final_model_best_score = -final_model_best_score

                    if warm_start:
                        if init_model_best_score >= final_model_best_score:
                            self.model = init_model
                        else:
                            best_iteration = init_model_tree_count + self.model.get_best_iteration()
                            self.model.shrink(ntree_start=0, ntree_end=best_iteration + 1)
                    else:
                        if init_model_best_score >= final_model_best_score:
                            self.model = init_model

        self.params_trained['iterations'] = self.model.tree_count_

    def _predict_proba(self, X, **kwargs):
        if self.problem_type != SOFTCLASS:
            return super()._predict_proba(X, **kwargs)
        # For SOFTCLASS problems, manually transform predictions into probabilities via softmax
        X = self.preprocess(X, **kwargs)
        y_pred_proba = self.model.predict(X, prediction_type='RawFormulaVal')
        y_pred_proba = np.exp(y_pred_proba)
        y_pred_proba = np.multiply(y_pred_proba, 1/np.sum(y_pred_proba, axis=1)[:, np.newaxis])
        if y_pred_proba.shape[1] == 2:
            y_pred_proba = y_pred_proba[:,1]
        return y_pred_proba

    def get_model_feature_importance(self):
        importance_df = self.model.get_feature_importance(prettified=True)
        importance_df['Importances'] = importance_df['Importances'] / 100
        importance_series = importance_df.set_index('Feature Id')['Importances']
        importance_dict = importance_series.to_dict()
        return importance_dict

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = dict(
            ignored_type_group_raw=[R_OBJECT],
        )
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params

    @staticmethod
    def _get_best_val_score(model, metric_name):
        """Necessary to trim extra args off of metric_name, such as 'F1:hints=skip_train~true' -> 'F1'"""
        model_best_scores = model.get_best_score()['validation']
        if metric_name in model_best_scores:
            best_score = model_best_scores[metric_name]
        else:
            metric_name_sub = metric_name.split(':')[0]
            best_score = model_best_scores[metric_name_sub]
        return best_score

    def _get_early_stopping_rounds(self, num_rows_train, strategy='auto'):
        return get_early_stopping_rounds(num_rows_train=num_rows_train, strategy=strategy)

    def _ag_params(self) -> set:
        return {'ag.early_stop'}
