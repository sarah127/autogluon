from collections import Counter
from statistics import mean

import numpy as np
import pandas as pd
from autogluon.core.models.ensemble.fold_fitting_strategy import *

from autogluon.core.constants import MULTICLASS, REGRESSION, SOFTCLASS, QUANTILE, REFIT_FULL_SUFFIX
from autogluon.core.utils.exceptions import TimeLimitExceeded
from autogluon.core.utils.loaders import load_pkl
from autogluon.core.utils.savers import save_pkl
from autogluon.core.utils.utils import generate_kfold, _compute_fi_with_stddev

from autogluon.core.models.abstract.abstract_model import AbstractModel

logger = logging.getLogger(__name__)


# TODO: Add metadata object with info like score on each model, train time on each model, etc.
class BaggedEnsembleModel(AbstractModel):
    """
    Bagged ensemble meta-model which fits a given model multiple times across different splits of the training data.

    For certain child models such as KNN, this may only train a single model and instead rely on the child model to generate out-of-fold predictions.
    """
    _oof_filename = 'oof.pkl'

    def __init__(self, model_base: AbstractModel, random_state=0, **kwargs):
        self.model_base = model_base
        self._child_type = type(self.model_base)
        self.models = []
        self._oof_pred_proba = None
        self._oof_pred_model_repeats = None
        self._n_repeats = 0  # Number of n_repeats with at least 1 model fit, if kfold=5 and 8 models have been fit, _n_repeats is 2
        self._n_repeats_finished = 0  # Number of n_repeats finished, if kfold=5 and 8 models have been fit, _n_repeats_finished is 1
        self._k_fold_end = 0  # Number of models fit in current n_repeat (0 if completed), if kfold=5 and 8 models have been fit, _k_fold_end is 3
        self._k = None  # k models per n_repeat, equivalent to kfold value
        self._k_per_n_repeat = []  # k-fold used for each n_repeat. == [5, 10, 3] if first kfold was 5, second was 10, and third was 3
        self._random_state = random_state
        self.low_memory = True
        self._bagged_mode = None
        # _child_oof currently is only set to True for KNN models, that are capable of LOO prediction generation to avoid needing bagging.
        # TODO: Consider moving `_child_oof` logic to a separate class / refactor OOF logic.
        # FIXME: Avoid unnecessary refit during refit_full on `_child_oof=True` models, just re-use the original model.
        self._child_oof = False  # Whether the OOF preds were taken from a single child model (Assumes child can produce OOF preds without bagging).

        try:
            feature_metadata = self.model_base.feature_metadata
        except:
            feature_metadata = None

        eval_metric = kwargs.pop('eval_metric', self.model_base.eval_metric)
        stopping_metric = kwargs.pop('stopping_metric', self.model_base.stopping_metric)

        super().__init__(problem_type=self.model_base.problem_type, eval_metric=eval_metric, stopping_metric=stopping_metric, feature_metadata=feature_metadata, **kwargs)

    def _set_default_params(self):
        default_params = {
            # 'use_child_oof': False,  # [Advanced] Whether to defer to child model for OOF preds and only train a single child.
            'save_bag_folds': True,
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)
        super()._set_default_params()

    def is_valid(self):
        return self.is_fit() and (self._n_repeats == self._n_repeats_finished)

    def can_infer(self):
        return self.is_fit() and self.params.get('save_bag_folds', True)

    def is_stratified(self):
        if self.problem_type in [REGRESSION, QUANTILE, SOFTCLASS]:
            return False
        else:
            return True

    def is_fit(self):
        return len(self.models) != 0

    def is_valid_oof(self):
        return self.is_fit() and (self._child_oof or self._bagged_mode)

    def get_oof_pred_proba(self, **kwargs):
        # TODO: Require is_valid == True (add option param to ignore is_valid)
        return self._oof_pred_proba_func(self._oof_pred_proba, self._oof_pred_model_repeats)

    @staticmethod
    def _oof_pred_proba_func(oof_pred_proba, oof_pred_model_repeats):
        oof_pred_model_repeats_without_0 = np.where(oof_pred_model_repeats == 0, 1, oof_pred_model_repeats)
        if oof_pred_proba.ndim == 2:
            oof_pred_model_repeats_without_0 = oof_pred_model_repeats_without_0[:, None]
        return oof_pred_proba / oof_pred_model_repeats_without_0

    def preprocess(self, X, preprocess_nonadaptive=True, model=None, **kwargs):
        if preprocess_nonadaptive:
            if model is None:
                if not self.models:
                    return X
                model = self.models[0]
            model = self.load_child(model)
            return model.preprocess(X, preprocess_stateful=False)
        else:
            return X

    def _fit(self,
             X,
             y,
             k_fold=5,
             k_fold_start=0,
             k_fold_end=None,
             n_repeats=1,
             n_repeat_start=0,
             time_limit=None,
             sample_weight=None,
             **kwargs):
        use_child_oof = self.params.get('use_child_oof', False)
        fold_fitting_strategy = self.params.get('fold_fitting_strategy', SequentialLocalFoldFittingStrategy)
        if use_child_oof:
            if self.is_fit():
                # TODO: We may want to throw an exception instead and avoid calling fit more than once
                return
            k_fold = 1
            k_fold_end = None
        if k_fold < 1:
            k_fold = 1
        if k_fold_end is None:
            k_fold_end = k_fold

        if self._oof_pred_proba is None and (k_fold_start != 0 or n_repeat_start != 0):
            self._load_oof()
        if n_repeat_start != self._n_repeats_finished:
            raise ValueError(f'n_repeat_start must equal self._n_repeats_finished, values: ({n_repeat_start}, {self._n_repeats_finished})')
        if n_repeats <= n_repeat_start:
            raise ValueError(f'n_repeats must be greater than n_repeat_start, values: ({n_repeats}, {n_repeat_start})')
        if k_fold_start != self._k_fold_end:
            raise ValueError(f'k_fold_start must equal previous k_fold_end, values: ({k_fold_start}, {self._k_fold_end})')
        if k_fold_start >= k_fold_end:
            # TODO: Remove this limitation if n_repeats > 1
            raise ValueError(f'k_fold_end must be greater than k_fold_start, values: ({k_fold_end}, {k_fold_start})')
        if (n_repeats - n_repeat_start) > 1 and k_fold_end != k_fold:
            # TODO: Remove this limitation
            raise ValueError(f'k_fold_end must equal k_fold when (n_repeats - n_repeat_start) > 1, values: ({k_fold_end}, {k_fold})')
        if self._k is not None and self._k != k_fold:
            raise ValueError(f'k_fold must equal previously fit k_fold value for the current n_repeat, values: (({k_fold}, {self._k})')

        model_base = self._get_model_base()
        model_base.rename(name='')
        if self.features is not None:
            model_base.features = self.features
        model_base.feature_metadata = self.feature_metadata  # TODO: Don't pass this here

        if self.model_base is not None:
            self.save_model_base(self.model_base)
            self.model_base = None

        if k_fold == 1:
            self._fit_single(X=X, y=y, model_base=model_base, use_child_oof=use_child_oof, time_limit=time_limit, sample_weight=sample_weight, **kwargs)
            return

        # TODO: Preprocess data here instead of repeatedly
        time_start = time.time()
        kfolds = generate_kfold(X=X, y=y, n_splits=k_fold, stratified=self.is_stratified(), random_state=self._random_state, n_repeats=n_repeats)

        oof_pred_proba, oof_pred_model_repeats = self._construct_empty_oof(X=X, y=y)

        models = []
        fold_start = n_repeat_start * k_fold + k_fold_start
        fold_end = (n_repeats - 1) * k_fold + k_fold_end
        folds_to_fit = fold_end - fold_start
        # noinspection PyCallingNonCallable
        fold_fitting_strategy: AbstractFoldFittingStrategy = fold_fitting_strategy(
            self, X, y, sample_weight, time_limit, time_start, models, oof_pred_proba, oof_pred_model_repeats)
        for j in range(n_repeat_start, n_repeats):  # For each n_repeat
            cur_repeat_count = j - n_repeat_start
            fold_start_n_repeat = fold_start + cur_repeat_count * k_fold
            fold_end_n_repeat = min(fold_start_n_repeat + k_fold, fold_end)

            for i in range(fold_start_n_repeat, fold_end_n_repeat):  # For each fold
                fold_num_in_repeat = i - (j * k_fold)  # The fold in the current repeat set (first fold in set = 0)

                fold_ctx = dict(
                    model_name_suffix=f'S{j + 1}F{fold_num_in_repeat + 1}',  # S5F3 = 3rd fold of the 5th repeat set
                    fold=kfolds[i],
                    is_last_fold=i != (fold_end - 1),
                    folds_to_fit=folds_to_fit,
                    folds_finished=i - fold_start,
                    folds_left=fold_end - i,
                )

                fold_fitting_strategy.schedule_fold_model_fit(model_base, fold_ctx, kwargs)
            if (fold_end_n_repeat != fold_end) or (k_fold == k_fold_end):
                self._k_per_n_repeat.append(k_fold)
        fold_fitting_strategy.after_all_folds_scheduled()
        self.models += models

        self._bagged_mode = True

        if self._oof_pred_proba is None:
            self._oof_pred_proba = oof_pred_proba
            self._oof_pred_model_repeats = oof_pred_model_repeats
        else:
            self._oof_pred_proba += oof_pred_proba
            self._oof_pred_model_repeats += oof_pred_model_repeats

        self._n_repeats = n_repeats
        if k_fold == k_fold_end:
            self._k = None
            self._k_fold_end = 0
            self._n_repeats_finished = self._n_repeats
        else:
            self._k = k_fold
            self._k_fold_end = k_fold_end
            self._n_repeats_finished = self._n_repeats - 1

    def predict_proba(self, X, normalize=None, **kwargs):
        model = self.load_child(self.models[0])
        X = self.preprocess(X, model=model, **kwargs)
        pred_proba = model.predict_proba(X=X, preprocess_nonadaptive=False, normalize=normalize)
        for model in self.models[1:]:
            model = self.load_child(model)
            pred_proba += model.predict_proba(X=X, preprocess_nonadaptive=False, normalize=normalize)
        pred_proba = pred_proba / len(self.models)

        return pred_proba

    def _predict_proba(self, X, normalize=False, **kwargs):
        return self.predict_proba(X=X, normalize=normalize, **kwargs)

    def score_with_oof(self, y, sample_weight=None):
        self._load_oof()
        valid_indices = self._oof_pred_model_repeats > 0
        y = y[valid_indices]
        y_pred_proba = self.get_oof_pred_proba()[valid_indices]
        if sample_weight is not None:
            sample_weight = sample_weight[valid_indices]
        return self.score_with_y_pred_proba(y=y, y_pred_proba=y_pred_proba, sample_weight=sample_weight)

    def _fit_single(self, X, y, model_base, use_child_oof, time_limit, **kwargs):
        if self.is_fit():
            raise AssertionError('Model is already fit.')
        if self._n_repeats != 0:
            raise ValueError(f'n_repeats must equal 0 when fitting a single model with k_fold == 1, value: {self._n_repeats}')
        model_base.name = f'{model_base.name}S1F1'
        model_base.set_contexts(path_context=self.path + model_base.name + os.path.sep)
        time_start_fit = time.time()
        model_base.fit(X=X, y=y, time_limit=time_limit, **kwargs)
        model_base.fit_time = time.time() - time_start_fit
        model_base.predict_time = None
        X_len = len(X)

        # Check if pred_proba is going to take too long
        if time_limit is not None and X_len >= 10000:

            max_allowed_time = time_limit * 1.3  # allow some buffer
            time_left = max(
                max_allowed_time - model_base.fit_time,
                time_limit * 0.1,  # At least 10% of time_limit
                10,  # At least 10 seconds
            )
            # Sample at most 500 rows to estimate prediction time of all rows
            # TODO: Consider moving this into end of abstract model fit for all models.
            #  Currently this only fixes problem when in bagged mode, if not bagging, then inference could still be problamatic
            n_sample = min(500, round(X_len * 0.1))
            frac = n_sample / X_len
            X_sample = X.sample(n=n_sample)
            time_start_predict = time.time()
            model_base.predict_proba(X_sample)
            time_predict_frac = time.time() - time_start_predict
            time_predict_estimate = time_predict_frac / frac
            logger.log(15, f'\t{round(time_predict_estimate, 2)}s\t= Estimated out-of-fold prediction time...')
            if time_predict_estimate > time_left:
                logger.warning(f'\tNot enough time to generate out-of-fold predictions for model. Estimated time required was {round(time_predict_estimate, 2)}s compared to {round(time_left, 2)}s of available time.')
                raise TimeLimitExceeded

        if use_child_oof:
            logger.log(15, '\t`use_child_oof` was specified for this model. It will function similarly to a bagged model, but will only fit one child model.')
            time_start_predict = time.time()
            if model_base._get_tags().get('valid_oof', False):
                self._oof_pred_proba = model_base.get_oof_pred_proba(X=X, y=y)
            else:
                logger.warning('\tWARNING: `use_child_oof` was specified but child model does not have a dedicated `get_oof_pred_proba` method. This model may have heavily overfit validation scores.')
                self._oof_pred_proba = model_base.predict_proba(X=X)
            self._child_oof = True
            model_base.predict_time = time.time() - time_start_predict
            model_base.val_score = model_base.score_with_y_pred_proba(y=y, y_pred_proba=self._oof_pred_proba)
        else:
            self._oof_pred_proba = model_base.predict_proba(X=X)  # TODO: Cheater value, will be overfit to valid set
        self._oof_pred_model_repeats = np.ones(shape=len(X), dtype=np.uint8)
        self._n_repeats = 1
        self._n_repeats_finished = 1
        self._k_per_n_repeat = [1]
        self._bagged_mode = False
        model_base.reduce_memory_size(remove_fit=True, remove_info=False, requires_save=True)
        if not self.params.get('save_bag_folds', True):
            model_base.model = None
        if self.low_memory:
            self.save_child(model_base, verbose=False)
            self.models = [model_base.name]
        else:
            self.models = [model_base]
        self._add_child_times_to_bag(model=model_base)

    # TODO: Augment to generate OOF after shuffling each column in X (Batching), this is the fastest way.
    # TODO: Reduce logging clutter during OOF importance calculation (Currently logs separately for each child)
    # Generates OOF predictions from pre-trained bagged models, assuming X and y are in the same row order as used in .fit(X, y)
    def compute_feature_importance(self,
                                   X,
                                   y,
                                   features=None,
                                   silent=False,
                                   time_limit=None,
                                   is_oof=False,
                                   **kwargs) -> pd.DataFrame:
        if features is None:
            features = self.load_child(model=self.models[0]).features
        if not is_oof:
            return super().compute_feature_importance(X, y, features=features, time_limit=time_limit, silent=silent, **kwargs)
        fi_fold_list = []
        model_index = 0
        num_children = len(self.models)
        if time_limit is not None:
            time_limit_per_child = time_limit / num_children
        else:
            time_limit_per_child = None
        if not silent:
            logging_message = f'Computing feature importance via permutation shuffling for {len(features)} features using out-of-fold (OOF) data aggregated across {num_children} child models...'
            if time_limit is not None:
                logging_message = f'{logging_message} Time limit: {time_limit}s...'
            logger.log(20, logging_message)

        time_start = time.time()
        early_stop = False
        children_completed = 0
        log_final_suffix = ''
        for n_repeat, k in enumerate(self._k_per_n_repeat):
            if is_oof:
                if self._child_oof or not self._bagged_mode:
                    raise AssertionError('Model trained with no validation data cannot get feature importances on training data, please specify new test data to compute feature importances (model=%s)' % self.name)
                kfolds = generate_kfold(X=X, y=y, n_splits=k, stratified=self.is_stratified(), random_state=self._random_state, n_repeats=n_repeat + 1)
                cur_kfolds = kfolds[n_repeat * k:(n_repeat + 1) * k]
            else:
                cur_kfolds = [(None, list(range(len(X))))] * k
            for i, fold in enumerate(cur_kfolds):
                _, test_index = fold
                model = self.load_child(self.models[model_index + i])
                fi_fold = model.compute_feature_importance(X=X.iloc[test_index, :], y=y.iloc[test_index], features=features, time_limit=time_limit_per_child,
                                                           silent=silent, log_prefix='\t', importance_as_list=True, **kwargs)
                fi_fold_list.append(fi_fold)

                children_completed += 1
                if time_limit is not None and children_completed != num_children:
                    time_now = time.time()
                    time_left = time_limit - (time_now - time_start)
                    time_child_average = (time_now - time_start) / children_completed
                    if time_left < (time_child_average * 1.1):
                        log_final_suffix = f' (Early stopping due to lack of time...)'
                        early_stop = True
                        break
            if early_stop:
                break
            model_index += k
        # TODO: DON'T THROW AWAY SAMPLES! USE LARGER N
        fi_list_dict = dict()
        for val in fi_fold_list:
            val = val['importance'].to_dict()  # TODO: Don't throw away stddev information of children
            for key in val:
                if key not in fi_list_dict:
                    fi_list_dict[key] = []
                fi_list_dict[key] += val[key]
        fi_df = _compute_fi_with_stddev(fi_list_dict)

        if not silent:
            logger.log(20, f'\t{round(time.time() - time_start, 2)}s\t= Actual runtime (Completed {children_completed} of {num_children} children){log_final_suffix}')

        return fi_df

    def load_child(self, model, verbose=False) -> AbstractModel:
        if isinstance(model, str):
            child_path = self.create_contexts(self.path + model + os.path.sep)
            return self._child_type.load(path=child_path, verbose=verbose)
        else:
            return model

    def save_child(self, model, verbose=False):
        child = self.load_child(model)
        child.set_contexts(self.path + child.name + os.path.sep)
        child.save(verbose=verbose)

    # TODO: Multiply epochs/n_iterations by some value (such as 1.1) to account for having more training data than bagged models
    def convert_to_refit_full_template(self):
        init_args = self._get_init_args()
        init_args['hyperparameters']['save_bag_folds'] = True  # refit full models must save folds
        init_args['model_base'] = self.convert_to_refitfull_template_child()
        init_args['name'] = init_args['name'] + REFIT_FULL_SUFFIX
        model_full_template = self.__class__(**init_args)
        return model_full_template

    def convert_to_refitfull_template_child(self):
        compressed_params = self._get_compressed_params()
        child_compressed = copy.deepcopy(self._get_model_base())
        child_compressed.feature_metadata = self.feature_metadata  # TODO: Don't pass this here
        child_compressed.params = compressed_params
        return child_compressed

    def _get_init_args(self):
        init_args = dict(
            model_base=self._get_model_base(),
            random_state=self._random_state,
        )
        init_args.update(super()._get_init_args())
        init_args.pop('problem_type')
        init_args.pop('feature_metadata')
        return init_args

    def _get_compressed_params(self, model_params_list=None):
        if model_params_list is None:
            model_params_list = [
                self.load_child(child).get_trained_params()
                for child in self.models
            ]

        model_params_compressed = dict()
        for param in model_params_list[0].keys():
            model_param_vals = [model_params[param] for model_params in model_params_list]
            if all(isinstance(val, bool) for val in model_param_vals):
                counter = Counter(model_param_vals)
                compressed_val = counter.most_common(1)[0][0]
            elif all(isinstance(val, int) for val in model_param_vals):
                compressed_val = round(mean(model_param_vals))
            elif all(isinstance(val, float) for val in model_param_vals):
                compressed_val = mean(model_param_vals)
            else:
                try:
                    counter = Counter(model_param_vals)
                    compressed_val = counter.most_common(1)[0][0]
                except TypeError:
                    compressed_val = model_param_vals[0]
            model_params_compressed[param] = compressed_val
        return model_params_compressed

    def _get_compressed_params_trained(self):
        model_params_list = [
            self.load_child(child).params_trained
            for child in self.models
        ]
        return self._get_compressed_params(model_params_list=model_params_list)

    def _get_model_base(self):
        if self.model_base is None:
            return self.load_model_base()
        else:
            return self.model_base

    def _add_child_times_to_bag(self, model):
        if self.fit_time is None:
            self.fit_time = model.fit_time
        else:
            self.fit_time += model.fit_time

        if self.predict_time is None:
            self.predict_time = model.predict_time
        else:
            self.predict_time += model.predict_time

    @classmethod
    def load(cls, path: str, reset_paths=True, low_memory=True, load_oof=False, verbose=True):
        model = super().load(path=path, reset_paths=reset_paths, verbose=verbose)
        if not low_memory:
            model.persist_child_models(reset_paths=reset_paths)
        if load_oof:
            model._load_oof()
        return model

    @classmethod
    def load_oof(cls, path, verbose=True):
        try:
            oof = load_pkl.load(path=path + 'utils' + os.path.sep + cls._oof_filename, verbose=verbose)
            oof_pred_proba = oof['_oof_pred_proba']
            oof_pred_model_repeats = oof['_oof_pred_model_repeats']
        except FileNotFoundError:
            model = cls.load(path=path, reset_paths=True, verbose=verbose)
            model._load_oof()
            oof_pred_proba = model._oof_pred_proba
            oof_pred_model_repeats = model._oof_pred_model_repeats
        return cls._oof_pred_proba_func(oof_pred_proba=oof_pred_proba, oof_pred_model_repeats=oof_pred_model_repeats)

    def _load_oof(self):
        if self._oof_pred_proba is not None:
            pass
        else:
            oof = load_pkl.load(path=self.path + 'utils' + os.path.sep + self._oof_filename)
            self._oof_pred_proba = oof['_oof_pred_proba']
            self._oof_pred_model_repeats = oof['_oof_pred_model_repeats']

    def persist_child_models(self, reset_paths=True):
        for i, model_name in enumerate(self.models):
            if isinstance(model_name, str):
                child_path = self.create_contexts(self.path + model_name + os.path.sep)
                child_model = self._child_type.load(path=child_path, reset_paths=reset_paths, verbose=True)
                self.models[i] = child_model

    def load_model_base(self):
        return load_pkl.load(path=self.path + 'utils' + os.path.sep + 'model_template.pkl')

    def save_model_base(self, model_base):
        save_pkl.save(path=self.path + 'utils' + os.path.sep + 'model_template.pkl', object=model_base)

    def save(self, path=None, verbose=True, save_oof=True, save_children=False) -> str:
        if path is None:
            path = self.path

        if save_children:
            model_names = []
            for child in self.models:
                child = self.load_child(child)
                child.set_contexts(path + child.name + os.path.sep)
                child.save(verbose=False)
                model_names.append(child.name)
            self.models = model_names

        if save_oof and self._oof_pred_proba is not None:
            save_pkl.save(path=path + 'utils' + os.path.sep + self._oof_filename, object={
                '_oof_pred_proba': self._oof_pred_proba,
                '_oof_pred_model_repeats': self._oof_pred_model_repeats,
            })
            self._oof_pred_proba = None
            self._oof_pred_model_repeats = None

        return super().save(path=path, verbose=verbose)

    # If `remove_fit_stack=True`, variables will be removed that are required to fit more folds and to fit new stacker models which use this model as a base model.
    #  This includes OOF variables.
    def reduce_memory_size(self, remove_fit_stack=False, remove_fit=True, remove_info=False, requires_save=True, reduce_children=False, **kwargs):
        super().reduce_memory_size(remove_fit=remove_fit, remove_info=remove_info, requires_save=requires_save, **kwargs)
        if remove_fit_stack:
            try:
                os.remove(self.path + 'utils' + os.path.sep + self._oof_filename)
            except FileNotFoundError:
                pass
            if requires_save:
                self._oof_pred_proba = None
                self._oof_pred_model_repeats = None
            try:
                os.remove(self.path + 'utils' + os.path.sep + 'model_template.pkl')
            except FileNotFoundError:
                pass
            if requires_save:
                self.model_base = None
            try:
                os.rmdir(self.path + 'utils')
            except OSError:
                pass
        if reduce_children:
            for model in self.models:
                model = self.load_child(model)
                model.reduce_memory_size(remove_fit=remove_fit, remove_info=remove_info, requires_save=requires_save, **kwargs)
                if requires_save and self.low_memory:
                    self.save_child(model=model)

    def _get_model_names(self):
        model_names = []
        for model in self.models:
            if isinstance(model, str):
                model_names.append(model)
            else:
                model_names.append(model.name)
        return model_names

    def get_info(self):
        info = super().get_info()
        children_info = self._get_child_info()
        child_memory_sizes = [child['memory_size'] for child in children_info.values()]
        sum_memory_size_child = sum(child_memory_sizes)
        if child_memory_sizes:
            max_memory_size_child = max(child_memory_sizes)
        else:
            max_memory_size_child = 0
        if self.low_memory:
            max_memory_size = info['memory_size'] + sum_memory_size_child
            min_memory_size = info['memory_size'] + max_memory_size_child
        else:
            max_memory_size = info['memory_size']
            min_memory_size = info['memory_size'] - sum_memory_size_child + max_memory_size_child

        bagged_info = dict(
            child_model_type=self._child_type.__name__,
            num_child_models=len(self.models),
            child_model_names=self._get_model_names(),
            _n_repeats=self._n_repeats,
            # _n_repeats_finished=self._n_repeats_finished,  # commented out because these are too technical
            # _k_fold_end=self._k_fold_end,
            # _k=self._k,
            _k_per_n_repeat=self._k_per_n_repeat,
            _random_state=self._random_state,
            low_memory=self.low_memory,  # If True, then model will attempt to use at most min_memory_size memory by having at most one child in memory. If False, model will use max_memory_size memory.
            bagged_mode=self._bagged_mode,
            max_memory_size=max_memory_size,  # Memory used when all children are loaded into memory at once.
            min_memory_size=min_memory_size,  # Memory used when only the largest child is loaded into memory.
            child_hyperparameters=self._get_model_base().params,
            child_hyperparameters_fit=self._get_compressed_params_trained(),
            child_ag_args_fit=self._get_model_base().params_aux,
        )
        info['bagged_info'] = bagged_info
        info['children_info'] = children_info

        child_features_full = list(set().union(*[child['features'] for child in children_info.values()]))
        info['features'] = child_features_full
        info['num_features'] = len(child_features_full)

        return info

    def get_memory_size(self):
        models = self.models
        self.models = None
        memory_size = super().get_memory_size()
        self.models = models
        return memory_size

    def _get_child_info(self):
        child_info_dict = dict()
        for model in self.models:
            if isinstance(model, str):
                child_path = self.create_contexts(self.path + model + os.path.sep)
                child_info_dict[model] = self._child_type.load_info(child_path)
            else:
                child_info_dict[model.name] = model.get_info()
        return child_info_dict

    def _construct_empty_oof(self, X, y):
        if self.problem_type == MULTICLASS:
            oof_pred_proba = np.zeros(shape=(len(X), len(y.unique())), dtype=np.float32)
        elif self.problem_type == SOFTCLASS:
            oof_pred_proba = np.zeros(shape=y.shape, dtype=np.float32)
        elif self.problem_type == QUANTILE:
            oof_pred_proba = np.zeros(shape=(len(X), len(self.quantile_levels)), dtype=np.float32)
        else:
            oof_pred_proba = np.zeros(shape=len(X), dtype=np.float32)
        oof_pred_model_repeats = np.zeros(shape=len(X), dtype=np.uint8)
        return oof_pred_proba, oof_pred_model_repeats

    def _preprocess_fit_resources(self, silent=False, **kwargs):
        """Pass along to child models to avoid altering up-front"""
        return kwargs

    # TODO: Currently double disk usage, saving model in HPO and also saving model in bag
    def _hyperparameter_tune(self, X, y, k_fold, scheduler_options, preprocess_kwargs=None, **kwargs):
        if len(self.models) != 0:
            raise ValueError('self.models must be empty to call hyperparameter_tune, value: %s' % self.models)

        self.model_base.feature_metadata = self.feature_metadata  # TODO: Move this
        self.model_base.set_contexts(self.path + 'hpo' + os.path.sep)

        # TODO: Preprocess data here instead of repeatedly
        if preprocess_kwargs is None:
            preprocess_kwargs = dict()
        X = self.preprocess(X=X, preprocess=False, fit=True, **preprocess_kwargs)
        kfolds = generate_kfold(X=X, y=y, n_splits=k_fold, stratified=self.is_stratified(), random_state=self._random_state, n_repeats=1)

        train_index, test_index = kfolds[0]
        X_fold, X_val_fold = X.iloc[train_index, :], X.iloc[test_index, :]
        y_fold, y_val_fold = y.iloc[train_index], y.iloc[test_index]
        orig_time = scheduler_options[1]['time_out']
        if orig_time:
            scheduler_options[1]['time_out'] = orig_time * 0.8  # TODO: Scheduler doesn't early stop on final model, this is a safety net. Scheduler should be updated to early stop
        hpo_models, hpo_model_performances, hpo_results = self.model_base.hyperparameter_tune(X=X_fold, y=y_fold, X_val=X_val_fold, y_val=y_val_fold, scheduler_options=scheduler_options, **kwargs)
        scheduler_options[1]['time_out'] = orig_time

        bags = {}
        bags_performance = {}
        for i, (model_name, model_path) in enumerate(hpo_models.items()):
            child: AbstractModel = self._child_type.load(path=model_path)
            y_pred_proba = child.predict_proba(X_val_fold)

            # TODO: Create new Ensemble Here
            bag = copy.deepcopy(self)
            bag.rename(f"{bag.name}{os.path.sep}T{i}")
            bag.set_contexts(self.path_root + bag.name + os.path.sep)

            oof_pred_proba, oof_pred_model_repeats = self._construct_empty_oof(X=X, y=y)
            oof_pred_proba[test_index] += y_pred_proba
            oof_pred_model_repeats[test_index] += 1

            bag.model_base = None
            child.rename('')
            child.set_contexts(bag.path + child.name + os.path.sep)
            bag.save_model_base(child.convert_to_template())

            bag._k = k_fold
            bag._k_fold_end = 1
            bag._n_repeats = 1
            bag._oof_pred_proba = oof_pred_proba
            bag._oof_pred_model_repeats = oof_pred_model_repeats
            child.rename('S1F1')
            child.set_contexts(bag.path + child.name + os.path.sep)
            if not self.params.get('save_bag_folds', True):
                child.model = None
            if bag.low_memory:
                bag.save_child(child, verbose=False)
                bag.models.append(child.name)
            else:
                bag.models.append(child)
            bag.val_score = child.val_score
            bag._add_child_times_to_bag(model=child)

            bag.save()
            bags[bag.name] = bag.path
            bags_performance[bag.name] = bag.val_score

        # TODO: hpo_results likely not correct because no renames
        return bags, bags_performance, hpo_results

    def _more_tags(self):
        return {'valid_oof': True}


""" MXNet neural networks for tabular data containing numerical, categorical, and text fields.
    First performs neural network specific pre-processing of the data.
    Contains separate input modules which are applied to different columns of the data depending on the type of values they contain:
    - Numeric columns are pased through single Dense layer (binary categorical variables are treated as numeric)
    - Categorical columns are passed through separate Embedding layers
    - Text columns are passed through separate LanguageModel layers
    Vectors produced by different input layers are then concatenated and passed to multi-layer MLP model with problem_type determined output layer.
    Hyperparameters are passed as dict params, including options for preprocessing stages.
"""
import json
import logging
import os
import random
import time
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, QuantileTransformer, FunctionTransformer  # PowerTransformer

from autogluon.core import Space
from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION, SOFTCLASS
from autogluon.core.features.types import R_OBJECT, S_TEXT_NGRAM, S_TEXT_AS_CATEGORY
from autogluon.core.utils import try_import_mxboard, try_import_mxnet
from autogluon.core.utils.exceptions import TimeLimitExceeded

#from autogluon.tabular.models
from autogluon.tabular.models.tabular_nn.categorical_encoders import OneHotMergeRaresHandleUnknownEncoder, OrdinalMergeRaresHandleUnknownEncoder
from autogluon.tabular.models.tabular_nn.hyperparameters.parameters import get_default_param
from autogluon.tabular.models.tabular_nn.hyperparameters.searchspaces import get_default_searchspace
from autogluon.core.models.abstract.abstract_model import AbstractNeuralNetworkModel
from autogluon.tabular.models.utils import fixedvals_from_searchspaces

warnings.filterwarnings("ignore", module='sklearn.preprocessing')  # sklearn processing n_quantiles warning
logger = logging.getLogger(__name__)
EPS = 1e-10  # small number


# TODO: Gets stuck after infering feature types near infinitely in nyc-jiashenliu-515k-hotel-reviews-data-in-europe dataset, 70 GB of memory, c5.9xlarge
#  Suspect issue is coming from embeddings due to text features with extremely large categorical counts.
class TabularNeuralNetModel(AbstractNeuralNetworkModel):
    """ Class for neural network models that operate on tabular data.
        These networks use different types of input layers to process different types of data in various columns.

        Attributes:
            _types_of_features (dict): keys = 'continuous', 'skewed', 'onehot', 'embed', 'language'; values = column-names of Dataframe corresponding to the features of this type
            feature_arraycol_map (OrderedDict): maps feature-name -> list of column-indices in df corresponding to this feature
        self.feature_type_map (OrderedDict): maps feature-name -> feature_type string (options: 'vector', 'embed', 'language')
        processor (sklearn.ColumnTransformer): scikit-learn preprocessor object.

        Note: This model always assumes higher values of self.eval_metric indicate better performance.

    """

    # Constants used throughout this class:
    # model_internals_file_name = 'model-internals.pkl' # store model internals here
    unique_category_str = '!missing!' # string used to represent missing values and unknown categories for categorical features. Should not appear in the dataset
    params_file_name = 'net.params' # Stores parameters of final network
    temp_file_name = 'temp_net.params' # Stores temporary network parameters (eg. during the course of training)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        """
        TabularNeuralNetModel object.

        Parameters
        ----------
        path (str): file-path to directory where to save files associated with this model
        name (str): name used to refer to this model
        problem_type (str): what type of prediction problem is this model used for
        eval_metric (func): function used to evaluate performance (Note: we assume higher = better)
        hyperparameters (dict): various hyperparameters for neural network and the NN-specific data processing
        features (list): List of predictive features to use, other features are ignored by the model.
        """
        self.feature_arraycol_map = None
        self.feature_type_map = None
        self.features_to_drop = []  # may change between different bagging folds. TODO: consider just removing these from self.features if it works with bagging
        self.processor = None  # data processor
        self.summary_writer = None
        self.ctx = None
        self.batch_size = None
        self.num_dataloading_workers = None
        self.num_dataloading_workers_inference = 0
        self.params_post_fit = None
        self.num_net_outputs = None
        self._architecture_desc = None
        self.optimizer = None
        self.verbosity = None

    def _set_default_params(self):
        """ Specifies hyperparameter values to use by default """
        default_params = get_default_param(self.problem_type)
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = dict(
            ignored_type_group_raw=[R_OBJECT],
            ignored_type_group_special=[S_TEXT_NGRAM, S_TEXT_AS_CATEGORY],
        )
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params

    def _get_default_searchspace(self):
        return get_default_searchspace(self.problem_type, num_classes=None)

    def set_net_defaults(self, train_dataset, params):
        """ Sets dataset-adaptive default values to use for our neural network """
        if (self.problem_type == MULTICLASS) or (self.problem_type == SOFTCLASS):
            self.num_net_outputs = train_dataset.num_classes
        elif self.problem_type == REGRESSION:
            self.num_net_outputs = 1
            if params['y_range'] is None:  # Infer default y-range
                y_vals = train_dataset.dataset._data[train_dataset.label_index].asnumpy()
                min_y = float(min(y_vals))
                max_y = float(max(y_vals))
                std_y = np.std(y_vals)
                y_ext = params['y_range_extend'] * std_y
                if min_y >= 0:  # infer y must be nonnegative
                    min_y = max(0, min_y-y_ext)
                else:
                    min_y = min_y-y_ext
                if max_y <= 0:  # infer y must be non-positive
                    max_y = min(0, max_y+y_ext)
                else:
                    max_y = max_y+y_ext
                params['y_range'] = (min_y, max_y)
        elif self.problem_type == BINARY:
            self.num_net_outputs = 2
        else:
            raise ValueError("unknown problem_type specified: %s" % self.problem_type)

        if params['layers'] is None:  # Use default choices for MLP architecture
            if self.problem_type == REGRESSION:
                default_layer_sizes = [256, 128]  # overall network will have 4 layers. Input layer, 256-unit hidden layer, 128-unit hidden layer, output layer.
            else:
                default_sizes = [256, 128]  # will be scaled adaptively
                # base_size = max(1, min(self.num_net_outputs, 20)/2.0) # scale layer width based on number of classes
                base_size = max(1, min(self.num_net_outputs, 100) / 50)  # TODO: Updated because it improved model quality and made training far faster
                default_layer_sizes = [defaultsize*base_size for defaultsize in default_sizes]
            layer_expansion_factor = 1  # TODO: consider scaling based on num_rows, eg: layer_expansion_factor = 2-np.exp(-max(0,train_dataset.num_examples-10000))

            max_layer_width = params['max_layer_width']
            params['layers'] = [int(min(max_layer_width, layer_expansion_factor*defaultsize)) for defaultsize in default_layer_sizes]

        if train_dataset.has_vector_features() and params['numeric_embed_dim'] is None:  # Use default choices for numeric embedding size
            vector_dim = train_dataset.dataset._data[train_dataset.vectordata_index].shape[1]  # total dimensionality of vector features
            prop_vector_features = train_dataset.num_vector_features() / float(train_dataset.num_features)  # Fraction of features that are numeric
            min_numeric_embed_dim = 32
            max_numeric_embed_dim = params['max_layer_width']
            params['numeric_embed_dim'] = int(min(max_numeric_embed_dim, max(min_numeric_embed_dim,
                                                    params['layers'][0]*prop_vector_features*np.log10(vector_dim+10) )))
        return

    def _fit(self, X, y, X_val=None, y_val=None,
             time_limit=None, sample_weight=None, num_cpus=1, num_gpus=0, reporter=None, **kwargs):
        """ X (pd.DataFrame): training data features (not necessarily preprocessed yet)
            X_val (pd.DataFrame): test data features (should have same column names as Xtrain)
            y (pd.Series):
            y_val (pd.Series): are pandas Series
            kwargs: Can specify amount of compute resources to utilize (num_cpus, num_gpus).
        """
        start_time = time.time()
        try_import_mxnet()
        import mxnet as mx
        self.verbosity = kwargs.get('verbosity', 2)
        if sample_weight is not None:  # TODO: support
            logger.log(15, "sample_weight not yet supported for TabularNeuralNetModel, this model will ignore them in training.")

        params = self._get_model_params()
        params = fixedvals_from_searchspaces(params)
        if self.feature_metadata is None:
            raise ValueError("Trainer class must set feature_metadata for this model")
        if num_cpus is not None:
            self.num_dataloading_workers = max(1, int(num_cpus/2.0))
        else:
            self.num_dataloading_workers = 1
        if self.num_dataloading_workers == 1:
            self.num_dataloading_workers = 0  # 0 is always faster and uses less memory than 1
        self.batch_size = params['batch_size']
        train_dataset, val_dataset = self.generate_datasets(X=X, y=y, params=params, X_val=X_val, y_val=y_val)
        logger.log(15, "Training data for neural network has: %d examples, %d features (%d vector, %d embedding, %d language)" %
              (train_dataset.num_examples, train_dataset.num_features,
               len(train_dataset.feature_groups['vector']), len(train_dataset.feature_groups['embed']),
               len(train_dataset.feature_groups['language']) ))
        # self._save_preprocessor()  # TODO: should save these things for hyperparam tunning. Need one HP tuner for network-specific HPs, another for preprocessing HPs.

        if num_gpus is not None and num_gpus >= 1:
            self.ctx = mx.gpu()  # Currently cannot use more than 1 GPU
        else:
            self.ctx = mx.cpu()
        self.get_net(train_dataset, params=params)

        if time_limit is not None:
            time_elapsed = time.time() - start_time
            time_limit_orig = time_limit
            time_limit = time_limit - time_elapsed
            if time_limit <= time_limit_orig * 0.4:  # if 60% of time was spent preprocessing, likely not enough time to train model
                raise TimeLimitExceeded

        self.train_net(train_dataset=train_dataset, params=params, val_dataset=val_dataset, initialize=True, setup_trainer=True, time_limit=time_limit, reporter=reporter)
        self.params_post_fit = params
        """
        # TODO: if we don't want to save intermediate network parameters, need to do something like saving in temp directory to clean up after training:
        with make_temp_directory() as temp_dir:
            save_callback = SaveModelCallback(self.model, monitor=self.metric, mode=save_callback_mode, name=self.name)
            with progress_disabled_ctx(self.model) as model:
                original_path = model.path
                model.path = Path(temp_dir)
                model.fit_one_cycle(self.epochs, self.lr, callbacks=save_callback)

                # Load the best one and export it
                model.load(self.name)
                print(f'Model validation metrics: {model.validate()}')
                model.path = original_path
        """

    def get_net(self, train_dataset, params):
        """ Creates a Gluon neural net and context for this dataset.
            Also sets up trainer/optimizer as necessary.
        """
        from .embednet import EmbedNet
        self.set_net_defaults(train_dataset, params)
        self.model = EmbedNet(train_dataset=train_dataset, params=params, num_net_outputs=self.num_net_outputs, ctx=self.ctx)

        # TODO: Below should not occur until at time of saving
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def train_net(self, train_dataset, params, val_dataset=None, initialize=True, setup_trainer=True, time_limit=None, reporter=None):
        """ Trains neural net on given train dataset, early stops based on test_dataset.
            Args:
                train_dataset (TabularNNDataset): training data used to learn network weights
                val_dataset (TabularNNDataset): validation data used for hyperparameter tuning
                initialize (bool): set = False to continue training of a previously trained model, otherwise initializes network weights randomly
                setup_trainer (bool): set = False to reuse the same trainer from a previous training run, otherwise creates new trainer from scratch
        """
        start_time = time.time()
        import mxnet as mx
        logger.log(15, "Training neural network for up to %s epochs..." % params['num_epochs'])
        seed_value = params.get('seed_value')
        if seed_value is not None:  # Set seed
            random.seed(seed_value)
            np.random.seed(seed_value)
            mx.random.seed(seed_value)
        if initialize:  # Initialize the weights of network
            logging.debug("initializing neural network...")
            self.model.collect_params().initialize(ctx=self.ctx)
            self.model.hybridize()
            logging.debug("initialized")
        if setup_trainer:
            # Also setup mxboard to monitor training if visualizer has been specified:
            visualizer = self.params_aux.get('visualizer', 'none')
            if visualizer == 'tensorboard' or visualizer == 'mxboard':
                try_import_mxboard()
                from mxboard import SummaryWriter
                self.summary_writer = SummaryWriter(logdir=self.path, flush_secs=5, verbose=False)
            self.optimizer = self.setup_trainer(params=params, train_dataset=train_dataset)
        best_val_metric = -np.inf  # higher = better
        val_metric = None
        best_val_epoch = 0
        val_improve_epoch = 0  # most recent epoch where validation-score strictly improved
        num_epochs = params['num_epochs']
        if val_dataset is not None:
            y_val = val_dataset.get_labels()
        else:
            y_val = None

        if params['loss_function'] is None:
            if self.problem_type == REGRESSION:
                params['loss_function'] = mx.gluon.loss.L1Loss()
            elif self.problem_type == SOFTCLASS:
                params['loss_function'] = mx.gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=False, from_logits=self.model.from_logits)
            else:
                params['loss_function'] = mx.gluon.loss.SoftmaxCrossEntropyLoss(from_logits=self.model.from_logits)

        loss_func = params['loss_function']
        epochs_wo_improve = params['epochs_wo_improve']
        loss_scaling_factor = 1.0  # we divide loss by this quantity to stabilize gradients

        rescale_losses = {mx.gluon.loss.L1Loss: 'std', mx.gluon.loss.HuberLoss: 'std', mx.gluon.loss.L2Loss: 'var'}  # dict of loss names where we should rescale loss, value indicates how to rescale.
        loss_torescale = [key for key in rescale_losses if isinstance(loss_func, key)]
        if loss_torescale:
            loss_torescale = loss_torescale[0]
            if rescale_losses[loss_torescale] == 'std':
                loss_scaling_factor = np.std(train_dataset.get_labels())/5.0 + EPS  # std-dev of labels
            elif rescale_losses[loss_torescale] == 'var':
                loss_scaling_factor = np.var(train_dataset.get_labels())/5.0 + EPS  # variance of labels
            else:
                raise ValueError("Unknown loss-rescaling type %s specified for loss_func==%s" % (rescale_losses[loss_torescale], loss_func))

        if self.verbosity <= 1:
            verbose_eval = -1  # Print losses every verbose epochs, Never if -1
        elif self.verbosity == 2:
            verbose_eval = 50
        elif self.verbosity == 3:
            verbose_eval = 10
        else:
            verbose_eval = 1

        net_filename = self.path + self.temp_file_name
        if num_epochs == 0:  # use dummy training loop that stops immediately (useful for using NN just for data preprocessing / debugging)
            logger.log(20, "Not training Neural Net since num_epochs == 0.  Neural network architecture is:")
            for batch_idx, data_batch in enumerate(train_dataset.dataloader):
                data_batch = train_dataset.format_batch_data(data_batch, self.ctx)
                with mx.autograd.record():
                    output = self.model(data_batch)
                    labels = data_batch['label']
                    loss = loss_func(output, labels) / loss_scaling_factor
                    # print(str(mx.nd.mean(loss).asscalar()), end="\r")  # prints per-batch losses
                loss.backward()
                self.optimizer.step(labels.shape[0])
                if batch_idx > 0:
                    break
            self.model.save_parameters(net_filename)
            logger.log(15, "untrained Neural Net saved to file")
            return

        start_fit_time = time.time()
        if time_limit is not None:
            time_limit = time_limit - (start_fit_time - start_time)

        # Training Loop:
        for e in range(num_epochs):
            if e == 0:  # special actions during first epoch:
                logger.log(15, "Neural network architecture:")
                logger.log(15, str(self.model))
            cumulative_loss = 0
            for batch_idx, data_batch in enumerate(train_dataset.dataloader):
                data_batch = train_dataset.format_batch_data(data_batch, self.ctx)
                with mx.autograd.record():
                    output = self.model(data_batch)
                    labels = data_batch['label']
                    loss = loss_func(output, labels) / loss_scaling_factor
                    # print(str(mx.nd.mean(loss).asscalar()), end="\r")  # prints per-batch losses
                loss.backward()
                self.optimizer.step(labels.shape[0])
                cumulative_loss += loss.sum()
            train_loss = cumulative_loss/float(train_dataset.num_examples)  # training loss this epoch
            if val_dataset is not None:
                val_metric = self.score(X=val_dataset, y=y_val, metric=self.stopping_metric)
                if np.isnan(val_metric):
                    if e == 0:
                        raise RuntimeError("NaNs encountered in TabularNeuralNetModel training. Features/labels may be improperly formatted or NN weights may have diverged.")
                    else:
                        logger.warning("Warning: NaNs encountered in TabularNeuralNetModel training. Reverting model to last checkpoint without NaNs.")
                        break
                if (val_metric >= best_val_metric) or (e == 0):
                    if val_metric > best_val_metric:
                        val_improve_epoch = e
                    best_val_metric = val_metric
                    best_val_epoch = e
                    # Until functionality is added to restart training from a particular epoch, there is no point in saving params without test_dataset
                    self.model.save_parameters(net_filename)
            else:
                best_val_epoch = e
            if val_dataset is not None:
                if verbose_eval > 0 and e % verbose_eval == 0:
                    logger.log(15, "Epoch %s.  Train loss: %s, Val %s: %s" %
                      (e, train_loss.asscalar(), self.stopping_metric.name, val_metric))
                if self.summary_writer is not None:
                    self.summary_writer.add_scalar(tag='val_'+self.stopping_metric.name,
                                                   value=val_metric, global_step=e)
            else:
                if verbose_eval > 0 and e % verbose_eval == 0:
                    logger.log(15, "Epoch %s.  Train loss: %s" % (e, train_loss.asscalar()))
            if self.summary_writer is not None:
                self.summary_writer.add_scalar(tag='train_loss', value=train_loss.asscalar(), global_step=e)  # TODO: do we want to keep mxboard support?
            if reporter is not None:
                # TODO: Ensure reporter/scheduler properly handle None/nan values after refactor
                if val_dataset is not None and (not np.isnan(val_metric)):  # TODO: This might work without the if statement
                    # epoch must be number of epochs done (starting at 1)
                    reporter(epoch=e + 1,
                             validation_performance=val_metric,  # Higher val_metric = better
                             train_loss=float(train_loss.asscalar()),
                             eval_metric=self.eval_metric.name,
                             greater_is_better=self.eval_metric.greater_is_better)
            if e - val_improve_epoch > epochs_wo_improve:
                break  # early-stop if validation-score hasn't strictly improved in `epochs_wo_improve` consecutive epochs
            if time_limit is not None:
                time_elapsed = time.time() - start_fit_time
                time_epoch_average = time_elapsed / (e+1)
                time_left = time_limit - time_elapsed
                if time_left < time_epoch_average:
                    logger.log(20, f"\tRan out of time, stopping training early. (Stopping on epoch {e})")
                    break

        if val_dataset is not None:
            self.model.load_parameters(net_filename)  # Revert back to best model
            try:
                os.remove(net_filename)
            except FileNotFoundError:
                pass
        if val_dataset is None:
            logger.log(15, "Best model found in epoch %d" % best_val_epoch)
        else:  # evaluate one final time:
            final_val_metric = self.score(X=val_dataset, y=y_val, metric=self.stopping_metric)
            if np.isnan(final_val_metric):
                final_val_metric = -np.inf
            logger.log(15, "Best model found in epoch %d. Val %s: %s" %
                  (best_val_epoch, self.stopping_metric.name, final_val_metric))
        self.params_trained['num_epochs'] = best_val_epoch + 1
        return

    def _predict_proba(self, X, **kwargs):
        """ To align predict with abstract_model API.
            Preprocess here only refers to feature processing steps done by all AbstractModel objects,
            not tabularNN-specific preprocessing steps.
            If X is not DataFrame but instead TabularNNDataset object, we can still produce predictions,
            but cannot use preprocess in this case (needs to be already processed).
        """
        from .tabular_nn_dataset import TabularNNDataset
        if isinstance(X, TabularNNDataset):
            return self._predict_tabular_data(new_data=X, process=False, predict_proba=True)
        elif isinstance(X, pd.DataFrame):
            X = self.preprocess(X, **kwargs)
            return self._predict_tabular_data(new_data=X, process=True, predict_proba=True)
        else:
            raise ValueError("X must be of type pd.DataFrame or TabularNNDataset, not type: %s" % type(X))

    def _predict_tabular_data(self, new_data, process=True, predict_proba=True):  # TODO ensure API lines up with tabular.Model class.
        """ Specific TabularNN method to produce predictions on new (unprocessed) data.
            Returns 1D numpy array unless predict_proba=True and task is multi-class classification (not binary).
            Args:
                new_data (pd.Dataframe or TabularNNDataset): new data to make predictions on.
                If you want to make prediction for just a single row of new_data, pass in: new_data.iloc[[row_index]]
                process (bool): should new data be processed (if False, new_data must be TabularNNDataset)
                predict_proba (bool): should we output class-probabilities (not used for regression)
        """
        from .tabular_nn_dataset import TabularNNDataset
        import mxnet as mx
        if process:
            new_data = self.process_test_data(new_data, batch_size=self.batch_size, num_dataloading_workers=self.num_dataloading_workers_inference, labels=None)
        if not isinstance(new_data, TabularNNDataset):
            raise ValueError("new_data must of of type TabularNNDataset if process=False")
        if self.problem_type == REGRESSION or not predict_proba:
            preds = mx.nd.zeros((new_data.num_examples,1))
        else:
            preds = mx.nd.zeros((new_data.num_examples, self.num_net_outputs))
        i = 0
        for batch_idx, data_batch in enumerate(new_data.dataloader):
            data_batch = new_data.format_batch_data(data_batch, self.ctx)
            preds_batch = self.model(data_batch)
            batch_size = len(preds_batch)
            if self.problem_type != REGRESSION:
                if not predict_proba: # need to take argmax
                    preds_batch = mx.nd.argmax(preds_batch, axis=1, keepdims=True)
                else: # need to take softmax
                    preds_batch = mx.nd.softmax(preds_batch, axis=1)
            preds[i:(i+batch_size)] = preds_batch
            i = i+batch_size
        if self.problem_type == REGRESSION or not predict_proba:
            return preds.asnumpy().flatten()  # return 1D numpy array
        elif self.problem_type == BINARY and predict_proba:
            return preds[:,1].asnumpy()  # for binary problems, only return P(Y==+1)

        return preds.asnumpy()  # return 2D numpy array

    def generate_datasets(self, X, y, params, X_val=None, y_val=None):
        impute_strategy = params['proc.impute_strategy']
        max_category_levels = params['proc.max_category_levels']
        skew_threshold = params['proc.skew_threshold']
        embed_min_categories = params['proc.embed_min_categories']
        use_ngram_features = params['use_ngram_features']

        from .tabular_nn_dataset import TabularNNDataset
        if isinstance(X, TabularNNDataset):
            train_dataset = X
        else:
            X = self.preprocess(X)
            if self.features is None:
                self.features = list(X.columns)
            train_dataset = self.process_train_data(
                df=X, labels=y, batch_size=self.batch_size, num_dataloading_workers=self.num_dataloading_workers,
                impute_strategy=impute_strategy, max_category_levels=max_category_levels, skew_threshold=skew_threshold, embed_min_categories=embed_min_categories, use_ngram_features=use_ngram_features,
            )
        if X_val is not None:
            if isinstance(X_val, TabularNNDataset):
                val_dataset = X_val
            else:
                X_val = self.preprocess(X_val)
                val_dataset = self.process_test_data(df=X_val, labels=y_val, batch_size=self.batch_size, num_dataloading_workers=self.num_dataloading_workers_inference)
        else:
            val_dataset = None
        return train_dataset, val_dataset

    def process_test_data(self, df, batch_size, num_dataloading_workers, labels=None):
        """ Process train or test DataFrame into a form fit for neural network models.
        Args:
            df (pd.DataFrame): Data to be processed (X)
            labels (pd.Series): labels to be processed (y)
            test (bool): Is this test data where each datapoint should be processed separately using predetermined preprocessing steps.
                         Otherwise preprocessor uses all data to determine propreties like best scaling factors, number of categories, etc.
        Returns:
            Dataset object
        """
        from .tabular_nn_dataset import TabularNNDataset
        warnings.filterwarnings("ignore", module='sklearn.preprocessing') # sklearn processing n_quantiles warning
        if labels is not None and len(labels) != len(df):
            raise ValueError("Number of examples in Dataframe does not match number of labels")
        if (self.processor is None or self._types_of_features is None
           or self.feature_arraycol_map is None or self.feature_type_map is None):
            raise ValueError("Need to process training data before test data")
        if self.features_to_drop:
            drop_cols = [col for col in df.columns if col in self.features_to_drop]
            if drop_cols:
                df = df.drop(columns=drop_cols)

        df = self.processor.transform(df) # 2D numpy array. self.feature_arraycol_map, self.feature_type_map have been previously set while processing training data.
        return TabularNNDataset(df, self.feature_arraycol_map, self.feature_type_map,
                                batch_size=batch_size, num_dataloading_workers=num_dataloading_workers,
                                problem_type=self.problem_type, labels=labels, is_test=True)

    def process_train_data(self, df, batch_size, num_dataloading_workers, impute_strategy, max_category_levels, skew_threshold, embed_min_categories, use_ngram_features, labels):
        """ Preprocess training data and create self.processor object that can be used to process future data.
            This method should only be used once per TabularNeuralNetModel object, otherwise will produce Warning.

        # TODO no label processing for now
        # TODO: language features are ignored for now
        # TODO: add time/ngram features
        # TODO: no filtering of data-frame columns based on statistics, e.g. categorical columns with all unique variables or zero-variance features.
                This should be done in default_learner class for all models not just TabularNeuralNetModel...
        """
        from .tabular_nn_dataset import TabularNNDataset
        warnings.filterwarnings("ignore", module='sklearn.preprocessing')  # sklearn processing n_quantiles warning
        if set(df.columns) != set(self.features):
            raise ValueError("Column names in provided Dataframe do not match self.features")
        if labels is None:
            raise ValueError("Attempting process training data without labels")
        if len(labels) != len(df):
            raise ValueError("Number of examples in Dataframe does not match number of labels")

        self._types_of_features, df = self._get_types_of_features(df, skew_threshold=skew_threshold, embed_min_categories=embed_min_categories, use_ngram_features=use_ngram_features)  # dict with keys: : 'continuous', 'skewed', 'onehot', 'embed', 'language', values = column-names of df
        logger.log(15, "AutoGluon Neural Network infers features are of the following types:")
        logger.log(15, json.dumps(self._types_of_features, indent=4))
        logger.log(15, "\n")
        self.processor = self._create_preprocessor(impute_strategy=impute_strategy, max_category_levels=max_category_levels)
        df = self.processor.fit_transform(df) # 2D numpy array
        self.feature_arraycol_map = self._get_feature_arraycol_map(max_category_levels=max_category_levels)  # OrderedDict of feature-name -> list of column-indices in df corresponding to this feature
        num_array_cols = np.sum([len(self.feature_arraycol_map[key]) for key in self.feature_arraycol_map])  # should match number of columns in processed array
        if num_array_cols != df.shape[1]:
            raise ValueError("Error during one-hot encoding data processing for neural network. Number of columns in df array does not match feature_arraycol_map.")

        self.feature_type_map = self._get_feature_type_map()  # OrderedDict of feature-name -> feature_type string (options: 'vector', 'embed', 'language')
        return TabularNNDataset(df, self.feature_arraycol_map, self.feature_type_map,
                                batch_size=batch_size, num_dataloading_workers=num_dataloading_workers,
                                problem_type=self.problem_type, labels=labels, is_test=False)

    def setup_trainer(self, params, train_dataset=None):
        """ Set up optimizer needed for training.
            Network must first be initialized before this.
        """
        import mxnet as mx
        optimizer_opts = {'learning_rate': params['learning_rate'], 'wd': params['weight_decay'], 'clip_gradient': params['clip_gradient']}
        if 'lr_scheduler' in params and params['lr_scheduler'] is not None:
            if train_dataset is None:
                raise ValueError("train_dataset cannot be None when lr_scheduler is specified.")
            base_lr = params.get('base_lr', 1e-6)
            target_lr = params.get('target_lr', 1.0)
            warmup_epochs = params.get('warmup_epochs', 10)
            lr_decay = params.get('lr_decay', 0.1)
            lr_mode = params['lr_scheduler']
            num_batches = train_dataset.num_examples // params['batch_size']
            lr_decay_epoch = [max(warmup_epochs, int(params['num_epochs']/3)), max(warmup_epochs+1, int(params['num_epochs']/2)),
                              max(warmup_epochs+2, int(2*params['num_epochs']/3))]
            from .utils.lr_scheduler import LRSequential, LRScheduler
            lr_scheduler = LRSequential([
                LRScheduler('linear', base_lr=base_lr, target_lr=target_lr, nepochs=warmup_epochs, iters_per_epoch=num_batches),
                LRScheduler(lr_mode, base_lr=target_lr, target_lr=base_lr, nepochs=params['num_epochs'] - warmup_epochs,
                            iters_per_epoch=num_batches, step_epoch=lr_decay_epoch, step_factor=lr_decay, power=2)
            ])
            optimizer_opts['lr_scheduler'] = lr_scheduler
        if params['optimizer'] == 'sgd':
            if 'momentum' in params:
                optimizer_opts['momentum'] = params['momentum']
            optimizer = mx.gluon.Trainer(self.model.collect_params(), 'sgd', optimizer_opts)
        elif params['optimizer'] == 'adam':  # TODO: Can we try AdamW?
            optimizer = mx.gluon.Trainer(self.model.collect_params(), 'adam', optimizer_opts)
        else:
            raise ValueError("Unknown optimizer specified: %s" % params['optimizer'])
        return optimizer

    def _get_feature_arraycol_map(self, max_category_levels):
        """ Returns OrderedDict of feature-name -> list of column-indices in processed data array corresponding to this feature """
        feature_preserving_transforms = set(['continuous','skewed', 'ordinal', 'language'])  # these transforms do not alter dimensionality of feature
        feature_arraycol_map = {}  # unordered version
        current_colindex = 0
        for transformer in self.processor.transformers_:
            transformer_name = transformer[0]
            transformed_features = transformer[2]
            if transformer_name in feature_preserving_transforms:
                for feature in transformed_features:
                    if feature in feature_arraycol_map:
                        raise ValueError("same feature is processed by two different column transformers: %s" % feature)
                    feature_arraycol_map[feature] = [current_colindex]
                    current_colindex += 1
            elif transformer_name == 'onehot':
                oh_encoder = [step for (name, step) in transformer[1].steps if name == 'onehot'][0]
                for i in range(len(transformed_features)):
                    feature = transformed_features[i]
                    if feature in feature_arraycol_map:
                        raise ValueError("same feature is processed by two different column transformers: %s" % feature)
                    oh_dimensionality = min(len(oh_encoder.categories_[i]), max_category_levels+1)
                    feature_arraycol_map[feature] = list(range(current_colindex, current_colindex+oh_dimensionality))
                    current_colindex += oh_dimensionality
            else:
                raise ValueError("unknown transformer encountered: %s" % transformer_name)
        return OrderedDict([(key, feature_arraycol_map[key]) for key in feature_arraycol_map])

    def _get_feature_type_map(self):
        """ Returns OrderedDict of feature-name -> feature_type string (options: 'vector', 'embed', 'language') """
        if self.feature_arraycol_map is None:
            raise ValueError("must first call _get_feature_arraycol_map() before _get_feature_type_map()")
        vector_features = self._types_of_features['continuous'] + self._types_of_features['skewed'] + self._types_of_features['onehot']
        feature_type_map = OrderedDict()
        for feature_name in self.feature_arraycol_map:
            if feature_name in vector_features:
                feature_type_map[feature_name] = 'vector'
            elif feature_name in self._types_of_features['embed']:
                feature_type_map[feature_name] = 'embed'
            elif feature_name in self._types_of_features['language']:
                feature_type_map[feature_name] = 'language'
            else:
                raise ValueError("unknown feature type encountered")
        return feature_type_map

    def _create_preprocessor(self, impute_strategy, max_category_levels):
        """ Defines data encoders used to preprocess different data types and creates instance variable which is sklearn ColumnTransformer object """
        if self.processor is not None:
            Warning("Attempting to process training data for TabularNeuralNetModel, but previously already did this.")
        continuous_features = self._types_of_features['continuous']
        skewed_features = self._types_of_features['skewed']
        onehot_features = self._types_of_features['onehot']
        embed_features = self._types_of_features['embed']
        language_features = self._types_of_features['language']
        transformers = []  # order of various column transformers in this list is important!
        if continuous_features:
            continuous_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy=impute_strategy)),
                ('scaler', StandardScaler())])
            transformers.append( ('continuous', continuous_transformer, continuous_features) )
        if skewed_features:
            power_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy=impute_strategy)),
                ('quantile', QuantileTransformer(output_distribution='normal')) ])  # Or output_distribution = 'uniform'
            transformers.append( ('skewed', power_transformer, skewed_features) )
        if onehot_features:
            onehot_transformer = Pipeline(steps=[
                # TODO: Consider avoiding converting to string for improved memory efficiency
                ('to_str', FunctionTransformer(convert_df_dtype_to_str)),
                ('imputer', SimpleImputer(strategy='constant', fill_value=self.unique_category_str)),
                ('onehot', OneHotMergeRaresHandleUnknownEncoder(max_levels=max_category_levels, sparse=False))])  # test-time unknown values will be encoded as all zeros vector
            transformers.append( ('onehot', onehot_transformer, onehot_features) )
        if embed_features:  # Ordinal transformer applied to convert to-be-embedded categorical features to integer levels
            ordinal_transformer = Pipeline(steps=[
                ('to_str', FunctionTransformer(convert_df_dtype_to_str)),
                ('imputer', SimpleImputer(strategy='constant', fill_value=self.unique_category_str)),
                ('ordinal', OrdinalMergeRaresHandleUnknownEncoder(max_levels=max_category_levels))])  # returns 0-n when max_category_levels = n-1. category n is reserved for unknown test-time categories.
            transformers.append( ('ordinal', ordinal_transformer, embed_features) )
        if language_features:
            raise NotImplementedError("language_features cannot be used at the moment")
        return ColumnTransformer(transformers=transformers)  # numeric features are processed in the same order as in numeric_features vector, so feature-names remain the same.

    def save(self, path: str = None, verbose=True) -> str:
        if self.model is not None:
            self._architecture_desc = self.model.architecture_desc
        temp_model = self.model
        temp_sw = self.summary_writer
        self.model = None
        self.summary_writer = None
        path_final = super().save(path=path, verbose=verbose)
        self.model = temp_model
        self.summary_writer = temp_sw
        self._architecture_desc = None

        # Export model
        if self.model is not None:
            params_filepath = path_final + self.params_file_name
            # TODO: Don't use os.makedirs here, have save_parameters function in tabular_nn_model that checks if local path or S3 path
            os.makedirs(os.path.dirname(path_final), exist_ok=True)
            self.model.save_parameters(params_filepath)
        return path_final

    @classmethod
    def load(cls, path: str, reset_paths=True, verbose=True):
        model: TabularNeuralNetModel = super().load(path=path, reset_paths=reset_paths, verbose=verbose)
        if model._architecture_desc is not None:
            from .embednet import EmbedNet
            model.model = EmbedNet(architecture_desc=model._architecture_desc, ctx=model.ctx)  # recreate network from architecture description
            model._architecture_desc = None
            # TODO: maybe need to initialize/hybridize?
            model.model.load_parameters(model.path + model.params_file_name, ctx=model.ctx)
            model.summary_writer = None
        return model

    def _hyperparameter_tune(self, X, y, X_val, y_val, scheduler_options, **kwargs):
        """ Performs HPO and sets self.params to best hyperparameter values """
        try_import_mxnet()
        from .tabular_nn_trial import tabular_nn_trial
        from .tabular_nn_dataset import TabularNNDataset

        time_start = time.time()
        self.verbosity = kwargs.get('verbosity', 2)
        logger.log(15, "Beginning hyperparameter tuning for Neural Network...")
        self._set_default_searchspace()  # changes non-specified default hyperparams from fixed values to search-spaces.
        if self.feature_metadata is None:
            raise ValueError("Trainer class must set feature_metadata for this model")
        scheduler_cls, scheduler_params = scheduler_options  # Unpack tuple
        if scheduler_cls is None or scheduler_params is None:
            raise ValueError("scheduler_cls and scheduler_params cannot be None for hyperparameter tuning")
        num_cpus = scheduler_params['resource']['num_cpus']

        params_copy = self._get_params()

        self.num_dataloading_workers = max(1, int(num_cpus/2.0))
        self.batch_size = params_copy['batch_size']
        train_dataset, val_dataset = self.generate_datasets(X=X, y=y, params=params_copy, X_val=X_val, y_val=y_val)
        train_path = self.path + "train"
        val_path = self.path + "validation"
        train_dataset.save(file_prefix=train_path)
        val_dataset.save(file_prefix=val_path)

        if not np.any([isinstance(params_copy[hyperparam], Space) for hyperparam in params_copy]):
            logger.warning("Warning: Attempting to do hyperparameter optimization without any search space (all hyperparameters are already fixed values)")
        else:
            logger.log(15, "Hyperparameter search space for Neural Network: ")
            for hyperparam in params_copy:
                if isinstance(params_copy[hyperparam], Space):
                    logger.log(15, str(hyperparam)+ ":   "+str(params_copy[hyperparam]))

        util_args = dict(
            train_path=train_path,
            val_path=val_path,
            model=self,
            time_start=time_start,
            time_limit=scheduler_params['time_out'],
            fit_kwargs=scheduler_params['resource'],
        )
        tabular_nn_trial.register_args(util_args=util_args, **params_copy)
        scheduler = scheduler_cls(tabular_nn_trial, **scheduler_params)
        if ('dist_ip_addrs' in scheduler_params) and (len(scheduler_params['dist_ip_addrs']) > 0):
            # TODO: Ensure proper working directory setup on remote machines
            # This is multi-machine setting, so need to copy dataset to workers:
            logger.log(15, "Uploading preprocessed data to remote workers...")
            scheduler.upload_files([
                train_path + TabularNNDataset.DATAOBJ_SUFFIX,
                train_path + TabularNNDataset.DATAVALUES_SUFFIX,
                val_path + TabularNNDataset.DATAOBJ_SUFFIX,
                val_path + TabularNNDataset.DATAVALUES_SUFFIX
            ])  # TODO: currently does not work.
            logger.log(15, "uploaded")

        scheduler.run()
        scheduler.join_jobs()

        return self._get_hpo_results(scheduler=scheduler, scheduler_params=scheduler_params, time_start=time_start)

    def get_info(self):
        info = super().get_info()
        info['hyperparameters_post_fit'] = self.params_post_fit
        return info

    def reduce_memory_size(self, remove_fit=True, requires_save=True, **kwargs):
        super().reduce_memory_size(remove_fit=remove_fit, requires_save=requires_save, **kwargs)
        if remove_fit and requires_save:
            self.optimizer = None

    def _get_default_stopping_metric(self):
        return self.eval_metric


def convert_df_dtype_to_str(df):
    return df.astype(str)


""" General TODOs:

- Automatically decrease batch-size if memory issue arises

- Retrain final NN on full dataset (train+val). How to ensure stability here?
- OrdinalEncoder class in sklearn currently cannot handle rare categories or unknown ones at test-time, so we have created our own Encoder in category_encoders.py
There is open PR in sklearn to address this: https://github.com/scikit-learn/scikit-learn/pull/13833/files
Currently, our code uses category_encoders package (BSD license) instead: https://github.com/scikit-learn-contrib/categorical-encoding
Once PR is merged into sklearn, may want to switch: category_encoders.Ordinal -> sklearn.preprocessing.OrdinalEncoder in preprocess_train_data()

- Save preprocessed data so that we can do HPO of neural net hyperparameters more efficiently, while also doing HPO of preprocessing hyperparameters?
      Naive full HPO method requires redoing preprocessing in each trial even if we did not change preprocessing hyperparameters.
      Alternative is we save each proprocessed dataset & corresponding TabularNeuralNetModel object with its unique param names in the file. Then when we try a new HP-config, we first try loading from file if one exists.

"""
