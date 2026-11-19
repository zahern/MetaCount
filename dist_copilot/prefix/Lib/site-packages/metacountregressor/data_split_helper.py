import numpy as np
import pandas as pd




class DataProcessor:
    def __init__(self, x_data, y_data, kwargs):
        self._obj_1 = kwargs.get('_obj_1')
        self._obj_2 = kwargs.get('_obj_2')
        self.test_percentage = float(kwargs.get('test_percentage', 0))
        self.val_percentage = float(kwargs.get('val_percentage', 0))
        self.is_multi = self.test_percentage != 0
        self._x_data = x_data
        self._y_data = y_data
        self._process_data(kwargs)

    def _process_data(self, kwargs):
        if self._obj_1 == 'MAE' or self._obj_2 in ["MAE", 'RMSE', 'MSE', 'RMSE_IN', 'RMSE_TEST']:
            self._handle_special_conditions(kwargs)
        else:
            self._standard_data_partition()

        self._characteristics_names = list(self._x_data.columns)
        self._max_group_all_means = 1
        self._exclude_this_test = [4]

    def _handle_special_conditions(self, kwargs):
        if 'panels' in kwargs:
            self._process_panels_data(kwargs)
        else:
            self._standard_data_partition()

    def _process_panels_data(self, kwargs):
        group_key = kwargs['group']
        panels_key = kwargs['panels']

        # Process groups and panels
        self._x_data[group_key] = self._x_data[group_key].astype('category').cat.codes
        try:
            self._x_data[panels_key] = self._x_data[panels_key].rank(method='dense').astype(int)
            self._x_data[panels_key] -= self._x_data[panels_key].min() - 1
        except KeyError:
            pass

        # Create training and test datasets
        unique_ids = np.unique(self._x_data[panels_key])
        training_size = int((1 - self.test_percentage - self.val_percentage) * len(unique_ids))
        training_ids = np.random.choice(unique_ids, training_size, replace=False)

        train_idx = self._x_data.index[self._x_data[panels_key].isin(training_ids)]
        test_idx = self._x_data.index[~self._x_data[panels_key].isin(training_ids)]

        self._create_datasets(train_idx, test_idx)

    def _standard_data_partition(self):
        total_samples = len(self._x_data)
        training_size = int((1 - self.test_percentage - self.val_percentage) * total_samples)
        training_indices = np.random.choice(total_samples, training_size, replace=False)

        train_idx = np.array([i for i in range(total_samples) if i in training_indices])
        test_idx = np.array([i for i in range(total_samples) if i not in training_indices])

        self._create_datasets(train_idx, test_idx)

    def _create_datasets(self, train_idx, test_idx):
        self.df_train = self._x_data.loc[train_idx, :]
        self.df_test = self._x_data.loc[test_idx, :]
        self.y_train = self._y_data.loc[train_idx, :]
        self.y_test = self._y_data.loc[test_idx, :]

        self._x_data_test = self.df_test.copy()
        self._y_data_test = self.y_test.astype('float').copy()
        self._x_data = self.df_train.copy()
        self._y_data = self.y_train.astype('float').copy()

        # Handle different shapes
        if self._x_data.ndim == 2:  # Typical DataFrame
            self._samples, self._characteristics = self._x_data.shape
            self._panels = None
        elif self._x_data.ndim == 3:  # 3D structure, e.g., Panel or similar
            self._samples, self._panels, self._characteristics = self._x_data.shape








