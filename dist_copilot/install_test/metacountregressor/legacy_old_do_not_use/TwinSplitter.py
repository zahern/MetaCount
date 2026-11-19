import pandas as pd
import numpy as np
import twinning
from sqlalchemy.dialects.mssql.information_schema import columns
from sqlalchemy.testing import skip_test

import helperprocess






class TwinSplitterWithPanels:
    def __init__(self, x_data, y_data, panel_id, train_ratio=0.7, test_ratio=0.2, val_ratio=0.1):
        """
        Initialize the TwinSplitter with x_data, y_data, panel_id, and split ratios.
        :param x_data: Input features (numpy array or pandas DataFrame).
        :param y_data: Labels corresponding to x_data.
        :param panel_id: Panel identifiers for each row in the dataset.
        :param train_ratio: Proportion of panels to use for training.
        :param test_ratio: Proportion of panels to use for testing.
        :param val_ratio: Proportion of panels to use for validation.
        """
        # Combine x_data, y_data, and panel_id into a single DataFrame
        self.data = pd.DataFrame(x_data, columns=[f'Feature{i+1}' for i in range(x_data.shape[1])])
        self.data['y'] = y_data
        self.data['panel_id'] = panel_id

        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio

        # Ensure the ratios sum to 1
        assert np.isclose(train_ratio + test_ratio + val_ratio, 1), "Ratios must sum to 1."

    def split_by_panel(self):
        """
        Split the data into train, test, and validation sets based on unique panels.
        :return: (train_x, train_y), (test_x, test_y), (val_x, val_y)
        """
        # Step 1: Extract unique panels
        unique_panels = self.data['panel_id'].unique()
        total_panels = len(unique_panels)
        k = 3  # Number of splits (train, test, validation)

        # Step 2: Use `twinning.multiplet` to split panels into train, test, and validation
        panel_indices = twinning.multiplet(unique_panels.reshape(-1, 1), k)

        # Step 3: Assign panels to train, test, and validation
        train_panels = unique_panels[panel_indices == 0]
        test_panels = unique_panels[panel_indices == 1]
        val_panels = unique_panels[panel_indices == 2]

        # Step 4: Filter rows in the original data based on panel assignments
        train_data = self.data[self.data['panel_id'].isin(train_panels)]
        test_data = self.data[self.data['panel_id'].isin(test_panels)]
        val_data = self.data[self.data['panel_id'].isin(val_panels)]

        # Step 5: Separate x_data and y_data for each split
        train_x, train_y = train_data.drop(columns=['y', 'panel_id']).values, train_data['y'].values
        test_x, test_y = test_data.drop(columns=['y', 'panel_id']).values, test_data['y'].values
        val_x, val_y = val_data.drop(columns=['y', 'panel_id']).values, val_data['y'].values

        return (train_x, train_y), (test_x, test_y), (val_x, val_y)









class TwinSplitterWithLabels:
    def __init__(self, x_data, y_data, train_ratio=0.7, test_ratio=0.2, val_ratio=0.1):
        """
        Initialize the TwinSplitter with x_data, y_data, and split ratios.
        :param x_data: Input features (numpy array or pandas DataFrame).
        :param y_data: Labels corresponding to x_data.
        :param train_ratio: Proportion of data to use for training.
        :param test_ratio: Proportion of data to use for testing.
        :param val_ratio: Proportion of data to use for validation.
        """
        # Join x_data and y_data into a single dataset for splitting
        self.data = np.hstack((x_data, y_data.reshape(-1, 1)))  # Combine features and labels
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio

        # Ensure the ratios sum to 1
        assert np.isclose(train_ratio + test_ratio + val_ratio, 1), "Ratios must sum to 1."

    def split_data(self):
        """
        Split the combined dataset into train, test, and validation sets.
        :return: (train_x, train_y), (test_x, test_y), (val_x, val_y)
        """
        total_samples = self.data.shape[0]
        k = int(1 / min(self.train_ratio, self.test_ratio, self.val_ratio))

        # Generate balanced splits using `twinning.multiplet`
        split_ids = twinning.multiplet(self.data, k)

        # Define split cutoffs
        train_cutoff = int(self.train_ratio * total_samples)
        test_cutoff = train_cutoff + int(self.test_ratio * total_samples)

        # Assign data points to splits
        train_data = self.data[split_ids < train_cutoff // (total_samples // k)]
        test_data = self.data[(split_ids >= train_cutoff // (total_samples // k)) &
                              (split_ids < test_cutoff // (total_samples // k))]
        val_data = self.data[split_ids >= test_cutoff // (total_samples // k)]

        # Separate x_data and y_data for each split
        train_x, train_y = train_data[:, :-1], train_data[:, -1]
        test_x, test_y = test_data[:, :-1], test_data[:, -1]
        val_x, val_y = val_data[:, :-1], val_data[:, -1]

        return (train_x, train_y), (test_x, test_y), (val_x, val_y)



import pandas as pd
import numpy as np
import twinning

class PanelSplitterWithTwinning:
    def __init__(self, x_data, y_data, panel_id, train_ratio=0.7, test_ratio=0.2, val_ratio=0.1):
        """
        Initialize the PanelSplitter with x_data, y_data, panel_id, and split ratios.
        :param x_data: Input features (numpy array or pandas DataFrame).
        :param y_data: Labels corresponding to x_data.
        :param panel_id: Panel identifiers for each row in the dataset.
        :param train_ratio: Proportion of panels to use for training.
        :param test_ratio: Proportion of panels to use for testing.
        :param val_ratio: Proportion of panels to use for validation.
        """
        # Combine x_data, y_data, and panel_id into a single DataFrame
        if hasattr(x_data, 'columns'):
            self.data = x_data
        else:
            self.data = pd.DataFrame(x_data.values, columns=[f'Feature{i+1}' for i in range(x_data.shape[1])])
        self.data['y'] = y_data
        self.data['panel_id'] = panel_id
        non_numeric_cols = self.data.select_dtypes(exclude=['number']).columns
        non_numeric_cols = non_numeric_cols.drop('panel_id',
                                                 errors='ignore')  # Ensure 'panel_id' is not dropped accidentally

        # Step 2: Save non-numerical columns with 'panel_id'
        non_numeric_data = self.data[['panel_id'] + list(non_numeric_cols)].copy()
        self.non_numeric_data = non_numeric_data
        # Step 3: Drop non-numerical columns from the original DataFrame
        self.data= self.data.drop(columns=non_numeric_cols)

        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio

        # Ensure the ratios sum to 1
        assert np.isclose(train_ratio + test_ratio + val_ratio, 1), "Ratios must sum to 1."

    #def split_panels_with_twinning(self):
    def split_panels_with_twinning(self):
        """
        Split the data into train, test, and validation sets using Twinning and panel ratios.
        :return: (train_x, train_y), (test_x, test_y), (val_x, val_y)
        """
        # Step 1: Identify unique panels (based on the first occurrence in the dataset)
        unique_panels = self.data['panel_id'].drop_duplicates().values

        # Step 2: Calculate the mean representation of panels (if needed for Twinning)
        mean_of_panels = (
            self.data.groupby('panel_id')
            .mean()
            .reset_index()
        )

        # Step 3: Use Twinning's multiplet to split panels into train, test, and validation
        # Ensure the `k` is based on the ratios provided
        total_ratio = self.train_ratio + self.test_ratio + self.val_ratio
        train_split = self.train_ratio / total_ratio
        test_split = self.test_ratio / total_ratio
        val_split = self.val_ratio / total_ratio

        k = np.round(1 / min(train_split, test_split, val_split)).astype(int)  # Number of splits
        panel_indices = twinning.multiplet(mean_of_panels.values, int(k))

        # Ensure `panel_indices` matches the number of unique panels
        if len(panel_indices) != len(unique_panels):
            raise ValueError("Length of panel indices does not match the number of unique panels.")

        # this is only 3 parts. so k
        k_train_split = np.round(train_split*k).astype(int)
        k_test_split = np.round(k*test_split).astype(int)
        k_val_split = np.round(k*val_split).astype(int)

        if (k_val_split+k_test_split+k_train_split) != k:
            raise ValueError('split does not work, rectify logic.')

        # Step 4: Assign panels to train, test, and validation
        train_panels = unique_panels[(panel_indices % k) < k_train_split]  # Train panels
        test_panels = unique_panels[
            ((panel_indices % k) >= k_train_split) & ((panel_indices % k) < (k - k_val_split))]  # Test panels
        val_panels = unique_panels[(panel_indices % k) >= (k - k_val_split)]  # Validation panelsValidation panels

        # Step 5: Filter rows based on panel assignments
        train_data = self.data[self.data['panel_id'].isin(train_panels)]
        test_data = self.data[self.data['panel_id'].isin(test_panels)]
        val_data = self.data[self.data['panel_id'].isin(val_panels)]

        # Check if the splits are non-empty
        if train_data.empty or test_data.empty or val_data.empty:
            raise ValueError("One of the data splits is empty. Check the panel ratios or dataset.")

        train_data = train_data.merge(self.non_numeric_data, on='panel_id', how='left')
        test_data = test_data.merge(self.non_numeric_data, on='panel_id', how='left')
        val_data = val_data.merge(self.non_numeric_data, on='panel_id', how='left')

        # Step 6: Separate x_data and y_data for each split
        train_x, train_y = train_data.drop(columns=['y', 'panel_id']).values, train_data['y'].values
        test_x, test_y = test_data.drop(columns=['y', 'panel_id']).values, test_data['y'].values
        val_x, val_y = val_data.drop(columns=['y', 'panel_id']).values, val_data['y'].values

        return (train_x, train_y), (test_x, test_y), (val_x, val_y)


def generate_panel_data(n_panels, rows_per_panel, n_features, random_state=42):
    """
    Generate synthetic data grouped by panels.

    :param n_panels: Number of unique panels.
    :param rows_per_panel: Number of rows per panel.
    :param n_features: Number of features in x_data.
    :param random_state: Seed for reproducibility.
    :return: x_data, y_data, panel_id
    """
    from sklearn.datasets import make_classification
    np.random.seed(random_state)
    x_data_list = []
    y_data_list = []
    panel_id_list = []

    for panel in range(1, n_panels + 1):
        # Generate synthetic classification data for each panel
        x_panel, y_panel = make_classification(
            n_samples=rows_per_panel,
            n_features=n_features,
            n_informative=n_features - 1,
            n_redundant=1,
            n_clusters_per_class=1,
            random_state=random_state + panel  # Vary seed per panel
        )

        # Add to lists
        x_data_list.append(x_panel)
        y_data_list.append(y_panel)
        panel_id_list += [panel] * rows_per_panel

    # Combine data from all panels
    x_data = np.vstack(x_data_list)
    y_data = np.hstack(y_data_list)
    panel_id = np.array(panel_id_list)

    return x_data, y_data, panel_id


# Example Usage
def main():
    # Example x_data, y_data, and panel_id
    x_data, y_data, panel_id = generate_panel_data(
        n_panels=1000,  # 10 unique panels
        rows_per_panel=10,  # Each panel has 10 rows
        n_features=5,  # 5 features in x_data
        random_state=42  # Set random state for reproducibility
    )

    # Initialize the splitter
    splitter = PanelSplitterWithTwinning(x_data, y_data, panel_id, train_ratio=0.6, test_ratio=0.3, val_ratio=0.1)

    # Perform the split
    (train_x, train_y), (test_x, test_y), (val_x, val_y) = splitter.split_panels_with_twinning()

    # Print results
    print("Train X:\n", train_x)
    print("Train Y:\n", train_y)
    print("Test X:\n", test_x)
    print("Test Y:\n", test_y)
    print("Validation X:\n", val_x)
    print("Validation Y:\n", val_y)






def read_in():
    df = pd.read_csv('./data/rural_int.csv')  # read in the data
    y_df = df[['crashes']].copy()  # only consider crashes
    y_df.rename(columns={"crashes": "Y"}, inplace=True)
    panels = df['orig_ID']
    try:
        x_df = df.drop(columns=['crashes', 'year', 'orig_ID',
                                'jurisdiction', 'town', 'maint_region', 'weather_station',
                                'dummy_winter_2'])  # was dropped postcode
        print('dropping for test')
        x_df = x_df.drop(columns=['month', 'inj.fat', 'PDO'])
        x_df = x_df.drop(columns=['zonal_ID', 'ln_AADT', 'ln_seg'])
        x_df['rumble_install_year'] = x_df['rumble_install_year'].astype('category').cat.codes
        x_df.rename(columns={"rumble_install_year": "has_rumble"}, inplace=True)
    except Exception as e:
        print(e)
        x_df = df.drop(columns=['Y'])  # was dropped postcode

    group_grab = x_df['county']
    x_df = x_df.drop(columns=['county'])
    x_df = helperprocess.interactions(x_df, drop_this_perc=0.8)
    x_df['county'] = group_grab
    splitter = PanelSplitterWithTwinning(x_df, y_df.values, panels)
    # Perform the split
    (train_x, train_y), (test_x, test_y), (val_x, val_y) = splitter.split_panels_with_twinning()

    # Print results
    print("Train X:\n", train_x)
    print("Train Y:\n", train_y)
    print("Test X:\n", test_x)
    print("Test Y:\n", test_y)
    print("Validation X:\n", val_x)
    print("Validation Y:\n", val_y)


if __name__ == "__main__":
    read_in()
    main()