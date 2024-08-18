import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings


class PSIAnalyzer:
    """
    PSIAnalyzer analyzes two list-like data in terms of PSI calculation.

    PSI (Population Stability Index) shows how much the expected data shifted according to actual data.

    PSI splits actual data in n groups then calculates percentages, lower and upper limits of groups. Then calculates
    percentages of expected data for each group. After percentages found, the formula below is calculated for every group:

    PSI = ∑[(actual_percentage_i - expected_percentage_i) * ln(expected_percentage_i / actual_percentage_i)]
    i € n

    PSI value can be interpreted as below:
    0 <= PSI < 0.1 -> No significant shifts between actual and expected
    0.1 <= PSI < 0.25 -> Moderate shifts between actual and expected
    0.25 <= PSI -> Major shifts between actual and expected

    Parameters:

        actual: list-like: pd.core.series.Series, np.ndarray, list
            Actual values that will be used in splitting data and determining lower and upper limits for groups.
        expected: list-like: pd.core.series.Series, np.ndarray, list
            Expected values that will be used to compare with actual data.
        group: Default=10, integer, must be >= 2
            Number of groups for splitting datas.

    Methods:

        group_creator():

            When called determines groups for actual and expected data.

            Parameters:
                None

            Returns:
                None

        _psi_calculator():

            When called calculates PSI for actual and expected data.

            Parameters:
                None

            Returns:
                psi: PSI value for actual and expected.
                group_psi_values: PSI value for each group between actual and expected. Sum of values in group gives `psi`.


    """

    def __init__(self, actual, expected, group=10):

        # initial controlling
        if type(actual) not in [pd.core.series.Series, np.ndarray, list]:
            raise ValueError(
                "actual must be one of these: pd.core.series.Series, np.ndarray, list"
            )
        if type(expected) not in [pd.core.series.Series, np.ndarray, list]:
            raise ValueError(
                "expected must be one of these: pd.core.series.Series, np.ndarray, list"
            )
        if type(group) != int:
            raise ValueError("group must be integer")
        if group < 2:
            raise ValueError("group must be bigger or equal to 2")
        elif group > 25:
            warnings.warn(
                "For better PSI visualization group should be lower than or equal to 25"
            )

        # setting initials
        self.actual = actual
        self.expected = expected
        self.group = group

        # type fixing
        if type(self.actual) == pd.core.series.Series:
            self.actual = self.actual.values
        elif type(self.actual) == list:
            self.actual = np.array(self.actual)

        if type(self.expected) == pd.core.series.Series:
            self.expected = self.expected.values
        elif type(self.expected) == list:
            self.expected = np.array(self.expected)

        # creating self variables
        self.cut_points = None
        self.actual_groups = []
        self.expected_groups = []
        self.actual_percentages = []
        self.expected_percentages = []
        self.group_psi_values = []
        self.psi = None

    # this function calculates cut_points and sets a group value for each instance in actual and expected
    def group_creator(self):

        sorted_actual = np.sort(self.actual)
        self.cut_points = np.percentile(
            sorted_actual,
            [(100 / self.group) * counter for counter in range(self.group + 1)],
        )
        self.actual_groups = (
            np.digitize(self.actual, bins=self.cut_points, right=True) + 1
        )
        self.expected_groups = (
            np.digitize(self.expected, bins=self.cut_points, right=True) + 1
        )

    # tihs function calculates PSI
    def _psi_calculator(self):

        if self.cut_points is None:

            self.group_creator()

        len_actual = len(self.actual)
        len_expected = len(self.expected)

        for cut_index in range(self.group):

            cut_lower = self.cut_points[cut_index]
            cut_upper = self.cut_points[cut_index + 1]

            actual_perc = (
                np.sum((self.actual >= cut_lower) & (self.actual < cut_upper))
                / len_actual
            )
            expected_perc = (
                np.sum((self.expected >= cut_lower) & (self.expected < cut_upper))
                / len_expected
            )

            if cut_index + 1 == self.group:
                expected_perc = (
                    np.sum((self.expected >= cut_lower) & (self.expected <= cut_upper))
                    / len_expected
                )

            if expected_perc == 0:
                expected_perc = 0.000001

            self.actual_percentages.append(actual_perc)
            self.expected_percentages.append(expected_perc)

            group_psi = (actual_perc - expected_perc) * np.log(
                actual_perc / expected_perc
            )

            self.group_psi_values.append(group_psi)

        self.psi = np.sum(self.group_psi_values)

        return self.psi, self.group_psi_values

    # this function visualize PSI for n group
    def _visualize(self):

        if self.cut_points is None:

            self.group_creator()

        if self.group_psi_values == []:

            _, _ = self._psi_calculator()

        actual_df = pd.DataFrame(
            data={
                "groups": list(range(1, self.group + 1)),
                "percentages": self.actual_percentages,
            }
        )
        actual_df[""] = "actual"
        expected_df = pd.DataFrame(
            data={
                "groups": list(range(1, self.group + 1)),
                "percentages": self.expected_percentages,
            }
        )
        expected_df[""] = "expected"
        self.group_df = pd.concat([actual_df, expected_df], axis=0)

        sns.set_theme(style="whitegrid")
        sns.barplot(data=self.group_df, x="groups", y="percentages", hue="")
        plt.show()
