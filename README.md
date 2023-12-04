# UFC Dataset Analysis

The purpose of this data analysis project is to explore and uncover meaningful insights within the UFC dataset which contains fighters' profiles, match outcomes, performance statistics and historical trends. Through statistical means, we will identify patterns and key factors that contribute to success in the UFC

## Authors

**Note:** We initially commited all our work to a repository outside SFU's git enterprise repository. This was a mistake we caught near the end of the project. We were able to migrate the project to SFU's git servers but could not rewrite our git history with our SFU computing IDs. We apologize for any inconvience and have put our personal github usernames alongside our SFU computing IDs below (this mitigation measure was approved by the professor):

- Jason Gill: jga132 == [@xgill15x](https://github.com/xgill15x)
- Yousef Haiba: yoh3 == [@YousefHaiba18](https://github.com/YousefHaiba18)
- Vishaal Bharadwaj: vbh == [@Svishyy](https://github.com/Svishyy)

## Required Libraries

**Note:** Python3 was the version used for this project (specifically >=3.9)

Install pandas and numpy for data manipulation

```bash
pip3 install pandas numpy
```

Install scipy, scikit-learn, pingouin and statsmodels for data analysis and machine learning tools

```bash
pip3 install scipy pingouin scikit-learn statsmodels
```

Install matplotlib and seaborn for data visualization

```bash
pip3 install matplotlib seaborn
```

## Order of Execution

**Note:** Running the preprocessing scripts (steps 1 and 2) are crucial for the rest of the programs to run

1. `python3 preprocessing_data.py`
2. `python3 preprocessing_fighter_details.py`
3. After preprocessing, the following files may be run in any order as they answer different analysis questions.

```bash
python3 colour_win_analysis.py
python3 effective_strikes_analysis.py
python3 stance_analysis.py
python3 p4p_analysis.py
python3 best_grappler_analysis.py
python3 best_striker_analysis.py
python3 takedown_knockdown_analysis.py
python3 strikeAcc_by_weight_analysis.py
python3 finishes_by_weight_analysis.py
python3 fight_result_predictor.py
python3 win_ratio_predictor.py
```

**Note:** Sample files were not included in our project because our datasets are not extremely large and our programs run fast. As a result, the above python analysis/predictor files do not require input/output arguments, as they inherently operate on the complete datasets.

## Files produced / Expected outputs

- The 2 preprocessing python files `preprocessing_data.py` and `preprocessing_fighter_details.py` will each produce preprocessed versions of the datasets found in the `data_sets` directory and will subsequently also be added to `data_sets`

- Most of the analyis programs produce plots that are explained in the report. Plots for a particular analysis question can be found in the `plots` directory after the corresponding program for it has been run.

- Some analysis questions will produce outputs in the terminal regarding p-values/posthoc used for analysis
