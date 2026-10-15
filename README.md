# Airbnb Listing Price Prediction Across Cities

## Project Overview

This project reproduces selected experiments from the Stanford CS229
project **"Predicting Airbnb Listing Price Across Different Cities"**
(2019) and extends the model comparison by adding **Extra Trees
Regression**.

The goal is to estimate Airbnb listing prices using listing attributes
from three cities: **New York City (NYC), Paris, and Berlin**. We
compare multiple regression models, evaluate how numerical, categorical,
and text features affect performance, and investigate whether Extra
Trees offers an improvement over established tree-based models.

> **Note:** This is a methodology reproduction using the datasets and
> feature columns available in our notebook. It is not an exact
> numerical replication of the original paper because the supplied
> datasets and feature coverage differ.

## Problem Statement

Airbnb hosts need to set nightly prices that are competitive while
reflecting the characteristics of their listings. Price depends on
several factors, including room and property attributes, location, and
information contained in listing descriptions or titles. Manually
estimating a reasonable price can be difficult.

**Problem:** Build and evaluate machine-learning regression models that
predict Airbnb listing prices for NYC, Paris, and Berlin, then compare
the results with the reference study and assess an additional model,
Extra Trees Regression.

### Objectives

1.  Reproduce selected experiments from the reference paper using the
    available city datasets.
2.  Compare regression models using a consistent evaluation setup.
3.  Study the effect of adding categorical and text features to
    numerical features.
4.  Compare our results with the values reported in the reference paper,
    while documenting differences in datasets and features.
5.  Extend the model comparison with Extra Trees Regression.

## Dataset

The notebook uses three CSV files:

-   `AB_NYC_2019.csv`
-   `Paris 2019.csv`
-   `Berlin 2019.csv`

The datasets contain Airbnb listing information for the corresponding
cities. The available columns are not identical across all three files.
To keep the city experiments comparable, the notebook uses numerical
columns shared by all three datasets, along with selected categorical
and text features.

The target is the listing price. Prices are filtered using a cap of
**500** to reduce the influence of very high-priced listings.

## Features

The notebook compares two feature configurations:

-   **C (Continuous):** Seven shared numerical features available across
    all three city datasets.
-   **C+O+T:** The numerical features plus selected categorical/other
    features and listing-title text.

For the C+O+T configuration:

-   Categorical features include `room_type` and `area`.
-   The text feature is the listing name/title.
-   Title text is represented with TF-IDF using unigrams and bigrams,
    then reduced to 50 dimensions using Truncated SVD.

These feature sets are narrower than those in the original paper, which
used a broader collection of continuous, categorical, and text features.

## Methodology

The notebook follows this general workflow:

1.  Load and inspect the three city datasets.
2.  Clean and filter the data, including applying the price cap.
3.  Select comparable feature sets for each city.
4.  Split the data into training, validation, and test subsets using a
    **70/20/10** split.
5.  Build preprocessing pipelines:
    -   Impute missing numerical values using the training-set median.
    -   Standardize numerical features where appropriate.
    -   One-hot encode categorical features.
    -   Convert listing titles to TF-IDF features and reduce their
        dimensionality with Truncated SVD.
6.  Train and evaluate the regression models.
7.  Compare test-set performance using **R²** and **mean squared error
    (MSE)**.
8.  Add Extra Trees Regression as an extension and compare it with
    Random Forest and XGBoost.

Preprocessing is fitted on the training data and then applied to
validation/test data to help prevent data leakage.

## Models Evaluated

The notebook evaluates the following models:

-   Ridge Regression
-   K-Nearest Neighbors (KNN) Regression
-   Random Forest Regression
-   Neural Network Regression
-   XGBoost Regression
-   **Extra Trees Regression** (project extension)

The Extra Trees model uses 300 estimators, `max_features=1.0`,
`min_samples_leaf=1`, and a fixed random state.

## Outcomes

### Test R² with C+O+T features

  ----------------------------------------------------------------------------
  City          Ridge        KNN     Random     Neural      XGBoost      Extra
                                     Forest    Network                   Trees
  -------- ---------- ---------- ---------- ---------- ------------ ----------
  NYC          0.4642     0.3087     0.5407     0.5310   **0.5727**     0.5505

  Paris        0.1957     0.1957     0.3760     0.3780   **0.4292**     0.3986

  Berlin       0.1380     0.1380     0.3379     0.3310   **0.3766**     0.3319
  ----------------------------------------------------------------------------

R² measures how much of the variation in listing prices is explained by
a model on the evaluated data. It is **not** the same as percentage
accuracy.

### Main Findings

-   **XGBoost achieved the highest test R²** among the evaluated models
    for NYC, Paris, and Berlin with C+O+T features.
-   Adding selected categorical and title-text features improved
    performance compared with the numerical-only C feature set in our
    experiments.
-   Extra Trees slightly outperformed Random Forest in NYC and Paris,
    but was slightly behind Random Forest in Berlin.
-   Extra Trees did not outperform XGBoost in these experiments, but it
    provided a useful additional comparison under the same evaluation
    framework.
-   Results varied by city, indicating that a model's performance
    depends on the available data and the characteristics represented in
    each dataset.

### Comparison with the Reference Paper

The reference paper reports an XGBoost test R² of **0.706** for NYC with
C+O+T features. Our corresponding NYC result is **0.5727**. The
difference should be interpreted in light of the different input files
and feature coverage: our notebook uses seven shared numerical features,
two selected categorical features, and one title-text field, rather than
the paper's broader feature set.

Therefore, our project reproduces the general experimental approach and
extends the model comparison, but should not be presented as an exact
replication of the paper's numerical results.

## Repository Files

Expected project files include:

-   `ML-Final_submission.ipynb` --- notebook containing data
    preparation, preprocessing, model training, evaluation, and the
    Extra Trees extension.
-   `AB_NYC_2019.csv` --- NYC listing dataset.
-   `Paris 2019.csv` --- Paris listing dataset.
-   `Berlin 2019.csv` --- Berlin listing dataset.

Ensure the CSV files are available at the paths expected by the notebook
before running it.

## How to Run

1.  Clone or download the project repository.
2.  Place the three CSV files in the locations expected by the notebook.
3.  Open `ML-Final_submission.ipynb` in Jupyter Notebook, JupyterLab, or
    VS Code.
4.  Install the Python packages required by the notebook if they are not
    already installed. Common dependencies used in this workflow
    include:
    -   `pandas`
    -   `numpy`
    -   `scikit-learn`
    -   `xgboost`
    -   `matplotlib`
    -   `seaborn`
5.  Run the notebook cells in order.
6.  Review the generated evaluation metrics and model comparisons.

The exact dependency versions should be taken from the environment used
to run the notebook; this README does not claim a tested, pinned
environment.

## Limitations

-   The city datasets do not have identical schemas.
-   The shared numerical feature set is smaller than the one used in the
    reference paper.
-   Only one listing-title text field is used in our C+O+T experiment.
-   Model performance depends on the available features, data quality,
    and model configuration.
-   Results from different datasets or feature pipelines should not be
    treated as a perfectly controlled numerical comparison.
-   The results describe predictive performance on the chosen test split
    and do not guarantee that predicted prices are optimal market
    prices.

## Reference

Luo, Yuanhang, Xuanyu Zhou, and Yulian Zhou. **"Predicting Airbnb
Listing Price Across Different Cities."** Stanford CS229 Project,
December 14, 2019.

Reference PDF:
https://cs229.stanford.edu/proj2019aut/data/assignment_308832_raw/26647491.pdf

## Summary

This project studies Airbnb listing-price prediction across NYC, Paris,
and Berlin. It compares multiple regression algorithms, evaluates
numerical and enriched feature sets, and extends the reference model
comparison with Extra Trees Regression. In our experiments, **XGBoost
was the strongest model across all three cities**, while Extra Trees
offered competitive but not best overall performance.
