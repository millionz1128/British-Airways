# British Airways Booking Completion Prediction

## Project Overview
This project was completed as part of the Forage Data Science Job Simulation for British Airways. The objective is to predict whether a customer will complete a flight booking.

Using customer-level flight search data, the project builds an **XGBoost classifier** to predict booking completion probability and identify the primary drivers influencing booking decisions.

## Problem Statement
Online flight searches do not always result in completed bookings. Given the competitive nature of the airline industry and relatively low conversion rates, understanding booking intent is critical.

This project aims to:
* Identify behavioral and geographic factors associated with booking completion
* Build a predictive model to classify high-intent customers
* Provide actionable insights to improve marketing allocation and conversion strategy

The target variable is: `booking_complete` (binary)

1 = booking completed<br>
0 = booking not completed

## Dataset
The dataset is from Forage's British Airways Data Science Job Simulation. It contains flight search and booking records from British Airways customers. Each row represents a single flight search attempt. Not all searches result in completed bookings.

| Variable              | Description                         |
| --------------------- | ----------------------------------- |
| num_passengers        | number of passengers travelling     |
| sales_channel         | sales channel (Internet / Mobile)   |
| trip_type             | RoundTrip, OneWay, CircleTrip       |
| purchase_lead         | days between booking and departure  |
| length_of_stay        | number of days spent at destination |
| flight_hour           | hour of flight departure            |
| flight_day            | day of week of flight departure     |
| route                 | origin -> destination flight route  |
| booking_origin        | country from where booking was made |
| wants_extra_baggage   | add-on selection                    |
| wants_preferred_seat  | add-on selection                    |
| wants_in_flight_meals | add-on selection                    |
| flight_duration       | total duration of flight (in hours) |
| booking_complete      | target variable                     |

## Methodology
The project follows a structured analytical pipeline:

**1. Data Preparation**
* Data validation and structural inspection

**2. Exploratory Data Analysis (EDA)**
* Feature distribution analysis
* Target vs feature booking rate analysis
* Binning analysis for non-linear numeric relationships
* Correlation analysis among numeric variables

**3. Feature Engineering**
* Label encoding for low-cardinality categorical variables
* K-Fold Target Encoding for high-cardinality features (`route`, `booking_origin`)
* Creation of `addon_sum` feature (aggregated add-on engagement)
* Hour binning for departure time segmentation

**4. XGBoost Model**
* XGBoost was selected because it handles non-linear relationships, performs well with heterogeneous features, and is effective for imbalanced datasets.
* Hyperparameter tuning performed using GridSearchCV with F1-score as the refit metric.
* Model evaluated using accuracy, precision, recall and F1-score.

## Results
Metric | Score
:--- | :---
**Accuracy** | 0.72
**Precision** | 0.31
**Recall** | 0.74
**F1**| 0.44

Top predictive drivers:
1. `booking_origin`
2. `route`
3. `sales_channel`

* This model successfully identifies high-intent customers with strong recall performance.
* Booking origin and route play a dominant role in predicting booking completion, followed by behavioral indicators such as sales channel and add-on purchases.
* Although precision remains moderate, the model is effective for identifying potential high-conversion segments.

## Business Recommendations
1. Allocate marketing budget toward countries/regions with high booking rates (e.g. Malaysia, Macau, Vietnam, Singapore, Indonesia).
2. Focus promotional efforts on high-conversion routes (e.g. PENTPE, DMKKIX, AKLKUL, MELPEN, ICNSIN).
3. Customers booking closer to departure are more likely to convert. Provide time-sensitive offers to short-lead travelers.
5. Internet channel significantly outperforms mobile in conversion rate. Investigate UI/UX differences.

## Tech Stack
* Python (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, XGBoost)
* Jupyter Notebook

## File Descriptions
* `British Airways.ipynb`: Jupyter notebook containing all the codes of the project.
* `customer_booking.csv`: CSV file containing the dataset
* `README.md`: This file, providing an overview of the project
