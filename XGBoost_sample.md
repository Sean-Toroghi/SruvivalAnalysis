<h1>Survival analysis - summary</h1>

References
- [Preventing churn like a bandit - with uplift modeling, causal inference, and Thompson sampling](https://bigdatarepublic.nl/articles/preventing-churn-bandit/)
- [Book: Survival analysis - 2012](https://link.springer.com/book/10.1007/978-1-4419-6646-9)


# Overview
Survival analysis is similar to a regular predictive model, witht the following important disitctions:
- the label is always positive
- the label may not be fully known, or a part of it may not be known during the observation period.

One example of susrvival analysis is predicting the time a cancer patient has to death. During the observation, we could have an open ended period, in which the patient has been released from the clinic before he/she died and we do not have access to information about the actual date of death. In this case the lower bound will be the duration from treatment until the patient was released. 

## Survival analysis with XGBoost [link](https://xgboost.readthedocs.io/en/stable/tutorials/aft_survival_analysis.html)
To model such a scenario, predictive models have developed survival analysis option. For example, XGBoost has accelerated failuire time model (survival analysis). The steps for developing survival analysis are as following:
1. express the label in range, with two bounds (y_lower_bound and y_upper_bound)
2. set:  `"objective" : "survival:aft"` and `"eval_metric" : "aft-nloglik"`
3. set `"aft_loss_distribution "` to one of the following options: `"normal"`, `"logistic",` or `"extreme"`
4. set `"aft_loss_distribution_scale"`

```python
import numpy as np
import xgboost as xgb

# 4-by-2 Data matrix
X = np.array([[1, -1], [-1, 1], [0, 1], [1, 0]])
dtrain = xgb.DMatrix(X)

# Associate ranged labels with the data matrix.
# This example shows each kind of censored labels.
#                         uncensored    right     left  interval
y_lower_bound = np.array([      2.0,     3.0,     0.0,     4.0])
y_upper_bound = np.array([      2.0, +np.inf,     4.0,     5.0])
dtrain.set_float_info('label_lower_bound', y_lower_bound)
dtrain.set_float_info('label_upper_bound', y_upper_bound)

params = {'objective': 'survival:aft',
          'eval_metric': 'aft-nloglik',
          'aft_loss_distribution': 'normal',
          'aft_loss_distribution_scale': 1.20,
          'tree_method': 'hist', 'learning_rate': 0.05, 'max_depth': 2}
bst = xgb.train(params, dtrain, num_boost_round=5,
                evals=[(dtrain, 'train')])
```
