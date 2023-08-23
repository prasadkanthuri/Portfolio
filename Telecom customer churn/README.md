# Customer Churn Prediction

In a landscape plagued by customer churn issues, particularly evident in the telecommunications industry with an average churn rate of 15-25%, businesses grapple with losing customers due to factors like inadequate customer service, high prices, and competitive shortfalls. However, by comprehending these churn contributors, businesses can rectify the situation.

The objective of this project is to develop a customer churn prediction model pinpointing high-risk customers for targeted interventions, such as discounts, to mitigate churn. Additionally, understanding the driving factors behind churn, like contract length and payment methods, will steer strategies for customer retention.

## About the data
The project's dataset encompasses details about departing customers, services subscribed to, account info, and demographics. Notably, the dataset is largely categorical, presenting insights like the prevalence of loyal, long-term customers and their varied spending habits.
## Insights Capturing
Examining tenure, monthly charges, and total charges distributions, it's clear new customers face retention challenges. Monthly charge distributions highlight a preference for low or high payments, and total charges distribution indicates a trend towards lower payments.

Analyzing churned customers reveals their shorter tenure, higher monthly charges, and lower total charges. Intriguingly, there's a cohort of high-paying customers who churned, emphasizing the need for tailored retention efforts. Moreover, correlations emerge, such as contract length's inverse relationship to churn and the influence of payment methods, tech support, and internet services.
## EDA at a glance
![Profile_summary](https://github.com/prasadkanthuri/Portfolio/assets/135444495/24d98e9f-2a41-443f-8e09-34849f0c0202)

## Feature Engineering and Model Building
After comprehensive feature engineering, a variety of classifier models underwent testing using the dataset. Among these, the XGBoost classifier emerged as the standout performer due to its robust performance. Subsequently, the model was trained employing the XGBoost classifier. With the model-building phase completed, predictions were executed on the entire dataset.

The projected outcomes, comprising these predictions, were harnessed as the data source for Power BI. This integration paved the way for a holistic evaluation of risks and customer profiles within the Power BI platform. By leveraging the power of data visualization and analysis, this process facilitated the synthesis of meaningful insights and actionable information to bolster decision-making and strategic planning.

In summary, the project encompassed rigorous classifier model testing, ultimately culminating in the selection of XGBoost as the optimal choice. The model was then trained and predictions applied to the entire dataset, further enhancing insights through Power BI's visualization capabilities, driving comprehensive risk assessment, and yielding valuable customer profiles.
![Risk_evaluation](https://github.com/prasadkanthuri/Portfolio/assets/135444495/299ee81e-a312-4241-b942-64f43eea82b8)

![Customer_details](https://github.com/prasadkanthuri/Portfolio/assets/135444495/efc64a16-3288-4ab5-a4e4-d9ed17c20519)

