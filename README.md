# Telco Customer Churn Project

# Abstract and Introduction
### Background

Customer churn is a major challenge for subscription-based businesses such as telecommunications providers. Using the Telco Customer Churn dataset with approximately 7,000 customers and an observed churn rate of around 25%, roughly 1,700 customers are expected to leave the service. Assuming an average customer lifetime value of $800, this could correspond to nearly $1.4 million in potential revenue loss.

However, retention actions such as promotional incentives or customer support interventions incur additional costs and cannot be applied to every customer. Therefore, the key challenge is to accurately identify high-risk customers and allocate retention resources efficiently. This motivates the need for predictive churn modeling and data-driven retention strategies that maximize return on investment (ROI).

### North Star Metrics(Aims and Goals)
• **High-Risk Customer Identification** – Accurately identify customers with the highest probability of churn using predictive modeling.

• **Customer Prioritization** – Rank customers based on predicted churn risk and potential business value to ensure retention resources focus on the most impactful segments.

• **Retention Targeting Strategy Optimization** – Determine the optimal targeting rate and intervention strategy to maximize the effectiveness of retention efforts under budget constraints.

### Main Outcomes

The results demonstrate that predictive churn modeling can effectively identify high-risk customers who are most likely to discontinue the service. Using logistic regression to estimate individual churn probabilities, customers can be ranked and prioritized based on their likelihood of churn and potential business impact. This enables the company to move beyond uniform retention campaigns and instead focus on targeted interventions for the most at-risk segments.

Building on these predictions, retention targeting simulations show that a mixed intervention strategy can substantially improve profitability when applied selectively. The analysis finds that the optimal targeting rate is approximately **65% of customers**, yielding a **maximum expected profit of about $300K** after accounting for retention costs. These results illustrate how combining churn prediction with targeted retention strategies allows companies to allocate limited retention resources more efficiently and maximize the return on investment of retention campaigns.

# Data Introduction

### Dataset Overview

The analysis uses the **Telco Customer Churn dataset**, which contains information on **7,043 customers** and **21 variables** describing customer demographics, service subscriptions, contract types, and billing characteristics.

The target variable is **Churn**, which indicates whether a customer discontinued the service. Among the 7,043 customers, **1,869 customers churned**, corresponding to a churn rate of approximately **26.5%**.

### Feature Characteristics

The dataset includes a mix of demographic, service usage, contract, and billing variables:

| Feature Category | Example Variables |
|---|---|
| Demographic | Gender, SeniorCitizen, Partner, Dependents |
| Service Usage | PhoneService, InternetService, StreamingTV, TechSupport |
| Contract Information | Contract type, tenure |
| Billing Information | MonthlyCharges, PaymentMethod, PaperlessBilling |
| Target Variable | Churn |

Among the variables, **tenure** and **MonthlyCharges** provide important numerical information about customer subscription duration and billing amount.

# EDA Framework 



<div style="border:1px solid #e0e0e0; padding:18px; border-radius:10px;  margin:20px 0;">
<h3>Overall Churn Rate</h3>
<hr>
<p>Churn rate = <b>26.5%</b></p>  
1,869 churners out of 7,043

<h4>Service Feature Impact on Churn</h4>

<p>
Beyond overall churn rate (26.5%), we examine service feature adoption 
to understand <b>why customers churn</b>.
</p>

<p><b>Key Findings:</b></p>
<ul>
  <li><b>Fiber optic users</b> exhibit a churn rate of <b>41.9%</b>.</li>
  <li>Customers without <b>Online Security</b> show a churn rate of <b>41.8%</b>.</li>
  <li>Customers without <b>Tech Support</b> show a churn rate of <b>41.6%</b>.</li>
  <li>Customers without <b>Online Backup</b> show a churn rate of <b>39.9%</b>.</li>
  <li>Customers without <b>Device Protection</b> show a churn rate of <b>39.1%</b>.</li>
</ul>


<p><b>Business Implication:</b><br>
Retention strategy should consider bundling or promoting support/security services 
to high-risk customers. Feature adoption may serve as both a predictive signal and 
an actionable intervention lever.
</p>

</div>
<div style="border:1px solid #e0e0e0; padding:16px; border-radius:10px;  margin:16px 0;">

<h3>Contract-Level Churn Comparison</h3>
<hr>

![image](https://hackmd.io/_uploads/SyraRULc-g.png)


<p><b>Key Insight:</b><br>
Month-to-month customers exhibit a churn rate nearly <b>15 times higher</b> than two-year contract customers. 
This indicates that <b>contract stability is a strong predictor of churn risk</b>.
</p>

<!-- <p><b>Business Implication:</b><br>
Short-term contract customers should be prioritized in retention campaigns. 
Incentivizing longer-term commitments may significantly reduce churn and stabilize recurring revenue.
</p> -->

</div>

<div style="border:1px solid #e0e0e0; padding:16px; border-radius:10px;  margin:16px 0;">

<h3>Tenure Bin Analysis</h3>
<hr>

/Users/terry/Documents/Telco-Customer-Churn-/image/image1.png

s
<p><b>Key Insight:</b><br>
Churn probability decreases significantly as customer tenure increases. 
Customers within the <b>first 3 months</b> exhibit the highest churn rate, 
indicating that <b>early-stage customers are the most vulnerable segment</b>.
</p>
<!-- <p><b>Business Implication:</b><br>
Retention efforts should prioritize onboarding-stage customers, 
where intervention can yield the highest marginal impact on churn reduction.
</p> -->
</div>


<div style="border:1px solid #e0e0e0; padding:16px; border-radius:10px;  margin:16px 0;">

<h3>Payment Method Analysis</h3>
<hr>

![image](https://hackmd.io/_uploads/r1UZkDIcWx.png)




<p><b>Key Insight:</b><br>
Customers using <b>electronic check</b> show significantly higher churn rates compared to those enrolled in automatic payment methods.  
This may indicate <b>lower commitment</b> or <b>higher payment friction</b>.
</p>
</div>

<div style="border:1px solid #e0e0e0; padding:16px; border-radius:10px;  margin:16px 0;">

<h3> Revenue Tier Segmentation</h3>
<hr>

![image](https://hackmd.io/_uploads/ryAylD8cWx.png)



Customers were segmented into <b>Low</b>, <b>Mid</b>, and <b>High</b> revenue tiers using quantile-based grouping of <b>MonthlyCharges</b>.
</p>

High-revenue customers exhibit the highest churn rate (<b>34.1%</b>), 
more than double the churn rate of low-revenue customers (<b>15.9%</b>).
</p>

<!-- <p><b>Business Implication:</b><br>
Revenue exposure is concentrated in the high-value segment. 
Targeted retention efforts toward high-revenue customers can generate 
disproportionately higher financial impact and improve overall retention ROI.
</p> -->

</div>


<div style="border:1px solid #e0e0e0; padding:18px; border-radius:10px;  margin:20px 0;">
<h3>CLV and Churn Relationship</h3>
<hr>
    
<p><b>Definition:</b><br>
Customer Lifetime Value (CLV) is approximated as:
</p>

<p style="font-size:16px; text-align:center;">
<b>CLV = MonthlyCharges × Tenure</b>
</p>

<p>
This proxy captures cumulative revenue contribution and long-term customer engagement.
</p>

![image](https://hackmd.io/_uploads/BkeiyD85Zx.png)


<p><b>Key Findings:</b></p>
<ul>
  <li>Mean CLV (Churn): <b>$1,532</b></li>
  <li>Mean CLV (Non-churn): <b>$2,550</b></li>
  <li>High CLV customers show the <b>lowest churn rate (16.6%)</b></li>
  <li>Low CLV customers show the <b>highest churn rate (39.3%)</b></li>
</ul>
Churn is primarily concentrated among lower accumulated-value customers, 
while high-CLV customers demonstrate stronger retention stability. 
Retention strategy should therefore prioritize value-weighted risk rather than churn rate alone.
</p>

</div>



<div style="border:1px solid #e0e0e0; padding:18px; border-radius:10px; margin:20px 0;">

<h3>Churn Risk Heatmap Analysis</h3>


<hr>


<h4>Contract Type vs Payment Method</h4>

<p>
This heatmap examines churn patterns across <b>contract commitment</b> and 
<b>payment behavior</b>. These two variables often reflect customer engagement 
and switching costs.
</p>

![image](https://hackmd.io/_uploads/SJ1b4Brcbe.png)

<p><b>Key Findings:</b></p>

<ul>
<li>Customers on <b>month-to-month contracts using electronic check</b> show the highest churn rate (<b>53.7%</b>).</li>
<li>Customers with <b>two-year contracts</b> exhibit the lowest churn rates (<b>below 8%</b> across payment methods).</li>
<li>Automatic payment methods generally correspond to <b>lower churn risk</b> than manual payment methods.</li>
</ul>

<p>
These results suggest that <b>low commitment combined with manual payment behavior</b> 
significantly increases churn risk. Encouraging customers to adopt <b>longer contracts 
or automatic payment options</b> could therefore improve retention.
</p>


<hr>


<h4>Internet Service Type vs Contract</h4>

<p>
This heatmap explores churn differences across <b>internet service types</b> and 
<b>contract structures</b>. Service tiers may affect customer expectations 
and perceived value.
</p>


![image](https://hackmd.io/_uploads/rJl3MNHSqZg.png)

<p><b>Key Findings:</b></p>

<ul>
<li><b>Fiber optic customers</b> consistently show the highest churn rates.</li>
<li>Fiber users with <b>month-to-month contracts</b> experience extremely high churn (<b>54.6%</b>).</li>
<li>Customers without internet service have the <b>lowest churn rates</b>.</li>
</ul>

<p>
The higher churn among fiber optic customers may indicate 
<b>price sensitivity or dissatisfaction with service value</b>. 
Targeted retention incentives or improved service offerings may 
help stabilize this segment.
</p>


<hr>


<h4>Tenure vs Monthly Charges</h4>

<p>
Customer tenure and monthly pricing jointly influence churn risk. 
The following heatmap visualizes churn rates across <b>tenure segments</b> 
and <b>monthly charge levels</b>.
</p>


![image](https://hackmd.io/_uploads/HkmKESH9Wg.png)

<p><b>Key Findings:</b></p>

<ul>
<li>Customers with <b>short tenure (0–12 months)</b> and <b>high monthly charges ($100+)</b> show the highest churn rate (<b>76%</b>).</li>
<li>Churn probability decreases significantly as <b>customer tenure increases</b>.</li>
<li>Long-term customers maintain <b>consistently lower churn rates</b> across pricing tiers.</li>
</ul>

<p>
These findings indicate that <b>early-stage customers exposed to higher pricing</b> 
are the most vulnerable to churn. Retention strategies should therefore prioritize 
<b>early customer engagement and pricing incentives</b>.
</p>
</div>


# Methods 

### Method 1: Churn Prediction Model

To estimate the probability that a customer will churn, we use a **logistic regression model**

The logistic regression model estimates the probability of churn as:
$$
P(\text{Churn}=1 \mid X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_k x_k)}}
$$
Where:
- $P(\text{Churn}=1 \mid X)$ = probability that a customer churns
- $x_1, x_2, \dots, x_k$ = customer features (e.g., tenure, contract type, internet service, monthly charges)
- $\beta_0$ = intercept
- $\beta_1, \beta_2, \dots, \beta_k$ = model coefficients
The logistic regression model allows us to estimate **individual churn probabilities for each customer**, which serves as the first step in identifying customers at risk of leaving.

---


### Method 1.a: Marginal Effects 

To interpret how each feature affects churn probability, we compute marginal effects.

The marginal effect of a feature $X_j$ is defined as:

$$
\frac{\partial P(Y = 1 \mid X)}{\partial X_j}
= \beta_j \cdot P(Y = 1 \mid X) \cdot (1 - P(Y = 1 \mid X))
$$

Marginal effects measure the change in predicted churn probability associated with a one-unit increase in a given feature, holding all other variables constant.

This allows us to quantify the relative importance of different factors, such as contract type, payment method, and value-added services, in influencing customer churn.

---

### Method 1.b: What-if Simulation (Scenario Analysis)

To evaluate the impact of potential retention strategies, we perform a what-if simulation based on the trained logistic regression model.

Given a customer profile $X$, we first compute the baseline churn probability:

$$
P(Y = 1 \mid X)
$$

We then modify selected features to simulate alternative scenarios (e.g., switching contract type or adding services), resulting in a new feature vector $X'$.

The change in churn probability is calculated as:

$$
\Delta P = P(Y = 1 \mid X') - P(Y = 1 \mid X)
$$

This approach enables us to quantify how specific interventions affect churn risk and provides actionable insights for designing effective retention strategies.

---


### Method 2: Expected Churn Loss Estimation

To quantify the financial impact of churn, we estimate the **expected revenue loss** for each customer:

$$
\text{Expected Loss}_i = P(\text{Churn}_i) \times CLV_i
$$

Where:

- $P(\text{Churn}_i)$ = predicted churn probability for customer $i$
- $CLV_i$ = estimated customer lifetime value for customer $i$

In this project, customer lifetime value is approximated as:

$$
CLV_i = \text{MonthlyCharges}_i \times 12
$$


---

### Method 3: Retention Profit Optimization
The expected profit of a retention strategy is defined as:

$$
\text{Expected Profit} = \big(P(\text{Churn}) \times CLV \times \text{SuccessRate}\big) - \text{RetentionCost}
$$

Where:

- $P(\text{Churn})$ = predicted probability that the customer will churn
- $CLV$ = estimated customer lifetime value
- $\text{SuccessRate}$ = probability that the retention intervention successfully prevents churn
- $\text{RetentionCost}$ = cost of the retention incentive

In this project, three retention strategies are considered, each representing a different level of retention intervention intensity.

- **Strategy A – Low-Cost Outreach**
  - Cost = \$5 per customer  
  - Success rate = 10%  
  - Represents lightweight retention actions such as reminder emails, personalized notifications, or small promotional incentives. 
- **Strategy B – Targeted Retention Offer**
  - Cost = \$30 per customer  
  - Success rate = 25%  
  - Represents moderate retention interventions such as contract adjustments, temporary service upgrades.
- **Strategy C – Aggressive Retention Incentive**
  - Cost = \$50 per customer  
  - Success rate = 30%  
  - Represents stronger retention actions such as large promotional credits, loyalty rewards.

# Results

### 1.High-Risk Customer Identification
We first examine the **logistic regression** coefficients to understand which features most strongly influence churn probability.
![image](https://hackmd.io/_uploads/Hkx2K42Kbl.png)
#### Factors associated with higher churn risk:
* Fiber optic internet service customers
* Customers using electronic check payment methods
* Customers with month-to-month contracts
* Customers without value-added services, such as online security, tech support, and online backup, are more likely to churn.



Next, we analyze where high-risk customers are concentrated by comparing the distribution of key service features across risk segments.

![image](https://hackmd.io/_uploads/ByFxKH2t-e.png)
#### High-Risk Customer Profile :
* Customers with month-to-month contracts exhibit significantly higher churn risk **(36.18%)** compared with long-term contract.
* Customers paying via electronic check exhibit the highest churn exposure, with **44.10%** of customers classified as high risk.
* Customers using fiber optic internet service also show elevated churn risk, with **39.79%** identified as high-risk customers.
* Customers without online security services have a high churn risk ratio of **38.58%**
* Similarly, customers without technical support show a high-risk ratio of **38.16%**

### 2. Drivers of Churn
![image](https://hackmd.io/_uploads/SJysqUI9Ze.png)
The analysis reveals that churn is primarily driven by contract flexibility, payment behavior, and service engagement. Customers with fiber optic internet, electronic check payment methods, and paperless billing exhibit significantly higher churn risk, while longer tenure and value-added services reduce churn probability.
### 3.Customer Prioritization
![image](https://hackmd.io/_uploads/B1DeqFTKZg.png)

To support retention decision-making, customers are visualized based on predicted churn risk and customer lifetime value (CLV). 

Using these two dimensions, customers are segmented into four groups based on churn risk (high vs. low) and customer avaverage value. For each segment, we estimate the **total expected revenue loss**, calculated as the sum of individual expected losses across customers within the segment.
![image](https://hackmd.io/_uploads/HJc-JT0YWl.png)

1. **High Risk – High Value (Upper Right) :** This segment contributes the largest expected revenue loss **($848K)**, indicating that retaining these customers would have the greatest financial impact.
2. **High Risk – Low Value (Lower Right) :** Despite higher churn risk, this segment contributes only **$49K** in expected loss, suggesting that expensive incentives may not be cost-effective.
3. **Low Risk – High Value (Upper Left) :** These customers generate substantial revenue and account for **$579K** in expected loss, meaning service quality and relationship management remain important.
4. **Low Risk – Low Value (Lower Left) :** This segment contributes **$192K** in expected loss, indicating relatively limited financial risk.

### 4.Retention Targeting Strategy Optimization![image](https://hackmd.io/_uploads/H1HsChRtWl.png)
To simulate a realistic retention campaign, we assume the company can deploy three different retention strategies with varying cost and effectiveness:
#### Strategy A – Low-Cost Outreach
#### Strategy B – Targeted Retention Offer
#### Strategy C – Aggressive Retention Incentive

#### Peak Profit by Strategy :
* **Best Mixed Strategy** reached its highest expected profit at **65%** target rate, generating approximately **$301.5K** in cumulative profit.
* **Strategy A** Only reached its peak at **65%** target rate, with approximately **$139.6K** in cumulative profit.
* **Strategy B** Only achieved its maximum profit at **53%** target rate, generating approximately **$277.3K**.
* **Strategy C** Only peaked earlier at **46%** target rate, with approximately **$284.1K** in cumulative profit.



### 5.Scenario Analysis for Retention Strategies
![image](https://hackmd.io/_uploads/ByS-iUI5Wx.png)
Scenario-based simulation shows that structural interventions, particularly transitioning customers to longer-term contracts, have the largest impact on churn reduction. A combined retention strategy reduces predicted churn probability by approximately 75%.
# Discussion and Recommendation

The analysis provides a comprehensive understanding of customer churn by integrating predictive modeling, customer value segmentation, and scenario-based simulations. The results highlight not only the key drivers of churn but also the effectiveness of targeted retention strategies in reducing potential revenue loss.

---

### Business Implications

* Customer churn is driven by a combination of behavioral and structural factors:

The results from the logistic regression model show that contract flexibility, payment behavior, and service engagement are the primary drivers of churn. Customers on month-to-month contracts, using electronic check payments, and lacking value-added services exhibit significantly higher churn risk, while longer tenure and service bundling reduce churn probability.

* High-risk customers are not necessarily high-value customers:

Customer segmentation analysis reveals that although some customers have high churn probability, their contribution to total revenue is relatively small. This suggests that applying costly retention strategies uniformly across all high-risk customers may not be economically efficient.

* Retention strategies should be targeted rather than uniform:

By combining churn probability with customer lifetime value (CLV), firms can identify high-risk, high-value customers who represent the largest potential revenue loss. Targeting this segment yields a more efficient allocation of retention resources compared to blanket campaigns.

* Structural interventions have a greater impact than marginal changes:

Scenario analysis indicates that long-term contract adoption and service bundling produce significantly larger reductions in churn probability compared to isolated actions such as changing payment methods. This suggests that retention strategies should focus on structural customer relationships rather than minor behavioral adjustments.

---

### Recommended Retention Strategy

Based on the analysis, the following strategies are recommended to improve customer retention and maximize long-term profitability:

* Prioritize high-risk, high-value customers:

Retention efforts should focus on customers who are both likely to churn and contribute substantial revenue. This segment represents the greatest financial risk and offers the highest return on retention investment.

* Encourage longer contract commitments:

The scenario analysis shows that transitioning customers from month-to-month plans to one- or two-year contracts leads to the largest reduction in churn probability. Incentives such as discounted pricing or bundled offerings can be used to encourage contract upgrades.

* Expand value-added service bundles:

Adding services such as online security, backup, and technical support significantly reduces churn risk. Bundled service packages can enhance perceived value and increase customer stickiness.
