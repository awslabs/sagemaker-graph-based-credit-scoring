## Solution

Credit ratings are traditionally generated using models that use financial statement data and market data, which is tabular only (numeric and categorical). This solution constructs a network of firms using [SEC filings](https://www.sec.gov/edgar/searchedgar/companysearch.html) and shows how to use the network of firm relationships with tabular data to generate accurate rating predictions. This solution demonstrates a methodology to use data on firm linkages to extend the traditionally tabular-based credit scoring models, which have been used by the ratings industry for decades, to the class of machine learning models on networks.


## What are the input datasets?

This solution uses synthetic datasets that are created to mimic typical examples of corporate credit rating datasets.

1. **Tabular** dataset: Contains various synthetically generated financials and accounting ratios (numerical) and industry codes (categorical). The dataset has **3286** rows. The label column `Rating` is also included. These are the node features to be used for tabular machine learning, as well as graph machine learning. 
2. **SEC filing text** dataset: Contains the Management Discussion and Analysis (MD&A) section of companies' SEC 10-K/Q filings, which are used to construct an undirected and unweighted corporate graph, denoted `CorpNet`. 

The goal is to use numerical and categorical features in the **tabular** dataset and text data in the **SEC filing text** dataset to create a corporate graph, 
and use the tabular and graph data to predict the rating of each company in the _Rating_ column.

Here are the first 5 observations of the **tabular** dataset.

| Observation  | Industry_code  | CurrentLiabs  |  TotalLiabs   |  RetainedEarnings   |  CurrentAssets   |  NetSales   |   EBIT   |  MktValueEquity   | Rating  |
|:------------:|:--------------:|:-------------:|:-------------:|:-------------------:|:----------------:|:-----------:|:--------:|:-----------------:|:-------:|
|      0       |       D        |    20.8683    |    50.5015    |       32.5473       |     24.8713      |   14.7749   | 1.33493  |      238.572      |    2    |
|      1       |       D        |    19.6230    |    50.3521    |       28.7603       |     29.1036      |   12.4473   | 1.13706  |      215.396      |    2    |
|      2       |       D        |    21.4741    |    55.0071    |       29.9036       |     24.5609      |   13.0745   | 1.39008  |      232.182      |    2    |
|      3       |       I        |    20.1353    |    50.2253    |       29.7489       |     26.0704      |   15.7651   | 1.20360  |      228.732      |    2    |
|      4       |       D        |    17.4178    |    49.8936    |       29.4803       |     24.1608      |   12.6121   | 1.18982  |      236.956      |    2    |


Here are the first 5 observations of the **SEC filing text** dataset. Due to its long length, only part of the SEC filing text is shown for each observation.

| Observation |                                                                                                                                                                                                                                                              SEC 10-K/Q filings: MD&A section                                                                                                                                                                                                                                                               |   
|:-----------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|      0      |                               Management's Discussion and Analysis of Financial Condition and Results of Operations Results of Operations The following table sets forth net sales and income by reportable segment and on a consolidated basis: (1)  After elimination of intra- and intersegment sales, which are not significant in amount.   (2)  Segment operating income represents net sales less all direct costs and expenses (including certain administrative and other expenses) applicable to each segment, ...                                |
|      1      |                                             Results of operations for the third quarter of 2013 compared with the third quarter of 2012 For the quarter ended September 30, 2013, the Company established records for sales and operating income. The Company achieved these results primarily through contributions from the acquisitions of Controls Southeast ("CSI") in August 2013 and Micro-Poise Measurement Systems ("Micro-Poise") in October 2012, as well as our Operational Excellence initiatives.                                             |
|      2      |    Total international sales for the third quarter of 2010 were $301.2 million or 46.7\% of consolidated net sales, an increase of $58.8 million or 24.3% when compared with international sales of $242.4 million or 48.8\% of consolidated net sales for the third quarter of 2009. The $58.8 million increase in international sales resulted from higher international order rates driven by the Company's net sales increase mentioned above, as well as continued expansion into Asia and includes the effect of foreign currency translation. ...    |
|      3      | In certain international geographies where we have not invested to build a local sales force, we rely on resellers that serve as outside sales agents for the sale of our Promoted Products. In 2013, we and our resellers sold our Promoted Products to advertisers in over 20 countries outside of the United States. We record advertising revenue based on the billing location of our advertisers, rather than the location of our users. We are headquartered in San Francisco, California, and have offices in over 20 cities around the world. ...  |
|      4      |    Selling expense increased $9.2 million or 13.9\% for the third quarter of 2011 driven by the increase in net sales noted above. Selling expenses, as a percentage of net sales, decreased to 10.0\% for the third quarter of 2011, compared with 10.2\% for the third quarter of 2010. Base business selling expense increased approximately 9\% for the third quarter of 2011, which was in line with internal sales growth. Corporate administrative expenses for the third quarter of 2011 were $11.2 million, an increase of $1.0 million or  ...    |

The following shows how the label column `Rating` is generated.

The model used is a well-known bankruptcy prediction approach, from the seminal paper by Altman (1968). For more information, see [description](https://www.creditguru.com/index.php/bankruptcy-and-insolvency/altman-z-score-insolvency-predictor).

The model uses 8 inputs converted into 5 financial ratios: (i) Current assets (**CA**); (ii) Current liabilities (**CL**); (iii) Total Liabilities (**TL**); (iv) **EBIT** (earnings before interest and taxes); (v) Total Assets (**TA**); (vi) Net Sales (**NS**); (vii) Retained earnings (**RE**); and (viii) Market value of equity (**MVE**). There are strict relationships between these 8 items, which are modeled as stipulated in the Altman model through the following five ratios: 

- A: EBIT / Total Assets 
- B: Net Sales / Total Assets
- C: Mkt Value of equity / Total Liabilities
- D: Working Capital / Total Assets
- E: Retained Earnings / Total Assets
 
The Z-score is then computed by the following formula. Specifically, it is estimated using Linear Discriminant Analysis (LDA) and the coefficients in the formula are published widely by Altman.

**Zscore = 3.3 A + 0.99 B + 0.6 C + 1.2 D + 1.4 E**

High quality companies have high Z-scores and low quality companies have low ones. The Z-scores are calibrated to broad balance-sheet, income-statement, and market data using averages for the U.S. economy from the following sources: 

- [Balance sheet data](https://fred.stlouisfed.org/release/tables?rid=434&eid=196197)
- [Income statement data](https://fred.stlouisfed.org/release/tables?rid=434&eid=195208)
- [Price to book](http://pages.stern.nyu.edu/~adamodar/New\_Home\_Page/datafile/pbvdata.html)

The numbers taken from these sources are used to extract total values for US companies for the eight inputs previously mentioned. In addition, we obtain the average price-to-book (**P2B**) ratio for US companies, book equity value (**EQ**) and then generate market value of equity (**MVE**) as **P2B (EQ+RE)**. Working capital (**WC**) is as usual, **CA-CL**, the difference between current assets and current liabilities. We also normalized the data by Total Assets. This normalization does not impact the Altman ratios in any way. 

This dataset of simulated companies and their financials is then enhanced with a column of ratings. In particular, the ratings are mapped (with some noise) to the Z-score, such that firms with higher Z-score are assigned higher quality ratings and firms with lower Z-score get lower quality ratings. 

We use the (MD&A) section of companies' SEC 10-K/Q filings to construct a corporate graph, `CorpNet`. The approach is based on the idea that companies facing similar risks will evidence similarities in their forward-looking statements in the MD&A section. Therefore, we construct a graph where each node is a company in the dataset and two companies are linked if the cosine similarity of their MD&A document embeddings is greater than 0.5. The notebook that accompanies this solution contains this generated graph data in addition to the tabular data described previously. 

>**<span style="color:RED">Important</span>**: 
>This solution is for demonstrative purposes only. It is not financial advice and should not be relied on as financial or investment advice. The associated notebooks, including the trained model, use synthetic data and are not intended for production. While text from SEC filings is used, the financial data is synthetically and randomly generated and has no relation to any company's true financials. Hence, the synthetically generated ratings also do not have any relation to a company's true rating. 

**The dataset is downloaded directly from [SEC](https://www.sec.gov/os/accessing-edgar-data). If you have any question about the dataset, please directly reach out to the owner.**


## What are the outputs?

* A [Graph Neural Network GraphSAGE](https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf) trained on the synthetic tabular dataset and the corporate graph. 
* An [AutoGluon Tabular](https://auto.gluon.ai/stable/tutorials/tabular_prediction/index.html) model trained on these datasets.
* Predictions of the probability for each company having an investment grade rating (which includes `AAA`, `AA`, `A`, `BBB`). If the estimated probability of a company is over a threshold, it is classified as an investment grade rating. Otherwise, its rating is classified as below investment grade. 



## Why is a Graph Neural Network (GNN) a good model for this problem?

Since the datasets used contain the numerical and categorical features for each company such as the current liabilities (_CurrentLiabs_), retained earnings (_RetainedEarnings_), and industry code (_industry_code_) as well as the corporate network based on the SEC filings text, graph neural networks are a fitting choice. 

Graph neural networks utilize all the constructed information above to learn a hidden representation (embedding) for each company. The hidden representation is then used as input for a linear classification layer to determine whether the company has an investment-grade or below investment-grade rating.


## What algorithms are used?

* We implemented [Graph Neural Network GraphSAGE](https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf) which is a state-of-the-art Graph Neural Network (GNN) model that leverages node feature information (e.g., financials of the companies) from the network neighbors to efficiently generate node embeddings for previously unseen data. 

* We used [AutoGluon Tabular](https://auto.gluon.ai/stable/tutorials/tabular_prediction/index.html) as the baseline model to compare the performance of tabular ML models with that of GNNs. 

## How do we do it?

This solution is presented in the following stages:

* Demonstrate how to construct a network of connected companies using the MD&A section from [SEC 10-K/Q filings](https://www.sec.gov/edgar/searchedgar/companysearch.html). Companies with similar forward looking statements are likely to be connected for credit events. These connections are represented in a graph. The solution provides a single function that generates the network. 

* For graph node features, the solution uses the variables in the Altman Z-score model and the industry category of each firm. These are provided in a synthetic dataset made available for demonstration purposes. 

* Train a rating classifier [Graph Neural Network GraphSAGE](https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf) using both the graph and tabular data.  

* Train an [AutoGluon Tabular](https://auto.gluon.ai/stable/tutorials/tabular_prediction/index.html) on the tabular data as a baseline to compare the performance of models with and without the graph information. 

* Demonstrate how to use hyper-parameter optimization (HPO) to find an optimal [Graph Neural Network GraphSAGE](https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf) model. 

* Show how to deploy this model to an endpoint and how to invoke this endpoint for inference. 

The trained [Graph Neural Network GraphSAGE](https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf) model with HPO shows better predictive performance than the baseline [AutoGluon Tabular](https://auto.gluon.ai/stable/tutorials/tabular_prediction/index.html) on a hold-out test data. The evaluation metrics include [F1 score](https://en.wikipedia.org/wiki/F-score), [Precision and Recall](https://en.wikipedia.org/wiki/Precision_and_recall), [Accuracy](https://en.wikipedia.org/wiki/Accuracy_and_precision), [ROC-AUC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic), [Matthews correlation coefficient (MCC)](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html), and [Balanced Accuracy](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html).

## What is the estimated cost?

Running the solution end-to-end costs less than $5 USD. Please make sure you have read the cleaning up instructions [here](#cleaning-up).

## Cleaning up

When you've finished with this solution, delete all unwanted AWS resources. AWS CloudFormation can be used to automatically delete all standard resources that have been created by the solution and notebook. Navigate to the [AWS CloudFormation Console](https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/cfn-using-console.html), 
and delete the parent stack. Choosing to delete the parent stack will automatically delete the nested stacks.

**Caution:** You need to manually delete any extra resources that you may have created in this notebook. Some examples include, extra Amazon S3 buckets (to the solution's default bucket), extra Amazon SageMaker endpoints (using a custom name), and extra Amazon ECR repositories.


## Citations

- Altman, Edward I. “Financial Ratios, Discriminant Analysis and the Prediction of Corporate Bankruptcy.” The Journal of Finance 23, no. 4 (1968): 589–609. https://doi.org/10.1111/j.1540-6261.1968.tb00843.x.

- Altman, Edward I. “A Fifty-Year Retrospective on Credit Risk Models, the Altman Z -Score Family of Models and Their Applications to Financial Markets and Managerial Strategies.” Journal of Credit Risk 14, no. 4 (December 11, 2018): 1–34.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.
