This note describes the steps and code used for generating synthetic data. This is done in two steps with two corresponding Jupyter notebooks. In the first step, we use a donwloaded collection of SEC filings and score them for various attributes that have positive or negative sentiment, to give net sentiment for each company. In the second step, synthetic financial data for the commpanies is generated, from which Altman's Z-score is calculated for credit quality. The dataset from the first step is then joined with the data from the second step such that the correlation between net sentiment (first step) and credit score (second step) is positive. 

# Files

1. `Synthetic_DataGen_CCR_1.ipynb`: Uses SEC filings, NLP scores them to get a net score. Stores an output file: `CCR_synthetic_text_nlp_scores.csv`. The processing here requires using AWS SageMaker JumpStart, so the output of this procedure has been provided in the file `CCR_synthetic_text_nlp_scores.csv`. 

2. `Synthetic_DataGen_CCR_2.ipynb`: Generates simulated financials, creates Altman's Z-score with noise and then also joins the text on ranked scores. This gives the complete dataset. 

3. `TEST_CCR_SYNTH_wGraphTab.ipynb`: 
  - Generates the graph from SEC text (MD&A) and prepares source and destination node lists. The graph is symmetric and unweighted (all links have weight 1). Uses the graph to create 3 features using the `networkx` library. 


