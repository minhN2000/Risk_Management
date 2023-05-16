# Credit_Risk_Management

Credit risk management plays a crucial role in assessing and mitigating potential losses in lending and financial institutions. In this repository, I explore the fundamental concepts of credit risk management and build three models to measure expected loss: Probability of Default (PD), Exposure at Default (EAD), and Loss Given Default (LGD) using the loan data set from All Lending Club (Kaggle)[https://www.kaggle.com/datasets/wordsforthewise/lending-club?datasetId=902&sortBy=voteCount&sort=votes]

## Models

### Probability of Default (PD)

The Probability of Default (PD) model quantifies the likelihood that a borrower will fail to make full repayment of the loan. This model helps us estimate the creditworthiness of borrowers by considering various factors such as financial indicators, credit history, and macroeconomic conditions. By developing a PD model, we aim to assess the probability of default accurately.

### Exposure at Default (EAD)

The Exposure at Default (EAD) model calculates the expected value of the loan at the time of default. It provides insights into the amount that remains unpaid when the borrower defaults. By understanding the EAD, we can better evaluate the potential loss exposure associated with credit risk. This model considers factors like collateral value, loan terms, and borrower behavior.

### Loss Given Default (LGD)

The Loss Given Default (LGD) model measures the potential loss amount in the event of default, expressed as a percentage of the Exposure at Default (EAD). LGD provides an estimate of the financial impact incurred when a borrower fails to repay the loan fully. This model considers recovery rates, collateral values, and other relevant factors to quantify the potential loss.

## Repository Structure

The repository is organized as follows:

```bash
├───data
│   ├───original
│   ├───test
│   └───train
├───log
├───models
│   └───scalers
├───notebooks
└───src
    └───util
```

## Getting Started

### a. Requirements

To explore the models and analyses developed in this repository, follow these steps:

1.  Clone the repository:

```bash
git clone https://github.com/minhN2000/Risk_Management.git
```
2.  Install the necessary dependencies. You can find the required packages and versions in the `requirements.txt` file.
```bash
`pip install -r requirements.txt` 
```

### b. Running 
1. Start with the `notebooks/PD_training.ipynb` to understand the Exploratory Data Analysis (EDA) process.
2. Continue on `notebooks/LGD_training.ipynb` then `notebooks/EAD_training.ipynb`
3. End with  `notebooks/evaluation.ipynb` which provides the evaluation for the 3 models.
4. Experiment with the provided data files stored in the `data/` directory. 

## Contributions

Contributions to this repository are welcome! If you have any suggestions, bug fixes, or enhancements related to risk management, credit risk, or the development of PD, EAD, and LGD models, please feel free to open an issue or submit a pull request. I really appreciate any comment from anybody because it will help me a lot to understand more about the field. Happy reading!