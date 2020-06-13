<h1 align="center">
    Predict Stock
</h1>

<p align="center">
    Classification model for predicting stock market trending, based on machine learning techniques, such as Extremely Randomized Trees, K-Means, Support Vector Machines and K-Fold Cross-Validation.
</p>

<p align="center">
    Coding presented as part of the Capstone Project in Computational Engineering of the <a href='http://www.ufjf.br/ufjf/'>Universidade Federal de Juiz de Fora</a>
</p>

<p align="center">
    <a href="#chart_with_upwards_trend-problem-presentation">Problem presentation</a>&nbsp;&nbsp;|&nbsp;&nbsp;
    <a href="#pencil-dependencies">Dependencies</a>&nbsp;&nbsp;|&nbsp;&nbsp;
    <a href="#runner-how-to-run">How to run</a>&nbsp;&nbsp;|&nbsp;&nbsp;
    <a href="#computer-technologies">Technologies</a>&nbsp;&nbsp;|&nbsp;&nbsp;
    <a href="#book-main-references">References</a>&nbsp;&nbsp;|&nbsp;&nbsp;
    <a href="#cookie-credits">Credits</a>&nbsp;&nbsp;
</p>

## :chart_with_upwards_trend: Problem presentation

This project aims to demonstrate an application of machine learning methods in predicting the oscillation of the stock market. Different techniques will be employed in order to create a more robust model and improve the predictions accuracy.

The pipeline and short description of the employed methods are as follows:

1. **Data acquiring:** Acquire stock history value using *pandas-datareader*.

2. **Data preparation:** Remove missing and unecessary data using *pandas*.

3. **Apply indicators:** Apply financial indicators in data collected using *pandas*.

4. **Feature selection: Extremely Randomized Trees**

    Supervised method used to solve classification and regression problems. It is a variation of the classic *Random Forests*, which adds more randomization in node partition and choice of training sets. These changes reduce the bias and the variance of the model, proposing to alleviate the problems of underfitting and overfitting, respectively.

    In the present problem, this method was used as a feature selector, measuring the importance of each financial indicator in the prediction.

5. **Clusterization: K-Means**

    Unsupervised method used in partitioning or clustering, which organizes the elements of a set into groups (clusters) so that the elements resemble each other. The number of clusters must be defined initially and this becomes the starting point of the method.

    This method was employed to clusterize the data and reduce the number of support vectors in the next step.

6. **Classification: Support Vector Machines**

    Supervised method used to solve classification and regression problems with linear or nonlinear data. This methods aims to find the hyperplane that separates the training samples of the problem in their respective classes.
    
    This is the main step of this pipeline, where the classified data stands for upward or downward stock oscillation.

7. **Parameter tuning: K-Fold Cross-validation**

    Finally, we need a method to evaluate the parameters of the chosen model and tell what is the best combination of them.

    This method randomly split the data set in *K* subsets. In each iteration, one set is used for test and the remaining *K-1* sets are employed for training, make possible to measure the accuracy and tuning the parameters.

## :pencil: Dependencies

Besides, of course, Python, you will need NumPy library for numerical operations, Matplotlib library for plotting, pandas and pandas-datareader to deal with datasets, and scikit-learn to perform the machine learning algorithms itself.

You may install all dependencies with the following command:

```sh
pip3 install numpy matplotlib pandas pandas-datareader scikit-learn
```

## :runner: How to run

After install <a href="#pencil-dependencies">dependencies</a>, open your terminal in the folder you want to clone the project:

```sh
git clone https://github.com/LorranSutter/PredictStock-SVM.git
```

First, you will need to acquire stocks data. The following command uses the file *db/NASDAQ.csv* as reference to list all stocks to get data. However, if you do not want to get the data from all the available stocks, just change the file removing unwanted stocks.

```sh
python3 initGetData.py
```

After acquire the stocks data, results will be stored in *db/stocks* folder. Then, you may run the main code changing the variable *ticker* inside the code with the desired ticker.

```sh
python3 main.py
```

## :computer: Technologies

- [Python](https://www.python.org/) - interpreted, high-level, general-purpose programming language
- [Pandas](https://pandas.pydata.org/) - data analysis and manipulation tool
- [Pandas datareader](https://pandas-datareader.readthedocs.io/en/latest/) - data access for pandas
- [Sklearn](https://scikit-learn.org/stable/) - machine learning library
- [NumPy](https://numpy.org/) - general-purpose array-processing package
- [Matplotlib](https://matplotlib.org/) - plotting library for the Python

## :book: Main references

- VO, V.; LUO, J.; VO, B. *Time series trend analysis based on k-means and support vector machine*. v. 35, p. 111–127, 1 2016
- LEE, M.-C. *Using support vector machine with a hybrid feature selection method to the stock trend prediction. Expert Systems with Applications*, v. 36, n. 8, p. 10896 – 10904, 2009. ISSN 0957-4174
- XU, Y.; LI, Z.; LUO, L. *A study on feature selection for trend prediction of stock trading price*. Jun 2013
- LIMA, M. L. *Um modelo para predição de bolsa de valores baseado em mineração de opinião*, 2016. Dissertação de Mestrado (Programa de Pós-Graduação em Engenharia de Eletricidade), UFMA (Universidade Federal do Maranhão), São Luı́s, Brasil

## :cookie: Credits

Thanks for indicators implementation of [Bruno Franca pandasImpl.py](https://github.com/panpanpandas/ultrafinance/blob/master/ultrafinance/pyTaLib/pandasImpl.py)
