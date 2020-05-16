<h1 align="center">
    Predict Stock
</h1>

<p align="center">
    Classification model for predicting stock market trending, based in machine learning techniques, such as Extremely Randomized Trees, K-Means, Support Vector Machines and K-Fold Cross-Validation.
</p>

<p align="center">
    Coding presented as part of the Final Term Paper in Computacional Engineering of the <a href='http://www.ufjf.br/ufjf/'>Universidade Federal de Juiz de Fora</a>
</p>

<p align="center">
    <a href="#pencil2-machine-learning">Machine learning</a>&nbsp;&nbsp;|&nbsp;&nbsp;
    <a href="#chart_with_upwards_trend-problem-presentation">Problem presentation</a>&nbsp;&nbsp;|&nbsp;&nbsp;
    <a href="#pencil-dependencies">Dependencies</a>&nbsp;&nbsp;|&nbsp;&nbsp;
    <a href="#runner-how-to-run">How to run</a>&nbsp;&nbsp;|&nbsp;&nbsp;
    <a href="#computer-technologies">Technologies</a>&nbsp;&nbsp;|&nbsp;&nbsp;
    <a href="#book-main-references">References</a>&nbsp;&nbsp;|&nbsp;&nbsp;
    <a href="#cookie-credits">Credits</a>&nbsp;&nbsp;
</p>

## :pencil2: Machine learning

Machine learning is a sub-field of [Artificial Intelligence](https://en.wikipedia.org/wiki/Artificial_intelligence) that aims to create algorithms to improve automatically through data.

## :chart_with_upwards_trend: Problem presentation

## :pencil: Dependencies

Besides, of course, Python, you will need NumPy library for numerical operations, Matplotlib library for plotting, pandas and pandas-datareader to deal with datasets, and scikit-learn to perform the machine learning algorithms itself.

You may install all dependencies with the following command:

```sh
pip3 install numpy matplotlib pandas pandas-datareader scikit-learn
```

## :runner: How to run

After install <a href="#pencil-dependencies">dependencies</a>, open your terminal in the folder you want to clone the project:

```sh
# Clone this repo
git clone https://github.com/LorranSutter/PredictStock.git
```

## :computer: Technologies

- [Python](https://www.python.org/)
- [Pandas](https://pandas.pydata.org/)
- [Pandas datareader](https://pandas-datareader.readthedocs.io/en/latest/)
- [Sklearn](https://scikit-learn.org/stable/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)

## :book: Main references

- VO, V.; LUO, J.; VO, B. *Time series trend analysis based on k-means and support vector machine*. v. 35, p. 111–127, 1 2016
- LEE, M.-C. *Using support vector machine with a hybrid feature selection method to the stock trend prediction. Expert Systems with Applications*, v. 36, n. 8, p. 10896 – 10904, 2009. ISSN 0957-4174
- XU, Y.; LI, Z.; LUO, L. *A study on feature selection for trend prediction of stock trading price*. Jun 2013
- LIMA, M. L. *Um modelo para predição de bolsa de valores baseado em mineração de opinião*, 2016. Dissertação de Mestrado (Programa de Pós-Graduação em Engenharia de Eletricidade), UFMA (Universidade Federal do Maranhão), São Luı́s, Brasil

## :cookie: Credits

Thanks for indicators implementation of [Bruno Franca pandasImpl.py](https://github.com/panpanpandas/ultrafinance/blob/master/ultrafinance/pyTaLib/pandasImpl.py)