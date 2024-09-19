# Awesome Python Tools


**Awesome Python Tools** is a curated repository featuring a wide range of Python libraries and tools across diverse fields, including programming, statistical analysis, data science, machine learning, deep learning, and more.

This collection is designed to serve developers, data scientists, and researchers by providing essential tools for various tasks, from software development to advanced data analysis.

In this repository, you’ll find an extensive range of Python tools categorized by their specific use cases.

This collection includes popular libraries for:

   - Programming and Software Development
   - Data Manipulation and Analysis
   - Web Development
   - Machine Learning and Deep Learning
   - Time Series Analysis
   - Image Processing, and much more.

The goal of this repository is to provide a centralized resource where developers and data scientists can explore the best tools available, saving time and effort when searching for the right library for their tasks.


**Author and Maintainer**: Dr. Saad Laouadi

## Table of Content

1. [Programming](#programming)
   - 1.1 [Development Tools](#development-tools)
   - 1.2 [Environment and Dependency Management](#environment-and-dependency-management)
   - 1.3 [Testing](#testing)
   - 1.4 [Desktop Applications](#desktop-applications)
   - 1.5 [Mobile Applications](#mobile-applications)
2. [Data Science](#data-science)
   - 2.1 [Data Manipulations](#data-manipulations)
3. [Data Visualizations](#data-visualizations)
4. [Machine Learning](#machine-learning)
5. [Deep Learning](#deep-learning)
6. [Text Analysis](#text-analysis)
7. [Speech Recognition](#speech-recognition)
8. [Time Series Analysis](#time-series-analysis)
9. [Image Processing](#image-processing)
10. [Statistical Analysis](#statistical-analysis)
11. [Web Development](#web-development)
12. [Animation](#animation)
13. [Command Line Tools](#command-line-tools)
14. [3D CAD](#3d-cad)
15. [GIS](#gis-geographic-information-systems)
16. [Remote Sensing](#remote-sensing)
17. [Natural Language Processing (NLP)](#natural-language-processing-nlp)
18. [Bioinformatics](#bioinformatics)
19. [Robotics](#robotics)
20. [Game Development](#game-development)
21. [Computer Vision](#computer-vision)
22. [Cryptography and Security](#cryptography-and-security)
23. [Audio Processing](#audio-processing)
24. [Financial and Quantitative Analysis](#financial-and-quantitative-analysis)
25. [Physics Simulations](#physics-simulations)
26. [Mathematics and Symbolic Computation](#mathematics-and-symbolic-computation)
27. [Automation](#automation)
28. [DevOps and Monitoring](#devops-and-monitoring)
29. [Quantum Computing](#quantum-computing)


## Programming


#### Development Tools
- [PyCharm](https://www.jetbrains.com/pycharm/): Python IDE for Professional Developers.
- [Visual Studio Code](https://code.visualstudio.com/): Open-source code editor that supports Python with extensions.
- [Sublime Text](https://www.sublimetext.com/): Lightweight text editor with Python support.
- [Eclipse + PyDev](https://www.eclipse.org/): IDE with the PyDev plugin for Python development.
- [Thonny](https://thonny.org/): Beginner-friendly Python IDE with a focus on education.
- [Wing](https://wingware.com/): Python IDE with powerful debugging and testing tools.
- [Jupyter Notebooks](https://jupyter.org/): Web-based notebook environment for Python, ideal for data science and interactive computing.
- [PyInstaller](https://www.pyinstaller.org/): Convert Python applications into standalone executables.
- [Black](https://github.com/psf/black): Python code formatter.
- [Flake8](https://flake8.pycqa.org/en/latest/): Python linting tool to ensure PEP8 compliance.
- [Prettier](https://prettier.io/): Code formatter that works with Python (via plugins).
- [Babel](https://babeljs.io/): JavaScript compiler that can be integrated with Python development.
- [Tox](https://tox.readthedocs.io/): Tool for automating testing across multiple Python environments.
- [Pylint](https://pylint.pycqa.org/en/latest/): A Python static code analysis tool that looks for programming errors, helps enforce a coding standard, and more.

#### Environment and Dependency Management
- [Poetry](https://python-poetry.org/): Dependency management and packaging tool for Python.
- [Pipenv](https://pipenv.pypa.io/en/latest/): Official Python tool for managing environments and dependencies.
- [Conda](https://docs.conda.io/): Package, dependency, and environment management for Python and other languages.
- [Virtualenv](https://virtualenv.pypa.io/en/latest/): Tool to create isolated Python environments.
- [Pip](https://pip.pypa.io/en/stable/): The Python package installer.
- [Anaconda](https://www.anaconda.com/): Distribution of Python and R for scientific computing and data science.
- [PyEnv](https://github.com/pyenv/pyenv): Simple Python version management.
- [Nox](https://nox.thea.codes/en/stable/): A tool that automates testing in multiple Python environments, similar to Tox but simpler.
- [Hatch](https://hatch.pypa.io/latest/): Modern project, package, and environment manager for Python.
- [asdf](https://asdf-vm.com/): Version manager for multiple runtime environments including Python.
- [Docker](https://www.docker.com/): Containerization platform often used to create isolated development environments for Python applications.
- [Venv](https://docs.python.org/3/library/venv.html): Built-in Python module to create lightweight, isolated virtual environments.
- [Pipx](https://pypa.github.io/pipx/): A tool to install and run Python applications in isolated environments.
- [PyScaffold](https://pyscaffold.org/): Framework for setting up Python projects with best practices.

- [pytest](https://pytest.org/): Simple, yet powerful testing framework for Python.
- [unittest](https://docs.python.org/3/library/unittest.html): Python's built-in testing framework.
- [tox](https://tox.readthedocs.io/en/latest/): Automate testing in multiple Python environments.
- [Hypothesis](https://hypothesis.readthedocs.io/en/latest/): Property-based testing library.
- [nose2](https://docs.nose2.io/en/latest/): Successor to `nose`, a tool that extends unittest to make testing easier.
- [Testify](https://github.com/Yelp/Testify): A test framework that aims to improve upon `unittest`.
- [Robot Framework](https://robotframework.org/): A generic open-source automation framework for acceptance testing.
- [Behave](https://behave.readthedocs.io/en/latest/): Behavior-driven development (BDD) framework for Python.
- [pytest-bdd](https://pytest-bdd.readthedocs.io/): BDD plugin for `pytest`.
- [coverage.py](https://coverage.readthedocs.io/): Code coverage measurement tool for Python.
- [mock](https://docs.python.org/3/library/unittest.mock.html): Built-in Python library for mocking during unit tests.
- [responses](https://github.com/getsentry/responses): Mock out HTTP requests made using `requests` during testing.
- [VCR.py](https://vcrpy.readthedocs.io/): Record HTTP interactions and replay them during tests.
- [webtest](https://webtest.readthedocs.io/en/latest/): Test your web applications with the `WSGI` interface.
- [Selenium](https://www.selenium.dev/): Browser automation tool used for end-to-end testing of web applications.
- [Locust](https://locust.io/): Scalable user load testing tool for web applications.
- [pytest-django](https://pytest-django.readthedocs.io/): Django plugin for pytest, enabling easy testing of Django applications.
- [pytest-flask](https://pytest-flask.readthedocs.io/): A pytest plugin for testing Flask applications.
- [Factory Boy](https://factoryboy.readthedocs.io/): A fixtures replacement based on the factory concept, helping to create test data.
- [allure-pytest](https://docs.qameta.io/allure/): Allure integration with `pytest` for advanced reporting.
- [Green](https://pypi.org/project/green/): A clean, colorful test runner that integrates with `unittest`.
- [testcontainers-python](https://github.com/testcontainers/testcontainers-python): Provides Python support for using Testcontainers, allowing lightweight, disposable containers for integration tests.
- [PyHamcrest](https://github.com/hamcrest/PyHamcrest): Matcher objects for writing clearer tests with assertions.
- [sure](https://sure.readthedocs.io/): Fluent assertion library for Python testing.


## Data Science

### Data Manipulations


#### General-Purpose Data Manipulation
- [Pandas](https://pandas.pydata.org/): The most widely used library for data manipulation and analysis with DataFrames.
- [PyJanitor](https://pyjanitor.readthedocs.io/): Built on Pandas, provides utilities for data cleaning and preprocessing.
- [SQLite](https://docs.python.org/3/library/sqlite3.html): Built-in Python library for handling structured data stored in SQLite databases.

#### High-Performance Data Manipulation
- [Polars](https://www.pola.rs/): A highly performant DataFrame library optimized for both small and large datasets.
- [Modin](https://modin.readthedocs.io/): Drop-in replacement for Pandas, designed to scale Pandas-like operations using parallel processing.
- [Dask](https://dask.org/): A parallel computing library for scalable data manipulation, especially useful for large datasets.
- [Vaex](https://vaex.io/): Out-of-core DataFrame library, optimized for working with datasets larger than memory.
- [cuDF](https://github.com/rapidsai/cudf): GPU-accelerated DataFrame library based on Pandas, part of the RAPIDS ecosystem for faster large-scale data processing.
- [Bcolz](https://bcolz.readthedocs.io/en/latest/): Optimized for in-memory analytics, provides columnar and compressed data containers for faster manipulation of large datasets.

#### Distributed and Parallel Data Processing
- [Dask](https://dask.org/): Facilitates parallel computing across multiple cores or clusters for scaling DataFrame operations.
- [PySpark](https://spark.apache.org/docs/latest/api/python/): Distributed data manipulation tool built on Apache Spark, ideal for big data processing.
- [Koalas](https://koalas.readthedocs.io/): Pandas-like API built on Apache Spark, allowing distributed DataFrame operations while maintaining a familiar syntax.
- [Mars](https://mars-project.readthedocs.io/en/latest/): Supports large-scale data computation using multi-dimensional arrays and DataFrames in distributed settings.

#### Specialized Data Handling
- [Xarray](http://xarray.pydata.org/): Best for multi-dimensional labeled arrays, used in scientific computing like climate data analysis.
- [GeoPandas](https://geopandas.org/): Specialized for geospatial data manipulation, supporting shapefiles, geoJSON, and spatial operations.
- [Awkward Array](https://awkward-array.org/): Designed for handling complex and nested data structures, useful for JSON-like data and scientific research.
- [SQLAlchemy](https://www.sqlalchemy.org/): SQL toolkit and ORM for handling complex relational database queries using Python code.

#### Efficient Storage and I/O
- [PyArrow](https://arrow.apache.org/): Enables efficient columnar storage and fast I/O for large-scale data manipulation, commonly used with Parquet and Feather file formats.
- [Vaex](https://vaex.io/): Optimized for reading large datasets from file formats like HDF5 and Arrow without loading them fully into memory.



## Data Visualizations

#### General-Purpose Plotting Libraries
- [Matplotlib](https://matplotlib.org/): The foundational 2D plotting library in Python, offering a variety of static, animated, and interactive plots. It is highly customizable and can create publication-quality figures.
- [Seaborn](https://seaborn.pydata.org/): Built on top of Matplotlib, Seaborn simplifies the process of creating complex statistical plots with aesthetically pleasing defaults and themes.
- [Plotly](https://plotly.com/python/): A powerful library for creating interactive, web-ready plots. It supports a wide range of chart types, including 3D plots, geographic maps, and dashboards.
- [Bokeh](https://bokeh.org/): Focused on creating interactive visualizations that can be deployed in web browsers. Bokeh is great for real-time streaming data and interactive dashboards.
- [Altair](https://altair-viz.github.io/): A declarative statistical visualization library that automates chart generation based on data structures. It is designed to work with Pandas dataframes and is highly effective for quick exploratory data analysis.
- [ggplot](https://github.com/yhat/ggpy): A Python implementation of the popular ggplot2 R package, providing a grammar of graphics-based approach to data visualization.

#### High-Performance & Large-Scale Visualization
- [Holoviews](https://holoviews.org/): Simplifies the creation of interactive visualizations across different plotting backends (e.g., Matplotlib, Bokeh, Plotly). Ideal for building dashboards and visualizing large datasets.
- [Datashader](https://datashader.org/): Designed to handle extremely large datasets, Datashader renders even billions of data points interactively by rasterizing data.
- [Vaex](https://vaex.io/): Primarily known for its data manipulation capabilities, Vaex also provides built-in fast visualization of large datasets, including scatter plots and histograms.
- [hvPlot](https://hvplot.holoviz.org/): High-level plotting API that simplifies the creation of interactive visualizations for Pandas, Dask, and Xarray data structures, integrating with Holoviews and Bokeh.

#### Web and Interactive Visualizations
- [Dash](https://dash.plotly.com/): A Python framework for building interactive web applications and dashboards. Built on top of Plotly, Dash is perfect for creating data-driven apps without requiring frontend knowledge.
- [Streamlit](https://streamlit.io/): A fast and easy way to build and share web applications directly from Python scripts. Streamlit simplifies the process of creating interactive dashboards with just a few lines of code.
- [PyWebIO](https://pywebio.readthedocs.io/): A framework for creating interactive web applications without needing front-end development skills. It supports plotting libraries like Matplotlib, Plotly, and Bokeh for visualizing data in a browser.
- [Flask-Dashboard](https://flask-dashboard.readthedocs.io/): A lightweight dashboarding tool for building web-based visualizations on top of Flask.

#### Geographic and Geospatial Visualizations
- [Folium](https://python-visualization.github.io/folium/): Simplifies the creation of interactive maps using Leaflet.js, allowing visualizations of geospatial data.
- [Geopandas](https://geopandas.org/): Extends Pandas with support for geospatial data, enabling geographic visualizations and spatial operations.
- [Cartopy](https://scitools.org.uk/cartopy/docs/latest/): A library for cartographic projections and geospatial visualizations.
- [Kepler.gl](https://kepler.gl/): A powerful geospatial visualization tool, developed by Uber, for large-scale geographic data visualizations with beautiful, interactive maps.

#### Specialized and Domain-Specific Visualization
- [NetworkX](https://networkx.github.io/): For visualizing and analyzing complex networks and graphs, NetworkX is widely used in network analysis and social network visualizations.
- [PyGraphviz](https://pygraphviz.github.io/): A Python interface to Graphviz, used for rendering graph visualizations, particularly useful in network and process flow visualizations.
- [mplfinance](https://github.com/matplotlib/mplfinance): Specializes in creating financial charts, such as candlestick, line, and volume plots for stock and trading data.
- [PyMC3](https://docs.pymc.io/plotting.html): Provides visualization tools for probabilistic programming and Bayesian inference.

#### Animation and 3D Visualizations
- [Manim](https://www.manim.community/): A Python library for creating high-quality mathematical animations and presentations, widely used for educational videos.
- [Mayavi](https://docs.enthought.com/mayavi/mayavi/): A 3D scientific data visualization tool that works well with NumPy arrays and has advanced rendering capabilities.
- [PyVista](https://docs.pyvista.org/): Simplifies the visualization of 3D data and meshes, providing an intuitive API for complex geometric and volumetric data.
- [Vispy](https://vispy.org/): A high-performance interactive 2D/3D data visualization library that leverages the power of OpenGL.
- [Matplotlib 3D](https://matplotlib.org/stable/gallery/mplot3d/index.html): Built-in 3D plotting capabilities in Matplotlib, ideal for basic 3D visualizations and surface plots.


## Machine Learning

### General Machine Learning Libraries
- [Scikit-learn](https://scikit-learn.org/): The foundational machine learning library for Python. It provides simple and efficient tools for data mining, data analysis, and machine learning with support for classification, regression, clustering, dimensionality reduction, and model evaluation.
- [PyCaret](https://pycaret.org/): An open-source, low-code machine learning library that automates many aspects of machine learning pipelines, including preprocessing, model selection, and tuning.

### Gradient Boosting Libraries
- [XGBoost](https://xgboost.readthedocs.io/en/latest/): A highly efficient and scalable implementation of gradient boosting that is widely used in machine learning competitions and real-world applications. It is particularly effective for structured/tabular data.
- [LightGBM](https://lightgbm.readthedocs.io/): A fast and high-performance gradient boosting framework that is optimized for speed and memory efficiency. It works well for large datasets with many features.
- [CatBoost](https://catboost.ai/): A gradient boosting library specifically designed to handle categorical features without the need for extensive preprocessing. CatBoost is efficient and performs well in both classification and regression tasks.

### Neural Networks and Deep Learning Integration
- [TensorFlow](https://www.tensorflow.org/): An open-source deep learning framework widely used for neural network modeling, machine learning, and artificial intelligence applications. It supports both high-level and low-level APIs for building, training, and deploying machine learning models.
- [Keras](https://keras.io/): A high-level neural networks API, built on top of TensorFlow, that simplifies the process of building and training neural networks. Keras is user-friendly and modular, making it ideal for rapid experimentation.
- [PyTorch](https://pytorch.org/): A popular deep learning library that offers flexibility and control over neural network building and training. PyTorch is favored in academic research and supports dynamic computation graphs.

### Model Explainability and Interpretability
- [SHAP](https://shap.readthedocs.io/): SHapley Additive exPlanations (SHAP) is a tool that explains the output of machine learning models. It provides consistent, game-theory-based explanations for why a model made a particular prediction.
- [LIME](https://github.com/marcotcr/lime): Local Interpretable Model-agnostic Explanations (LIME) is a library that explains the predictions of any machine learning classifier or regressor by approximating it locally with interpretable models.

### Hyperparameter Tuning and Optimization
- [Optuna](https://optuna.org/): A hyperparameter optimization framework that allows for efficient sampling and tuning of machine learning models. Optuna supports both simple and complex search spaces and integrates with many machine learning frameworks.
- [Hyperopt](http://hyperopt.github.io/hyperopt/): A Python library for optimizing machine learning hyperparameters through random search, grid search, and Bayesian optimization.
- [Ray Tune](https://docs.ray.io/en/latest/tune/index.html): A scalable library for hyperparameter tuning that supports distributed model training and integrates with popular libraries like TensorFlow, PyTorch, and XGBoost.

### Automated Machine Learning (AutoML)
- [Auto-sklearn](https://automl.github.io/auto-sklearn/stable/): An AutoML library built on top of scikit-learn, automating model selection, hyperparameter optimization, and data preprocessing.
- [TPOT](https://epistasislab.github.io/tpot/): A genetic programming-based AutoML library that automates the selection of models and hyperparameters.
- [H2O.ai](https://www.h2o.ai/): A scalable open-source platform that includes H2O AutoML, providing automated model selection, tuning, and deployment.

### Model Deployment and Monitoring
- [MLflow](https://mlflow.org/): An open-source platform for managing the machine learning lifecycle. MLflow tracks experiments, packages code into reproducible runs, and manages model deployment.
- [DVC](https://dvc.org/): Data Version Control (DVC) is an open-source tool for managing datasets and machine learning models. It helps version data, manage pipelines, and reproduce results.

### Dimensionality Reduction
- [UMAP](https://umap-learn.readthedocs.io/): Uniform Manifold Approximation and Projection (UMAP) is a powerful dimensionality reduction technique for visualization and modeling of high-dimensional data.
- [t-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html): t-Distributed Stochastic Neighbor Embedding is a non-linear dimensionality reduction technique useful for visualizing high-dimensional data.
- [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html): Principal Component Analysis (PCA) is a standard linear technique used for reducing the dimensionality of datasets.

### Specialized Machine Learning Libraries
- [Hugging Face Transformers](https://huggingface.co/transformers/): A library for state-of-the-art natural language processing, allowing for the easy use of pretrained transformers like BERT, GPT, and T5.
- [FastAI](https://www.fast.ai/): A high-level library built on top of PyTorch, designed to simplify the implementation of deep learning models with minimal code.
- [OpenCV](https://opencv.org/): A powerful computer vision library that provides machine learning tools for image and video analysis.


## Text Analysis

- [NLTK](https://www.nltk.org/): Natural Language Toolkit for Python.
- [spaCy](https://spacy.io/): Industrial-strength Natural Language Processing.
- [TextBlob](https://textblob.readthedocs.io/en/dev/): Simplified text processing.
- [Gensim](https://radimrehurek.com/gensim/): Topic modeling and document similarity library.

## Speech Recognition

- [SpeechRecognition](https://pypi.org/project/SpeechRecognition/): Library for performing speech recognition.
- [DeepSpeech](https://github.com/mozilla/DeepSpeech): Open-source Speech-to-Text engine.
- [Wav2Vec](https://github.com/pytorch/fairseq/tree/main/examples/wav2vec): Pre-trained speech models for speech recognition tasks.

## Time Series Analysis

- [statsmodels](https://www.statsmodels.org/): A Python library for estimating and testing statistical models, including time series analysis with ARIMA, SARIMAX, and other advanced techniques.
- [Prophet](https://facebook.github.io/prophet/): Developed by Facebook, Prophet is a powerful tool for forecasting time series data, especially for data with daily observations and seasonal patterns.
- [tslearn](https://tslearn.readthedocs.io/en/stable/): A machine learning library specifically designed for time series data, offering clustering, classification, and regression tools.
- [GluonTS](https://ts.gluon.ai/): An open-source library built on Apache MXNet for probabilistic time series modeling, featuring state-of-the-art deep learning models for forecasting.
- [Darts](https://github.com/unit8co/darts): A user-friendly Python library that provides a wide range of models for time series forecasting, including ARIMA, exponential smoothing, and deep learning models like RNNs, N-BEATS, and TCNs.
- [sktime](https://www.sktime.org/): A unified framework for machine learning with time series data, covering time series forecasting, classification, and regression.
- [PyFlux](https://github.com/RJT1990/pyflux): A library for time series analysis using Bayesian methods. It includes models like ARIMA, GARCH, and state space models.
- [Kats](https://github.com/facebookresearch/Kats): A toolkit developed by Facebook for analyzing time series data, providing tools for forecasting, anomaly detection, and signal decomposition.
- [TSA (Time Series Analysis)](https://pypi.org/project/TSA/): A simple library for analyzing time series data with tools like smoothing, differencing, and statistical tests.
- [AutoTS](https://github.com/winedarksea/AutoTS): Automated time series forecasting in Python, with built-in support for a variety of statistical and machine learning-based forecasting models.
- [Pmdarima](https://www.alkaline-ml.com/pmdarima/): A library that provides statistical time series models, especially ARIMA, with tools for model selection and hyperparameter tuning.
- [PyCaret Time Series](https://pycaret.gitbook.io/docs/get-started/time-series-forecasting): Part of the PyCaret low-code machine learning library, providing easy-to-use time series forecasting models.
- [tsfresh](https://tsfresh.readthedocs.io/en/latest/): A package for automatic extraction of relevant features from time series data, used to support machine learning models.
- [tsfeatures](https://github.com/Nixtla/tsfeatures): Extracts features from time series for machine learning and forecasting tasks. Can be used for model selection or feature engineering.
- [pyts](https://pyts.readthedocs.io/en/stable/): A Python package for time series classification, offering a variety of tools to transform and classify time series data.
- [aeon](https://github.com/aeon-toolkit/aeon): An extension of sktime, providing advanced algorithms for time series forecasting, classification, and feature extraction.
- [PyWavelets](https://pywavelets.readthedocs.io/en/latest/): A comprehensive library for wavelet transforms, often used in time series feature extraction and denoising.
- [tsmoothie](https://github.com/cerlymarco/tsmoothie): A library to smooth time series data and perform uncertainty intervals.
- [River](https://riverml.xyz/): A machine learning library for real-time analysis of streaming time series data.
- [sktime-dl](https://github.com/sktime/sktime-dl): Deep learning extension for sktime, specifically for time series classification and regression using neural networks.
- [MatrixProfile](https://matrixprofile.org/): Efficient time series analysis techniques for finding patterns, motifs, and anomalies in time series data.
- [TensorFlow Probability](https://www.tensorflow.org/probability): Offers probabilistic models and tools for time series forecasting using TensorFlow.
- [Tsfel](https://github.com/fraunhoferportugal/tsfel): Time Series Feature Extraction Library focused on generating descriptive features from raw time series data.
- [Merlion](https://github.com/salesforce/Merlion): A machine learning library for time series forecasting and anomaly detection.


## Statistical Analysis

- [SciPy](https://scipy.org/): Python library used for scientific and technical computing. Includes modules for statistics, linear algebra, optimization, integration, and signal processing.
- [StatsModels](https://www.statsmodels.org/): Provides classes and functions for the estimation of statistical models, conducting tests, and performing data exploration.
- [Pingouin](https://pingouin-stats.org/): A simple yet comprehensive statistical package for Python with functionalities for correlation, regression, ANOVA, and more.
- [PyMC3](https://docs.pymc.io/): A probabilistic programming library focused on Bayesian statistical modeling and inference.
- [ArviZ](https://arviz-devs.github.io/arviz/): Visualization and diagnostic tools for Bayesian inference.
- [Lifelines](https://lifelines.readthedocs.io/en/latest/): A complete survival analysis library that offers statistical tools for estimating survival functions and hazards.
- [Pandas-Stubs](https://pandas-stubs.github.io/pandas-stubs/): Provides better type annotations for Pandas, assisting in statistical data analysis.
- [BayesianMethodsForHackers](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers): A library for learning Bayesian methods, based on Jupyter Notebooks.
- [rpy2](https://rpy2.github.io/): Interface to use R from Python, providing access to R’s extensive statistical libraries.
- [scikit-posthocs](https://github.com/maximtrp/scikit-posthocs): A library that provides statistical post-hoc tests to complement statistical analysis.
- [PyGAM](https://pygam.readthedocs.io/en/latest/): A package for Generalized Additive Models (GAMs), providing flexible regression models.
- [pandas](https://pandas.pydata.org/): Although more often used for data manipulation, Pandas provides various statistical methods such as groupby, aggregation, and descriptive statistics.
- [MLE-Toolkit](https://pypi.org/project/MLE-Toolkit/): Provides functions for Maximum Likelihood Estimation (MLE) and various tests for model diagnostics.
- [PyABC](https://pyabc.readthedocs.io/en/latest/): Python library for Approximate Bayesian Computation, useful for parameter estimation and model comparison.
- [linearmodels](https://bashtage.github.io/linearmodels/): A library focused on econometrics, providing tools for panel data models, instrumental variables, and more.
- [HDDM](http://ski.clps.brown.edu/hddm_docs/): Hierarchical Bayesian estimation of Drift Diffusion Models (DDM) used in cognitive science research.
- [SkiKit-Bio](https://scikit-bio.org/): Bioinformatics library, providing statistical tools for phylogenetic analysis and biological sequence data.
- [pyJanitor](https://pyjanitor.readthedocs.io/): An extension of Pandas with added functionality for data cleaning and preprocessing, used in preparation for statistical analysis.

### Specialized Libraries

- **Survival Analysis**:
  - [Lifelines](https://lifelines.readthedocs.io/en/latest/): Provides functions for Kaplan-Meier estimates, Cox regression models, and more.

- **Bayesian Inference**:
  - [PyMC3](https://docs.pymc.io/): Focuses on probabilistic programming using Bayesian methods.
  - [ArviZ](https://arviz-devs.github.io/arviz/): Provides diagnostic tools and visualizations for Bayesian inference models.

- **Econometrics**:
  - [linearmodels](https://bashtage.github.io/linearmodels/): Includes panel data models, instrumental variable regression, and other econometric tools.

- **Bioinformatics & Phylogenetics**:
  - [SkiKit-Bio](https://scikit-bio.org/): Provides bioinformatics tools, focusing on statistical analysis of biological data, such as sequence alignment and phylogenetic tree generation.


## Web Development

- [Django](https://www.djangoproject.com/): A high-level Python web framework that encourages rapid development and clean, pragmatic design. Includes ORM, admin interface, and authentication.
- [Flask](https://flask.palletsprojects.com/): A lightweight WSGI web application framework designed with simplicity in mind. Popular for smaller, more straightforward applications.
- [FastAPI](https://fastapi.tiangolo.com/): A modern, fast (high-performance) web framework for building APIs with Python, based on standard Python type hints. Known for its speed and ease of use.
- [Tornado](https://www.tornadoweb.org/): A Python web framework and asynchronous networking library. Known for its ability to handle large amounts of simultaneous connections, making it ideal for WebSockets and real-time services.
- [Pyramid](https://trypyramid.com/): A flexible, "start small, finish big" web framework that allows developers to scale from simple to complex applications.
- [Bottle](https://bottlepy.org/): A fast, simple, and lightweight WSGI micro web framework for Python. It's perfect for small web applications, and it has no dependencies outside of the Python standard library.
- [Sanic](https://sanic.dev/): An asynchronous web framework designed for fast HTTP responses, supporting asynchronous request handlers to boost speed for I/O-heavy applications.
- [Dash](https://dash.plotly.com/): A Python framework for building analytical web applications, primarily used for creating interactive dashboards.
- [Web2py](http://www.web2py.com/): A full-stack, scalable web framework designed for ease of use. Includes web-based interface for development and administration.
- [Starlette](https://www.starlette.io/): A lightweight ASGI framework/toolkit that is ideal for building asynchronous APIs and services. It's the foundation for FastAPI.
- [Hug](https://www.hug.rest/): A fast API framework for Python 3 with automatic documentation generation.
- [CherryPy](https://cherrypy.org/): A minimalist Python web framework that allows developers to build web applications similarly to writing Python programs. It’s easy to integrate with other Python frameworks and libraries.
- [Quart](https://pgjones.gitlab.io/quart/): An asyncio-based web framework inspired by Flask, providing complete support for asynchronous request handlers.
- [Falcon](https://falconframework.org/): A minimalist web framework designed for building fast and reliable large-scale RESTful APIs.
- [Masonite](https://docs.masoniteproject.com/): A modern and developer-friendly Python web framework, focused on simplicity and performance.
- [TurboGears](https://turbogears.org/): A full-stack web framework that combines the convenience of a microframework with the power of a full-stack solution.
- [Responder](https://python-responder.org/): A minimalist web framework built on Starlette and intended for building APIs with a modern and simple syntax.

### Supporting Tools & Libraries:

- **ORMs**:
  - [SQLAlchemy](https://www.sqlalchemy.org/): The Python SQL toolkit and Object-Relational Mapping (ORM) library.
  - [Tortoise ORM](https://tortoise-orm.readthedocs.io/): An easy-to-use asyncio ORM inspired by Django’s ORM.
  - [Peewee](http://docs.peewee-orm.com/): A small, expressive ORM that provides a simple and clean interface for working with databases.

- **Authentication & Security**:
  - [Flask-Security](https://pythonhosted.org/Flask-Security/): Security layer for Flask-based applications, including login, registration, and role-based access control.
  - [Django Allauth](https://django-allauth.readthedocs.io/): Integrated set of Django applications addressing authentication, registration, account management, and third-party account authentication.
  - [Authlib](https://authlib.org/): Powerful authentication library for OAuth, OpenID Connect, and JWT.

- **GraphQL**:
  - [Graphene](https://graphene-python.org/): A library for building GraphQL APIs in Python.
  - [Ariadne](https://ariadnegraphql.org/): A Python library for implementing GraphQL servers.

- **WebSockets**:
  - [Channels](https://channels.readthedocs.io/): Extensions for Django that add support for handling WebSockets and other real-time protocols.
  - [SocketIO](https://python-socketio.readthedocs.io/): A Python implementation of the Socket.IO protocol for WebSockets communication.

### Specialized Frameworks:

- **API-First Development**:
  - [Connexion](https://connexion.readthedocs.io/): A framework for creating REST APIs and building OpenAPI-compliant web applications.

- **Asynchronous Web Frameworks**:
  - [Sanic](https://sanic.dev/): Supports asynchronous request handlers and is known for its speed.
  - [Quart](https://pgjones.gitlab.io/quart/): Like Flask but supports async/await.

### Full-Stack Solutions:

- **Server-Side Rendering**:
  - [Masonite](https://docs.masoniteproject.com/): Full-stack web framework with support for server-side rendering and developer-friendly tools.


### Desktop Applications

- [PyQt](https://riverbankcomputing.com/software/pyqt/intro): Python bindings for Qt, used to create cross-platform applications with a native look and feel.
- [PySide (Qt for Python)](https://doc.qt.io/qtforpython/): Official Python bindings for Qt, similar to PyQt but licensed under LGPL.
- [Kivy](https://kivy.org/): Open-source Python library for developing multitouch applications, suitable for desktop and mobile platforms.
- [Tkinter](https://wiki.python.org/moin/TkInter): Standard Python interface to the Tk GUI toolkit for creating simple, lightweight GUIs.
- [wxPython](https://wxpython.org/): Python bindings for the wxWidgets C++ library, allowing the creation of native applications on Windows, macOS, and Linux.
- [PyGTK](https://pygtk.org/): Python bindings for GTK, used to create graphical user interfaces for GNOME and other environments.
- [BeeWare](https://beeware.org/): A collection of tools and libraries to help you write cross-platform native applications using Python.
  - **Toga**: GUI toolkit for Python that allows for native applications across multiple platforms.
- [Dear PyGui](https://github.com/hoffstadt/DearPyGui): A simple-to-use, GPU-accelerated Python GUI framework that provides a lot of flexibility for developing applications.
- [Flexx](https://flexx.app/): Pure Python toolkit for creating desktop applications and web apps with a modern, flexible design.
- [PyForms](https://pyforms.readthedocs.io/en/v4.1/): A Python framework to develop GUI applications easily using a simple API.
- [Enaml](https://nucleic.github.io/enaml/): A declarative framework for building desktop applications in Python, similar to QML for Qt.
- [EasyGUI](https://easygui.readthedocs.io/en/latest/): A module for very simple, basic GUIs, making it easy to create message boxes, dialogs, and input boxes.
- [Gooey](https://github.com/chriskiehl/Gooey): Turns command-line programs into a full-fledged GUI application with minimal code changes.
- [PySimpleGUI](https://pysimplegui.readthedocs.io/): Simple-to-use Python GUI framework that wraps around other GUI frameworks like Tkinter, Qt, or WxPython.
- [Flet](https://flet.dev/): A high-level Python framework that simplifies the development of cross-platform desktop apps, with a modern UI.
- [Electron](https://www.electronjs.org/) with Python: Use Electron, commonly used with JavaScript, along with Python for creating cross-platform desktop apps.
- [pywebview](https://pywebview.flowrl.com/): Lightweight cross-platform wrapper for web-based content in Python desktop apps, providing a webview window.
- [Shoebot](https://shoebot.readthedocs.io/en/latest/): A Python-based simple, cross-platform framework for creating generative art and animations.
- [PyGObject](https://pygobject.readthedocs.io/en/latest/): Python bindings for GObject-based libraries such as GTK, suitable for creating complex GUI applications.
- [Remi](https://github.com/dddomodossola/remi): Python library to create complex and fully featured web applications with GUIs in just a few lines of code, which can also run locally in the browser.


### Mobile Applications

- [BeeWare](https://beeware.org/): Native mobile applications in Python.
- [Kivy](https://kivy.org/#home): Open-source Python library for rapid development of mobile applications.

## Animation

- [Manim](https://www.manim.community/): Engine for creating explanatory math videos.
- [Blender](https://www.blender.org/): Free and open-source 3D creation suite.

## Command Line Tools

- [Click](https://click.palletsprojects.com/): Python package for creating beautiful command line interfaces.
- [argparse](https://docs.python.org/3/library/argparse.html): Built-in Python library for command-line parsing.

## 3D CAD

- [FreeCAD](https://www.freecadweb.org/): Open-source parametric 3D modeler.
- [Blender](https://www.blender.org/): 3D modeling and rendering.


## GIS (Geographic Information Systems)

- [GeoPandas](https://geopandas.org/): Extends Pandas to support spatial data operations and analysis.
- [Fiona](https://fiona.readthedocs.io/): A Python library for reading and writing vector data formats like shapefiles.
- [Shapely](https://shapely.readthedocs.io/): Manipulates and analyzes planar geometric objects such as points, lines, and polygons.
- [PyProj](https://pyproj4.github.io/pyproj/): Python interface to PROJ (a library for cartographic projections).
- [Rasterio](https://rasterio.readthedocs.io/): Reads and writes geospatial raster data using formats such as GeoTIFF.
- [GDAL](https://gdal.org/): Geospatial Data Abstraction Library for working with raster and vector geospatial data formats.
- [Cartopy](https://scitools.org.uk/cartopy/docs/latest/): A library for cartographic projections and geospatial visualizations.
- [PySAL](https://pysal.org/): Python Spatial Analysis Library, used for spatial data analysis.
- [OsmPy](https://osmpy.readthedocs.io/): Tools for processing OpenStreetMap (OSM) data with Python.
- [WhiteboxTools](https://www.whiteboxgeo.com/whitebox-tools/): Advanced geospatial analysis platform.
- [MapClassify](https://pysal.org/mapclassify/): Part of PySAL for choropleth map classification schemes.

## Remote Sensing

- [SentinelHub-Py](https://sentinelhub-py.readthedocs.io/): Python package for downloading and processing Sentinel satellite images.
- [EarthPy](https://earthpy.readthedocs.io/): A Python package that makes it easier to plot and analyze spatial raster and vector data.
- [SentiPy](https://github.com/cosinekitty/sentipy): A Python interface for accessing and processing Sentinel satellite data.
- [Py6S](https://py6s.readthedocs.io/): A Python interface to the 6S Radiative Transfer Model for atmospheric correction.
- [Rasterio](https://rasterio.readthedocs.io/): Handles geospatial raster data, ideal for remote sensing.
- [Google Earth Engine Python API](https://developers.google.com/earth-engine/guides/python_install): Python API for Google Earth Engine to work with satellite imagery and geospatial datasets.
- [eoreader](https://eoreader.readthedocs.io/en/latest/): Python library for reading and processing EO data, including various satellite imagery sources.
- [EO-learn](https://eo-learn.readthedocs.io/en/latest/): A Python package for processing satellite images in a plug-and-play manner, ideal for Earth observation tasks.
- [Pyresample](https://pyresample.readthedocs.io/en/latest/): Python package for geolocating and resampling Earth observation satellite data.
- [Snappy](http://step.esa.int/main/toolboxes/snap/): Python API for ESA's SNAP, used for Sentinel-1, Sentinel-2, and Sentinel-3 data processing.
- [RSGISLib](https://rsgislib.org/): Remote Sensing and GIS software library for raster processing, classification, and analysis.
- [PyRate](https://github.com/GeoscienceAustralia/PyRate): InSAR time series analysis for studying ground motion and surface deformation.
- [scikit-image](https://scikit-image.org/): A collection of algorithms for image processing, commonly used in remote sensing tasks.
- [Orfeo Toolbox](https://www.orfeo-toolbox.org/): A powerful library for processing remote sensing images.


## Natural Language Processing (NLP)

- [spaCy](https://spacy.io/): Industrial-strength natural language processing library.
- [NLTK](https://www.nltk.org/): Natural Language Toolkit for working with human language data.
- [Transformers](https://huggingface.co/transformers/): Library for working with transformer-based models like GPT and BERT.
- [Gensim](https://radimrehurek.com/gensim/): Topic modeling and document similarity.

## Bioinformatics

- [Biopython](https://biopython.org/): Tools for biological computation.
- [scikit-bio](https://scikit-bio.org/): Bioinformatics library for statistical analysis of biological data.
- [PyMOL](https://pymol.org/2/): Molecular visualization system.
- [BioPandas](https://rasbt.github.io/biopandas/): Extends Pandas for working with biomolecular data.


## Robotics

- [PyRobot](https://pyrobot.org/): Robotics and AI integration library.
- [ROS (Robot Operating System)](https://www.ros.org/): Python interface for working with robotic systems.
- [VREP-Py](https://www.coppeliarobotics.com/helpFiles/en/b0PythonBindings.htm): Python bindings for robot simulations in the V-REP (CoppeliaSim) environment.

## Game Development

- [Pygame](https://www.pygame.org/): Popular game development library.
- [Panda3D](https://www.panda3d.org/): Framework for real-time 3D games, simulations, and visualizations.
- [Arcade](https://arcade.academy/): Easy-to-learn Python library for 2D games.

## Computer Vision

- [OpenCV](https://opencv.org/): Most popular computer vision library for Python.
- [PyTorch Vision](https://pytorch.org/vision/stable/index.html): Image classification and object detection models.
- [TensorFlow Object Detection API](https://tensorflow-object-detection-api-tutorial.readthedocs.io/): Tools for training object detection models.

## Cryptography and Security

- [PyCryptodome](https://www.pycryptodome.org/): Self-contained Python package of low-level cryptographic primitives.
- [Cryptography](https://cryptography.io/): Popular Python library for cryptographic recipes.
- [Paramiko](https://www.paramiko.org/): SSH protocol for remote access to servers and systems.

## Audio Processing

- [Librosa](https://librosa.org/): Python library for music and audio analysis.
- [PyDub](https://github.com/jiaaro/pydub): Simple and easy high-level API for audio processing.
- [SoundFile](https://pysoundfile.readthedocs.io/): Library for reading and writing sound files.

## Financial and Quantitative Analysis

- [QuantLib](https://www.quantlib.org/): Library for modeling, trading, and risk management in real-life.
- [TA-Lib](https://mrjbq7.github.io/ta-lib/): Technical analysis library for financial market data.
- [zipline](http://www.zipline.io/): Pythonic algorithmic trading library.

## Physics Simulations

- [PyBullet](https://pybullet.org/): Physics simulation for robotics and machine learning.
- [FEniCS](https://fenicsproject.org/): Automated solution of differential equations via finite element methods.
- [SimPy](https://simpy.readthedocs.io/): Process-based discrete-event simulation framework.

## Mathematics and Symbolic Computation

- [SymPy](https://www.sympy.org/en/index.html): Python library for symbolic mathematics.
- [SciPy](https://www.scipy.org/): Python library used for scientific and technical computing.
- [NumPy](https://numpy.org/): Core library for numerical and matrix computations.

## Automation

- [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/): Python library for web scraping.

## DevOps and Monitoring

- [Ansible](https://www.ansible.com/): Automation framework for provisioning and deployment.
- [Fabric](https://www.fabfile.org/): Python library for streamlining the use of SSH for application deployment.
- [NagiosPy](https://github.com/NagiosEnterprises/nagiospy): Monitoring software integration with Python.

## Quantum Computing

- [Qiskit](https://qiskit.org/): A Python-based quantum computing framework developed by IBM, designed for creating, simulating, and running quantum algorithms on quantum computers. It provides tools for quantum circuit creation, simulation, and optimization.

- [Cirq](https://quantumai.google/cirq): An open-source Python library developed by Google for designing, simulating, and executing quantum circuits. It is specifically designed for Noisy Intermediate-Scale Quantum (NISQ) algorithms, enabling developers to build and experiment with quantum systems.

- [ProjectQ](https://projectq.ch/): A Python library for quantum computing that allows users to compile and simulate quantum programs and execute them on various quantum devices. It includes a modular architecture for building custom quantum gates and circuits.

- [QuTiP](https://qutip.org/): Quantum Toolbox in Python (QuTiP) is used for simulating the dynamics of quantum systems. It is widely used in quantum optics, many-body physics, and quantum information processing.

- [PennyLane](https://pennylane.ai/): A library focused on quantum machine learning, PennyLane bridges quantum computing and machine learning frameworks, enabling hybrid quantum-classical computations. It integrates well with TensorFlow and PyTorch.

- [Pyquil](https://pyquil-docs.rigetti.com/en/stable/): Developed by Rigetti Computing, PyQuil is a Python library for writing quantum programs, simulating quantum circuits, and running experiments on real quantum hardware through Rigetti’s Forest platform.

- [Strawberry Fields](https://strawberryfields.ai/): A Python library for quantum computing built specifically for continuous-variable (CV) quantum systems, providing tools for photonic quantum computing.

- [D-Wave Ocean SDK](https://docs.ocean.dwavesys.com/en/stable/): A set of Python libraries for programming quantum annealers from D-Wave. It includes tools for defining, solving, and optimizing problems on D-Wave’s quantum computers.

# Contribution Guidelines

Thank you for considering contributing to **Awesome Python Tools**! This repository is dedicated to curating the best Python libraries and frameworks across various fields. We welcome contributions that help make this list more comprehensive and up-to-date. Below are the guidelines for contributing to the repository.

## How to Contribute

### 1. Fork the Repository
Fork this repository by clicking the “Fork” button at the top of the page. This will create a copy of the repository under your GitHub account.

### 2. Clone Your Fork
Clone your forked repository to your local machine:
```bash
git clone https://github.com/your-username/awesome-python-tools.git
cd awesome-python-tools
```

3. Create a New Branch

Create a new branch for your contribution. It’s a good practice to name your branch based on the specific section or tool you are adding or modifying:

```bash
git checkout -b add-tool-to-section
```

4. Add Your Contribution

Add a new library, framework, or tool to the appropriate section. Make sure to:

   - Provide a brief, clear description of the tool.
   - Add a link to the tool’s official website or documentation.
   - Ensure your addition is placed in the correct category (e.g., Machine Learning, Data Science, etc.).

For example:

`- [SomeLibrary](https://somelibrary.org/): A Python library for X that helps with Y.`


5. Update the Table of Contents

If your contribution adds a new section or significantly alters the content, make sure to update the Table of Contents in the README so that others can easily find it.

6. Commit and Push Your Changes

Once you’re happy with your contribution, commit your changes with a clear and descriptive message:

```bash
git add .
git commit -m "Added [Library/Framework Name] to [Section Name]"
```

Then push your branch to your forked repository:

```bash
git push origin add-tool-to-section
```


7. **Open a Pull Request**

Submit a pull request (PR) back to this repository. Provide a description of your changes, including why the tool you added is important or how it enhances the repository. We will review your contribution and merge it if it meets the guidelines.

## Contribution Guidelines

   - New Libraries or Frameworks: Contributions should only include well-established or highly useful libraries and frameworks. Please avoid adding tools that are not actively maintained or have very limited use cases.
   - Categories: If you feel the existing categories are not sufficient, you can propose a new section by creating an issue or adding it directly in your pull request.
   - Documentation: If you’re improving the documentation (e.g., fixing typos, improving explanations), please make your changes as clear and concise as possible.
   - Completeness: Ensure your contribution includes a name, link to the official website or documentation, and a brief but clear description.
   - No Code Contributions: This repository focuses solely on listing Python libraries and frameworks. If you have code-related contributions, please refer to other repositories that focus on code samples or open-source development.
   - Respect: Be respectful to maintainers and other contributors. We value collaborative and constructive feedback.

**Suggesting a New Category**

If you want to suggest an entirely new section or area of focus, feel free to open an issue and discuss your ideas with the maintainers.



## LICENSE

This repository is licensed under the [MIT License](./LICENSE)



