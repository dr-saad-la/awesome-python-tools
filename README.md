# Awesome Python Tools

Awesome Python Tools is a repository about different Python Tools used in various fields ranging from programming, statistical analysis, data science, machine learning, deep learning, and much more.

**Author and Maintainer**: Dr. Saad Laouadi

## Table of Content


1. [Programming](#programming)
   - 1.1. [Software Development](#software-development)
   - 1.2. [Testing](#testing)
   - 1.3. [Desktop Applications](#desktop-applications)
   - 1.4. [Mobile Applications](#mobile-applications)
2. [Data Science](#data-science)
   - 2.1. [Data Manipulations](#data-manipulations)
   - 2.2. [Data Visualizations](#data-visualizations)
3. [Machine Learning](#machine-learning)
4. [Deep Learning](#deep-learning)
5. [Text Analysis](#text-analysis)
6. [Speech Recognition](#speech-recognition)
7. [Time Series Analysis](#time-series-analysis)
8. [Image Processing](#image-processing)
9. [Statistical Analysis](#statistical-analysis)
10. [Web Development](#web-development)
11. [Animation](#animation)
12. [Command Line Tools](#command-line-tools)
13. [3D CAD](#3d-cad)

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

#### Big Data and Out-of-Core Manipulation
- [Vaex](https://vaex.io/): Out-of-core DataFrame library enabling fast manipulation of big data that doesn’t fit into memory.
- [BlazingSQL](https://blazingsql.com/): GPU-accelerated SQL engine for processing large datasets, part of the RAPIDS ecosystem.
- [DataTables](https://pypi.org/project/py-datatables/): Provides memory-efficient data handling for larger-than-memory datasets.
- [TensorFlow Data API](https://www.tensorflow.org/api_docs/python/tf/data): Efficiently loads and processes large-scale datasets for machine learning and deep learning workflows.

#### Specialized Data Handling
- [Xarray](http://xarray.pydata.org/): Best for multi-dimensional labeled arrays, used in scientific computing like climate data analysis.
- [GeoPandas](https://geopandas.org/): Specialized for geospatial data manipulation, supporting shapefiles, geoJSON, and spatial operations.
- [Awkward Array](https://awkward-array.org/): Designed for handling complex and nested data structures, useful for JSON-like data and scientific research.
- [SQLAlchemy](https://www.sqlalchemy.org/): SQL toolkit and ORM for handling complex relational database queries using Python code.

#### Efficient Storage and I/O
- [PyArrow](https://arrow.apache.org/): Enables efficient columnar storage and fast I/O for large-scale data manipulation, commonly used with Parquet and Feather file formats.
- [Vaex](https://vaex.io/): Optimized for reading large datasets from file formats like HDF5 and Arrow without loading them fully into memory.



### Data Visualizations

- [Matplotlib](https://matplotlib.org/): 2D plotting library for Python.
- [Seaborn](https://seaborn.pydata.org/): Statistical data visualization based on Matplotlib.
- [Plotly](https://plotly.com/python/): Interactive data visualization library.
- [Bokeh](https://bokeh.org/): Interactive visualization library that targets modern web browsers.
- [Altair](https://altair-viz.github.io/): Declarative statistical visualization library.

## Machine Learning

- [Scikit-learn](https://scikit-learn.org/): Simple and efficient tools for data mining and data analysis.
- [XGBoost](https://xgboost.readthedocs.io/en/latest/): Gradient boosting framework.
- [LightGBM](https://lightgbm.readthedocs.io/): A fast, distributed, high-performance gradient boosting framework.
- [CatBoost](https://catboost.ai/): Gradient boosting library that handles categorical features automatically.

## Deep Learning

- [TensorFlow](https://www.tensorflow.org/): End-to-end open-source platform for machine learning.
- [PyTorch](https://pytorch.org/): Deep learning framework for research and production.
- [Keras](https://keras.io/): High-level neural networks API.
- [MXNet](https://mxnet.apache.org/): Scalable deep learning framework.

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

- [statsmodels](https://www.statsmodels.org/): Statistical models, hypothesis tests, and data exploration.
- [Prophet](https://facebook.github.io/prophet/): Forecasting tool by Facebook for time series data.
- [tslearn](https://tslearn.readthedocs.io/en/stable/): Machine learning library for time series.

## Image Processing

- [Pillow](https://python-pillow.org/): The Python Imaging Library.
- [OpenCV](https://opencv.org/): Open-source computer vision library.
- [scikit-image](https://scikit-image.org/): Image processing library for Python.
- [tifffile](https://pypi.org/project/tifffile/): Read and write TIFF files.

## Statistical Analysis

- [SciPy](https://scipy.org/): Python library used for scientific and technical computing.
- [StatsModels](https://www.statsmodels.org/): Provides classes and functions for the estimation of statistical models.
- [Pingouin](https://pingouin-stats.org/): Statistical package for Python.

## Web Development

- [Django](https://www.djangoproject.com/): High-level Python web framework.
- [Flask](https://flask.palletsprojects.com/): Lightweight WSGI web application framework.
- [FastAPI](https://fastapi.tiangolo.com/): Modern, fast web framework for building APIs with Python.



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

## LICENSE

This repository is licensed under the [MIT License](./LICENSE)



