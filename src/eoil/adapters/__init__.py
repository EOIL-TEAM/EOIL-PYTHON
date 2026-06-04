"""EOIL framework adapters (scipy, sklearn, pytorch, pandas).

All adapters are optional — import the one you need directly::

    from eoil.adapters.scipy import minimize
    from eoil.adapters.sklearn import EOILSearchCV
    from eoil.adapters.pytorch import EOILOptimizer
    from eoil.adapters.pandas import EOILPortfolioOptimizer

Each adapter raises ``ImportError`` with a clear message if its optional
third-party dependency (scipy, scikit-learn, torch, pandas) is not installed.
"""
