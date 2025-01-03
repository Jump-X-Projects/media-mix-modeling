from setuptools import setup, find_packages

setup(
    name="media-mix-modeling",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "pandas",
        "numpy",
        "scikit-learn",
        "lightgbm",
        "xgboost",
        "pytest",
        "httpx",
        "python-multipart",
    ],
) 