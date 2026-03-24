from setuptools import setup, find_packages

setup(
    name="edatoolkit", 
    version="0.1.0",
    author="Elvin Aliyev",
    description="A professional OOP-based EDA toolkit with statistical tests",
    long_description=open("README.md",encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "plotly",
        "scipy",
        "statsmodels",
        "nbformat",
        "kaleido"
    ],
    python_requires=">=3.7",
)