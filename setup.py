from setuptools import setup, find_packages
import os

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="edatoolkit", 
    version="0.2.0",
    author="Elvin Aliyev",
    description="A professional OOP-based EDA toolkit with statistical tests",
    long_description=long_description,
    long_description_content_type="text/markdown",
    
    # KÖK QOVLUĞU tap və yalnız onun daxilindəki paketləri götür
    packages=find_packages(where="."), 
    
    install_requires=[
        "pandas",
        "numpy",
        "seaborn",
        "matplotlib",
        "scipy"
    ],
    python_requires=">=3.7",
)
