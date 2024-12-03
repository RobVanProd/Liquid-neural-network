from setuptools import setup, find_packages

setup(
    name="liquid_neural_network",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.1",
        "numpy>=1.24.3",
        "matplotlib>=3.7.1",
        "scikit-learn>=1.2.2",
        "tensorboard",
        "seaborn",
        "optuna"
    ],
    python_requires=">=3.8",
)
