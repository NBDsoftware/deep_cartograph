from setuptools import setup, find_packages

setup(name='deep_cartograph', 
    version='0.1', 
    packages=find_packages(include=['deep_cartograph']),
    install_requires=[
        'numpy',
        'dask',
        'pandas',
        'torch',
        'lightning',
        'mdanalysis',
        'seaborn',
        'matplotlib',
        'scikit-learn>=1.3',
        'mlcolvar',
        'KDEpy',
        'diptest',
        'plumed'
    ])
