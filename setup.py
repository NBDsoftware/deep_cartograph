from setuptools import setup, find_packages

setup(name='mlcolvar_utils', 
    version='0.1', 
    packages=find_packages(include=['mlcolvar_utils']),
    install_requires=[
        'numpy',
        'pandas',
        'pytorch',
        'pydantic<2',
        'lightning',
        'mdanalysis',
        'scikit-learn>=1.3',
        'seaborn',
        'scipy',
        'matplotlib',
        'mlcolvar'
    ])
