from setuptools import setup, find_packages

setup(name='deep_cartograph', 
    version='0.1', 
    packages=find_packages(),
    install_requires=[
        'numpy',
        'dask',
        'pandas',
        'pydantic',
        'torch',
        'lightning',
        'mdanalysis',
        'seaborn',
        'matplotlib',
        'scikit-learn>=1.3',
        'KDEpy',
        'diptest',
        'plumed',
        'jupyter'   
    ],
    entry_points={
        "console_scripts": [
            "deep_carto=deep_cartograph.run:main",
            "compute_features=deep_cartograph.tools.compute_features.compute_features:main",
            "filter_features=deep_cartograph.tools.filter_features.filter_features:main",
            "train_colvars=deep_cartograph.tools.train_colvars.train_colvars:main",
            "analyze_geometry=deep_cartograph.tools.analyze_geometry.analyze_geometry:main",
        ]
    },
    include_package_data=True)