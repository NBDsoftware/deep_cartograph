from setuptools import setup, find_packages

setup(name='deep_cartograph', 
    version='0.1.0', 
    packages=find_packages(),
    python_requires=">=3.8,<=3.10",
    install_requires=[
        'numpy',
        'pytorch<2.3',
        'pandas',
        "pydantic",
        'lightning',
        "mdanalysis",
        'plumed==2.9.0',
        'seaborn',
        'matplotlib',
        'scikit-learn>=1.3',
        "scipy",
        'jupyter',
        "ipython",
        "ipykernel",
    ],
    extras_require={
        "pip_only": [
            "kdepy",
            "diptest",
            "mlcolvar"
        ]
    },
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