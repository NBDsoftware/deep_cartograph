from deep_cartograph.tools.train_colvars.cv_calculator import PCA, TICA, AE, DeepTICA

cv_classes = {
    'pca': PCA,
    'ae': AE,
    'tica': TICA,
    'deep_tica': DeepTICA
}

cv_labels = {
    'pca': 'PCA',
    'ae': 'AE',
    'tica': 'TICA',
    'deep_tica': 'DeepTICA'
}