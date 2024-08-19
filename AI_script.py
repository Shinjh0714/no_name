# neutral pdb files directory = neu_1, pathogenic pdb files directory = highly_pathogen
from Bio.PDB import PDBParser
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# function to extract C-alpha atomic coordinates from the overall PDB file
def extract_coordinates(pdb_file):
    parser = PDBParser()
    structure = parser.get_structure('protein', pdb_file)
    coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    coords.append(residue["CA"].get_coord())
    return np.array(coords).flatten()

# Import pdb files
neu_files = [os.path.join("neu_1", f) for f in os.listdir("neu_1") if f.endswith('.pdb')]
pathogen_files = [os.path.join("highly_pathogen", f) for f in os.listdir("highly_pathogen") if f.endswith('.pdb')]

#coordinate extraction
neu_coords = [extract_coordinates(f) for f in neu_files]
pathogen_coords = [extract_coordinates(f) for f in pathogen_files]
all_coords = neu_coords + pathogen_coords
labels = ['neutral'] * len(neu_coords) + ['pathogenic'] * len(pathogen_coords)

# Since the atom has different lengths, apply zero padding
max_length = max(len(coords) for coords in all_coords) # Longest of all coordinates
padded_coords = []
for coords in all_coords:
    padding_length = max_length - len(coords)
    padded_coords.append(np.pad(coords, (0, padding_length), 'constant'))

# Data normalization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
normalized_coords = scaler.fit_transform(padded_coords)

#PCA perform
pca = PCA(n_components=4)
principal_components = pca.fit_transform(padded_coords)
principalDf = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2', 'PC3', 'PC4'])
principalDf['Class'] = labels
principalDf.to_csv('pca_results.csv', index=False) # PCA data save

# Draw a PCA plot through a combination of all the main ingredients
import itertools
combinations = list(itertools.combinations(['PC1', 'PC2', 'PC3', 'PC4'], 2))
for comb in combinations:
    plt.figure(figsize=(10,6))
    for label, color in zip(['neutral', 'pathogenic'], ['blue','red']):
        mask = principalDf['Class']==label
        plt.scatter(principalDf[mask][comb[0]], principalDf[mask][comb[1]], label=label, c=color, alpha=0.5)
    plt.xlabel(comb[0])
    plt.ylabel(comb[1])
    plt.legend()
    plt.title(f'PCA of Protein structures using {comb[0]} and {comb[1]}')
    plt.show()

# Perform abnormality detection and draw ROC curves (one - class SVM model)
from sklearn import svm
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelBinarizer

data = pd.read_csv("pca_results.csv")
lb = LabelBinarizer()
data['binary_class'] = lb.fit_transform(data['Class'])
#need a label reversal
data['binary_class'] = [-1 if label == 'neutral' else 1 for label in data['Class']]

neutral_data = data[data['Class'] == 'neutral']
combinations = [('PC1', 'PC2'), ('PC1', 'PC3'), ('PC1', 'PC4'), 
                ('PC2', 'PC3'), ('PC2', 'PC4'), ('PC3', 'PC4')]
plt.figure(figsize=(10, 10))
for comb in combinations:
    clf= svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    clf.fit(neutral_data[list(comb)])
    scores = -clf.decision_function(data[list(comb)])
    fpr, tpr, _ = roc_curve(data['binary_class'], scores)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{comb[0]} & {comb[1]} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for PC Combinations')
plt.legend(loc="lower right")
plt.show()

# Draw average ROC curves with overfitting protection and bootstrapping
## Function for bootstrapping
def bootstrap_sample(data, n=1):
    return [data.sample(n=len(data), replace=True) for _ in range(n)]

clf = svm.OneClassSVM(nu=0.5, kernel="rbf", gamma=0.1)
plt.figure(figsize=(10, 8))
N_ITER = 100  # Number of bootstrapping iterations

for comb in combinations:
    mean_fpr = np.linspace(0,1,100)
    tprs=[]
    for sample in bootstrap_sample(data, N_ITER):
        neutral_sample = sample[sample['Class']=='neutral']
        clf.fit(neutral_sample[list(comb)])
        fpr, tpr, _ = roc_curve(sample['binary_class'], scores)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
    mean_tpr = np.mean(tprs, axis=0)
    roc_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, lw=2, label=f'{comb[0]} & {comb[1]} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Average ROC Curve from Bootstrapping for PC Combinations')
plt.legend(loc="lower right")
plt.show()

######## Finaly abnormality detection ROC curve
data = pd.read_csv('pca_results.csv')
lb = LabelBinarizer(neg_label=-1)
data['binary_class'] = lb.fit_transform(data['Class'])

# color setting
color_map = {
    ('PC1', 'PC2'): '#0015FF',
    ('PC1', 'PC3'): '#FF00A1',
    ('PC1', 'PC4'): '#6BC800',
    ('PC2', 'PC3'): '#8400FF',
    ('PC2', 'PC4'): '#00BEB2',
    ('PC3', 'PC4'): '#FF7300'
}

combinations = list(color_map.keys())

# Bootstrapping function
def bootstrap_sample(data, n=1):
    return [data.sample(n=len(data), replace=True) for _ in range(n)]

# Draw ROC Curve
plt.figure(figsize=(10, 10))
N_ITER = 100
auc_scores = {}

for comb in combinations:
    mean_fpr = np.linspace(0,1,100)
    tprs=[]
    aucs = []
    for sample in bootstrap_sample(data, N_ITER):
        neutral_sample = sample[sample['Class']=='neutral']
        clf = svm.OneClassSVM(nu=0.5, kernel="rbf", gamma=0.1)
        clf.fit(neutral_sample[list(comb)])
        scores = -clf.decision_function(data[list(comb)])
        fpr, tpr, _ = roc_curve(data['binary_class'], scores)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        aucs.append(auc(fpr, tpr))
    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    auc_scores[comb] = mean_auc
    plt.plot(mean_fpr, mean_tpr, color=color_map[comb], label=f'{comb[0]} & {comb[1]} (AUC = {mean_auc:.2f} $\pm$ {std_auc:.2f})')
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color=color_map[comb], alpha=0.2)

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Average ROC Curve from Bootstrapping for PC Combinations')

# Change the legend order and display the plot again
order = np.argsort([-auc_scores[tuple(label.split(' (')[0].split(' & '))] if " & " in label else (-np.inf,) for label in labels])
ordered_handles = [handles[idx] for idx in order]
ordered_labels = [labels[idx] for idx in order]

# Don't add random Guess if it's already in Legend
if 'Random Guess' not in ordered_labels:
    ordered_handles.append(plt.Line2D([0], [0], color='navy', lw=2, linestyle='--'))
    ordered_labels.append('Random Guess')

plt.legend(ordered_handles, ordered_labels, loc="lower right")

plt.show()


#unsupervised learning perform
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import pandas as pd

data = pd.read_csv('pca_results.csv')

combinations = [('PC1', 'PC2'), ('PC1', 'PC3'), ('PC1', 'PC4'), 
                ('PC2', 'PC3'), ('PC2', 'PC4'), ('PC3', 'PC4')]

param_grid = {'n_neighbors': [3, 5, 7, 9, 11], 
              'weights': ['uniform', 'distance'], 
              'metric': ['euclidean', 'manhattan']}

N_ITER = 1000

results = []

for comb in combinations:
    print(f"Processing for combination: {comb}")
    
    X = data[list(comb)].values
    y = (data['Class'] == 'neutral').astype(int).values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    
    best_knn = grid_search.best_estimator_
    
    accuracies = []
    f1_scores = []
    recalls = []
    precisions = []
    
    for _ in range(N_ITER):
        sample_X, sample_y = resample(X_train, y_train)
        best_knn.fit(sample_X, sample_y)
        y_pred = best_knn.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))
        recalls.append(recall_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred))
        
    results.append({
        'Combination': ' & '.join(comb),
        'Accuracy': f"{np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}",
        'F1 Score': f"{np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}",
        'Recall': f"{np.mean(recalls):.4f} ± {np.std(recalls):.4f}",
        'Precision': f"{np.mean(precisions):.4f} ± {np.std(precisions):.4f}"
    })

results_df = pd.DataFrame(results)
results_df.to_csv('knn_performance_results.csv', index=False)
