# Lab 1 – Deep Learning avec PyTorch (MLP : régression et classification)

**Université Abdelmalek Essaadi – FST Tanger – Master MBD**
**Module** : Deep Learning – **Pr.** ELAACHAK Lotfi
**Étudiant(e)** : Ahadri Botaina
**Outils** : Kaggle Notebooks, PyTorch, skorch, scikit-learn, imbalanced-learn

## 1. Objectif

Se familiariser avec PyTorch pour résoudre une tâche de **régression** et une tâche de **classification multi-classe** avec des réseaux de neurones MLP, en utilisant `GridSearchCV` (via `skorch`) pour le choix des hyperparamètres et en comparant plusieurs techniques de régularisation.

## 2. Contenu du dépôt

| Fichier | Description |
|---|---|
| `part1-regression-nyse.ipynb` | Partie 1 : régression (NYSE) |
| `part2-classification-maintenance.ipynb` | Partie 2 : classification (Predictive Maintenance) |

Les graphiques (EDA, courbes Loss / Epochs, matrices de confusion, etc.) sont visibles directement dans les sorties des cellules de chaque notebook.

---

## 3. Partie 1 : Régression (NYSE)

**Dataset** : [New York Stock Exchange](https://www.kaggle.com/datasets/dgawlik/nyse), fichier `prices-split-adjusted.csv` (851 264 lignes, 7 colonnes, aucune valeur manquante).
**Problème** : prédire le prix de clôture `close` à partir de `open`, `low`, `high` et `volume`.

### 3.1 Analyse exploratoire (EDA)

- Le dataset contient 501 symboles boursiers, de 2010 à 2016, sans valeur manquante.
- Les prix `open`, `close`, `low`, `high` sont **corrélés à 1** : ils sont quasi redondants. Le volume est faiblement corrélé aux prix (-0.13).
- Les distributions sont très asymétriques (la plupart des actions valent moins de 100 $, quelques-unes dépassent 1 500 $). Ces valeurs élevées sont des actions réellement chères et non des erreurs, donc elles sont conservées.
- Le volume est extrêmement asymétrique (jusqu'à 8.6e8) : transformation `log1p`.
- Les échelles de prix diffèrent beaucoup d'une action à l'autre (AAPL autour de 100 $, GOOG et AMZN jusqu'à 800 $).

### 3.2 Préparation et architecture

- Échantillon aléatoire de 200 000 lignes (temps de calcul), découpage 80 % / 20 % (160 000 train, 40 000 test).
- `StandardScaler` sur les variables et sur la cible, **ajusté uniquement sur le train**. Les métriques sont recalculées en dollars après `inverse_transform`.
- **MLP** : `Linear → (BatchNorm) → ReLU → (Dropout)` répété `n_layers` fois, puis une couche de sortie à 1 neurone. Loss : MSE.

### 3.3 GridSearch (scikit-learn + skorch)

| Hyperparamètre | Valeurs testées |
|---|---|
| Learning rate | 0.01, 0.001 |
| Optimiseur | Adam, SGD, RMSprop |
| Epochs | 20, 40 |
| Neurones par couche | 32, 64 |
| Nombre de couches | 1, 2 |

48 combinaisons × 3 folds = 144 entraînements, score = R², sur un sous-échantillon de 30 000 lignes du train.

**Meilleure configuration** : Adam, lr = 0.01, 40 epochs, 32 neurones, 1 couche cachée, **R² (CV) = 0.99984**.
Les cinq meilleures configurations ont toutes un R² compris entre 0.9998 et 0.9999 : la différence est négligeable. Un petit réseau suffit car le problème est presque linéaire.

### 3.4 Courbes Loss / Epochs et R² / Epochs

> **Remarque** : en régression, l'accuracy n'existe pas. Le **R²** est utilisé comme équivalent de l'accuracy.

**Interprétation** : la loss chute dès les premières epochs et se stabilise vers l'epoch 30-40. Les courbes train et test sont superposées : il n'y a **pas d'overfitting**. Deux petits pics isolés apparaissent sur la courbe de test, dus à un learning rate relativement élevé (0.01) ; ils se résorbent d'eux-mêmes.

| Modèle de base | RMSE | MAE | R² |
|---|---|---|---|
| Train | 0.942 $ | 0.395 $ | 0.99985 |
| Test | 0.891 $ | 0.389 $ | 0.99985 |

### 3.5 Régularisation et comparaison

Techniques testées avec les mêmes hyperparamètres : Dropout (0.3), L2 (`weight_decay = 1e-3`), Batch Normalization, Early Stopping (patience 5, jusqu'à 100 epochs) et une combinaison (Dropout 0.2 + L2 1e-4 + Early Stopping).

| Modèle | RMSE test | MAE test | R² train | R² test |
|---|---|---|---|---|
| **Base** | **0.891** | **0.389** | 0.99985 | **0.99985** |
| Early Stopping | 0.905 | 0.447 | 0.99984 | 0.99985 |
| L2 (wd = 1e-3) | 1.311 | 0.862 | 0.99969 | 0.99968 |
| Combiné | 2.552 | 1.851 | 0.99886 | 0.99878 |
| BatchNorm | 3.468 | 2.329 | 0.99776 | 0.99774 |
| Dropout 0.3 (2 exécutions) | 4.089 à 7.526 | 2.276 à 3.713 | 0.9970 à 0.9892 | 0.9969 à 0.9894 |

**Conclusion** : le modèle de base ne souffre pas d'overfitting (train ≈ test), donc la régularisation n'a rien à corriger. Elle dégrade légèrement (L2) ou nettement (Dropout, BatchNorm, Combiné) les résultats, car le réseau (32 neurones, 1 couche) est déjà petit et le problème presque linéaire. Seul l'**Early Stopping** conserve la performance du modèle de base. Avec Dropout, les métriques du train sont moins bonnes que celles du test parce que le dropout est actif uniquement pendant l'entraînement. Les résultats du Dropout varient d'une exécution à l'autre (deux exécutions : RMSE 7.53 puis 4.09), ce qui montre qu'un seul run n'est pas totalement fiable.

**Limites** : le découpage est aléatoire alors que les données sont une série temporelle (un découpage par date serait plus réaliste), et `close` est très proche de `open/low/high`, ce qui rend la tâche facile.

---

## 4. Partie 2 : Classification multi-classe (Predictive Maintenance)

**Dataset** : [Machine Predictive Maintenance Classification](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification) (10 000 lignes). La cible est `Failure Type` (6 classes).

### 4.1 Prétraitement

- Aucune valeur manquante, aucun doublon.
- 9 lignes incohérentes supprimées (`Target = 1` mais `Failure Type = No Failure`).
- Colonnes supprimées : `UDI`, `Product ID` (identifiants) et `Target` (redondante avec la cible, risque de fuite de données).
- `Type` (L/M/H) encodé en one-hot, `Failure Type` avec `LabelEncoder`, variables numériques standardisées (`StandardScaler`, ajusté sur le train uniquement).
- Découpage stratifié 80 % / 20 % : 7 992 lignes en train, 1 999 en test.

### 4.2 Analyse exploratoire (EDA)

- **Classes très déséquilibrées** : environ 96.6 % de « No Failure ». Avant le découpage : Heat Dissipation 112, Power 95, Overstrain 78, Tool Wear 45, **Random Failures 18**. Un modèle qui prédit toujours « No Failure » aurait déjà ~96 % d'accuracy, donc l'accuracy est trompeuse ici.
- `air_temp` / `proc_temp` fortement corrélées (0.88) ; `rpm` / `torque` fortement corrélées négativement (-0.87).
- Les pannes se concentrent aux extrêmes : torque élevé et rpm bas (Overstrain, Heat Dissipation), usure d'outil > 200 min (Tool Wear). `Power Failure` a un torque très dispersé.
- `Random Failures` ne se distingue pas des autres sur les variables, ce qui est cohérent avec leur nature aléatoire.
- `Type` : L 60 %, M 30 %, H 10 %.

### 4.3 Équilibrage (data augmentation)

**SMOTE** appliqué **après le split et uniquement sur le train** (pour éviter toute fuite de données), avec `k_neighbors = 3` à cause de la classe très rare.

- Avant : `[90, 7714, 62, 76, 14, 36]`
- Après : 7 714 exemples par classe (46 284 lignes)

Le jeu de test garde sa distribution réelle.

### 4.4 Architecture

MLP : `Linear → (BatchNorm) → ReLU → (Dropout)` répété `n_layers` fois, puis une couche linéaire à 6 sorties (logits). Loss : `CrossEntropyLoss`.

### 4.5 GridSearch

| Hyperparamètre | Valeurs testées |
|---|---|
| Learning rate | 0.01, 0.001 |
| Optimiseur | Adam, SGD |
| Epochs | 30, 60 |
| Neurones par couche | 32, 64, 128 |
| Nombre de couches | 2, 3 |

48 combinaisons × 3 folds, score = F1 macro. Meilleure configuration retenue pour le modèle final : **Adam, lr = 0.001, 60 epochs, 128 neurones, 3 couches** (F1 macro en validation croisée ≈ 0.97).

> **Limite** : la validation croisée est faite sur des données déjà suréchantillonnées par SMOTE, donc des exemples synthétiques proches d'exemples réels se retrouvent à la fois dans les folds d'entraînement et de validation. Le score (≈ 0.97) est **optimiste** ; le vrai F1 macro sur le test est de 0.43. Une approche plus rigoureuse intègre SMOTE dans un `Pipeline` de `imblearn`, ce qui n'a pas été fait ici.

### 4.6 Courbes Loss / Epochs et Accuracy / Epochs

- **Train** : la loss tend vers 0 et l'accuracy atteint environ 0.99.
- **Test** : la loss démarre très haut (≈ 7) et diminue ; l'accuracy part de ~0 et monte vers 0.89.
- Ce décalage s'explique par la différence de distribution : le train est équilibré (SMOTE) alors que le test est déséquilibré comme en réalité. Au début, le modèle prédit surtout des pannes et se trompe presque toujours sur le test.
- L'écart train/test reste important : le modèle s'adapte fortement aux exemples synthétiques (surapprentissage).

### 4.7 Métriques (modèle de base)

Le train utilisé pour les métriques est le **train d'origine non augmenté** (7 992 lignes).

| | Accuracy | Précision (macro) | Sensibilité / Recall (macro) | F1 (macro) | F1 (weighted) |
|---|---|---|---|---|---|
| Train | 0.910 | 0.409 | 0.983 | 0.515 | 0.936 |
| Test | 0.890 | 0.354 | 0.677 | 0.430 | 0.923 |

Résultats par classe sur le test :

| Classe | Précision | Sensibilité | Support |
|---|---|---|---|
| Heat Dissipation | 0.36 | 1.00 | 22 |
| No Failure | 0.99 | 0.89 | 1929 |
| Overstrain | 0.28 | 1.00 | 16 |
| Power Failure | 0.46 | 0.95 | 19 |
| Random Failures | 0.00 | 0.00 | 4 |
| Tool Wear | 0.03 | 0.22 | 9 |

**Interprétation** :
- La **sensibilité est élevée** (presque toutes les pannes de Heat Dissipation, Overstrain et Power sont détectées) mais la **précision est faible** : beaucoup de machines saines sont signalées en panne (par exemple 99 « No Failure » prédits « Tool Wear » sur le test). C'est une conséquence directe du rééquilibrage : le modèle apprend que toutes les classes sont aussi fréquentes.
- **Random Failures** n'est jamais détectée (0/4) : ces pannes sont aléatoires et non prédictibles à partir des mesures.
- **Tool Wear** est mal prédite : les valeurs d'usure élevées se mélangent à beaucoup de machines normales.
- Le **F1 macro et la sensibilité par classe** sont les indicateurs pertinents ; l'accuracy seule surestime la qualité du modèle.

### 4.8 Régularisation et comparaison

Mêmes techniques qu'en partie 1, entraînées sur le train suréchantillonné et évaluées sur le test réel (Early Stopping : patience 5, jusqu'à 100 epochs).

| Modèle | Acc. train | F1 macro train | Acc. test | Précision test | Sensibilité test | F1 macro test | F1 weighted test |
|---|---|---|---|---|---|---|---|
| Base | 0.910 | 0.515 | 0.890 | 0.354 | 0.677 | 0.430 | 0.923 |
| Dropout 0.3 | 0.785 | 0.360 | 0.759 | 0.289 | 0.692 | 0.339 | 0.840 |
| L2 (wd = 1e-3) | 0.366 | 0.175 | 0.353 | 0.237 | 0.534 | 0.201 | 0.499 |
| BatchNorm | 0.066 | 0.040 | 0.070 | 0.232 | 0.339 | 0.080 | 0.109 |
| **Early Stopping** | 0.986 | 0.867 | **0.957** | **0.443** | 0.582 | **0.493** | **0.963** |
| Combiné | 0.756 | 0.349 | 0.730 | 0.291 | 0.750 | 0.343 | 0.821 |

**Conclusion** :
- L'**Early Stopping** donne le meilleur accuracy, F1 macro et F1 weighted, avec moins de fausses alertes (précision plus élevée), au prix d'une sensibilité plus basse.
- Le **Dropout** et le **modèle combiné** ont la meilleure sensibilité (0.69 à 0.75) mais détectent plus de fausses pannes.
- **L2** et **BatchNorm** dégradent fortement les résultats et rendent l'entraînement instable. BatchNorm est particulièrement sensible au décalage de distribution entre le train équilibré par SMOTE et le test réel.
- Le gain de l'Early Stopping vient probablement en grande partie d'un entraînement plus long (100 epochs au lieu de 60) : la courbe montre que l'arrêt ne s'est apparemment pas déclenché avant la fin.

**Limites** : le jeu de test sert aussi de jeu de validation pour l'Early Stopping (léger biais ; idéalement un jeu de validation séparé) ; un seul run par configuration, donc des résultats qui peuvent varier à cause de l'initialisation aléatoire ; SMOTE à 100 % crée beaucoup d'exemples synthétiques (surtout pour la classe de 14 exemples). Des pistes d'amélioration : rééquilibrage partiel, pondération des classes dans la loss, SMOTE intégré à la validation croisée.

---

## 5. Synthèse personnelle : ce que j'ai appris



Ce TP m'a permis de prendre en main PyTorch : définir une architecture MLP avec `nn.Module`, choisir la loss adaptée (MSE pour la régression, `CrossEntropyLoss` pour la classification) et suivre la loss et les métriques epoch par epoch. J'ai appris à utiliser `skorch` pour relier un modèle PyTorch à `GridSearchCV` de scikit-learn et chercher les meilleurs hyperparamètres (learning rate, optimiseur, epochs, taille du réseau).

J'ai compris que les métriques dépendent de la tâche : l'accuracy n'a pas de sens en régression (R², RMSE, MAE à la place) et elle est trompeuse sur des classes déséquilibrées (F1 macro, sensibilité, précision par classe). J'ai aussi appris à éviter la fuite de données : scaler ajusté sur le train seulement, SMOTE appliqué après le split et uniquement sur le train. J'ai constaté que la validation croisée sur des données déjà suréchantillonnées donne des scores trop optimistes (0.97 en CV contre 0.43 en test).

Enfin, j'ai vu que la régularisation n'est pas toujours bénéfique : en partie 1, il n'y avait pas d'overfitting donc elle a dégradé les résultats, et en partie 2 seul l'Early Stopping a amélioré le modèle, tandis que BatchNorm et L2 l'ont dégradé. Il faut donc diagnostiquer le problème (écart train/test) avant de choisir une technique, et rester honnête sur les limites des résultats.

## 6. Reproduction

1. Ouvrir les notebooks sur Kaggle et ajouter les datasets correspondants.
2. Activer Internet (pour `pip install skorch imbalanced-learn`) et, si possible, le GPU.
3. Exécuter les cellules dans l'ordre.
