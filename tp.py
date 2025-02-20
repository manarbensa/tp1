import numpy as np

# Exercice 1
liste = [5, 10, 15, 20, 25]
premier = np.array(liste, dtype=np.float64)
premier

liste_imbriquee = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
deuxieme = np.array(liste_imbriquee)
print(deuxieme.shape)
print(deuxieme.size)

tableau = np.random.rand(2, 3, 4)
print(tableau)
print(tableau.shape)
print(tableau.ndim)

# Exercice 2
tableau1 = np.arange(10)
tableau1
inverse = np.flip(tableau1)
inverse

tableau2 = np.arange(12).reshape(3, 4)
tableau2
tableau2[:2, 2:]

tableau3 = np.random.uniform(0, 11, size=(5, 5))
tableau3[tableau3 > 5] = 0
tableau3

# Exercice 3
identite = np.eye(3)
print(identite.ndim)
print(identite.shape)
print(identite.size)
print(identite.itemsize)
print(identite.nbytes)

tableau4 = np.linspace(0, 5, 10)
tableau4

# Exercice 4
tableau5 = np.random.randint(0, 50, size=20)
indices = [2, 5, 7, 10, 15]
elements_extraits = tableau5[indices]
print("Éléments extraits:", elements_extraits)

tableau6 = np.random.randint(0, 30, size=(4, 5))
masque_booleen = tableau6 > 15
elements_selectionnes = tableau6[masque_booleen]
print("Plus que 15:", elements_selectionnes)

tableau7 = np.random.randint(-10, 10, size=10)
tableau7[tableau7 < 0] = 0
print("Négatif à zéro:", tableau7)

# Exercice 5
tableau8 = np.random.randint(0, 10, size=5)
tableau9 = np.random.randint(0, 10, size=5)
concatenation = np.concatenate((tableau8, tableau9))
print("Concaténé:", concatenation)

tableau_2d_6x4 = np.random.randint(0, 10, size=(6, 4))
tableau_2d_6x4

coupe_2d_6x4 = np.split(tableau_2d_6x4, 2, axis=0)
print("Première partie:")
print(coupe_2d_6x4[0])
print("Deuxième partie:")
print(coupe_2d_6x4[1])

# Exercice 6
tableau_1d = np.random.randint(1, 100, size=15)
print("Tableau 1D:", tableau_1d)

moyenne_1d = np.mean(tableau_1d)
mediane_1d = np.median(tableau_1d)
ecart_type_1d = np.std(tableau_1d)
variance_1d = np.var(tableau_1d)

print("Moyenne:", moyenne_1d)
print("Médiane:", mediane_1d)
print("Écart-type:", ecart_type_1d)
print("Variance:", variance_1d)

tableau_2d = np.random.randint(1, 50, size=(4, 4))
print("Tableau 2D:\n", tableau_2d)

somme_lignes = np.sum(tableau_2d, axis=1)
somme_colonnes = np.sum(tableau_2d, axis=0)

print("Somme de chaque ligne:", somme_lignes)
print("Somme de chaque colonne:", somme_colonnes)

tableau_3d = np.random.randint(1, 20, size=(2, 3, 4))
print("Tableau 3D:\n", tableau_3d)

# Exercice 7
tableau_1d_1_12 = np.arange(1, 13)
tableau_2d_3x4 = tableau_1d_1_12.reshape(3, 4)
print("Tableau 2D de forme (3, 4):\n", tableau_2d_3x4)

tableau_2d_aleatoire_3x4 = np.random.randint(1, 11, size=(3, 4))
print(tableau_2d_aleatoire_3x4)
transpose = tableau_2d_aleatoire_3x4.T
print("Tableau 2D transposé:\n", transpose)

tableau_2d_aleatoire_2x3 = np.random.randint(1, 11, size=(2, 3))
print(tableau_2d_aleatoire_2x3)
flatten = tableau_2d_aleatoire_2x3.flatten()
print("Aplati:\n", flatten)

# Exercice 8
tableau_2d_3x4_aleatoire = np.random.randint(1, 11, size=(3, 4))
print(tableau_2d_3x4_aleatoire)

moyenne_colonnes = np.mean(tableau_2d_3x4_aleatoire, axis=0)
resultat = tableau_2d_3x4_aleatoire - moyenne_colonnes
print("Résultat après soustraction:\n", resultat)

# Exercice 9
tableau_1d_aleatoire = np.random.randint(1, 21, size=10)
print(tableau_1d_aleatoire)
tableau_1d_trie = np.sort(tableau_1d_aleatoire)
print("Tableau 1D trié:", tableau_1d_trie)
tableau_1d_aleatoire = np.random.randint(1, 21, size=10)
print(tableau_1d_aleatoire)
tableau_1d_trie = np.sort(tableau_1d_aleatoire)
print("Tableau 1D trié:", tableau_1d_trie)

tableau_2d_aleatoire = np.random.randint(1, 51, size=(3, 5))
tableau_2d_trie = tableau_2d_aleatoire[tableau_2d_aleatoire[:, 1].argsort()]
print("Trié par la deuxième colonne:\n", tableau_2d_trie)

tableau_1d_aleatoire_100 = np.random.randint(1, 101, size=15)
print(tableau_1d_aleatoire_100)
indices_superieurs_a_50 = np.where(tableau_1d_aleatoire_100 > 50)[0]
print("Indices des éléments supérieurs à 50:", indices_superieurs_a_50)


#exercice 10
tableau_2x2 = np.random.randint(1, 11, size=(2, 2))
print(tableau_2x2)
determinant = np.linalg.det(tableau_2x2)
print("Déterminant:", determinant)

tableau_3x3 = np.random.randint(1, 6, size=(3, 3))
print(tableau_3x3)
valeurs_propres, vecteurs_propres = np.linalg.eig(tableau_3x3)
print("Valeurs propres:", valeurs_propres)
print("Vecteurs propres:\n", vecteurs_propres)


tableau_2x3 = np.random.randint(1, 11, size=(2, 3))
tableau_3x2 = np.random.randint(1, 11, size=(3, 2))
print("Tableau 2x3:\n", tableau_2x3)
print("Tableau 3x2:\n", tableau_3x2)
produit_matriciel = np.dot(tableau_2x3, tableau_3x2)
print("Produit matriciel:\n", produit_matriciel)


#exercice 11
# Tableau 1D de distribution uniforme entre 0 et 1
tableau_uniforme = np.random.uniform(0, 1, 10)
print("Tableau 1D de distribution uniforme :", tableau_uniforme)

tableau_normal = np.random.normal(0, 1, (3, 3))
print("Tableau 2D de distribution normale :\n", tableau_normal)

tableau_entiers_aleatoires = np.random.randint(1, 101, 20)
print("Tableau 1D d'entiers aléatoires :", tableau_entiers_aleatoires)
histogramme, intervalles = np.histogram(tableau_entiers_aleatoires, bins=5)
print("Histogramme :", histogramme)
print("Intervalles :", intervalles)

#exercise 12
tableau_2d_5x5 = np.random.randint(1, 21, size=(5, 5))
print(tableau_2d_5x5)
elements_diagonaux = np.diagonal(tableau_2d_5x5)
print("Éléments diagonaux :", elements_diagonaux)

tableau_1d_10 = np.random.randint(1, 51, size=10)
print("Tableau 1D de 10 entiers aléatoires :", tableau_1d_10)

# Fonction pour vérifier si un nombre est premier
def est_premier(nombre):
    if nombre < 2:
        return False
    for i in range(2, int(np.sqrt(nombre)) + 1):
        if nombre % i == 0:
            return False
    return True

elements_premiers = tableau_1d_10[np.vectorize(est_premier)(tableau_1d_10)]
print("Éléments premiers :", elements_premiers)


# Nombres pairs
tableau_2d_4x4 = np.random.randint(1, 11, size=(4, 4))
print(tableau_2d_4x4)
elements_pairs = tableau_2d_4x4[tableau_2d_4x4 % 2 == 0]
print("Éléments pairs :", elements_pairs)


#exercie 13
tableau_1d_aleatoire_10 = np.random.randint(1, 11, size=10).astype(float)
print("Tableau 1D :", tableau_1d_aleatoire_10)
indices_nan = np.random.choice(tableau_1d_aleatoire_10.size, size=3, replace=False)
tableau_1d_aleatoire_10[indices_nan] = np.nan
print("Tableau 1D avec np.nan à des positions aléatoires :", tableau_1d_aleatoire_10)

tableau_2d_aleatoire_3x4 = np.random.randint(1, 11, size=(3, 4)).astype(float)
print(tableau_2d_aleatoire_3x4)
# Remplacement des valeurs < 5 par np.nan
tableau_2d_aleatoire_3x4[tableau_2d_aleatoire_3x4 < 5] = np.nan
print("Tableau 2D après remplacement des éléments < 5 par np.nan :\n", tableau_2d_aleatoire_3x4)

tableau_1d_aleatoire_15 = np.random.randint(1, 21, size=15).astype(float)
# Introduction de np.nan à des positions aléatoires
indices_nan_15 = np.random.choice(tableau_1d_aleatoire_15.size, size=3, replace=False)
tableau_1d_aleatoire_15[indices_nan_15] = np.nan
indices_nan_dans_tableau = np.where(np.isnan(tableau_1d_aleatoire_15))[0]
print("Indices des valeurs np.nan :", indices_nan_dans_tableau)
print("Tableau avec des valeurs np.nan :", tableau_1d_aleatoire_15)


#exercice 14

import time
grand_tableau_1d = np.random.randint(1, 101, size=1000000)

# Mesure du temps pris pour calculer la moyenne et l'écart-type
debut_temps = time.time()
moyenne_grand_tableau_1d = np.mean(grand_tableau_1d)
ecart_type_grand_tableau_1d = np.std(grand_tableau_1d)
temps_calcul_1d = time.time() - debut_temps
print("Moyenne du grand tableau 1D :", moyenne_grand_tableau_1d)
print("Écart-type du grand tableau 1D :", ecart_type_grand_tableau_1d)
print("Temps pris pour les opérations sur le tableau 1D :", temps_calcul_1d)

grand_tableau_2d_1 = np.random.randint(1, 11, size=(1000, 1000))
grand_tableau_2d_2 = np.random.randint(1, 11, size=(1000, 1000))

# Mesure du temps pris pour effectuer l'addition élément par élément
debut_temps = time.time()
somme_grands_tableaux_2d = np.add(grand_tableau_2d_1, grand_tableau_2d_2)
temps_calcul_2d = time.time() - debut_temps
print("Temps pris pour l'addition élément par élément des tableaux 2D :", temps_calcul_2d)

grand_tableau_3d = np.random.randint(1, 11, size=(100, 100, 100))
# Mesure du temps pris pour effectuer des sommes selon les axes
debut_temps = time.time()
somme_axis0 = np.sum(grand_tableau_3d, axis=0)
somme_axis1 = np.sum(grand_tableau_3d, axis=1)
somme_axis2 = np.sum(grand_tableau_3d, axis=2)
temps_calcul_3d = time.time() - debut_temps
print("Temps pris pour les opérations de somme sur le tableau 3D :", temps_calcul_3d)


#exercice 15
tableau_1_a_10 = np.arange(1, 11)
print("Tableau 1D de 1 à 10 :", tableau_1_a_10)
# Somme et produit cumulatifs
somme_cumulative = np.cumsum(tableau_1_a_10)
produit_cumulatif = np.cumprod(tableau_1_a_10)
print("Somme cumulative :", somme_cumulative)
print("Produit cumulatif :", produit_cumulatif)

tableau_2d_4x4_aleatoire = np.random.randint(1, 21, size=(4, 4))
print("Tableau 2D de taille (4, 4) :\n", tableau_2d_4x4_aleatoire)
# Somme cumulative le long des lignes et des colonnes
somme_cumulative_lignes = np.cumsum(tableau_2d_4x4_aleatoire, axis=1)
somme_cumulative_colonnes = np.cumsum(tableau_2d_4x4_aleatoire, axis=0)
print("Somme cumulative le long des lignes :\n", somme_cumulative_lignes)
print("Somme cumulative le long des colonnes :\n", somme_cumulative_colonnes)

tableau_1d_aleatoire_1_a_50 = np.random.randint(1, 51, size=10)
print("Tableau 1D avec des entiers aléatoires entre 1 et 50 :", tableau_1d_aleatoire_1_a_50)
# Calcul du minimum, maximum et somme du tableau
valeur_min = np.min(tableau_1d_aleatoire_1_a_50)
valeur_max = np.max(tableau_1d_aleatoire_1_a_50)
somme_totale = np.sum(tableau_1d_aleatoire_1_a_50)
print("Valeur minimale :", valeur_min)
print("Valeur maximale :", valeur_max)
print("Somme du tableau :", somme_totale)
 


#exercice 16

aujourdhui = np.datetime64('today', 'D')

# Création d'un tableau de 10 dates à partir d'aujourd'hui avec une fréquence quotidienne
dates_quotidiennes = np.arange(aujourdhui, aujourdhui + 10, dtype='datetime64[D]')
print("Tableau de 10 dates à partir d'aujourd'hui avec une fréquence quotidienne :\n", dates_quotidiennes)

# Création d'un tableau de dates mensuelles de janvier 2022 à mai 2022
dates_mensuelles = np.arange('2022-01', '2022-06', dtype='datetime64[M]')
print("\nTableau de 5 dates à partir de janvier 2022 avec une fréquence mensuelle :\n", dates_mensuelles)

# Génération de 10 dates aléatoires dans l'année 2023
horodatages_aleatoires = np.datetime64('2023-01-01') + np.random.randint(0, 365, size=10).astype('timedelta64[D]')
print("\nTableau 1D avec 10 horodatages aléatoires en 2023 :\n", horodatages_aleatoires)


#exercice 17
# Définition d'un dtype structuré pour stocker un entier et sa représentation binaire
dt = np.dtype([('integer', np.int32), ('binary', 'U10')])  
data = np.array([(i, bin(i)[2:]) for i in range(5)], dtype=dt)
print("\nReprésentation binaire des entiers :\n", data)

# Définition d'un dtype pour stocker des nombres complexes sous forme de parties réelle et imaginaire
complex_dtype = np.dtype([('real', np.float64), ('imag', np.float64)])
array = np.array([(1+2j, 3+4j, 5+6j)], dtype=complex_dtype)
print("\nTableau 2D de nombres complexes :\n", array)

# Définition d'un dtype structuré pour stocker des informations sur des livres
book_dtype = np.dtype([('title', 'U50'), ('author', 'U50'), ('pages', np.int32)])
books = np.array([('Book1', 'Author1', 300), ('Book2', 'Author2', 250), ('Book3', 'Author3', 400)], dtype=book_dtype)
print("\nTableau structuré pour stocker des informations sur les livres :\n", books)


