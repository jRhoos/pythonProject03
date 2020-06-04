# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

my_data = pd.read_csv("artists_billboard_fix3.csv", delimiter= ",")
my_data.head(20)
my_data.columns
my_data.dtypes

## 'id', 'title', 'artist', 'mood', 'tempo', 'genre', 'artist_type', 'chart_date', 'durationSeg', 'top', 'anioNacimiento'

X = my_data[['mood', 'tempo', 'genre', 'artist_type', 'chart_date', 'durationSeg', 'anioNacimiento']].values
y = my_data[['top',]].values

# en los modelos no aceptan string hay que limpiar
from sklearn import preprocessing

le_mood = preprocessing.LabelEncoder()
le_mood.fit(['Brooding', 'Energizing', 'Excited', 'Yearning', 'Upbeat', 'Cool',
       'Urgent', 'Aggressive', 'Sophisticated', 'Defiant', 'Sensual',
       'Empowering', 'Gritty', 'Romantic', 'Rowdy', 'Other', 'Fiery',
       'Sentimental', 'Easygoing', 'Stirring', 'Melancholy', 'Peaceful',
       'Lively'])
X[:,0] = le_mood.transform(X[:,0])

le_tempo = preprocessing.LabelEncoder()
le_tempo.fit(['Slow Tempo', 'Fast Tempo', 'Medium Tempo'])
X[:,1] = le_tempo.transform(X[:,1])

le_genre = preprocessing.LabelEncoder()
le_genre.fit(['Traditional', 'Pop', 'Urban', 'Alternative & Punk', 'Electronica', 'Other', 'Soundtrack', 'Rock', 'Jazz'])
X[:,2] = le_genre.transform(X[:,2])

le_artista_type = preprocessing.LabelEncoder()
le_artista_type.fit(['Male', 'Female', 'Mixed'])
X[:,3] = le_artista_type.transform(X[:,3])


pd.unique(my_data['mood'])
pd.unique(my_data['genre'])

X[0:20]

# my_data['artista_edad'] = df['chart_date']
# my_data['chart_date'].replace(np.nan, 0, inplace = True)
my_data['chart_date'].astype('str')
my_data['chart_date'] = my_data['chart_date'].str[:2]


# dividimos nuestro dataset para el entrenamientoy para su evaluacion
from sklearn.model_selection import train_test_split
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size = 0.2, random_state = 4)


# importar el modelo
from sklearn.tree import DecisionTreeClassifier

fartree = DecisionTreeClassifier(criterion = "entropy", max_depth=4)
fartree.fit(X_trainset, y_trainset)
fartree
y_hat_tree= fartree.predict(X_testset)


from sklearn import metrics
print("Exactitud del Modelo de DesicionTree es:", metrics.accuracy_score(y_testset, y_hat_tree))


# generar grafico 
from sklearn.externals.six import StringIO

import pydotplus
import matplotlib.image as mpimg
from sklearn import tree

import matplotlib.pyplot as plt


dot_data = StringIO()
filename = "DecissionTree_Artista.png"
featureNames = my_data.columns[0:7]
targetNames = my_data["top"].unique().tolist()
out = tree.export_graphviz(fartree, 
                           out_file = dot_data,
                           feature_names = featureNames,
                           class_names = targetNames,
                           filled = True,
                           special_characters = True,
                           rotate = False
                           )

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize = (100, 200))
plt.imshow(img, interpolation = "nearest")
