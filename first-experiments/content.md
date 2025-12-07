# Contention L2 Cache sur GPU
Ce dossier contient des scripts et des kernels pour mener des expériences sur la contention du cache L2 dans les GPU NVIDIA. L'objectif est d'étudier comment différents paramètres de configuration affectent les performances d'un kernel "victime" lorsqu'il est exécuté simultanément avec un kernel "aggresseur" conçu pour saturer le cache L2.
## Structure du Dossier
- `main.cu` : Lancement du kernel "victime" et le kernel "aggresseur". Sans nvidia MPS pour l'instant.
- `interferenceL2.py` : Script principal pour exécuter les expériences de contention L2. Il génère des configurations, exécute les kernels et collecte les métriques de performance.
- `victim_kernel.md` : Documentation expliquant le fonctionnement du kernel "victime".
- `results/` : Dossier contenant les résultats des expériences, y compris les fichiers CSV et les visualisations.  
    Chaque sous-dossier dans `results/` correspond à une expérience spécifique et contient un fichier `observation.md` qui analyse les résultats obtenus.

## Objectif des Expériences
L'objectif principal de ces expériences est de comprendre comment la contention du cache L2 affecte les performances des applications GPU. En ajustant divers paramètres, nous pouvons observer comment ces facteurs influencent le temps d'exécution. 

## Instructions pour l'Exécution
Pour exécuter les expériences, suivez les instructions détaillées dans le fichier `commands.md`. Assurez-vous d'avoir un environnement CUDA configuré et les outils Nsight installés.
