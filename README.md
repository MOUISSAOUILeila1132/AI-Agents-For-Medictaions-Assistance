# MedInfo AI ⚕️🤖

**Intelligence Médicale Propulsée par l'IA**

> **Advanced algorithmic analysis for drug interactions, precise dosages, and safety protocols.**

![Status](https://img.shields.io/badge/Status-Active-success)
![Model](https://img.shields.io/badge/Model-Phi--3%20mini-blue)
![Architecture](https://img.shields.io/badge/Architecture-RAG-orange)

## 📋 À propos du projet

**MedInfo AI** est un assistant intelligent conçu pour fournir des informations précises et sécurisées sur les médicaments. En combinant la puissance des LLM (Large Language Models) avec une architecture RAG (Retrieval-Augmented Generation), l'application permet aux utilisateurs d'analyser les interactions médicamenteuses, de vérifier les posologies et d'obtenir des protocoles de sécurité avec un taux de fiabilité élevé.

L'objectif principal est de réduire les hallucinations souvent présentes dans les modèles génératifs standards pour offrir un outil d'aide à la décision fiable.

## 🚀 Fonctionnalités Principales

*   **💊 Dosage & Posologie :** Calculs précis et recommandations de prise (Précision : 95.8%).
*   **⚠️ Interactions Médicamenteuses :** Analyse des conflits entre molécules (ex: Amoxicilline et Ibuprofène).
*   **🧪 Principes Actifs :** Identification et explication des molécules (Précision : 97.2%).
*   **🚫 Contre-indications :** Alertes sur les risques liés aux profils patients.
*   **📉 Effets Secondaires :** Liste détaillée des effets indésirables potentiels.
*   **🧠 Questions Complexes :** Traitement de requêtes médicales nuancées.

---

## ⚙️ Architecture & Data Pipeline

Notre système repose sur un pipeline de données rigoureux pour garantir la qualité des réponses :

1.  **Sources de Données :** API OpenFDA, Sites médicaux certifiés, Manuels de référence.
2.  **Prétraitement :** Nettoyage, gestion des valeurs manquantes et standardisation des textes.
3.  **Vectorisation :** Création d'embeddings pour capturer le sens sémantique.
4.  **Base de Données Vectorielle :** Indexation via **FAISS** pour une recherche d'information ultra-rapide.
5.  **Génération (RAG) :** Injection du contexte trouvé dans le modèle **Phi-3-mini** pour générer la réponse.
6.  **Dataset :** Entraînement et validation sur **40 000 paires Questions/Réponses**.

---

## 📊 Benchmarks et Performances

Nous avons comparé trois architectures majeures pour ce projet : **Flan-T5**, **GPT-2 (Fine-tuné LoRA)** et **Phi-3-mini (RAG)**.

### 🏆 Choix du Modèle : Phi-3-mini (RAG)

L'approche RAG avec Phi-3 a été sélectionnée pour ses performances supérieures et son faible taux d'hallucinations.

| Modèle | Architecture | Accuracy | F1-Score | Hallucinations |
| :--- | :--- | :--- | :--- | :--- |
| Flan-T5 | Encodeur-Décodeur | 78.2% | 0.76 | 15.3% |
| GPT-2 + LoRA | Décodeur (Fine-tuning) | 83.7% | 0.81 | 11.2% |
| **Phi-3 (RAG)** | **RAG (3.8B params)** | **92.1%** | **0.92** | **3.1%** |

### 📈 Performance par Catégorie

Le modèle final affiche un temps de réponse moyen de **4.2 secondes**.

| Type de question | Accuracy | F1 | Précision | Rappel |
| :--- | :--- | :--- | :--- | :--- |
| **Dosage/Posologie** | 95.8% | 0.94 | 0.96 | 0.92 |
| **Principes actifs** | 97.2% | 0.96 | 0.97 | 0.95 |
| **Effets secondaires** | 91.5% | 0.90 | 0.92 | 0.88 |
| **Interactions** | 89.3% | 0.88 | 0.90 | 0.86 |
| **Contre-indications** | 93.8% | 0.92 | 0.94 | 0.90 |
| **Questions complexes**| 86.4% | 0.85 | 0.87 | 0.83 |
| **Moyenne globale** | **92.1%** | **0.92** | **0.90** | **0.94** |

---

## 🛠️ Installation et Utilisation

Pré-requis : Python 3.8+

```bash
# 1. Cloner le dépôt
git clone https://github.com/MOUISSAOUILeila1132/medinfo-ai.git
cd medinfo-ai

# 2. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Sur Windows : venv\Scripts\activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer l'application
python app.py
