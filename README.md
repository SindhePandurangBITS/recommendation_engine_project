# 📽 Content-Based Movie Recommendation System

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)  
[![Stars](https://img.shields.io/github/stars/your_username/recommendation_system_project.svg)](https://github.com/your_username/recommendation_system_project/stargazers)

---

## 🍿 My Movie Obsession ( Motivation )

I absolutely love watching movies from  Mission: Impossible marathons to epic John Wick showdowns but too often Netflix and other platforms feel like they’re nudging me toward in-house hits or the latest releases. I’ve found myself scratching my head at biased, cookie-cutter suggestions when I just want my classic action thrillers! So, I decided to build my own recommendation engines. Now I can binge all my favorites, from Ethan Hunt’s stunts to Wick’s unstoppable vengeance, without any agenda sneaking into my watch list.


---

## 🧩 Why Content-Based & Hybrid?

- **Rich Metadata Utilization**: Combines plot “overview”, “tagline”, keywords, genres, cast/crew, production entities and languages into a unified feature vector.  
- **Term Weighting & Sparsity**: TF-IDF down-weights ubiquitous terms and preserves sparsity for efficient cosine computations [Salton & Buckley, 1988].  
- **Dimensionality Reduction**: Truncated SVD (LSA) captures latent thematic structure in 300 dimensions, balancing expressiveness and performance [Landauer & Dumais, 1997].  
- **Hybrid Fusion**: Stacks TF-IDF vectors with scaled numeric features (budget, popularity, revenue, runtime, vote_average, vote_count) to blend content relevance with popularity and quality signals [Burke, 2002].  
- **Cold-Start & Explainability**: Pure metadata models support new-item recommendations and provide human-readable “why” explanations via top term overlaps.

---

## 🚀 Project Goals

1. **Data Ingestion** from Kaggle’s “the-movies-dataset” (metadata, ratings, credits, keywords).  
2. **Preprocessing & Merging**: ID normalization, JSON parsing (`ast.literal_eval`), DataFrame consolidation.  
3. **Feature Engineering**:  
   - Extract `collection`, `production_houses`, `production_countries_clean`, `spoken_languages_list`  
   - Parse and map `parsed_crew` roles (Director, Producer, Editor, Writer, Screenplay)  
   - Derive cast info (`lead_actor`, `lead_character`, top‐5 actors/characters).  
4. **Missing-Data Analysis & Imputation**:  
   - Numeric: `KNNImputer(n_neighbors=5)` for correlated fill-in.  
   - Categorical: `SimpleImputer(strategy='most_frequent')` and placeholder text for free-form fields.  
5. **Exploratory Data Analysis**: Distribution plots for ratings, genres, popularity, release years, and actor frequencies via Matplotlib/Seaborn.  
6. **Content Assembly**: Clean, lowercase, punctuation-stripped `content` and stemmed `content2` for text vectorizers.  
7. **Modeling & Recommendation Engines**:  
   - **System 1**: `CountVectorizer(max_features=3000)` + cosine similarity  
   - **System 2**: Same counts + `NearestNeighbors(metric='cosine')` (KNN)  
   - **System 3**: `TfidfVectorizer()` + KNN  
   - **System 4**: `CountVectorizer(2000)` + `TruncatedSVD(n_components=300)` + cosine  
   - **System 5**: Hybrid TF-IDF + numeric KNN (`hstack` of sparse+CSR)  
   - **System 6**: Precomputed hybrid cosine similarity matrix  
   - **System 7**: Metadata-only CountVectorizer + cosine  
8. **Evaluation & Metrics**: Precision@K, Recall@K, NDCG@K on “Jason Bourne” query to compare relevance, completeness, and rank quality.  
9. **Year-Bin Analysis**: Decade-based release-year binning (`pd.cut`) to reveal temporal trends.

---

## 📦 Requirements

```text
numpy>=1.24
pandas>=2.1
scikit-learn>=1.2
scipy
matplotlib
seaborn
kaggle
nltk

---
```
## 🚀 Quick Start

# 1. Clone repository
```
git clone https://github.com/your_username/recommendation_system_project.git
cd recommendation_system_project
```

# 2. Install dependencies
```bash
pip install -r requirements.txt
```

# 3. Download & preprocess data
```bash
python src/data/collect.py --output data/raw
python src/preprocessing.py --input data/raw --output data/processed
```

# 4. Run full pipeline
```bash
python scripts/run_pipeline.py
```

# 5. Serve recommendations
```bash
python scripts/serve_recommender.py
```

---

## 📖 Documentation & Notes

The Jupyter notebooks and accompanying PDF include in-depth explanations tailored to our Content-Based Movie Recommendation Engine:

- **Modeling Workflow**: Step-by-step breakdown of each recommendation system (CountVectorizer, TF-IDF, SVD, Hybrid, etc.).
- **Design Rationale**: Discussion of why we selected specific vectorizers, similarity metrics, dimensionality reduction, and hybrid feature combinations.
- **Diagnostic Procedures**: How we ensure data integrity, validate imputation strategies, and verify similarity computations.
- **Evaluation Framework**: Definitions and calculations of Precision@K, Recall@K, and NDCG@K, with practical guidance on interpreting results and choosing the best model.

---

## ⭐ Support
If you find this project helpful:
Give it a ⭐ star on GitHub


---

## 📑 Key References

1. Pazzani, M. J., & Billsus, D. (2007). _Content-based recommendation systems_. In *The Adaptive Web* (pp. 325–341). Springer.

2. Burke, R. (2002). _Hybrid recommender systems: Survey and experiments_. *User Modeling and User-Adapted Interaction*, 12(4), 331–370.

3. Salton, G., & Buckley, C. (1988). _Term-weighting approaches in automatic text retrieval_. *Information Processing & Management*, 24(5), 513–523.
