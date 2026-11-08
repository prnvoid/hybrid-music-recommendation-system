# Hybrid Music Recommendation & Discovery System

A hybrid music recommendation system built on the [Free Music Archive (FMA)](https://github.com/mdeff/fma) dataset, exploring how audio-based representations and collaborative filtering can improve personalized music discovery, particularly for tracks with limited listening history.

**Research Focus:** Recommendation Systems · Music Information Retrieval · Serendipitous Recommendation · Long-Tail Discovery · User Preference Modelling

---

## Motivation

Conventional music recommendation systems often rely heavily on historical user interactions to predict listening preferences. While effective for popular tracks and established listening patterns, these approaches can struggle with less-exposed music, unfamiliar artists and niche genres.

Tracks with limited listening history may receive insufficient exposure despite being musically relevant to particular listeners.

This project explores a complementary approach: combining **audio-based content representations with collaborative filtering** to support personalized recommendations while investigating opportunities for long-tail music discovery.

The broader motivation is to move beyond simply predicting what users are likely to consume and investigate how recommendation systems can facilitate meaningful, unexpected discoveries.

## The Problem

Traditional collaborative filtering presents several challenges:

- **Cold-start problem:** New or less-exposed tracks have insufficient interaction data.
- **Popularity bias:** Frequently interacted-with tracks can dominate recommendation lists.
- **Long-tail discovery:** Music from independent artists and niche genres can remain underrepresented.
- **Limited personalization:** Listening history alone may not capture the acoustic characteristics or evolving preferences of a listener.

This project investigates how content-based retrieval can complement collaborative filtering to address these challenges.

## Research Objectives

- Develop a hybrid recommendation framework combining content-based and collaborative signals.
- Investigate cold-start handling for tracks without interaction history.
- Explore long-tail music discovery and popularity bias.
- Study the relationship between musical similarity and learned user preferences.
- Investigate serendipitous recommendation through an adjustable exploration-versus-familiarity mechanism.

---

## Dataset

**[Free Music Archive (FMA)](https://github.com/mdeff/fma)**

The FMA dataset provides music tracks, genre labels, metadata and audio material suitable for music information retrieval experiments.

- Full FMA collection: 106,574 tracks.
- Subset used: `fma_small`
- Selected catalogue: 8,000 tracks across 8 balanced genres.

### Current Experimental Dataset

| Characteristic | Value |
|---|---:|
| Catalogue tracks | 8,000 |
| Genres | 8 |
| Simulated users | 500 |
| Tracks represented in interaction matrix | 4,978 |
| Non-zero user-track pairs | 14,873 |
| Interaction matrix | 500 × 4,978 |
| Matrix sparsity | 99.4% |

The collaborative filtering experiments use simulated user interaction histories rather than actual FMA listening histories. This provides a controlled environment for developing and testing the recommendation pipeline.

---

## System Architecture & Tech Stack

### 1. Audio Feature Extraction

Musical representations are extracted using **Librosa**, producing 45-dimensional audio feature vectors.

Features include:

- Mel-Frequency Cepstral Coefficients (MFCC)
- Chroma features
- Spectral descriptors
- Tonnetz features

The extracted features are normalized before similarity-based retrieval.

**Technologies:** Python, Librosa, NumPy, Pandas

### 2. Content-Based Recommendation

Content-based retrieval identifies tracks with similar acoustic characteristics through normalized feature representations and cosine similarity.

This provides a recommendation pathway that does not depend exclusively on historical user interactions.

**Technologies:** Scikit-learn, NumPy, Cosine Similarity

### 3. Collaborative Filtering

An implicit user-track interaction matrix is constructed using 500 simulated user profiles.

Singular Value Decomposition (SVD) is used to learn latent representations of users and tracks.

| Component | Dimensions |
|---|---:|
| User latent matrix | 500 × 50 |
| Track latent matrix | 4,978 × 50 |
| Latent factors | 50 |
| Matrix sparsity | 99.4% |

The resulting latent representations support personalized top-N recommendation generation.

**Technologies:** Python, Pandas, NumPy, Scikit-learn, SVD

### 4. Vector Search & Retrieval

FAISS is used to index normalized audio and unified item representations for efficient similarity-based retrieval.

Current implementation includes:

- Audio feature index.
- Unified item representation index.
- `IndexFlatIP` for exact inner-product search.

For L2-normalized vectors, inner-product similarity is equivalent to cosine similarity.

The pipeline currently maintains:

- 45-dimensional audio representations.
- 64-dimensional unified item representations.

**Technologies:** FAISS, NumPy, Vector Representations

### 5. Hybrid Recommendation & Cold-Start Handling

The recommendation workflow brings together content-based retrieval and collaborative filtering.

The intended routing strategy is:

1. Generate candidate tracks through audio-based similarity.
2. Generate personalized candidates using collaborative filtering where interaction history is available.
3. Combine recommendation signals through the hybrid workflow.
4. Use content-based retrieval as a fallback for tracks without interaction history.

An adjustable **Explore vs Familiar** mechanism is also being developed to investigate how recommendations change when users are exposed to more unfamiliar tracks.

**Research direction:** Hybrid personalization, cold-start recommendation and serendipitous discovery.

### 6. API Integration

A FastAPI-based REST interface is planned to expose recommendation functionality through API endpoints.

**Planned technology:** FastAPI

---

## Experimental Results

### Collaborative Filtering

The current SVD implementation uses 50 latent factors to represent users and tracks.

**Factorization output:**

| Component | Shape |
|---|---:|
| User matrix (U) | 500 × 50 |
| Singular values | 50 |
| Track matrix (Vt) | 50 × 4,978 |
| Predicted interaction matrix | 500 × 4,978 |

The current pipeline reports the following distribution of retained singular-value energy:

| Retained factors | Reported variance/energy |
|---|---:|
| Top 10 | 25.1% |
| Top 20 | 45.4%
