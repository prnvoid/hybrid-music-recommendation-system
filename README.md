# Hybrid Music Recommendation & Discovery System

A hybrid music recommendation system built on the [Free Music Archive (FMA)](https://github.com/mdeff/fma) dataset, exploring how audio-based representations and collaborative filtering can improve personalized music discovery, particularly for tracks with limited listening history.

**Research Focus:** Recommendation Systems · Music Information Retrieval · Hybrid Recommendation · Long-Tail Discovery · Serendipitous Recommendation · User-Controlled Exploration

---

## Motivation

Conventional music recommendation systems often rely heavily on historical user interactions to predict listening preferences. While effective for established listening patterns, these approaches can struggle with less-exposed music, unfamiliar artists and niche genres.

Tracks with limited interaction history may receive insufficient exposure despite being acoustically relevant to particular listeners.

This project explores a complementary approach: combining **audio-based content representations with collaborative filtering** to support personalized recommendations while investigating the trade-off between relevance and music discovery.

The broader motivation is to move beyond simply predicting what users are likely to consume and investigate how recommendation systems can facilitate meaningful, user-controlled discovery.

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
- Explore relative popularity bias, catalogue exposure and recommendation diversity.
- Study the relationship between musical similarity and learned user preferences.
- Investigate serendipitous recommendation through adjustable similarity-versus-exploration controls.

---

## Dataset

**[Free Music Archive (FMA)](https://github.com/mdeff/fma)**

The FMA dataset provides music tracks, genre labels, metadata and audio material suitable for music information retrieval experiments.

- Full FMA metadata collection: 106,574 tracks.
- Subset used: `fma_small`
- Selected catalogue: 8,000 tracks across 8 balanced top-level genres.

### Current Experimental Dataset

| Characteristic | Value |
|---|---:|
| Metadata catalogue | 8,000 tracks |
| Usable direct-Librosa audio vectors | 7,997 (45 dimensions each) |
| Top-level genres | 8 |
| Simulated users | 500 |
| Tracks represented in interaction matrix | 6,605 |
| Synthetic user-track interactions | 14,993 |
| User-item matrix | 500 × 6,605; 99.55% sparse |

Three source MP3s (track IDs `99134`, `108925` and `133297`) are malformed in the distributed FMA Small archive and cannot be decoded. They remain in the metadata catalogue and are explicitly recorded in the audio-exclusion report; no artificial audio vectors are created for them. Audio feature coverage is 99.9625%.

The collaborative filtering experiments use simulated user interaction histories rather than actual FMA listening histories. This provides a controlled environment for developing and testing the recommendation pipeline, but does not establish real-listener performance. The FMA metadata also includes EchoNest descriptors for 1,294 tracks (16.2% coverage); these partial descriptors are not part of the main 45-dimensional retrieval representation.

FMA Small contains no tracks below the project's absolute threshold of 100 recorded listens. Popularity-aware experiments therefore use **relative popularity tiers within FMA Small**; the low-popularity tier must not be interpreted as tracks with fewer than 100 listens.

---

## System Architecture & Tech Stack

### 1. Audio Feature Extraction

Musical representations are extracted directly from FMA Small MP3 files using **Librosa 0.11.0**, producing 45-dimensional track-level feature vectors.

Features include:

- Mel-Frequency Cepstral Coefficients (MFCC): 20 dimensions.
- Chroma CENS: 12 dimensions.
- Spectral contrast: 7 dimensions.
- Tonnetz: 6 dimensions.

The four feature families are standardized and balanced before L2 normalization for cosine-similarity retrieval. Extraction uses checkpoints so successful tracks do not need to be reprocessed after interruption.

**Technologies:** Python, Librosa, NumPy, Pandas

### 2. Content-Based Recommendation

Content-based retrieval identifies tracks with similar acoustic descriptors through family-balanced feature representations and cosine similarity.

This provides a recommendation pathway that does not depend exclusively on historical user interactions and can surface audio-eligible tracks without collaborative interaction history.

**Technologies:** Scikit-learn, NumPy, FAISS, Cosine Similarity

### 3. Collaborative Filtering

An implicit user-track interaction matrix is constructed using 500 simulated user profiles and 14,993 synthetic interactions. A per-user holdout separates training interactions from evaluation interactions.

Truncated Singular Value Decomposition (SVD) is used to learn 50-dimensional latent representations of users and tracks.

| Component | Dimensions / value |
|---|---:|
| User latent matrix | 500 × 50 |
| Track latent matrix | 8,000 × 50 |
| Latent factors | 50 |
| Training interactions | 12,005 |
| Held-out interactions | 2,988 |

For held-out evaluation, SVD is fitted on the 12,005 training interactions only; the separate serving factors are refitted using the available synthetic interaction history after the evaluation configuration is fixed. The collaborative filtering candidate catalogue includes all 8,000 metadata tracks, including the three tracks without usable audio vectors.

**Technologies:** Python, Pandas, NumPy, SciPy, Scikit-learn, TruncatedSVD

### 4. Vector Search & Retrieval

FAISS indexes the 45-dimensional, L2-normalized audio representations for exact similarity search.

Current implementation includes:

- Audio feature index containing 7,997 eligible tracks.
- `IndexFlatIP` for exact inner-product search.
- A persisted ordered track-ID array to map FAISS result positions back to FMA track IDs.

For L2-normalized vectors, inner-product similarity is equivalent to cosine similarity. `IndexFlatIP` performs exact search; it is **not** an approximate-nearest-neighbour index.

**Technologies:** FAISS, NumPy, Vector Representations

### 5. Hybrid Recommendation & Cold-Start Handling

The recommendation workflow brings together content-based retrieval and collaborative filtering.

The current routing strategy is:

1. Generate audio-similar candidates for a valid seed track.
2. Generate personalized candidates using collaborative filtering from a user's synthetic interaction history.
3. Combine collaborative and audio-profile ranking signals through percentile-based rank fusion, with a separate novelty component.
4. Use audio-based signals for interaction-cold items where audio is available, and CF-only fallback for the three catalogue tracks without usable audio vectors.

An adjustable **Explore vs Similar** mechanism lets users trade acoustic similarity against relative popularity-based novelty. The personalized interface also exposes collaborative-versus-audio weighting and a separate exploration control.

**Research direction:** Hybrid personalization, item cold-start handling, relative long-tail exposure and user-controlled relevance–discovery trade-offs.

### 6. API Integration

A local FastAPI application exposes health, catalogue, similar-track and personalized recommendation endpoints. A browser-based Music Discovery interface provides track search, valid seed selection, recommendation controls and result tables.

The API uses saved Phase 1 and Phase 2 artifacts; it does not require the notebook kernel to remain active. Public deployment is not yet claimed.

**Technologies:** FastAPI, Pydantic, Uvicorn, HTML/CSS/JavaScript

---

## Experimental Results

### Audio Retrieval

The audio retrieval evaluation uses 6,397 reference tracks and 1,600 held-out query tracks. Reference and query IDs do not overlap; preprocessing is fitted on the reference set. Top-level genre agreement is used as a broad-label proxy, not as ground-truth musical similarity.

| Metric | Result |
|---|---:|
| Genre Precision@10 proxy | 0.3036 (30.36%) |
| Random expected genre precision | 0.1250 (12.50%) |
| Genre precision lift over random | 2.429× |
| Retrieval catalogue coverage | 78.99% |
| Mean intra-list diversity | 0.3723 |
| Maximum FAISS vs NumPy score difference | 0.0 |

The retrieval result indicates broad genre coherence above the random expectation. It does not by itself prove that users perceive the returned tracks as musically similar. FAISS scores matched the independent NumPy exact-score calculation in the reported test.

### Collaborative Filtering

The current CF evaluation uses 12,005 training interactions and 2,988 held-out interactions across 500 synthetic users. The 50-factor SVD model is fitted using training interactions only.

| Component | Shape |
|---|---:|
| User factors | 500 × 50 |
| Item factors | 8,000 × 50 |
| Training interaction matrix | 500 × 8,000 |

### Hybrid and Baseline Recommendation Evaluation

Current held-out synthetic-interaction results at top 10:

| Metric | Collaborative SVD | Hybrid rank fusion |
|---|---:|---:|
| Precision@10 | 0.00240 | 0.00200 |
| Recall@10 | 0.004255 | 0.003802 |
| NDCG@10 | 0.003990 | 0.003390 |

The hybrid model does **not** currently outperform SVD on held-out relevance metrics. It does, however, increase catalogue coverage from 12.94% to 22.53% (+74.1% relative), increase mean distinct genres per top-10 list from 2.696 to 3.734 (+38.5%), and increase the relative low-popularity share from 12.42% to 13.88% (+1.46 percentage points). The hybrid's low-popularity share remains below the tier's 20% share of the FMA Small catalogue. This indicates a relevance–discovery trade-off rather than an across-the-board accuracy improvement.

The expected random baseline has Precision@10 of 0.00075 and Recall@10 of 0.001254 in the reported evaluation. Collaborative SVD and hybrid results are above these expectations, but the absolute values remain low. All collaborative evaluation figures are based on synthetic interactions and must not be interpreted as validated real-world recommendation performance.

### Example Recommendation Output

The local interface supports:

- Similar-track retrieval from a valid audio-indexed seed track.
- A user-controlled Explore vs Similar slider for audio retrieval.
- Personalized recommendations for synthetic users.
- Separate collaborative-versus-audio and relevance-versus-exploration controls.

The displayed recommendation scores are ranking scores or percentiles, not probabilities that a listener will like a track.

---

## Complete Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Audio Analysis | Librosa 0.11.0 |
| Machine Learning | Scikit-learn, SciPy |
| Recommendation Algorithms | Content-Based Retrieval, Collaborative Filtering, TruncatedSVD, Hybrid Rank Fusion |
| Audio Representation | 45-D MFCC, Chroma CENS, Spectral Contrast and Tonnetz features |
| Vector Search | FAISS `IndexFlatIP` |
| Similarity and Ranking | Cosine Similarity, Percentile Rank Fusion, Popularity-Based Novelty |
| API & Interface | FastAPI, Pydantic, Uvicorn, HTML, CSS, JavaScript |
| Development Environment | Jupyter Notebook, VS Code |
| Version Control | Git, GitHub |

---

## Installation & Setup

Clone the repository:

```bash
git clone https://github.com/prnvoid/hybrid-music-recommendation-system.git

cd hybrid-music-recommendation-system
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

```bash
# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Download the Dataset

### FMA Metadata

```bash
wget https://os.unil.cloud.switch.ch/fma/fma_metadata.zip

unzip fma_metadata.zip -d data/
```

### FMA Small Audio Subset

```bash
wget https://os.unil.cloud.switch.ch/fma/fma_small.zip

unzip fma_small.zip -d data/
```

The audio subset requires several gigabytes of storage. FFmpeg should be installed and available on `PATH` for decoding MP3 files that cannot be read by the SoundFile backend.

Dataset documentation: https://github.com/mdeff/fma

### Run the local API and interface

From the repository root, start the FastAPI application:

```bash
python -m uvicorn api.main:app --reload
```

Open `http://127.0.0.1:8000/` for the Music Discovery interface and `http://127.0.0.1:8000/docs` for API documentation. The API expects the saved Phase 1 and Phase 2 artifacts under `data/processed/`.

---

## Development Roadmap

- [x] Phase 1 — FMA dataset loading, preprocessing, exploratory analysis and synthetic interaction generation.
- [x] Phase 2 — Direct 45-D Librosa feature extraction, audio validation and content-based similarity.
- [x] Phase 3 — Synthetic user interactions and SVD-based collaborative filtering.
- [x] Phase 4 — FAISS audio index generation and exact vector retrieval.
- [x] Phase 5 — Hybrid rank fusion, novelty scoring and unavailable-audio fallback.
- [x] Phase 6 — Preliminary held-out evaluation against popularity, random and individual-model baselines.
- [ ] Phase 7 — Repeated-split evaluation, cold-start analysis and stronger long-tail experiments.
- [x] Phase 8 — Explore vs Similar and personalized recommendation controls in the local interface.
- [x] Phase 9 — Local FastAPI integration and browser interface; public deployment remains pending.

---

## Future Research Directions

The next stage focuses on evaluating the system beyond conventional recommendation accuracy.

Planned experiments include:

- **Model comparison:** Comparing content-based, collaborative and hybrid recommendation approaches across repeated, leakage-aware splits.
- **Implicit-feedback modelling:** Exploring Alternating Least Squares (ALS) and Bayesian Personalized Ranking (BPR).
- **Recommendation diversity:** Measuring catalogue coverage, novelty, intra-list diversity and artist concentration.
- **Long-tail discovery:** Evaluating exposure and ranking quality across clearly defined relative-popularity segments.
- **Serendipitous recommendation:** Investigating the relationship between relevance and unexpectedness through controlled slider experiments.
- **Preference modelling:** Studying how user preferences and exploration behaviour influence recommendation outcomes.
- **Human-centered evaluation:** Exploring real-user feedback and perceived recommendation usefulness through a listening study.

The broader objective is to investigate recommendation systems that do more than reproduce existing listening patterns: systems that can also facilitate meaningful, user-controlled discovery.

---

**Project Status:** Working Research Prototype | Master's Thesis Development | 2026–Present

**Author:** Pranav R Kumar
