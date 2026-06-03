# hybrid-music-recommendation-system
A content-based music recommendation system built on the Free Music Archive (FMA) dataset, 
specifically designed to handle the long-tail problem or the cold start problem - recommending obscure songs that have little to no listening history.

## The problem this solves

Most recommendation systems (Spotify, YouTube) over-recommend popular songs because they rely on user interaction data. 95% of songs have almost no play history. 
This system recommends those songs using audio features instead.

## 🔬 Motivation

Traditional recommendation systems suffer from popularity bias.

Songs with:

- few listens
- new artists
- niche genres

often receive no recommendations because they lack interaction data.

This project explores how content-based methods can improve discovery for these tracks.

## Tech stack

- **Embeddings**: 45-dimensional audio feature vectors (MFCC, chroma, spectral contrast, tonnetz) extracted with librosa  
- **Similarity**: Cosine similarity on normalized feature embeddings  
- **Collaborative filtering**: SVD matrix factorization for songs with interaction data  
- **Cold start handling**: Content-based fallback routing for zero-interaction tracks  
- **API**: FastAPI REST endpoint  
- **Vector search**: FAISS ANN indexing

- ## Dataset

[FMA: Free Music Archive](https://github.com/mdeff/fma) — 106,574 tracks, heavily 
independent/underground artists, purpose-built for studying the long tail of music.

Subset used: `fma_small` (8,000 tracks, 8 genres)
