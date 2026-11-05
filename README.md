# Netflix Content-Based Recommendation System

A machine-learning recommender that suggests similar Netflix titles based on genres, categories and other content attributes. Built with Python, pandas and scikit-learn.

## Overview

Given the name of a Netflix title, the system returns the 10 most similar titles. Similarity is computed purely from content metadata (content-based filtering), so no user ratings or viewing history are required.

## Dataset

- **Source:** Netflix Titles dataset (Kaggle)
- **Size:** 8,790 titles, 10 columns
- **Columns:** `show_id, type, title, director, country, date_added, release_year, rating, duration, listed_in`
- **Missing values:** none in this version of the dataset (missing directors are stored as the text "Not Given")

## Method

The workflow follows five steps.

### 1. Prepare content-related features
The following columns are cleaned (lowercased, spaces removed inside names so that "Kirsten Johnson" becomes the single token `kirstenjohnson`) and combined into one text field called the **soup**:

- `type` (Movie / TV Show)
- `listed_in` (genres), included twice to give it more weight
- `director`
- `country`
- `rating`

Example soup: `movie documentaries documentaries kirstenjohnson unitedstates pg13`

### 2. Convert text into a machine-readable format
The soup is converted into numeric vectors using **TF-IDF** (`TfidfVectorizer`), which down-weights common tokens and up-weights rare, distinctive ones.

- Resulting matrix shape: **(8790, 5403)**

### 3. Calculate content similarity scores
**Cosine similarity** (computed with `linear_kernel`) is calculated between every pair of titles, producing an 8790 x 8790 similarity matrix.

### 4. Generate recommendations
`recommend(title, n=10)` looks up the title, sorts all other titles by similarity, and returns the top `n`.

```python
recommend("Ganglands")
```

Example output (abridged):

| Title | Type | Country | Similarity |
|---|---|---|---|
| Lupin | TV Show | France | 0.874 |
| Crime Time | TV Show | France | 0.874 |
| Fatal Destiny | TV Show | Pakistan | 0.810 |
| Agent Raghav | TV Show | India | 0.803 |
| Smoking | TV Show | Japan | 0.803 |

All recommendations share the Crime TV Shows genre, and the French titles score highest because they also share the country with the query title.

### 5. Evaluate recommendation quality
Since no user ratings are available, quality was evaluated with a **genre precision@10** metric: for 300 randomly sampled titles, the fraction of the top-10 recommendations sharing at least one genre with the query title, averaged over the sample (seed = 42).

| Model | Genre precision@10 |
|---|---|
| Soup with genre | 0.997 |
| Soup without genre (ablation) | 0.457 |

**Interpretation:**
- The 0.997 score is inflated because genre is used both as a feature and in the evaluation metric. It confirms the system uses genre correctly but is not an independent measure of quality.
- The ablation (genre removed from the soup) gives a fairer picture: 0.457. This shows that genre is the most important feature, while director, country, type and rating still carry useful signal.

## Limitations

- The dataset has no `description` or `cast` columns, so plot and actor similarity cannot be captured.
- The evaluation metric is based on genre overlap, which is only a proxy for true user satisfaction.
- The system is content-based only, so it does not personalise by user and does not use popularity.
- Title lookup is an exact (case-insensitive) match.

## Future Improvements

- Add `description` text and `cast` if available, and use n-grams or sentence embeddings.
- Tune feature weights (for example genre, director and country).
- Add fuzzy title matching for typos.
- Add a user-profile layer for personalised suggestions based on a user's watch history.
- Evaluate with real user feedback, for example click-through or ratings.

## How to Run

1. Open the notebook in Google Colab or Jupyter.
2. Upload `netflix.csv` and update the file path in the first cell.
3. Run the cells in order.

```bash
pip install pandas numpy scikit-learn
```

## Tech Stack

Python, pandas, NumPy, scikit-learn (`TfidfVectorizer`, `linear_kernel`)
