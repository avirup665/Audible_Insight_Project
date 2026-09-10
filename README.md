# Audible Insights: Intelligent Book Recommendations

Audible Insights is a small recommendation-system project built from two Audible catalog CSV files. The focus is on a reproducible data-cleaning pipeline, useful exploratory analysis, and recommendations that can be explained in a project review or viva.

The application is written in Python and Streamlit and is designed to run locally from VS Code or deploy directly to Streamlit Community Cloud.

## What the project does

- Cleans and deduplicates both supplied catalog files.
- Matches records using normalized book title + author.
- Treats `-1` values as missing where the source uses them as placeholders.
- Extracts usable genre labels from the `Ranks and Genre` text.
- Converts listening times such as `3 hours and 23 minutes` to minutes.
- Builds a TF-IDF content model from title, author, genres and descriptions.
- Uses cosine nearest neighbours for similar-book recommendations.
- Builds SVD + K-Means clusters for another view of book similarity.
- Adds a hybrid ranking that combines content similarity, rating quality and popularity.
- Provides genre recommendations, hidden-gem suggestions and an EDA dashboard.

## Important data limitations

The supplied files contain aggregate book ratings and review counts, not individual user-book interactions. Because of that, this project does **not** pretend to implement collaborative filtering or report user-level RMSE/precision/recall. The Methodology page shows content and clustering diagnostics instead.

The source files also do not contain publication years, so publication-year trends are not fabricated.

## Project structure

```text
Audible_Insights_Project/
├── data/
│   ├── raw/
│   │   ├── Audible_Catlog.csv
│   │   └── Audible_Catlog_Advanced_Features.csv
│   └── processed/
│       └── audible_cleaned.csv
├── models/
│   ├── model_bundle.joblib
│   └── model_metrics.json
├── src/
│   ├── app_helpers.py
│   ├── config.py
│   ├── data_pipeline.py
│   └── recommender.py
├── scripts/
│   └── rebuild_artifacts.py
├── assets/
│   └── styles.css
├── notebooks/
│   └── 01_project_walkthrough.ipynb
├── tests/
│   ├── test_pipeline.py
│   └── test_recommender.py
├── .streamlit/
│   └── config.toml
├── streamlit_app.py
├── requirements.txt
├── PROJECT_NOTES.md
└── README.md
```

## Run in VS Code (Windows)

### 1. Open the folder

In VS Code choose **File → Open Folder** and select `Audible_Insights_Project`.

### 2. Create a virtual environment

Open the VS Code terminal in the project root:

```powershell
py -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, either use Command Prompt:

```cmd
.venv\Scripts\activate.bat
```

or change the execution policy only if you are comfortable doing so.

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Start the app

```powershell
streamlit run streamlit_app.py
```

Streamlit will print a local address, usually `http://localhost:8501`.

## Rebuild the cleaned data and models

The project already includes generated artifacts, so this is optional. Run it if you change the raw CSV files or the modeling code:

```powershell
python scripts\rebuild_artifacts.py
```

This recreates:

- `data/processed/audible_cleaned.csv`
- `models/model_bundle.joblib`
- `models/model_metrics.json`

## Run tests

Install pytest if you want to run the project checks:

```powershell
pip install pytest
pytest -q
```

## Deploy to Streamlit Community Cloud

1. Push this project to a GitHub repository.
2. Open Streamlit Community Cloud and choose **Create app**.
3. Select your repository and branch.
4. Set the entry-point file to:

```text
streamlit_app.py
```

5. Deploy.

No API key or external service is required for this version.

## Recommendation methods

### Content similarity

The project creates a combined text field from:

- book title
- author
- cleaned genres
- description

TF-IDF converts that text into numeric vectors. Nearest-neighbour search with cosine distance returns books whose text features are most similar.

### Hybrid score

The app uses:

```text
0.72 × content similarity
+ 0.18 × rating quality
+ 0.10 × review popularity
```

This keeps content similarity as the main signal while avoiding recommendations that are only textually close but weakly rated.

### Clustering

TF-IDF vectors are reduced with Truncated SVD and grouped with K-Means. Several values of `k` are tried, and the project keeps the candidate with the best sampled silhouette score.

## Suggested GitHub files

Upload the full project except local environment and editor folders. In particular, keep:

- `streamlit_app.py`
- `src/`
- `scripts/`
- `data/raw/`
- `data/processed/`
- `models/`
- `assets/`
- `.streamlit/config.toml`
- `requirements.txt`
- `README.md`
- `PROJECT_NOTES.md`

Do **not** upload `.venv/`, `venv/`, `__pycache__/`, or `.vscode/`.

## Project workflow in one line

```text
Raw CSVs → cleaning/merge → feature engineering → TF-IDF → nearest neighbours + SVD/K-Means → hybrid ranking → Streamlit
```
