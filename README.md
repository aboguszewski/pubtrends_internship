# Linked Datasets Clusterer

This web application allows users to upload a list of PubMed IDs (PMIDs) and clusters the metadata of linked GEO datasets. The resulting clusters are visualized as an interactive plot.

## Running the App

1. **Clone the repository** or download the files into a directory.

2. **Create a virtual environment** (optional):

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. **Install the required dependencies**:

```bash
pip install -r requirements.txt
```

4. **Start the Flask server**:

```bash
python app.py
```

5. **Open your browser** and go to:

```
http://127.0.0.1:5000
```

6. **Upload a `.txt` file** with a list of PMIDs (one per line) to start the clustering process.

## Requirements

Dependencies are listed in `requirements.txt`:
``` 
Flask==3.1.0
bokeh==3.7.2
scikit-learn==1.6.1
pandas==2.2.3
numpy==2.2.4
ratelimit==2.2.1
requests==2.32.3
```

## File Format

The input file should be a plain `.txt` file with one PMID per line, for example:
```    
30530648
31820734
31018141
35440059
```

## Demonstration

A sample file `PMIDs_list.txt` is included in the repository. It contains a list of valid PubMed IDs for testing the application. Due to request rate limits imposed by the PubMed and GEO APIs, generating the plot may take approximately two to three minutes.
