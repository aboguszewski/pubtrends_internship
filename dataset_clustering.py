import requests
import xml.etree.ElementTree as et
from bokeh.palettes import Paired
from bokeh.transform import linear_cmap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.embed import components
from bokeh.resources import CDN

# Below are some constants used throughout the module.
# Most of them are chosen arbitrarily, except N_COMPONENTS.
# This constant defines the number of dimensions to which the original tf-idf vectors are reduced.
# After testing, 100 seemed to be a reasonable number,
# because it accounted for 93% of variation present in the vectors calculated from the example list.
# Any bigger number resulted in diminishing returns.

SEED = 42
MIN_DF = 0.03
MAX_DF = 0.95
MAX_CLUSTERS = 10
N_COMPONENTS = 100
KMEANS_N_INIT = 10
TSNE_PERPLEXITY = 30


# Generate a cluster plot for the datasets linked to PMIDs listed in file_path.
# Return components for embedding the plot in HTML.
def generate_html_cluster_plot(file_path):
    plot = generate_cluster_plot(file_path)
    script, div = components(plot)
    cdn_jss = CDN.js_files[0]

    return script, div, cdn_jss


# Generate a cluster plot for the datasets linked to PMIDs listed in file_path.
# Return a Bokeh cluster plot.
def generate_cluster_plot(file_path):
    # Read all PMIDs from the input file.
    pmids = set()
    with open(file_path, 'r') as pmid_list:
        for line in pmid_list:
            pmid = int(line)
            pmids.add(pmid)

    # Get all the datasets' UIDs linked to given PMID.
    linked_datasets = get_linked_datasets(pmids)

    # Create a corpus dictionary (with all datasets' metadata).
    dataset_metadata = get_datasets_metadata(linked_datasets)

    # Build metadata text corpus for tf-idf.
    metadata_corpus = list(dataset_metadata.values())
    corpus_index_to_uid = list(dataset_metadata.keys())  # For conserving tf-idf vector -> UID -> PMID correspondence.

    # Calculate tf-idf, reduce dimensions and normalize.
    normalized_reduced_matrix = tf_idf(metadata_corpus)

    # Cluster resulting vectors.
    cluster_labels = cluster(normalized_reduced_matrix)

    # Reduce the vectors for plotting.
    x, y = reduce_to_2d(normalized_reduced_matrix)

    # Generate the plot.
    plot = figure(title="Datasets' Clusters",
                  x_axis_label='first t-SNE component',
                  y_axis_label='second t-SNE component')
    source = ColumnDataSource(data={
        'x': x,
        'y': y,
        'cluster': cluster_labels,
        'pmid': [linked_datasets[corpus_index_to_uid[index]] for index in range(len(metadata_corpus))]
    })
    colors = linear_cmap(field_name='cluster',
                         palette=Paired[max(max(cluster_labels), 3)],
                         low=min(cluster_labels),
                         high=max(cluster_labels))
    hover = HoverTool()
    hover.tooltips = [('PMID', '@pmid'), ('Cluster', '@cluster')]
    plot.add_tools(hover)
    plot.scatter(x='x', y='y', source=source, color=colors, size=10, alpha=0.7)

    return plot


# Retrieve UIDs of all the datasets in GEO database linked to given PMIDs.
# Return a dictionary with UID, PMID pairs.
def get_linked_datasets(pmids):
    linked_dataset = {}  # key: UID of the linked dataset, value: PMID
    elink_query_url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi'
    params = {
        'dbfrom': 'pubmed',
        'db': "gds",
        'linkname': 'pubmed_gds',
        'id': None,
        'retmode': 'xml'
    }

    for pmid in pmids:
        params['id'] = pmid
        response = requests.get(url=elink_query_url, params=params)
        if response.status_code != 200:
            continue  # Skip the PMIDs that don't get a response.

        # Parse XML response to find the UIDs.
        elink_result = et.fromstring(response.text)
        linked_uids = [int(id_element.text) for id_element in elink_result.findall('.//Link/Id')]

        for uid in linked_uids:
            linked_dataset[uid] = pmid

    return linked_dataset


# Retrieve 'Title', 'Organism', 'Summary', 'Experiment Type' and 'Overall Design' metadata fields for datasets.
# Return a dictionary with UID, its concatenated metadata as a string pairs.
def get_datasets_metadata(linked_datasets):
    datasets_metadata = {}  # key: dataset UID, value: metadata text (all fields combined)

    # Get accession key and 'Organism' field of the metadata for each dataset
    efetch_query_url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi'
    params = {
        'db': 'gds',
        'id': None
    }

    accession_key = {}  # key: dataset UID, value: accession key
    for uid in linked_datasets.keys():
        params['id'] = uid
        response = requests.get(url=efetch_query_url, params=params)
        if response.status_code != 200:
            continue  # Skip the UIDs that don't get a response.

        efetch_result = response.text.strip()
        lines = efetch_result.split('\n')
        organism_line = ''
        accession_line = ''
        for line in lines:
            if line.startswith('Organism:'):
                organism_line = line
            elif line.startswith('Series') or line.startswith('DataSet'):
                accession_line = line

        organism_name = ''
        for word in organism_line.split()[1:]:  # Organism name starts after 'Organism:'.
            organism_name += word + ' '

        accession = accession_line.split()[2]  # Accession key is the third word in the line.
        datasets_metadata[uid] = organism_name
        accession_key[uid] = accession

    # Use the accession key to get the rest of the metadata for each dataset.
    geo_accession_url = 'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi'
    params = {
        'form': 'xml',
        'acc': None
    }

    for uid in linked_datasets.keys():
        if uid not in accession_key.keys():  # Accession key is not available.
            _ = datasets_metadata.pop(uid)  # Drop UID, because retrieval of the majority of metadata failed.
            continue
        params['acc'] = accession_key[uid]
        response = requests.get(url=geo_accession_url, params=params)
        if response.status_code != 200:
            _ = datasets_metadata.pop(uid)  # Drop UID, because retrieval of the majority of metadata failed.

            continue  # Skip the UIDs that don't get a response.

        # Retrieve the chosen text fields from the metadata.
        try:
            accession_result = et.fromstring(response.text)
        except et.ParseError as error:  # The response wasn't an XML.
            _ = datasets_metadata.pop(uid)  # Drop UID, because retrieval of the majority of metadata failed.
            continue

        tag_prefix = '{http://www.ncbi.nlm.nih.gov/geo/info/MINiML}'
        title = get_text_from_xml(tag=tag_prefix + 'Title', xml=accession_result)
        summary = get_text_from_xml(tag=tag_prefix + 'Summary', xml=accession_result)
        overall_design = get_text_from_xml(tag=tag_prefix + 'Overall-Design', xml=accession_result)
        experiment_type = get_text_from_xml(tag=tag_prefix + 'Type', xml=accession_result)

        # Add retrieved text fields to dataset's metadata.
        datasets_metadata[uid] += title + ' ' + experiment_type + ' ' + summary + ' ' + overall_design

    return datasets_metadata


# Find the first occurrence of a tag in a xml document and its embedded text.
# Return the text stripped of whitespace (if the tag is not present in the document return None).
def get_text_from_xml(tag, xml):
    element = xml.find(f'.//{tag}')
    if element is not None and element.text is not None:
        return element.text.strip()
    return None


# Calculate tf-idf vectors, reduce dimensions and normalize them.
# Return resulting reduced and normalized tf-idf matrix.
def tf_idf(metadata_corpus):
    pipeline = make_pipeline(
        TfidfVectorizer(min_df=MIN_DF, max_df=MAX_DF),
        TruncatedSVD(n_components=N_COMPONENTS, random_state=SEED),
        Normalizer())

    return pipeline.fit_transform(metadata_corpus)


# Cluster tf-idf vectors using k-means algorithm.
# Return a list of cluster labels from the best clustering (labels are integers starting at 0).
def cluster(tf_idf_matrix):
    # Choose best k for k-means clustering.
    best_cluster_num = 2
    best_clustering_score = 0
    cluster_number_range = [n for n in range(2, min(MAX_CLUSTERS + 1, len(tf_idf_matrix)))]
    for n in cluster_number_range:
        cluster_labels = KMeans(
            n_clusters=n,
            n_init=KMEANS_N_INIT,
            random_state=SEED
        ).fit_predict(tf_idf_matrix)

        silhouette_average = silhouette_score(tf_idf_matrix, cluster_labels)
        if silhouette_average > best_clustering_score:
            best_cluster_num = n
            best_clustering_score = silhouette_average

    # Cluster.
    cluster_labels = KMeans(
        n_clusters=best_cluster_num,
        n_init=KMEANS_N_INIT,
        random_state=SEED
    ).fit_predict(tf_idf_matrix)
    return cluster_labels


# Reduce tf-idf vectors to 2 dimensions using t-SNE.
# Return lists x and y, where i-th reduced vector = [x[i], y[i]].
def reduce_to_2d(tf_idf_matrix):
    visualization_vectors = TSNE(
        n_components=2,
        perplexity=min(TSNE_PERPLEXITY, len(tf_idf_matrix) - 1),
        random_state=SEED
    ).fit_transform(tf_idf_matrix)
    x = []
    y = []
    for vector in visualization_vectors:
        x.append(vector[0])
        y.append(vector[1])
    return x, y
