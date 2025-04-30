from enum import Enum
import logging
import math
import random
import string
import time
from types import MethodType
from typing import List, Optional, Tuple, Type, Union
import os

import numpy as np
import pandas as pd
from IPython.display import clear_output  # type: ignore
from sklearn.metrics import (  # type: ignore
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.utils import shuffle  # type: ignore
from sklearn.base import clone  # type: ignore

from embiggen.embedding_transformers import EdgePredictionTransformer  # type: ignore
from embiggen.utils import AbstractEdgeFeature  # type: ignore
from ensmallen import Graph  # type: ignore
from grape import EmbeddingResult  # type: ignore
from helper_lib import cache
from helper_lib.graph import build_triples_df, check_if_in_graph


def generate_negative_graph(
    positive_graph: Graph,
    train_size: float,
    testing_unbalance_rate: float = 1.0,
    training_unbalance_rate: float = 1.0,
    random_state: int = 42,
    pair_to_predict: Optional[Tuple[str, str]] = None,
    directed: bool = True,
    n_links_to_generate: Optional[int] = None,
) -> Graph:
    """
    Generate a negative graph for edge prediction.

    Parameters:
    - positive_graph: Positive graph to generate the negative graph from.
    - train_size: Proportion of the graph to use for training.
    - testing_unbalance_rate: Unbalance rate for the testing set.
    - training_unbalance_rate: Unbalance rate for the training set.
    - random_state: Random seed for reproducibility.
    - pair_to_predict: Tuple of source and destination node types to filter by.
    - directed: Whether to use count the number of edges of the graph as directed or not.
    - n_links_to_generate: Number of negative links to generate (before considering the unbalance rates).

    Returns:
    - Correctly-sized negative graph.
    """
    graph_to_consider = (
        positive_graph
        if pair_to_predict is None
        else positive_graph.filter_from_names(
            source_node_type_name_to_keep=[pair_to_predict[0]],
            destination_node_type_name_to_keep=[pair_to_predict[1]],
        )
    )

    if n_links_to_generate is None:
        n_positive_links = (
            graph_to_consider.get_number_of_directed_edges()
            if directed
            else graph_to_consider.get_number_of_edges()
        )
    else:
        n_positive_links = n_links_to_generate

    n_negative_links_training = int(
        n_positive_links * training_unbalance_rate * train_size
    )
    n_negative_links_testing = int(
        n_positive_links * testing_unbalance_rate * (1 - train_size)
    )
    n_negative_links = n_negative_links_training + n_negative_links_testing
    negative_graph = (
        positive_graph.sample_negative_graph(
            number_of_negative_samples=n_negative_links,
            random_state=random_state,
            use_scale_free_distribution=True,
            only_from_same_component=True,
            source_node_types_names=[pair_to_predict[0]],
            destination_node_types_names=[pair_to_predict[1]],
        )
        if pair_to_predict
        else positive_graph.sample_negative_graph(
            number_of_negative_samples=n_negative_links,
            random_state=random_state,
            use_scale_free_distribution=True,
            only_from_same_component=True,
        )
    )
    return negative_graph


def generate_holdouts(
    positive_graph: Graph,
    negative_graph: Graph,
    train_size: float,
    training_unbalance_rate: float = 1.0,
    number_of_holdouts: int = 10,
    seed: int = 42,
    pair_to_predict: Optional[Tuple[str, str]] = None,
) -> List[dict]:
    """
    Generate holdout sets for edge prediction.

    Parameters:
    - positive_graph: Graph object to split into positive training and testing sets.
    - negative_graph: Graph object to split into negative training and testing sets.
    - train_size: Proportion of the graph to use for training.
    - training_unbalance_rate: Unbalance rate for the training set, the testing unbalance rate will be worked out from this.
    - number_of_holdouts: Number of holdout sets to generate.
    - seed: Random seed for reproducibility.
    - pair_to_predict: Tuple of source and destination node types to filter by.

    Returns:
    - List of dictionaries containing the holdout sets.
        Each dictionary contains the following keys:
        - embedding_graph: Graph used to generate the embeddings.
        - positive_train_graph: Positive training graph, equal to 'embedding_graph' if pair_to_predict is not set.
        - positive_test_graph: Positive testing graph, used to evaluate the model.
        - negative_train_graph: Negative training graph, used to train the model.
        - negative_test_graph: Negative testing graph, used to evaluate the model.
    """
    holdouts = []
    # set seed for reproducibility
    random.seed(seed)

    for i in range(number_of_holdouts):
        holdout = {}
        # generate a random state for the connected holdout
        random_state = random.randrange(0, 100000)
        # use connected monte carlo to obtain a training set that has the same connectivity guarantees as full graph
        logging.info(f"Generating holdout {i+1}/{number_of_holdouts}")
        positive_train_graph, positive_test_graph = positive_graph.connected_holdout(
            train_size=train_size, random_state=random_state
        )
        # store the graph used to generate the embeddings, this is the graph without filtering to preserve the connectivity guarantees
        holdout["embedding_graph"] = positive_train_graph
        # if pair_to_predict is set, filter the graphs to only contain the edges we are interested in
        if pair_to_predict:
            logging.info("Filtering train graph by source/destination node type")
            positive_train_graph = positive_train_graph.filter_from_names(
                source_node_type_name_to_keep=[pair_to_predict[0]],
                destination_node_type_name_to_keep=[pair_to_predict[1]],
            )
            logging.info("Filtering test graph by source/destination node type")
            positive_test_graph = positive_test_graph.filter_from_names(
                source_node_type_name_to_keep=[pair_to_predict[0]],
                destination_node_type_name_to_keep=[pair_to_predict[1]],
            )
        # add the positive graphs to the holdout
        holdout["positive_train_graph"] = positive_train_graph
        holdout["positive_test_graph"] = positive_test_graph

        # no need for testing_unbalance_rate since the negative graph is already generated
        # by splitting it using the training_unbalance_rate we already obtain the desired unbalance rate for the test set

        # split the negative graph into train and test set
        n_negative_train_samples = int(
            positive_train_graph.get_number_of_directed_edges()
            * training_unbalance_rate
        )
        # Calculate the train_size parameter for the negative graph split
        negative_train_size = (
            n_negative_train_samples / negative_graph.get_number_of_directed_edges()
        )
        negative_train_graph, negative_test_graph = negative_graph.random_holdout(
            train_size=negative_train_size, random_state=random_state
        )

        # if pair_to_predict is set, filter the graphs to only contain the edges we are interested in
        if pair_to_predict:
            negative_train_graph = negative_train_graph.filter_from_names(
                source_node_type_name_to_keep=[pair_to_predict[0]],
                destination_node_type_name_to_keep=[pair_to_predict[1]],
            )
            negative_test_graph = negative_test_graph.filter_from_names(
                source_node_type_name_to_keep=[pair_to_predict[0]],
                destination_node_type_name_to_keep=[pair_to_predict[1]],
            )

        # add the negative graphs to the holdout
        holdout["negative_train_graph"] = negative_train_graph
        holdout["negative_test_graph"] = negative_test_graph

        holdouts.append(holdout)

    return holdouts


def generate_embedding(
    positive_training_graph: Graph,
    embedder,
    cache_embedding_externally: bool = False,
    holdout_number: int = 0,
) -> EmbeddingResult:
    """
    Generate an embedding for a graph using an embedder.

    Parameters:
    - positive_training_graph: Graph to generate the embedding for.
    - embedder: Embedder object to use for embedding generation.
    - cache_embedding_externally: Whether to cache the embedding externally.
    - holdout_number: Number of the holdout set.

    Returns:
    - Embedding for the graph.
    """
    # calculate the embedding on the train graph
    logging.info("Training embedding on the train graph")
    before = time.time()
    train_embedding = embedder.fit_transform(positive_training_graph)
    logging.info(f"Embedding time:{time.time()-before:.2f}")
    # the embedding could be cached to avoid recalculating it
    if cache_embedding_externally:
        # params = embedder.parameters()
        # params["holdout_number"] = holdout_number
        # cache.cache_embedding(
        #     train_embedding, f"{embedder.model_name()}_holdout{holdout_number}", params
        # )
        logging.info("Embedding caching is not implemented yet")
    return train_embedding


class EmbeddingCombinationMethod(Enum):
    Concatenate = "Concatenate"
    Hadamard = "Hadamard"

    def get(s: string):
        try:
            return EmbeddingCombinationMethod(s)
        except ValueError:
            raise ValueError(
                f"Unknown node_embedding_concatenation_method: {s}, must be one of {EmbeddingCombinationMethod.get_available_methods()}"
            )

    def combine(self, source_node_embedding, destination_node_embedding):
        if self.value == "Concatenate":
            return np.hstack((source_node_embedding, destination_node_embedding))
        elif self.value == "Hadamard":
            return source_node_embedding * destination_node_embedding
        else:
            raise ValueError(
                f"Unknown node_embedding_concatenation_method: {self.value}, must be one of {EmbeddingCombinationMethod.get_available_methods()}"
            )

    def get_available_methods() -> List[str]:
        return [method.value for method in EmbeddingCombinationMethod]


def graph_to_edge_embeddings(
    embedding: EmbeddingResult,
    graph: Graph,
    node_embeddings_concatenation_method: string = "Concatenate",
    treat_edges_as_bidirectional: bool = True,
) -> List[np.ndarray]:
    """
    Extract embeddings for the edges in a graph.

    Parameters:
    - embedding: Embedding object to extract the embeddings from.
    - graph: Graph object to extract the embeddings for.
    - node_embeddings_concatenation_method: Method to combine the node embeddings.
    - treat_edges_as_bidirectional: Whether to treat edges as bidirectional when the graph is undirected.

    Returns:
    - List of embeddings for the edges in the graph.
    """
    node_embeddings_concatenation_method = EmbeddingCombinationMethod.get(
        node_embeddings_concatenation_method
    )
    before = time.time()
    # Extract the embeddings for the nodes
    node_embedding = embedding.get_all_node_embedding()
    # Get source and destination node names for each edge in the graph
    if graph.is_directed():
        logging.info("Graph is directed, extracting directed edge embeddings")
        edge_node_names = np.array(graph.get_edge_node_names(directed=True))
    else:
        logging.info(
            f"Graph is undirected, extracting {'directed' if treat_edges_as_bidirectional else 'undirected'} edge embeddings"
        )
        edge_node_names = np.array(
            graph.get_edge_node_names(directed=treat_edges_as_bidirectional)
        )

    logging.info(f"Number of edges: {len(edge_node_names)}")

    # Preallocate memory for the embeddings
    embeddings = np.empty((len(edge_node_names), node_embedding[0].shape[1] * 2))

    # Use vectorized operations to extract and concatenate embeddings
    source_embeddings = node_embedding[0].loc[edge_node_names[:, 0]].values
    destination_embeddings = node_embedding[0].loc[edge_node_names[:, 1]].values

    embeddings = node_embeddings_concatenation_method.combine(
        source_embeddings, destination_embeddings
    )

    logging.info(f"Embedding extraction time: {time.time() - before:.2f}")
    return embeddings


def train_sklearn_model(
    model,
    embedding: EmbeddingResult,
    positive_train_graph: Graph | np.ndarray,
    negative_train_graph: Graph | np.ndarray,
    random_state: int = 42,
    node_embeddings_concatenation_method: string = "Concatenate",
    treat_edges_as_bidirectional: bool = True,
) -> Tuple:
    """
    Train a model using positive and negative training data.

    Parameters:
    - model: Model object to train.
    - embedding: Embedding used to train the model.
    - positive_train_graph: Positive training graph.
    - negative_train_graph: Negative training graph.
    - random_state: Random seed for reproducibility.
    - treat_edges_as_bidirectional: Whether to consider both directions of an edge in an undirected graph for the generation of train/test sets.

    Returns:
    - Trained model.
    - Tuple of positive precision score, negative precision score, AUC score, AUC-PRA score and training set sizes.
    """
    # extract embeddings for the positive and negative training data
    if isinstance(positive_train_graph, Graph):
        positive_train_X = graph_to_edge_embeddings(
            embedding,
            positive_train_graph,
            node_embeddings_concatenation_method=node_embeddings_concatenation_method,
            treat_edges_as_bidirectional=treat_edges_as_bidirectional,
        )
        negative_train_X = graph_to_edge_embeddings(
            embedding,
            negative_train_graph,
            node_embeddings_concatenation_method=node_embeddings_concatenation_method,
            treat_edges_as_bidirectional=treat_edges_as_bidirectional,
        )
    elif isinstance(positive_train_graph, np.ndarray):
        positive_train_X = positive_train_graph
        negative_train_X = negative_train_graph
    else:
        raise ValueError(
            "positive_train_graph and negative_train_graph must be either Graph or np.ndarray"
        )

    # combine positive and negative training data
    positive_train_X = np.array(positive_train_X)
    negative_train_X = np.array(negative_train_X)
    X = np.concatenate([positive_train_X, negative_train_X])
    # create labels for the training data
    y = np.concatenate(
        [np.ones(len(positive_train_X)), np.zeros(len(negative_train_X))]
    )
    # shuffle the data
    X, y = shuffle(X, y, random_state=random_state)

    # train the model
    model.fit(X, y)
    # calculate the Accuracy score for the positive training data
    pos_precision_score = accuracy_score(
        np.ones(len(positive_train_X)), model.predict(positive_train_X)
    )
    # calculate the Accuracy score for the negative training data
    neg_precision_score = accuracy_score(
        np.zeros(len(negative_train_X)), model.predict(negative_train_X)
    )

    logging.info(f"Predicting on overall training set of size {len(X)}")
    y_score = model.predict_proba(X)[:, 1]
    # calculate the overall AUC score
    auc_score = roc_auc_score(y, y_score)
    # calculate the overall AUC-PRA score
    auc_pr_score = average_precision_score(y, y_score)

    return model, {
        "training_positive_set_size": len(positive_train_X),
        "training_negative_set_size": len(negative_train_X),
        "training_set_size": len(X),
        "training_positive_accuracy_score": pos_precision_score,
        "training_negative_accuracy_score": neg_precision_score,
        "training_auc_score": auc_score,
        "training_auc_pr_score": auc_pr_score,
    }


def test_sklearn_model(
    model,
    train_embedding: EmbeddingResult,
    positive_test_graph: Graph | np.ndarray,
    negative_test_graph: Graph | np.ndarray,
    node_embeddings_concatenation_method: string = "Concatenate",
    treat_edges_as_bidirectional: bool = True,
) -> Tuple:
    """
    Test a model on positive and negative test data.

    Parameters:
    - model: Model object to test.
    - train_embedding: Embedding used to train the model.
    - positive_test_graph: Positive test graph.
    - negative_test_graph: Negative test graph.
    - node_embedding_concatenation_method: Method to combine the node embeddings.
    - treat_edges_as_bidirectional: Whether to consider both directions of an edge in an undirected graph for the generation of train/test sets.

    Returns:
    - Dictionary containing the positive accuracy score, negative accuracy score, AUC score, AUC-PRA score and test set sizes.
    - Dictionary containing the true labels, predicted labels and predicted scores.
    """
    if isinstance(positive_test_graph, Graph):
        X_test_pos = graph_to_edge_embeddings(
            train_embedding,
            positive_test_graph,
            node_embeddings_concatenation_method=node_embeddings_concatenation_method,
            treat_edges_as_bidirectional=treat_edges_as_bidirectional,
        )
        X_test_neg = graph_to_edge_embeddings(
            train_embedding,
            negative_test_graph,
            node_embeddings_concatenation_method=node_embeddings_concatenation_method,
            treat_edges_as_bidirectional=treat_edges_as_bidirectional,
        )
    elif isinstance(positive_test_graph, np.ndarray):
        X_test_pos = positive_test_graph
        X_test_neg = negative_test_graph
    else:
        raise ValueError(
            "positive_test_graph and negative_test_graph must be either Graph or np.ndarray"
        )

    logging.info(f"Predicting on positive test set of size {len(X_test_pos)}")
    pred_test_pos = model.predict(X_test_pos)
    logging.info(f"Predicting on negative test set of size {len(X_test_neg)}")
    pred_test_neg = model.predict(X_test_neg)

    # calculate the Accuracy score for the positive training data
    pos_accuracy_score = accuracy_score(np.ones(len(X_test_pos)), pred_test_pos)
    # calculate the Accuracy score for the negative training data
    neg_accuracy_score = accuracy_score(np.zeros(len(X_test_neg)), pred_test_neg)

    # combine positive and negative test data
    X = np.concatenate([X_test_pos, X_test_neg])
    y = np.concatenate([np.ones(len(X_test_pos)), np.zeros(len(X_test_neg))])

    logging.info(f"Predicting on overall test set of size {len(X)}")
    y_pred = model.predict(X)
    y_score = model.predict_proba(X)[:, 1]

    # calculate the overall AUC score
    auc_score = roc_auc_score(y, y_score)
    # calculate the overall AUC-PRA score
    auc_pra_score = average_precision_score(y, y_score)

    return (
        {
            "testing_positive_set_size": len(X_test_pos),
            "testing_negative_set_size": len(X_test_neg),
            "testing_set_size": len(X),
            "testing_positive_accuracy_score": pos_accuracy_score,
            "testing_negative_accuracy_score": neg_accuracy_score,
            "testing_auc_score": auc_score,
            "testing_auc_pra_score": auc_pra_score,
        },
        {
            "y_true": y,
            "y_pred": y_pred,
            "y_score": y_score,
        },
    )


def edge_prediction_pipeline_sklearn(
    graph: Graph,
    model,
    embedder,
    pair_to_predict: Optional[Tuple[str, str]] = None,
    train_size: float = 0.7,
    number_of_holdouts: int = 5,
    seed: int = 42,
    testing_unbalance_rate: float = 1.0,
    training_unbalance_rate: Optional[float] = None,
    cache_embedding_externally: bool = True,
    graph_to_generate_negatives_from: Optional[
        Graph
    ] = None,  # This could be removed if we move to the new holdout generation method
    node_embeddings_concatenation_method: str = "Concatenate",
    treat_edges_as_bidirectional: bool = True,
) -> Tuple[List[dict], pd.DataFrame]:
    """
    Pipeline for edge prediction using sklearn models.

    Parameters:
    - graph: Graph object to generate holdouts from.
    - model: Model object to use for edge prediction.
    - embedder: Embedder object to use for embedding generation.
    - pair_to_predict: Tuple of source and destination node types to filter by.
    - train_size: Proportion of the graph to use for training.
    - number_of_holdouts: Number of holdout sets to generate.
    - seed: Random seed for reproducibility.
    - testing_unbalance_rate: Unbalance rate for the testing set.
    - training_unbalance_rate: Unbalance rate for the training set.
    - cache_embedding_externally: Whether to cache the embedding externally.
    - graph_to_generate_negatives_from: Graph object to use to generate the negative samples, if not set, the negative samples will be generated from the graph itself.
    - treat_edges_as_bidirectional: Whether to consider both directions of an edge in an undirected graph for the generation of train/test sets.

    Returns:
    - List of holdouts.
    - Dataframe containing the results of the pipeline.
    """
    # generating holdouts
    before = time.time()
    negative_graph = generate_negative_graph(
        positive_graph=graph,
        train_size=train_size,
        testing_unbalance_rate=testing_unbalance_rate,
        training_unbalance_rate=training_unbalance_rate,
        random_state=seed,
        pair_to_predict=pair_to_predict,
    )
    holdouts = generate_holdouts(
        positive_graph=graph,
        negative_graph=negative_graph,
        train_size=train_size,
        training_unbalance_rate=training_unbalance_rate,
        number_of_holdouts=number_of_holdouts,
        seed=seed,
        pair_to_predict=pair_to_predict,
    )

    holdouts_generation_time = time.time() - before
    logging.info(f"Holdout generation time: {holdouts_generation_time:.2f}")

    # create dataframe to store the results
    results_df = pd.DataFrame(
        columns=[
            "graph",
            "embedder",
            "model",
            "source_node_type",
            "destination_node_type",
            "holdout_number",
            "training_positive_set_size",
            "training_negative_set_size",
            "training_set_size",
            "testing_positive_set_size",
            "testing_negative_set_size",
            "testing_set_size",
            "training_positive_accuracy_score",
            "training_negative_accuracy_score",
            "training_auc_score",
            "training_auc_pra_score",
            "testing_positive_accuracy_score",
            "testing_negative_accuracy_score",
            "testing_auc_score",
            "testing_auc_pra_score",
            "model_parameters",
            "embedder_parameters",
            "experiment_parameters",
            "times",
        ]
    )

    model_parameters = model.get_params()
    embedder_parameters = embedder.parameters()
    experiment_parameters = {
        "pair_to_predict": pair_to_predict,
        "train_size": train_size,
        "number_of_holdouts": number_of_holdouts,
        "seed": seed,
        "testing_unbalance_rate": testing_unbalance_rate,
        "training_unbalance_rate": training_unbalance_rate,
        "cache_embedding_externally": cache_embedding_externally,
        "graph_to_generate_negatives_from": (
            graph_to_generate_negatives_from.get_name()
            if graph_to_generate_negatives_from
            else None
        ),
        "node_embeddings_concatenation_method": node_embeddings_concatenation_method,
        "treat_edges_as_bidirectional": treat_edges_as_bidirectional,
    }

    for i, holdout in enumerate(holdouts):
        times = {
            "holdout_generation_time": holdouts_generation_time,
        }
        logging.info(f"Holdout {i + 1}")
        # calculate the embedding on the not filtered train graph
        before = time.time()
        holdout["embedding"] = generate_embedding(
            holdout["embedding_graph"],
            embedder,
            cache_embedding_externally=cache_embedding_externally,
            holdout_number=i,
        )
        times["embedding_generation_time"] = time.time() - before
        logging.info(
            f"Embedding generation time: {times['embedding_generation_time']:.2f}"
        )

        # train the model
        # we clone the model to avoid overwriting the model for each holdout
        before = time.time()
        holdout_model = clone(model)

        # train the model
        model, train_metrics = train_sklearn_model(
            holdout_model,
            holdout["embedding"],
            holdout["positive_train_graph"],
            holdout["negative_train_graph"],
            random_state=seed,
            node_embeddings_concatenation_method=node_embeddings_concatenation_method,
            treat_edges_as_bidirectional=treat_edges_as_bidirectional,
        )

        logging.info(train_metrics)
        holdout["model"] = holdout_model
        holdout["train_metrics"] = train_metrics
        times["training_time"] = time.time() - before
        logging.info(f"Training time: {times['training_time']:.2f}")

        # test the model
        before = time.time()
        test_metrics, test_predictions = test_sklearn_model(
            holdout_model,
            holdout["embedding"],
            holdout["positive_test_graph"],
            holdout["negative_test_graph"],
            node_embeddings_concatenation_method=node_embeddings_concatenation_method,
            treat_edges_as_bidirectional=treat_edges_as_bidirectional,
        )
        logging.info(test_metrics)
        holdout["test_metrics"] = test_metrics
        holdout["test_predictions"] = test_predictions

        times["testing_time"] = time.time() - before
        logging.info(f"Testing time: {times['testing_time']:.2f}")

        results_df.loc[i] = [
            graph.get_name(),
            embedder.model_name(),
            model.__class__.__name__,
            pair_to_predict[0] if pair_to_predict else None,
            pair_to_predict[1] if pair_to_predict else None,
            i,
            train_metrics["training_positive_set_size"],
            train_metrics["training_negative_set_size"],
            train_metrics["training_set_size"],
            test_metrics["testing_positive_set_size"],
            test_metrics["testing_negative_set_size"],
            test_metrics["testing_set_size"],
            train_metrics["training_positive_accuracy_score"],
            train_metrics["training_negative_accuracy_score"],
            train_metrics["training_auc_score"],
            train_metrics["training_auc_pra_score"],
            test_metrics["testing_positive_accuracy_score"],
            test_metrics["testing_negative_accuracy_score"],
            test_metrics["testing_auc_score"],
            test_metrics["testing_auc_pra_score"],
            model_parameters,
            embedder_parameters,
            experiment_parameters,
            times,
        ]
        # predict values for negative test set
        negative_test_pred = pd.DataFrame(
            holdout["negative_test_graph"].get_edge_node_names(
                directed=treat_edges_as_bidirectional
            ),
            columns=["source", "destination"],
        )

        negative_test_pred["scores"] = (
            holdout["test_predictions"]["y_score"][
                holdout["positive_test_graph"].get_number_of_directed_edges() :
            ]
            if treat_edges_as_bidirectional
            else holdout["test_predictions"]["y_score"][
                holdout["positive_test_graph"].get_number_of_edges() :
            ]
        )
        holdout["negative_test_predictions"] = negative_test_pred
    return (holdouts, results_df)


def calculate_mask(src_node_names, dst_node_names, edges_to_filter):
    """
    Calculate a mask to filter out a set of edges.

    Parameters:
    - src_node_names: Source node names.
    - dst_node_names: Destination node names.
    - edges_to_filter: Set of training edges.
    """

    src_node_names_repeated = np.repeat(src_node_names, len(dst_node_names))
    dst_node_names_tiled = np.tile(dst_node_names, len(src_node_names))
    mask = np.array(
        [
            (src, dst) in edges_to_filter
            for src, dst in zip(src_node_names_repeated, dst_node_names_tiled)
        ]
    )
    return mask


def estimate_chunk_ram_size(chunk_size):
    """
    Estimate the amount of RAM needed to store a chunk of predictions.

    Parameters:
    - chunk_size: Number of predictions in the chunk.

    Returns:
    - Estimated RAM size in MB NOTE: The value is slightly underestimated because it does not consider the index.
    """
    # chunk_size * float64(prediction score) * 2 (a,b and b,a) + chunk_size * bool * 2 (train and positive masks)

    return ((chunk_size * 8 * 2) + (chunk_size * 1 * 2)) / 1024**2


def save_chunk(
    forward_scores,
    reverse_scores,
    train_mask,
    positive_mask,
    current_chunk,
    filename,
    path,
):
    """
    Save a chunk of predictions to a parquet file.

    Parameters:
    - forward_scores: Scores for the forward edges.
    - reverse_scores: Scores for the reverse edges.
    - train_mask: Mask to filter out the training edges.
    - positive_mask: Mask to filter out the positive edges.
    - current_chunk: Current chunk number.
    - filename: Filename for the chunks.
    - path: Path to store the chunks.

    Returns:
    - Filename for the saved chunk.
    """
    chunk = pd.DataFrame(
        {
            "forward_score": forward_scores,
            "reverse_score": reverse_scores,
            "is_train": train_mask,
            "is_positive": positive_mask,
        }
    )
    filename = f"{filename}_{current_chunk}.parquet"
    os.makedirs(path, exist_ok=True)
    chunk.to_parquet(
        f"{path}{filename}",
        compression="brotli",
    )
    return f"{path}{filename}"


def predict_all_edges(
    embedding_name,
    model,
    node_type_pair,
    max_chunk_size=500_000_000,
    chunks_filename="predictions",
    chunks_path="predictions/",
    holdout_path="holdouts/",
):
    """
    Predict all edges between two node types using the trained model. The predictions are saved in parquet chunks to avoid memory issues.
    The predictions are saved in the following format:
    - forward_scores: Scores for the forward edges.
    - reverse_scores: Scores for the reverse edges.
    - train_mask: Mask to filter out the training edges.
    - positive_mask: Mask to filter out the positive edges.
    It's important to keep track of the edge order to be able to reconstruct the predictions later.

    Parameters:
    - embedding_name: Name of the embedding to use.
    - model: Model object to use for edge prediction.
    - node_type_pair: Tuple of source and destination node types to predict edges between.
    - max_chunk_size: Maximum number of predictions to store in a single chunk.
    - chunks_filename: Filename for the chunks.
    - chunks_path: Path to store the chunks.
    - holdout_path: Path to load the holdouts from.

    Returns:
    - List of filenames for the saved chunks.
    """
    # Load node lists
    type_1_nodes = pd.read_parquet(
        f"{holdout_path}{embedding_name}/_nodes/{node_type_pair[0]}_nodes_embeddings.parquet"
    )
    type_2_nodes = pd.read_parquet(
        f"{holdout_path}{embedding_name}/_nodes/{node_type_pair[1]}_nodes_embeddings.parquet"
    )

    n_src_nodes = len(type_1_nodes)
    n_dst_nodes = len(type_2_nodes)
    total_combinations = n_src_nodes * n_dst_nodes

    need_chunking = total_combinations >= max_chunk_size

    # Pre-allocate numpy arrays
    chunk_size = (
        max_chunk_size // n_dst_nodes * n_dst_nodes
        if need_chunking
        else total_combinations
    )
    forward_scores = np.zeros(chunk_size)
    reverse_scores = np.zeros(chunk_size)
    # n_chunks = total_combinations // chunk_size + 1

    # Setting up to build the training mask
    # Load the training edges dataframe to make sure we are not testing on training data
    training_edges = pd.read_parquet(
        f"{holdout_path}/filters/{node_type_pair[0]}-{node_type_pair[1]}_train_edges_names.parquet"
    )
    training_edges_set = set(tuple(x) for x in training_edges.to_numpy())

    # Setting up to build the positive mask
    positive_edges = pd.read_parquet(
        f"{holdout_path}/filters/{node_type_pair[0]}-{node_type_pair[1]}_positive_edges_names.parquet"
    )
    positive_edges_set = set(tuple(x) for x in positive_edges.to_numpy())

    if not need_chunking:
        # Generate the training mask to filter out the training edges
        # in one go, should be faster than doing it in chunks

        train_mask = calculate_mask(
            type_1_nodes.index.values, type_2_nodes.index.values, training_edges_set
        )
        positive_mask = calculate_mask(
            type_1_nodes.index.values, type_2_nodes.index.values, positive_edges_set
        )

    i = 0
    total_processed = 0
    current_chunk = 0
    chunk_filenames = []
    for _, src_node_embedding in type_1_nodes.iterrows():
        if need_chunking and i + n_dst_nodes > chunk_size:
            type_1_nodes_chunk = type_1_nodes.index[
                total_processed // n_dst_nodes : (total_processed + i) // n_dst_nodes
            ]
            type_2_nodes_chunk = type_2_nodes.index.values
            # Calculate the train_mask for this chunk
            train_mask = calculate_mask(
                type_1_nodes_chunk,
                type_2_nodes_chunk,
                training_edges_set,
            )
            # Calculate the positive mask for this chunk
            positive_mask = calculate_mask(
                type_1_nodes_chunk,
                type_2_nodes_chunk,
                positive_edges_set,
            )

            # Store the results in a dataframe
            chunk_filename = save_chunk(
                forward_scores,
                reverse_scores,
                train_mask,
                positive_mask,
                current_chunk,
                chunks_filename,
                chunks_path,
            )
            chunk_filenames.append(chunk_filename)

            # Prepare for the next chunk
            total_processed += i
            i = 0
            current_chunk += 1
            # if chunk_i == n_chunks - 1:
            #     # Last chunk, adjust the size
            #     chunk_size = total_combinations - tot_i
            forward_scores = np.zeros(chunk_size)
            reverse_scores = np.zeros(chunk_size)

        src_node_embedding = src_node_embedding.to_numpy()
        repeated_src_node_embedding = np.tile(src_node_embedding, (n_dst_nodes, 1))

        X_forward = np.hstack([repeated_src_node_embedding, type_2_nodes.values])
        X_reverse = np.hstack([type_2_nodes.values, repeated_src_node_embedding])

        forward_scores[i : i + n_dst_nodes] = model.predict_proba(X_forward)[:, 1]
        reverse_scores[i : i + n_dst_nodes] = model.predict_proba(X_reverse)[:, 1]

        i += n_dst_nodes

    if need_chunking:
        train_mask = calculate_mask(
            type_1_nodes.index[total_processed // n_dst_nodes :],
            type_2_nodes.index.values,
            training_edges_set,
        )
        positive_mask = calculate_mask(
            type_1_nodes.index[total_processed // n_dst_nodes :],
            type_2_nodes.index.values,
            positive_edges_set,
        )
    # Move the last chunk to disk
    chunk_filename = save_chunk(
        forward_scores[:i],
        reverse_scores[:i],
        train_mask,
        positive_mask,
        current_chunk,
        chunks_filename,
        chunks_path,
    )
    chunk_filenames.append(chunk_filename)
    return chunk_filenames


def edge_prediction_pipeline(
    graph,
    model,
    embedder,
    pair_to_predict=None,
    train_on_filtered=True,
    train_size=0.7,
    number_of_holdouts=5,
    seed=42,
    verbose=False,
    clear_output_holdout=True,
    use_scale_free_distribution=True,
    testing_unbalance_rate=1.0,
    cache_embedding_externally=True,
):
    random.seed(seed)

    results = []

    predictions = []
    models = []

    for i in range(number_of_holdouts):
        # clean the cell output at each iteration to avoid huge cell outputs
        if clear_output_holdout:
            clear_output(wait=True)
        # use connected monte carlo to obtain a training set that has the same connectivity guarantees as full graph
        before = time.time()
        logging.info(f"Generating holdout {i+1}/{number_of_holdouts}")
        random_state = random.randrange(0, 100000)
        train_graph, positive_test_graph = graph.connected_holdout(
            train_size=train_size, random_state=random_state
        )
        logging.info(f"Training graph size: {train_graph.get_number_of_edges()}")
        logging.info(
            f"Positive test graph size: {positive_test_graph.get_number_of_edges()}"
        )

        # check if number of connected components is the same in the training set and full graph
        logging.debug(train_graph.get_number_of_connected_components())
        assert (
            train_graph.get_number_of_connected_components()
            == graph.get_number_of_connected_components()
        )

        logging.info("Filtering train and test graph by source/destination node type")
        # keep only the edges (source-destination node type) we are interested in
        train_graph_filtered = (
            train_graph.filter_from_names(
                source_node_type_name_to_keep=[pair_to_predict[0]],
                destination_node_type_name_to_keep=[pair_to_predict[1]],
            )
            if pair_to_predict
            else train_graph
        )
        test_graph_filtered = (
            positive_test_graph.filter_from_names(
                source_node_type_name_to_keep=[pair_to_predict[0]],
                destination_node_type_name_to_keep=[pair_to_predict[1]],
            )
            if pair_to_predict
            else positive_test_graph
        )
        logging.info(f"Generating positive holdout: {time.time()-before}")

        if verbose:
            df_train = build_triples_df(train_graph)
            # df_test = build_triples_df(positive_test_graph)
            df_train_filtered = build_triples_df(train_graph_filtered)
            # df_test_filtered = build_triples_df(test_graph_filtered)
            logging.info(f"Positive training set size: {len(df_train)}")
            logging.info(
                f"Positive training set FILTERED size: {len(df_train_filtered)}"
            )
            # print relative frequencies of top 10 source_type - destination_type in train and test set compared to full graph
            # for i,(key,count) in enumerate(df_view0['complete_label'].value_counts().items()):
            #   logging.info(f"{key}: {df_train['complete_label'].value_counts()[key]/count} - {df_test['complete_label'].value_counts()[key]/count}")
            #   if i == 9:break

        # calculate the embedding on the not filtered train graph
        logging.info("Training embedding on unfiltered train graph")
        before = time.time()
        train_embedding = embedder.fit_transform(train_graph)
        logging.info(f"Embedding time:{time.time()-before}")
        # the embedding could be cached to avoid recalculating it
        if cache_embedding_externally:
            params = embedder.parameters()
            params["number_of_holdouts"] = number_of_holdouts
            params["train_size"] = train_size

            cache.cache_embedding(
                train_embedding, f"{embedder.model_name()}_holdout{i}", params
            )

        before = time.time()
        logging.info("Training model using the filtered train graph")
        # train the model using the node embeddings and with the filtered train graph => train only on miRNA-Disease edges
        # the edge prediction models from grape generate their own negative edges for training
        model.fit(
            graph=train_graph_filtered if train_on_filtered else train_graph,
            node_features=train_embedding,
            support=train_graph,
        )

        logging.info("Evaluating model on positive train set")
        train_pred = model.predict_proba(
            graph=train_graph_filtered if train_on_filtered else train_graph,
            node_features=train_embedding,
            return_predictions_dataframe=True,
            support=train_graph,
        )

        training_set_size = len(train_pred)
        print(f"Positive Training set size: {training_set_size}")
        print(
            f"Graph filtered undirected edges: {train_graph_filtered.get_number_of_edges()}"
        )
        print(
            f"Graph filtered directed edges: {train_graph_filtered.get_number_of_directed_edges()}"
        )
        print(f"Graph undirected edges: {train_graph.get_number_of_edges()}")
        print(f"Graph directed edges: {train_graph.get_number_of_directed_edges()}")

        pos_train_score = balanced_accuracy_score(
            [True for _ in range(len(train_pred))],
            train_pred["prediction"].apply(lambda x: x > 0.5),
        )
        if verbose:
            # pred_train_edge_presence = train_pred.apply(lambda row:check_if_in_graph(graph,row['sources'],row['destinations'],pair_to_predict),axis=1)
            # train_score = balanced_accuracy_score(pred_train_edge_presence, train_pred['prediction'].apply(lambda x:x>0.5))
            logging.info(
                f"Balanced accuracy positive score TRAINING: {pos_train_score}"
            )
        logging.info(f"Training time: {time.time()-before:.2f}")

        logging.info("Creating a graph with the negative edges for testing")
        # create graph with negative edges for testing
        negative_test_graph = (
            graph.sample_negative_graph(
                # number_of_negative_samples=test_graph_filtered.get_number_of_edges(), # this option creates only half the edges
                number_of_negative_samples=int(
                    test_graph_filtered.get_number_of_directed_edges()
                    * testing_unbalance_rate
                ),
                source_node_types_names=[pair_to_predict[0]],
                destination_node_types_names=[pair_to_predict[1]],
                random_state=random_state,
                use_scale_free_distribution=use_scale_free_distribution,
            )
            if pair_to_predict
            else graph.sample_negative_graph(
                # number_of_negative_samples=test_graph_filtered.get_number_of_edges(), # this option creates only half the edges
                number_of_negative_samples=int(
                    test_graph_filtered.get_number_of_directed_edges()
                    * testing_unbalance_rate
                ),
                random_state=random_state,
                use_scale_free_distribution=use_scale_free_distribution,
            )
        )

        if verbose:
            logging.info("Positive test set:")
            df_test_positive = build_triples_df(test_graph_filtered)
            logging.info(df_test_positive["complete_label"].value_counts())
            logging.info("Negative test set:")
            df_test_negative = build_triples_df(negative_test_graph)
            logging.info(df_test_negative["complete_label"].value_counts())

        if verbose:
            logging.info(
                f"#edges in positive test graph: {test_graph_filtered.get_number_of_directed_edges()}"
            )
            logging.info(
                f"#edges in negative test graph: {negative_test_graph.get_number_of_directed_edges()}"
            )
        logging.info(
            f"#edges in positive test graph: {test_graph_filtered.get_number_of_directed_edges()}"
        )
        logging.info(
            f"#edges in negative test graph: {negative_test_graph.get_number_of_directed_edges()}"
        )

        before = time.time()
        # use model to predict on the positive edges
        logging.info("Using the model to predict the existence of positive edges")
        pos_pred = model.predict_proba(
            graph=test_graph_filtered,
            node_features=train_embedding,
            return_predictions_dataframe=True,
            support=train_graph,
        )

        if verbose:
            # check if all edges of positive test set are in the original graph
            pos_pred_edge_presence = pos_pred.apply(
                lambda row: check_if_in_graph(
                    graph, row["sources"], row["destinations"], pair_to_predict
                ),
                axis=1,
            )
            logging.info(
                f"Are all positive edges present in the positive test set also in the original graph? {pos_pred_edge_presence.all()}"
            )
            logging.info(pos_pred_edge_presence.value_counts())

        # use model to predict on the negative edges
        logging.info("Using the model to predict the non-existence of negative edges")
        neg_pred = model.predict_proba(
            graph=negative_test_graph,
            node_features=train_embedding,
            return_predictions_dataframe=True,
            support=train_graph,
        )

        testing_set_size = len(pos_pred) + len(neg_pred)

        if verbose:
            # check if all edges of negative test set are not in the original graph
            neg_pred_edge_presence = neg_pred.apply(
                lambda row: check_if_in_graph(
                    graph, row["sources"], row["destinations"], pair_to_predict
                ),
                axis=1,
            )
            logging.info(
                f"Are all negative edges present in the negative test set NOT in the original graph? {~neg_pred_edge_presence.all()}"
            )
            logging.info(neg_pred_edge_presence.value_counts())

        # calculate balanced accuracy score for positive and negative predictions
        pos_score = balanced_accuracy_score(
            [True for _ in range(len(pos_pred))],
            pos_pred["prediction"].apply(lambda x: x > 0.5),
        )
        neg_score = balanced_accuracy_score(
            [False for _ in range(len(neg_pred))],
            neg_pred["prediction"].apply(lambda x: x > 0.5),
        )
        logging.info(f"Balanced accuracy positive score: {pos_score}")
        logging.info(f"Balanced accuracy negative score: {neg_score}")
        avg_score = (pos_score + neg_score) / 2
        logging.info(f"Balanced accuracy mean score: {avg_score}")

        auc_score = roc_auc_score(
            [True for _ in range(len(pos_pred))]
            + [False for _ in range(len(neg_pred))],
            pd.concat(
                [
                    pos_pred["prediction"],
                    neg_pred["prediction"],
                ]
            ),
        )
        logging.info(f"AUC score: {auc_score}")

        logging.info(f"Testing time: {time.time()-before:.2f}")

        predictions.append({"positive": pos_pred, "negative": neg_pred})
        models.append(model)

        results.append(
            (
                graph.get_name(),
                embedder.model_name(),
                model.model_name(),
                pair_to_predict[0] if pair_to_predict else None,
                pair_to_predict[1] if pair_to_predict else None,
                train_on_filtered,
                training_set_size,
                testing_set_size,
                pos_train_score,
                pos_score,
                neg_score,
                avg_score,
                auc_score,
            )
        )
    return results, predictions, models


def _new_fit(
    self,
    graph: Graph,
    support: Optional[Graph] = None,
    node_features: Optional[List[np.ndarray]] = None,
    node_type_features: Optional[List[np.ndarray]] = None,
    edge_type_features: Optional[List[np.ndarray]] = None,
    edge_features: Optional[  # type: ignore
        Union[Type[AbstractEdgeFeature], List[Type[AbstractEdgeFeature]]]
    ] = None,
):
    logging.debug("UPDATED _FIT")
    lpt = EdgePredictionTransformer(
        methods=self._edge_embedding_methods,
        aligned_mapping=True,
        include_both_undirected_edges=False,
    )
    lpt.fit(
        node_features,
        node_type_feature=node_type_features,
        edge_type_features=edge_type_features,
    )

    if support is None:
        support = graph

    if edge_features is None:
        edge_features: List[Type[AbstractEdgeFeature]] = []

    for edge_feature in edge_features:  # type: ignore
        if not issubclass(type(edge_feature), AbstractEdgeFeature):
            raise NotImplementedError(
                f"Edge features of type {type(edge_feature)} are not supported."
                "We currently only support edge features of type AbstractEdgeFeature."
            )

    number_of_negative_samples = int(
        math.ceil(graph.get_number_of_directed_edges() * self._training_unbalance_rate)
    )
    logging.info(graph.get_number_of_directed_edges())
    logging.info(f"{number_of_negative_samples} negative training samples")

    # custom code
    if self._original_graph:
        # generate the negative graph from the original graph instead of the negative graph
        negative_graph = self._original_graph.sample_negative_graph(
            number_of_negative_samples=number_of_negative_samples,
            only_from_same_component=True,
            random_state=self._random_state,
            use_scale_free_distribution=self._use_scale_free_distribution,
            sample_edge_types=len(edge_type_features) > 0,  # type: ignore
            source_node_types_names=(
                [self._pair_to_predict[0]] if self._pair_to_predict else None
            ),
            destination_node_types_names=(
                [self._pair_to_predict[1]] if self._pair_to_predict else None
            ),
        )
    else:
        # default GRAPE code
        negative_graph = graph.sample_negative_graph(
            number_of_negative_samples=number_of_negative_samples,
            only_from_same_component=True,
            random_state=self._random_state,
            use_scale_free_distribution=self._use_scale_free_distribution,
            sample_edge_types=len(edge_type_features) > 0,  # type: ignore
            source_node_types_names=(
                [self._pair_to_predict[0]] if self._pair_to_predict else None
            ),
            destination_node_types_names=(
                [self._pair_to_predict[1]] if self._pair_to_predict else None
            ),
        )

    # debug stuff
    # # create df with the negative graph edge data to explore what is inside
    # df_negative_graph_triples = helper_lib.graph.build_triples_df(negative_graph).drop_duplicates()

    # # calcualate the intersection between the view0 dataframe and the negative dataframe to check for overlap
    # intersection_neg = pd.merge(df_view, df_negative_graph_triples, how='inner', on=['source','destination'])
    # print(f"Percentage of negative train set that is actually positive: {len(intersection_neg)/len(df_negative_graph_triples)}")
    # print(f"Absolute value: {len(intersection_neg)}")
    # false_negatives.append(len(intersection_neg)/len(df_negative_graph_triples))

    # print(f"Positive training set size: {len(df_positive_graph_triples)}")
    # print relative frequencies of top 10 source_type - destination_type in train and test set compared to full graph
    # for i,(key,count) in enumerate(df_positive_graph_triples['complete_label'].value_counts().items()):
    #    print(f"{key}: {count}")
    #    if i == 9 :
    #        break
    # print(f"Negative training set size: {len(df_negative_graph_triples)}")
    # print relative frequencies of top 10 source_type - destination_type in train and test set compared to full graph
    # for i,(key,count) in enumerate(df_negative_graph_triples['complete_label'].value_counts().items()):
    #    print(f"{key}: {count}")
    #    if i == 9 :
    #        break

    # does not work properly
    # if(len(intersection_neg)>0):
    #     # remove positive edges from negative test set
    #     print("Removing positive edges from negative graph")
    #     # can't filter by edge id since it could make the graph directed (error from grape)
    #     edge_ids_to_remove = intersection_neg['edge_y'].unique()
    #     negative_graph = negative_graph.to_directed()
    #     negative_graph = negative_graph.filter_from_ids(edge_ids_to_remove=edge_ids_to_remove)
    #     negative_graph = negative_graph.to_undirected()
    #     # negative_graph = negative_graph.filter_from_ids(edge_ids_to_keep=edge_ids_to_keep)

    #     df_negative_graph_triples = helper_lib.graph.build_triples_df(negative_graph).drop_duplicates()
    #     intersection_neg = pd.merge(df_view, df_negative_graph_triples, how='inner', on=['source','destination'])
    #     print(f"[CLEANED] Percentage of negative train set that is actually positive: {len(intersection_neg)/len(df_negative_graph_triples)}")
    #     print(f"[CLEANED] Absolute value: {len(intersection_neg)}")

    assert negative_graph.has_edges()

    # the assert negative_graph.has_selfloops() from GRAPE almost always causes an exception
    if negative_graph.has_selfloops():
        # assert graph.has_selfloops(), (
        #     "The negative graph contains self loops, "
        #     "but the positive graph does not."
        # )
        if not graph.has_selfloops():
            logging.warning(
                "WARNING: The negative graph contains self loops, but the positive graph does not."
            )

    if self._training_unbalance_rate == 1.0:
        number_of_negative_edges = negative_graph.get_number_of_directed_edges()
        number_of_positive_edges = graph.get_number_of_directed_edges()
        self_loop_message = (
            ("The graph contains self loops.")
            if negative_graph.has_selfloops()
            else ("The graph does not contain self loops.")
        )
        if number_of_negative_edges not in (
            number_of_positive_edges + 1,
            number_of_positive_edges,
        ):
            print(
                "The negative graph should have the same number of edges as the "
                "positive graph when using a training unbalance rate of 1.0. "
                "We expect the negative graph to have "
                f"{number_of_positive_edges} or {number_of_positive_edges + 1} edges, but found "
                f"{number_of_negative_edges}. {self_loop_message} "
                f"The exact number requested was {number_of_negative_samples}"
            )

    rasterized_edge_features = []

    for edge_feature in edge_features:  # type: ignore
        for positive_edge_features, negative_edge_features in zip(
            edge_feature.get_edge_feature_from_graph(
                graph=graph,
                support=support,
            ).values(),
            edge_feature.get_edge_feature_from_graph(
                graph=negative_graph,
                support=support,
            ).values(),
        ):
            rasterized_edge_features.append(
                np.vstack((positive_edge_features, negative_edge_features))
            )

    if self._use_edge_metrics:
        rasterized_edge_features.append(
            np.vstack(
                (
                    support.get_all_edge_metrics(  # type: ignore
                        normalize=True,
                        subgraph=graph,
                    ),
                    support.get_all_edge_metrics(  # type: ignore
                        normalize=True,
                        subgraph=negative_graph,
                    ),
                )
            )
        )

    self._model_instance.fit(
        *lpt.transform(
            positive_graph=graph,
            negative_graph=negative_graph,
            edge_features=rasterized_edge_features,
            shuffle=True,
            random_state=self._random_state,
        )
    )


def update_fit(edge_prediction_model, pair_to_predict, original_graph):
    edge_prediction_model._pair_to_predict = pair_to_predict
    edge_prediction_model._original_graph = original_graph
    edge_prediction_model._fit = MethodType(_new_fit, edge_prediction_model)
    return edge_prediction_model
