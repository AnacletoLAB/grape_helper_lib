import pandas as pd


def check_if_in_graph(graph, src_node, dst_node, pair_to_predict=None):
    try:
        if pair_to_predict:
            # check if the node type match
            src_node_type = graph.get_node_type_name_from_node_name(src_node)[0]
            dst_node_type = graph.get_node_type_name_from_node_name(dst_node)[0]
            # print(src_node_type)
            # print(dst_node_type)
            if src_node_type not in pair_to_predict:
                print("Wrong source type")
                return False
            if dst_node_type not in pair_to_predict:
                print("Wrong destination type")
                return False
            if (
                src_node_type == pair_to_predict[0]
                and dst_node_type != pair_to_predict[1]
            ):
                print("Wrong type combination 1")
                return False
            if (
                src_node_type == pair_to_predict[1]
                and dst_node_type != pair_to_predict[0]
            ):
                print("Wrong type combination 2")
                return False
        # print('Type combination: OK!')
        # check if the edge exists
        graph.get_edge_id_from_node_names(src_node, dst_node)
        return True
    except:
        return False


def build_triples_df(graph):
    df = pd.DataFrame()
    if graph.is_directed():
        edge_node_ids = (
            graph.get_directed_source_node_ids(),
            graph.get_directed_destination_node_ids(),
        )
    else:
        edge_node_ids = (
            graph.get_source_node_ids(directed=False),
            graph.get_destination_node_ids(directed=False),
        )
    sources = edge_node_ids[0]
    destinations = edge_node_ids[1]
    sources_types = [
        graph.get_node_type_ids_from_node_id(node_id)[0] for node_id in sources
    ]
    sources_types_labels = [
        graph.get_node_type_name_from_node_type_id(node_type_id)
        for node_type_id in sources_types
    ]
    destinations_types = [
        graph.get_node_type_ids_from_node_id(node_id)[0] for node_id in destinations
    ]
    destinations_types_labels = [
        graph.get_node_type_name_from_node_type_id(node_type_id)
        for node_type_id in destinations_types
    ]

    edge_node_ids = graph.get_edge_node_ids(directed=graph.is_directed())
    edge_ids = [
        graph.get_edge_id_from_node_ids(node_ids[0], node_ids[1])
        for node_ids in edge_node_ids
    ]
    edge_types = [graph.get_edge_type_id_from_edge_id(edge_id) for edge_id in edge_ids]
    edge_type_labels = [
        graph.get_edge_type_name_from_edge_id(edge_id) for edge_id in edge_ids
    ]

    df["source"] = sources
    df["destination"] = destinations
    df["edge"] = edge_ids
    df["source_type"] = sources_types
    df["destination_type"] = destinations_types
    df["edge_type"] = edge_types
    df["source_type_label"] = sources_types_labels
    df["destination_type_label"] = destinations_types_labels
    df["edge_type_label"] = edge_type_labels
    df["complete_label"] = (
        df["source_type_label"] + " - " + df["destination_type_label"]
    )
    return df
