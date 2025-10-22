import pandas as pd

from graph import builder


def build_star_graph_for_row(row: pd.Series, vocabs: dict, config):
    """Build a star heterograph for a single encounter (row) using train vocabs for inductive inference."""
    df = pd.DataFrame([row])
    # We can reuse the builder for a single row
    star_graph = builder.build_heterodata(df, vocabs, config, include_target=False)
    return star_graph
