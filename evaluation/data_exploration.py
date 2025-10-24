from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import dcor
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from minepy import MINE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler


@dataclass
class PCAResult:
    components: pd.DataFrame
    explained_variance: pd.DataFrame
    projection: pd.DataFrame
    scatter_figure: go.Figure


@dataclass
class ClusteringResult:
    assignments: pd.DataFrame
    centroid_frame: pd.DataFrame
    silhouette: float
    scatter_figure: go.Figure


class DataExplorer:
    """Utility class that generates descriptive tables and interactive figures."""

    def __init__(
        self,
        df: pd.DataFrame,
        target_column: str,
        *,
        max_sample: int = 5000,
        random_state: int = 42,
    ) -> None:
        if target_column not in df.columns:
            raise KeyError(f"Target column '{target_column}' not present in dataframe.")
        self.target_column = target_column
        self.random_state = random_state
        self.df = df.copy()
        if len(df) > max_sample:
            self.sample = df.sample(n=max_sample, random_state=random_state).reset_index(drop=True)
        else:
            self.sample = df.copy().reset_index(drop=True)

    # ------------------------------------------------------------------
    # Helper properties
    # ------------------------------------------------------------------
    @property
    def numeric_columns(self) -> List[str]:
        return [c for c in self.sample.columns if pd.api.types.is_numeric_dtype(self.sample[c]) and c != self.target_column]

    @property
    def categorical_columns(self) -> List[str]:
        return [c for c in self.sample.columns if pd.api.types.is_categorical_dtype(self.sample[c]) or self.sample[c].dtype == object]

    def _subset(self, columns: Optional[Sequence[str]]) -> List[str]:
        if columns is None:
            return self.numeric_columns
        return [c for c in columns if c in self.sample.columns and c != self.target_column]

    # ------------------------------------------------------------------
    # Summary statistics & class balance
    # ------------------------------------------------------------------
    def summary_table(self) -> pd.DataFrame:
        desc = self.sample.describe(include="all").transpose()
        missing_pct = self.sample.isna().mean() * 100
        desc["missing_pct"] = missing_pct
        return desc.sort_index()

    def class_balance_table(self) -> pd.DataFrame:
        counts = self.sample[self.target_column].value_counts().rename("count")
        frac = counts / counts.sum()
        return pd.DataFrame({"count": counts, "fraction": frac})

    def class_balance_plot(self) -> go.Figure:
        tbl = self.class_balance_table().reset_index().rename(columns={"index": self.target_column})
        fig = px.bar(tbl, x=self.target_column, y="count", color=self.target_column, text="count", title="Class balance")
        fig.update_layout(showlegend=False)
        return fig

    # ------------------------------------------------------------------
    # Distribution visualisations
    # ------------------------------------------------------------------
    def violin_plot(self, columns: Optional[Sequence[str]] = None) -> go.Figure:
        cols = self._subset(columns)
        melted = self.sample[[self.target_column, *cols]].melt(id_vars=self.target_column, var_name="feature", value_name="value")
        fig = px.violin(melted, x="feature", y="value", color=self.target_column, box=True, points="outliers")
        fig.update_layout(title="Feature distributions by class")
        return fig

    def class_density_plot(self, columns: Optional[Sequence[str]] = None) -> go.Figure:
        cols = self._subset(columns)
        if len(cols) < 2:
            raise ValueError("Class density plot requires at least two numeric columns.")
        df_plot = self.sample[[self.target_column, cols[0], cols[1]]].dropna()
        fig = px.density_contour(
            df_plot,
            x=cols[0],
            y=cols[1],
            color=self.target_column,
            title=f"Class density: {cols[0]} vs {cols[1]}",
        )
        return fig

    # ------------------------------------------------------------------
    # Outlier analysis
    # ------------------------------------------------------------------
    def outlier_summary(self, columns: Optional[Sequence[str]] = None, z_threshold: float = 3.0) -> pd.DataFrame:
        cols = self._subset(columns)
        data = self.sample[cols].select_dtypes(include=[np.number]).dropna()
        if data.empty:
            return pd.DataFrame()
        z_scores = np.abs((data - data.mean()) / data.std(ddof=0))
        result = (z_scores > z_threshold).sum().to_frame(name="outlier_count")
        result["outlier_fraction"] = result["outlier_count"] / len(data)
        return result.sort_values("outlier_count", ascending=False)

    # ------------------------------------------------------------------
    # Dependence and correlation analysis
    # ------------------------------------------------------------------
    def distance_correlation_heatmap(self, columns: Optional[Sequence[str]] = None) -> go.Figure:
        cols = self._subset(columns)
        df_num = self.sample[cols].dropna()
        matrix = np.zeros((len(cols), len(cols)))
        for i, c1 in enumerate(cols):
            for j, c2 in enumerate(cols):
                if j < i:
                    matrix[i, j] = matrix[j, i]
                else:
                    matrix[i, j] = dcor.distance_correlation(df_num[c1], df_num[c2])
        fig = go.Figure(data=go.Heatmap(z=matrix, x=cols, y=cols, colorscale="Viridis"))
        fig.update_layout(title="Distance correlation heatmap")
        return fig

    def mutual_information_table(self, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
        cols = self._subset(columns)
        if not cols:
            return pd.DataFrame(columns=["feature", "mutual_information"])
        df_feat = self.sample[cols].copy()
        df_feat = pd.get_dummies(df_feat, drop_first=True)
        df_feat = df_feat.fillna(df_feat.median())
        y = pd.factorize(self.sample[self.target_column])[0]
        mi = mutual_info_classif(df_feat.values, y, discrete_features=False, random_state=self.random_state)
        return pd.DataFrame({"feature": df_feat.columns, "mutual_information": mi}).sort_values("mutual_information", ascending=False)

    def mic_table(self, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
        cols = self._subset(columns)
        if not cols:
            return pd.DataFrame(columns=["feature", "mic", "mas", "tic"])
        mine = MINE()
        y = pd.factorize(self.sample[self.target_column])[0]
        scores: List[Dict[str, float]] = []
        for col in cols:
            x = self.sample[col].values
            mask = ~pd.isna(x)
            mine.compute_score(x[mask], y[mask])
            scores.append({"feature": col, "mic": mine.mic(), "mas": mine.mas(), "tic": mine.tic()})
        return pd.DataFrame(scores).sort_values("mic", ascending=False)

    # ------------------------------------------------------------------
    # PCA & clustering
    # ------------------------------------------------------------------
    def pca_analysis(self, n_components: int = 3) -> PCAResult:
        cols = self.numeric_columns
        if not cols:
            raise ValueError("PCA analysis requires at least one numeric feature.")
        df_num = self.sample[cols].fillna(self.sample[cols].median())
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df_num)
        pca = PCA(n_components=min(n_components, scaled.shape[1]))
        proj = pca.fit_transform(scaled)
        comp_df = pd.DataFrame(pca.components_, columns=cols)
        explained = pd.DataFrame({"component": np.arange(1, pca.n_components_ + 1), "explained_variance": pca.explained_variance_ratio_})
        proj_df = pd.DataFrame(proj, columns=[f"PC{i+1}" for i in range(pca.n_components_)])
        proj_df[self.target_column] = self.sample[self.target_column].values[: len(proj_df)]
        fig = px.scatter_3d(proj_df, x="PC1", y="PC2", z="PC3" if pca.n_components_ >= 3 else "PC2", color=self.target_column, title="PCA projection")
        return PCAResult(components=comp_df, explained_variance=explained, projection=proj_df, scatter_figure=fig)

    def clustering_analysis(self, columns: Optional[Sequence[str]] = None, n_clusters: int = 3) -> ClusteringResult:
        cols = self._subset(columns)
        if not cols:
            raise ValueError("Clustering analysis requires at least one numeric feature.")
        df_num = self.sample[cols].fillna(self.sample[cols].median())
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df_num)
        km = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        labels = km.fit_predict(scaled)
        silhouette = silhouette_score(scaled, labels) if n_clusters > 1 else float("nan")
        assign_df = pd.DataFrame({"cluster": labels, self.target_column: self.sample[self.target_column].values[: len(labels)]})
        centroids = pd.DataFrame(km.cluster_centers_, columns=cols)
        scatter = px.scatter_matrix(assign_df.join(df_num.reset_index(drop=True)), dimensions=cols[: min(len(cols), 4)], color="cluster", title="Cluster scatter matrix")
        return ClusteringResult(assignments=assign_df, centroid_frame=centroids, silhouette=float(silhouette), scatter_figure=scatter)


__all__ = ["DataExplorer", "PCAResult", "ClusteringResult"]
