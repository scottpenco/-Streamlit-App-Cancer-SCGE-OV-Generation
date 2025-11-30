import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from generate import generate_synthetic
from evaluate_synthetic_data import evaluate
import seaborn as sns
import numpy as np
from scipy.stats import ks_2samp
import streamlit as st


test_df_real = pd.read_csv('data/processed/test_combo_scaled.csv')
continuous_cols = [col for col in test_df_real.columns if col != 'death_event']

def main():
    # Title 
    st.title("SDGE TCGA OV Data generator")


    st.sidebar.header('Select **n** number of samples to generate')
    n_samples = st.sidebar.slider('Number of samples', min_value=1, max_value=1000, value=43, step=1)

    df = generate_synthetic(n_samples=n_samples)

    st.subheader("Generated Synthetic Data")
    st.dataframe(df)

    # Download button
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='synthetic_tcga_ov_data.csv',
        mime='text/csv',
    )
    # # Evaluation of synthetic data
    # st.subheader("Evaluation of Synthetic Data")
    # test_df_real = pd.read_csv('./data/processed/test_combo_scaled.csv')
    # continuous_cols = [col for col in test_df_real.columns if col != 'death_event']
    # metrics = evaluate(real_df=test_df_real, df=df, continuous_cols=continuous_cols)

    # st.dataframe(metrics['ks_test_pass_rate'], use_container_width=True)
    st.markdown("## Histogram Comparison (First 3 Features)")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for idx, col in enumerate(continuous_cols[:3]):
        ax = axes[idx]

        ax.hist(test_df_real[col], bins=20, alpha=0.6, label="Real", color="blue", density=True)
        ax.hist(df[col],       bins=20, alpha=0.6, label="Synthetic", color="orange", density=True)

        ax.set_title(f"{col}", fontweight="bold")
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

    # KDE PLOTS
    st.markdown("## KDE Distributions + KS Test (First 4 Features)")

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    axes = axes.flatten()

    for idx, col in enumerate(continuous_cols[:4]):
        ax = axes[idx]

        real_data = test_df_real[col].dropna()
        syn_data = df[col].dropna()

        sns.kdeplot(real_data, ax=ax, label="Real", linewidth=2, color="blue", fill=True, alpha=0.3)
        sns.kdeplot(syn_data, ax=ax, label="Synthetic", linewidth=2, color="orange", fill=True, alpha=0.3)

        # KS test
        stat, pval = ks_2samp(real_data, syn_data)

        ax.set_title(f"{col}\nKS-test p-value: {pval:.4f}", fontweight="bold")
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    
    # SCATTER PLOTS
       
    st.markdown("## Scatter Relationships (Real vs Synthetic)")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    col_pairs = [(0, 1), (1, 2), (2, 3), (3, 4)]

    for idx, (i, j) in enumerate(col_pairs):
        if i < len(continuous_cols) and j < len(continuous_cols):
            ax = axes[idx]

            col1 = continuous_cols[i]
            col2 = continuous_cols[j]

            ax.scatter(
                test_df_real[col1], test_df_real[col2],
                alpha=0.5, s=30, label="Real", color="blue"
            )
            ax.scatter(
                df[col1], df[col2],
                alpha=0.5, s=30, label="Synthetic", color="orange"
            )

            ax.set_xlabel(col1, fontsize=9)
            ax.set_ylabel(col2, fontsize=9)
            ax.set_title(f"{col1} vs {col2}", fontweight="bold", fontsize=10)
            ax.legend()
            ax.grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

    
    # BOX PLOTS
    st.markdown("## Boxplot Comparison (First 4 Features)")

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    axes = axes.flatten()

    for idx, col in enumerate(continuous_cols[:4]):
        ax = axes[idx]

        data_to_plot = [test_df_real[col], df[col]]
        bp = ax.boxplot(data_to_plot, labels=["Real", "Synthetic"], patch_artist=True)

        bp["boxes"][0].set_facecolor("lightblue")
        bp["boxes"][1].set_facecolor("lightyellow")

        ax.set_title(f"{col}", fontweight="bold")
        ax.set_ylabel("Value")
        ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    st.pyplot(fig)

    
    # CORRELATION HEATMAPS
    st.markdown("## Correlation Heatmaps (First 15 PC of Genetic Features)")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    real_corr = test_df_real[continuous_cols[:15]].corr()
    syn_corr = df[continuous_cols[:15]].corr()

    sns.heatmap(
        real_corr, ax=axes[0], cmap="coolwarm", center=0, vmin=-1, vmax=1,
        square=True, cbar_kws={"label": "Correlation"}
    )
    axes[0].set_title("Real Data Correlations", fontsize=14, fontweight="bold")

    sns.heatmap(
        syn_corr, ax=axes[1], cmap="coolwarm", center=0, vmin=-1, vmax=1,
        square=True, cbar_kws={"label": "Correlation"}
    )
    axes[1].set_title("Synthetic Data Correlations", fontsize=14, fontweight="bold")

    plt.tight_layout()
    st.pyplot(fig)

    
    # Q-Q PLOTS
    st.markdown("## Q-Q Plots (First 4 Features)")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for idx, col in enumerate(continuous_cols[:4]):
        ax = axes[idx]

        real_data = test_df_real[col].dropna()
        syn_data = df[col].dropna()

        # Get quantiles
        real_quantiles = np.percentile(real_data, np.linspace(0, 100, 50))
        syn_quantiles = np.percentile(syn_data, np.linspace(0, 100, 50))

        # Plot
        ax.scatter(real_quantiles, syn_quantiles, alpha=0.6, s=50)

        # Diagonal line (perfect match)
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()]),
        ]
        ax.plot(lims, lims, "r--", alpha=0.75, linewidth=2, zorder=0)

        ax.set_xlabel("Real Data Quantiles")
        ax.set_ylabel("Synthetic Data Quantiles")
        ax.set_title(f"{col} - Q-Q Plot", fontweight="bold")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

    
    # STATS SUMMARY TABLE (st.dataframe)
    st.markdown("## Statistical Comparison Table (Real vs Synthetic)")

    stats_rows = []
    for col in continuous_cols[:10]:
        real_mean = test_df_real[col].mean()
        real_std = test_df_real[col].std()
        syn_mean = df[col].mean()
        syn_std = df[col].std()

        stat, pval = ks_2samp(test_df_real[col], df[col])

        stats_rows.append(
            {
                "Feature": col[:20],
                "Real Mean": round(real_mean, 4),
                "Real Std": round(real_std, 4),
                "Syn Mean": round(syn_mean, 4),
                "Syn Std": round(syn_std, 4),
                "KS p-value": round(pval, 4),
                "Pass (p>0.05)": "✓" if pval > 0.05 else "✗",
            }
        )

    stats_df = pd.DataFrame(stats_rows)
    st.dataframe(stats_df, use_container_width=True)


if __name__ == "__main__":
    st.set_page_config(
        page_title="SDGE TCGA OV Data generator", page_icon=":chart_with_upwards_trend:"
    )
    main()
