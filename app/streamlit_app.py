"""PerturbAI research app. Reads artifacts produced by notebooks 05-13 from the data root.
Run:  streamlit run app/streamlit_app.py     (set PERTURBAI_DATA if your data root is not ~/PerturbAI_data)"""
import json
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from perturbai.paths import artifact, get_paths

TAG = os.environ.get("PERTURBAI_TAG", "replogle_k562_essential")
P = get_paths()
st.set_page_config(page_title="PerturbAI", layout="wide")
st.title("PerturbAI: perturbation response modelling")
st.caption("Computational hypotheses only. Predictions are not validated drug targets.")


@st.cache_data
def load_table(path):
    p = str(path)
    return pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p, index_col=0)


@st.cache_resource
def load_model():
    return joblib.load(P.models / artifact("best_model", TAG))


def need(path):
    if not path.exists():
        st.warning(f"Missing artifact: {path.name}. Run the corresponding notebook first.")
        return False
    return True


page = st.sidebar.radio("Page", ["Dataset explorer", "Perturbation explorer", "Response predictor",
                                 "Network viewer", "Model evaluation", "Target prioritization"])

if page == "Dataset explorer":
    f = P.manifests / f"{TAG.replace('_synthetic', '')}_audit.json"
    if need(f):
        a = json.load(open(f))
        c = st.columns(4)
        c[0].metric("Cells", f"{a['n_cells']:,}"); c[1].metric("Genes", f"{a['n_genes']:,}")
        c[2].metric("Perturbations", f"{a['n_perturbations']:,}"); c[3].metric("Control cells", f"{a['n_control_cells']:,}")
        st.json(a)
    nc = P.processed / artifact("n_cells", TAG)
    if nc.exists():
        s = load_table(nc).iloc[:, 0]
        st.subheader("Cells per perturbation (analysis subset)"); st.bar_chart(s.value_counts().sort_index())

elif page == "Perturbation explorer":
    ef = P.processed / artifact("effects", TAG)
    if need(ef):
        D = load_table(ef)
        p = st.selectbox("Perturbation (measured)", D.index)
        d = D.loc[p].sort_values()
        k = st.slider("Top genes", 5, 50, 15)
        c1, c2 = st.columns(2)
        c1.subheader("Most down"); c1.dataframe(d.head(k).to_frame("delta"))
        c2.subheader("Most up"); c2.dataframe(d.tail(k)[::-1].to_frame("delta"))
        st.bar_chart(pd.concat([d.head(k), d.tail(k)]))

elif page == "Response predictor":
    if need(P.models / artifact("best_model", TAG)):
        B = load_model(); feats = B["feature_table"]
        st.write(f"Model: **{B['name']}** trained on {len(B['trained_on'])} perturbations.")
        g = st.selectbox("Gene to perturb (CRISPRi knockdown)", sorted(feats.index))
        pred = pd.Series(B["model"].predict(feats.loc[[g]].values)[0], index=B["genes"]).sort_values()
        st.info("This gene was in the training set." if g in set(B["trained_on"]) else "Unseen perturbation: prediction.")
        k = st.slider("Top genes", 5, 50, 15, key="k2")
        c1, c2 = st.columns(2)
        c1.subheader("Predicted down"); c1.dataframe(pred.head(k).to_frame("predicted_delta"))
        c2.subheader("Predicted up"); c2.dataframe(pred.tail(k)[::-1].to_frame("predicted_delta"))
        ef = P.processed / artifact("effects", TAG)
        if ef.exists() and g in load_table(ef).index:
            obs = load_table(ef).loc[g, pred.index]
            st.write("Pearson(observed delta, predicted delta):", round(float(np.corrcoef(obs, pred)[0, 1]), 3))

elif page == "Network viewer":
    ef = P.processed / artifact("effects", TAG)
    if need(ef):
        D = load_table(ef); p = st.selectbox("Perturbation", D.index, key="net")
        C = pd.Series(np.corrcoef(D.values)[list(D.index).index(p)], index=D.index).drop(p).sort_values(ascending=False)
        st.subheader("Perturbations with the most similar response profile")
        st.dataframe(C.head(20).to_frame("pearson_of_delta"))

elif page == "Model evaluation":
    bf = P.results / artifact("benchmark", TAG)
    if need(bf):
        from perturbai.evaluation.benchmark import compare_to_baseline, summarize
        long = pd.read_csv(bf)
        st.subheader("Unseen-perturbation benchmark (mean ± sd over seeds)")
        st.dataframe(summarize(long).round(3))
        st.subheader("Paired difference vs mean baseline (Pearson on delta)")
        st.dataframe(compare_to_baseline(long, "mean", "pearson_delta").round(3))
        fig = P.figures / f"{TAG}_benchmark.png"
        if fig.exists(): st.image(str(fig))

else:
    pf = P.results / artifact("priority", TAG)
    if need(pf):
        R = load_table(pf)
        st.error("Ranked computational hypotheses for follow-up. NOT validated drug targets. "
                 "Validate experimentally and assess safety, tractability and therapeutic window.")
        n = st.slider("Rows", 10, 200, 30)
        st.dataframe(R.head(n))
        st.download_button("Download CSV", R.to_csv().encode(), "target_priority.csv")
