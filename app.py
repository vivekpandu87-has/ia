import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_curve, auc,
                             mean_squared_error, r2_score)
from mlxtend.frequent_patterns import apriori, association_rules

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Cricket Command",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── COLOUR SYSTEM ─────────────────────────────────────────────────────────────
BG      = "#0a0f1e"
CARD    = "#111827"
BORDER  = "#1e293b"
GOLD    = "#f59e0b"
GREEN   = "#10b981"
BLUE    = "#3b82f6"
RED     = "#ef4444"
TEXT    = "#f1f5f9"
MUTED   = "#94a3b8"
PALETTE = [GOLD, GREEN, BLUE, RED, "#8b5cf6", "#ec4899", "#06b6d4", "#84cc16"]

BASE_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Georgia, serif", color=TEXT, size=13),
    title_font=dict(family="Georgia, serif", size=17, color=GOLD),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT)),
    margin=dict(l=20, r=20, t=50, b=20),
    xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, tickfont=dict(color=MUTED)),
    yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, tickfont=dict(color=MUTED)),
)

def L(fig, title=""):
    fig.update_layout(**BASE_LAYOUT)
    if title:
        fig.update_layout(title_text=title)
    return fig

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@600;700&family=Lato:wght@300;400;700&display=swap');

html, body, [class*="css"] {{
    background-color:{BG} !important;
    color:{TEXT};
    font-family:'Lato',sans-serif;
}}
[data-testid="stSidebar"] {{
    background:linear-gradient(180deg,#0d1424 0%,{BG} 100%) !important;
    border-right:1px solid {BORDER};
}}
[data-testid="stSidebar"] * {{ color:{TEXT} !important; }}
[data-testid="metric-container"] {{
    background:{CARD};
    border:1px solid {BORDER};
    border-radius:12px;
    padding:18px 22px !important;
    box-shadow:0 4px 24px rgba(0,0,0,.4);
}}
[data-testid="metric-container"] label {{
    font-size:.78rem; letter-spacing:.12em;
    text-transform:uppercase; color:{MUTED} !important;
}}
[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    font-family:'Cinzel',serif; font-size:2rem !important; color:{GOLD} !important;
}}
h1 {{ font-family:'Cinzel',serif !important; color:{GOLD} !important;
      border-bottom:2px solid {GOLD}33; padding-bottom:8px; margin-bottom:20px; }}
h2, h3 {{ font-family:'Cinzel',serif !important; color:{TEXT} !important; }}
hr {{ border-color:{BORDER} !important; margin:22px 0; }}
[data-testid="stDataFrame"] {{ border-radius:10px; overflow:hidden; }}
[data-testid="stFileUploader"] {{
    background:{CARD}; border:1px dashed {BLUE}; border-radius:10px; padding:12px;
}}
.pill {{
    display:inline-block; padding:3px 14px; border-radius:999px;
    font-size:.74rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase;
    background:{GOLD}22; color:{GOLD}; border:1px solid {GOLD}55; margin-bottom:10px;
}}
.card-box {{
    background:{CARD}; border:1px solid {BORDER}; border-radius:14px;
    padding:20px 24px; margin-bottom:16px; box-shadow:0 2px 16px rgba(0,0,0,.35);
}}
</style>
""", unsafe_allow_html=True)


# ── DATA ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        return pd.read_csv("dataset.csv")
    except Exception:
        st.error("⚠️  dataset.csv not found. Place it in the same folder as app.py.")
        return None


df = load_data()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
st.sidebar.markdown(f"""
<div style='text-align:center;padding:20px 0 10px;'>
  <span style='font-family:Cinzel,serif;font-size:1.3rem;color:{GOLD};letter-spacing:.08em;'>
    🏏 Smart Cricket<br>Command Centre
  </span>
</div>
<hr style='border-color:{BORDER};margin:8px 0 18px;'/>
""", unsafe_allow_html=True)

PAGES = [
    "📊  Overview & Descriptive",
    "🔍  Diagnostic Analysis",
    "👥  Segmentation (Clustering)",
    "🎯  Predictive (Classification)",
    "🔗  Predictive (Association Rules)",
    "💰  Predictive (Regression)",
    "🚀  Lead Scorer",
]
nav  = st.sidebar.radio("", PAGES, label_visibility="collapsed")
page = nav.split("  ", 1)[1]


# ═════════════════════════════════════════════════════════════════════════════
if df is not None:

    # ── 1. OVERVIEW & DESCRIPTIVE ────────────────────────────────────────────
    if page == "Overview & Descriptive":
        st.markdown('<div class="pill">Market Intelligence</div>', unsafe_allow_html=True)
        st.title("📊 Market Snapshot")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Surveyed",   f"{len(df):,}")
        c2.metric("Hot Leads",        f"{df['Switch_Intent'].sum():,}")
        c3.metric("Avg Annual Spend", f"₹{df['Annual_Spend_Estimate'].mean():,.0f}")
        c4.metric("Conversion Rate",  f"{df['Switch_Intent'].mean()*100:.1f}%")

        st.markdown("---")

        # Pie + Age histogram
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Respondents by City Tier")
            fig = px.pie(df, names="City_Tier", hole=0.5,
                         color_discrete_sequence=PALETTE)
            fig.update_traces(textposition="outside", textinfo="percent+label",
                              marker=dict(line=dict(color=BG, width=2)))
            st.plotly_chart(L(fig, "City Tier Distribution"), use_container_width=True)

        with col2:
            st.subheader("Age & Gender Distribution")
            fig = px.histogram(df, x="Age", color="Gender", nbins=20, barmode="overlay",
                               opacity=0.8,
                               color_discrete_sequence=[GOLD, BLUE])
            fig.update_traces(marker_line_width=0)
            st.plotly_chart(L(fig, "Age Groups by Gender"), use_container_width=True)

        st.markdown("---")

        # Switch intent bar + Income distribution
        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Switch Intent by City Tier")
            d = (df.groupby("City_Tier")["Switch_Intent"]
                 .mean().mul(100).round(1).reset_index())
            d.columns = ["City_Tier", "Intent_Rate"]
            fig = px.bar(d, x="City_Tier", y="Intent_Rate",
                         color="Intent_Rate",
                         color_continuous_scale=["#1e293b", GOLD],
                         text="Intent_Rate")
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside",
                              marker_line_width=0)
            fig.update_coloraxes(showscale=False)
            st.plotly_chart(L(fig, "Switch Intent Rate (%) by City Tier"),
                            use_container_width=True)

        with col4:
            st.subheader("Income Distribution")
            fig = px.histogram(df, x="Income_Lakhs", nbins=25,
                               color_discrete_sequence=[GREEN])
            fig.update_traces(marker_line_width=0)
            st.plotly_chart(L(fig, "Annual Income Distribution (₹ Lakhs)"),
                            use_container_width=True)


    # ── 2. DIAGNOSTIC ANALYSIS ───────────────────────────────────────────────
    elif page == "Diagnostic Analysis":
        st.markdown('<div class="pill">Root Cause</div>', unsafe_allow_html=True)
        st.title("🔍 Why Do They Switch?")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Tech Comfort vs Switch Intent")
            fig = px.violin(df, x="Switch_Intent", y="Digital_Usage_Score",
                            box=True, points="all", color="Switch_Intent",
                            color_discrete_sequence=[BLUE, GOLD])
            st.plotly_chart(L(fig, "Digital Usage Score vs Intent"), use_container_width=True)

        with col2:
            st.subheader("Switch Intent by Cricket Skill Level")
            d = (df.groupby("Cricket_Skill")["Switch_Intent"]
                 .mean().mul(100).round(1).reset_index())
            d.columns = ["Cricket_Skill", "Intent_Rate"]
            fig = px.bar(d, x="Cricket_Skill", y="Intent_Rate",
                         color="Cricket_Skill",
                         color_discrete_sequence=PALETTE,
                         text="Intent_Rate")
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside",
                              marker_line_width=0, showlegend=False)
            st.plotly_chart(L(fig, "Switch Intent Rate (%) by Cricket Skill"),
                            use_container_width=True)

        st.markdown("---")
        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Income vs Switch Intent")
            fig = px.box(df, x="Switch_Intent", y="Income_Lakhs",
                         color="Switch_Intent",
                         color_discrete_sequence=[GREEN, RED])
            st.plotly_chart(L(fig, "Income Distribution by Switch Intent"),
                            use_container_width=True)

        with col4:
            st.subheader("Digital Score by Gender")
            fig = px.box(df, x="Gender", y="Digital_Usage_Score",
                         color="Gender",
                         color_discrete_sequence=[GOLD, BLUE])
            st.plotly_chart(L(fig, "Digital Usage Score by Gender"),
                            use_container_width=True)


    # ── 3. SEGMENTATION (CLUSTERING) ─────────────────────────────────────────
    elif page == "Segmentation (Clustering)":
        st.markdown('<div class="pill">ML · KMeans</div>', unsafe_allow_html=True)
        st.title("👥 Customer Segmentation")

        k = st.sidebar.slider("Number of Segments", 2, 6, 4)
        X_sc = StandardScaler().fit_transform(df[["Income_Lakhs", "Digital_Usage_Score"]])
        km   = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_sc)
        df["Segment"] = ("Segment " + (km.labels_ + 1).astype(str))

        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Income vs Tech Savviness Groups")
            fig = px.scatter(df, x="Income_Lakhs", y="Digital_Usage_Score",
                             color="Segment", symbol="Segment",
                             color_discrete_sequence=PALETTE, opacity=0.85)
            fig.update_traces(marker=dict(size=8,
                                          line=dict(width=0.5, color=BG)))
            st.plotly_chart(L(fig, "Customer Clusters — Income × Digital Score"),
                            use_container_width=True)

        with col2:
            st.subheader("Segment Size")
            sc = df["Segment"].value_counts().reset_index()
            sc.columns = ["Segment", "Count"]
            fig = px.pie(sc, names="Segment", values="Count", hole=0.55,
                         color_discrete_sequence=PALETTE)
            fig.update_traces(textinfo="percent+label",
                              marker=dict(line=dict(color=BG, width=2)))
            st.plotly_chart(L(fig, "Segment Share"), use_container_width=True)

        st.markdown("---")
        st.subheader("Segment Profile Summary")
        profile = df.groupby("Segment").agg(
            Members=("Segment", "count"),
            Avg_Income=("Income_Lakhs", "mean"),
            Avg_Digital_Score=("Digital_Usage_Score", "mean"),
            Switch_Rate=("Switch_Intent", "mean"),
            Avg_Spend=("Annual_Spend_Estimate", "mean"),
        ).round(2).reset_index()
        profile["Switch_Rate"] = (profile["Switch_Rate"]*100).round(1).astype(str) + "%"
        st.dataframe(profile, use_container_width=True)


    # ── 4. PREDICTIVE — CLASSIFICATION ───────────────────────────────────────
    elif page == "Predictive (Classification)":
        st.markdown('<div class="pill">ML · Random Forest</div>', unsafe_allow_html=True)
        st.title("🎯 Switch Prediction Model")

        X = pd.get_dummies(df[["Income_Lakhs", "Cricket_Skill",
                                "Digital_Usage_Score", "Age"]])
        y = df["Switch_Intent"]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_tr, y_tr)
        y_pred  = clf.predict(X_te)
        y_probs = clf.predict_proba(X_te)[:, 1]
        fpr, tpr, _ = roc_curve(y_te, y_probs)
        roc_auc = auc(fpr, tpr)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy",  f"{accuracy_score(y_te, y_pred)*100:.1f}%")
        m2.metric("Precision", f"{precision_score(y_te, y_pred)*100:.1f}%")
        m3.metric("Recall",    f"{recall_score(y_te, y_pred)*100:.1f}%")
        m4.metric("F1 Score",  f"{f1_score(y_te, y_pred)*100:.1f}%")

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("ROC Curve")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                     name=f"AUC = {roc_auc:.2f}",
                                     line=dict(color=GOLD, width=2.5),
                                     fill="tozeroy", fillcolor=f"{GOLD}22"))
            fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Random",
                                     line=dict(color=MUTED, width=1, dash="dash")))
            st.plotly_chart(L(fig, "Receiver Operating Characteristic"),
                            use_container_width=True)

        with col2:
            st.subheader("Feature Importance (Bar Graph)")
            feat = (pd.Series(clf.feature_importances_, index=X.columns)
                    .nlargest(10).reset_index())
            feat.columns = ["Feature", "Importance"]
            fig = px.bar(feat.sort_values("Importance"),
                         x="Importance", y="Feature", orientation="h",
                         color="Importance",
                         color_continuous_scale=["#1e293b", GOLD],
                         text=feat.sort_values("Importance")["Importance"].round(3))
            fig.update_traces(textposition="outside", marker_line_width=0)
            fig.update_coloraxes(showscale=False)
            st.plotly_chart(L(fig, "Top 10 Predictive Features"),
                            use_container_width=True)

        st.markdown("---")
        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Class Balance (Pie)")
            bal = y.value_counts().reset_index()
            bal.columns = ["Switch_Intent", "Count"]
            bal["Label"] = bal["Switch_Intent"].map({0: "No Switch", 1: "Will Switch"})
            fig = px.pie(bal, names="Label", values="Count", hole=0.5,
                         color_discrete_sequence=[BLUE, GOLD])
            fig.update_traces(textinfo="percent+label",
                              marker=dict(line=dict(color=BG, width=2)))
            st.plotly_chart(L(fig, "Target Class Distribution"),
                            use_container_width=True)

        with col4:
            st.subheader("Predicted Probability Distribution")
            fig = px.histogram(x=y_probs, nbins=30,
                               color_discrete_sequence=[GREEN],
                               labels={"x": "Predicted Probability"})
            fig.update_traces(marker_line_width=0)
            st.plotly_chart(L(fig, "Distribution of Switch Probabilities"),
                            use_container_width=True)


    # ── 5. ASSOCIATION RULES ─────────────────────────────────────────────────
    elif page == "Predictive (Association Rules)":
        st.markdown('<div class="pill">Market Basket</div>', unsafe_allow_html=True)
        st.title("🔗 Lifestyle Associations")

        basket_cols = ["Bought_Saree", "Bought_Cookware", "Bought_AirFryer",
                       "Bought_Premium_Pen", "Switch_Intent"]
        basket   = df[basket_cols].astype(bool)
        min_sup  = st.sidebar.slider("Min Support",  0.01, 0.30, 0.05, 0.01)
        min_lift = st.sidebar.slider("Min Lift",     0.50, 3.00, 1.00, 0.10)

        freq  = apriori(basket, min_support=min_sup, use_colnames=True)
        rules = association_rules(freq, metric="lift", min_threshold=min_lift)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Top Rules by Lift (Bar Graph)")
            top = rules.nlargest(10, "lift").copy()
            top["rule"] = (top["antecedents"].astype(str)
                           + " → " + top["consequents"].astype(str))
            fig = px.bar(top.sort_values("lift"),
                         x="lift", y="rule", orientation="h",
                         color="lift",
                         color_continuous_scale=["#1e293b", GOLD],
                         text=top.sort_values("lift")["lift"].round(2))
            fig.update_traces(textposition="outside", marker_line_width=0)
            fig.update_coloraxes(showscale=False)
            st.plotly_chart(L(fig, "Top Association Rules — Lift Score"),
                            use_container_width=True)

        with col2:
            st.subheader("Confidence vs Support (Bubble Chart)")
            fig = px.scatter(rules, x="support", y="confidence", size="lift",
                             color="lift",
                             color_continuous_scale=["#1e293b", GREEN],
                             hover_data=["antecedents", "consequents", "lift"])
            st.plotly_chart(L(fig, "Support × Confidence — Bubble Size = Lift"),
                            use_container_width=True)

        st.markdown("---")
        st.subheader("Full Association Rules Table")
        out = (rules[["antecedents", "consequents", "support", "confidence", "lift"]]
               .sort_values("lift", ascending=False).round(4))
        st.dataframe(out, use_container_width=True)


    # ── 6. PREDICTIVE — REGRESSION ───────────────────────────────────────────
    elif page == "Predictive (Regression)":
        st.markdown('<div class="pill">ML · Random Forest</div>', unsafe_allow_html=True)
        st.title("💰 Spending Forecast")

        X_r = pd.get_dummies(df[["Income_Lakhs", "Digital_Usage_Score", "Cricket_Skill"]])
        y_r = df["Annual_Spend_Estimate"]
        X_tr, X_te, y_tr, y_te = train_test_split(X_r, y_r, test_size=0.2, random_state=42)
        reg    = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_tr, y_tr)
        y_pred = reg.predict(X_te)
        rmse   = np.sqrt(mean_squared_error(y_te, y_pred))
        r2     = r2_score(y_te, y_pred)

        m1, m2, m3 = st.columns(3)
        m1.metric("RMSE",               f"₹{rmse:,.0f}")
        m2.metric("R² Score",           f"{r2:.3f}")
        m3.metric("Avg Predicted Spend", f"₹{y_pred.mean():,.0f}")

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Actual vs Predicted Spend")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=y_te.values, y=y_pred, mode="markers",
                marker=dict(color=GOLD, size=7, opacity=0.7,
                            line=dict(width=0.5, color=BG)),
                name="Predictions",
            ))
            mn, mx = float(y_te.min()), float(y_te.max())
            fig.add_trace(go.Scatter(x=[mn,mx], y=[mn,mx], mode="lines",
                                     line=dict(color=GREEN, dash="dash", width=1.5),
                                     name="Perfect Fit"))
            st.plotly_chart(L(fig, "Actual vs Predicted Annual Spend"),
                            use_container_width=True)

        with col2:
            st.subheader("Feature Importance (Bar Graph)")
            feat = (pd.Series(reg.feature_importances_, index=X_r.columns)
                    .nlargest(10).reset_index())
            feat.columns = ["Feature", "Importance"]
            fig = px.bar(feat.sort_values("Importance"),
                         x="Importance", y="Feature", orientation="h",
                         color="Importance",
                         color_continuous_scale=["#1e293b", GREEN],
                         text=feat.sort_values("Importance")["Importance"].round(3))
            fig.update_traces(textposition="outside", marker_line_width=0)
            fig.update_coloraxes(showscale=False)
            st.plotly_chart(L(fig, "Drivers of Spending"), use_container_width=True)

        st.markdown("---")
        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Residual Distribution")
            fig = px.histogram(x=(y_te.values - y_pred), nbins=30,
                               color_discrete_sequence=[BLUE],
                               labels={"x": "Residual (Actual − Predicted)"})
            fig.update_traces(marker_line_width=0)
            st.plotly_chart(L(fig, "Residual Distribution"), use_container_width=True)

        with col4:
            st.subheader("Avg Spend Share by City Tier (Pie)")
            d = df.groupby("City_Tier")["Annual_Spend_Estimate"].mean().reset_index()
            fig = px.pie(d, names="City_Tier", values="Annual_Spend_Estimate",
                         hole=0.5, color_discrete_sequence=PALETTE)
            fig.update_traces(textinfo="percent+label",
                              marker=dict(line=dict(color=BG, width=2)))
            st.plotly_chart(L(fig, "Average Spend Share by City Tier"),
                            use_container_width=True)


    # ── 7. LEAD SCORER ───────────────────────────────────────────────────────
    elif page == "Lead Scorer":
        st.markdown('<div class="pill">Conversion Engine</div>', unsafe_allow_html=True)
        st.title("🚀 Lead Scoring")

        st.markdown("""
        <div class="card-box">
          Upload a CSV of prospects. The model will score each lead from <b>0–100</b>
          based on income, digital usage, cricket skill, and age — then rank them
          for your sales team.
        </div>
        """, unsafe_allow_html=True)

        # train on full dataset
        X_ls = pd.get_dummies(df[["Income_Lakhs", "Cricket_Skill",
                                   "Digital_Usage_Score", "Age"]])
        clf_ls = RandomForestClassifier(n_estimators=100, random_state=42).fit(
            X_ls, df["Switch_Intent"])

        up = st.file_uploader("Upload Prospect CSV", type="csv")

        if up:
            leads = pd.read_csv(up)
            try:
                X_new = pd.get_dummies(leads[["Income_Lakhs", "Cricket_Skill",
                                               "Digital_Usage_Score", "Age"]])
                X_new = X_new.reindex(columns=X_ls.columns, fill_value=0)
                leads["Lead_Score"] = (clf_ls.predict_proba(X_new)[:, 1] * 100).round(1)
                leads["Priority"] = pd.cut(
                    leads["Lead_Score"],
                    bins=[0, 40, 70, 100],
                    labels=["🟡 Warm", "🟠 Hot", "🔴 Priority"],
                )
                scored = leads.sort_values("Lead_Score", ascending=False).reset_index(drop=True)

                c1, c2 = st.columns(2)
                c1.metric("Total Leads",         len(scored))
                c2.metric("High Priority (>70)", int((scored["Lead_Score"] > 70).sum()))

                st.markdown("---")
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Lead Score Distribution")
                    fig = px.histogram(scored, x="Lead_Score", nbins=20,
                                       color_discrete_sequence=[GOLD])
                    fig.update_traces(marker_line_width=0)
                    st.plotly_chart(L(fig, "Distribution of Lead Scores"),
                                    use_container_width=True)

                with col2:
                    st.subheader("Priority Tier Breakdown (Pie)")
                    pc = scored["Priority"].value_counts().reset_index()
                    pc.columns = ["Priority", "Count"]
                    fig = px.pie(pc, names="Priority", values="Count", hole=0.5,
                                 color_discrete_sequence=[GOLD, RED, GREEN])
                    fig.update_traces(textinfo="percent+label",
                                      marker=dict(line=dict(color=BG, width=2)))
                    st.plotly_chart(L(fig, "Priority Tier Distribution"),
                                    use_container_width=True)

                st.subheader("Scored & Ranked Prospects")
                st.dataframe(scored, use_container_width=True)

            except KeyError as e:
                st.error(f"Missing column: {e}. CSV must include: "
                         "Income_Lakhs, Cricket_Skill, Digital_Usage_Score, Age")
        else:
            st.info("📂 Upload a CSV file above to begin lead scoring.")
