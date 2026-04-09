import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def show_eda_page(df):
    
    st.subheader("📊 Exploratory Data Analysis")
    st.caption("Visualize patterns in the original heart health dataset.")
    
    # Identify column types automatically
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Custom CSS for styling
    st.markdown("""
    <style>
    .eda-title {
        border-left: 6px solid #1565C0;
        padding-left: 0.7rem;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # --- SECTION 1: CATEGORICAL DISTRIBUTIONS (Gender, etc.) ---
    st.markdown('<h3 class="eda-title"> Categorical Distribution (Gender & Others)</h3>', unsafe_allow_html=True)
    
    # Try to set 'Sex' or 'Gender' as default if they exist in cat_cols
    default_cat = [c for c in cat_cols if c.lower() in ['sex', 'hadheartattack', 'hadstroke']]
    
    selected_cat_cols = st.multiselect(
        "Select categorical variables:", 
        cat_cols, 
        default=default_cat if default_cat else cat_cols[:1]
    )

    if selected_cat_cols:
        cols_cat = st.columns(3)
        for i, col in enumerate(selected_cat_cols):
            with cols_cat[i % 3]:
                fig, ax = plt.subplots(figsize=(4, 3))
                
                # Using a Count Plot for categorical data
                # We use a distinct palette to differentiate from numerical charts
                sns.countplot(data=df, x=col, palette="Blues_d", ax=ax, order=df[col].value_counts().index)
                
                ax.set_title(f'{col} Distribution', fontsize=9, fontweight='bold')
                ax.tick_params(labelsize=7)
                ax.set_xlabel('')
                ax.set_ylabel('Count', fontsize=7)
                
                # Rotate labels if they are long (like Age Categories)
                if df[col].nunique() > 3:
                    plt.xticks(rotation=45)
                
                sns.despine()
                st.pyplot(fig)
                plt.close(fig)

    st.markdown("---")
    # --- SECTION 2: NUMERICAL DISTRIBUTIONS ---
    st.markdown('<h3 class="eda-title"> Numerical Variable Distribution</h3>', unsafe_allow_html=True)
    
    selected_hist_cols = st.multiselect(
        "Select numerical variables:", 
        num_cols, 
        default=num_cols[:3] if len(num_cols) > 3 else num_cols
    )
    
    if selected_hist_cols:
        cols = st.columns(3) 
        for i, col in enumerate(selected_hist_cols):
            with cols[i % 3]: 
                fig, ax = plt.subplots(figsize=(4, 3))
                sns.histplot(df[col], kde=True, bins=20, color="#1565C0", alpha=0.6, ax=ax)
                ax.set_title(f'{col}', fontsize=9, fontweight='bold')
                ax.tick_params(labelsize=7)
                ax.set_xlabel('')
                ax.set_ylabel('')
                sns.despine()
                st.pyplot(fig)
                plt.close(fig)

    