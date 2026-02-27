# Kitchen Prep Time (KPT) Optimization

Files added:

- pipeline.py — data loading, cleaning, baseline model training, and dispatch simulation
- app_streamlit.py — lightweight Streamlit app to inspect results per city
- requirements.txt — Python packages required

Quick start

1. (Optional) Create and activate a virtualenv

   python -m venv venv
   source venv/bin/activate

2. Install dependencies

   pip install -r requirements.txt

3. Run the pipeline to train baseline and run simulation

   python pipeline.py

4. Launch the Streamlit dashboard

   streamlit run app_streamlit.py

Notes

- The scripts expect the dataset file `Zomato_Enterprise_Level_Dataset_50k.csv` in the repository root. Change the path in the Streamlit sidebar if needed.
- The code is a concise template following the outline you provided: cleaning, simple feature engineering, a RandomForest baseline for KPT, a simple travel-time heuristic, and a dispatch simulation comparing immediate vs. KPT-aware dispatch.
- Adjust column names in `pipeline.py` if your CSV uses different headings.
