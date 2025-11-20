import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pickle
import time
from datetime import datetime
import csv
import os

st.set_page_config(page_title="Network Flow IDS", layout="wide", page_icon=":shield:")

def load_data():
    # Ensure this path points to your actual data file
    # You may need to create a dummy csv or point to real data for this to run
    try:
        df = pd.read_csv('./data/network_traffic.csv')
    except FileNotFoundError:
        st.error("Data file not found. Please ensure './data/network_traffic.csv' exists.")
        return pd.DataFrame() 
    return df

def preprocess_data(df):
    if df.empty:
        return df
    df_processed = df.copy()
    df_processed['Label'] = df_processed['Label'].apply(lambda x: 1 if x != 'BENIGN' else 0)
    return df_processed

def train_model(X_train, y_train):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    return clf

def create_feature_importance_plot(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    fig = go.Figure(data=[go.Bar(
        x=[feature_names[i] for i in indices],
        y=[importances[i] for i in indices],
        text=[f"{importances[i]:.3f}" for i in indices],
        textposition='auto',
    )])
    
    fig.update_layout(
        title='Feature Importance',
        xaxis_title='Features',
        yaxis_title='Importance Score',
        xaxis_tickangle=-45
    )
    
    return fig

def save_flow_data(flow_data, predictions, filename):
    """
    Save flow data along with predictions to a CSV file.
    """
    save_df = flow_data.copy()
    
    save_df['detection_time'] = predictions['detection_time']
    save_df['predicted_label'] = predictions['prediction']
    save_df['prediction_confidence'] = predictions['confidence']
   
    os.makedirs('captured_data', exist_ok=True)
    
    filepath = os.path.join('captured_data', filename)
    
    if os.path.exists(filepath):
        save_df.to_csv(filepath, mode='a', header=False, index=False)
    else:
        save_df.to_csv(filepath, index=False)
    
    return filepath

def live_detection():
    st.header("Live Flow Analysis")
    
    try:
        with open('ids_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('ids_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        st.sidebar.subheader("Data Capture Settings")
        save_data = st.sidebar.checkbox("Save flow data", value=True)
        custom_filename = st.sidebar.text_input(
            "Custom filename (optional)", 
            value=f"flow_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        
        if st.button("Start Flow Monitoring"):
            placeholder = st.empty()
            chart_placeholder = st.empty()
            
            flow_history = []
            saved_flows_count = 0
            
            # Simulation loop
            for i in range(50):
                df = load_data()
                if df.empty:
                    break

                random_flow = df.iloc[np.random.randint(len(df))].copy()
                # Ensure we drop Label before scaling/predicting
                features = random_flow.drop('Label').values.reshape(1, -1)
                scaled_flow = scaler.transform(features)
                
                prediction = model.predict(scaled_flow)[0]
                prediction_prob = model.predict_proba(scaled_flow)[0]
                
                prediction_info = {
                    'detection_time': datetime.now(),
                    'prediction': prediction,
                    'confidence': prediction_prob.max()
                }
                
                filepath = ""
                if save_data:
                    filepath = save_flow_data(
                        random_flow.to_frame().T,
                        prediction_info,
                        custom_filename
                    )
                    saved_flows_count += 1
                
                flow_history.append({
                    'time': i,
                    'prediction': prediction,
                    'confidence': prediction_prob.max()
                })
                
                with placeholder.container():
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Flow ID", f"#{i+1}")
                    with col2:
                        if prediction == 0:
                            st.success("Benign Flow")
                        else:
                            st.error("Suspicious Flow")
                    with col3:
                        st.metric("Confidence", f"{prediction_prob.max():.2%}")
                    
                    if save_data:
                        st.info(f"Saved flows: {saved_flows_count}")
                        st.text(f"Saving to: {filepath}")
                    
                    # Display some flow features
                    st.json({
                        'Flow Duration': f"{random_flow.get('Flow Duration', 0):.2f}",
                        'Fwd Packets': f"{random_flow.get('Total Fwd Packets', 0):.2f}",
                        'Bwd Packets': f"{random_flow.get('Total Backward Packets', 0):.2f}",
                        'Fwd Bytes': f"{random_flow.get('Total Length of Fwd Packets', 0):.2f}",
                        'Bwd Bytes': f"{random_flow.get('Total Length of Bwd Packets', 0):.2f}"
                    })
                
                if len(flow_history) > 1:
                    history_df = pd.DataFrame(flow_history)
                    fig = px.line(history_df, x='time', y='confidence',
                                color=history_df['prediction'].astype(str),
                                title='Flow Analysis History',
                                color_discrete_map={'0': 'green', '1': 'red'})
                    chart_placeholder.plotly_chart(fig)
                
                time.sleep(0.5)
            
            if save_data:
                st.success(f"""
                Flow monitoring completed!
                - Total flows captured: {saved_flows_count}
                - Data saved to: {filepath}
                """)
                
    except FileNotFoundError:
        st.error("Please train the model first!")    

def save_feedback_to_csv(feedback):
    """
    Save the user feedback to a CSV file.
    """
    file_path = "feedback.csv"
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Feedback"])
        
        writer.writerow([feedback])        

def main():
    st.title("🛡️ Network Flow-Based Intrusion Detection System")
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox("Choose a page", 
                               ["Home", "Training", "Live Detection", "Analytics"])
    
    if page == "Home":
        st.markdown("""
        ## Welcome to Network Flow IDS
        This application analyzes network flow data to detect potential intrusions.
        
        ### Features:
        - Flow-based traffic analysis
        - Machine learning-based detection
        - Interactive visualizations
        """)
        
        st.sidebar.title("Quick Actions")
        st.sidebar.write("Access the app's main features quickly:")
        
        if st.sidebar.button("View Analytics", key="analytics_btn"):
            st.session_state.page = "Analytics"
            st.rerun()
        
        if st.sidebar.button("Retrain Model", key="retrain_btn"):
            st.session_state.page = "Training"
            st.rerun()
        
        if st.sidebar.button("Live Detection", key="live_detection_btn"):
            st.session_state.page = "Live Detection"
            st.rerun()

    elif page == "Training":
        st.header("Model Training")
        
        if st.button("Load and Process Data"):
            with st.spinner("Loading data..."):
                df = load_data()
                if not df.empty:
                    st.success(f"Loaded {len(df)} flow records!")
                    
                    st.subheader("Sample Flow Data")
                    st.dataframe(df.head())
                    
                    df_processed = preprocess_data(df)
                    X = df_processed.drop('Label', axis=1)
                    y = df_processed['Label']
                    
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42)
                    
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    with st.spinner("Training model..."):
                        model = train_model(X_train_scaled, y_train)
                        
                    with open('ids_model.pkl', 'wb') as f:
                        pickle.dump(model, f)
                    with open('ids_scaler.pkl', 'wb') as f:
                        pickle.dump(scaler, f)
                    
                    y_pred = model.predict(X_test_scaled)
                    
                    st.subheader("Model Performance")
                    st.text("Classification Report:")
                    st.text(classification_report(y_test, y_pred))
                    
                    cm = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt='d', ax=ax, 
                              xticklabels=['Benign', 'Malicious'],
                              yticklabels=['Benign', 'Malicious'])
                    plt.xlabel('Predicted')
                    plt.ylabel('True')
                    st.pyplot(fig)
                    
                    importance_fig = create_feature_importance_plot(model, X.columns)
                    st.plotly_chart(importance_fig)

    elif page == "Live Detection":
        live_detection()

    elif page == "Analytics":
        st.header("Flow Analytics Dashboard")
        
        try:
            df = load_data()
            if not df.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'Label' in df.columns:
                        label_counts = df['Label'].value_counts()
                        fig = px.pie(values=label_counts.values, 
                                    names=label_counts.index,
                                    title='Flow Classification Distribution')
                        st.plotly_chart(fig)
                
                with col2:
                    if 'Total Fwd Packets' in df.columns and 'Total Backward Packets' in df.columns:
                        fig = px.scatter(df, x='Total Fwd Packets', y='Total Backward Packets',
                                       color='Label' if 'Label' in df.columns else None, 
                                       title='Forward vs Backward Packets',
                                       opacity=0.6)
                        st.plotly_chart(fig)
                
                if 'Flow Duration' in df.columns:
                    fig = px.histogram(df, x='Flow Duration', 
                                     color='Label' if 'Label' in df.columns else None,
                                     title='Flow Duration Distribution',
                                     marginal='box')
                    st.plotly_chart(fig)
                
                iat_cols = [col for col in df.columns if 'IAT' in col]
                if iat_cols and 'Label' in df.columns:
                    iat_data = df[iat_cols + ['Label']]
                    
                    fig = px.box(iat_data.melt(id_vars=['Label'], 
                                             value_vars=iat_cols),
                                x='variable', y='value', color='Label',
                                title="IAT Analysis")
                    st.plotly_chart(fig)
            
        except FileNotFoundError:
            st.error("Data is not loaded yet. Please load the data first!")

    st.sidebar.write("---")
    st.sidebar.title("About")
    st.sidebar.write("Network Flow IDS v1.0")
    st.sidebar.write("Created by Group 196")
    
    st.sidebar.write("---")
    st.sidebar.title("Let us know how we can improve!")
            
if __name__ == "__main__":
    if "page" not in st.session_state:
        st.session_state.page = "Home"
    main()
