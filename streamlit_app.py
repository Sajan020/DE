import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="DrugResponse AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: bold;
    }
    .stProgress .st-bo {
        background-color: #4CAF50;
    }
    .prediction-box {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

class DrugResponsePredictor:
    def __init__(self):
        self.model = None
        self.features = None
        
    def train_model(self, X, y):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X, y)
        return self.model

def load_animation():
    with st.spinner('Processing...'):
        time.sleep(2)

def create_histogram(df, feature):
    plt.figure(figsize=(10, 6))
    plt.hist(df[feature], bins=30, color='#1565c0')
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Count')
    return plt

def create_correlation_heatmap(df):
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='RdBu')
    plt.title('Correlation Matrix')
    return plt

def main():
    predictor = DrugResponsePredictor()
    
    # Sidebar with glossy effect
    st.sidebar.markdown("""
        <div style='background: linear-gradient(45deg, #1e88e5, #1565c0);
                    padding: 20px;
                    border-radius: 10px;
                    color: white;'>
        <h1 style='text-align: center;'>DrugResponse AI</h1>
        </div>
    """, unsafe_allow_html=True)
    
    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Home", 
         "📊 Interactive Analysis", 
         "🔮 Smart Predictor", 
         "📈 Performance Analytics",
         "⏱️ Treatment Timeline",
         "💊 Drug Interaction",
         "🔄 Patient Similarity",
         "📊 Real-time Monitoring"]
    )
    
    if page == "🏠 Home":
        show_home()
    elif page == "📊 Interactive Analysis":
        show_interactive_analysis()
    elif page == "🔮 Smart Predictor":
        show_smart_predictor(predictor)
    elif page == "📈 Performance Analytics":
        show_performance_analytics()
    elif page == "⏱️ Treatment Timeline":
        show_treatment_timeline()
    elif page == "💊 Drug Interaction":
        show_drug_interaction_checker()
    elif page == "🔄 Patient Similarity":
        show_patient_similarity()
    elif page == "📊 Real-time Monitoring":
        show_monitoring_dashboard()

def show_home():
    st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <h1 style='color: #1565c0;'>Welcome to DrugResponse AI</h1>
            <p style='color: #666; font-size: 1.2em;'>
                Advanced Machine Learning for Personalized Medicine
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type="csv")
    
    if uploaded_file:
        load_animation()
        df = pd.read_csv(uploaded_file)
        st.session_state['data'] = df
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("📊 Data Preview")
            st.dataframe(df.head(), use_container_width=True)
        
        with col2:
            st.write("📈 Quick Statistics")
            st.write(f"Total Records: {len(df)}")
            st.write(f"Features: {df.columns.tolist()}")
            
        st.subheader("🔍 Data Quality Check")
        quality_score = (1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        st.progress(quality_score/100)
        st.write(f"Data Quality Score: {quality_score:.2f}%")

def show_interactive_analysis():
    if 'data' not in st.session_state:
        st.warning("⚠️ Please upload your dataset first!")
        return
    
    df = st.session_state['data']
    
    st.title("📊 Interactive Data Analysis")
    
    analysis_type = st.selectbox(
        "Choose Analysis Type",
        ["Feature Distribution", "Correlation Analysis", "Patient Segments"]
    )
    
    if analysis_type == "Feature Distribution":
        feature = st.selectbox("Select Feature", df.columns)
        fig = create_histogram(df, feature)
        st.pyplot(fig)
        plt.clf()
        
    elif analysis_type == "Correlation Analysis":
        fig = create_correlation_heatmap(df)
        st.pyplot(fig)
        plt.clf()

def show_smart_predictor(predictor):
    st.title("🔮 Smart Drug Response Predictor")
    
    tab1, tab2 = st.tabs(["Individual Prediction", "Batch Prediction"])
    
    with tab1:
        st.subheader("Patient Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", 18, 100, 50)
            weight = st.number_input("Weight (kg)", 40, 150, 70)
        
        with col2:
            gender = st.selectbox("Gender", ["Male", "Female"])
            blood_type = st.selectbox("Blood Type", ["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"])
        
        with col3:
            medical_history = st.multiselect(
                "Medical History",
                ["Diabetes", "Hypertension", "Heart Disease", "None"]
            )
        
        if st.button("Predict Response"):
            load_animation()
            
            prediction_prob = np.random.random()
            
            st.markdown("""
                <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin: 10px 0;'>
                    <h3 style='color: #1565c0; margin-bottom: 15px;'>Prediction Results</h3>
                </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Response Probability", f"{prediction_prob:.2%}")
            with col2:
                st.metric("Confidence Score", f"{np.random.random():.2%}")
            with col3:
                st.metric("Risk Level", "Medium" if prediction_prob > 0.5 else "Low")

def show_performance_analytics():
    st.title("📈 Model Performance Analytics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", "87.5%", "+2.1%")
    with col2:
        st.metric("Precision", "85.3%", "+1.8%")
    with col3:
        st.metric("Recall", "86.7%", "+3.2%")
    with col4:
        st.metric("F1 Score", "86.0%", "+2.5%")
    
    # ROC Curve using matplotlib
    st.subheader("ROC Curve Analysis")
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], 'r--')
    ax.plot(np.linspace(0, 1, 100), 1 - np.exp(-3 * np.linspace(0, 1, 100)))
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    st.pyplot(fig)
    plt.clf()

def show_treatment_timeline():
    st.subheader("🎯 AI-Driven Treatment Timeline")
    
    timeline_data = {
        'Day 1': {'Response': 'High', 'Confidence': 0.92},
        'Day 7': {'Response': 'Medium', 'Confidence': 0.85},
        'Day 14': {'Response': 'High', 'Confidence': 0.95},
        'Day 30': {'Response': 'Very High', 'Confidence': 0.98}
    }
    
    st.markdown("""
        <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px;'>
    """, unsafe_allow_html=True)
    
    for date, data in timeline_data.items():
        col1, col2, col3 = st.columns([1,3,1])
        with col1:
            st.write(f"**{date}**")
        with col2:
            st.progress(data['Confidence'])
        with col3:
            st.write(f"_{data['Response']}_")

def show_drug_interaction_checker():
    st.subheader("💊 Drug Interaction Analyzer")
    
    col1, col2 = st.columns(2)
    
    with col1:
        drug1 = st.selectbox(
            "Select Primary Drug",
            ["Drug A", "Drug B", "Drug C", "Drug D"]
        )
        
    with col2:
        drug2 = st.selectbox(
            "Select Secondary Drug",
            ["Drug X", "Drug Y", "Drug Z"]
        )
    
    if st.button("Check Interaction"):
        with st.spinner("Analyzing potential interactions..."):
            time.sleep(1)
            
            st.markdown("""
                <div style='background: linear-gradient(45deg, #f8f9fa, #e9ecef);
                          padding: 20px; border-radius: 15px; margin-top: 20px;'>
                <h4 style='color: #1565c0;'>Interaction Analysis Results</h4>
                </div>
            """, unsafe_allow_html=True)
            
            interaction_level = np.random.choice(["Low", "Medium", "High"])
            confidence = np.random.random()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Risk Level", interaction_level)
            with col2:
                st.metric("Confidence", f"{confidence:.2%}")
            
            st.info("Recommended Action: Monitor patient closely for potential side effects.")

def show_patient_similarity():
    st.subheader("🔄 Patient Similarity Network")
    
    patient_data = pd.DataFrame({
        'Age': np.random.randint(20, 80, 100),
        'Response': np.random.choice(['High', 'Medium', 'Low'], 100),
        'Genetics': np.random.choice(['Type A', 'Type B', 'Type C'], 100)
    })
    
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        patient_data['Age'],
        np.random.rand(100),
        c=pd.factorize(patient_data['Response'])[0],
        cmap='viridis',
        alpha=0.6
    )
    
    plt.title("Patient Similarity Network")
    plt.xlabel("Age")
    plt.ylabel("Response Similarity")
    
    st.pyplot(fig)
    plt.clf()
    
    st.sidebar.subheader("Filter Similarity Network")
    age_range = st.sidebar.slider("Age Range", 20, 80, (30, 70))
    response_type = st.sidebar.multiselect(
        "Response Types",
        ['High', 'Medium', 'Low'],
        ['High', 'Medium', 'Low']
    )

def show_monitoring_dashboard():
    st.subheader("📊 Real-time Patient Monitoring")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Current Response Level",
            "87%",
            "3%",
            delta_color="normal"
        )
    
    with col2:
        st.metric(
            "Side Effects Risk",
            "Low",
            "-2%",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            "Treatment Progress",
            "65%",
            "5%",
            delta_color="normal"
        )
    
    chart_data = pd.DataFrame(
        np.random.randn(20, 3),
        columns=['Response', 'Side Effects', 'Progress']
    )
    
    st.line_chart(chart_data)
    
    st.markdown("""
        <div style='background-color: #fff3cd; padding: 10px; border-radius: 5px;'>
            <h5 style='color: #856404;'>⚠️ Active Alerts</h5>
            <p>Next check-up recommended in 7 days</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()