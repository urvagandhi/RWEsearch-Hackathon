# RWEsearch - Healthcare Analytics Platform

A comprehensive healthcare analytics platform for predicting hospital readmissions and providing clinical insights using real-world evidence (RWE).

## 🎯 Why RWEsearch?

- **⚡ Lightning Fast**: Smart model loading means instant access to analytics
- **🧠 Intelligent**: Automatically discovers and uses existing trained models
- **💰 Cost-Effective**: Immediate cost analysis and intervention planning
- **📊 Comprehensive**: Full ML pipeline from data to actionable insights
- **🔄 Efficient**: No unnecessary retraining - work with what you have
- **📈 Professional**: Production-ready dashboard for healthcare analytics

## Features

- **Readmission Prediction**: Predict 30, 60, and 90-day readmission rates
- **Machine Learning Models**: Support for multiple ML algorithms including:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
  - XGBoost (optional)
  - Deep Learning with TensorFlow (optional)
- **Smart Model Loading**: Automatically loads existing trained models without retraining
- **Interactive Dashboard**: Streamlit-based web interface
- **Data Visualization**: Comprehensive charts and insights
- **Model Persistence**: Save and load trained models with full functionality
- **Clinical Insights**: Risk factor analysis and treatment recommendations
- **Cost Analysis**: Treatment cost estimation and intervention planning

## New Features (Latest Update)

### 🚀 **Smart Model Loading**
- **Automatic Discovery**: System finds and loads existing trained models on startup
- **Instant Performance**: Get model metrics without retraining
- **Direct Insights**: Generate risk factors and cost analysis from loaded models
- **Efficient Workflow**: Only retrain when you need to update models

### 💡 **Enhanced User Experience**
- **Status Indicators**: Clear feedback on model availability
- **Smart Controls**: "Train" vs "Retrain" buttons based on model status
- **Immediate Results**: Performance metrics, ROC curves, and insights available instantly
- **Better Navigation**: Improved workflow with automatic model management

## Quick Start

### Option 1: Run with Docker (Recommended)

1. **Build and run with Docker Compose:**
   ```bash
   docker-compose up --build
   ```

2. **Or build and run with Docker:**
   ```bash
   # Build the image
   docker build -t rwesearch .
   
   # Run the container
   docker run -p 8501:8501 -v $(pwd)/data:/app/data -v $(pwd)/saved_models:/app/saved_models rwesearch
   ```

3. **Access the application:**
   Open your browser and go to `http://localhost:8501`

### Option 2: Run Locally

1. **Install Python 3.11 or higher**

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   # source .venv/bin/activate  # On macOS/Linux
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   streamlit run streamlit_app.py
   ```

5. **Access the application:**
   Open your browser and go to `http://localhost:8501`

## Data Requirements

Place your healthcare data files in the `data/` directory:

- `DE1_0_2008_Beneficiary_Summary_File_Sample_1.csv`
- `DE1_0_2009_Beneficiary_Summary_File_Sample_1.csv`
- `DE1_0_2010_Beneficiary_Summary_File_Sample_1.csv`
- `DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv`
- `DE1_0_2008_to_2010_Outpatient_Claims_Sample_1.csv`
- `condition_occurrence.csv`
- `drug_exposure.csv`
- `person.csv`

## Project Structure

```
RWEsearch/
├── streamlit_app.py          # Main Streamlit application
├── healthcare_pipeline.py    # Core analytics pipeline with smart loading
├── requirements.txt          # Python dependencies
├── Dockerfile               # Docker configuration
├── docker-compose.yml       # Docker Compose configuration
├── data/                    # Data directory
├── saved_models/            # Trained models directory (auto-discovered)
├── MODEL_LOADING_IMPROVEMENTS.md  # Technical documentation
└── __pycache__/            # Python cache files
```

## Key Improvements

### Performance Enhancements
- **⚡ Faster Startup**: Models load in seconds instead of minutes
- **📊 Instant Analytics**: All insights available immediately from saved models
- **🔄 Smart Training**: Avoid unnecessary retraining
- **💾 Persistent Results**: Full model functionality retained across sessions

### User Experience
- **🎯 Clear Status**: Know exactly what models are available
- **🚀 One-Click Insights**: Generate analysis without waiting
- **📈 Immediate Visualization**: ROC curves, feature importance instantly available
- **💰 Direct Cost Analysis**: Treatment cost estimates from existing models

## Dependencies

### Core Libraries
- Streamlit (Web interface)
- Pandas & NumPy (Data processing)
- Scikit-learn (Machine learning)
- Matplotlib & Seaborn (Visualization)

### Optional Libraries
- XGBoost (Enhanced gradient boosting)
- TensorFlow (Deep learning)

*Note: The application will run with core libraries even if optional libraries are not installed.*

## Usage

### Enhanced Workflow (New!)

1. **First Time Setup**:
   - Upload data files to the `data/` directory
   - Use the sidebar to configure data loading
   - Train models for the first time

2. **Subsequent Usage**:
   - **Automatic Loading**: Existing models are loaded automatically
   - **Instant Results**: Performance metrics available immediately
   - **Direct Analysis**: Generate insights and cost analysis without waiting
   - **Optional Retraining**: Use "Retrain" only when updating models

### Step-by-Step Guide

1. **Load Data**: Use the interface to load your healthcare datasets
2. **Check Model Status**: System automatically detects existing trained models
3. **View Results**: Analyze model performance and predictions instantly
4. **Generate Insights**: Get risk factors and treatment recommendations
5. **Cost Analysis**: Review potential savings from interventions
6. **Export Reports**: Download insights and visualizations

### Model Management

- **Auto-Discovery**: System finds existing models in `saved_models/` directory
- **Performance Calculation**: Metrics generated from saved models without retraining
- **Smart Training**: Only train when models don't exist or you want updates
- **Full Functionality**: All features work with loaded models (ROC curves, feature importance, etc.)

## Environment Variables

- `STREAMLIT_SERVER_PORT`: Port for Streamlit server (default: 8501)
- `STREAMLIT_SERVER_ADDRESS`: Server address (default: 0.0.0.0)

## Troubleshooting

### Common Issues

1. **Port already in use**: Change the port in docker-compose.yml or use a different port
2. **Memory issues**: Reduce dataset size or increase Docker memory allocation
3. **Missing dependencies**: Ensure all required packages are installed
4. **Models not loading**: Check `saved_models/` directory permissions and file integrity

### Model Loading Issues

- **No existing models found**: This is normal on first run - train models once
- **Performance metrics missing**: Models will auto-load and calculate metrics
- **Slow loading**: Large models may take a few seconds to load initially
- **Feature importance errors**: Ensure tree-based models (Random Forest, etc.) are available

### Docker Issues

- Ensure Docker is running
- Check if ports are available
- Verify data volume mounts are correct
- Confirm `saved_models/` directory is properly mounted

## Technical Details

### Model Loading Process
1. **Auto-Discovery**: System scans `saved_models/` for existing model files
2. **Smart Loading**: Loads models and generates performance metrics without retraining
3. **Full Integration**: All features (insights, cost analysis, visualizations) work with loaded models
4. **Efficient Workflow**: Only retrain when actually needed

For detailed technical information, see `MODEL_LOADING_IMPROVEMENTS.md`.

## License

This project is for educational and research purposes.
