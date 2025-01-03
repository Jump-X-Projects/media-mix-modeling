# Media Mix Modeling Web App

A web application for analyzing and predicting the relationship between media spend and revenue using various machine learning models.

## Features

- Upload and analyze media spend data (CSV/XLS)
- Multiple modeling options (BMMM, LightGBM, XGBoost)
- Interactive visualizations and insights
- Revenue prediction and spend optimization
- Model performance evaluation

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Project Structure

```
├── frontend/           # Streamlit components and UI
├── backend/           # Data processing and ML models
├── tests/            # Unit and integration tests
└── app.py            # Main application entry
```

## Development

- Follow PEP8 coding standards
- Run tests: `pytest tests/`
- Format code: `black .`

## License

MIT 