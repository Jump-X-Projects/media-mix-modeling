### Project Requirements Document (PRD)

1.  **Project Overview**

This media mix modeling web app is designed to analyze and predict the relationship between media expenditure across various channels and corresponding daily revenues. The core idea is to provide businesses with insights into how effectively their marketing spend on platforms like Google, Facebook, YouTube, Instagram, TikTok, and others, translate into revenue. By utilizing predictive analytics, this app allows users to explore and understand the influence their media investments have on their financial returns.

The app is being built to facilitate data-driven decision-making in marketing strategies. The key objectives are to enable users to make informed media spending decisions, optimize budget allocations across channels, and ultimately maximize revenue. Success in this project is defined by the app’s ability to provide accurate, actionable insights through intuitive and interactive analytics, positioning users to improve marketing ROI substantially.

1.  **In-Scope vs. Out-of-Scope**

*   **In-Scope:**

    *   **Data Ingestion**: Users can upload media spend and revenue data in CSV or XLS formats.
    *   **Model Selection**: Option to choose from BMMM, LightGBM, XGBoost, and Meta Robyn for analysis.
    *   **Analysis Outputs**: Display of feature importance, a correlation matrix, and insights specific to media channels.
    *   **Revenue Predictions**: Users can input spends to forecast expected revenue.
    *   **Spending Suggestions**: Recommendations for optimal channel spend to achieve target revenue.
    *   **Model Evaluation**: Display of metrics like R-squared, RMSE, MAPE, and statistical significance indicators.

*   **Out-of-Scope:**

    *   Integration with external third-party APIs.
    *   Long-term storage of user-uploaded data.
    *   User authentication and registration systems for external users.
    *   Advanced AI models or datasets beyond the current scope.

1.  **User Flow**

A typical user journey begins on the Home Screen, which provides an overview and guides navigating the app. Users start by uploading their media spend and revenue datasets in CSV or XLS formats through the Data Upload Module. Here, data is initially validated and cleaned to ensure quality input for analysis.

Once data is prepped, users proceed to the Model Selection Page, where they can select an analytical model from options like BMMM, LightGBM, XGBoost, or Meta Robyn. Consequently, users are directed to the Analysis Results page, which visually displays the feature importance, correlation matrix, and preliminary insights. Moving forward, users can interact with the Revenue Predictor Module to simulate outcomes based on hypothetical spend scenarios and use the Spend Optimizer Module to get reverse predictions on achieving desired revenue targets. Finally, the Model Performance Dashboard shows comprehensive metrics that validate the prediction accuracy and reliability.

1.  **Core Features**

*   **Data Upload Interface**: Supports media and revenue data upload in CSV/XLS formats.

*   **Model Selection**: Allows choice between BMMM, LightGBM, XGBoost, and Meta Robyn.

*   **Analysis Dashboard**:

    *   **Feature Importance Visualization**
    *   **Correlation Matrix Visualization**

*   **Revenue Predictor**: Simulates potential revenue based on user inputted media spends.

*   **Spend Optimizer**: Provides spend recommendations for target revenue achievements.

*   **Model Evaluation**: Displays analytical metrics including R-squared, RMSE, MAPE, etc.

1.  **Tech Stack & Tools**

*   **Frontend**: Streamlit
*   **Backend**: Python
*   **Machine Learning Libraries**: scikit-learn, LightGBM, XGBoost, optional Pyro or TensorFlow Probability for BMMM
*   **Data Handling**: Pandas
*   **AI Tools**: Claude AI for coding assistance, Cursor for IDE support

1.  **Non-Functional Requirements**

*   **Performance**: Efficient handling of datasets up to 5000 rows with responsive analysis output.
*   **Security**: Data security with HTTPS for secure transmissions; temporary data storage during sessions.
*   **Usability**: Inspired by platforms like Tableau or Google Analytics for user-friendly navigation.
*   **Compliance**: Adheres to data privacy standards, ensuring no long-term data retention without consent.

1.  **Constraints & Assumptions**

*   The app is built for internal team use initially; hence, sophisticated authentication features are excluded.
*   Dependency on Streamlit's capabilities for deployment and user interface realization.
*   Assumes that chosen predictive models are sufficient for current scope without the need for third-party integrations.

1.  **Known Issues & Potential Pitfalls**

*   **Scalability Limitations**: Current design might restrict scalability beyond 5000-row datasets.
*   **Model Flexibility**: Diving deeper might require additional setup for models like Pyro or TensorFlow Probability.
*   **User Engagement**: Lack of user support or help features may undermine user experience initially.

These details should guide the development process, ensuring clarity and facilitating successful project execution without unexpected hurdles.
