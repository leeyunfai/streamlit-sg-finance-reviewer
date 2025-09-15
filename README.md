# Government Finance Insights App

This project is a **Streamlit application** that leverages the Singapore Department of Statistics (SingStat) `GovernmentFinanceMonthly.csv` dataset to provide **interactive, AI-driven financial insights**.

## 🚀 Features
- **Interactive Filters**  
  Manually adjust filters in the Streamlit UI to explore government finance data.

- **Natural Language Filtering (NLP)**  
  Update filters using natural language queries, making the data exploration more intuitive.

- **AI-Generated Insights**  
  The filtered dataset is sent to the **OpenAI API**, which generates narrative insights and highlights key trends.

- **Insight Validation with Gemini**  
  The **Gemini API** acts as a secondary checker to validate or challenge OpenAI’s insights, ensuring reliability.

## 🛠 Tech Stack
- [Streamlit](https://streamlit.io/) – App framework  
- [SingStat GovernmentFinanceMonthly.csv](https://www.singstat.gov.sg/) – Data source  
- [OpenAI API](https://platform.openai.com/) – Insight generation  
- [Gemini API](https://ai.google/) – Insight validation  
- NLP – Natural language filter interpretation  

## 🎯 Purpose
This project demonstrates how **AI can augment financial data analysis** by combining:
- Structured government statistical data  
- Natural language interaction  
- Multi-model validation  

The result: a **hands-on exploration tool** that makes government financial data more accessible, insightful, and trustworthy.

---
