# ⚖️ Legal AI Assistant

An intelligent Streamlit application that connects to Salesforce/Litify legal databases and provides natural language querying capabilities with full transparency.

## 🌟 Features

- **Natural Language Queries**: Ask questions about your legal data in plain English
- **Real-time Database Access**: Direct connection to Salesforce/Litify
- **AI-Powered**: Uses OpenAI GPT for SOQL query generation
- **Full Transparency**: See exact SOQL queries and explore raw data
- **Interactive Data Explorer**: Customize table columns and download results
- **Conversational Interface**: Chat mode with context awareness
- **Three Interaction Modes**: Predefined questions, custom queries, and chat

## 🔍 Transparency Features

- **Query Visibility**: See the exact SOQL query used for each answer
- **Interactive Tables**: Explore database results with customizable columns
- **Field Selection**: Choose from all available Litify fields
- **Data Download**: Export query results as CSV files
- **Visual Highlighting**: Key fields are highlighted in data tables

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Salesforce org with Litify installed
- Salesforce Connected App with Client Credentials flow
- OpenAI API key

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/legal-ai-assistant.git
   cd legal-ai-assistant