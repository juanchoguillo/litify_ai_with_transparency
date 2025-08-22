"""
Streamlit Web Interface for Legal AI Assistant with Salesforce/Litify Integration
Enhanced with conversational context, history awareness, and TRANSPARENCY FEATURES
Run with: streamlit run app.py
"""

import streamlit as st
import os
import requests
import pandas as pd
from typing import Dict, Any, List, Tuple
import json
from pathlib import Path
from dotenv import load_dotenv
import warnings
from datetime import datetime, date, timedelta
import urllib.parse
import pytz
from openai import OpenAI

# Load environment variables (fallback only)
load_dotenv()

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Set page config
st.set_page_config(
    page_title="Legal AI Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def configuration_form():
    """Display configuration form for user credentials"""
    st.title("⚖️ Legal AI Assistant - Configuration")
    st.markdown("**Please enter your Salesforce and OpenAI credentials to get started**")
    st.markdown("---")
    
    # Check if already configured
    if st.session_state.get('configured', False):
        st.success("✅ Configuration completed! You can now use the Legal AI Assistant.")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Continue to Assistant", use_container_width=True):
                st.session_state.show_config = False
                st.rerun()
        
        st.markdown("---")
        if st.button("🔧 Reconfigure Credentials"):
            st.session_state.configured = False
            st.session_state.show_config = True
            # Clear existing credentials
            for key in ['sf_client_id', 'sf_client_secret', 'sf_instance_url', 'openai_api_key']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
        return True
    
    # Configuration form
    with st.form("config_form"):
        st.markdown("### 🔗 Salesforce/Litify Configuration")
        
        # Salesforce credentials
        sf_client_id = st.text_input(
            "Salesforce Client ID",
            value=os.getenv('SALESFORCE_CLIENT_ID', ''),
            help="Your Salesforce Connected App Client ID",
            type="password"
        )
        
        sf_client_secret = st.text_input(
            "Salesforce Client Secret",
            value=os.getenv('SALESFORCE_CLIENT_SECRET', ''),
            help="Your Salesforce Connected App Client Secret",
            type="password"
        )
        
        sf_instance_url = st.text_input(
            "Salesforce Instance URL",
            value=os.getenv('SALESFORCE_INSTANCE_URL', 'https://bridgeconsulting2.my.salesforce.com'),
            help="Your Salesforce instance URL (e.g., https://yourorg.my.salesforce.com)"
        )
        
        st.markdown("### 🤖 OpenAI Configuration")
        
        openai_api_key = st.text_input(
            "OpenAI API Key",
            value=os.getenv('OPENAI_API_KEY', ''),
            help="Your OpenAI API key for AI agents",
            type="password"
        )
        
        st.markdown("---")
        
        # Instructions
        with st.expander("ℹ️ Configuration Help"):
            st.markdown("""
            **Salesforce Setup:**
            1. Create a Connected App in Salesforce
            2. Enable OAuth settings with Client Credentials flow
            3. Copy the Consumer Key (Client ID) and Consumer Secret (Client Secret)
            4. Your instance URL is typically: https://[yourorg].my.salesforce.com
            
            **OpenAI Setup:**
            1. Go to https://platform.openai.com/api-keys
            2. Create a new API key
            3. Copy the key (starts with sk-...)
            
            **Security Note:**
            Your credentials are stored only in your browser session and are not saved permanently.
            """)
        
        # Submit button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            submitted = st.form_submit_button("🔐 Connect & Validate", use_container_width=True)
        
        if submitted:
            # Validate inputs
            if not sf_client_id:
                st.error("❌ Please enter Salesforce Client ID")
                return False
            if not sf_client_secret:
                st.error("❌ Please enter Salesforce Client Secret")
                return False
            if not sf_instance_url:
                st.error("❌ Please enter Salesforce Instance URL")
                return False
            if not openai_api_key:
                st.error("❌ Please enter OpenAI API Key")
                return False
            
            # Store in session state
            st.session_state.sf_client_id = sf_client_id
            st.session_state.sf_client_secret = sf_client_secret
            st.session_state.sf_instance_url = sf_instance_url
            st.session_state.openai_api_key = openai_api_key
            
            # Set OpenAI API key for the session
            os.environ['OPENAI_API_KEY'] = openai_api_key
            
            # Test connection
            with st.spinner("🔄 Testing Salesforce connection..."):
                test_api = SalesforceAPI(sf_client_id, sf_client_secret, sf_instance_url)
                if test_api.get_access_token():
                    st.success("✅ Salesforce connection successful!")
                    st.session_state.configured = True
                    st.session_state.show_config = False
                    st.rerun()
                else:
                    st.error("❌ Failed to connect to Salesforce. Please check your credentials.")
                    return False
    
    return False

class SalesforceAPI:
    """Handles Salesforce API authentication and queries"""
    
    def __init__(self, client_id=None, client_secret=None, instance_url=None):
        # Use provided credentials or fall back to session state or environment
        self.client_id = (client_id or 
                         st.session_state.get('sf_client_id') or 
                         os.getenv('SALESFORCE_CLIENT_ID'))
        self.client_secret = (client_secret or 
                             st.session_state.get('sf_client_secret') or 
                             os.getenv('SALESFORCE_CLIENT_SECRET'))
        self.instance_url = (instance_url or 
                            st.session_state.get('sf_instance_url') or 
                            os.getenv('SALESFORCE_INSTANCE_URL', 'https://bridgeconsulting2.my.salesforce.com'))
        
        self.access_token = None
        self.token_expires_at = None
    
    def get_access_token(self) -> bool:
        """Authenticate with Salesforce using Client Credentials flow"""
        
        if not all([self.client_id, self.client_secret, self.instance_url]):
            print("❌ Missing required Salesforce credentials")
            return False
        
        token_url = f"{self.instance_url}/services/oauth2/token"
        
        data = {
            'grant_type': 'client_credentials',
            'client_id': self.client_id,
            'client_secret': self.client_secret
        }
        
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        try:
            print(f"🔄 Attempting authentication to {token_url}")
            print(f"🔄 Client ID: {self.client_id[:10]}...")
            
            response = requests.post(token_url, data=data, headers=headers, timeout=30)
            
            if response.status_code != 200:
                print(f"❌ HTTP {response.status_code}: {response.reason}")
                try:
                    error_data = response.json()
                    print(f"❌ Error details: {error_data}")
                except:
                    print(f"❌ Raw response: {response.text}")
                return False
            
            token_data = response.json()
            self.access_token = token_data.get('access_token')
            self.instance_url = token_data.get('instance_url', self.instance_url)
            
            if self.access_token:
                print(f"✅ Successfully authenticated with Salesforce")
                return True
            else:
                print(f"❌ No access token received in response: {token_data}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Network error during authentication: {e}")
            return False
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse authentication response: {e}")
            return False
    
    def execute_soql_query(self, soql_query: str) -> List[Dict]:
        """Execute SOQL query against Salesforce"""
        
        if not self.access_token:
            if not self.get_access_token():
                return []
        
        encoded_query = urllib.parse.quote(soql_query)
        query_url = f"{self.instance_url}/services/data/v63.0/query?q={encoded_query}"
        
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }
        
        try:
            response = requests.get(query_url, headers=headers, timeout=30)
            
            if response.status_code == 401:
                print("🔄 Token expired, re-authenticating...")
                if self.get_access_token():
                    headers['Authorization'] = f'Bearer {self.access_token}'
                    response = requests.get(query_url, headers=headers, timeout=30)
                else:
                    return []
            
            response.raise_for_status()
            data = response.json()
            
            records = data.get('records', [])
            
            # Clean up records (remove Salesforce metadata)
            cleaned_records = []
            for record in records:
                clean_record = {k: v for k, v in record.items() if k != 'attributes'}
                cleaned_records.append(clean_record)
            
            print(f"✅ Successfully executed query, returned {len(cleaned_records)} records")
            return cleaned_records
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Query execution failed: {e}")
            return []
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse query response: {e}")
            return []
    
    def get_all_fields_for_object(self, object_name: str = "litify_pm__Matter__c") -> List[Dict]:
        """Get all available fields for the Litify Matter object"""
        
        if not self.access_token:
            if not self.get_access_token():
                return []
        
        describe_url = f"{self.instance_url}/services/data/v63.0/sobjects/{object_name}/describe"
        
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }
        
        try:
            response = requests.get(describe_url, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            fields = data.get('fields', [])
            
            # Extract relevant field information
            field_list = []
            for field in fields:
                field_info = {
                    'name': field.get('name'),
                    'label': field.get('label'),
                    'type': field.get('type'),
                    'length': field.get('length'),
                    'updateable': field.get('updateable', False),
                    'createable': field.get('createable', False)
                }
                field_list.append(field_info)
            
            print(f"✅ Retrieved {len(field_list)} fields for {object_name}")
            return field_list
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to get fields for {object_name}: {e}")
            return []

class LegalAIAssistant:
    def __init__(self):
        self.salesforce_api = SalesforceAPI()
        self.matter_fields = self._get_matter_field_mapping()
        
        # Initialize OpenAI client
        api_key = st.session_state.get('openai_api_key') or os.getenv('OPENAI_API_KEY')
        if api_key:
            self.openai_client = OpenAI(api_key=api_key)
        else:
            self.openai_client = None
        
        # Set up current date context for date-related queries
        self.current_datetime = datetime.now()
        self.current_date = self.current_datetime.date()
        self.current_year = self.current_date.year
        self.current_month = self.current_date.month
        
        # Try to get timezone-aware datetime (default to UTC if pytz not available)
        try:
            # You can change this to your local timezone
            local_tz = pytz.timezone('America/New_York')  # Eastern Time
            self.current_datetime_tz = datetime.now(local_tz)
        except:
            self.current_datetime_tz = self.current_datetime
        
        print(f"🕒 Legal AI Assistant initialized with current date context:")
        print(f"   Current Date: {self.current_date}")
        print(f"   Current Year: {self.current_year}")
        print(f"   Current DateTime: {self.current_datetime_tz.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        
    def get_date_context(self) -> str:
        """Get formatted date context for AI agents"""
        # Calculate common date ranges
        last_30_days = self.current_date - timedelta(days=30)
        last_90_days = self.current_date - timedelta(days=90)
        last_6_months = self.current_date - timedelta(days=180)
        
        # Calculate last month
        first_day_current_month = self.current_date.replace(day=1)
        last_month = first_day_current_month - timedelta(days=1)
        
        return f"""
        CURRENT DATE CONTEXT:
        - Today's Date: {self.current_date} ({self.current_date.strftime('%B %d, %Y')})
        - Current Year: {self.current_year}
        - Current Month: {self.current_date.strftime('%B %Y')}
        - Current Quarter: Q{((self.current_month - 1) // 3) + 1} {self.current_year}
        - Current DateTime: {self.current_datetime_tz.strftime('%Y-%m-%d %H:%M:%S %Z')}
        
        DATE CALCULATION HELPERS:
        - This year means: {self.current_year}
        - Last year means: {self.current_year - 1}
        - This month means: {self.current_date.strftime('%Y-%m')}
        - Last month: {last_month.strftime('%Y-%m')}
        - This quarter: Q{((self.current_month - 1) // 3) + 1} {self.current_year}
        - Year to date (YTD): From {self.current_year}-01-01 to {self.current_date}
        - Last 30 days: From {last_30_days} to {self.current_date}
        - Last 90 days: From {last_90_days} to {self.current_date}
        - Last 6 months: From {last_6_months} to {self.current_date}
        """
        
    def _get_matter_field_mapping(self) -> Dict[str, str]:
        """Define the mapping between simplified field names and actual Litify API names"""
        return {
            # Basic identification
            'Id': 'Id',
            'Name': 'Name',
            'Display_Name': 'litify_pm__Display_Name__c',
            'Case_Number': 'Case_Number__c',
            'DB_ID': 'DB_ID__c',
            
            # Client information
            'Client_Id': 'litify_pm__Client__c',
            
            # Dates
            'Open_Date': 'litify_pm__Open_Date__c',
            'Close_Date': 'litify_pm__Close_Date__c',
            'Closed_Date': 'litify_pm__Closed_Date__c',
            'Incident_Date': 'litify_pm__Incident_date__c',
            'Filed_Date': 'litify_pm__Filed_Date__c',
            'Trial_Date': 'litify_pm__Trial_Date__c',
            'Statute_Of_Limitations': 'litify_pm__Statute_Of_Limitations__c',
            'Created_Date': 'CreatedDate',
            'Last_Modified_Date': 'LastModifiedDate',
            
            # Case details
            'Status': 'litify_pm__Status__c',
            'Case_Type_Id': 'litify_pm__Case_Type__c',
            'Record_Type_Id': 'RecordTypeId',
            'Practice_Area': 'litify_pm__Practice_Area2__c',
            'Case_Title': 'litify_pm__Case_Title__c',
            'Description': 'litify_pm__Description__c',
            'Docket_Number': 'litify_pm__Docket_Number__c',
            'Court': 'litify_pm__Court__c',
            'Opposing_Party': 'litify_pm__OpposingParty__c',
            
            # Attorneys and team
            'Principal_Attorney': 'litify_pm__Principal_Attorney__c',
            'Originating_Attorney': 'litify_pm__Originating_Attorney__c',
            'Case_Manager': 'litify_pm__lit_Case_Manager__c',
            'Default_Matter_Team': 'litify_pm__Default_Matter_Team__c',
            'Owner_Id': 'OwnerId',
            
            # Financial information
            'Billing_Type': 'litify_pm__Billing_Type__c',
            'Contingency_Fee_Rate': 'litify_pm__Contingency_Fee_Rate__c',
            'Hourly_Rate': 'litify_pm__Hourly_Rate__c',
            'Total_Matter_Value': 'litify_pm__Total_Matter_Value__c',
            'Gross_Recovery': 'litify_pm__Gross_Recovery__c',
            'Net_Recovery': 'litify_pm__Net_Recovery__c',
            'Total_Damages': 'litify_pm__Total_Damages__c',
            'Hard_Costs': 'litify_pm__Hard_Costs__c',
            'Soft_Costs': 'litify_pm__Soft_Costs__c',
            'Fee_Amount': 'litify_pm__Fee_Amount__c',
            'Net_Attorney_Fee': 'litify_pm__Net_Attorney_Fee__c',
            'Amount_Due_to_Client': 'litify_pm__Amount_Due_to_Client__c',
            'Total_Client_Payout': 'litify_pm__lit_Total_Client_Payout__c',
        }
    
    def execute_query(self, query: str) -> List[Dict]:
        """Execute query against Salesforce/Litify database"""
        
        try:
            # Clean up the query
            soql_query = query.strip()
            if not soql_query.upper().startswith('SELECT'):
                return []
            
            print(f"🔍 Executing SOQL: {soql_query}")
            
            # Execute the SOQL query
            results = self.salesforce_api.execute_soql_query(soql_query)
            
            # Format results consistently
            formatted_results = []
            for record in results:
                # Handle nested objects from Salesforce
                flat_record = {}
                for key, value in record.items():
                    if isinstance(value, dict) and value is not None:
                        # Handle related objects
                        for nested_key, nested_value in value.items():
                            if nested_key != 'attributes':
                                flat_record[f"{key}.{nested_key}"] = nested_value
                    else:
                        flat_record[key] = value if value is not None else ""
                
                formatted_results.append(flat_record)
            
            return formatted_results
            
        except Exception as e:
            print(f"❌ Query execution error: {e}")
            return []
    
    def generate_soql_query(self, user_question: str) -> str:
        """Generate SOQL query from natural language using OpenAI"""
        
        if not self.openai_client:
            return "SELECT COUNT(Id) FROM litify_pm__Matter__c"
        
        system_prompt = f"""You are an expert Salesforce developer who specializes in legal databases and SOQL queries. 
        
        {self.get_date_context()}
        
        Convert natural language questions into accurate SOQL queries for the Salesforce/Litify database.
        
        The database has a main object called 'litify_pm__Matter__c' with these key fields:
        
        IDENTIFICATION: Id, Name, litify_pm__Display_Name__c, Case_Number__c
        STATUS & DATES: litify_pm__Status__c (Active, Closed, Pending, Open), litify_pm__Open_Date__c, litify_pm__Close_Date__c, CreatedDate
        FINANCIAL: litify_pm__Total_Matter_Value__c, litify_pm__Gross_Recovery__c, litify_pm__Net_Recovery__c, litify_pm__Fee_Amount__c
        SPECIAL: IsMinor__c, IsDeceased__c, Serious_Injury__c, litify_pm__Billable_Matter__c
        ATTORNEYS: litify_pm__Principal_Attorney__c, litify_pm__Originating_Attorney__c
        
        SOQL Rules:
        - Use COUNT(Id) instead of COUNT(*)
        - Always use FROM litify_pm__Matter__c
        - For boolean fields use = true or = false
        - For dates use YYYY-MM-DD format
        - For aggregation, use SUM(), AVG(), COUNT()
        
        DATE EXAMPLES:
        - "this year": WHERE litify_pm__Open_Date__c >= {self.current_year}-01-01
        - "last 30 days": WHERE litify_pm__Open_Date__c >= {self.current_date - timedelta(days=30)}
        - "YTD": WHERE litify_pm__Open_Date__c >= {self.current_year}-01-01
        
        Return ONLY the SOQL query, no explanations."""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Convert this question to SOQL: {user_question}"}
                ],
                max_tokens=200,
                temperature=0.1
            )
            
            soql_query = response.choices[0].message.content.strip()
            soql_query = soql_query.replace('```sql', '').replace('```soql', '').replace('```', '').strip()
            
            return soql_query
            
        except Exception as e:
            print(f"Error generating SOQL: {e}")
            return "SELECT COUNT(Id) FROM litify_pm__Matter__c"
    
    def analyze_results(self, user_question: str, query_results: List[Dict]) -> str:
        """Analyze query results using OpenAI"""
        
        if not self.openai_client:
            if query_results and len(query_results) > 0:
                if 'expr0' in query_results[0]:
                    return f"Result: {query_results[0]['expr0']}"
                else:
                    return f"Found {len(query_results)} records."
            return "No results found."
        
        system_prompt = f"""You are a legal data analyst who provides SHORT, DIRECT answers about legal database results.
        
        {self.get_date_context()}
        
        Analyze Salesforce/Litify database results and provide clear, concise business insights.
        
        Response style:
        - Keep answers SHORT (2-4 sentences maximum)
        - Start with the direct answer to the question
        - Use exact numbers from the database
        - Include relevant business insights
        - Format currency properly
        - Provide date context when relevant
        
        Stay factual and professional."""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Question: {user_question}\nDatabase Results: {query_results}\n\nProvide a short, direct answer."}
                ],
                max_tokens=150,
                temperature=0.1
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error analyzing results: {e}")
            if query_results and len(query_results) > 0:
                if 'expr0' in query_results[0]:
                    value = query_results[0]['expr0']
                    if isinstance(value, (int, float)) and value > 1000:
                        return f"Result: ${value:,.0f}"
                    return f"Result: {value}"
                else:
                    return f"Found {len(query_results)} records."
            return "No results found."
    
    def process_query_with_transparency(self, user_query: str) -> Tuple[str, str, List[Dict]]:
        """Process user query and return response, SOQL, and raw results for transparency"""
        
        try:
            # Step 1: Generate SOQL Query
            soql_query = self.generate_soql_query(user_query)
            
            # Step 2: Execute the query
            query_results = self.execute_query(soql_query)
            
            if not query_results and soql_query != "SELECT COUNT(Id) FROM litify_pm__Matter__c":
                # Try fallback
                fallback_query = "SELECT COUNT(Id) FROM litify_pm__Matter__c"
                query_results = self.execute_query(fallback_query)
                soql_query = fallback_query
            
            # Step 3: Analyze results
            response = self.analyze_results(user_query, query_results)
            
            return response, soql_query, query_results
            
        except Exception as e:
            print(f"Error processing query: {e}")
            return "I encountered an error processing your question. Please try rephrasing it.", "", []
    
    def process_query(self, user_query: str) -> str:
        """Process user query through the AI system (backward compatibility)"""
        response, _, _ = self.process_query_with_transparency(user_query)
        return response
    
    # Enhanced chat methods for conversation handling
    def process_chat(self, user_message: str, conversation_history: list = None) -> str:
        """Process chat message with proper conversational context understanding"""
        
        if conversation_history is None:
            conversation_history = []
        
        # STEP 1: Check if this is a follow-up question that needs conversational context
        context_info = self._analyze_conversational_context(user_message, conversation_history)
        
        if context_info['needs_context']:
            # This is a follow-up question - generate query with context
            return self._handle_contextual_query(user_message, conversation_history, context_info)
        else:
            # This is a new question - check simple history first, then database
            history_context = self._check_simple_history(user_message, conversation_history)
            if history_context:
                return self._generate_history_based_response(user_message, history_context, conversation_history)
            else:
                return self._query_database_and_respond(user_message, conversation_history)

    def process_chat_with_transparency(self, user_message: str, conversation_history: list = None) -> Tuple[str, str, List[Dict]]:
        """Process chat message and return response, SOQL, and raw results for transparency"""
        
        if conversation_history is None:
            conversation_history = []
        
        # STEP 1: Check if this is a follow-up question that needs conversational context
        context_info = self._analyze_conversational_context(user_message, conversation_history)
        
        if context_info['needs_context']:
            # This is a follow-up question - generate query with context
            response = self._handle_contextual_query(user_message, conversation_history, context_info)
            # For contextual queries, try to extract the SOQL and results from session state
            soql = st.session_state.get('last_contextual_soql', '')
            results = st.session_state.get('last_contextual_results', [])
            return response, soql, results
        else:
            # This is a new question - check simple history first, then database
            history_context = self._check_simple_history(user_message, conversation_history)
            if history_context:
                response = self._generate_history_based_response(user_message, history_context, conversation_history)
                return response, '', []  # No new query for history-based responses
            else:
                return self._query_database_and_respond_with_transparency(user_message, conversation_history)

    def _query_database_and_respond_with_transparency(self, user_message: str, conversation_history: list) -> Tuple[str, str, List[Dict]]:
        """Query database and generate response with transparency data"""
        
        try:
            # Generate SOQL query
            soql_query = self.generate_soql_query(user_message)
            
            # Execute the SOQL query
            print(f"🔍 Executing SOQL: {soql_query}")
            query_results = self.execute_query(soql_query)
            
            if not query_results:
                # Query failed, try a simple fallback
                fallback_query = "SELECT COUNT(Id) FROM litify_pm__Matter__c"
                query_results = self.execute_query(fallback_query)
                if query_results:
                    soql_query = fallback_query
                    print(f"🔄 Fallback query used: {fallback_query}")
                else:
                    return "I'm having trouble accessing the database right now. Could you try again?", "", []
            
            # Generate response with database results
            response = self._generate_chat_response_simple(user_message, conversation_history, query_results, soql_query)
            
            return response, soql_query, query_results
            
        except Exception as e:
            print(f"Database query error: {e}")
            return "I encountered an error while accessing the database. Could you try rephrasing your question?", "", []

    def _analyze_conversational_context(self, user_message: str, conversation_history: list) -> dict:
        """Analyze if the user message needs conversational context from previous questions"""
        
        if not conversation_history:
            return {'needs_context': False}
        
        # Look for follow-up indicators - expanded list for better detection
        follow_up_phrases = [
            'these cases', 'those cases', 'these matters', 'those matters',
            'them', 'those', 'these', 'that group', 'same cases', 'same matters',
            'how much money', 'what value', 'total value', 'financial',
            'how many of those', 'what about those', 'details on those',
            'breakdown of these', 'more info on these', 'the ones that are',
            'just the ones', 'only the ones', 'the active', 'the open',
            'active or open', 'those active', 'those open', 'value in money',
            'money that', 'all these cases', 'together'
        ]
        
        user_lower = user_message.lower()
        
        # Check if this looks like a follow-up question
        is_followup = any(phrase in user_lower for phrase in follow_up_phrases)
        
        # Additional check for contextual references
        contextual_words = ['these', 'those', 'them', 'that', 'the ones', 'just the', 'only the']
        has_contextual_reference = any(word in user_lower for word in contextual_words)
        
        if not is_followup and not has_contextual_reference:
            return {'needs_context': False}
        
        # Find the most recent question that defined a specific subset of data
        context_from_history = None
        
        # Look through recent conversation for context - check more history
        for msg in reversed(conversation_history[-5:]):  # Check last 5 exchanges
            user_question = msg['user'].lower()
            assistant_response = msg['assistant'].lower()
            
            # Look for questions that defined a specific subset - improved detection
            subset_indicators = [
                'active', 'open', 'closed', 'pending', 'status', 'type', 'involve', 
                'with', 'that have', 'where', 'matters', 'cases', 'minors', 
                'currently', 'this year', 'last month', 'recent'
            ]
            
            if any(word in user_question for word in subset_indicators):
                # This question likely defined a subset - extract the context
                context_from_history = {
                    'original_question': msg['user'],
                    'original_response': msg['assistant'],
                    'subset_description': user_question
                }
                break
        
        return {
            'needs_context': True,
            'context': context_from_history,
            'followup_type': 'financial' if any(word in user_lower for word in ['money', 'value', 'financial', 'cost', 'revenue', 'total', 'amount']) else 'general'
        }

    def _handle_contextual_query(self, user_message: str, conversation_history: list, context_info: dict) -> str:
        """Handle follow-up questions that need conversational context"""
        
        if not context_info.get('context'):
            return "I'm not sure which cases you're referring to. Could you clarify?"
        
        original_question = context_info['context']['original_question']
        followup_type = context_info['followup_type']
        
        try:
            # Generate contextual SOQL using OpenAI
            contextual_soql = self._generate_contextual_soql(original_question, user_message)
            
            print(f"🔍 Contextual SOQL: {contextual_soql}")
            
            # Execute the contextual query
            query_results = self.execute_query(contextual_soql)
            
            # Store for transparency
            st.session_state.last_contextual_soql = contextual_soql
            st.session_state.last_contextual_results = query_results
            
            if not query_results:
                return f"I understand you're asking about the {original_question.lower()}, but I couldn't retrieve the data. Could you try rephrasing?"
            
            # Generate response with context awareness
            return self._generate_contextual_response_simple(user_message, original_question, query_results)
            
        except Exception as e:
            print(f"Contextual query error: {e}")
            return f"I understand you're asking about the cases from '{original_question}', but I had trouble processing that. Could you try being more specific?"

    def _generate_contextual_soql(self, original_question: str, followup_question: str) -> str:
        """Generate SOQL query that combines context from previous question with new follow-up"""
        
        if not self.openai_client:
            return "SELECT COUNT(Id) FROM litify_pm__Matter__c"
        
        system_prompt = f"""You are an expert at understanding conversational context in legal database queries.
        
        {self.get_date_context()}
        
        You understand that when someone asks a follow-up question, they want to apply it to the same subset of data from the previous question.
        
        Available Litify fields:
        - litify_pm__Status__c (values: 'Active', 'Open', 'Closed', 'Pending')
        - IsMinor__c (true/false for cases involving minors)
        - IsDeceased__c (true/false)
        - Serious_Injury__c (true/false)
        - litify_pm__Total_Matter_Value__c (case value)
        - litify_pm__Gross_Recovery__c (recovery amount)
        - litify_pm__Net_Recovery__c (net recovery)
        - litify_pm__Fee_Amount__c (fee amount)
        - litify_pm__Total_Amount_Billed__c (billed amount)
        - litify_pm__Billing_Type__c (Contingency, Hourly, etc.)
        - litify_pm__Open_Date__c, litify_pm__Close_Date__c
        - litify_pm__Principal_Attorney__c
        - litify_pm__Case_Type__c
        
        CRITICAL: For active/open cases, use WHERE (litify_pm__Status__c = 'Active' OR litify_pm__Status__c = 'Open')
        
        Examples of contextual queries:
        
        Previous: "How many matters are currently active or open?"
        Follow-up: "How much money is involved in these cases?"
        → SELECT SUM(litify_pm__Total_Matter_Value__c) FROM litify_pm__Matter__c WHERE (litify_pm__Status__c = 'Active' OR litify_pm__Status__c = 'Open') AND litify_pm__Total_Matter_Value__c != null
        
        Previous: "How many matters involve minors?"
        Follow-up: "What's the total value of those?"
        → SELECT SUM(litify_pm__Total_Matter_Value__c) FROM litify_pm__Matter__c WHERE IsMinor__c = true AND litify_pm__Total_Matter_Value__c != null
        
        Previous: "Show me active cases"
        Follow-up: "What's the total value of those?"
        → SELECT SUM(litify_pm__Total_Matter_Value__c) FROM litify_pm__Matter__c WHERE (litify_pm__Status__c = 'Active' OR litify_pm__Status__c = 'Open') AND litify_pm__Total_Matter_Value__c != null
        
        Previous: "Cases opened this year"
        Follow-up: "How much have we recovered from them?"
        → SELECT SUM(litify_pm__Gross_Recovery__c) FROM litify_pm__Matter__c WHERE litify_pm__Open_Date__c >= {self.current_year}-01-01 AND litify_pm__Gross_Recovery__c != null
        
        IMPORTANT RULES:
        1. Always maintain the WHERE clause from the original question
        2. For "active or open" always use: (litify_pm__Status__c = 'Active' OR litify_pm__Status__c = 'Open')
        3. Add appropriate null checks for numeric fields
        4. Use SUM() for financial totals, COUNT() for counts, AVG() for averages
        
        Return ONLY the SOQL query."""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"PREVIOUS QUESTION: '{original_question}'\nFOLLOW-UP QUESTION: '{followup_question}'\n\nGenerate a SOQL query that applies the follow-up to the same subset from the previous question:"}
                ],
                max_tokens=200,
                temperature=0.1
            )
            
            soql_query = response.choices[0].message.content.strip()
            soql_query = soql_query.replace('```sql', '').replace('```soql', '').replace('```', '').strip()
            soql_query = soql_query.replace('SOQL:', '').replace('Query:', '').strip()
            
            print(f"🔍 Generated contextual SOQL: {soql_query}")
            
            return soql_query
            
        except Exception as e:
            print(f"Error generating contextual SOQL: {e}")
            return "SELECT COUNT(Id) FROM litify_pm__Matter__c"

    def _generate_contextual_response_simple(self, user_message: str, original_question: str, query_results: list) -> str:
        """Generate response that acknowledges the conversational context"""
        
        if not self.openai_client:
            # Simple fallback response
            if query_results and len(query_results) > 0:
                value = query_results[0].get('expr0', 0)
                if 'money' in user_message.lower() or 'value' in user_message.lower():
                    return f"For those cases, the total value is ${value:,.0f}."
                else:
                    return f"For those cases, the result is {value}."
            return "I had trouble generating a response for that follow-up question."
        
        system_prompt = f"""Generate a conversational response that acknowledges the context and provides the requested information:

        ORIGINAL QUESTION: "{original_question}"
        FOLLOW-UP QUESTION: "{user_message}"
        DATABASE RESULTS: {query_results}
        
        Instructions:
        1. Acknowledge that you understand the connection to the previous question
        2. Provide the specific data requested
        3. Keep response SHORT (1-2 sentences)
        4. Use exact numbers from database results
        5. Format currency properly
        
        Examples:
        
        Original: "How many matters involve minors?"
        Follow-up: "How much money is involved in these cases?"
        Results: [{{'expr0': 250000}}]
        Response: "For the matters involving minors, the total value is $250,000."
        
        Original: "Show me active cases"  
        Follow-up: "What's the average value?"
        Results: [{{'expr0': 45000}}]
        Response: "The average value of those active cases is $45,000."
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Generate contextual response using exact database values"}
                ],
                max_tokens=100,
                temperature=0.1
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Contextual response error: {e}")
            # Fallback response
            if query_results and len(query_results) > 0:
                value = query_results[0].get('expr0', 0)
                if 'money' in user_message.lower() or 'value' in user_message.lower():
                    return f"For those cases, the total value is ${value:,.0f}."
                else:
                    return f"For those cases, the result is {value}."
            return "I had trouble generating a response for that follow-up question."

    def _query_database_and_respond(self, user_message: str, conversation_history: list) -> str:
        """Query database and generate response with the results"""
        
        try:
            # Generate SOQL query
            soql_query = self.generate_soql_query(user_message)
            
            # Execute the SOQL query
            print(f"🔍 Executing SOQL: {soql_query}")
            query_results = self.execute_query(soql_query)
            
            if not query_results:
                # Query failed, try a simple fallback
                fallback_query = "SELECT COUNT(Id) FROM litify_pm__Matter__c"
                query_results = self.execute_query(fallback_query)
                if query_results:
                    soql_query = fallback_query
                    print(f"🔄 Fallback query used: {fallback_query}")
                else:
                    return "I'm having trouble accessing the database right now. Could you try again?"
            
            # Generate response with database results
            return self._generate_chat_response_simple(user_message, conversation_history, query_results, soql_query)
            
        except Exception as e:
            print(f"Database query error: {e}")
            return "I encountered an error while accessing the database. Could you try rephrasing your question?"

    def _generate_chat_response_simple(self, user_message: str, conversation_history: list, query_results: list, soql_query: str) -> str:
        """Generate chat response with or without database results"""
        
        if not self.openai_client:
            # Simple fallback without OpenAI
            if query_results and len(query_results) > 0:
                if 'expr0' in query_results[0]:
                    value = query_results[0]['expr0']
                    if isinstance(value, (int, float)) and value > 1000:
                        return f"Result: ${value:,.0f}"
                    return f"Result: {value}"
                else:
                    return f"Found {len(query_results)} records."
            return "No results found."
        
        # Build conversation context
        context = ""
        if conversation_history:
            context = "Recent conversation:\n"
            for msg in conversation_history[-2:]:
                context += f"User: {msg['user']}\nAssistant: {msg['assistant']}\n\n"
        
        # Build database context
        database_context = f"""
        DATABASE QUERY EXECUTED: {soql_query}
        DATABASE RESULTS: {query_results}
        
        These are REAL results from the Salesforce/Litify database. Use these exact numbers and data to answer.
        """
        
        system_prompt = f"""Answer this user message in a SHORT, DIRECT, conversational way:

        {context}
        Current User Message: "{user_message}"
        {database_context}
        
        CRITICAL INSTRUCTIONS:
        1. Keep answers SHORT (1-3 sentences maximum)
        2. USE THE EXACT DATABASE RESULTS - don't make up numbers
        3. Be direct and factual with the database numbers
        4. For greetings/general chat, be friendly and mention you can help with legal data
        
        EXAMPLES:
        
        With database results showing COUNT = 150:
        ✅ Good: "We have 150 matters in the system."
        ❌ Bad: "Based on the data, it appears there might be around 150 matters."
        
        With database results showing SUM = 2500000:
        ✅ Good: "The total matter value is $2,500,000."
        ❌ Bad: "The database indicates significant value."
        
        NEVER make up specific numbers that aren't in the database results!
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Generate a short, direct response using the exact database values"}
                ],
                max_tokens=150,
                temperature=0.1
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Debug logging
            print(f"✅ Used database query: {soql_query}")
            print(f"✅ Results count: {len(query_results)}")
                
            return response_text
            
        except Exception as e:
            print(f"Chat response error: {e}")
            return "I'm having trouble processing that request. Could you try asking in a different way?"

    def _check_simple_history(self, user_message: str, conversation_history: list) -> str:
        """Simple check for exact matches in conversation history (not contextual follow-ups)"""
        
        if not conversation_history:
            return ""
        
        # Only check for very similar questions, not contextual follow-ups
        user_lower = user_message.lower()
        
        # Skip if this looks like a contextual follow-up - EXPANDED detection
        contextual_words = [
            'these', 'those', 'them', 'that group', 'same', 'the ones', 
            'just the', 'only the', 'active or open', 'money', 'value', 
            'financial', 'total', 'amount', 'how much', 'what value'
        ]
        if any(word in user_lower for word in contextual_words):
            print(f"DEBUG - Skipping history check for contextual follow-up: {user_message}")
            return ""
        
        # Look for very similar questions in history (only for non-contextual questions)
        for msg in conversation_history[-3:]:
            hist_question = msg['user'].lower()
            
            # Check if questions are very similar (same key words)
            user_words = set(user_lower.split())
            hist_words = set(hist_question.split())
            
            # If 80% of words match, consider it the same question (increased threshold)
            if len(user_words.intersection(hist_words)) / len(user_words.union(hist_words)) > 0.8:
                print(f"DEBUG - Found similar question in history: {hist_question}")
                return msg['assistant']
        
        return ""

    def _generate_history_based_response(self, user_message: str, history_context: str, conversation_history: list) -> str:
        """Generate response based on conversation history"""
        
        if not self.openai_client:
            return f"Based on our earlier discussion, {history_context}"
        
        # Build recent conversation context
        context = ""
        if conversation_history:
            context = "Recent conversation:\n"
            for msg in conversation_history[-2:]:
                context += f"User: {msg['user']}\nAssistant: {msg['assistant']}\n\n"
        
        system_prompt = f"""Answer this user question using information from our conversation history:

        {context}
        Current User Message: "{user_message}"
        
        Relevant Information from History: {history_context}
        
        Instructions:
        1. Use the information from our conversation history to answer
        2. Keep response SHORT (1-2 sentences)
        3. Reference that this info was discussed earlier if appropriate
        4. Be conversational and natural
        5. Don't repeat unnecessary details
        
        Example responses:
        - "As I mentioned earlier, we have 15 legal matters."
        - "Based on our previous discussion, the total value is $2.3M."
        - "We covered this - there are 7 closed cases from this quarter."
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Generate response using history information"}
                ],
                max_tokens=100,
                temperature=0.1
            )
            
            response_text = response.choices[0].message.content.strip()
            print(f"DEBUG - Used conversation history for: {user_message}")
            return response_text
            
        except Exception as e:
            print(f"History response error: {e}")
            # Fallback to simple response
            return f"Based on our earlier discussion, {history_context}"

# TRANSPARENCY FEATURES

def create_transparency_panel(response: str, soql_query: str, query_results: List[Dict], user_question: str, panel_id: str = None):
    """Create transparency panel showing SOQL and interactive table"""
    
    # Only show if we have actual query data
    if not soql_query or not query_results:
        return
    
    # Use provided panel_id or create unique key for this transparency panel
    if panel_id is None:
        panel_id = f"transparency_{hash(soql_query + str(len(query_results)))}"
    
    # Check if this panel should be shown (from session state)
    show_key = f"show_transparency_{panel_id}"
    
    # Transparency button - MOVED OUTSIDE OF FORMS
    if st.button("🔍 Show Query Details & Data", key=f"show_{panel_id}", help="See the SOQL query used and explore the data"):
        st.session_state[show_key] = True
    
    # Show transparency panel if button was clicked
    if st.session_state.get(show_key, False):
        
        with st.expander("🔍 Query Transparency", expanded=True):
            
            # SOQL Query Section
            st.markdown("### 📝 SOQL Query Used")
            st.code(soql_query, language="sql")
            
            # Copy button for SOQL
            if st.button("📋 Copy Query", key=f"copy_{panel_id}"):
                st.success("✅ Query copied to clipboard! (Note: Manual copy from code block above)")
            
            st.markdown("---")
            
            # Data Table Section
            st.markdown("### 📊 Query Results Data")
            
            if not query_results:
                st.info("No data returned from this query.")
                return
            
            # Convert to DataFrame for better handling
            df = pd.DataFrame(query_results)
            
            if df.empty:
                st.info("No data returned from this query.")
                return
            
            # Show summary
            st.info(f"📈 **{len(df)} records** returned from Salesforce/Litify")
            
            # Get all available fields from Salesforce for this object
            all_fields = st.session_state.assistant.salesforce_api.get_all_fields_for_object("litify_pm__Matter__c")
            
            if all_fields:
                # Create field selector
                st.markdown("#### 🎛️ Customize Table Columns")
                
                # Current columns (from query results)
                current_columns = list(df.columns)
                
                # Available fields for selection
                field_options = []
                field_labels = {}
                
                for field in all_fields:
                    field_name = field['name']
                    field_label = field.get('label', field_name)
                    field_type = field.get('type', 'text')
                    
                    # Create display name
                    display_name = f"{field_label} ({field_name})"
                    if field_type:
                        display_name += f" - {field_type}"
                    
                    field_options.append(field_name)
                    field_labels[field_name] = display_name
                
                # Default selection: current columns + some key fields
                key_fields = ['Id', 'Name', 'litify_pm__Display_Name__c', 'litify_pm__Status__c', 
                             'litify_pm__Open_Date__c', 'litify_pm__Total_Matter_Value__c', 
                             'litify_pm__Principal_Attorney__c', 'CreatedDate']
                
                # Combine current columns with key fields, remove duplicates
                # IMPORTANT FIX: Only include fields that actually exist in field_options
                default_selection = []
                for field in current_columns + key_fields:
                    if field in field_options and field not in default_selection:
                        default_selection.append(field)
                
                # Limit to first 12 for initial display
                if len(default_selection) > 12:
                    default_selection = default_selection[:12]
                
                # Multi-select for columns
                selected_fields = st.multiselect(
                    "Select fields to display in table:",
                    options=field_options,
                    default=default_selection,
                    format_func=lambda x: field_labels.get(x, x),
                    key=f"fields_{panel_id}",
                    help="Choose which fields to show in the data table. Key fields are highlighted by default."
                )
                
                if selected_fields:
                    # Create new query with selected fields
                    if st.button("🔄 Refresh Table with Selected Fields", key=f"refresh_{panel_id}"):
                        with st.spinner("Fetching data with selected fields..."):
                            # Build new SOQL query with selected fields
                            try:
                                if 'WHERE' in soql_query.upper():
                                    # Split on WHERE and reconstruct
                                    query_parts = soql_query.split('WHERE', 1)
                                    if len(query_parts) == 2:
                                        where_clause = query_parts[1].strip()
                                        new_query = f"SELECT {', '.join(selected_fields)} FROM litify_pm__Matter__c WHERE {where_clause}"
                                    else:
                                        new_query = f"SELECT {', '.join(selected_fields)} FROM litify_pm__Matter__c"
                                elif 'FROM' in soql_query.upper():
                                    # Extract the FROM clause and any additional clauses
                                    parts = soql_query.split('FROM', 1)
                                    if len(parts) > 1:
                                        from_part = parts[1].strip()
                                        # Check if there are additional clauses after the table name
                                        from_parts = from_part.split()
                                        if len(from_parts) > 1:
                                            table_name = from_parts[0]
                                            additional_clauses = ' '.join(from_parts[1:])
                                            new_query = f"SELECT {', '.join(selected_fields)} FROM {table_name} {additional_clauses}"
                                        else:
                                            new_query = f"SELECT {', '.join(selected_fields)} FROM {from_part}"
                                    else:
                                        new_query = f"SELECT {', '.join(selected_fields)} FROM litify_pm__Matter__c"
                                else:
                                    new_query = f"SELECT {', '.join(selected_fields)} FROM litify_pm__Matter__c"
                                
                                # Execute new query
                                new_results = st.session_state.assistant.execute_query(new_query)
                                
                                if new_results:
                                    df = pd.DataFrame(new_results)
                                    st.success(f"✅ Updated table with {len(df)} records and {len(selected_fields)} fields")
                                else:
                                    st.error("❌ Failed to fetch data with selected fields")
                                    
                            except Exception as e:
                                st.error(f"❌ Error building new query: {e}")
                                st.info("💡 Using original data instead")
            
            # Display the data table
            if not df.empty:
                st.markdown("#### 📋 Data Table")
                
                # Highlight key columns
                def highlight_key_columns(val, col_name):
                    """Highlight important columns"""
                    key_columns = ['Id', 'Name', 'litify_pm__Display_Name__c', 'litify_pm__Status__c']
                    if col_name in key_columns:
                        return 'background-color: #e8f4f8; font-weight: bold;'
                    return ''
                
                try:
                    # Style the dataframe
                    styled_df = df.style.apply(lambda x: [highlight_key_columns(val, x.name) for val in x], axis=0)
                    
                    # Display with formatting
                    st.dataframe(
                        styled_df,
                        use_container_width=True,
                        height=400
                    )
                except Exception as e:
                    # Fallback to plain dataframe if styling fails
                    st.dataframe(
                        df,
                        use_container_width=True,
                        height=400
                    )
                
                # Download option
                try:
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="💾 Download Data as CSV",
                        data=csv,
                        file_name=f"legal_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        key=f"download_{panel_id}"
                    )
                except Exception as e:
                    st.warning(f"⚠️ Download not available: {e}")
            
            # Close button
            if st.button("❌ Close Query Details", key=f"close_{panel_id}"):
                st.session_state[show_key] = False
                st.rerun()

def display_enhanced_chat_message(message, is_user=True, message_id=None, show_transparency_data=None):
    """Display chat message with enhanced indicators and transparency option"""
    
    if is_user:
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>👤 You:</strong><br>
            {message}
        </div>
        """, unsafe_allow_html=True)
    else:
        # Check if this was a history-based response
        history_indicator = ""
        if (message_id is not None and 
            message_id < len(st.session_state.chat_history) and 
            st.session_state.chat_history[message_id].get('used_history', False)):
            history_indicator = '<span style="color: #28a745; font-size: 12px;">💭 From conversation history</span><br>'
        
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <strong>🤖 Assistant:</strong><br>
            {history_indicator}
            {message}
        </div>
        """, unsafe_allow_html=True)
        
        # Add transparency panel if we have query data
        # FIXED: Create unique panel ID and move outside form context
        if show_transparency_data and len(show_transparency_data) == 3:
            response, soql_query, query_results = show_transparency_data
            if soql_query and query_results:  # Only show if we have actual query data
                unique_panel_id = f"chat_msg_{message_id}_{hash(str(show_transparency_data))}"
                create_transparency_panel(response, soql_query, query_results, message, unique_panel_id)

def get_smart_placeholder(chat_history):
    """Generate smart placeholder text based on conversation history"""
    
    if not chat_history:
        return "e.g., How many matters do we have?"
    
    # Analyze last few messages to suggest follow-ups
    recent_topics = []
    for msg in chat_history[-3:]:
        assistant_response = msg['assistant'].lower()
        if 'matter' in assistant_response or 'case' in assistant_response:
            recent_topics.append('matters')
        if 'attorney' in assistant_response or 'lawyer' in assistant_response:
            recent_topics.append('attorneys')
        if 'value' in assistant_response or 'recovery' in assistant_response or ' in assistant_response':
            recent_topics.append('financial')
        if 'active' in assistant_response or 'closed' in assistant_response or 'pending' in assistant_response:
            recent_topics.append('status')
    
    # Smart suggestions based on recent topics
    if 'matters' in recent_topics and 'attorneys' not in recent_topics:
        return "e.g., Which attorney handles those matters?"
    elif 'attorneys' in recent_topics:
        return "e.g., How many matters does each attorney have?"
    elif 'financial' in recent_topics:
        return "e.g., What's the breakdown by case type?"
    elif 'status' in recent_topics:
        return "e.g., When were those cases opened?"
    else:
        return "e.g., Tell me more about that..."

def check_if_history_sufficient(user_input, chat_history):
    """Quick check to predict if history will be used (for UI indicators) - IMPROVED"""
    
    if not chat_history:
        return False
    
    # Simple heuristics for common follow-up patterns - EXPANDED
    follow_up_indicators = [
        'what about', 'how many of those', 'and the', 'tell me more', 
        'which ones', 'any others', 'what else', 'more details',
        'those cases', 'those matters', 'that attorney', 'these cases',
        'these matters', 'them', 'those', 'that group', 'the ones',
        'just the', 'only the', 'active or open', 'money', 'value',
        'how much', 'what value', 'total', 'amount'
    ]
    
    user_lower = user_input.lower()
    
    # Check if this looks like a follow-up question
    for indicator in follow_up_indicators:
        if indicator in user_lower:
            print(f"DEBUG - Detected follow-up indicator: {indicator}")
            return False  # Don't use simple history for follow-ups, use contextual processing
    
    # Check if asking about data mentioned in recent responses (for exact matches only)
    recent_responses = ' '.join([msg['assistant'].lower() for msg in chat_history[-2:]])
    
    # If user mentions numbers, names, or terms from recent responses (but not contextual)
    user_words = user_lower.split()
    for word in user_words:
        if len(word) > 3 and word in recent_responses:
            # Check if it's not a contextual reference
            contextual_words = ['these', 'those', 'them', 'that', 'the', 'active', 'open', 'money', 'value']
            if word not in contextual_words:
                return True
    
    return False

# Custom CSS for better styling with improved chat visibility and transparency panels
st.markdown("""
<style>
.main {
    padding-top: 1rem;
}

.stButton > button {
    width: 100%;
    border-radius: 10px;
    border: 1px solid #e0e0e0;
    padding: 0.5rem 1rem;
    margin: 0.25rem 0;
}

.stButton > button:hover {
    border-color: #1f77b4;
    background-color: #f0f8ff;
}

/* Chat message styling - applies to both custom and Streamlit default messages */
.chat-message,
[data-testid="stChatMessageContent"] {
    padding: 1.2rem;
    border-radius: 12px;
    margin: 0.8rem 0;
    border: 2px solid #ddd;
    font-size: 16px;
    line-height: 1.5;
}

/* User message styling */
.user-message,
[data-testid="stChatMessageContent"]:has([data-testid="chatAvatarIcon-user"]) {
    background-color: #0084ff;
    color: white;
    border-left: 4px solid #0066cc;
    margin-left: 20%;
}

.user-message strong,
[data-testid="stChatMessageContent"]:has([data-testid="chatAvatarIcon-user"]) strong {
    color: #ffffff !important;
    font-weight: bold;
}

/* Assistant message styling - targets both custom and Streamlit default */
.assistant-message,
[data-testid="stChatMessageContent"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background-color: #2c3e50 !important;
    color: #f8f9fa !important;
    border-left: 4px solid #28a745;
    margin-right: 20%;
}

.assistant-message strong,
[data-testid="stChatMessageContent"]:has([data-testid="chatAvatarIcon-assistant"]) strong {
    color: #28a745 !important;
    font-weight: bold;
}

/* Force text color in all messages */
.chat-message p,
.chat-message div,
.chat-message span,
[data-testid="stChatMessageContent"] p,
[data-testid="stChatMessageContent"] div,
[data-testid="stChatMessageContent"] span {
    color: inherit !important;
}

/* Specific fix for first assistant message */
.stChatMessage:first-child div:first-child div:first-child {
    background-color: #2c3e50 !important;
    color: #f8f9fa !important;
}

/* Transparency panel styling */
.stExpander {
    border: 2px solid #17a2b8;
    border-radius: 10px;
    margin: 1rem 0;
}

.stExpander > div:first-child {
    background-color: #d1ecf1;
    color: #0c5460;
    font-weight: bold;
}

/* Code block styling for SOQL */
.stCode {
    background-color: #f8f9fa;
    border: 1px solid #e9ecef;
    border-radius: 6px;
}

/* Dataframe styling */
.stDataFrame {
    border: 1px solid #dee2e6;
    border-radius: 8px;
}

.metric-container {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 10px;
    border: 1px solid #e9ecef;
    margin: 0.5rem 0;
}

.connection-status {
    padding: 0.5rem 1rem;
    border-radius: 8px;
    margin: 0.5rem 0;
    font-weight: bold;
}

.connected {
    background-color: #d4edda;
    color: #155724;
    border: 1px solid #c3e6cb;
}

.disconnected {
    background-color: #f8d7da;
    color: #721c24;
    border: 1px solid #f5c6cb;
}

.config-form {
    background-color: #f8f9fa;
    padding: 2rem;
    border-radius: 15px;
    border: 2px solid #e9ecef;
    margin: 1rem 0;
}

.config-success {
    background-color: #d1ecf1;
    color: #0c5460;
    padding: 1rem;
    border-radius: 10px;
    border: 1px solid #bee5eb;
    margin: 1rem 0;
}

/* Transparency button styling */
button[data-testid="baseButton-secondary"]:has([title*="Show Query Details"]) {
    background-color: #17a2b8;
    color: white;
    border: none;
    margin-top: 0.5rem;
}

button[data-testid="baseButton-secondary"]:has([title*="Show Query Details"]):hover {
    background-color: #138496;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_app():
    """Initialize the application and session state"""
    
    # Check if configuration is needed
    if not st.session_state.get('configured', False):
        st.session_state.show_config = True
        return False
    
    if 'assistant' not in st.session_state:
        try:
            with st.spinner("🔄 Initializing Legal AI Assistant with Salesforce/Litify..."):
                st.session_state.assistant = LegalAIAssistant()
                
                # Test Salesforce connection
                test_connection = st.session_state.assistant.salesforce_api.get_access_token()
                
                if test_connection:
                    st.session_state.initialized = True
                    st.session_state.connection_status = "connected"
                    st.success("✅ Legal AI Assistant initialized successfully!")
                    st.success("🔗 Connected to Salesforce/Litify database!")
                else:
                    st.session_state.initialized = False
                    st.session_state.connection_status = "disconnected"
                    st.error("❌ Failed to connect to Salesforce/Litify")
                    st.info("💡 Please check your Salesforce credentials")
                    return False
                    
        except Exception as e:
            st.session_state.initialized = False
            st.session_state.connection_status = "disconnected"
            st.error(f"❌ Failed to initialize Legal AI Assistant: {e}")
            st.info("💡 Make sure your credentials are correct and try again")
            return False

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if 'mode' not in st.session_state:
        st.session_state.mode = "Chat Mode"
    
    return st.session_state.get('initialized', False)

def display_connection_status():
    """Display Salesforce connection status"""
    status = st.session_state.get('connection_status', 'disconnected')
    
    if status == "connected":
        st.markdown("""
        <div class="connection-status connected">
            🔗 Connected to Salesforce/Litify Database
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="connection-status disconnected">
            ❌ Disconnected from Salesforce/Litify Database
        </div>
        """, unsafe_allow_html=True)

def display_chat_message(message, is_user=True):
    """Display a chat message with proper styling (backward compatibility)"""
    if is_user:
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>👤 You:</strong><br>
            {message}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <strong>🤖 Assistant:</strong><br>
            {message}
        </div>
        """, unsafe_allow_html=True)

def chat_mode():
    """Enhanced Chat Mode Interface with transparency features - FIXED FORM ISSUE"""
    st.markdown("### 💬 Chat with your Salesforce/Litify Database")
    st.markdown("💡 **Ask quick questions about your legal data in a conversational way!**")
    
    # Display connection status
    display_connection_status()
    
    # Enhanced examples with real field awareness
    with st.expander("💡 Example Questions"):
        st.markdown("""
        **Case & Matter Questions:**
        - "How many matters do we have?"
        - "Show me active cases"
        - "What matters opened this year?"
        - "How many new cases this month?"
        
        **Financial Questions:**
        - "What's our total matter value YTD?"
        - "Show me matters over $100k this year"
        - "How much recovered in the last month?"
        - "What's our average case value?"
        
        **Time-Based Questions:**
        - "Recent case openings?"
        - "Matters closed last quarter?"
        - "Year over year comparison?"
        - "This month's performance?"
        
        **Follow-up Examples:**
        - After asking "How many matters involve minors?" try "How much money is involved in these cases?"
        - After asking "Show me active cases" try "What's the total value of those?"
        
        **🔍 Transparency Feature:**
        - After each answer, click "Show Query Details & Data" to see the exact SOQL query used
        - Explore the actual database results in an interactive table
        - Customize which fields to display and download the data
        """)
    
    # Performance indicator
    if st.session_state.chat_history:
        total_messages = len(st.session_state.chat_history)
        st.info(f"🧠 **Smart Memory**: {total_messages} exchanges remembered • 🔍 **Full Transparency**: Click any 'Show Query Details' button to see SOQL queries and data")
    
    # Chat container
    chat_container = st.container()
    
    # Display chat history with enhanced indicators and transparency
    with chat_container:
        if st.session_state.chat_history:
            for i, msg in enumerate(st.session_state.chat_history):
                display_enhanced_chat_message(msg['user'], is_user=True, message_id=i)
                
                # Get transparency data if available
                transparency_data = None
                if 'transparency' in msg:
                    transparency_data = (msg['assistant'], msg['transparency'].get('soql', ''), msg['transparency'].get('results', []))
                
                display_enhanced_chat_message(
                    msg['assistant'], 
                    is_user=False, 
                    message_id=i,
                    show_transparency_data=transparency_data
                )
        else:
            st.info("👋 Start a conversation by asking a question about your legal matters!")
    
    # FIXED: Separate form handling and transparency display
    # Keep form simple - only for input
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([6, 1])
        with col1:
            # Smart placeholder based on conversation history
            placeholder = get_smart_placeholder(st.session_state.chat_history)
            user_input = st.text_input("Ask a question:", placeholder=placeholder)
        with col2:
            send_button = st.form_submit_button("Send", use_container_width=True)
    
    # Handle form submission OUTSIDE the form
    if send_button and user_input:
        # Check if we're connected to Salesforce
        if st.session_state.get('connection_status') != 'connected':
            st.error("❌ Not connected to Salesforce/Litify. Please check your credentials and try reconnecting.")
            return
        
        # Add user message to history immediately
        display_enhanced_chat_message(user_input, is_user=True, message_id=len(st.session_state.chat_history))
        
        # Enhanced response generation with transparency
        with st.spinner("🤖 Analyzing..."):
            try:
                # Check if we'll use history or database
                will_use_history = check_if_history_sufficient(user_input, st.session_state.chat_history)
                
                if will_use_history:
                    st.info("💭 Using conversation history (faster response)")
                else:
                    st.info("🔍 Querying Salesforce/Litify database for fresh data")
                
                # Get response with transparency data
                response, soql_query, query_results = st.session_state.assistant.process_chat_with_transparency(
                    user_input, st.session_state.chat_history
                )
                
                # Prepare transparency data
                transparency_data = None
                if soql_query and query_results:
                    transparency_data = (response, soql_query, query_results)
                
                # Display assistant response with transparency
                display_enhanced_chat_message(
                    response, 
                    is_user=False, 
                    message_id=len(st.session_state.chat_history),
                    show_transparency_data=transparency_data
                )
                
                # Prepare chat history entry
                chat_entry = {
                    'user': user_input,
                    'assistant': response,
                    'timestamp': datetime.now().isoformat(),
                    'used_history': will_use_history
                }
                
                # Add transparency data if available
                if soql_query and query_results:
                    chat_entry['transparency'] = {
                        'soql': soql_query,
                        'results': query_results
                    }
                
                # Add to chat history
                st.session_state.chat_history.append(chat_entry)
                
                # Rerun to update the display
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.info("💡 This might be a connection issue. Try refreshing the page or reconfiguring your credentials.")
    elif send_button:
        st.warning("Please enter a question.")

def custom_query_mode():
    """Custom Query Interface with transparency features - FIXED FORM ISSUE"""
    st.markdown("### 🔍 Custom Data Query")
    st.markdown("💡 **Ask any question about your Salesforce/Litify data in natural language!**")
    
    # Display connection status
    display_connection_status()
    
    if st.session_state.get('connection_status') != 'connected':
        st.warning("⚠️ Not connected to Salesforce/Litify. Please check your credentials and refresh the page.")
        return
    
    # Initialize session state for custom query results
    if 'custom_query_result' not in st.session_state:
        st.session_state.custom_query_result = None
    
    # Examples
    with st.expander("💡 Example Custom Questions"):
        current_date = st.session_state.assistant.current_date if st.session_state.get('assistant') else date.today()
        current_year = current_date.year
        current_month = current_date.strftime('%B')
        current_quarter = f"Q{((current_date.month - 1) // 3) + 1}"
        
        st.markdown(f"""
        **Financial Queries:**
        - "What's our average matter value?"
        - "Show me matters with recovery over $50,000"
        - "How much have we collected in contingency fees this year?"
        - "What's our total revenue year to date?"
        
        **Time-Based Queries (Current date: {current_date}):**
        - "How many matters opened this year?" 
        - "Show me cases filed in the last 6 months"
        - "What's our monthly intake for {current_month}?"
        - "Compare this year vs last year case volume"
        - "Show me matters closed in {current_quarter}"
        
        **Case Management:**
        - "How many matters are pending settlement?"
        - "Which matters have upcoming trial dates?"
        - "Show me cases past their statute of limitations"
        - "What's our average case duration?"
        
        **Attorney Analysis:**
        - "Which attorney opened the most cases this year?"
        - "Show me recent case assignments"
        - "How many new matters per attorney this quarter?"
        
        **🔍 Transparency:**
        - Every answer includes a "Show Query Details & Data" button
        - See the exact SOQL query that was executed
        - Explore the raw data in an interactive table
        - Download results as CSV
        """)
    
    # FIXED: Simple form for input only
    with st.form("custom_query_form"):
        query = st.text_area(
            "Enter your question:",
            placeholder="e.g., What's our total matter value for active cases?",
            height=120,
            help="Ask any question about your Salesforce/Litify database in plain English"
        )
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            submitted = st.form_submit_button("🔍 Ask Question", use_container_width=True)
    
    # Handle form submission OUTSIDE the form to avoid button conflicts
    if submitted and query:
        with st.spinner("🤖 Analyzing your question and querying Salesforce/Litify..."):
            try:
                response, soql_query, query_results = st.session_state.assistant.process_query_with_transparency(query)
                
                # Store results in session state so they persist across reruns
                st.session_state.custom_query_result = {
                    'query': query,
                    'response': response,
                    'soql': soql_query,
                    'results': query_results,
                    'timestamp': datetime.now().isoformat()
                }
                
                st.rerun()  # Rerun to display the results
                
            except Exception as e:
                st.error(f"❌ Error processing query: {e}")
                st.info("💡 Try rephrasing your question or check if your Salesforce connection is working.")
                
    elif submitted:
        st.warning("⚠️ Please enter a question before submitting.")
    
    # Display stored results if available
    if st.session_state.custom_query_result:
        result = st.session_state.custom_query_result
        
        st.markdown("---")
        st.markdown(f"**Your Question:** {result['query']}")
        st.success("✅ Query completed successfully!")
        
        # Display results
        st.markdown("**Answer:**")
        st.info(result['response'])
        
        # Add transparency panel with unique ID for custom query
        if result['soql'] and result['results']:
            unique_panel_id = f"custom_query_{hash(result['query'] + result['timestamp'])}"
            create_transparency_panel(result['response'], result['soql'], result['results'], result['query'], unique_panel_id)
        
        # Add button to clear results and ask new question
        if st.button("🗑️ Clear Results & Ask New Question", key="clear_custom_results"):
            st.session_state.custom_query_result = None
            st.rerun()

def predefined_questions_mode():
    """Predefined Questions Interface with transparency features - FIXED FORM ISSUE"""
    st.markdown("### 📋 Quick Demo Questions")
    st.markdown("💡 **Click any question to get instant insights from your Salesforce/Litify database**")
    
    # Display connection status
    display_connection_status()
    
    if st.session_state.get('connection_status') != 'connected':
        st.warning("⚠️ Not connected to Salesforce/Litify. Please check your credentials and refresh the page.")
        return
    
    # Demo queries
    demo_queries = [
        "How many total matters do we have in the system?",
        "What's our total matter value across all cases?",
        "How many matters are currently active or open?",
        "Show me the breakdown of matters by status",
        "How many contingency fee cases do we have?",
        "What's the total gross recovery amount?",
        "How many billable matters do we track?",
        "Show me matters opened this year",
        "How many cases involve minors?",
        "What's our total billed amount across all matters?",
        "How many matters have serious injuries?",
        "Show me the total hours tracked across all cases"
    ]
    
    # Initialize session state for tracking answered questions
    if 'answered_questions' not in st.session_state:
        st.session_state.answered_questions = {}
    
    # Display questions with answers below each one
    for i, query in enumerate(demo_queries):
        # Create a container for each question
        question_container = st.container()
        
        with question_container:
            # Question header with button
            col1, col2 = st.columns([5, 1])
            
            with col1:
                st.markdown(f"**{i+1}.** {query}")
            
            with col2:
                button_key = f"demo_{i}"
                # FIXED: These buttons are NOT in forms, so they work fine
                if st.button("🔍 Ask", key=button_key, use_container_width=True):
                    # Process the query when button is clicked
                    with st.spinner("🔄 Processing query..."):
                        try:
                            response, soql_query, query_results = st.session_state.assistant.process_query_with_transparency(query)
                            # Store the answer with transparency data in session state
                            st.session_state.answered_questions[i] = {
                                'response': response,
                                'soql': soql_query,
                                'results': query_results,
                                'status': 'success'
                            }
                            st.rerun()  # Refresh to show the answer
                        except Exception as e:
                            st.session_state.answered_questions[i] = {
                                'response': f"Error: {e}",
                                'soql': '',
                                'results': [],
                                'status': 'error'
                            }
                            st.rerun()
            
            # Display answer if this question has been answered
            if i in st.session_state.answered_questions:
                answer_data = st.session_state.answered_questions[i]
                
                if answer_data['status'] == 'success':
                    # Success answer styling
                    st.success("✅ Query completed!")
                    st.info(f"**Answer:** {answer_data['response']}")
                    
                    # Add transparency panel for predefined questions with unique ID
                    if answer_data.get('soql') and answer_data.get('results'):
                        unique_panel_id = f"predefined_{i}_{hash(query)}"
                        create_transparency_panel(
                            answer_data['response'], 
                            answer_data['soql'], 
                            answer_data['results'], 
                            query,
                            unique_panel_id
                        )
                    
                else:
                    # Error answer styling
                    st.error("❌ Query failed!")
                    st.error(answer_data['response'])
                    st.info("💡 Please check your Salesforce connection and configuration")
                
                # Add a clear button for this answer
                if st.button(f"🗑️ Clear Answer", key=f"clear_{i}", help="Clear this answer"):
                    del st.session_state.answered_questions[i]
                    st.rerun()
            
            # Add separator between questions (except for the last one)
            if i < len(demo_queries) - 1:
                st.markdown("---")
    
    # Add a clear all answers button at the bottom
    if st.session_state.answered_questions:
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🗑️ Clear All Answers", use_container_width=True):
                st.session_state.answered_questions = {}
                st.rerun()

def sidebar_content():
    """Enhanced sidebar content with configuration management"""
    st.sidebar.title("⚖️ Legal AI Assistant")
    st.sidebar.markdown("---")
    
    # Configuration status
    if st.session_state.get('configured', False):
        st.sidebar.success("✅ Configuration Complete")
        
        # Connection status in sidebar
        status = st.session_state.get('connection_status', 'disconnected')
        if status == "connected":
            st.sidebar.success("🔗 Connected to Salesforce/Litify")
        else:
            st.sidebar.error("❌ Salesforce/Litify Disconnected")
            if st.sidebar.button("🔄 Retry Connection"):
                if st.session_state.get('assistant'):
                    with st.spinner("Reconnecting..."):
                        success = st.session_state.assistant.salesforce_api.get_access_token()
                        if success:
                            st.session_state.connection_status = "connected"
                            st.rerun()
        
        # Reconfigure button
        if st.sidebar.button("🔧 Reconfigure Credentials"):
            st.session_state.configured = False
            st.session_state.show_config = True
            # Clear existing credentials
            for key in ['sf_client_id', 'sf_client_secret', 'sf_instance_url', 'openai_api_key']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
            
    else:
        st.sidebar.warning("⚠️ Configuration Required")
        st.sidebar.info("Please complete the configuration form to continue.")
    
    st.sidebar.markdown("---")
    
    # Mode selection (only if configured)
    if st.session_state.get('configured', False):
        st.sidebar.markdown("### 🎯 Select Mode")
        mode = st.sidebar.radio(
            "Choose how you want to interact:",
            ["📋 Predefined Questions", "🔍 Custom Query", "💬 Chat Mode"],
            index=2  # Default to Chat Mode
        )
        
        # Extract mode name
        st.session_state.mode = mode.split(" ", 1)[1]
        
        st.sidebar.markdown("---")
        
        # Transparency info
        st.sidebar.markdown("### 🔍 Transparency Features")
        st.sidebar.info("""
        **New in this version:**
        - 🔍 See exact SOQL queries used
        - 📊 Interactive data tables
        - 🎛️ Customize table columns
        - 💾 Download query results
        - 🔄 Refresh with selected fields
        
        Look for "Show Query Details & Data" buttons after each answer!
        """)
        
        st.sidebar.markdown("---")
        
        # Quick stats from Salesforce
        if st.session_state.get('initialized', False) and st.session_state.get('connection_status') == "connected":
            st.sidebar.markdown("### 📊 Live Database Stats")
            
            try:
                # Get quick statistics
                total_matters = st.session_state.assistant.execute_query("SELECT COUNT(Id) FROM litify_pm__Matter__c")
                
                if total_matters and len(total_matters) > 0:
                    count_key = 'expr0' if 'expr0' in total_matters[0] else list(total_matters[0].keys())[0]
                    st.sidebar.metric("📁 Total Matters", total_matters[0].get(count_key, 0))
                    
            except Exception as e:
                st.sidebar.warning("⚠️ Could not load live stats")
        
        st.sidebar.markdown("---")
        
        # Clear history
        if st.session_state.mode == "Chat Mode" and st.session_state.chat_history:
            if st.sidebar.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()
        
        # System info
        current_date = st.session_state.assistant.current_date.strftime('%B %d, %Y') if st.session_state.get('assistant') else 'Loading...'
        st.sidebar.markdown("### ℹ️ System Info")
        st.sidebar.success(f"""
        **📅 Current Date**: {current_date}
        
        **🔗 Live Connection**: Direct access to Salesforce/Litify
        
        **🤖 AI Powered**: OpenAI GPT for natural language processing
        
        **⚡ Real-time**: Live database queries
        
        **🔍 Transparent**: Full query visibility
        """)
    
    # Configuration help
    with st.sidebar.expander("🔧 Configuration Help"):
        st.sidebar.markdown("""
        **Setup Requirements:**
        - Salesforce Connected App with Client Credentials flow
        - OpenAI API key for AI processing
        - Proper permissions for Litify data access
        
        **Security:**
        - Credentials stored only in browser session
        - No permanent storage of sensitive data
        - Direct connection to your Salesforce instance
        
        **Transparency Features:**
        - Every query shows the exact SOQL used
        - Interactive data exploration
        - Field customization and CSV download
        """)
    
    return st.session_state.get('mode', 'Chat Mode')

def main():
    """Main application function with configuration flow"""
    
    # Check if we need to show configuration
    if st.session_state.get('show_config', True) and not st.session_state.get('configured', False):
        if configuration_form():
            st.rerun()
        return
    
    # Initialize the app
    if not initialize_app():
        st.stop()
    
    # Sidebar
    mode = sidebar_content()
    
    # Main content area
    st.title("⚖️ Legal AI Assistant")
    st.markdown("**Intelligent Salesforce/Litify Database Analysis with AI**")
    if st.session_state.get('assistant'):
        current_date = st.session_state.assistant.current_date.strftime('%B %d, %Y')
        st.markdown(f"*Natural language queries • Current date: {current_date} • Real-time data access • **🔍 Full transparency with query details***")
    else:
        st.markdown("*Natural language queries • Real-time data access • **🔍 Full transparency with query details***")
    st.markdown("---")
    
    # Route to appropriate mode
    if st.session_state.mode == "Predefined Questions":
        predefined_questions_mode()
    elif st.session_state.mode == "Custom Query":
        custom_query_mode()
    elif st.session_state.mode == "Chat Mode":
        chat_mode()

if __name__ == "__main__":
    main()