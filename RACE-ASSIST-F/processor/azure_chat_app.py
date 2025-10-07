from flask import Flask, render_template, request, jsonify, session
import json
import os
import uuid
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = ""

class RaceAzureAI:
    def __init__(self):
        self.setup_azure_client()
        self.load_unified_database()
        
    def setup_azure_client(self):
        """Initialize Azure AI Project Client - Mock for now"""
        try:
            # Mock Azure setup - replace with real Azure code when packages are installed
            self.azure_available = False  # Set to True when Azure packages are installed
            print("⚠️ Azure AI Agent - Mock mode (install Azure packages to enable)")
            
            # When Azure packages are installed, uncomment this:
            # from azure.ai.projects import AIProjectClient
            # from azure.identity import DefaultAzureCredential
            # from azure.ai.agents.models import ListSortOrder
            # self.project = AIProjectClient(
            #     credential=DefaultAzureCredential(),
            #     endpoint="https://varshitharesource.services.ai.azure.com/api/projects/varshitharesource"
            # )
            # self.agent = self.project.agents.get_agent("asst_dNrgMZm0ypOo4HLn4a059P7X")
            # self.azure_available = True
            
        except Exception as e:
            print(f"❌ Azure AI setup failed: {str(e)}")
            self.azure_available = False
    
    def load_unified_database(self):
        """Load the unified FAISS database"""
        try:
            # Load the unified metadata
            metadata_path = "faiss_databases/complete_unified_metadata.pkl"
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    self.unified_metadata = pickle.load(f)
                
                # Load the FAISS index
                index_path = "faiss_databases/complete_unified_index.faiss"
                if os.path.exists(index_path):
                    self.unified_index = faiss.read_index(index_path)
                    self.model = SentenceTransformer('all-MiniLM-L6-v2')
                    self.database_available = True
                    print("✅ Unified FAISS database loaded successfully!")
                else:
                    self.database_available = False
                    print("❌ Unified FAISS index not found")
            else:
                self.database_available = False
                print("❌ Unified metadata not found")
                
        except Exception as e:
            print(f"❌ Database loading failed: {str(e)}")
            self.database_available = False
    
    def search_unified_database(self, query: str, top_k: int = 3):
        """Search the unified database for relevant context"""
        if not self.database_available:
            return "No knowledge base available."
        
        try:
            # Encode query
            query_embedding = self.model.encode([query])
            query_embedding = np.array(query_embedding).astype('float32')
            faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = self.unified_index.search(query_embedding, top_k)
            
            # Get relevant chunks
            context_sections = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.unified_metadata['chunks']):
                    chunk = self.unified_metadata['chunks'][idx]
                    context_sections.append(f"Context {i+1}:\n{chunk}")
            
            return "\n\n".join(context_sections)
            
        except Exception as e:
            print(f"Search error: {str(e)}")
            return "Error retrieving information."
    
    def create_thread(self):
        """Create a new conversation thread"""
        if self.azure_available:
            try:
                thread = self.project.agents.threads.create()
                print(f"Created thread, ID: {thread.id}")
                return thread.id
            except Exception as e:
                print(f"Error creating thread: {str(e)}")
                return None
        else:
            # Generate a mock thread ID for local mode
            thread_id = f"local_thread_{uuid.uuid4().hex[:8]}"
            print(f"Created local thread, ID: {thread_id}")
            return thread_id
    
    def send_message(self, thread_id: str, user_message: str):
        """Send message to Azure AI Agent or use local knowledge base"""
        
        # If Azure is available, use Azure AI Agent
        if self.azure_available:
            try:
                # Create user message
                message = self.project.agents.messages.create(
                    thread_id=thread_id,
                    role="user",
                    content=user_message
                )
                
                # Run the agent
                run = self.project.agents.runs.create_and_process(
                    thread_id=thread_id,
                    agent_id=self.agent.id
                )
                
                if run.status == "failed":
                    print(f"Run failed: {run.last_error}")
                    return "I apologize, but I encountered an error processing your request. Please try again."
                
                # Get all messages in the thread
                messages = self.project.agents.messages.list(
                    thread_id=thread_id, 
                    order=ListSortOrder.ASCENDING
                )
                
                # Get the latest assistant message
                for message in reversed(messages):
                    if message.role == "assistant" and message.text_messages:
                        return message.text_messages[-1].text.value
                
                return "I apologize, but I didn't receive a proper response. Please try again."
                
            except Exception as e:
                print(f"Error in send_message: {str(e)}")
                return f"I encountered an error: {str(e)}. Please try again."
        
        # Fallback: Use local knowledge base with simple responses
        else:
            try:
                # Get context from unified database
                context = self.search_unified_database(user_message)
                
                # Generate simple response based on context
                if "fees" in user_message.lower() or "cost" in user_message.lower():
                    response = f"""**Program Fee Information**

Based on our program details:

{context}

For the most current fee structure and payment options, please contact our admissions team:
- **Email:** admissions@race.reva.edu.in
- **Phone:** +91-80-4696-6966

Would you like to know more about any specific program?"""

                elif "admission" in user_message.lower() or "requirement" in user_message.lower():
                    response = f"""**Admission Requirements**

Here's what I found about our admission requirements:

{context}

For detailed admission guidance and application support, please reach out to our admissions team:
- **Email:** admissions@race.reva.edu.in
- **Phone:** +91-80-4696-6966

Is there a specific program you're interested in?"""

                else:
                    response = f"""Thank you for your question! Here's what I found:

{context}

For more detailed information and personalized guidance, I recommend speaking with our admissions team:
- **Email:** admissions@race.reva.edu.in
- **Phone:** +91-80-4696-6966

How else can I help you today?"""

                return response
                
            except Exception as e:
                return """I apologize, but I'm experiencing technical difficulties. 

For immediate assistance, please contact our RACE admissions team:
- **Email:** admissions@race.reva.edu.in
- **Phone:** +91-80-4696-6966
- **Website:** https://race.reva.edu.in

They'll be happy to help you with all your questions!"""

# Initialize Azure AI
race_azure = RaceAzureAI()

@app.route('/')
def index():
    """Main website page with chat button"""
    return render_template('azure_index.html')

@app.route('/chat')
def chat_page():
    """Full-page chat interface"""
    return render_template('azure_chat.html')

@app.route('/api/azure_chat', methods=['POST'])
def azure_chat():
    """Handle Azure AI chat messages"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'Message is required'}), 400
        
        # Get or create thread ID
        thread_id = session.get('azure_thread_id')
        if not thread_id:
            thread_id = race_azure.create_thread()
            if not thread_id:
                return jsonify({'error': 'Failed to create conversation thread'}), 500
            session['azure_thread_id'] = thread_id
        
        # Initialize chat history
        if 'azure_chat_history' not in session:
            session['azure_chat_history'] = []
        
        # Add user message to history
        session['azure_chat_history'].append({
            'role': 'user',
            'content': user_message,
            'timestamp': str(uuid.uuid4())
        })
        
        # Get AI response
        ai_response = race_azure.send_message(thread_id, user_message)
        
        # Add AI response to history
        session['azure_chat_history'].append({
            'role': 'assistant',
            'content': ai_response,
            'timestamp': str(uuid.uuid4())
        })
        
        # Keep only last 50 messages
        session['azure_chat_history'] = session['azure_chat_history'][-50:]
        
        return jsonify({
            'response': ai_response,
            'thread_id': thread_id
        })
        
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/clear_azure_chat', methods=['POST'])
def clear_azure_chat():
    """Clear Azure chat history and create new thread"""
    session['azure_chat_history'] = []
    # Create new thread for fresh conversation
    thread_id = race_azure.create_thread()
    if thread_id:
        session['azure_thread_id'] = thread_id
    return jsonify({'success': True})

@app.route('/api/azure_status')
def azure_status():
    """Get Azure AI status"""
    return jsonify({
        'azure_available': race_azure.azure_available,
        'database_available': race_azure.database_available
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)