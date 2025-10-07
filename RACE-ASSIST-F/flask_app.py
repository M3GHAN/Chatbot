from flask import Flask, render_template, request, jsonify, session
import json
import os
import uuid
from openai import OpenAI
from dotenv import load_dotenv
from process_brochures import BrochureProcessor

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'race-ai-secret-key-2024')

class RaceAIFlask:
    def __init__(self):
        self.processor = BrochureProcessor()
        self.setup_azure_openai()
        self.load_unified_database()
        
    def setup_azure_openai(self):
        """Initialize Azure OpenAI API"""
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "https://varshitharesource.openai.azure.com/openai/v1/")
        deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
        
        if api_key and api_key != "your_api_key_here":
            try:
                self.client = OpenAI(
                    base_url=endpoint,
                    api_key=api_key
                )
                self.deployment_name = deployment_name
                self.openai_available = True
            except Exception as e:
                print(f"Azure OpenAI setup failed: {str(e)}")
                self.openai_available = False
        else:
            self.openai_available = False
            
    def load_unified_database(self):
        """Load unified database information"""
        try:
            if os.path.exists(self.processor.mapping_file):
                with open(self.processor.mapping_file, 'r') as f:
                    self.mapping = json.load(f)
                self.available_files = self.processor.get_available_files()
                self.unified_db_info = self.mapping.get('unified_database', {})
            else:
                self.mapping = {}
                self.available_files = []
                self.unified_db_info = {}
        except Exception as e:
            print(f"Error loading unified database info: {str(e)}")
            self.mapping = {}
            self.available_files = []
            self.unified_db_info = {}
    

    
    def get_context_from_documents(self, query: str, conversation_history: list, top_k: int = 6):
        """Get relevant context from unified FAISS database"""
        try:
            # Use unified search
            results = self.processor.search_similar(query=query, top_k=top_k)
            
            if not results:
                print("No results found in unified database")
                return "INSUFFICIENT_CONTEXT", []
            
            context_sections = []
            similarity_scores = []
            
            print(f"Query: {query}")
            print(f"Found {len(results)} results in unified database")
            
            for i, (chunk, score, source_file) in enumerate(results):
                context_sections.append(f"From {source_file}:\n{chunk}")
                similarity_scores.append(score)
            
            context_text = "\n\n".join(context_sections)
            print(f"Final context length: {len(context_text)}")
            return context_text, similarity_scores
            
        except Exception as e:
            print(f"Error in get_context_from_documents: {str(e)}")
            return "INSUFFICIENT_CONTEXT", []
    
    def generate_response(self, user_input: str, conversation_history: list):
        """Generate AI response with enhanced context and conversation flow using Azure OpenAI"""
        if not self.openai_available:
            return "AI service is currently unavailable. Please try again later.", []
        
        try:
            # Get context from documents with conversation history
            document_context, similarity_scores = self.get_context_from_documents(
                user_input, conversation_history
            )
            
            # Handle insufficient context
            if document_context == "INSUFFICIENT_CONTEXT":
                return "INSUFFICIENT_CONTEXT", []
            
            # Build comprehensive conversation context
            recent_history = ""
            for msg in conversation_history[-8:]:
                role = "User" if msg["role"] == "user" else "Assistant"
                recent_history += f"{role}: {msg['content']}\n"
            
            # Comprehensive RACE-AI system prompt
            system_prompt = """You are RACE-AI, the official admission counselor and student support specialist for REVA University's RACE (REVA Academy for Corporate Excellence) programs. You are a knowledgeable, professional, and empathetic counselor with comprehensive expertise in all RACE degree programs and professional certifications.

YOUR IDENTITY & ROLE
- Name: RACE-AI
- Position: Senior Admission Counselor, REVA Academy for Corporate Excellence
- Institution: REVA University, Bangalore
- Expertise: Complete knowledge of RACE programs, admissions, career guidance, and student success
- Personality: Professional, encouraging, supportive, detail-oriented, and results-driven

COMPLETE PROGRAM PORTFOLIO

🎓 DEGREE PROGRAMS (Masters Level)

M.Sc. in Business Analytics
- Duration: 2 Years (4 Semesters, 84 Credits)
- Recognition: UGC Approved
- Fee: INR 4.8 Lakhs
- Start Date: September 2025
- Partnership: AWS Academy & Microsoft Azure
- Placement Stats: 39 LPA Average Salary, 354% Average Hike, 30 LPA Median Salary
- Key Features: 
  - 14 Modules, 10+ Mini Projects, 2 Capstone Projects
  - 2 Global Certifications (Azure AI-102, Azure DP-100)
  - 1 Research Paper Publication
  - Industry mentorship and 100+ hiring partners
- Curriculum: Full-stack analytics, ML/AI, Deep Learning, NLP, Domain Analytics (Marketing, Supply Chain, Finance)

M.Tech. in Artificial Intelligence
- Recognition: UGC Approved
- Focus: Advanced AI/ML applications for enterprise solutions
- Industry Integration: Hands-on projects with real-world applications

PG Diploma/M.Sc. in Artificial Intelligence
- Flexible Options: Both diploma and degree pathways
- Specialization: AI engineering and implementation

M.Tech. in Cybersecurity
- Focus: Advanced cybersecurity for enterprise protection
- Industry Relevance: Current threat landscape and defense strategies

PG Diploma/M.Sc. in Cybersecurity
- Dual Pathway: Professional diploma or academic degree options
- Practical Focus: Hands-on security implementation

PG Diploma/M.Sc. in Cloud Architecture and Security
- Cloud Focus: AWS and Azure specialization
- Security Integration: Cloud security best practices

📜 PROFESSIONAL CERTIFICATIONS

Certified AI Engineer
- Focus: Industry-standard AI implementation skills
- Recognition: Global certification for AI professionals

Certified Ethical Hacker (CEH)
- Specialization: Ethical hacking and penetration testing
- Industry Recognition: Globally recognized cybersecurity certification

Certified Penetration Testing Professional (CPENT)
- Advanced Level: Professional penetration testing expertise
- Hands-on Focus: Real-world testing scenarios

Advanced Diploma in Cybersecurity and Privacy Management
- Comprehensive: Full spectrum cybersecurity management
- Privacy Focus: Data protection and compliance

Certified DevOps Specialist
- Technologies: Terraform, Kubernetes, Jenkins, DevSecOps, AIOps
- Industry Demand: High-demand DevOps and cloud automation skills

KEY PROGRAM FEATURES

Academic Excellence
- UGC approved programs with international standards
- Outcome-based education system (OBE)
- Continuous evaluation and project-based learning
- Research publication opportunities

Industry Integration
- 50+ industry mentors from top companies (Dell, Intel, Oracle, Goldman Sachs, PwC)
- 100+ hiring partners including Fortune 500 companies
- Real-time industry projects and case studies
- Partnership with AWS Academy, Microsoft Azure, CloudxLabs

Career Success
- Average salary hike: 50-200%
- Placement support with resume building and mock interviews
- One-on-one career guidance sessions
- Alumni network of 1000+ professionals

Learning Methodology
- Weekend classes for working professionals
- 24/7 LMS access with recorded sessions
- Hands-on labs and simulations
- Industry-grade projects and capstone work

ADMISSION PROCESS & SUPPORT

Application Process
1. Online Application: Fill application form
2. Evaluation: Documentation and screening call with Director's office
3. Admission: Receive offer letter and secure seat with admission fee

Financial Support
- Merit Scholarship: Up to INR 20,000 discount for 60%+ scores
- Additional Discounts: Early bird, group, and referral discounts available
- Educational Loans: Available from banks/NBFCs at 9-14% interest
- Tax Benefits: Income tax benefits on educational loans

Contact & Support
- Phone: +91 89040 58866
- Immediate WhatsApp Support: https://web.whatsapp.com/send/?phone=919945395881&text=Hello%21+I+am+interested+to+know+more+about+RACE+Programs&type=phone_number&app_absent=0
- Campus: REVA University, Bangalore

YOUR RESPONSE GUIDELINES

Communication Style
- Professional yet warm and encouraging
- Use specific data points (salaries, percentages, durations)
- Create structured responses with bullet points and tables
- Be enthusiastic about REVA University and RACE programs
- Show genuine interest in student career goals

Response Structure
1. Warm Greeting (acknowledge their interest)
2. Direct Answer to their specific question
3. Relevant Additional Information 
4. Program Recommendations based on their background
5. Next Steps and contact options
6. Encouragement and offer for further assistance

Key Messaging Points
- RACE programs are designed for working professionals
- Industry-driven curriculum with real-world applications
- Exceptional placement support and salary growth
- UGC recognition and global partnerships
- Flexible learning with weekend schedules
- Strong alumni network and industry connections

WhatsApp Integration
Always offer immediate support through WhatsApp:
"For immediate assistance and to connect with our admission team, you can reach us on WhatsApp: https://web.whatsapp.com/send/?phone=919945395881&text=Hello%21+I+am+interested+to+know+more+about+RACE+Programs&type=phone_number&app_absent=0"

IMPORTANT GUIDELINES
- Always provide specific numbers (fees, durations, salary data)
- Mention partnerships with AWS, Microsoft, and industry leaders
- Highlight success stories and career transformations
- Offer WhatsApp support for immediate assistance
- Focus on ROI and career advancement opportunities
- Never oversell - be honest about program requirements and commitments
- Always end with encouragement and next steps

Remember: Your goal is to help prospective students find the perfect RACE program match while showcasing the exceptional value proposition of RACE REVA University's industry-integrated education approach."""
            
            user_prompt = f"""KNOWLEDGE BASE CONTEXT:
{document_context}

CONVERSATION HISTORY:
{recent_history}

STUDENT QUESTION: {user_input}

Please respond as RACE-AI following all the guidelines above."""
            
            # Generate response using Azure OpenAI
            completion = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            full_response = completion.choices[0].message.content
            return full_response, similarity_scores
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "INSUFFICIENT_CONTEXT", []

# Initialize RACE-AI
race_ai = RaceAIFlask()

@app.route('/')
def index():
    """Main website page"""
    total_files = len(race_ai.available_files)
    total_chunks = race_ai.unified_db_info.get('total_chunks', 0)
    return render_template('index.html', 
                         total_files=total_files,
                         total_chunks=total_chunks,
                         openai_available=race_ai.openai_available)

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'Message is required'}), 400
        
        # Initialize session chat history
        if 'chat_history' not in session:
            session['chat_history'] = []
        
        # Add user message to history
        session['chat_history'].append({
            'role': 'user',
            'content': user_message,
            'timestamp': str(uuid.uuid4())
        })
        
        # Generate AI response
        ai_response, similarity_scores = race_ai.generate_response(
            user_message,
            session['chat_history']
        )
        
        # Handle insufficient context with professional fallback
        if ai_response == "INSUFFICIENT_CONTEXT" or "INSUFFICIENT_CONTEXT" in ai_response:
            ai_response = f"""Hello! I'm RACE-AI, your dedicated admission counselor for REVA University's RACE programs. 

While I have comprehensive knowledge about all our programs, I want to ensure you get the most accurate and up-to-date information for your specific inquiry.

I'd recommend connecting directly with our admissions team for detailed, personalized guidance:

**🔗 Connect with Our Admissions Team:**
- **Phone:** +91 89040 58866
- **WhatsApp:** https://web.whatsapp.com/send/?phone=919945395881&text=Hello%21+I+am+interested+to+know+more+about+RACE+Programs&type=phone_number&app_absent=0
- **Campus:** REVA University, Bangalore

**📚 Our Program Portfolio includes:**
- M.Sc. Business Analytics (INR 4.8L, 39 LPA avg salary)
- M.Tech./M.Sc. Artificial Intelligence 
- M.Tech./M.Sc. Cybersecurity
- M.Sc. Cloud Architecture & Security
- Professional Certifications (CEH, CPENT, DevOps)

Our counselors can discuss program details, career prospects, admission requirements, and help you find the perfect program for your goals!

Feel free to ask me anything about RACE programs - I'm here to help! 🚀"""
        
        # Add AI response to history
        session['chat_history'].append({
            'role': 'assistant',
            'content': ai_response,
            'similarity_scores': similarity_scores,
            'avg_score': sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0,
            'timestamp': str(uuid.uuid4())
        })
        
        # Keep only last 20 messages
        session['chat_history'] = session['chat_history'][-20:]
        
        return jsonify({
            'response': ai_response,
            'similarity_scores': similarity_scores,
            'avg_score': sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0,
            'unified_search': True
        })
        
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/clear_chat', methods=['POST'])
def clear_chat():
    """Clear chat history"""
    session['chat_history'] = []
    return jsonify({'success': True})

@app.route('/api/database_info')
def get_database_info():
    """Get unified database information"""
    return jsonify({
        'unified_database': race_ai.unified_db_info,
        'available_files': race_ai.available_files,
        'total_files': len(race_ai.available_files)
    })

@app.route('/api/status')
def get_status():
    """Get system status"""
    return jsonify({
        'openai_available': race_ai.openai_available,
        'unified_database_available': bool(race_ai.unified_db_info),
        'total_files': len(race_ai.available_files),
        'total_chunks': race_ai.unified_db_info.get('total_chunks', 0)
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)