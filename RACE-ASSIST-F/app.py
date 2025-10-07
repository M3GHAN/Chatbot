import streamlit as st
import json
import os
import time
from openai import OpenAI
from dotenv import load_dotenv
from process_brochures import BrochureProcessor
from typing import List, Tuple, Dict

# Load environment variables
load_dotenv()

class RaceAIApp:
   def __init__(self):
       self.processor = BrochureProcessor()
       self.setup_azure_openai()
       self.load_categories()
   
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
               st.warning(f"Azure OpenAI setup failed: {str(e)}")
               self.openai_available = False
       else:
           self.openai_available = False
   
   def load_categories(self):
       """Load available files from unified database"""
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
           st.error(f"Error loading unified database info: {str(e)}")
           self.mapping = {}
           self.available_files = []
           self.unified_db_info = {}
   
   def display_search_results(self, results: List[Tuple[str, float, str]], query: str):
       """Display search results in a formatted way"""
       if not results:
           st.warning("No relevant information found for your query.")
           return
       
       st.success(f"Found {len(results)} relevant sections:")
       
       for i, (chunk, score, source_file) in enumerate(results, 1):
           with st.expander(f"Result {i} (Relevance: {score:.3f}) - Source: {source_file}", expanded=(i == 1)):
               st.write(chunk)
   
   def get_context_from_documents(self, query: str, top_k: int = 3) -> tuple:
       """Get relevant context from unified FAISS database with similarity scores"""
       try:
           results = self.processor.search_similar(
               query=query,
               top_k=top_k
           )
           
           if not results:
               return "No relevant information found in the knowledge base.", []
           
           context_sections = []
           similarity_scores = []
           
           for i, (chunk, score, source_file) in enumerate(results):
               context_sections.append(f"Section {i+1} (Similarity: {score:.3f}, Source: {source_file}):\n{chunk}")
               similarity_scores.append(score)
           
           context_text = "\n\n".join(context_sections)
           return context_text, similarity_scores
           
       except Exception as e:
           return f"Error retrieving context: {str(e)}", []
   
   
   def run_chat_interface(self):
       """Clean chat interface using unified database"""
       
       # Check if unified database is available
       if not self.unified_db_info:
           st.warning("No unified database available! Please run the unified processor to create the database.")
           return
       
       # Show unified database info
       total_chunks = self.unified_db_info.get('total_chunks', 0)
       total_files = self.unified_db_info.get('total_files', 0)
       st.success(f"✅ Connected to RACE-AI Unified Database - {total_chunks} topics from {total_files} files available")
       
       # Show available files expander
       with st.expander(f"📁 Available Files ({len(self.available_files)}):", expanded=False):
           if self.available_files:
               for file in self.available_files:
                   st.write(f"• {file}")
           else:
               st.write("No files loaded")
       
       # API status
       if not self.openai_available:
           st.error("❌ Azure OpenAI not available. Please set AZURE_OPENAI_API_KEY in .env file")
           return
       
       # Initialize chat messages
       if "messages" not in st.session_state:
           st.session_state.messages = []
       
       # Display chat messages
       for message in st.session_state.messages:
           with st.chat_message(message["role"]):
               st.markdown(message["content"])
       
       # Chat input
       if prompt := st.chat_input("Ask about RACE programs..."):
           # Add user message to chat history
           st.session_state.messages.append({"role": "user", "content": prompt})
           
           # Display user message
           with st.chat_message("user"):
               st.markdown(prompt)
           
           # Generate AI response
           with st.chat_message("assistant"):
               # Get context from documents with similarity scores
               document_context, similarity_scores = self.get_context_from_documents(prompt)
               
               # Build conversation context
               recent_history = ""
               for msg in st.session_state.messages[-4:]:
                   role = "User" if msg["role"] == "user" else "Assistant"
                   recent_history += f"{role}: {msg['content']}\n"
               
               # Create comprehensive RACE-AI prompt
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

USER QUESTION: {prompt}

Please respond as RACE-AI following all the guidelines above."""

               try:
                   completion = self.client.chat.completions.create(
                       model=self.deployment_name,
                       messages=[
                           {"role": "system", "content": system_prompt},
                           {"role": "user", "content": user_prompt}
                       ],
                       temperature=0.7,
                       max_tokens=1000,
                       stream=True
                   )
                   
                   # Stream the response
                   response = st.write_stream(self.stream_generator(completion))
                   
                   # Show similarity scores after the response
                   if similarity_scores:
                       st.caption(f"📊 **Relevance Scores:** {', '.join([f'{score:.3f}' for score in similarity_scores])} | **Avg:** {sum(similarity_scores)/len(similarity_scores):.3f}")
                   
               except Exception as e:
                   response = f"I apologize, but I'm experiencing technical difficulties. Error: {str(e)}"
                   st.markdown(response)
                   similarity_scores = []
           
           # Add assistant response to chat history with metadata
           response_with_scores = response
           if similarity_scores:
               avg_score = sum(similarity_scores) / len(similarity_scores)
               response_with_scores += f"\n\n*[Relevance: {avg_score:.3f}]*"
           
           st.session_state.messages.append({"role": "assistant", "content": response_with_scores})
   
   def stream_generator(self, stream):
       """Generator for streaming OpenAI response"""
       for chunk in stream:
           if chunk.choices[0].delta.content is not None:
               yield chunk.choices[0].delta.content
   
   def run_search_mode(self):
       """Search mode interface using unified database"""
       st.header("🔍 Document Search")
       st.markdown("*Search through all RACE program documents*")
       
       # Show unified database info
       if self.unified_db_info:
           total_chunks = self.unified_db_info.get('total_chunks', 0)
           total_files = self.unified_db_info.get('total_files', 0)
           st.info(f"📄 {total_chunks} text sections from {total_files} files available")
           
           # Query input
           query = st.text_input(
               "Enter your research question:",
               placeholder="e.g., What are the admission requirements for AI program?",
               key="search_query"
           )
           
           # Search parameters
           col1, col2 = st.columns([3, 1])
           with col2:
               top_k = st.slider("Number of results:", 1, 10, 5)
           
           # Search button
           if st.button("🔍 Search", type="primary") and query:
               with st.spinner("Searching..."):
                   try:
                       results = self.processor.search_similar(
                           query=query,
                           top_k=top_k
                       )
                       self.display_search_results(results, query)
                   except Exception as e:
                       st.error(f"Search error: {str(e)}")
   
   def run(self):
       """Main application interface"""
       # Header
       st.title("🤖 RACE-AI Customer Support")
       st.write(
           "Select a program below and ask questions about it – RACE-AI will answer based on the your interested program"
       )
       
       # Run the chat interface
       self.run_chat_interface()

def main():
   # Page configuration
   st.set_page_config(
       page_title="RACE-AI Platform",
       page_icon="🤖",
       layout="wide",
       initial_sidebar_state="expanded"
   )
   
   # Clean chat styling
   st.markdown("""
   <style>
   /* Chat message styling */
   .stChatMessage {
       padding: 0.8rem 1.2rem;
       border-radius: 18px;
       margin-bottom: 0.5rem;
       max-width: 75%;
   }
   
   /* User messages - blue, right side */
   .stChatMessage[data-testid="user"] {
       background-color: #007bff;
       color: white;
       margin-left: auto;
       margin-right: 0;
   }
   
   /* AI messages - gray, left side */
   .stChatMessage[data-testid="assistant"] {
       background-color: #f1f3f4;
       color: #333;
       margin-left: 0;
       margin-right: auto;
   }
   
   /* Hide streamlit branding */
   #MainMenu {visibility: hidden;}
   footer {visibility: hidden;}
   
   /* Clean layout */
   .block-container {
       padding-top: 2rem;
   }
   </style>
   """, unsafe_allow_html=True)
   
   app = RaceAIApp()
   app.run()

if __name__ == "__main__":
   main()