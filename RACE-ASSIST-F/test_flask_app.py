#!/usr/bin/env python3
"""
Test script for Flask app with unified FAISS database
"""

import requests
import json
import time

def test_flask_app():
    """Test the Flask app endpoints"""
    base_url = "http://localhost:5000"
    
    print("Testing Flask App with Unified FAISS Database")
    print("=" * 50)
    
    try:
        # Test status endpoint
        print("1. Testing /api/status endpoint...")
        response = requests.get(f"{base_url}/api/status")
        if response.status_code == 200:
            status = response.json()
            print(f"   ✅ OpenAI Available: {status.get('openai_available')}")
            print(f"   ✅ Total Files: {status.get('total_files')}")
            print(f"   ✅ Total Chunks: {status.get('total_chunks')}")
        else:
            print(f"   ❌ Status check failed: {response.status_code}")
            return False
        
        # Test database info endpoint
        print("\\n2. Testing /api/database_info endpoint...")
        response = requests.get(f"{base_url}/api/database_info")
        if response.status_code == 200:
            db_info = response.json()
            print(f"   ✅ Files available: {db_info.get('total_files', 0)}")
            available_files = db_info.get('available_files', [])
            print(f"   ✅ Sample files: {available_files[:3] if available_files else 'None'}")
        else:
            print(f"   ❌ Database info check failed: {response.status_code}")
        
        # Test chat endpoint
        print("\\n3. Testing /api/chat endpoint...")
        test_messages = [
            "What programs do you offer?",
            "Tell me about the Business Analytics program",
            "What are the admission requirements?"
        ]
        
        session = requests.Session()  # Use session to maintain chat history
        
        for i, message in enumerate(test_messages, 1):
            print(f"   Testing message {i}: {message}")
            
            chat_data = {"message": message}
            response = session.post(f"{base_url}/api/chat", json=chat_data)
            
            if response.status_code == 200:
                chat_response = response.json()
                print(f"   ✅ Response received (avg_score: {chat_response.get('avg_score', 0):.3f})")
                print(f"   ✅ Response preview: {chat_response.get('response', '')[:100]}...")
            else:
                print(f"   ❌ Chat failed: {response.status_code}")
                print(f"   Error: {response.text}")
            
            time.sleep(1)  # Small delay between requests
        
        print("\\n✅ All Flask app tests completed!")
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to Flask app. Make sure it's running on http://localhost:5000")
        return False
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        return False

if __name__ == "__main__":
    print("Make sure to start the Flask app first:")
    print("python flask_app.py")
    print("\\nPress Enter when the Flask app is running...")
    input()
    
    success = test_flask_app()
    if success:
        print("\\n🎉 Flask app is working correctly with unified search!")
    else:
        print("\\n❌ Some tests failed. Check the Flask app logs for details.")