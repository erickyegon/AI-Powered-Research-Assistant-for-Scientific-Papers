# 🔌 AI Research Assistant - API Documentation

## 📋 **Overview**

The AI Research Assistant provides a comprehensive REST API for processing research queries, document summarization, and system monitoring. Built with FastAPI, it offers high-performance async processing with automatic OpenAPI documentation.

**Base URL**: `https://ai-research-assistant-backend.onrender.com`

## 🚀 **Quick Start**

### **Interactive Documentation**
- **Swagger UI**: https://ai-research-assistant-backend.onrender.com/docs
- **ReDoc**: https://ai-research-assistant-backend.onrender.com/redoc
- **OpenAPI Spec**: https://ai-research-assistant-backend.onrender.com/openapi.json

### **Authentication**
Currently, the API is open for public use. Future versions will include API key authentication.

## 📡 **Endpoints**

### **1. Query Processing**

#### `POST /api/query`
Process research queries with AI-powered analysis.

**Request Body:**
```json
{
  "query": "What are the latest developments in machine learning?",
  "include_enhancement": true,
  "max_results": 5
}
```

**Response:**
```json
{
  "response": "Machine learning has seen significant developments...",
  "enhanced_query": "Please provide a comprehensive research analysis of: What are the latest developments in machine learning?",
  "metadata": {
    "original_query": "What are the latest developments in machine learning?",
    "processing_method": "direct_api",
    "model": "gpt-4.1-nano",
    "enhanced": true,
    "environment": "production"
  }
}
```

**Parameters:**
- `query` (string, required): The research question
- `include_enhancement` (boolean, optional): Whether to enhance the query
- `max_results` (integer, optional): Maximum number of results (default: 5)

---

### **2. Document Summarization**

#### `POST /api/summarize`
Generate summaries of research documents.

**Request Body:**
```json
{
  "text": "Your research paper content here...",
  "summary_type": "detailed"
}
```

**Response:**
```json
{
  "summary": "This research paper discusses...",
  "summary_type": "detailed",
  "original_length": 5000,
  "summary_length": 500
}
```

**Summary Types:**
- `brief`: 2-3 sentences
- `detailed`: Comprehensive summary
- `bullet_points`: Structured bullet points

---

### **3. System Health**

#### `GET /health`
Check system health and component status.

**Response:**
```json
{
  "status": "healthy",
  "version": "production-1.0.0",
  "startup_complete": true,
  "components": {
    "euri_available": true,
    "euri_configured": true,
    "langsmith_configured": true,
    "llm_initialized": true
  },
  "environment": "production"
}
```

---

### **4. Configuration**

#### `GET /api/config`
Get current system configuration (non-sensitive data only).

**Response:**
```json
{
  "euri": {
    "api_key_configured": true,
    "model": "gpt-4.1-nano",
    "temperature": "0.7",
    "max_tokens": "2000"
  },
  "app": {
    "environment": "production",
    "debug": "false",
    "version": "1.0.0"
  }
}
```

---

### **5. API Test**

#### `POST /api/test`
Test API connectivity and LLM integration.

**Response:**
```json
{
  "status": "success",
  "message": "Euri API connection successful",
  "model": "gpt-4.1-nano",
  "environment": "production"
}
```

## 🔧 **Error Handling**

### **Error Response Format**
```json
{
  "detail": "Error description",
  "type": "error_type"
}
```

### **HTTP Status Codes**
| Code | Description | Common Causes |
|------|-------------|---------------|
| 200 | Success | Request processed successfully |
| 400 | Bad Request | Invalid request format or parameters |
| 422 | Validation Error | Request body validation failed |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server-side processing error |
| 503 | Service Unavailable | External service (LLM) unavailable |

### **Error Examples**

**400 Bad Request:**
```json
{
  "detail": "Query cannot be empty",
  "type": "validation_error"
}
```

**429 Rate Limited:**
```json
{
  "detail": "Rate limit exceeded. Please try again later.",
  "type": "rate_limit_error"
}
```

**500 Internal Server Error:**
```json
{
  "detail": "Internal server error",
  "type": "server_error"
}
```

## 📊 **Rate Limiting**

### **Current Limits**
- **Requests per minute**: 60 per IP address
- **Concurrent requests**: 10 per IP address
- **Request timeout**: 60 seconds

### **Rate Limit Headers**
```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1640995200
```

## 🔍 **Request/Response Examples**

### **Python Example**
```python
import requests
import json

# Configuration
BASE_URL = "https://ai-research-assistant-backend.onrender.com"
HEADERS = {"Content-Type": "application/json"}

def query_research_assistant(question, enhance=True):
    """Query the research assistant API."""
    url = f"{BASE_URL}/api/query"
    payload = {
        "query": question,
        "include_enhancement": enhance,
        "max_results": 5
    }
    
    try:
        response = requests.post(url, headers=HEADERS, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return None

# Usage
result = query_research_assistant("Explain quantum computing")
if result:
    print(f"Response: {result['response']}")
```

### **JavaScript Example**
```javascript
const API_BASE = 'https://ai-research-assistant-backend.onrender.com';

async function queryAPI(endpoint, data) {
    try {
        const response = await fetch(`${API_BASE}${endpoint}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('API request failed:', error);
        throw error;
    }
}

// Usage
queryAPI('/api/query', {
    query: 'What is artificial intelligence?',
    include_enhancement: true
}).then(result => {
    console.log('Response:', result.response);
}).catch(error => {
    console.error('Error:', error);
});
```

### **cURL Examples**
```bash
# Query processing
curl -X POST "https://ai-research-assistant-backend.onrender.com/api/query" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "Explain neural networks",
       "include_enhancement": true,
       "max_results": 3
     }'

# Document summarization
curl -X POST "https://ai-research-assistant-backend.onrender.com/api/summarize" \
     -H "Content-Type: application/json" \
     -d '{
       "text": "Neural networks are computing systems inspired by biological neural networks...",
       "summary_type": "brief"
     }'

# Health check
curl "https://ai-research-assistant-backend.onrender.com/health"

# Configuration
curl "https://ai-research-assistant-backend.onrender.com/api/config"
```

## 🛡️ **Security & Best Practices**

### **Security Considerations**
- All requests use HTTPS encryption
- No sensitive data is logged or stored
- Rate limiting prevents abuse
- Input validation prevents injection attacks

### **Best Practices**
1. **Implement retry logic** with exponential backoff
2. **Handle timeouts gracefully** (30-60 seconds recommended)
3. **Cache responses** when appropriate to reduce API calls
4. **Monitor rate limits** using response headers
5. **Validate responses** before processing
6. **Use appropriate error handling** for different status codes

### **Recommended Retry Logic**
```python
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def create_session_with_retries():
    session = requests.Session()
    
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session

# Usage
session = create_session_with_retries()
response = session.post(url, json=data, timeout=30)
```

## 📈 **Performance Optimization**

### **Response Times**
- **Simple queries**: < 2 seconds
- **Complex queries**: 2-10 seconds
- **Document summarization**: 3-15 seconds (depending on length)

### **Optimization Tips**
1. **Use brief summaries** for faster processing
2. **Limit query complexity** for better performance
3. **Implement client-side caching** for repeated queries
4. **Use appropriate timeouts** based on query type

## 🔗 **Related Resources**

- **Live Demo**: https://ai-research-assistant-frontend.onrender.com
- **GitHub Repository**: https://github.com/erickyegon/AI-Powered-Research-Assistant-for-Scientific-Papers
- **Interactive API Docs**: https://ai-research-assistant-backend.onrender.com/docs
- **System Health**: https://ai-research-assistant-backend.onrender.com/health

## 📞 **Support**

For API support and questions:
- **GitHub Issues**: [Create an issue](https://github.com/erickyegon/AI-Powered-Research-Assistant-for-Scientific-Papers/issues)
- **Email**: keyegon@gmail.com

---

**Last Updated**: January 2025  
**API Version**: 1.0.0  
**Documentation Version**: 1.0.0
