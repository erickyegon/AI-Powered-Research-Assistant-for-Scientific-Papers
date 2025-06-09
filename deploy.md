# 🚀 Deployment Guide for AI Research Assistant

## 📋 Prerequisites

1. **GitHub Repository**: https://github.com/erickyegon/AI-Powered-Research-Assistant-for-Scientific-Papers.git
2. **Render Account**: Sign up at https://render.com
3. **API Keys**:
   - Euri API Key: `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJlZDljYzlkYy0xZmQ2LTRiMGMtODcyZS1lYmJlMmRjNjZiNzQiLCJlbWFpbCI6ImtleWVnb25AZ21haWwuY29tIiwiaWF0IjoxNzQ3MDM0Nzc2LCJleHAiOjE3Nzg1NzA3NzZ9.6m2jzZ_A7eGmoBYjzP7lLazn1luxIFUIYOsbS6ttKS0`
   - LangSmith API Key: `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`

## 🔧 Step 1: Push to GitHub

```bash
# Initialize git repository
git init

# Add remote repository
git remote add origin https://github.com/erickyegon/AI-Powered-Research-Assistant-for-Scientific-Papers.git

# Add all files
git add .

# Commit changes
git commit -m "Complete AI Research Assistant with LangChain stack and Render deployment"

# Push to GitHub
git push -u origin main
```

## 🚀 Step 2: Deploy Backend on Render

### Option A: Using render.yaml (Recommended)

1. **Connect Repository**:
   - Go to https://render.com/dashboard
   - Click "New" → "Blueprint"
   - Connect your GitHub repository
   - Select the repository: `AI-Powered-Research-Assistant-for-Scientific-Papers`

2. **Configure Environment Variables**:
   - `EURI_API_KEY`: `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJlZDljYzlkYy0xZmQ2LTRiMGMtODcyZS1lYmJlMmRjNjZiNzQiLCJlbWFpbCI6ImtleWVnb25AZ21haWwuY29tIiwiaWF0IjoxNzQ3MDM0Nzc2LCJleHAiOjE3Nzg1NzA3NzZ9.6m2jzZ_A7eGmoBYjzP7lLazn1luxIFUIYOsbS6ttKS0`
   - `LANGSMITH_API_KEY`: `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`

### Option B: Manual Deployment

1. **Create Backend Service**:
   - Go to https://render.com/dashboard
   - Click "New" → "Web Service"
   - Connect your GitHub repository
   - Configure:
     - **Name**: `ai-research-assistant-backend`
     - **Environment**: `Python 3`
     - **Build Command**: `pip install -r backend/requirements.txt`
     - **Start Command**: `python working_server.py`
     - **Plan**: `Starter` (Free)

2. **Environment Variables**:
   ```
   EURI_API_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJlZDljYzlkYy0xZmQ2LTRiMGMtODcyZS1lYmJlMmRjNjZiNzQiLCJlbWFpbCI6ImtleWVnb25AZ21haWwuY29tIiwiaWF0IjoxNzQ3MDM0Nzc2LCJleHAiOjE3Nzg1NzA3NzZ9.6m2jzZ_A7eGmoBYjzP7lLazn1luxIFUIYOsbS6ttKS0
   LANGSMITH_API_KEY=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
   LANGSMITH_PROJECT=ai-research-assistant-production
   ENVIRONMENT=production
   DEBUG=false
   LOG_LEVEL=INFO
   CORS_ORIGINS=*
   EURI_BASE_URL=https://api.euron.one/api/v1/euri/alpha/chat/completions
   EURI_MODEL=gpt-4.1-nano
   EURI_TEMPERATURE=0.7
   EURI_MAX_TOKENS=2000
   ```

## 🎨 Step 3: Deploy Frontend on Render

1. **Create Frontend Service**:
   - Click "New" → "Web Service"
   - Connect the same GitHub repository
   - Configure:
     - **Name**: `ai-research-assistant-frontend`
     - **Environment**: `Python 3`
     - **Build Command**: `pip install -r frontend/requirements.txt`
     - **Start Command**: `streamlit run frontend/app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true --server.enableCORS false --server.enableXsrfProtection false`
     - **Plan**: `Starter` (Free)

2. **Environment Variables**:
   ```
   BACKEND_URL=https://ai-research-assistant-backend.onrender.com
   ENVIRONMENT=production
   ```

## 🔗 Step 4: Update Frontend Configuration

After backend deployment, update the frontend environment variable:
- `BACKEND_URL`: Use the URL provided by Render for your backend service
- Example: `https://ai-research-assistant-backend.onrender.com`

## 🧪 Step 5: Test Deployment

### Backend Testing:
- Health Check: `https://your-backend-url.onrender.com/health`
- API Docs: `https://your-backend-url.onrender.com/docs`
- Test API: `https://your-backend-url.onrender.com/api/test`

### Frontend Testing:
- Access your Streamlit app: `https://your-frontend-url.onrender.com`
- Test query processing
- Verify API connectivity

## 📋 Expected URLs

After deployment, you'll have:
- **Backend API**: `https://ai-research-assistant-backend.onrender.com`
- **Frontend App**: `https://ai-research-assistant-frontend.onrender.com`
- **API Documentation**: `https://ai-research-assistant-backend.onrender.com/docs`

## 🔧 Troubleshooting

### Common Issues:

1. **Build Failures**:
   - Check `requirements.txt` files
   - Verify Python version compatibility
   - Check build logs in Render dashboard

2. **Environment Variables**:
   - Ensure all required variables are set
   - Check for typos in variable names
   - Verify API keys are correct

3. **CORS Issues**:
   - Update `CORS_ORIGINS` to include frontend URL
   - Check frontend `BACKEND_URL` configuration

4. **API Connection Issues**:
   - Verify backend is deployed and healthy
   - Check network connectivity
   - Ensure HTTPS is used for production

## 🎯 Production Features

Your deployed application will have:
- ✅ **Professional API** with comprehensive error handling
- ✅ **LangSmith Tracing** for monitoring and debugging
- ✅ **Health Monitoring** with detailed status checks
- ✅ **Secure Configuration** with environment-based settings
- ✅ **CORS Support** for cross-origin requests
- ✅ **Production Logging** for troubleshooting
- ✅ **Scalable Architecture** ready for high traffic

## 🚀 Next Steps

1. **Custom Domain**: Configure custom domains in Render
2. **SSL Certificates**: Automatic HTTPS with Render
3. **Monitoring**: Set up alerts and monitoring
4. **Scaling**: Upgrade to paid plans for better performance
5. **CI/CD**: Set up automatic deployments on git push

## 📞 Support

If you encounter issues:
1. Check Render build logs
2. Verify environment variables
3. Test API endpoints individually
4. Check GitHub repository structure
