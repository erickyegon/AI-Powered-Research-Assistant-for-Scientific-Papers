# 🔐 Authentication & Authorization Guide

This document provides comprehensive information about the Role-Based Access Control (RBAC) authentication system implemented in the AI-Powered Research Assistant.

## 🏗️ Authentication Architecture

### System Overview
- **JWT Token-based Authentication**: Secure token-based authentication with configurable expiration
- **API Key Authentication**: Alternative authentication method for programmatic access
- **Role-Based Access Control (RBAC)**: Hierarchical permission system with predefined roles
- **Session Management**: Persistent session tracking with activity monitoring
- **Middleware Integration**: Automatic request logging and authentication validation

### User Roles & Permissions

#### Role Hierarchy
1. **Super Admin** (`super_admin`) - Full system access
2. **Admin** (`admin`) - Administrative access with some restrictions
3. **Researcher** (`researcher`) - Research-focused permissions
4. **User** (`user`) - Basic user permissions
5. **Guest** (`guest`) - Limited read-only access

#### Permission Matrix

| Permission | Super Admin | Admin | Researcher | User | Guest |
|------------|-------------|-------|------------|------|-------|
| **Documents** |
| Read Documents | ✅ | ✅ | ✅ | ✅ | ✅ |
| Write Documents | ✅ | ✅ | ✅ | ❌ | ❌ |
| Delete Documents | ✅ | ✅ | ❌ | ❌ | ❌ |
| Admin Documents | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Queries** |
| Execute Queries | ✅ | ✅ | ✅ | ✅ | ✅ |
| View Query History | ✅ | ✅ | ✅ | ❌ | ❌ |
| Admin Queries | ✅ | ✅ | ❌ | ❌ | ❌ |
| **Users** |
| Read Users | ✅ | ✅ | ✅ | ✅ | ❌ |
| Write Users | ✅ | ✅ | ❌ | ❌ | ❌ |
| Delete Users | ✅ | ❌ | ❌ | ❌ | ❌ |
| Admin Users | ✅ | ❌ | ❌ | ❌ | ❌ |
| **System** |
| Read System | ✅ | ✅ | ❌ | ❌ | ❌ |
| Write System | ✅ | ❌ | ❌ | ❌ | ❌ |
| Admin System | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Analytics** |
| Read Analytics | ✅ | ✅ | ✅ | ❌ | ❌ |
| Admin Analytics | ✅ | ❌ | ❌ | ❌ | ❌ |

## 🚀 Getting Started

### Default Users
The system comes with pre-configured default users:

| Username | Password | Role | Description |
|----------|----------|------|-------------|
| `superadmin` | `admin123` | Super Admin | Full system access |
| `admin` | `admin123` | Admin | Administrative access |
| `researcher` | `admin123` | Researcher | Research-focused access |

⚠️ **Security Warning**: Change default passwords in production!

### First Login
1. **Start the application**:
   ```bash
   ./start.sh
   ```

2. **Login via API**:
   ```bash
   curl -X POST "http://localhost:8000/auth/login" \
        -H "Content-Type: application/json" \
        -d '{"username": "admin", "password": "admin123"}'
   ```

3. **Use the returned token**:
   ```bash
   curl -X GET "http://localhost:8000/auth/me" \
        -H "Authorization: Bearer YOUR_TOKEN_HERE"
   ```

## 📡 API Authentication

### JWT Token Authentication

#### Login
```http
POST /auth/login
Content-Type: application/json

{
  "username": "admin",
  "password": "admin123"
}
```

**Response**:
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "user": {
    "user_id": "admin_001",
    "username": "admin",
    "email": "admin@research-assistant.com",
    "role": "admin",
    "permissions": ["document:read", "document:write", ...]
  }
}
```

#### Using JWT Token
Include the token in the Authorization header:
```http
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...
```

### API Key Authentication

#### Create API Key
```http
POST /auth/api-key
Authorization: Bearer YOUR_JWT_TOKEN
Content-Type: application/json

{
  "name": "my-api-key",
  "expires_in_days": 30
}
```

#### Using API Key
Include the API key in the Authorization header:
```http
Authorization: Bearer your-api-key-here
```

## 👥 User Management

### Create New User
```http
POST /auth/register
Authorization: Bearer YOUR_ADMIN_TOKEN
Content-Type: application/json

{
  "username": "newuser",
  "email": "newuser@example.com",
  "password": "securepassword",
  "role": "user"
}
```

### List Users
```http
GET /auth/users?role_filter=researcher
Authorization: Bearer YOUR_TOKEN
```

### Get Current User Info
```http
GET /auth/me
Authorization: Bearer YOUR_TOKEN
```

## 🔒 Protected Endpoints

### Document Operations
- `POST /documents/upload` - Requires `DOCUMENT_WRITE` permission
- `GET /documents` - Requires `DOCUMENT_READ` permission
- `DELETE /documents/{id}` - Requires `DOCUMENT_DELETE` permission

### Query Operations
- `POST /rag/query` - Requires `QUERY_EXECUTE` permission
- `GET /queries/history` - Requires `QUERY_HISTORY` permission

### Administrative Operations
- `POST /auth/register` - Requires `USER_WRITE` permission
- `GET /auth/users` - Requires `USER_READ` permission
- `GET /system/info` - Public endpoint (no authentication required)

## 🛡️ Security Features

### Password Security
- **Bcrypt Hashing**: Passwords are hashed using bcrypt with salt
- **Password Validation**: Configurable password complexity requirements
- **Secure Storage**: Passwords are never stored in plain text

### Token Security
- **JWT Tokens**: Signed with configurable secret key
- **Configurable Expiration**: Default 30 minutes, configurable
- **Automatic Refresh**: Tokens can be refreshed before expiration

### Session Management
- **Activity Tracking**: Last activity timestamp for each session
- **Session Cleanup**: Automatic cleanup of expired sessions
- **Concurrent Sessions**: Multiple sessions per user supported

### Rate Limiting
- **Request Throttling**: Configurable rate limits per IP
- **Abuse Prevention**: Automatic blocking of suspicious activity
- **Whitelist Support**: Bypass rate limits for trusted IPs

## 🔧 Configuration

### Environment Variables
```bash
# JWT Configuration
SECRET_KEY=your-secret-key-here
ACCESS_TOKEN_EXPIRE_MINUTES=30
ALGORITHM=HS256

# Rate Limiting
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=3600
```

### Custom Roles & Permissions
To add custom roles or permissions, modify the `ROLE_PERMISSIONS` mapping in `backend/utils/auth.py`:

```python
# Add new permission
class Permission(str, Enum):
    CUSTOM_PERMISSION = "custom:permission"

# Add new role
class UserRole(str, Enum):
    CUSTOM_ROLE = "custom_role"

# Update role permissions
ROLE_PERMISSIONS[UserRole.CUSTOM_ROLE] = {
    Permission.DOCUMENT_READ,
    Permission.CUSTOM_PERMISSION,
}
```

## 🚨 Security Best Practices

### Production Deployment
1. **Change Default Passwords**: Update all default user passwords
2. **Use Strong Secret Keys**: Generate cryptographically secure secret keys
3. **Enable HTTPS**: Always use HTTPS in production
4. **Configure CORS**: Restrict CORS origins to trusted domains
5. **Monitor Access**: Enable logging and monitoring
6. **Regular Updates**: Keep dependencies updated

### Password Policy
- Minimum 8 characters
- Include uppercase, lowercase, numbers, and special characters
- Avoid common passwords and dictionary words
- Regular password rotation for administrative accounts

### API Security
- Use API keys for programmatic access
- Implement proper rate limiting
- Validate all input parameters
- Log all authentication attempts
- Monitor for suspicious activity

## 🔍 Troubleshooting

### Common Issues

#### "Invalid authentication credentials"
- Check if token is expired
- Verify token format (should start with "Bearer ")
- Ensure user account is active

#### "Permission denied"
- Check user role and permissions
- Verify endpoint requires the correct permission
- Contact administrator for role upgrade

#### "Rate limit exceeded"
- Wait for rate limit window to reset
- Contact administrator for rate limit adjustment
- Use API key for higher limits

### Debug Mode
Enable debug logging to troubleshoot authentication issues:
```bash
export LOG_LEVEL=DEBUG
```

## 📞 Support

For authentication-related issues:
1. Check the application logs
2. Verify environment configuration
3. Test with default users
4. Contact system administrator

---

**Security Notice**: This authentication system is designed for production use but should be regularly audited and updated according to security best practices.
