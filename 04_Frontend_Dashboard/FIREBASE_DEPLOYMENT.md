# Firebase Deployment & Configuration Notes

## Firebase Console Setup

### Prerequisites
- Active Firebase account
- Project: `nutritwin-cd71b` (created and configured)

### Authentication Setup Checklist

- [x] Firebase project created
- [ ] **TO DO**: Enable Email/Password authentication in Firebase Console
  1. Go to https://console.firebase.google.com/
  2. Select project: nutritwin-cd71b
  3. Go to Authentication > Sign-in method
  4. Enable "Email/Password" provider

- [ ] **TO DO**: Configure Email Template (Optional but recommended)
  1. In Firebase Console, go to Authentication > Templates
  2. Customize email templates for reset password

### Firestore Database Setup Checklist

- [ ] **TO DO**: Create Firestore Database
  1. Go to Firestore Database
  2. Create database in production mode
  3. Choose region closest to users

- [ ] **TO DO**: Set Security Rules
  1. Go to Firestore > Rules tab
  2. Replace with the security rules below:

```javascript
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    // Users can read and write their own data
    match /users/{uid} {
      allow read, write: if request.auth.uid == uid;
    }
    
    // Optional: Admins can read all users
    // match /users/{document=**} {
    //   allow read: if request.auth.token.role == 'admin';
    // }
  }
}
```

### API Keys Configuration (Security)

1. Go to Firebase Console > Project Settings
2. Click "API Keys" tab
3. Restrict the API key:
   - **Restrict keys** > Application restrictions > Web apps
   - Add domain: yourdomain.com
   - Add domain: localhost:3000 (for development)
   - **API restrictions** > Restrict to Firestore APIs

### Deployment Platforms

#### Vercel Deployment
```bash
# Deploy to Vercel
npm install -g vercel
vercel
```

No environment variables needed since Firebase config is public.

#### Netlify Deployment
```bash
# Build
npm run build

# Deploy to Netlify
netlify deploy --prod --dir=.next
```

#### Docker Deployment
```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
```

### Production Checklist

#### Security
- [ ] Verify Firestore security rules are correct
- [ ] Set up domain restrictions on Firebase API key
- [ ] Enable HTTPS only
- [ ] Configure CORS if using from external domains

#### Features to Enable (Optional)
- [ ] Email verification for new accounts
- [ ] Password reset email customization
- [ ] Custom sender email domain
- [ ] Social authentication (Google, GitHub, etc.)
- [ ] Two-factor authentication
- [ ] Session timeout policies

#### Monitoring
- [ ] Set up Firebase alerts for quota usage
- [ ] Monitor authentication usage
- [ ] Set up Firestore usage alerts
- [ ] Configure error reporting

#### Testing
- [ ] Test all auth flows in production
- [ ] Test email reset functionality
- [ ] Verify user data is stored in Firestore
- [ ] Test with real users
- [ ] Performance testing

### Development Environment

#### Local Development
```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Open browser
# http://localhost:3000
```

#### Environment Variables (Optional)
If you want to use different Firebase projects for different environments:

```
# .env.local (development)
NEXT_PUBLIC_FIREBASE_API_KEY=AIzaSyBRqumqQoWU5FdnuN_WTn4O0tjiCy1YZSc
NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN=nutritwin-cd71b.firebaseapp.com
# ... other vars

# .env.production (production)
NEXT_PUBLIC_FIREBASE_API_KEY=production-key
# ... other vars
```

Then update `lib/firebase.ts`:
```typescript
const firebaseConfig = {
  apiKey: process.env.NEXT_PUBLIC_FIREBASE_API_KEY,
  authDomain: process.env.NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN,
  // ... other properties
};
```

### Testing Accounts

Create test accounts for different roles:

| Email | Password | Role | Purpose |
|-------|----------|------|---------|
| user@test.com | Test123! | user | Regular user testing |
| nutrition@test.com | Test123! | nutritionist | Nutritionist features |
| admin@test.com | Test123! | admin | Admin features |

### Monitoring & Analytics

#### Firebase Analytics (Optional)
- Already configured in `lib/firebase.ts`
- View analytics in Firebase Console > Analytics

#### Error Tracking
- Monitor authentication errors in Firebase Console
- Set up alerts for unusual activity

### Scaling Considerations

1. **Database Growth**
   - Monitor Firestore document count
   - Consider indexing for larger collections
   - Archive old data if needed

2. **Authentication Volume**
   - Firebase Auth scales automatically
   - No action needed for typical apps
   - Monitor quota if expecting high load

3. **Performance**
   - Enable Firestore caching
   - Use batch operations for bulk updates
   - Monitor query performance

### Backup & Recovery

1. **Regular Backups**
   - Firebase automatically backs up data
   - Export Firestore data periodically:
   ```bash
   gcloud firestore export gs://your-bucket/backups
   ```

2. **User Data Export**
   - Implement data export feature for users
   - Allow users to download their data (GDPR)

### Migration from PHP Backend

The old PHP authentication system can be retired after:
1. All users migrated to Firebase
2. Testing complete in production
3. No references to old system remain
4. Database backup taken

### Cost Optimization

1. **Firebase Pricing**
   - Free tier covers most small projects
   - Monitor usage at https://console.firebase.google.com/
   - Set up budget alerts

2. **Optimize Queries**
   - Use composite indexes for complex queries
   - Avoid inefficient queries
   - Use pagination for large datasets

### Support & Contacts

- Firebase Status: https://status.firebase.google.com/
- Firebase Support: https://firebase.google.com/support
- Documentation: https://firebase.google.com/docs

---

**Last Updated**: April 23, 2026
**Status**: Ready for Deployment
**Next Action**: Enable Email/Password auth in Firebase Console
