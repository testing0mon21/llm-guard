function verifyJwtToken(token, secretKey) {
  // Verify a JWT token
  const jwt = require('jsonwebtoken');
  
  try {
    const decoded = jwt.verify(token, secretKey);
    return { valid: true, payload: decoded };
  } catch (err) {
    return { valid: false, payload: null };
  }
}