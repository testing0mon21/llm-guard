function hashPassword(password, salt = null) {
  // Hash a password using PBKDF2
  const crypto = require('crypto');
  
  if (!salt) {
    salt = crypto.randomBytes(16).toString('hex');
  }
  
  const hash = crypto.pbkdf2Sync(password, salt, 1000, 64, 'sha512').toString('hex');
  return { salt, hash };
}