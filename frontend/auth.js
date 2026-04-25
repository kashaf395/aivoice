const API = 'http://localhost:5000';

function getToken() {
  return localStorage.getItem('token');
}

function getUser() {
  const u = localStorage.getItem('user');
  return u ? JSON.parse(u) : null;
}

function setUser(user, token) {
  localStorage.setItem('user', JSON.stringify(user));
  if (token) localStorage.setItem('token', token);
}

function clearAuth() {
  localStorage.removeItem('user');
  localStorage.removeItem('token');
}

async function apiCall(endpoint, options = {}) {
  const token = getToken();
  const headers = { 'Content-Type': 'application/json' };
  if (token) headers['Authorization'] = `Bearer ${token}`;

  try {
    const resp = await fetch(`${API}${endpoint}`, { ...options, headers });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || 'Request failed');
    return data;
  } catch (err) {
    throw err;
  }
}
