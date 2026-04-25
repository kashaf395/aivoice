function renderNavbar() {
  const user = getUser();
  const currentPage = window.location.pathname.split('/').pop();

  const navHtml = `
    <a href="home.html" class="nav-brand">
      <img src="logo.jpg" alt="Logo" style="height: 32px; width: 32px; border-radius: 8px; object-fit: cover;">
      <span>AI Voice Detector</span>
    </a>
    <div class="nav-links">
      <a href="home.html" class="nav-link ${currentPage === 'home.html' ? 'active' : ''}">Home</a>
      <a href="analysis.html" class="nav-link ${currentPage === 'analysis.html' ? 'active' : ''}">Analyze</a>
      <a href="about.html" class="nav-link ${currentPage === 'about.html' ? 'active' : ''}">About Us</a>
      <a href="contact.html" class="nav-link ${currentPage === 'contact.html' ? 'active' : ''}">Contact</a>
      ${user && user.role === 'admin' ? `<a href="admin.html" class="nav-link ${currentPage === 'admin.html' ? 'active' : ''}">Admin</a>` : ''}
    </div>
    <div class="nav-auth">
      ${user ? `
        <a href="profile.html" class="btn-nav btn-login"><i class="fas fa-user" style="margin-right:4px"></i>${user.name.split(' ')[0]}</a>
        <button onclick="handleLogout()" class="btn-nav btn-logout-nav" title="Log Out"><i class="fas fa-sign-out-alt"></i></button>
      ` : `
        <a href="login.html" class="btn-nav btn-login">Log In</a>
        <a href="signup.html" class="btn-nav btn-signup">Get Started</a>
      `}
    </div>
  `;

  const navbar = document.getElementById('navbar');
  if (navbar) {
    navbar.innerHTML = navHtml;
  }
}

async function handleLogout() {
  try {
    await apiCall('/api/logout', { method: 'POST' });
  } catch(e) {}
  clearAuth();
  window.location.href = 'login.html';
}

document.addEventListener('DOMContentLoaded', renderNavbar);
