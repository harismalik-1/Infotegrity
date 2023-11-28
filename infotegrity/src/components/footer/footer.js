import React from 'react';
import './footer.css'; // Import the CSS file for styling

const Footer = () => {
  return (
    <footer className="footer-container">
      <p>&copy; {new Date().getFullYear()} Infotegrity. All rights reserved.</p>
      {/* Add any additional information you'd like to include in the footer */}
    </footer>
  );
};

export default Footer;
