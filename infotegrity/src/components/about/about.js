import React from 'react';
import './about.css'; // Import the CSS file for styling

const About = () => {
  return (
    <div className="about-container">
      <h1>About Infotegrity</h1>
      
      <section className="mission-section">
        <h2>Our Mission</h2>
        <p>
          Infotegrity aims to provide reliable and unbiased information analysis, 
          helping individuals and organizations make informed decisions. Our future goals 
          include expanding our analytical capabilities and reaching a broader audience.
        </p>
      </section>

      <section className="team-section">
        <h2>Our Team</h2>
        <div className="team-members">
          {/* Replace these with actual images and names */}
          <div className="team-member">
            <img src="/haris.png" alt="Haris Malik" />
            <h3>Haris Malik</h3>
            <p>
            Computer Science Specialist @ University of Toronto - Software Engineering Ex-SWE Intern @TD
            </p>
          </div>
          <div className="team-member">
            <img src="/kody.png" alt="Kody Chik" />
            <h3>Kody Chik</h3>
            <p>Computer Science Specialist @ University of Toronto - Software Engineering​Lead Developer @Hellopayz</p>
          </div>
          {/* Repeat for other team members */}
        </div>
      </section>
    </div>
  );
};

export default About;
