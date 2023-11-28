import React from 'react';
import { Outlet } from 'react-router-dom';
import Navbar from "../navbar/Navbar";
import Footer from '../footer/footer';

const Layout = () => {
  return (
    <>
      <Navbar />
      <Outlet />
      <Footer />
    </>
  );
};

export default Layout;
