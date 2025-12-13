import React, { createContext, useContext, useState, useEffect } from 'react';
import { ThemeProvider as StyledThemeProvider } from 'styled-components';
import theme from '../styles/theme';

const ThemeContext = createContext();

export const useTheme = () => {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within ThemeProvider');
  }
  return context;
};

export const ThemeProvider = ({ children }) => {
  // Check system preference first
  const getSystemTheme = () => {
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
      return 'dark';
    }
    return 'light';
  };

  // Load saved preference or use system theme
  const [mode, setMode] = useState(() => {
    const saved = localStorage.getItem('theme');
    return saved || getSystemTheme();
  });

  // Listen to system theme changes
  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    
    const handleChange = (e) => {
      // Only auto-switch if user hasn't set a preference
      if (!localStorage.getItem('theme')) {
        setMode(e.matches ? 'dark' : 'light');
      }
    };

    mediaQuery.addEventListener('change', handleChange);
    return () => mediaQuery.removeEventListener('change', handleChange);
  }, []);

  // Apply theme to document
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', mode);
    document.documentElement.style.colorScheme = mode;
    
    // Update meta theme-color for mobile browsers
    const metaTheme = document.querySelector('meta[name="theme-color"]');
    if (metaTheme) {
      metaTheme.setAttribute(
        'content',
        mode === 'dark' ? theme.colors.dark.background : theme.colors.light.background
      );
    }
  }, [mode]);

  const toggleTheme = () => {
    const newMode = mode === 'light' ? 'dark' : 'light';
    setMode(newMode);
    localStorage.setItem('theme', newMode);
  };

  const setTheme = (newMode) => {
    if (newMode === 'light' || newMode === 'dark') {
      setMode(newMode);
      localStorage.setItem('theme', newMode);
    }
  };

  const value = {
    mode,
    theme: theme.colors[mode],
    typography: theme.typography,
    spacing: theme.spacing,
    borderRadius: theme.borderRadius,
    animation: theme.animation,
    breakpoints: theme.breakpoints,
    zIndex: theme.zIndex,
    toggleTheme,
    setTheme,
    isDark: mode === 'dark'
  };

  // Combine all theme properties for styled-components
  const styledTheme = {
    ...theme.colors[mode],
    typography: theme.typography,
    spacing: theme.spacing,
    borderRadius: theme.borderRadius,
    animation: theme.animation,
    breakpoints: theme.breakpoints,
    zIndex: theme.zIndex,
    isDark: mode === 'dark'
  };

  return (
    <ThemeContext.Provider value={value}>
      <StyledThemeProvider theme={styledTheme}>
        {children}
      </StyledThemeProvider>
    </ThemeContext.Provider>
  );
};

export default ThemeContext;
