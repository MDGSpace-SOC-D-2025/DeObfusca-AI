import { createGlobalStyle } from 'styled-components';
import theme from './theme';

const GlobalStyles = createGlobalStyle`
  /* CSS Reset & Base Styles */
  *, *::before, *::after {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
  }

  html {
    font-size: 16px;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    text-rendering: optimizeLegibility;
    scroll-behavior: smooth;
  }

  body {
    font-family: ${theme.typography.fontFamily.primary};
    background-color: ${props => props.theme.background};
    color: ${props => props.theme.text.primary};
    transition: background-color ${theme.animation.duration.normal} ${theme.animation.easing.apple},
                color ${theme.animation.duration.normal} ${theme.animation.easing.apple};
    line-height: 1.5;
    overflow-x: hidden;
  }

  /* Apple-style Selection */
  ::selection {
    background-color: ${props => props.isDark ? 'rgba(10, 132, 255, 0.3)' : 'rgba(0, 122, 255, 0.3)'};
    color: ${props => props.theme.text.primary};
  }

  /* Scrollbar Styling (WebKit) */
  ::-webkit-scrollbar {
    width: 12px;
    height: 12px;
  }

  ::-webkit-scrollbar-track {
    background: ${props => props.theme.surface};
  }

  ::-webkit-scrollbar-thumb {
    background: ${props => props.theme.border};
    border-radius: ${theme.borderRadius.md};
    border: 2px solid ${props => props.theme.surface};
    transition: background ${theme.animation.duration.fast};
  }

  ::-webkit-scrollbar-thumb:hover {
    background: ${props => props.theme.text.tertiary};
  }

  /* Firefox Scrollbar */
  * {
    scrollbar-width: thin;
    scrollbar-color: ${props => props.theme.border} ${props => props.theme.surface};
  }

  /* Focus Styles - Apple-like */
  *:focus {
    outline: none;
  }

  *:focus-visible {
    outline: 2px solid ${props => props.theme.primary};
    outline-offset: 2px;
    border-radius: ${theme.borderRadius.sm};
  }

  /* Typography */
  h1, h2, h3, h4, h5, h6 {
    font-weight: 600;
    letter-spacing: -0.01em;
    line-height: 1.2;
  }

  h1 {
    font-size: ${theme.typography.fontSize.largeTitle.size};
    line-height: ${theme.typography.fontSize.largeTitle.lineHeight};
    font-weight: ${theme.typography.fontSize.largeTitle.fontWeight};
    letter-spacing: ${theme.typography.fontSize.largeTitle.letterSpacing};
  }

  h2 {
    font-size: ${theme.typography.fontSize.title1.size};
    line-height: ${theme.typography.fontSize.title1.lineHeight};
    font-weight: ${theme.typography.fontSize.title1.fontWeight};
    letter-spacing: ${theme.typography.fontSize.title1.letterSpacing};
  }

  h3 {
    font-size: ${theme.typography.fontSize.title2.size};
    line-height: ${theme.typography.fontSize.title2.lineHeight};
    font-weight: ${theme.typography.fontSize.title2.fontWeight};
    letter-spacing: ${theme.typography.fontSize.title2.letterSpacing};
  }

  p {
    font-size: ${theme.typography.fontSize.body.size};
    line-height: ${theme.typography.fontSize.body.lineHeight};
    letter-spacing: ${theme.typography.fontSize.body.letterSpacing};
  }

  /* Links */
  a {
    color: ${props => props.theme.primary};
    text-decoration: none;
    transition: opacity ${theme.animation.duration.fast} ${theme.animation.easing.apple};
  }

  a:hover {
    opacity: 0.8;
  }

  a:active {
    opacity: 0.6;
  }

  /* Code Blocks */
  code, pre {
    font-family: ${theme.typography.fontFamily.mono};
    font-size: 0.9em;
  }

  pre {
    background: ${props => props.theme.surface};
    padding: ${theme.spacing.md};
    border-radius: ${theme.borderRadius.md};
    overflow-x: auto;
    border: 1px solid ${props => props.theme.border};
  }

  code {
    background: ${props => props.theme.surface};
    padding: 2px 6px;
    border-radius: ${theme.borderRadius.xs};
    font-size: 0.875em;
  }

  pre code {
    background: none;
    padding: 0;
  }

  /* Buttons - Apple Style */
  button {
    font-family: inherit;
    cursor: pointer;
    border: none;
    background: none;
    transition: all ${theme.animation.duration.fast} ${theme.animation.easing.apple};
  }

  /* Inputs - Apple Style */
  input, textarea, select {
    font-family: inherit;
    font-size: inherit;
    border: 1px solid ${props => props.theme.border};
    border-radius: ${theme.borderRadius.md};
    padding: ${theme.spacing.sm} ${theme.spacing.md};
    background: ${props => props.theme.surface};
    color: ${props => props.theme.text.primary};
    transition: all ${theme.animation.duration.fast} ${theme.animation.easing.apple};
  }

  input:hover, textarea:hover, select:hover {
    border-color: ${props => props.theme.text.tertiary};
  }

  input:focus, textarea:focus, select:focus {
    border-color: ${props => props.theme.primary};
    box-shadow: 0 0 0 3px ${props => props.isDark ? 'rgba(10, 132, 255, 0.15)' : 'rgba(0, 122, 255, 0.15)'};
  }

  /* Glassmorphism effect */
  .glass {
    background: ${props => props.theme.glass};
    backdrop-filter: blur(20px) saturate(180%);
    -webkit-backdrop-filter: blur(20px) saturate(180%);
    border: 1px solid rgba(255, 255, 255, 0.125);
  }

  /* Loading Animation */
  @keyframes shimmer {
    0% {
      background-position: -1000px 0;
    }
    100% {
      background-position: 1000px 0;
    }
  }

  .skeleton {
    background: linear-gradient(
      90deg,
      ${props => props.theme.surface} 0%,
      ${props => props.theme.surfaceSecondary} 50%,
      ${props => props.theme.surface} 100%
    );
    background-size: 1000px 100%;
    animation: shimmer 2s infinite linear;
  }

  /* Fade In Animation */
  @keyframes fadeIn {
    from {
      opacity: 0;
      transform: translateY(10px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }

  .fade-in {
    animation: fadeIn ${theme.animation.duration.normal} ${theme.animation.easing.apple};
  }

  /* Scale Animation */
  @keyframes scaleIn {
    from {
      opacity: 0;
      transform: scale(0.95);
    }
    to {
      opacity: 1;
      transform: scale(1);
    }
  }

  .scale-in {
    animation: scaleIn ${theme.animation.duration.normal} ${theme.animation.easing.apple};
  }

  /* Utility Classes */
  .text-center { text-align: center; }
  .text-left { text-align: left; }
  .text-right { text-align: right; }
  
  .flex { display: flex; }
  .flex-col { flex-direction: column; }
  .items-center { align-items: center; }
  .justify-center { justify-content: center; }
  .justify-between { justify-content: space-between; }
  
  .hidden { display: none; }
  .visible { visibility: visible; }
  .invisible { visibility: hidden; }

  /* Responsive Typography */
  @media (max-width: ${theme.breakpoints.md}) {
    html {
      font-size: 14px;
    }
    
    h1 {
      font-size: 28px;
    }
    
    h2 {
      font-size: 24px;
    }
  }

  /* Print Styles */
  @media print {
    body {
      background: white;
      color: black;
    }
    
    .no-print {
      display: none;
    }
  }
`;

export default GlobalStyles;
