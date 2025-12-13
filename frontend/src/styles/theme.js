/**
 * Apple-Style Design System for DeObfusca-AI
 * 
 * Inspired by Apple's Human Interface Guidelines:
 * - Clarity: Typography, icons, and layout emphasize important content
 * - Deference: Fluid motion and crisp, beautiful interface
 * - Depth: Visual layers and realistic motion convey hierarchy
 */

export const colors = {
  light: {
    // Primary Colors
    primary: '#007AFF',         // Apple Blue
    secondary: '#5856D6',       // Apple Purple
    success: '#34C759',         // Apple Green
    warning: '#FF9500',         // Apple Orange
    danger: '#FF3B30',          // Apple Red
    
    // Neutrals
    background: '#FFFFFF',
    surface: '#F2F2F7',         // Apple Gray 1
    surfaceSecondary: '#E5E5EA', // Apple Gray 2
    border: '#C6C6C8',
    
    // Text
    text: {
      primary: '#000000',
      secondary: '#3C3C43',     // 60% opacity
      tertiary: '#3C3C4399',    // 30% opacity
      quaternary: '#3C3C4366',  // 18% opacity
    },
    
    // Glassmorphism
    glass: 'rgba(255, 255, 255, 0.8)',
    glassHover: 'rgba(255, 255, 255, 0.9)',
    
    // Shadows
    shadow: {
      sm: '0 2px 8px rgba(0, 0, 0, 0.04)',
      md: '0 4px 16px rgba(0, 0, 0, 0.08)',
      lg: '0 12px 32px rgba(0, 0, 0, 0.12)',
      xl: '0 24px 64px rgba(0, 0, 0, 0.16)',
    }
  },
  
  dark: {
    // Primary Colors - adjusted for dark mode
    primary: '#0A84FF',         // Brighter Apple Blue
    secondary: '#5E5CE6',       // Brighter Purple
    success: '#32D74B',         // Brighter Green
    warning: '#FF9F0A',         // Brighter Orange
    danger: '#FF453A',          // Brighter Red
    
    // Neutrals
    background: '#000000',
    surface: '#1C1C1E',         // Apple Dark Gray 1
    surfaceSecondary: '#2C2C2E', // Apple Dark Gray 2
    border: '#38383A',
    
    // Text
    text: {
      primary: '#FFFFFF',
      secondary: '#EBEBF5',     // 60% opacity
      tertiary: '#EBEBF599',    // 30% opacity
      quaternary: '#EBEBF54D',  // 18% opacity
    },
    
    // Glassmorphism
    glass: 'rgba(28, 28, 30, 0.8)',
    glassHover: 'rgba(28, 28, 30, 0.9)',
    
    // Shadows
    shadow: {
      sm: '0 2px 8px rgba(0, 0, 0, 0.24)',
      md: '0 4px 16px rgba(0, 0, 0, 0.32)',
      lg: '0 12px 32px rgba(0, 0, 0, 0.48)',
      xl: '0 24px 64px rgba(0, 0, 0, 0.64)',
    }
  }
};

export const typography = {
  // SF Pro Display (Apple's system font)
  fontFamily: {
    primary: '-apple-system, BlinkMacSystemFont, "Segoe UI", "Roboto", "Oxygen", "Ubuntu", "Cantarell", "Fira Sans", "Droid Sans", "Helvetica Neue", sans-serif',
    mono: 'ui-monospace, SFMono-Regular, "SF Mono", Monaco, "Cascadia Code", "Roboto Mono", Menlo, Consolas, "Courier New", monospace',
  },
  
  // Apple Typography Scale
  fontSize: {
    // Large Titles
    largeTitle: {
      size: '34px',
      lineHeight: '41px',
      fontWeight: 700,
      letterSpacing: '-0.02em'
    },
    
    // Title Hierarchy
    title1: {
      size: '28px',
      lineHeight: '34px',
      fontWeight: 700,
      letterSpacing: '-0.015em'
    },
    title2: {
      size: '22px',
      lineHeight: '28px',
      fontWeight: 700,
      letterSpacing: '-0.012em'
    },
    title3: {
      size: '20px',
      lineHeight: '25px',
      fontWeight: 600,
      letterSpacing: '-0.01em'
    },
    
    // Headline
    headline: {
      size: '17px',
      lineHeight: '22px',
      fontWeight: 600,
      letterSpacing: '-0.008em'
    },
    
    // Body Text
    body: {
      size: '17px',
      lineHeight: '22px',
      fontWeight: 400,
      letterSpacing: '-0.006em'
    },
    callout: {
      size: '16px',
      lineHeight: '21px',
      fontWeight: 400,
      letterSpacing: '-0.005em'
    },
    subheadline: {
      size: '15px',
      lineHeight: '20px',
      fontWeight: 400,
      letterSpacing: '-0.004em'
    },
    footnote: {
      size: '13px',
      lineHeight: '18px',
      fontWeight: 400,
      letterSpacing: '-0.003em'
    },
    caption1: {
      size: '12px',
      lineHeight: '16px',
      fontWeight: 400,
      letterSpacing: '0'
    },
    caption2: {
      size: '11px',
      lineHeight: '13px',
      fontWeight: 400,
      letterSpacing: '0.001em'
    }
  }
};

export const spacing = {
  xs: '4px',
  sm: '8px',
  md: '16px',
  lg: '24px',
  xl: '32px',
  '2xl': '48px',
  '3xl': '64px',
  '4xl': '96px'
};

export const borderRadius = {
  xs: '4px',
  sm: '8px',
  md: '12px',
  lg: '16px',
  xl: '20px',
  '2xl': '24px',
  full: '9999px'
};

export const animation = {
  // Apple's signature spring animations
  spring: {
    damping: 0.8,
    stiffness: 100,
    mass: 0.5
  },
  
  // Easing curves
  easing: {
    standard: 'cubic-bezier(0.4, 0.0, 0.2, 1)',      // Material standard
    decelerate: 'cubic-bezier(0.0, 0.0, 0.2, 1)',     // Ease out
    accelerate: 'cubic-bezier(0.4, 0.0, 1, 1)',       // Ease in
    sharp: 'cubic-bezier(0.4, 0.0, 0.6, 1)',          // Sharp
    apple: 'cubic-bezier(0.25, 0.1, 0.25, 1)',        // Apple's signature
  },
  
  // Duration
  duration: {
    fast: '150ms',
    normal: '250ms',
    slow: '350ms',
    slower: '500ms'
  }
};

export const breakpoints = {
  xs: '320px',
  sm: '640px',
  md: '768px',
  lg: '1024px',
  xl: '1280px',
  '2xl': '1536px'
};

export const zIndex = {
  dropdown: 1000,
  sticky: 1100,
  modal: 1200,
  popover: 1300,
  tooltip: 1400,
  toast: 1500
};

export default {
  colors,
  typography,
  spacing,
  borderRadius,
  animation,
  breakpoints,
  zIndex
};
