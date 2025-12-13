import React from 'react';
import styled from 'styled-components';
import { useTheme } from '../context/ThemeContext';
import { Sun, Moon } from 'lucide-react';

const ToggleButton = styled.button`
  position: relative;
  width: 52px;
  height: 32px;
  background: ${props => props.theme.surfaceSecondary};
  border-radius: ${props => props.theme.borderRadius.full};
  padding: 4px;
  cursor: pointer;
  transition: background ${props => props.theme.animation.duration.normal} ${props => props.theme.animation.easing.apple};
  border: 1px solid ${props => props.theme.border};
  
  &:hover {
    background: ${props => props.theme.surface};
    box-shadow: ${props => props.theme.shadow.sm};
  }
  
  &:active {
    transform: scale(0.98);
  }
`;

const ToggleThumb = styled.div`
  position: absolute;
  top: 4px;
  left: ${props => props.isDark ? '24px' : '4px'};
  width: 24px;
  height: 24px;
  background: ${props => props.theme.primary};
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: left ${props => props.theme.animation.duration.normal} ${props => props.theme.animation.easing.apple},
              transform ${props => props.theme.animation.duration.fast} ${props => props.theme.animation.easing.apple};
  box-shadow: ${props => props.theme.shadow.md};
  
  svg {
    width: 14px;
    height: 14px;
    color: white;
  }
`;

const ThemeToggle = () => {
  const { isDark, toggleTheme } = useTheme();

  return (
    <ToggleButton onClick={toggleTheme} aria-label="Toggle theme">
      <ToggleThumb isDark={isDark}>
        {isDark ? <Moon /> : <Sun />}
      </ToggleThumb>
    </ToggleButton>
  );
};

export default ThemeToggle;
