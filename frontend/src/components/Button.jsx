import React, { forwardRef } from 'react';
import styled from 'styled-components';

const ButtonBase = styled.button`
  font-family: ${props => props.theme.typography.fontFamily.primary};
  font-size: ${props => props.theme.typography.fontSize.body.size};
  font-weight: 600;
  letter-spacing: -0.01em;
  padding: ${props => getSizePadding(props.size)};
  border-radius: ${props => props.theme.borderRadius.md};
  cursor: pointer;
  border: none;
  transition: all ${props => props.theme.animation.duration.fast} ${props => props.theme.animation.easing.apple};
  position: relative;
  overflow: hidden;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: ${props => props.theme.spacing.sm};
  
  /* Variant Styles */
  ${props => getVariantStyles(props)}
  
  /* Disabled State */
  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
    pointer-events: none;
  }
  
  /* Hover Effect */
  &:hover:not(:disabled) {
    transform: translateY(-1px);
    box-shadow: ${props => props.theme.shadow.md};
  }
  
  /* Active State */
  &:active:not(:disabled) {
    transform: translateY(0) scale(0.98);
  }
  
  /* Loading State */
  ${props => props.loading && `
    pointer-events: none;
    opacity: 0.7;
  `}
  
  /* Full Width */
  ${props => props.fullWidth && `
    width: 100%;
  `}
  
  /* Icon Button */
  ${props => props.iconOnly && `
    padding: ${props.theme.spacing.sm};
    aspect-ratio: 1;
  `}
`;

const LoadingSpinner = styled.span`
  display: inline-block;
  width: 16px;
  height: 16px;
  border: 2px solid ${props => props.color || 'currentColor'};
  border-radius: 50%;
  border-top-color: transparent;
  animation: spin 0.6s linear infinite;
  
  @keyframes spin {
    to { transform: rotate(360deg); }
  }
`;

const getSizePadding = (size) => {
  switch (size) {
    case 'small':
      return '8px 16px';
    case 'large':
      return '14px 28px';
    default:
      return '10px 20px';
  }
};

const getVariantStyles = (props) => {
  const { variant, theme } = props;
  
  switch (variant) {
    case 'primary':
      return `
        background: ${theme.primary};
        color: white;
        box-shadow: ${theme.shadow.sm};
        
        &:hover:not(:disabled) {
          background: ${theme.primary};
          filter: brightness(1.1);
        }
      `;
      
    case 'secondary':
      return `
        background: ${theme.surface};
        color: ${theme.text.primary};
        border: 1px solid ${theme.border};
        
        &:hover:not(:disabled) {
          background: ${theme.surfaceSecondary};
          border-color: ${theme.text.tertiary};
        }
      `;
      
    case 'ghost':
      return `
        background: transparent;
        color: ${theme.text.primary};
        
        &:hover:not(:disabled) {
          background: ${theme.surface};
        }
      `;
      
    case 'danger':
      return `
        background: ${theme.danger};
        color: white;
        box-shadow: ${theme.shadow.sm};
        
        &:hover:not(:disabled) {
          filter: brightness(1.1);
        }
      `;
      
    case 'success':
      return `
        background: ${theme.success};
        color: white;
        box-shadow: ${theme.shadow.sm};
        
        &:hover:not(:disabled) {
          filter: brightness(1.1);
        }
      `;
      
    case 'glass':
      return `
        background: ${theme.glass};
        backdrop-filter: blur(20px) saturate(180%);
        -webkit-backdrop-filter: blur(20px) saturate(180%);
        color: ${theme.text.primary};
        border: 1px solid rgba(255, 255, 255, 0.125);
        
        &:hover:not(:disabled) {
          background: ${theme.glassHover};
        }
      `;
      
    default:
      return `
        background: ${theme.primary};
        color: white;
      `;
  }
};

const Button = forwardRef(({
  children,
  variant = 'primary',
  size = 'medium',
  loading = false,
  disabled = false,
  fullWidth = false,
  iconOnly = false,
  leftIcon,
  rightIcon,
  onClick,
  type = 'button',
  className,
  ...props
}, ref) => {
  return (
    <ButtonBase
      ref={ref}
      variant={variant}
      size={size}
      loading={loading}
      disabled={disabled || loading}
      fullWidth={fullWidth}
      iconOnly={iconOnly}
      onClick={onClick}
      type={type}
      className={className}
      {...props}
    >
      {loading && <LoadingSpinner />}
      {!loading && leftIcon && leftIcon}
      {!iconOnly && children}
      {!loading && rightIcon && rightIcon}
    </ButtonBase>
  );
});

Button.displayName = 'Button';

export default Button;
