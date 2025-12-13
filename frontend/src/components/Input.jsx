import React, { forwardRef } from 'react';
import styled from 'styled-components';

const InputWrapper = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${props => props.theme.spacing.xs};
  width: 100%;
`;

const Label = styled.label`
  font-size: ${props => props.theme.typography.fontSize.subheadline.size};
  font-weight: 500;
  color: ${props => props.theme.text.secondary};
  letter-spacing: -0.01em;
`;

const InputBase = styled.input`
  font-family: ${props => props.theme.typography.fontFamily.primary};
  font-size: ${props => props.theme.typography.fontSize.body.size};
  padding: ${props => props.theme.spacing.sm} ${props => props.theme.spacing.md};
  background: ${props => props.theme.surface};
  color: ${props => props.theme.text.primary};
  border: 1px solid ${props => props.theme.border};
  border-radius: ${props => props.theme.borderRadius.md};
  transition: all ${props => props.theme.animation.duration.fast} ${props => props.theme.animation.easing.apple};
  width: 100%;
  
  &::placeholder {
    color: ${props => props.theme.text.tertiary};
  }
  
  &:hover:not(:disabled) {
    border-color: ${props => props.theme.text.tertiary};
  }
  
  &:focus {
    outline: none;
    border-color: ${props => props.theme.primary};
    box-shadow: 0 0 0 3px ${props => props.theme.primary}25;
    background: ${props => props.theme.background};
  }
  
  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  
  ${props => props.error && `
    border-color: ${props.theme.danger};
    
    &:focus {
      box-shadow: 0 0 0 3px ${props.theme.danger}25;
    }
  `}
  
  ${props => props.success && `
    border-color: ${props.theme.success};
    
    &:focus {
      box-shadow: 0 0 0 3px ${props.theme.success}25;
    }
  `}
`;

const TextAreaBase = styled.textarea`
  font-family: ${props => props.theme.typography.fontFamily.primary};
  font-size: ${props => props.theme.typography.fontSize.body.size};
  padding: ${props => props.theme.spacing.sm} ${props => props.theme.spacing.md};
  background: ${props => props.theme.surface};
  color: ${props => props.theme.text.primary};
  border: 1px solid ${props => props.theme.border};
  border-radius: ${props => props.theme.borderRadius.md};
  transition: all ${props => props.theme.animation.duration.fast} ${props => props.theme.animation.easing.apple};
  width: 100%;
  min-height: 120px;
  resize: vertical;
  
  &::placeholder {
    color: ${props => props.theme.text.tertiary};
  }
  
  &:hover:not(:disabled) {
    border-color: ${props => props.theme.text.tertiary};
  }
  
  &:focus {
    outline: none;
    border-color: ${props => props.theme.primary};
    box-shadow: 0 0 0 3px ${props => props.theme.primary}25;
    background: ${props => props.theme.background};
  }
  
  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  
  ${props => props.error && `
    border-color: ${props.theme.danger};
    
    &:focus {
      box-shadow: 0 0 0 3px ${props.theme.danger}25;
    }
  `}
`;

const HelperText = styled.span`
  font-size: ${props => props.theme.typography.fontSize.caption1.size};
  color: ${props => props.error ? props.theme.danger : props.theme.text.tertiary};
  margin-top: ${props => props.theme.spacing.xs};
`;

const Input = forwardRef(({
  label,
  error,
  success,
  helperText,
  multiline = false,
  rows = 4,
  className,
  ...props
}, ref) => {
  const Component = multiline ? TextAreaBase : InputBase;
  
  return (
    <InputWrapper className={className}>
      {label && <Label>{label}</Label>}
      <Component
        ref={ref}
        error={error}
        success={success}
        rows={multiline ? rows : undefined}
        {...props}
      />
      {helperText && <HelperText error={error}>{helperText}</HelperText>}
    </InputWrapper>
  );
});

Input.displayName = 'Input';

export default Input;
