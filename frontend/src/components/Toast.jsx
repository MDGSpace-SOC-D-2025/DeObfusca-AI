import React, { useEffect } from 'react';
import styled from 'styled-components';
import { X, CheckCircle, AlertCircle, Info } from 'lucide-react';

const ToastContainer = styled.div`
  position: fixed;
  top: ${props => props.theme.spacing.lg};
  right: ${props => props.theme.spacing.lg};
  z-index: ${props => props.theme.zIndex.toast};
  display: flex;
  flex-direction: column;
  gap: ${props => props.theme.spacing.sm};
  max-width: 400px;
`;

const ToastItem = styled.div`
  display: flex;
  align-items: flex-start;
  gap: ${props => props.theme.spacing.md};
  padding: ${props => props.theme.spacing.md};
  background: ${props => props.glass ? props.theme.glass : props.theme.surface};
  ${props => props.glass && `
    backdrop-filter: blur(20px) saturate(180%);
    -webkit-backdrop-filter: blur(20px) saturate(180%);
  `}
  border-radius: ${props => props.theme.borderRadius.lg};
  border: 1px solid ${props => getToastBorderColor(props)};
  box-shadow: ${props => props.theme.shadow.lg};
  animation: slideIn ${props => props.theme.animation.duration.normal} ${props => props.theme.animation.easing.apple};
  
  @keyframes slideIn {
    from {
      transform: translateX(120%);
      opacity: 0;
    }
    to {
      transform: translateX(0);
      opacity: 1;
    }
  }
  
  &.exiting {
    animation: slideOut ${props => props.theme.animation.duration.fast} ${props => props.theme.animation.easing.apple};
  }
  
  @keyframes slideOut {
    to {
      transform: translateX(120%);
      opacity: 0;
    }
  }
`;

const IconWrapper = styled.div`
  flex-shrink: 0;
  display: flex;
  align-items: center;
  color: ${props => getToastColor(props)};
  
  svg {
    width: 20px;
    height: 20px;
  }
`;

const ToastContent = styled.div`
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: ${props => props.theme.spacing.xs};
`;

const ToastTitle = styled.div`
  font-weight: 600;
  font-size: ${props => props.theme.typography.fontSize.callout.size};
  color: ${props => props.theme.text.primary};
`;

const ToastMessage = styled.div`
  font-size: ${props => props.theme.typography.fontSize.footnote.size};
  color: ${props => props.theme.text.secondary};
  line-height: 1.4;
`;

const CloseButton = styled.button`
  flex-shrink: 0;
  background: none;
  border: none;
  padding: 0;
  cursor: pointer;
  color: ${props => props.theme.text.tertiary};
  transition: color ${props => props.theme.animation.duration.fast};
  
  &:hover {
    color: ${props => props.theme.text.primary};
  }
  
  svg {
    width: 18px;
    height: 18px;
  }
`;

const getToastColor = (props) => {
  switch (props.type) {
    case 'success': return props.theme.success;
    case 'error': return props.theme.danger;
    case 'warning': return props.theme.warning;
    default: return props.theme.primary;
  }
};

const getToastBorderColor = (props) => {
  switch (props.type) {
    case 'success': return props.theme.success;
    case 'error': return props.theme.danger;
    case 'warning': return props.theme.warning;
    default: return props.theme.border;
  }
};

const getIcon = (type) => {
  switch (type) {
    case 'success': return CheckCircle;
    case 'error': return AlertCircle;
    case 'warning': return AlertCircle;
    default: return Info;
  }
};

export const Toast = ({ 
  title, 
  message, 
  type = 'info',
  onClose,
  duration = 5000,
  glass = true
}) => {
  const Icon = getIcon(type);
  
  useEffect(() => {
    if (duration && onClose) {
      const timer = setTimeout(onClose, duration);
      return () => clearTimeout(timer);
    }
  }, [duration, onClose]);
  
  return (
    <ToastItem type={type} glass={glass}>
      <IconWrapper type={type}>
        <Icon />
      </IconWrapper>
      
      <ToastContent>
        {title && <ToastTitle>{title}</ToastTitle>}
        {message && <ToastMessage>{message}</ToastMessage>}
      </ToastContent>
      
      {onClose && (
        <CloseButton onClick={onClose} aria-label="Close">
          <X />
        </CloseButton>
      )}
    </ToastItem>
  );
};

export const ToastProvider = ({ toasts = [], onRemove }) => {
  return (
    <ToastContainer>
      {toasts.map(toast => (
        <Toast
          key={toast.id}
          {...toast}
          onClose={() => onRemove(toast.id)}
        />
      ))}
    </ToastContainer>
  );
};

export default Toast;
