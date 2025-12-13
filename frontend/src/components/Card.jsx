import React from 'react';
import styled from 'styled-components';

const CardContainer = styled.div`
  background: ${props => props.glass ? props.theme.glass : props.theme.surface};
  ${props => props.glass && `
    backdrop-filter: blur(20px) saturate(180%);
    -webkit-backdrop-filter: blur(20px) saturate(180%);
  `}
  border-radius: ${props => props.theme.borderRadius.lg};
  padding: ${props => props.theme.spacing.lg};
  border: 1px solid ${props => props.theme.border};
  box-shadow: ${props => props.theme.shadow.md};
  transition: all ${props => props.theme.animation.duration.normal} ${props => props.theme.animation.easing.apple};
  
  /* Hover Effect */
  ${props => props.hoverable && `
    cursor: pointer;
    
    &:hover {
      transform: translateY(-4px);
      box-shadow: ${props.theme.shadow.lg};
      border-color: ${props.theme.text.tertiary};
    }
    
    &:active {
      transform: translateY(-2px);
      box-shadow: ${props.theme.shadow.md};
    }
  `}
  
  /* Clickable */
  ${props => props.onClick && !props.hoverable && `
    cursor: pointer;
    
    &:active {
      transform: scale(0.98);
    }
  `}
`;

const CardHeader = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: ${props => props.theme.spacing.md};
  padding-bottom: ${props => props.theme.spacing.md};
  border-bottom: 1px solid ${props => props.theme.border};
`;

const CardTitle = styled.h3`
  font-size: ${props => props.theme.typography.fontSize.title3.size};
  font-weight: ${props => props.theme.typography.fontSize.title3.fontWeight};
  color: ${props => props.theme.text.primary};
  margin: 0;
`;

const CardBody = styled.div`
  color: ${props => props.theme.text.secondary};
  font-size: ${props => props.theme.typography.fontSize.body.size};
  line-height: 1.6;
`;

const CardFooter = styled.div`
  margin-top: ${props => props.theme.spacing.md};
  padding-top: ${props => props.theme.spacing.md};
  border-top: 1px solid ${props => props.theme.border};
  display: flex;
  align-items: center;
  justify-content: space-between;
`;

const Card = ({
  children,
  title,
  headerAction,
  footer,
  hoverable = false,
  glass = false,
  onClick,
  className,
  ...props
}) => {
  return (
    <CardContainer
      hoverable={hoverable}
      glass={glass}
      onClick={onClick}
      className={className}
      {...props}
    >
      {title && (
        <CardHeader>
          <CardTitle>{title}</CardTitle>
          {headerAction && <div>{headerAction}</div>}
        </CardHeader>
      )}
      
      <CardBody>
        {children}
      </CardBody>
      
      {footer && (
        <CardFooter>
          {footer}
        </CardFooter>
      )}
    </CardContainer>
  );
};

export default Card;
