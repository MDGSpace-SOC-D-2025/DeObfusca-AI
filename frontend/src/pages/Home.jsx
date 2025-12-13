import React from 'react';
import styled from 'styled-components';
import { Link } from 'react-router-dom';
import { useTheme } from '../context/ThemeContext';
import ThemeToggle from '../components/ThemeToggle';
import Button from '../components/Button';
import Card from '../components/Card';
import { 
  Code, 
  Zap, 
  Shield, 
  Cpu, 
  Brain, 
  GitBranch, 
  Target,
  ArrowRight,
  CheckCircle,
  Sparkles
} from 'lucide-react';

const HomeContainer = styled.div`
  min-height: 100vh;
  background: ${props => props.theme.background};
  color: ${props => props.theme.text.primary};
`;

const Navigation = styled.nav`
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  z-index: ${props => props.theme.zIndex.sticky};
  background: ${props => props.theme.glass};
  backdrop-filter: blur(20px) saturate(180%);
  -webkit-backdrop-filter: blur(20px) saturate(180%);
  border-bottom: 1px solid ${props => props.theme.border};
  padding: ${props => props.theme.spacing.md} ${props => props.theme.spacing.xl};
  display: flex;
  align-items: center;
  justify-content: space-between;
`;

const Logo = styled.div`
  font-size: ${props => props.theme.typography.fontSize.title3.size};
  font-weight: 700;
  background: linear-gradient(135deg, ${props => props.theme.primary}, ${props => props.theme.secondary});
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.sm};
  
  svg {
    color: ${props => props.theme.primary};
  }
`;

const NavActions = styled.div`
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.md};
`;

const Hero = styled.section`
  padding: 140px ${props => props.theme.spacing.xl} 80px;
  text-align: center;
  max-width: 1200px;
  margin: 0 auto;
`;

const HeroTitle = styled.h1`
  font-size: 64px;
  font-weight: 700;
  letter-spacing: -0.02em;
  margin-bottom: ${props => props.theme.spacing.lg};
  background: linear-gradient(135deg, ${props => props.theme.text.primary}, ${props => props.theme.text.secondary});
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  animation: fadeIn 0.6s ease-out;
  
  @keyframes fadeIn {
    from {
      opacity: 0;
      transform: translateY(20px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }
  
  @media (max-width: 768px) {
    font-size: 40px;
  }
`;

const HeroSubtitle = styled.p`
  font-size: ${props => props.theme.typography.fontSize.title3.size};
  color: ${props => props.theme.text.secondary};
  margin-bottom: ${props => props.theme.spacing['2xl']};
  max-width: 700px;
  margin-left: auto;
  margin-right: auto;
  animation: fadeIn 0.6s ease-out 0.2s both;
`;

const CTAButtons = styled.div`
  display: flex;
  gap: ${props => props.theme.spacing.md};
  justify-content: center;
  animation: fadeIn 0.6s ease-out 0.4s both;
  
  @media (max-width: 768px) {
    flex-direction: column;
    align-items: stretch;
  }
`;

const FeaturesSection = styled.section`
  padding: 80px ${props => props.theme.spacing.xl};
  background: ${props => props.theme.surface};
  border-top: 1px solid ${props => props.theme.border};
  border-bottom: 1px solid ${props => props.theme.border};
`;

const SectionTitle = styled.h2`
  font-size: ${props => props.theme.typography.fontSize.title1.size};
  font-weight: 700;
  text-align: center;
  margin-bottom: ${props => props.theme.spacing['2xl']};
  color: ${props => props.theme.text.primary};
`;

const FeaturesGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: ${props => props.theme.spacing.lg};
  max-width: 1200px;
  margin: 0 auto;
`;

const FeatureCard = styled(Card)`
  text-align: center;
  padding: ${props => props.theme.spacing.xl};
  transition: all ${props => props.theme.animation.duration.normal} ${props => props.theme.animation.easing.apple};
  
  &:hover {
    transform: translateY(-8px) scale(1.02);
  }
`;

const FeatureIcon = styled.div`
  width: 64px;
  height: 64px;
  margin: 0 auto ${props => props.theme.spacing.lg};
  background: linear-gradient(135deg, ${props => props.theme.primary}, ${props => props.theme.secondary});
  border-radius: ${props => props.theme.borderRadius.lg};
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  
  svg {
    width: 32px;
    height: 32px;
  }
`;

const FeatureTitle = styled.h3`
  font-size: ${props => props.theme.typography.fontSize.title3.size};
  font-weight: 600;
  margin-bottom: ${props => props.theme.spacing.sm};
  color: ${props => props.theme.text.primary};
`;

const FeatureDescription = styled.p`
  color: ${props => props.theme.text.secondary};
  font-size: ${props => props.theme.typography.fontSize.callout.size};
  line-height: 1.6;
`;

const ArchitectureSection = styled.section`
  padding: 80px ${props => props.theme.spacing.xl};
  max-width: 1200px;
  margin: 0 auto;
`;

const ArchitectureGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: ${props => props.theme.spacing.md};
  margin-bottom: ${props => props.theme.spacing['2xl']};
`;

const ArchStep = styled.div`
  padding: ${props => props.theme.spacing.lg};
  background: ${props => props.theme.surface};
  border: 1px solid ${props => props.theme.border};
  border-radius: ${props => props.theme.borderRadius.lg};
  position: relative;
  
  &::after {
    content: '→';
    position: absolute;
    right: -24px;
    top: 50%;
    transform: translateY(-50%);
    font-size: 24px;
    color: ${props => props.theme.text.tertiary};
  }
  
  &:last-child::after {
    display: none;
  }
  
  @media (max-width: 768px) {
    &::after {
      content: '↓';
      right: 50%;
      top: auto;
      bottom: -24px;
      transform: translateX(50%);
    }
  }
`;

const StepNumber = styled.div`
  width: 32px;
  height: 32px;
  background: ${props => props.theme.primary};
  color: white;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 700;
  margin-bottom: ${props => props.theme.spacing.sm};
`;

const StepTitle = styled.h4`
  font-size: ${props => props.theme.typography.fontSize.headline.size};
  font-weight: 600;
  margin-bottom: ${props => props.theme.spacing.xs};
  color: ${props => props.theme.text.primary};
`;

const StepDescription = styled.p`
  font-size: ${props => props.theme.typography.fontSize.footnote.size};
  color: ${props => props.theme.text.secondary};
  line-height: 1.4;
`;

const Footer = styled.footer`
  padding: ${props => props.theme.spacing.xl};
  background: ${props => props.theme.surface};
  border-top: 1px solid ${props => props.theme.border};
  text-align: center;
  color: ${props => props.theme.text.tertiary};
  font-size: ${props => props.theme.typography.fontSize.footnote.size};
`;

export default function Home() {
  const themeContext = useTheme();
  const { theme, typography, spacing, borderRadius, animation, zIndex } = themeContext;

  const features = [
    {
      icon: <Brain />,
      title: 'Advanced AI Models',
      description: 'Utilizes Edge-Augmented Graph Transformers, Hierarchical LLMs with RAG, and Neural-Symbolic execution for SOTA deobfuscation.'
    },
    {
      icon: <GitBranch />,
      title: 'Code Property Graphs',
      description: 'Fuses CFG, AST, and PDG into hypergraphs with dominator trees for precise obfuscation detection.'
    },
    {
      icon: <Target />,
      title: 'Verify-Refine Loop',
      description: 'Iterative refinement with Z3 symbolic execution proves mathematical equivalence.'
    },
    {
      icon: <Zap />,
      title: 'Grammar-Guided Decoding',
      description: 'BNF constraints eliminate LLM hallucinations, ensuring 100% syntactically correct C code.'
    },
    {
      icon: <Shield />,
      title: 'Batch Processing',
      description: 'Upload multiple binaries for parallel processing with comprehensive history tracking.'
    },
    {
      icon: <Sparkles />,
      title: 'Beautiful Interface',
      description: 'Apple-inspired design with dark/light modes and fluid animations.'
    }
  ];

  const architectureSteps = [
    { title: 'Binary Upload', desc: 'Upload obfuscated binary' },
    { title: 'Ghidra Analysis', desc: 'Extract P-Code' },
    { title: 'CPG Builder', desc: 'Build hypergraph' },
    { title: 'Graph Transformer', desc: 'Detect junk code' },
    { title: 'Hierarchical LLM', desc: 'Decompile with RAG' },
    { title: 'Z3 Verification', desc: 'Prove equivalence' }
  ];

  return (
    <HomeContainer>
      <Navigation>
        <Logo>
          <Code />
          DeObfusca-AI
        </Logo>
        
        <NavActions>
          <ThemeToggle />
          <Link to="/login">
            <Button variant="ghost">Login</Button>
          </Link>
          <Link to="/signup">
            <Button variant="primary">Get Started</Button>
          </Link>
        </NavActions>
      </Navigation>

      <Hero>
        <HeroTitle>
          Neural Reverse Engineering
        </HeroTitle>
        <HeroSubtitle>
          State-of-the-art AI deobfuscation with Code Property Graphs, 
          Graph Transformers, and Symbolic Execution
        </HeroSubtitle>
        <CTAButtons>
          <Link to="/signup">
            <Button 
              variant="primary" 
              size="large"
              rightIcon={<ArrowRight />}
            >
              Start Deobfuscating
            </Button>
          </Link>
          <Link to="/login">
            <Button variant="glass" size="large">
              Learn More
            </Button>
          </Link>
        </CTAButtons>
      </Hero>

      <FeaturesSection>
        <SectionTitle>
          Research-Grade Architecture
        </SectionTitle>
        <FeaturesGrid>
          {features.map((feature, index) => (
            <FeatureCard 
              key={index}
              hoverable
             
             
             
             
            >
              <FeatureIcon>
                {feature.icon}
              </FeatureIcon>
              <FeatureTitle>
                {feature.title}
              </FeatureTitle>
              <FeatureDescription>
                {feature.description}
              </FeatureDescription>
            </FeatureCard>
          ))}
        </FeaturesGrid>
      </FeaturesSection>

      <ArchitectureSection>
        <SectionTitle>
          Verify-Refine Pipeline
        </SectionTitle>
        <ArchitectureGrid>
          {architectureSteps.map((step, index) => (
            <ArchStep key={index}>
              <StepNumber>
                {index + 1}
              </StepNumber>
              <StepTitle>
                {step.title}
              </StepTitle>
              <StepDescription>
                {step.desc}
              </StepDescription>
            </ArchStep>
          ))}
        </ArchitectureGrid>
      </ArchitectureSection>

      <Footer>
        © 2024 DeObfusca-AI. Powered by Neural-Symbolic AI.
      </Footer>
    </HomeContainer>
  );
}
