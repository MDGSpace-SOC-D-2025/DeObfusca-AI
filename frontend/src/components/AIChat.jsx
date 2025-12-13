// AI Chat Assistant Component
import React, { useState, useEffect, useRef } from 'react';
import { io } from 'socket.io-client';
import { useAuth } from '../context/AuthContext';
import Card from './Card';
import Button from './Button';
import './AIChat.css';

export default function AIChat({ context = {} }) {
  const { user } = useAuth();
  const [socket, setSocket] = useState(null);
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: '👋 Hi! I\'m your AI decompilation assistant. Ask me anything about your code, decompilation process, or binary analysis.',
      timestamp: new Date()
    }
  ]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    const newSocket = io('http://localhost:8000');
    
    newSocket.on('connect', () => {
      console.log('AI Chat connected');
      newSocket.emit('join', user.uid);
    });
    
    newSocket.on('ai_response', (data) => {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: data.response,
        reasoning: data.reasoning,
        timestamp: new Date()
      }]);
      setIsTyping(false);
    });
    
    newSocket.on('ai_error', (data) => {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `Sorry, I encountered an error: ${data.error}`,
        timestamp: new Date()
      }]);
      setIsTyping(false);
    });
    
    setSocket(newSocket);
    
    return () => {
      newSocket.close();
    };
  }, [user.uid]);

  const sendMessage = (e) => {
    e.preventDefault();
    
    if (!input.trim() || !socket) return;
    
    const userMessage = {
      role: 'user',
      content: input,
      timestamp: new Date()
    };
    
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsTyping(true);
    
    // Send to AI service
    socket.emit('ai_chat', {
      message: input,
      context,
      userId: user.uid
    });
  };

  const quickQuestions = [
    'Explain this decompiled code',
    'What optimization was applied?',
    'How accurate is this decompilation?',
    'What does this function do?',
    'Can you simplify this code?'
  ];

  const handleQuickQuestion = (question) => {
    setInput(question);
  };

  return (
    <div className="ai-chat">
      <Card>
        <div className="chat-header">
          <h3>🤖 AI Assistant</h3>
          <span className="status-indicator">
            {socket?.connected ? '🟢 Online' : '🔴 Offline'}
          </span>
        </div>
        
        <div className="chat-messages">
          {messages.map((msg, idx) => (
            <div key={idx} className={`message message-${msg.role}`}>
              <div className="message-avatar">
                {msg.role === 'user' ? '👤' : '🤖'}
              </div>
              <div className="message-content">
                <div className="message-text">{msg.content}</div>
                {msg.reasoning && (
                  <details className="message-reasoning">
                    <summary>View reasoning</summary>
                    <pre>{JSON.stringify(msg.reasoning, null, 2)}</pre>
                  </details>
                )}
                <span className="message-timestamp">
                  {msg.timestamp.toLocaleTimeString()}
                </span>
              </div>
            </div>
          ))}
          
          {isTyping && (
            <div className="message message-assistant">
              <div className="message-avatar">🤖</div>
              <div className="message-content">
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>
        
        <div className="quick-questions">
          {quickQuestions.map((q, idx) => (
            <button
              key={idx}
              className="quick-question-btn"
              onClick={() => handleQuickQuestion(q)}
            >
              {q}
            </button>
          ))}
        </div>
        
        <form onSubmit={sendMessage} className="chat-input-form">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask me anything..."
            className="chat-input"
            disabled={isTyping}
          />
          <Button type="submit" disabled={!input.trim() || isTyping}>
            Send
          </Button>
        </form>
      </Card>
    </div>
  );
}
