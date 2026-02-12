'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkGemoji from 'remark-gemoji';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  sources?: Source[];
}

interface Source {
  title: string;
  source_file: string;
  url: string;
  category: string;
  topics: string[];
  score: number;
  content_type: string;
  all_urls: string[];
}

interface ChatProps {
  initialQuery: string;
  onQueryConsumed: () => void;
}

const SUGGESTIONS = [
  'What are the best free courses to learn about RAG?',
  'Explain the difference between fine-tuning and RLHF',
  'What agent frameworks are covered in the resources?',
  'Give me a 5-day roadmap to learn LLM foundations',
  'What evaluation techniques are recommended for LLMs?',
  'Which courses cover multimodal AI models?',
];

export default function Chat({ initialQuery, onQueryConsumed }: ChatProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Handle initial query from explore mode
  useEffect(() => {
    if (initialQuery) {
      setInput(`Tell me about ${initialQuery} — what resources and courses are available?`);
      onQueryConsumed();
      inputRef.current?.focus();
    }
  }, [initialQuery, onQueryConsumed]);

  // Auto-scroll
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const sendMessage = useCallback(async (text?: string) => {
    const messageText = text || input.trim();
    if (!messageText || isLoading) return;

    const userMessage: Message = { role: 'user', content: messageText };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const history = messages.map(m => ({ role: m.role, content: m.content }));
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: messageText, history }),
      });

      if (!res.ok) throw new Error(`API error: ${res.status}`);

      const data = await res.json();
      const assistantMessage: Message = {
        role: 'assistant',
        content: data.answer,
        sources: data.sources,
      };
      setMessages(prev => [...prev, assistantMessage]);
    } catch (err) {
      setMessages(prev => [
        ...prev,
        {
          role: 'assistant',
          content: 'Sorry, I encountered an error connecting to the API. Please make sure the backend is running.',
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  }, [input, isLoading, messages]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const categoryColor = (cat: string) => {
    switch (cat) {
      case 'free_courses': return 'text-emerald-400';
      case 'resources': return 'text-violet-400';
      case 'interview_prep': return 'text-amber-400';
      case 'research_updates': return 'text-cyan-400';
      default: return 'text-slate-400';
    }
  };

  const categoryLabel = (cat: string) => {
    switch (cat) {
      case 'free_courses': return 'Course';
      case 'resources': return 'Resource';
      case 'interview_prep': return 'Interview';
      case 'research_updates': return 'Research';
      default: return cat;
    }
  };

  return (
    <div className="h-full flex flex-col max-w-4xl mx-auto">
      {/* Messages area */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto px-6 py-6 space-y-4">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full animate-fade-in">
            <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-violet-500/20 to-cyan-500/20 border border-violet-500/20 flex items-center justify-center mb-6">
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="var(--nebula)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="3"/>
                <path d="M12 1v4M12 19v4M4.2 4.2l2.8 2.8M17 17l2.8 2.8M1 12h4M19 12h4M4.2 19.8l2.8-2.8M17 7l2.8-2.8"/>
              </svg>
            </div>
            <h2 className="font-display text-2xl text-[var(--text-primary)] mb-2">
              Ask anything about GenAI
            </h2>
            <p className="text-sm text-[var(--text-muted)] mb-8 text-center max-w-md">
              I can help you navigate 90+ courses, research papers, roadmaps, and interview prep materials
              from the awesome-generative-ai-guide collection.
            </p>
            <div className="grid grid-cols-2 gap-2 w-full max-w-lg">
              {SUGGESTIONS.map((s, i) => (
                <button
                  key={i}
                  onClick={() => sendMessage(s)}
                  className="text-left text-xs text-[var(--text-secondary)] bg-[var(--surface)] hover:bg-[var(--surface-hover)] border border-[var(--border)] rounded-lg px-3 py-2.5 transition-all hover:border-violet-500/30"
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={i} className={`chat-message ${msg.role} rounded-xl px-5 py-4 max-w-3xl ${
            msg.role === 'user' ? 'ml-auto max-w-xl' : ''
          }`}>
            {/* Role label */}
            <div className={`text-[10px] uppercase tracking-wider font-medium mb-2 ${
              msg.role === 'user' ? 'text-[var(--text-muted)]' : 'text-violet-400'
            }`}>
              {msg.role === 'user' ? 'You' : 'Explorer'}
            </div>

            {/* Content */}
            <div className="prose-chat text-sm text-[var(--text-primary)]">
              <ReactMarkdown
                remarkPlugins={[remarkGfm, remarkGemoji]}
                components={{
                  a: ({ href, children }) => (
                    <a href={href} target="_blank" rel="noopener noreferrer">{children}</a>
                  ),
                }}
              >
                {msg.content}
              </ReactMarkdown>
            </div>

            {/* Sources */}
            {msg.sources && msg.sources.length > 0 && (
              <div className="mt-4 pt-3 border-t border-[var(--border)]">
                <div className="text-[10px] uppercase tracking-wider text-[var(--text-muted)] mb-2">
                  Sources
                </div>
                <div className="space-y-2">
                  {msg.sources.filter(s => s.title).slice(0, 6).map((source, j) => (
                    <div key={j} className="flex items-start gap-2 text-xs group">
                      {/* Content type badge */}
                      <span className={`flex-shrink-0 inline-flex items-center gap-1 font-medium min-w-[70px] ${
                        source.content_type === 'paper'
                          ? 'text-rose-400'
                          : categoryColor(source.category)
                      }`}>
                        {source.content_type === 'paper' ? (
                          <>
                            <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                              <polyline points="14 2 14 8 20 8"/>
                            </svg>
                            Paper
                          </>
                        ) : categoryLabel(source.category)}
                      </span>

                      {/* Title */}
                      <span className="text-[var(--text-secondary)] truncate flex-1">
                        {source.title}
                      </span>

                      {/* Action buttons */}
                      <span className="flex items-center gap-1.5 flex-shrink-0 opacity-60 group-hover:opacity-100 transition-opacity">
                        {/* Paper link — prominent */}
                        {source.content_type === 'paper' && source.url && (
                          <a
                            href={source.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-rose-500/15 text-rose-300 hover:bg-rose-500/25 transition-colors text-[10px] font-medium"
                          >
                            <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                              <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>
                              <polyline points="15 3 21 3 21 9"/>
                              <line x1="10" y1="14" x2="21" y2="3"/>
                            </svg>
                            Read Paper
                          </a>
                        )}
                        {/* Other URLs */}
                        {source.content_type !== 'paper' && source.url && (
                          <a
                            href={source.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-violet-500/10 text-violet-300 hover:bg-violet-500/20 transition-colors text-[10px] font-medium"
                          >
                            <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                              <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>
                              <polyline points="15 3 21 3 21 9"/>
                              <line x1="10" y1="14" x2="21" y2="3"/>
                            </svg>
                            Open
                          </a>
                        )}
                        {/* Additional URLs for papers */}
                        {source.all_urls && source.all_urls.length > 1 && (
                          <span className="text-[10px] text-[var(--text-muted)]">
                            +{source.all_urls.length - 1} links
                          </span>
                        )}
                        <span className="text-[10px] text-[var(--text-muted)]">
                          {(source.score * 100).toFixed(0)}%
                        </span>
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        ))}

        {/* Loading indicator */}
        {isLoading && (
          <div className="chat-message assistant rounded-xl px-5 py-4">
            <div className="text-[10px] uppercase tracking-wider font-medium mb-2 text-violet-400">
              Explorer
            </div>
            <div className="flex items-center gap-2">
              <div className="flex gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-violet-400 animate-bounce" style={{ animationDelay: '0ms' }}/>
                <span className="w-1.5 h-1.5 rounded-full bg-violet-400 animate-bounce" style={{ animationDelay: '150ms' }}/>
                <span className="w-1.5 h-1.5 rounded-full bg-violet-400 animate-bounce" style={{ animationDelay: '300ms' }}/>
              </div>
              <span className="text-xs text-[var(--text-muted)]">Searching knowledge base...</span>
            </div>
          </div>
        )}
      </div>

      {/* Input area */}
      <div className="flex-shrink-0 px-6 pb-6">
        <div className="chat-input-wrapper rounded-xl flex items-end">
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about GenAI courses, techniques, research..."
            rows={1}
            className="flex-1 bg-transparent text-sm text-[var(--text-primary)] placeholder:text-[var(--text-muted)] px-4 py-3 resize-none focus:outline-none"
            style={{ maxHeight: '120px' }}
          />
          <button
            onClick={() => sendMessage()}
            disabled={!input.trim() || isLoading}
            className="p-3 text-[var(--text-muted)] hover:text-violet-400 disabled:opacity-30 transition-colors"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/>
            </svg>
          </button>
        </div>
        <p className="text-[10px] text-[var(--text-muted)] text-center mt-2">
          Powered by RAG over awesome-generative-ai-guide · OpenAI gpt-4.1-mini · Responses may contain inaccuracies
        </p>
      </div>
    </div>
  );
}
