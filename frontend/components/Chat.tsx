'use client';

import { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkGemoji from 'remark-gemoji';
import { sanitizeUrl } from '@/lib/sanitizeUrl';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  sources?: Source[];
  isError?: boolean;
  feedback?: 'up' | 'down' | null;
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
  text_snippet?: string;
}

interface ChatProps {
  initialQuery: string;
  onQueryConsumed: () => void;
  onExploreInGraph?: (topicId: string) => void;
  onSwitchToExplore?: () => void;
}

const SUGGESTIONS = [
  'How does RAG work, and what retrieval strategies are compared in the resources?',
  'What types of AI agents are described, and how do they use tools and planning?',
  'Explain LoRA and QLoRA — how do parameter-efficient fine-tuning methods work?',
  'What are the main safety and alignment challenges for LLMs, and how does RLHF address them?',
];

const MAX_SESSION_MESSAGES = 50;

/** Parse SSE stream and dispatch events. */
async function readSSEStream(
  response: Response,
  onSources: (sources: Source[]) => void,
  onToken: (text: string) => void,
  onCorrected: (answer: string) => void,
  onError: (detail: string) => void,
  onDone: () => void,
) {
  const reader = response.body?.getReader();
  if (!reader) { onError('No response stream'); return; }

  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // Parse SSE events from buffer
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    let currentEvent = '';
    for (const line of lines) {
      if (line.startsWith('event: ')) {
        currentEvent = line.slice(7).trim();
      } else if (line.startsWith('data: ')) {
        const data = line.slice(6);
        try {
          const parsed = JSON.parse(data);
          switch (currentEvent) {
            case 'sources':
              onSources(parsed as Source[]);
              break;
            case 'token':
              onToken(parsed.text);
              break;
            case 'corrected':
              onCorrected(parsed.answer);
              break;
            case 'error':
              onError(parsed.detail || 'Unknown error');
              break;
            case 'done':
              onDone();
              break;
          }
        } catch {
          // skip malformed JSON
        }
        currentEvent = '';
      }
    }
  }
}

export default function Chat({ initialQuery, onQueryConsumed, onExploreInGraph, onSwitchToExplore }: ChatProps) {
  const [messages, setMessages] = useState<Message[]>(() => {
    if (typeof window === 'undefined') return [];
    try {
      const saved = sessionStorage.getItem('chat-messages');
      return saved ? JSON.parse(saved) : [];
    } catch { return []; }
  });
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [cooldown, setCooldown] = useState(false);
  const [expandedSources, setExpandedSources] = useState<Set<string>>(new Set());
  const cooldownRef = useRef<ReturnType<typeof setTimeout>>();
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const streamContentRef = useRef('');

  // Cleanup cooldown timer
  useEffect(() => {
    return () => { if (cooldownRef.current) clearTimeout(cooldownRef.current); };
  }, []);

  // Persist messages to sessionStorage
  useEffect(() => {
    try {
      sessionStorage.setItem('chat-messages', JSON.stringify(messages));
    } catch { /* quota exceeded — ignore */ }
  }, [messages]);

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
  }, [messages, isStreaming]);

  const userMessageCount = useMemo(
    () => messages.filter(m => m.role === 'user').length,
    [messages],
  );

  const toggleSourceExpanded = useCallback((msgIdx: number, srcIdx: number) => {
    const key = `${msgIdx}-${srcIdx}`;
    setExpandedSources(prev => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }, []);

  const setFeedback = useCallback((msgIdx: number, feedback: 'up' | 'down') => {
    setMessages(prev => prev.map((msg, i) => {
      if (i !== msgIdx) return msg;
      return { ...msg, feedback: msg.feedback === feedback ? null : feedback };
    }));
  }, []);

  /** Scroll a source card into view */
  const scrollToSource = useCallback((msgIdx: number, sourceNum: number) => {
    const el = document.getElementById(`source-${msgIdx}-${sourceNum - 1}`);
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      el.classList.add('ring-1', 'ring-violet-500/50');
      setTimeout(() => el.classList.remove('ring-1', 'ring-violet-500/50'), 2000);
    }
  }, []);

  const sendMessage = useCallback(async (text?: string) => {
    const messageText = text || input.trim();
    if (!messageText || isLoading || isStreaming || cooldown) return;

    if (userMessageCount >= MAX_SESSION_MESSAGES) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'You\'ve reached the message limit for this session. Click "New Chat" above to start a new conversation.',
        isError: true,
      }]);
      return;
    }

    const userMessage: Message = { role: 'user', content: messageText };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const history = messages.map(m => ({ role: m.role, content: m.content }));

      // Use streaming mode
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: messageText, history, stream: true }),
      });

      if (!res.ok) {
        throw new Error(res.status === 429 ? 'RATE_LIMITED' : `API error: ${res.status}`);
      }

      // Check if server returned SSE or JSON
      const contentType = res.headers.get('content-type') || '';
      if (contentType.includes('text/event-stream')) {
        // Streaming mode
        setIsLoading(false);
        setIsStreaming(true);
        streamContentRef.current = '';

        // Add placeholder assistant message
        setMessages(prev => [...prev, { role: 'assistant', content: '', sources: [] }]);

        await readSSEStream(
          res,
          (sources) => {
            setMessages(prev => {
              const copy = [...prev];
              const last = copy[copy.length - 1];
              if (last.role === 'assistant') {
                copy[copy.length - 1] = { ...last, sources };
              }
              return copy;
            });
          },
          (token) => {
            streamContentRef.current += token;
            const content = streamContentRef.current;
            setMessages(prev => {
              const copy = [...prev];
              const last = copy[copy.length - 1];
              if (last.role === 'assistant') {
                copy[copy.length - 1] = { ...last, content };
              }
              return copy;
            });
          },
          (correctedAnswer) => {
            setMessages(prev => {
              const copy = [...prev];
              const last = copy[copy.length - 1];
              if (last.role === 'assistant') {
                copy[copy.length - 1] = { ...last, content: correctedAnswer };
              }
              return copy;
            });
          },
          (detail) => {
            setMessages(prev => {
              const copy = [...prev];
              const last = copy[copy.length - 1];
              if (last.role === 'assistant') {
                copy[copy.length - 1] = { ...last, content: detail, isError: true };
              }
              return copy;
            });
          },
          () => { /* done */ },
        );
      } else {
        // Fallback: non-streaming JSON response
        const data = await res.json();
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: data.answer,
          sources: data.sources,
        }]);
      }
    } catch (err) {
      const isRateLimited = err instanceof Error && err.message === 'RATE_LIMITED';
      setMessages(prev => [
        ...prev,
        {
          role: 'assistant',
          content: isRateLimited
            ? 'You\'re sending messages too quickly. Please wait a moment before trying again.'
            : 'Sorry, I encountered an error connecting to the API. Please make sure the backend is running.',
          isError: true,
        },
      ]);
    } finally {
      setIsLoading(false);
      setIsStreaming(false);
      streamContentRef.current = '';
      setCooldown(true);
      cooldownRef.current = setTimeout(() => setCooldown(false), 2000);
    }
  }, [input, isLoading, isStreaming, cooldown, messages, userMessageCount]);

  const resetChat = useCallback(() => {
    setMessages([]);
    setInput('');
    setIsLoading(false);
    setIsStreaming(false);
    setCooldown(false);
    setExpandedSources(new Set());
    if (cooldownRef.current) clearTimeout(cooldownRef.current);
    try { sessionStorage.removeItem('chat-messages'); } catch {}
    setTimeout(() => inputRef.current?.focus(), 100);
  }, []);

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
      case 'root': return 'text-slate-400';
      default: return 'text-slate-400';
    }
  };

  const categoryLabel = (cat: string) => {
    switch (cat) {
      case 'free_courses': return 'Course';
      case 'resources': return 'Resource';
      case 'interview_prep': return 'Interview';
      case 'research_updates': return 'Research';
      case 'root': return 'Guide';
      default: return cat;
    }
  };

  /** Render markdown with clickable [Source N] citations */
  const renderContent = (content: string, msgIdx: number) => {
    // Replace [Source N] with clickable links
    const processed = content.replace(
      /\[Source (\d+)\]/g,
      (match, num) => `[${match}](#source-ref-${msgIdx}-${num})`
    );

    return (
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkGemoji]}
        components={{
          a: ({ href, children }) => {
            // Handle source citation clicks
            const sourceMatch = href?.match(/#source-ref-(\d+)-(\d+)/);
            if (sourceMatch) {
              const mi = parseInt(sourceMatch[1], 10);
              const sn = parseInt(sourceMatch[2], 10);
              return (
                <button
                  onClick={() => scrollToSource(mi, sn)}
                  className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-semibold bg-violet-500/15 text-violet-300 hover:bg-violet-500/25 transition-colors cursor-pointer no-underline align-baseline"
                >
                  {String(children)}
                </button>
              );
            }
            const safeHref = sanitizeUrl(href);
            if (!safeHref) return <span>{children}</span>;
            return <a href={safeHref} target="_blank" rel="noopener noreferrer">{children}</a>;
          },
        }}
      >
        {processed}
      </ReactMarkdown>
    );
  };

  return (
    <div className="h-full flex flex-col max-w-4xl mx-auto">
      {/* Messages area */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto px-6 py-6 space-y-4">
        {messages.length > 0 && (
          <div className="sticky top-0 z-10 flex justify-end pb-2">
            <button
              onClick={resetChat}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--surface)] border border-transparent hover:border-[var(--border)] transition-all"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <line x1="12" y1="5" x2="12" y2="19"/>
                <line x1="5" y1="12" x2="19" y2="12"/>
              </svg>
              New Chat
            </button>
          </div>
        )}
        {messages.length === 0 && (
          <div className="flex flex-col items-start justify-center h-full animate-fade-in max-w-2xl mx-auto w-full">
            <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-violet-500/20 to-cyan-500/20 border border-violet-500/20 flex items-center justify-center mb-6">
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="var(--nebula)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="3"/>
                <path d="M12 1v4M12 19v4M4.2 4.2l2.8 2.8M17 17l2.8 2.8M1 12h4M19 12h4M4.2 19.8l2.8-2.8M17 7l2.8-2.8"/>
              </svg>
            </div>
            <h2 className="font-display text-2xl text-[var(--text-primary)] mb-2">
              Ask anything about GenAI
            </h2>
            <p className="text-sm text-[var(--text-muted)] mb-8 max-w-md">
              I can help you navigate 90+ courses, research papers, roadmaps, and interview prep materials
              from the awesome-generative-ai-guide collection.
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 w-full max-w-lg">
              {SUGGESTIONS.map((s, i) => (
                <button
                  key={i}
                  onClick={() => sendMessage(s)}
                  className="text-left text-sm text-[var(--text-secondary)] bg-[var(--surface)] hover:bg-[var(--surface-hover)] border border-[var(--border)] rounded-lg px-4 py-3.5 min-h-[48px] transition-all hover:border-violet-500/30 leading-snug"
                >
                  {s}
                </button>
              ))}
            </div>

            {/* Explore mode promotion */}
            {onSwitchToExplore && (
              <>
                <div className="mt-6 flex items-center gap-3 text-xs text-[var(--text-muted)]">
                  <span className="w-8 h-px bg-[var(--border)]" />
                  <span>or</span>
                  <span className="w-8 h-px bg-[var(--border)]" />
                </div>
                <button
                  onClick={onSwitchToExplore}
                  className="mt-3 flex items-center gap-2 text-xs text-cyan-300 hover:text-cyan-200 transition-colors"
                >
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <circle cx="12" cy="12" r="10"/>
                    <polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"/>
                  </svg>
                  Explore the knowledge graph visually
                </button>
              </>
            )}
          </div>
        )}

        {messages.map((msg, msgIdx) => (
          <div key={msgIdx} className={`chat-message ${msg.role} rounded-xl px-5 py-4 max-w-3xl ${
            msg.role === 'user' ? 'ml-auto max-w-xl' : ''
          } ${msg.isError ? 'border-l-2 !border-l-rose-500/50 !bg-rose-500/5' : ''}`}>
            {/* Role label */}
            <div className={`text-xs uppercase tracking-wider font-medium mb-2 ${
              msg.role === 'user' ? 'text-[var(--text-muted)]' : 'text-violet-400'
            }`}>
              {msg.role === 'user' ? 'You' : 'Explorer'}
            </div>

            {/* Content */}
            <div className="prose-chat text-sm text-[var(--text-primary)]">
              {renderContent(msg.content, msgIdx)}
            </div>

            {/* Feedback buttons */}
            {msg.role === 'assistant' && msg.content && !msg.isError && (
              <div className="flex items-center gap-1 mt-3">
                <button
                  onClick={() => setFeedback(msgIdx, 'up')}
                  className={`p-2.5 rounded-lg transition-colors ${
                    msg.feedback === 'up'
                      ? 'text-emerald-400 bg-emerald-500/15'
                      : 'text-[var(--text-muted)] hover:text-emerald-400 hover:bg-emerald-500/10'
                  }`}
                  title="Helpful"
                >
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"/>
                  </svg>
                </button>
                <button
                  onClick={() => setFeedback(msgIdx, 'down')}
                  className={`p-2.5 rounded-lg transition-colors ${
                    msg.feedback === 'down'
                      ? 'text-rose-400 bg-rose-500/15'
                      : 'text-[var(--text-muted)] hover:text-rose-400 hover:bg-rose-500/10'
                  }`}
                  title="Not helpful"
                >
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm7-13h2.67A2.31 2.31 0 0 1 22 4v7a2.31 2.31 0 0 1-2.33 2H17"/>
                  </svg>
                </button>
              </div>
            )}

            {/* Sources */}
            {msg.sources && msg.sources.length > 0 && (
              <div className="mt-4 pt-3 border-t border-[var(--border)]">
                <div className="text-xs uppercase tracking-wider text-[var(--text-muted)] mb-2">
                  Sources
                </div>
                <div className="space-y-2">
                  {msg.sources.filter(s => s.title).slice(0, 8).map((source, j) => {
                    const isExpanded = expandedSources.has(`${msgIdx}-${j}`);
                    return (
                      <div key={j} id={`source-${msgIdx}-${j}`} className="text-xs rounded-lg transition-all">
                        <div className="flex items-start gap-2 group">
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

                          {/* Title — clickable to expand */}
                          <button
                            onClick={() => toggleSourceExpanded(msgIdx, j)}
                            className="text-[var(--text-secondary)] truncate flex-1 text-left hover:text-[var(--text-primary)] transition-colors"
                            title={source.text_snippet ? 'Click to show context' : source.title}
                          >
                            {source.title}
                          </button>

                          {/* Action buttons */}
                          <span className="flex items-center gap-1.5 flex-shrink-0 opacity-60 group-hover:opacity-100 transition-opacity">
                            {source.content_type === 'paper' && sanitizeUrl(source.url) && (
                              <a
                                href={sanitizeUrl(source.url)!}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-rose-500/15 text-rose-300 hover:bg-rose-500/25 transition-colors text-xs font-medium"
                              >
                                <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                  <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>
                                  <polyline points="15 3 21 3 21 9"/>
                                  <line x1="10" y1="14" x2="21" y2="3"/>
                                </svg>
                                Read Paper
                              </a>
                            )}
                            {source.content_type !== 'paper' && sanitizeUrl(source.url) && (
                              <a
                                href={sanitizeUrl(source.url)!}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-violet-500/10 text-violet-300 hover:bg-violet-500/20 transition-colors text-xs font-medium"
                              >
                                <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                  <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>
                                  <polyline points="15 3 21 3 21 9"/>
                                  <line x1="10" y1="14" x2="21" y2="3"/>
                                </svg>
                                Open
                              </a>
                            )}
                            {source.all_urls && source.all_urls.length > 1 && (
                              <span className="text-xs text-[var(--text-muted)]">
                                +{source.all_urls.length - 1} links
                              </span>
                            )}
                            <span className={`text-xs tabular-nums ${
                              source.score > 0.85 ? 'text-emerald-400' :
                              source.score > 0.7 ? 'text-[var(--text-secondary)]' :
                              'text-[var(--text-muted)]'
                            }`}>
                              {(source.score * 100).toFixed(0)}%
                            </span>
                          </span>
                        </div>

                        {/* Expandable source context */}
                        {isExpanded && source.text_snippet && (
                          <div className="mt-1.5 ml-[78px] p-2 rounded bg-[var(--surface)] border border-[var(--border)] text-[var(--text-muted)] text-[11px] leading-relaxed">
                            {source.text_snippet}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>

                {/* Explore topic chips — bidirectional bridge */}
                {onExploreInGraph && (() => {
                  const allTopics = Array.from(new Set(msg.sources!.flatMap(s => s.topics || []))).slice(0, 5);
                  return allTopics.length > 0 ? (
                    <div className="flex flex-wrap gap-1.5 mt-3 pt-2 border-t border-[var(--border)]">
                      <span className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider mr-1 self-center">Explore</span>
                      {allTopics.map(topic => (
                        <button
                          key={topic}
                          onClick={() => onExploreInGraph(topic.toLowerCase().replace(/\s+/g, '_'))}
                          className="text-[11px] px-2 py-0.5 rounded-full bg-cyan-500/10 text-cyan-300 hover:bg-cyan-500/20 transition-colors"
                        >
                          {topic}
                        </button>
                      ))}
                    </div>
                  ) : null;
                })()}
              </div>
            )}
          </div>
        ))}

        {/* Loading indicator */}
        {isLoading && (
          <div className="chat-message assistant rounded-xl px-5 py-4">
            <div className="text-xs uppercase tracking-wider font-medium mb-2 text-violet-400">
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
            onInput={(e) => {
              const el = e.currentTarget;
              el.style.height = 'auto';
              el.style.height = Math.min(el.scrollHeight, 120) + 'px';
            }}
            onKeyDown={handleKeyDown}
            placeholder="Ask about GenAI courses, techniques, research..."
            rows={1}
            className="flex-1 bg-transparent text-sm text-[var(--text-primary)] placeholder:text-[var(--text-muted)] px-4 py-3 resize-none focus:outline-none"
            style={{ maxHeight: '120px' }}
          />
          <button
            onClick={() => sendMessage()}
            disabled={!input.trim() || isLoading || isStreaming || cooldown}
            className="p-3 text-[var(--text-muted)] hover:text-violet-400 disabled:opacity-30 transition-colors"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/>
            </svg>
          </button>
        </div>
        <p className="text-xs text-[var(--text-muted)] text-center mt-2">
          Powered by RAG over awesome-generative-ai-guide · OpenAI gpt-4.1-mini · Responses may contain inaccuracies
        </p>
      </div>
    </div>
  );
}
