'use client';

import { useEffect, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkGemoji from 'remark-gemoji';
import { nameToEmoji } from 'gemoji';

/** Common non-standard shortcode aliases */
const EMOJI_ALIASES: Record<string, string> = {
  mortarboard: 'mortar_board',
  notebook: 'notebook',
};

/** Replace :shortcode: with actual emoji in plain strings */
function emojify(text: string): string {
  return text.replace(/:([a-z0-9_+-]+):/g, (match, name) => {
    const resolved = EMOJI_ALIASES[name] || name;
    return nameToEmoji[resolved] ?? match;
  });
}

interface TopicData {
  topic: {
    id: string;
    label: string;
    category: string;
    weight: number;
  };
  connected_topics: Array<{
    id: string;
    label: string;
    weight: number;
  }>;
  resources: Array<{
    title: string;
    source_file: string;
    category: string;
    url: string;
    text_preview: string;
    content_type: string;
    all_urls: string[];
  }>;
}

interface Props {
  topicId: string;
  onClose: () => void;
  onExploreInChat: (topicLabel: string) => void;
}

const CATEGORY_BADGES: Record<string, { bg: string; text: string; label: string }> = {
  concept: { bg: 'bg-violet-500/15', text: 'text-violet-300', label: 'Concept' },
  technique: { bg: 'bg-cyan-500/15', text: 'text-cyan-300', label: 'Technique' },
  tool: { bg: 'bg-emerald-500/15', text: 'text-emerald-300', label: 'Tool' },
  provider: { bg: 'bg-orange-500/15', text: 'text-orange-300', label: 'Provider' },
};

const FILE_CATEGORY_COLORS: Record<string, string> = {
  free_courses: 'text-emerald-400',
  resources: 'text-violet-400',
  interview_prep: 'text-amber-400',
  research_updates: 'text-cyan-400',
};

const FILE_CATEGORY_LABELS: Record<string, string> = {
  free_courses: 'Course',
  resources: 'Resource',
  interview_prep: 'Interview Prep',
  research_updates: 'Research',
};

export default function TopicDetail({ topicId, onClose, onExploreInChat }: Props) {
  const [data, setData] = useState<TopicData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);

    fetch(`/api/topics/${topicId}`)
      .then(r => r.json())
      .then(d => {
        if (!cancelled) {
          setData(d.error ? null : d);
          setLoading(false);
        }
      })
      .catch(() => {
        if (!cancelled) setLoading(false);
      });

    return () => { cancelled = true; };
  }, [topicId]);

  const badge = data ? CATEGORY_BADGES[data.topic.category] || CATEGORY_BADGES.concept : null;

  return (
    <div className="detail-panel w-[380px] h-full overflow-y-auto flex-shrink-0">
      {/* Header */}
      <div className="sticky top-0 bg-[var(--surface)] border-b border-[var(--border)] px-5 py-4 z-10">
        <div className="flex items-start justify-between">
          <div className="flex-1 min-w-0">
            {loading ? (
              <div className="skeleton h-6 w-40 mb-2" />
            ) : data ? (
              <>
                <h3 className="font-display text-lg text-[var(--text-primary)] truncate">
                  {data.topic.label}
                </h3>
                <div className="flex items-center gap-2 mt-1">
                  {badge && (
                    <span className={`${badge.bg} ${badge.text} text-[10px] font-medium px-2 py-0.5 rounded-full uppercase tracking-wider`}>
                      {badge.label}
                    </span>
                  )}
                  <span className="text-[10px] text-[var(--text-muted)]">
                    {data.topic.weight} references
                  </span>
                </div>
              </>
            ) : (
              <p className="text-sm text-[var(--text-muted)]">Topic not found</p>
            )}
          </div>
          <button
            onClick={onClose}
            className="p-1 text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
            </svg>
          </button>
        </div>

        {/* Action buttons */}
        {data && (
          <div className="flex gap-2 mt-3">
            <button
              onClick={() => onExploreInChat(data.topic.label)}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-violet-500/15 text-violet-300 hover:bg-violet-500/25 transition-colors"
            >
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
              </svg>
              Ask in Chat
            </button>
          </div>
        )}
      </div>

      {loading && (
        <div className="p-5 space-y-3">
          <div className="skeleton h-4 w-24" />
          <div className="skeleton h-10 w-full" />
          <div className="skeleton h-10 w-full" />
          <div className="skeleton h-10 w-full" />
        </div>
      )}

      {data && (
        <div className="p-5 space-y-6">
          {/* Connected Topics */}
          {data.connected_topics.length > 0 && (
            <div>
              <h4 className="text-[10px] uppercase tracking-wider text-[var(--text-muted)] font-medium mb-3">
                Connected Topics
              </h4>
              <div className="flex flex-wrap gap-1.5">
                {data.connected_topics.map(ct => (
                  <button
                    key={ct.id}
                    onClick={() => {
                      // This would navigate to the topic in the graph
                      // For now, we re-select
                      const event = new CustomEvent('selectTopic', { detail: ct.id });
                      window.dispatchEvent(event);
                    }}
                    className="text-[11px] px-2.5 py-1 rounded-full bg-[var(--midnight)] border border-[var(--border)] text-[var(--text-secondary)] hover:border-violet-500/40 hover:text-violet-300 transition-all"
                  >
                    {ct.label}
                    <span className="text-[var(--text-muted)] ml-1">({ct.weight})</span>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Resources */}
          {data.resources.length > 0 && (
            <div>
              <h4 className="text-[10px] uppercase tracking-wider text-[var(--text-muted)] font-medium mb-3">
                Resources ({data.resources.length})
              </h4>
              <div className="space-y-2">
                {data.resources.map((resource, i) => (
                  <div
                    key={i}
                    className="rounded-lg bg-[var(--midnight)] border border-[var(--border)] p-3 hover:border-[var(--border-bright)] transition-colors"
                  >
                    <div className="flex items-center gap-2 mb-1.5">
                      <span className={`text-[10px] font-medium ${
                        resource.content_type === 'paper'
                          ? 'text-rose-400'
                          : FILE_CATEGORY_COLORS[resource.category] || 'text-slate-400'
                      }`}>
                        {resource.content_type === 'paper'
                          ? '📄 Paper'
                          : FILE_CATEGORY_LABELS[resource.category] || resource.category}
                      </span>
                    </div>
                    {resource.url ? (
                      <a
                        href={resource.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-sm text-[var(--text-primary)] hover:text-violet-400 transition-colors font-medium leading-tight block"
                      >
                        {emojify(resource.title)}
                      </a>
                    ) : (
                      <p className="text-sm text-[var(--text-primary)] font-medium leading-tight">
                        {emojify(resource.title)}
                      </p>
                    )}
                    {resource.text_preview && (
                      <div className="prose-resource text-[11px] text-[var(--text-muted)] mt-1.5 leading-relaxed line-clamp-3">
                        <ReactMarkdown
                          remarkPlugins={[remarkGfm, remarkGemoji]}
                          components={{
                            a: ({ href, children }) => (
                              <a href={href} target="_blank" rel="noopener noreferrer">{children}</a>
                            ),
                            p: ({ children }) => <span>{children} </span>,
                          }}
                        >
                          {resource.text_preview}
                        </ReactMarkdown>
                      </div>
                    )}
                    {/* Action row: file path + paper link */}
                    <div className="flex items-center justify-between mt-2">
                      <p className="text-[10px] text-[var(--text-muted)] font-mono truncate flex-1">
                        {resource.source_file}
                      </p>
                      {resource.content_type === 'paper' && resource.url && (
                        <a
                          href={resource.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="flex-shrink-0 ml-2 inline-flex items-center gap-1 px-2 py-0.5 rounded bg-rose-500/15 text-rose-300 hover:bg-rose-500/25 transition-colors text-[10px] font-medium"
                        >
                          <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>
                            <polyline points="15 3 21 3 21 9"/>
                            <line x1="10" y1="14" x2="21" y2="3"/>
                          </svg>
                          Read Paper
                        </a>
                      )}
                    </div>
                    {/* Additional links */}
                    {resource.all_urls && resource.all_urls.length > 1 && (
                      <div className="flex flex-wrap gap-1.5 mt-1.5">
                        {resource.all_urls.filter(u => u !== resource.url).map((u, j) => {
                          let label = u;
                          try { label = new URL(u).hostname.replace('www.', ''); } catch {}
                          return (
                            <a
                              key={j}
                              href={u}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-violet-500/10 text-[10px] text-violet-300 hover:bg-violet-500/20 transition-colors truncate max-w-[200px]"
                            >
                              <svg width="8" height="8" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                                <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>
                                <polyline points="15 3 21 3 21 9"/>
                                <line x1="10" y1="14" x2="21" y2="3"/>
                              </svg>
                              {label}
                            </a>
                          );
                        })}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
