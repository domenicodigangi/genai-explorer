'use client';

import { useState, useCallback } from 'react';
import Chat from '@/components/Chat';
import TopicGraph from '@/components/TopicGraph';
import TopicTree from '@/components/TopicTree';
import TopicDetail from '@/components/TopicDetail';

type Tab = 'chat' | 'explore';
type ExploreView = 'graph' | 'tree';

export default function Home() {
  const [tab, setTab] = useState<Tab>('explore');
  const [exploreView, setExploreView] = useState<ExploreView>('graph');
  const [selectedTopic, setSelectedTopic] = useState<string | null>(null);
  const [searchFromExplore, setSearchFromExplore] = useState<string>('');

  const handleTopicClick = useCallback((topicId: string) => {
    setSelectedTopic(topicId);
  }, []);

  const handleExploreInChat = useCallback((topicLabel: string) => {
    setSearchFromExplore(topicLabel);
    setTab('chat');
  }, []);

  return (
    <div className="h-screen flex flex-col overflow-hidden">
      {/* ---- Header ---- */}
      <header className="flex-shrink-0 border-b border-[var(--border)] bg-[var(--abyss)]">
        <div className="flex items-center justify-between px-6 py-3">
          {/* Logo */}
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-violet-500 to-cyan-500 flex items-center justify-center">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="3"/>
                <path d="M12 1v4M12 19v4M4.2 4.2l2.8 2.8M17 17l2.8 2.8M1 12h4M19 12h4M4.2 19.8l2.8-2.8M17 7l2.8-2.8"/>
              </svg>
            </div>
            <div>
              <h1 className="font-display text-xl text-[var(--text-primary)] leading-none">
                GenAI Knowledge Explorer
              </h1>
              <p className="text-[11px] text-[var(--text-muted)] mt-0.5 tracking-wide">
                Navigate the generative AI landscape
              </p>
            </div>
          </div>

          {/* Tab switcher */}
          <div className="flex items-center gap-1 bg-[var(--midnight)] rounded-lg p-1">
            <button
              onClick={() => setTab('chat')}
              className={`flex items-center gap-2 px-4 py-1.5 rounded-md text-sm font-medium transition-all ${
                tab === 'chat'
                  ? 'bg-[var(--surface)] text-[var(--text-primary)] shadow-sm'
                  : 'text-[var(--text-muted)] hover:text-[var(--text-secondary)]'
              }`}
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
              </svg>
              Chat
            </button>
            <button
              onClick={() => setTab('explore')}
              className={`flex items-center gap-2 px-4 py-1.5 rounded-md text-sm font-medium transition-all ${
                tab === 'explore'
                  ? 'bg-[var(--surface)] text-[var(--text-primary)] shadow-sm'
                  : 'text-[var(--text-muted)] hover:text-[var(--text-secondary)]'
              }`}
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="10"/>
                <polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"/>
              </svg>
              Explore
            </button>
          </div>

          {/* Source link */}
          <a
            href="https://github.com/aishwaryanr/awesome-generative-ai-guide"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 text-xs text-[var(--text-muted)] hover:text-[var(--text-secondary)] transition-colors"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z"/>
            </svg>
            Source Repo
          </a>
        </div>
      </header>

      {/* ---- Main Content ---- */}
      <main className="flex-1 overflow-hidden relative">
        {/* Chat Mode */}
        {tab === 'chat' && (
          <Chat initialQuery={searchFromExplore} onQueryConsumed={() => setSearchFromExplore('')} />
        )}

        {/* Explore Mode */}
        {tab === 'explore' && (
          <div className="h-full flex">
            {/* Visualization area */}
            <div className="flex-1 flex flex-col overflow-hidden">
              {/* Sub-tab bar */}
              <div className="flex items-center gap-4 px-6 py-2 border-b border-[var(--border)] bg-[var(--abyss)]">
                <button
                  onClick={() => setExploreView('graph')}
                  className={`flex items-center gap-1.5 px-3 py-1 rounded-md text-xs font-medium transition-all ${
                    exploreView === 'graph'
                      ? 'bg-violet-500/15 text-violet-300'
                      : 'text-[var(--text-muted)] hover:text-[var(--text-secondary)]'
                  }`}
                >
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <circle cx="6" cy="6" r="3"/><circle cx="18" cy="18" r="3"/><circle cx="18" cy="6" r="3"/>
                    <line x1="8.5" y1="7.5" x2="15.5" y2="16.5"/><line x1="15.5" y1="7.5" x2="8.5" y2="16.5"/>
                  </svg>
                  Knowledge Graph
                </button>
                <button
                  onClick={() => setExploreView('tree')}
                  className={`flex items-center gap-1.5 px-3 py-1 rounded-md text-xs font-medium transition-all ${
                    exploreView === 'tree'
                      ? 'bg-cyan-500/15 text-cyan-300'
                      : 'text-[var(--text-muted)] hover:text-[var(--text-secondary)]'
                  }`}
                >
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <rect x="3" y="3" width="18" height="18" rx="2"/><line x1="3" y1="9" x2="21" y2="9"/>
                    <line x1="9" y1="21" x2="9" y2="9"/>
                  </svg>
                  Topic Treemap
                </button>
                <div className="flex-1" />
                <span className="text-[10px] text-[var(--text-muted)] tracking-wider uppercase">
                  Click a topic to explore
                </span>
              </div>

              {/* Graph or Tree */}
              <div className="flex-1 overflow-hidden">
                {exploreView === 'graph' && (
                  <TopicGraph onTopicClick={handleTopicClick} selectedTopic={selectedTopic} />
                )}
                {exploreView === 'tree' && (
                  <TopicTree onTopicClick={handleTopicClick} />
                )}
              </div>
            </div>

            {/* Detail panel */}
            {selectedTopic && (
              <TopicDetail
                topicId={selectedTopic}
                onClose={() => setSelectedTopic(null)}
                onExploreInChat={handleExploreInChat}
              />
            )}
          </div>
        )}
      </main>
    </div>
  );
}
