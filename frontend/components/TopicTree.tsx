'use client';

import { useEffect, useRef, useState, useCallback } from 'react';
import * as d3 from 'd3';

interface Props {
  onTopicClick: (topicId: string) => void;
}

interface TreeNode {
  name: string;
  id?: string;
  value?: number;
  category?: string;
  children?: TreeNode[];
}

const CATEGORY_PALETTES: Record<string, string[]> = {
  'Foundations':           ['#7c3aed', '#8b5cf6', '#a78bfa', '#c4b5fd'],
  'Techniques':           ['#0891b2', '#06b6d4', '#22d3ee', '#67e8f9'],
  'RAG & Retrieval':      ['#059669', '#10b981', '#34d399', '#6ee7b7'],
  'Agents':               ['#d97706', '#f59e0b', '#fbbf24', '#fcd34d'],
  'Frameworks & Tools':   ['#e11d48', '#f43f5e', '#fb7185', '#fda4af'],
  'Evaluation & Ops':     ['#7c2d12', '#c2410c', '#ea580c', '#fb923c'],
  'Models':               ['#4338ca', '#6366f1', '#818cf8', '#a5b4fc'],
  'Multimodal':           ['#be185d', '#db2777', '#ec4899', '#f472b6'],
  'Providers & Platforms': ['#065f46', '#047857', '#059669', '#10b981'],
};

export default function TopicTree({ onTopicClick }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });
  const renderTreemapRef = useRef<((hierarchy: TreeNode) => void) | null>(null);
  const hierarchyDataRef = useRef<TreeNode | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function loadAndRender() {
      try {
        const res = await fetch('/api/graph');
        if (!res.ok) throw new Error('Failed to load');
        const data = await res.json();
        if (cancelled) return;

        const hierarchy = data.hierarchy;
        if (!hierarchy || !hierarchy.children?.length) {
          setError('No hierarchy data available.');
          setLoading(false);
          return;
        }

        hierarchyDataRef.current = hierarchy;
        renderTreemapRef.current = renderTreemap;
        setLoading(false);
        renderTreemap(hierarchy);
      } catch {
        if (!cancelled) {
          setError('Could not load topic hierarchy.');
          setLoading(false);
        }
      }
    }

    function renderTreemap(hierarchy: TreeNode) {
      const container = containerRef.current;
      const svgEl = svgRef.current;
      if (!container || !svgEl) return;

      const width = container.clientWidth;
      const height = container.clientHeight;

      const svg = d3.select(svgEl);
      svg.attr('viewBox', `0 0 ${width} ${height}`);
      svg.selectAll('*').remove();

      // Build hierarchy
      const root = d3.hierarchy<TreeNode>(hierarchy)
        .sum(d => d.value || 1)
        .sort((a, b) => (b.value || 0) - (a.value || 0));

      const treemapRoot = d3.treemap<TreeNode>()
        .size([width, height])
        .paddingOuter(6)
        .paddingInner(3)
        .paddingTop(28)
        .round(true)(root);

      // Color scale per parent category
      const getColor = (d: d3.HierarchyRectangularNode<TreeNode>): string => {
        const parentName = d.parent?.data.name || d.data.name;
        const palette = CATEGORY_PALETTES[parentName] || ['#6366f1', '#818cf8', '#a5b4fc'];
        if (d.depth === 1) return palette[0];
        const siblings = d.parent?.children || [];
        const index = siblings.indexOf(d);
        return palette[Math.min(index, palette.length - 1)] || palette[palette.length - 1];
      };

      // Draw category groups (depth 1)
      const groups = svg.selectAll('g.category')
        .data(treemapRoot.children || [])
        .join('g')
        .attr('class', 'category');

      // Category background
      groups.append('rect')
        .attr('x', d => d.x0!)
        .attr('y', d => d.y0!)
        .attr('width', d => d.x1! - d.x0!)
        .attr('height', d => d.y1! - d.y0!)
        .attr('fill', d => getColor(d))
        .attr('fill-opacity', 0.08)
        .attr('rx', 8)
        .attr('stroke', d => getColor(d))
        .attr('stroke-opacity', 0.2)
        .attr('stroke-width', 1);

      // Category labels
      groups.append('text')
        .attr('x', d => d.x0! + 10)
        .attr('y', d => d.y0! + 18)
        .text(d => d.data.name)
        .style('font-family', "'DM Sans', sans-serif")
        .style('font-size', '11px')
        .style('font-weight', '600')
        .style('fill', d => getColor(d))
        .style('fill-opacity', 0.8)
        .style('text-transform', 'uppercase')
        .style('letter-spacing', '0.05em');

      // Draw leaf nodes (depth 2)
      const leaves = svg.selectAll('g.leaf')
        .data(treemapRoot.leaves())
        .join('g')
        .attr('class', 'leaf tree-node')
        .attr('tabindex', '0')
        .attr('role', 'button')
        .attr('aria-label', d => `${d.data.name}, ${d.data.value || 0} references`)
        .style('cursor', 'pointer')
        .on('click', (event, d) => {
          if (d.data.id) onTopicClick(d.data.id);
        })
        .on('keydown', (event, d) => {
          if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            if (d.data.id) onTopicClick(d.data.id);
          }
        })
        .on('mouseenter', (event, d) => {
          setHoveredNode(d.data.name);
          const rect = container!.getBoundingClientRect();
          setMousePos({ x: event.clientX - rect.left, y: event.clientY - rect.top });
        })
        .on('mousemove', (event) => {
          const rect = container!.getBoundingClientRect();
          setMousePos({ x: event.clientX - rect.left, y: event.clientY - rect.top });
        })
        .on('mouseleave', () => {
          setHoveredNode(null);
        });

      // Leaf rectangles
      leaves.append('rect')
        .attr('x', d => d.x0!)
        .attr('y', d => d.y0!)
        .attr('width', d => Math.max(0, d.x1! - d.x0!))
        .attr('height', d => Math.max(0, d.y1! - d.y0!))
        .attr('fill', d => getColor(d))
        .attr('fill-opacity', 0.6)
        .attr('rx', 4)
        .attr('stroke', d => getColor(d))
        .attr('stroke-opacity', 0.3)
        .attr('stroke-width', 0.5)
        .on('mouseenter', function() {
          d3.select(this).attr('fill-opacity', 0.85).attr('stroke-opacity', 0.6);
        })
        .on('mouseleave', function() {
          d3.select(this).attr('fill-opacity', 0.6).attr('stroke-opacity', 0.3);
        });

      // Leaf labels (only if cell is large enough)
      leaves.append('text')
        .attr('x', d => d.x0! + 6)
        .attr('y', d => d.y0! + 16)
        .text(d => d.data.name)
        .style('font-family', "'DM Sans', sans-serif")
        .style('font-size', d => {
          const w = d.x1! - d.x0!;
          return w > 100 ? '11px' : w > 60 ? '9px' : '8px';
        })
        .style('font-weight', '500')
        .style('fill', 'white')
        .style('fill-opacity', 0.9)
        .style('pointer-events', 'none')
        .each(function(d) {
          const cellWidth = d.x1! - d.x0! - 12;
          const textEl = d3.select(this);
          // Truncate if too wide
          if ((this as SVGTextElement).getComputedTextLength() > cellWidth) {
            let text = d.data.name;
            while (text.length > 3 && (this as SVGTextElement).getComputedTextLength() > cellWidth) {
              text = text.slice(0, -1);
              textEl.text(text + '…');
            }
          }
        });

      // Value labels (smaller, below name)
      leaves.append('text')
        .attr('x', d => d.x0! + 6)
        .attr('y', d => d.y0! + 30)
        .text(d => {
          const w = d.x1! - d.x0!;
          const h = d.y1! - d.y0!;
          return (w > 60 && h > 40) ? `${d.data.value || 0} refs` : '';
        })
        .style('font-family', "'DM Sans', sans-serif")
        .style('font-size', '9px')
        .style('fill', 'white')
        .style('fill-opacity', 0.5)
        .style('pointer-events', 'none');
    }

    loadAndRender();
    return () => { cancelled = true; };
  }, [onTopicClick]);

  // Re-render on container resize
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const observer = new ResizeObserver(() => {
      if (renderTreemapRef.current && hierarchyDataRef.current) {
        renderTreemapRef.current(hierarchyDataRef.current);
      }
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  return (
    <div ref={containerRef} className="w-full h-full relative bg-[var(--midnight)]">
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-center animate-fade-in">
            <div className="w-12 h-12 rounded-xl bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center mx-auto mb-4 animate-pulse">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--cyan)" strokeWidth="2">
                <rect x="3" y="3" width="18" height="18" rx="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="9" y1="21" x2="9" y2="9"/>
              </svg>
            </div>
            <p className="text-sm text-[var(--text-muted)]">Loading topic hierarchy...</p>
          </div>
        </div>
      )}
      {error && (
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-center max-w-sm animate-fade-in">
            <p className="text-sm text-red-400 mb-2">{error}</p>
            <p className="text-xs text-[var(--text-muted)]">
              Make sure the API is running and data has been ingested.
            </p>
          </div>
        </div>
      )}
      <svg ref={svgRef} className="w-full h-full" role="img" aria-label="Treemap showing GenAI topic categories and their relative sizes" />

      {/* Tooltip */}
      {hoveredNode && (
        <div
          className="absolute bg-[var(--surface)] border border-[var(--border)] rounded-lg px-3 py-2 text-xs text-[var(--text-secondary)] pointer-events-none z-10 shadow-lg"
          style={{ left: mousePos.x + 12, top: mousePos.y - 8 }}
        >
          {hoveredNode}
        </div>
      )}
    </div>
  );
}
