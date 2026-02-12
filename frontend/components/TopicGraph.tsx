'use client';

import { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';

interface GraphNode {
  id: string;
  label: string;
  category: string;
  weight: number;
  // d3 simulation fields
  x?: number;
  y?: number;
  fx?: number | null;
  fy?: number | null;
}

interface GraphEdge {
  source: string | GraphNode;
  target: string | GraphNode;
  weight: number;
}

interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

interface Props {
  onTopicClick: (topicId: string) => void;
  selectedTopic: string | null;
}

const CATEGORY_COLORS: Record<string, string> = {
  concept: '#8b5cf6',    // violet
  technique: '#06b6d4',  // cyan
  tool: '#10b981',       // emerald
  provider: '#f97316',   // orange
};

const CATEGORY_GLOWS: Record<string, string> = {
  concept: 'rgba(139, 92, 246, 0.4)',
  technique: 'rgba(6, 182, 212, 0.4)',
  tool: 'rgba(16, 185, 129, 0.4)',
  provider: 'rgba(249, 115, 22, 0.4)',
};

export default function TopicGraph({ onTopicClick, selectedTopic }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const simulationRef = useRef<d3.Simulation<GraphNode, GraphEdge> | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function loadAndRender() {
      try {
        const res = await fetch('/api/graph');
        if (!res.ok) throw new Error('Failed to load graph');
        const data: GraphData = await res.json();

        if (cancelled) return;
        if (!data.nodes || data.nodes.length === 0) {
          setError('No topic data available. Run the ingestion pipeline first.');
          setLoading(false);
          return;
        }

        setLoading(false);
        renderGraph(data);
      } catch (err) {
        if (!cancelled) {
          setError('Could not load topic graph. Is the API running?');
          setLoading(false);
        }
      }
    }

    function renderGraph(data: GraphData) {
      const svg = d3.select(svgRef.current);
      const container = containerRef.current;
      if (!container) return;

      const width = container.clientWidth;
      const height = container.clientHeight;

      svg.attr('viewBox', `0 0 ${width} ${height}`);
      svg.selectAll('*').remove();

      // Defs for glow filters
      const defs = svg.append('defs');
      Object.entries(CATEGORY_GLOWS).forEach(([cat, color]) => {
        const filter = defs.append('filter').attr('id', `glow-${cat}`);
        filter.append('feGaussianBlur').attr('stdDeviation', '4').attr('result', 'blur');
        filter.append('feFlood').attr('flood-color', color).attr('result', 'color');
        filter.append('feComposite').attr('in', 'color').attr('in2', 'blur').attr('operator', 'in').attr('result', 'glow');
        const merge = filter.append('feMerge');
        merge.append('feMergeNode').attr('in', 'glow');
        merge.append('feMergeNode').attr('in', 'SourceGraphic');
      });

      const g = svg.append('g');

      // Zoom behavior
      const zoom = d3.zoom<SVGSVGElement, unknown>()
        .scaleExtent([0.3, 4])
        .on('zoom', (event) => {
          g.attr('transform', event.transform);
        });
      svg.call(zoom);

      // Scale node size by weight
      const maxWeight = d3.max(data.nodes, d => d.weight) || 1;
      const nodeScale = d3.scaleSqrt().domain([1, maxWeight]).range([6, 28]);

      // Force simulation
      const simulation = d3.forceSimulation<GraphNode>(data.nodes)
        .force('link', d3.forceLink<GraphNode, GraphEdge>(data.edges)
          .id(d => d.id)
          .distance(d => 100 - Math.min(d.weight * 5, 60))
          .strength(d => Math.min(d.weight * 0.08, 0.5))
        )
        .force('charge', d3.forceManyBody().strength(d => -nodeScale((d as GraphNode).weight) * 8))
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('collision', d3.forceCollide().radius(d => nodeScale((d as GraphNode).weight) + 8))
        .force('x', d3.forceX(width / 2).strength(0.03))
        .force('y', d3.forceY(height / 2).strength(0.03));

      simulationRef.current = simulation;

      // Draw edges
      const links = g.append('g')
        .selectAll('line')
        .data(data.edges)
        .join('line')
        .attr('class', 'graph-link')
        .attr('stroke-width', d => Math.max(0.5, Math.min(d.weight * 0.3, 3)));

      // Draw nodes
      const nodes = g.append('g')
        .selectAll('g')
        .data(data.nodes)
        .join('g')
        .attr('class', 'graph-node')
        .on('click', (event, d) => {
          event.stopPropagation();
          onTopicClick(d.id);
        })
        .call(d3.drag<SVGGElement, GraphNode>()
          .on('start', (event, d) => {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
          })
          .on('drag', (event, d) => {
            d.fx = event.x;
            d.fy = event.y;
          })
          .on('end', (event, d) => {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
          })
        );

      // Node circles
      nodes.append('circle')
        .attr('r', d => nodeScale(d.weight))
        .attr('fill', d => CATEGORY_COLORS[d.category] || '#8b5cf6')
        .attr('fill-opacity', 0.7)
        .attr('stroke', d => CATEGORY_COLORS[d.category] || '#8b5cf6')
        .attr('stroke-width', 1.5)
        .attr('filter', d => `url(#glow-${d.category})`);

      // Node labels
      nodes.append('text')
        .attr('class', 'graph-label')
        .attr('dy', d => nodeScale(d.weight) + 14)
        .text(d => d.label)
        .style('font-size', d => d.weight > maxWeight * 0.5 ? '12px' : '10px')
        .style('fill', d => d.weight > maxWeight * 0.3 ? '#e2e8f0' : '#94a3b8');

      // Hover effects
      nodes.on('mouseenter', function(event, d) {
        // Highlight connected edges
        links.attr('class', l => {
          const src = typeof l.source === 'string' ? l.source : l.source.id;
          const tgt = typeof l.target === 'string' ? l.target : l.target.id;
          return (src === d.id || tgt === d.id) ? 'graph-link highlighted' : 'graph-link';
        });
        // Dim unconnected nodes
        const connectedIds = new Set<string>();
        connectedIds.add(d.id);
        data.edges.forEach(l => {
          const src = typeof l.source === 'string' ? l.source : l.source.id;
          const tgt = typeof l.target === 'string' ? l.target : l.target.id;
          if (src === d.id) connectedIds.add(tgt);
          if (tgt === d.id) connectedIds.add(src);
        });
        nodes.select('circle').attr('fill-opacity', n => connectedIds.has(n.id) ? 0.9 : 0.2);
        nodes.select('text').style('opacity', n => connectedIds.has(n.id) ? 1 : 0.2);
      })
      .on('mouseleave', function() {
        links.attr('class', 'graph-link');
        nodes.select('circle').attr('fill-opacity', 0.7);
        nodes.select('text').style('opacity', 1);
      });

      // Simulation tick
      simulation.on('tick', () => {
        links
          .attr('x1', d => (d.source as GraphNode).x!)
          .attr('y1', d => (d.source as GraphNode).y!)
          .attr('x2', d => (d.target as GraphNode).x!)
          .attr('y2', d => (d.target as GraphNode).y!);

        nodes.attr('transform', d => `translate(${d.x},${d.y})`);
      });

      // Initial zoom to fit
      setTimeout(() => {
        svg.transition().duration(800).call(
          zoom.transform,
          d3.zoomIdentity.translate(width * 0.1, height * 0.1).scale(0.8)
        );
      }, 1500);
    }

    loadAndRender();
    return () => { cancelled = true; simulationRef.current?.stop(); };
  }, [onTopicClick]);

  // Highlight selected topic
  useEffect(() => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll('.graph-node circle')
      .attr('stroke-width', (d: any) => d.id === selectedTopic ? 3 : 1.5)
      .attr('stroke', (d: any) =>
        d.id === selectedTopic ? '#ffffff' : (CATEGORY_COLORS[d.category] || '#8b5cf6')
      );
  }, [selectedTopic]);

  return (
    <div ref={containerRef} className="w-full h-full graph-container relative bg-[var(--midnight)]">
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-center animate-fade-in">
            <div className="w-12 h-12 rounded-xl bg-violet-500/10 border border-violet-500/20 flex items-center justify-center mx-auto mb-4 animate-pulse-glow">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--nebula)" strokeWidth="2">
                <circle cx="12" cy="12" r="10"/><polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88"/>
              </svg>
            </div>
            <p className="text-sm text-[var(--text-muted)]">Loading knowledge graph...</p>
          </div>
        </div>
      )}
      {error && (
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-center max-w-sm animate-fade-in">
            <p className="text-sm text-red-400 mb-2">{error}</p>
            <p className="text-xs text-[var(--text-muted)]">
              Run <code className="text-violet-400">python scripts/ingest.py</code> then start the API
            </p>
          </div>
        </div>
      )}
      <svg ref={svgRef} className="w-full h-full" />

      {/* Legend */}
      {!loading && !error && (
        <div className="absolute bottom-4 left-4 flex items-center gap-4 bg-[var(--abyss)]/80 backdrop-blur-sm rounded-lg px-4 py-2 border border-[var(--border)]">
          {Object.entries(CATEGORY_COLORS).map(([cat, color]) => (
            <div key={cat} className="flex items-center gap-1.5">
              <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: color }} />
              <span className="text-[10px] text-[var(--text-muted)] capitalize">{cat}</span>
            </div>
          ))}
          <span className="text-[10px] text-[var(--text-muted)] ml-2">Scroll to zoom · Drag to pan</span>
        </div>
      )}
    </div>
  );
}
