'use client';

import { useEffect, useRef, useState, useCallback } from 'react';
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

// Category cluster positions (normalized 0-1, mapped to width/height)
const CATEGORY_POSITIONS: Record<string, { x: number; y: number }> = {
  concept: { x: 0.3, y: 0.3 },     // top-left
  technique: { x: 0.7, y: 0.3 },   // top-right
  tool: { x: 0.3, y: 0.7 },        // bottom-left
  provider: { x: 0.7, y: 0.7 },    // bottom-right
};

const DEFAULT_EDGE_THRESHOLD = 10;
const MIN_THRESHOLD = 2;
const MAX_THRESHOLD = 30;
// Nodes with weight below this fraction of max get labels hidden (shown on hover)
const LABEL_WEIGHT_FRACTION = 0.15;

export default function TopicGraph({ onTopicClick, selectedTopic }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [edgeThreshold, setEdgeThreshold] = useState(DEFAULT_EDGE_THRESHOLD);
  const simulationRef = useRef<d3.Simulation<GraphNode, GraphEdge> | null>(null);
  const graphDataRef = useRef<GraphData | null>(null);
  const renderRef = useRef<((threshold: number) => void) | null>(null);

  // Rebuild visible edges when threshold changes
  useEffect(() => {
    if (renderRef.current) {
      renderRef.current(edgeThreshold);
    }
  }, [edgeThreshold]);

  const onTopicClickRef = useRef(onTopicClick);
  onTopicClickRef.current = onTopicClick;

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

        graphDataRef.current = data;
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
      const nodeScale = d3.scaleSqrt().domain([1, maxWeight]).range([6, 30]);
      const labelThreshold = maxWeight * LABEL_WEIGHT_FRACTION;

      // Filter edges for simulation — use all edges for layout but only show strong ones
      const filteredEdges = data.edges.filter(e => e.weight >= DEFAULT_EDGE_THRESHOLD);

      // Edge weight scale for opacity
      const maxEdgeWeight = d3.max(data.edges, d => d.weight) || 1;
      const edgeOpacity = d3.scaleLog().domain([2, maxEdgeWeight]).range([0.15, 0.6]).clamp(true);

      // Force simulation with category clustering
      const simulation = d3.forceSimulation<GraphNode>(data.nodes)
        .force('link', d3.forceLink<GraphNode, GraphEdge>(filteredEdges)
          .id(d => d.id)
          .distance(d => 150 - Math.min(d.weight * 2, 80))
          .strength(d => Math.min(d.weight * 0.03, 0.3))
        )
        .force('charge', d3.forceManyBody().strength(d => -nodeScale((d as GraphNode).weight) * 18))
        .force('collision', d3.forceCollide().radius(d => nodeScale((d as GraphNode).weight) + 18))
        // Category clustering: pull toward quadrant positions
        .force('x', d3.forceX<GraphNode>().x(d => {
          const pos = CATEGORY_POSITIONS[d.category] || { x: 0.5 };
          return pos.x * width;
        }).strength(0.12))
        .force('y', d3.forceY<GraphNode>().y(d => {
          const pos = CATEGORY_POSITIONS[d.category] || { y: 0.5 };
          return pos.y * height;
        }).strength(0.12))
        .alphaDecay(0.04);  // settle faster

      simulationRef.current = simulation;

      // Draw edges group
      const linkGroup = g.append('g');

      // Draw all edges but control visibility via threshold
      const links = linkGroup
        .selectAll('line')
        .data(data.edges)
        .join('line')
        .attr('class', 'graph-link')
        .attr('stroke-width', d => Math.max(0.5, Math.min(d.weight * 0.15, 2)))
        .attr('stroke-opacity', d => d.weight >= DEFAULT_EDGE_THRESHOLD ? edgeOpacity(d.weight) : 0)
        .style('display', d => d.weight >= DEFAULT_EDGE_THRESHOLD ? null : 'none');

      // Expose threshold update function
      renderRef.current = (threshold: number) => {
        links
          .attr('stroke-opacity', d => d.weight >= threshold ? edgeOpacity(d.weight) : 0)
          .style('display', d => d.weight >= threshold ? null : 'none');
      };

      // Draw nodes
      const nodes = g.append('g')
        .selectAll('g')
        .data(data.nodes)
        .join('g')
        .attr('class', 'graph-node')
        .style('cursor', 'pointer')
        .on('click', (event, d) => {
          event.stopPropagation();
          onTopicClickRef.current(d.id);
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

      // Node labels — only show for prominent nodes
      nodes.append('text')
        .attr('class', 'graph-label')
        .attr('dy', d => nodeScale(d.weight) + 14)
        .text(d => d.label)
        .style('font-size', d => d.weight > maxWeight * 0.5 ? '12px' : '10px')
        .style('fill', d => d.weight > maxWeight * 0.3 ? '#e2e8f0' : '#94a3b8')
        .style('opacity', d => d.weight >= labelThreshold ? 1 : 0);

      // Hover effects
      nodes.on('mouseenter', function(event, d) {
        // Show this node's label regardless of weight
        d3.select(this).select('text').style('opacity', 1);

        // Highlight connected edges
        links.attr('class', l => {
          const src = typeof l.source === 'string' ? l.source : l.source.id;
          const tgt = typeof l.target === 'string' ? l.target : l.target.id;
          return (src === d.id || tgt === d.id) ? 'graph-link highlighted' : 'graph-link';
        });

        // Show all connected edges (even below threshold) on hover
        const connectedIds = new Set<string>();
        connectedIds.add(d.id);
        data.edges.forEach(l => {
          const src = typeof l.source === 'string' ? l.source : l.source.id;
          const tgt = typeof l.target === 'string' ? l.target : l.target.id;
          if (src === d.id) connectedIds.add(tgt);
          if (tgt === d.id) connectedIds.add(src);
        });

        links
          .style('display', l => {
            const src = typeof l.source === 'string' ? l.source : l.source.id;
            const tgt = typeof l.target === 'string' ? l.target : l.target.id;
            return (src === d.id || tgt === d.id) ? null : (l.weight >= edgeThreshold ? null : 'none');
          })
          .attr('stroke-opacity', l => {
            const src = typeof l.source === 'string' ? l.source : l.source.id;
            const tgt = typeof l.target === 'string' ? l.target : l.target.id;
            return (src === d.id || tgt === d.id) ? 0.7 : (l.weight >= edgeThreshold ? edgeOpacity(l.weight) * 0.3 : 0);
          });

        nodes.select('circle').attr('fill-opacity', n => connectedIds.has(n.id) ? 0.9 : 0.15);
        nodes.select('text').style('opacity', n => {
          if (n.id === d.id) return 1;
          if (connectedIds.has(n.id)) return 1;
          return 0.1;
        });
      })
      .on('mouseleave', function(event, d) {
        links.attr('class', 'graph-link');
        renderRef.current?.(edgeThreshold); // restore threshold-based visibility
        nodes.select('circle').attr('fill-opacity', 0.7);
        nodes.select('text').style('opacity', n => n.weight >= labelThreshold ? 1 : 0);
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

      // Fit to view once simulation settles
      simulation.on('end', () => {
        const nodePositions = data.nodes.map(n => ({ x: n.x || 0, y: n.y || 0 }));
        const xExtent = d3.extent(nodePositions, d => d.x) as [number, number];
        const yExtent = d3.extent(nodePositions, d => d.y) as [number, number];
        const graphWidth = xExtent[1] - xExtent[0] + 80;
        const graphHeight = yExtent[1] - yExtent[0] + 80;
        const scale = Math.min(width / graphWidth, height / graphHeight, 1) * 0.85;
        const cx = (xExtent[0] + xExtent[1]) / 2;
        const cy = (yExtent[0] + yExtent[1]) / 2;

        svg.transition().duration(600).call(
          zoom.transform,
          d3.zoomIdentity
            .translate(width / 2, height / 2)
            .scale(scale)
            .translate(-cx, -cy)
        );
      });
    }

    loadAndRender();
    return () => { cancelled = true; simulationRef.current?.stop(); };
  }, []);

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

  const visibleEdgeCount = graphDataRef.current
    ? graphDataRef.current.edges.filter(e => e.weight >= edgeThreshold).length
    : 0;

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

      {/* Controls: Legend + Edge threshold slider */}
      {!loading && !error && (
        <>
          {/* Legend */}
          <div className="absolute bottom-4 left-4 flex items-center gap-4 bg-[var(--abyss)]/80 backdrop-blur-sm rounded-lg px-4 py-2 border border-[var(--border)]">
            {Object.entries(CATEGORY_COLORS).map(([cat, color]) => (
              <div key={cat} className="flex items-center gap-1.5">
                <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: color }} />
                <span className="text-[10px] text-[var(--text-muted)] capitalize">{cat}</span>
              </div>
            ))}
            <span className="text-[10px] text-[var(--text-muted)] ml-2">Scroll to zoom · Drag to pan</span>
          </div>

          {/* Edge threshold slider */}
          <div className="absolute bottom-4 right-4 flex items-center gap-3 bg-[var(--abyss)]/80 backdrop-blur-sm rounded-lg px-4 py-2 border border-[var(--border)]">
            <span className="text-[10px] text-[var(--text-muted)] whitespace-nowrap">Connections</span>
            <input
              type="range"
              min={MIN_THRESHOLD}
              max={MAX_THRESHOLD}
              value={edgeThreshold}
              onChange={(e) => setEdgeThreshold(Number(e.target.value))}
              className="w-20 h-1 accent-violet-500 cursor-pointer"
            />
            <span className="text-[10px] text-[var(--text-secondary)] tabular-nums w-6 text-right">
              {visibleEdgeCount}
            </span>
          </div>
        </>
      )}
    </div>
  );
}
