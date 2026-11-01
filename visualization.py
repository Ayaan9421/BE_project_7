from typing import List, Set
import pandas as pd
from pyvis.network import Network


def create_subgraph_network_html(
    subgraph_df: pd.DataFrame,
    seed_names: List[str] = None,
    max_triples: int = 75,
) -> str:
    """
    Generate an interactive PyVis HTML network visualization for a retrieved subgraph.
    """
    if subgraph_df is None or subgraph_df.empty:
        return "<div style='text-align:center; padding: 30px; color: #888;'>No subgraph data to visualize.</div>"

    seed_set: Set[str] = set(s.strip().lower() for s in (seed_names or []))

    # Limit to max_triples for fluid rendering
    df_plot = subgraph_df.drop_duplicates(subset=["head", "relation", "tail"]).head(max_triples)

    net = Network(
        height="500px",
        width="100%",
        bgcolor="#0f172a",
        font_color="#f8fafc",
        directed=True,
    )

    # Physics configuration for smooth dynamic layout
    net.set_options("""
    {
      "nodes": {
        "borderWidth": 2,
        "borderWidthSelected": 4,
        "shadow": true,
        "font": { "size": 14, "face": "system-ui, sans-serif" }
      },
      "edges": {
        "color": { "color": "#94a3b8", "highlight": "#38bdf8", "hover": "#38bdf8" },
        "arrows": { "to": { "enabled": true, "scaleFactor": 0.7 } },
        "smooth": { "type": "continuous", "roundness": 0.4 },
        "font": { "size": 11, "align": "middle", "strokeWidth": 2, "strokeColor": "#0f172a" },
        "shadow": false
      },
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -4000,
          "centralGravity": 0.2,
          "springLength": 120,
          "springConstant": 0.04,
          "damping": 0.09
        },
        "stabilization": { "iterations": 120 }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 200,
        "navigationButtons": true,
        "keyboard": true
      }
    }
    """)

    added_nodes = set()

    for _, row in df_plot.iterrows():
        head = str(row["head"]).strip()
        rel = str(row["relation"]).strip()
        tail = str(row["tail"]).strip()
        source = str(row.get("source", ""))
        page = str(row.get("page", ""))

        # Head node
        if head not in added_nodes:
            is_seed = head.lower() in seed_set
            net.add_node(
                head,
                label=head[:30] + ("..." if len(head) > 30 else ""),
                title=f"Entity: {head}\n{'★ SEED NODE' if is_seed else ''}",
                color="#f59e0b" if is_seed else "#38bdf8",
                size=26 if is_seed else 18,
                shape="dot",
            )
            added_nodes.add(head)

        # Tail node
        if tail not in added_nodes:
            is_seed = tail.lower() in seed_set
            net.add_node(
                tail,
                label=tail[:30] + ("..." if len(tail) > 30 else ""),
                title=f"Entity: {tail}\n{'★ SEED NODE' if is_seed else ''}",
                color="#f59e0b" if is_seed else "#a78bfa",
                size=26 if is_seed else 16,
                shape="dot",
            )
            added_nodes.add(tail)

        # Edge
        net.add_edge(
            head,
            tail,
            label=rel,
            title=f"Relation: {rel}\nSource: {source} (p.{page})",
        )

    return net.generate_html()
