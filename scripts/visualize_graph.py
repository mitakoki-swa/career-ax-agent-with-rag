"""BYOG / LLM生成Graphを、依存なしの対話型HTMLへ書き出す。"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from graph_core import DATA_DIR, load_projects, read_csv


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "graph_visualizations"


def build_visualization_payload(data_dir: Path, source: str) -> dict[str, object]:
    graph_dir = data_dir if source == "byog" else data_dir / "llm_generated"
    node_rows = read_csv(graph_dir / "nodes.csv")
    edge_rows = read_csv(graph_dir / "edges.csv")
    tag_rows = read_csv(graph_dir / "tags.csv")
    projects = load_projects(data_dir)

    projects_by_node: dict[str, set[str]] = defaultdict(set)
    tags_by_project: dict[str, set[str]] = defaultdict(set)
    for row in tag_rows:
        project_id = (row.get("案件ID") or "").strip()
        node_id = (row.get("ノードID") or "").strip()
        if project_id and node_id:
            projects_by_node[node_id].add(project_id)
            tags_by_project[project_id].add(node_id)

    nodes: list[dict[str, object]] = []
    known_node_ids: set[str] = set()
    for row in node_rows:
        node_id = (row.get("ノードID") or "").strip()
        if not node_id:
            continue
        known_node_ids.add(node_id)
        nodes.append(
            {
                "id": node_id,
                "label": (row.get("正規名") or node_id).strip(),
                "kind": (row.get("種別") or "概念").strip(),
                "description": (row.get("説明") or "").strip(),
                "projectIds": sorted(projects_by_node.get(node_id, set())),
                "isProject": False,
            }
        )

    edges: list[dict[str, object]] = []
    relation_counts: dict[str, int] = defaultdict(int)
    for row in edge_rows:
        source_id = (row.get("出発ノードID") or "").strip()
        target_id = (row.get("到着ノードID") or "").strip()
        if source_id not in known_node_ids or target_id not in known_node_ids:
            continue
        relation = (row.get("関係") or "関連").strip()
        relation_counts[relation] += 1
        edges.append(
            {
                "id": (row.get("エッジID") or "").strip(),
                "source": source_id,
                "target": target_id,
                "relation": relation,
                "weight": float((row.get("強さ") or "1").strip() or "1"),
                "isTag": False,
            }
        )

    tag_count = 0
    for project_id, node_ids in sorted(tags_by_project.items()):
        project = projects.get(project_id)
        if project is None:
            continue
        visual_id = f"PROJECT:{project_id}"
        nodes.append(
            {
                "id": visual_id,
                "label": f"{project_id} {project.name}",
                "kind": "案件",
                "description": project.document(),
                "projectIds": [project_id],
                "isProject": True,
            }
        )
        for node_id in sorted(node_ids & known_node_ids):
            tag_count += 1
            edges.append(
                {
                    "id": f"TAG:{project_id}:{node_id}",
                    "source": visual_id,
                    "target": node_id,
                    "relation": "案件タグ",
                    "weight": 1.0,
                    "isTag": True,
                }
            )

    kinds = sorted({str(node["kind"]) for node in nodes})
    return {
        "source": source,
        "title": "BYOG（人手オントロジー）" if source == "byog" else "LLM生成Graph",
        "nodes": nodes,
        "edges": edges,
        "kinds": kinds,
        "relations": sorted(relation_counts),
        "stats": {
            "conceptNodes": len(known_node_ids),
            "projectNodes": sum(bool(node["isProject"]) for node in nodes),
            "graphEdges": len(edge_rows),
            "tagEdges": tag_count,
            "relationCounts": dict(sorted(relation_counts.items())),
        },
    }


def render_html(payload: dict[str, object]) -> str:
    graph_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return (
        HTML_TEMPLATE.replace("__GRAPH_JSON__", graph_json)
        .replace("__GRAPH_TITLE__", str(payload["title"]))
        .replace("__GRAPH_SOURCE__", str(payload["source"]).upper())
    )


def write_visualization(data_dir: Path, output_dir: Path, source: str) -> Path:
    payload = build_visualization_payload(data_dir, source)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{source}.html"
    path.write_text(render_html(payload), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Knowledge GraphをHTMLで可視化する")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--source", choices=("byog", "llm", "both"), default="both"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    sources = ("byog", "llm") if args.source == "both" else (args.source,)
    for source in sources:
        path = write_visualization(args.data_dir, args.output_dir, source)
        print(f"{source.upper()}: {path}")
    return 0


HTML_TEMPLATE = r"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__GRAPH_TITLE__</title>
  <style>
    :root { color-scheme: dark; --bg:#111418; --panel:#191e24; --line:#3a434d; --text:#e7edf3; --muted:#9ba8b5; --accent:#63b3ed; }
    * { box-sizing: border-box; }
    body { margin:0; height:100vh; overflow:hidden; font:13px/1.45 system-ui, sans-serif; color:var(--text); background:var(--bg); }
    header { height:56px; display:flex; align-items:center; gap:18px; padding:0 18px; border-bottom:1px solid var(--line); background:var(--panel); }
    h1 { margin:0; font-size:17px; font-weight:650; }
    .badge { padding:3px 8px; border:1px solid var(--line); border-radius:999px; color:var(--accent); font-size:11px; }
    #stats { color:var(--muted); }
    main { display:grid; grid-template-columns:250px 1fr 290px; height:calc(100vh - 56px); }
    aside { overflow:auto; padding:15px; background:var(--panel); }
    aside.left { border-right:1px solid var(--line); }
    aside.right { border-left:1px solid var(--line); }
    section { margin-bottom:18px; }
    h2 { margin:0 0 9px; color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.08em; }
    input[type="search"] { width:100%; padding:8px 10px; color:var(--text); background:var(--bg); border:1px solid var(--line); border-radius:5px; }
    label { display:flex; align-items:center; gap:7px; margin:5px 0; cursor:pointer; }
    button { padding:7px 10px; color:var(--text); background:var(--bg); border:1px solid var(--line); border-radius:5px; cursor:pointer; }
    button:hover { border-color:var(--accent); }
    .button-row { display:flex; gap:7px; }
    #viewport { position:relative; min-width:0; }
    canvas { display:block; width:100%; height:100%; cursor:grab; }
    canvas.dragging { cursor:grabbing; }
    .hint { position:absolute; left:12px; bottom:10px; padding:5px 8px; color:var(--muted); background:rgba(17,20,24,.82); border-radius:4px; pointer-events:none; }
    .swatch { width:10px; height:10px; border-radius:50%; flex:0 0 auto; }
    .count { margin-left:auto; color:var(--muted); font-variant-numeric:tabular-nums; }
    #details h3 { margin:0 0 5px; font-size:16px; }
    #details .id { color:var(--muted); overflow-wrap:anywhere; }
    #details .kind { display:inline-block; margin:8px 0; padding:2px 7px; border:1px solid var(--line); border-radius:999px; }
    #details p { white-space:pre-wrap; overflow-wrap:anywhere; }
    #details ul { margin:7px 0; padding-left:18px; }
    .empty { color:var(--muted); }
  </style>
</head>
<body>
  <header>
    <h1>__GRAPH_TITLE__</h1><span class="badge">__GRAPH_SOURCE__</span><span id="stats"></span>
  </header>
  <main>
    <aside class="left">
      <section><h2>検索</h2><input id="search" type="search" placeholder="ノード名・ID・案件ID"></section>
      <section><h2>表示</h2>
        <label><input id="showProjects" type="checkbox"> 案件ノードとタグ線</label>
        <label><input id="neighborsOnly" type="checkbox"> 選択ノードの周辺だけ</label>
      </section>
      <section><h2>ノード種別</h2><div id="kindFilters"></div></section>
      <section><h2>エッジ種別</h2><div id="relationFilters"></div></section>
      <section><h2>操作</h2><div class="button-row"><button id="fit">全体表示</button><button id="reset">選択解除</button></div></section>
    </aside>
    <div id="viewport"><canvas id="graph"></canvas><div class="hint">ホイール: 拡大縮小 / ドラッグ: 移動 / ノードをクリック: 詳細</div></div>
    <aside class="right"><section><h2>ノード詳細</h2><div id="details" class="empty">ノードを選択してください。</div></section></aside>
  </main>
  <script>
    const DATA = __GRAPH_JSON__;
    const COLORS = {"職種":"#63b3ed","業務":"#68d391","スキル":"#f6ad55","ドメイン":"#b794f4","クラウド":"#4fd1c5","案件":"#f56565","概念":"#a0aec0","その他":"#cbd5e0"};
    const RELATION_COLORS = {"包含":"#63b3ed","関連":"#68d391","使用":"#f6ad55","案件タグ":"#59636e"};
    const canvas = document.getElementById("graph"), ctx = canvas.getContext("2d"), viewport = document.getElementById("viewport");
    const state = {scale:1, offsetX:0, offsetY:0, selected:null, hover:null, dragging:null, panning:false, lastX:0, lastY:0};
    const nodeMap = new Map(DATA.nodes.map(n => [n.id, {...n, x:0, y:0}]));
    const adjacency = new Map(DATA.nodes.map(n => [n.id, new Set()]));
    DATA.edges.forEach(e => { if(adjacency.has(e.source)&&adjacency.has(e.target)){ adjacency.get(e.source).add(e.target); adjacency.get(e.target).add(e.source); }});
    const hash = s => [...s].reduce((a,c)=>((a*31+c.charCodeAt(0))>>>0),2166136261);
    const color = kind => COLORS[kind] || "#a0aec0";
    const relationColor = relation => RELATION_COLORS[relation] || "#718096";

    function addFilters() {
      const kindCounts = Object.fromEntries(DATA.kinds.map(k => [k, DATA.nodes.filter(n=>n.kind===k).length]));
      document.getElementById("kindFilters").innerHTML = DATA.kinds.map(k =>
        `<label><input type="checkbox" data-kind="${escapeAttr(k)}" checked><span class="swatch" style="background:${color(k)}"></span>${escapeHtml(k)}<span class="count">${kindCounts[k]}</span></label>`).join("");
      document.getElementById("relationFilters").innerHTML = DATA.relations.map(r =>
        `<label><input type="checkbox" data-relation="${escapeAttr(r)}" checked><span class="swatch" style="background:${relationColor(r)}"></span>${escapeHtml(r)}<span class="count">${DATA.edges.filter(e=>!e.isTag&&e.relation===r).length}</span></label>`).join("");
      document.querySelectorAll("input").forEach(el => el.addEventListener("input", refresh));
    }
    function escapeHtml(v){ const d=document.createElement("div"); d.textContent=v; return d.innerHTML; }
    function escapeAttr(v){ return escapeHtml(v).replaceAll('"',"&quot;"); }

    function visibleGraph() {
      const kinds = new Set([...document.querySelectorAll("[data-kind]:checked")].map(x=>x.dataset.kind));
      const relations = new Set([...document.querySelectorAll("[data-relation]:checked")].map(x=>x.dataset.relation));
      const showProjects = document.getElementById("showProjects").checked;
      const query = document.getElementById("search").value.trim().toLowerCase();
      let ids = new Set(DATA.nodes.filter(n => kinds.has(n.kind) && (showProjects || !n.isProject)).map(n=>n.id));
      if(query) {
        const matches = new Set(DATA.nodes.filter(n =>
          `${n.id} ${n.label} ${n.description} ${n.projectIds.join(" ")}`.toLowerCase().includes(query)).map(n=>n.id));
        const expanded = new Set(matches);
        matches.forEach(id => (adjacency.get(id)||[]).forEach(other=>expanded.add(other)));
        ids = new Set([...ids].filter(id=>expanded.has(id)));
      }
      if(state.selected && document.getElementById("neighborsOnly").checked) {
        const near = new Set([state.selected, ...(adjacency.get(state.selected)||[])]);
        ids = new Set([...ids].filter(id=>near.has(id)));
      }
      const nodes = [...ids].map(id=>nodeMap.get(id));
      const edges = DATA.edges.filter(e => ids.has(e.source) && ids.has(e.target) && (e.isTag ? showProjects : relations.has(e.relation)));
      return {nodes, edges};
    }

    function layout(nodes) {
      const groups = new Map();
      nodes.forEach(n => { if(!groups.has(n.kind)) groups.set(n.kind, []); groups.get(n.kind).push(n); });
      const width = Math.max(viewport.clientWidth, 600), height = Math.max(viewport.clientHeight, 500);
      const entries = [...groups.entries()].sort((a,b)=>a[0].localeCompare(b[0],"ja"));
      entries.forEach(([kind, items], groupIndex) => {
        const groupAngle = Math.PI*2*groupIndex/Math.max(entries.length,1) - Math.PI/2;
        const groupRadius = entries.length === 1 ? 0 : Math.min(width,height)*.29;
        const centerX = width/2 + Math.cos(groupAngle)*groupRadius;
        const centerY = height/2 + Math.sin(groupAngle)*groupRadius;
        const localRadius = Math.min(115, 24 + Math.sqrt(items.length)*10);
        items.sort((a,b)=>a.id.localeCompare(b.id,"ja"));
        items.forEach((n,i) => {
          const jitter = (hash(n.id)%100)/500;
          const angle = Math.PI*2*i/Math.max(items.length,1) + jitter + groupIndex*.4;
          n.x = centerX + Math.cos(angle)*localRadius;
          n.y = centerY + Math.sin(angle)*localRadius;
        });
      });
    }

    function resize() {
      const ratio = window.devicePixelRatio || 1;
      canvas.width = Math.floor(viewport.clientWidth * ratio); canvas.height = Math.floor(viewport.clientHeight * ratio);
      canvas.style.width = `${viewport.clientWidth}px`; canvas.style.height = `${viewport.clientHeight}px`;
      ctx.setTransform(ratio,0,0,ratio,0,0); draw();
    }
    function screen(n){ return {x:n.x*state.scale+state.offsetX, y:n.y*state.scale+state.offsetY}; }
    function draw() {
      const {nodes, edges} = visibleGraph(), ids = new Set(nodes.map(n=>n.id));
      ctx.clearRect(0,0,viewport.clientWidth,viewport.clientHeight);
      edges.forEach(e => {
        const a=screen(nodeMap.get(e.source)), b=screen(nodeMap.get(e.target));
        const active = state.selected && (e.source===state.selected || e.target===state.selected);
        ctx.beginPath(); ctx.moveTo(a.x,a.y); ctx.lineTo(b.x,b.y);
        ctx.strokeStyle = active ? relationColor(e.relation) : (e.isTag ? "#343c45" : "#46515d");
        ctx.globalAlpha = active ? .95 : .35; ctx.lineWidth = active ? 2 : Math.max(.5,e.weight||1); ctx.stroke();
        if(active) {
          const mx=(a.x+b.x)/2, my=(a.y+b.y)/2;
          ctx.globalAlpha=1; ctx.fillStyle="#cbd5e0"; ctx.font="11px system-ui"; ctx.fillText(e.relation,mx+3,my-3);
        }
      });
      ctx.globalAlpha=1;
      nodes.forEach(n => {
        const p=screen(n), selected=n.id===state.selected, hover=n.id===state.hover, neighbor=state.selected && adjacency.get(state.selected)?.has(n.id);
        const radius=(n.isProject?5:7)*(selected?1.5:1);
        ctx.beginPath(); ctx.arc(p.x,p.y,radius,0,Math.PI*2);
        ctx.fillStyle=color(n.kind); ctx.fill();
        if(selected||hover||neighbor){ ctx.strokeStyle=selected?"#ffffff":"#a9c5dc"; ctx.lineWidth=2; ctx.stroke(); }
        if(selected||hover||state.scale>1.15) {
          ctx.font=`${selected?"600 ":""}11px system-ui`; ctx.fillStyle="#e7edf3";
          ctx.fillText(n.label,p.x+radius+4,p.y+4);
        }
      });
      document.getElementById("stats").textContent = `表示 ${nodes.length} ノード / ${edges.length} エッジ　全体 ${DATA.stats.conceptNodes} ノード / ${DATA.stats.graphEdges} エッジ / ${DATA.stats.tagEdges} タグ`;
    }
    function refresh(){ const g=visibleGraph(); layout(g.nodes); showDetails(); draw(); }
    function fit(){ state.scale=1; state.offsetX=0; state.offsetY=0; refresh(); }
    function hitNode(x,y) {
      const nodes=visibleGraph().nodes;
      for(let i=nodes.length-1;i>=0;i--){ const p=screen(nodes[i]); if(Math.hypot(p.x-x,p.y-y)<11) return nodes[i]; }
      return null;
    }
    function showDetails() {
      const box=document.getElementById("details"), n=state.selected&&nodeMap.get(state.selected);
      if(!n){ box.className="empty"; box.textContent="ノードを選択してください。"; return; }
      box.className=""; box.innerHTML="";
      const h=document.createElement("h3"); h.textContent=n.label; box.append(h);
      const id=document.createElement("div"); id.className="id"; id.textContent=n.id; box.append(id);
      const kind=document.createElement("div"); kind.className="kind"; kind.textContent=n.kind; box.append(kind);
      if(n.description){ const p=document.createElement("p"); p.textContent=n.description; box.append(p); }
      const projects=n.projectIds||[]; if(projects.length){ const h2=document.createElement("h2"); h2.textContent=`接続案件 (${projects.length})`; box.append(h2); const ul=document.createElement("ul"); projects.forEach(v=>{const li=document.createElement("li");li.textContent=v;ul.append(li)}); box.append(ul); }
      const neighbors=[...(adjacency.get(n.id)||[])].map(id=>nodeMap.get(id)).filter(Boolean); if(neighbors.length){ const h2=document.createElement("h2"); h2.textContent=`隣接ノード (${neighbors.length})`; box.append(h2); const ul=document.createElement("ul"); neighbors.slice(0,80).forEach(v=>{const li=document.createElement("li");li.textContent=`${v.label} [${v.kind}]`;ul.append(li)}); box.append(ul); }
    }
    canvas.addEventListener("mousedown", e => { const n=hitNode(e.offsetX,e.offsetY); state.lastX=e.offsetX;state.lastY=e.offsetY; if(n){state.dragging=n}else{state.panning=true} canvas.classList.add("dragging"); });
    window.addEventListener("mousemove", e => {
      const r=canvas.getBoundingClientRect(), x=e.clientX-r.left, y=e.clientY-r.top;
      if(state.dragging){state.dragging.x=(x-state.offsetX)/state.scale;state.dragging.y=(y-state.offsetY)/state.scale;draw()}
      else if(state.panning){state.offsetX+=x-state.lastX;state.offsetY+=y-state.lastY;state.lastX=x;state.lastY=y;draw()}
      else { const n=hitNode(x,y); const id=n?.id||null; if(id!==state.hover){state.hover=id;draw()} }
    });
    window.addEventListener("mouseup", () => {state.dragging=null;state.panning=false;canvas.classList.remove("dragging")});
    canvas.addEventListener("click", e => { const n=hitNode(e.offsetX,e.offsetY); if(n){state.selected=n.id;showDetails();draw()} });
    canvas.addEventListener("wheel", e => { e.preventDefault(); const factor=e.deltaY<0?1.12:.89, beforeX=(e.offsetX-state.offsetX)/state.scale, beforeY=(e.offsetY-state.offsetY)/state.scale; state.scale=Math.max(.2,Math.min(4,state.scale*factor)); state.offsetX=e.offsetX-beforeX*state.scale;state.offsetY=e.offsetY-beforeY*state.scale;draw() }, {passive:false});
    document.getElementById("fit").onclick=fit;
    document.getElementById("reset").onclick=()=>{state.selected=null;document.getElementById("neighborsOnly").checked=false;showDetails();draw()};
    window.addEventListener("resize",resize);
    addFilters(); resize(); fit();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
