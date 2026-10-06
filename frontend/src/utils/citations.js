// Citation-marker handling for report rendering.

// [N] markers — but not markdown links like [1](url), mirroring the backend.
const CITATION_TOKEN = /\[(\d{1,3})\](?!\()/g;

/**
 * Recursively transform React children, replacing [N] citation markers in
 * text nodes with the result of `renderMarker(id, key)`. Element children
 * are recursed into so markers inside bold/italic/link/code nodes are
 * still rendered as citations. Structure and other content are preserved.
 */
export function mapCitationTokens(children, renderMarker) {
  return mapChildren(children, renderMarker, 0).nodes;
}

function mapChildren(children, renderMarker, depth) {
  if (children === null || children === undefined || depth > 20) {
    return { nodes: children };
  }
  if (typeof children === "string") {
    return splitText(children, renderMarker);
  }
  if (typeof children === "number") {
    return { nodes: String(children) };
  }
  if (Array.isArray(children)) {
    const out = [];
    let keySeed = 0;
    for (const child of children) {
      const { nodes } = mapChildren(child, renderMarker, depth + 1);
      if (Array.isArray(nodes)) {
        for (const node of nodes) {
          out.push(node);
          keySeed += 1;
        }
      } else {
        out.push(nodes);
        keySeed += 1;
      }
    }
    return { nodes: out };
  }
  if (typeof children === "object" && "type" in children) {
    const element = children;
    const { nodes: mappedPropsChildren } = mapChildren(
      element.props?.children,
      renderMarker,
      depth + 1,
    );
    if (mappedPropsChildren === element.props?.children) {
      return { nodes: element };
    }
    return {
      nodes: {
        ...element,
        props: { ...element.props, children: mappedPropsChildren },
      },
    };
  }
  return { nodes: children };
}

function splitText(text, renderMarker) {
  if (!text.includes("[")) return { nodes: text };
  const nodes = [];
  let lastIndex = 0;
  let match;
  CITATION_TOKEN.lastIndex = 0;
  let counter = 0;
  while ((match = CITATION_TOKEN.exec(text)) !== null) {
    if (match.index > lastIndex) {
      nodes.push(text.slice(lastIndex, match.index));
    }
    const id = Number(match[1]);
    nodes.push(renderMarker(id, `cite-${match.index}-${counter}`));
    counter += 1;
    lastIndex = match.index + match[0].length;
  }
  if (lastIndex < text.length) {
    nodes.push(text.slice(lastIndex));
  }
  return { nodes: nodes.length === 1 ? nodes[0] : nodes };
}

/**
 * Extract all citation IDs used in a markdown string (for tests/analysis).
 */
export function extractCitationIds(markdown) {
  const ids = new Set();
  for (const match of String(markdown ?? "").matchAll(CITATION_TOKEN)) {
    ids.add(Number(match[1]));
  }
  return [...ids].sort((a, b) => a - b);
}
