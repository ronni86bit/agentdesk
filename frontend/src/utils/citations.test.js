import { describe, expect, it } from "vitest";
import { extractCitationIds, mapCitationTokens } from "./citations";

describe("extractCitationIds", () => {
  it("finds all unique markers in order", () => {
    expect(extractCitationIds("A [1] and [3] and [1].")).toEqual([1, 3]);
  });

  it("ignores markdown links that look like citations", () => {
    expect(extractCitationIds("[1](https://example.com)")).toEqual([]);
  });

  it("handles empty input", () => {
    expect(extractCitationIds("")).toEqual([]);
    expect(extractCitationIds(undefined)).toEqual([]);
  });
});

describe("mapCitationTokens", () => {
  it("splits text into strings and rendered markers", () => {
    const out = mapCitationTokens("See [2] now.", (id, key) => ({
      marker: id,
      key,
    }));
    expect(out).toHaveLength(3);
    expect(out[0]).toBe("See ");
    expect(out[1]).toEqual({ marker: 2, key: expect.any(String) });
    expect(out[2]).toBe(" now.");
  });

  it("recurses into element children", () => {
    const el = {
      type: "strong",
      props: { children: ["bold [4] text"] },
    };
    const out = mapCitationTokens(el, (id) => ({ marker: id }));
    expect(out.type).toBe("strong");
    expect(out.props.children[0]).toBe("bold ");
    expect(out.props.children[1]).toEqual({ marker: 4 });
  });

  it("returns plain text untouched when no markers exist", () => {
    expect(mapCitationTokens("no markers", () => "X")).toBe("no markers");
  });
});
