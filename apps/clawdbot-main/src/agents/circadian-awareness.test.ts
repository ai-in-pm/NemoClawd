import { describe, expect, it } from "vitest";
import { detectJobKind, evaluateCircadianDecision } from "./circadian-awareness.js";

describe("circadian-awareness", () => {
  it("classifies purpose-driven work", () => {
    expect(detectJobKind("Investigate production outage and deploy a hotfix")).toBe(
      "purpose-driven",
    );
  });

  it("classifies generic chat as text-based", () => {
    expect(detectJobKind("Hello there, how are you?")).toBe("text-based");
  });

  it("blocks text-based jobs during sleep hours", () => {
    const result = evaluateCircadianDecision("Summarize yesterday notes", new Date("2026-03-15T02:10:00"));
    expect(result.allowed).toBe(false);
    expect(result.phase).toBe("sleep");
    expect(result.kind).toBe("text-based");
  });

  it("allows urgent purpose-driven jobs during sleep hours", () => {
    const result = evaluateCircadianDecision(
      "Critical incident: production down, need hotfix",
      new Date("2026-03-15T02:10:00"),
    );
    expect(result.allowed).toBe(true);
    expect(result.kind).toBe("purpose-driven");
  });
});
