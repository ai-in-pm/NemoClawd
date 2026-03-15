export type JobKind = "text-based" | "purpose-driven";

export type CircadianDecision = {
  kind: JobKind;
  allowed: boolean;
  localHour: number;
  phase: "sleep" | "startup" | "focus" | "winddown";
  reason: string;
};

const PURPOSE_KEYWORDS = [
  "build",
  "deploy",
  "ship",
  "release",
  "incident",
  "outage",
  "deadline",
  "production",
  "bug",
  "fix",
  "urgent",
  "critical",
  "migration",
  "security",
  "hotfix",
  "rollback",
  "customer",
  "payment",
  "data loss",
  "breach",
  "postmortem",
  "on-call",
  "oncall",
  "sprint",
  "milestone",
  "risk",
  "blocker",
];

const URGENCY_KEYWORDS = [
  "urgent",
  "critical",
  "asap",
  "p0",
  "sev1",
  "sev-1",
  "incident",
  "outage",
  "production down",
  "data loss",
  "security",
  "breach",
  "hotfix",
];

function includesAny(text: string, keywords: string[]): boolean {
  return keywords.some((keyword) => text.includes(keyword));
}

function resolvePhase(hour: number): CircadianDecision["phase"] {
  if (hour >= 22 || hour < 6) return "sleep";
  if (hour < 9) return "startup";
  if (hour < 18) return "focus";
  return "winddown";
}

export function detectJobKind(message: string): JobKind {
  const normalized = message.toLowerCase();
  return includesAny(normalized, PURPOSE_KEYWORDS) ? "purpose-driven" : "text-based";
}

export function evaluateCircadianDecision(message: string, now = new Date()): CircadianDecision {
  const kind = detectJobKind(message);
  const localHour = now.getHours();
  const phase = resolvePhase(localHour);
  const normalized = message.toLowerCase();
  const urgent = includesAny(normalized, URGENCY_KEYWORDS);

  if (phase === "sleep" && kind === "text-based") {
    return {
      kind,
      allowed: false,
      localHour,
      phase,
      reason:
        "Circadian guard: deferring non-purpose text work during sleep hours. Retry between 06:00-22:00 local time or mark the task as urgent and purpose-driven.",
    };
  }

  if (phase === "sleep" && kind === "purpose-driven" && !urgent) {
    return {
      kind,
      allowed: false,
      localHour,
      phase,
      reason:
        "Circadian guard: purpose-driven work is allowed overnight only for urgent tasks. Add urgency context (for example: incident, critical, urgent) to continue.",
    };
  }

  return {
    kind,
    allowed: true,
    localHour,
    phase,
    reason: `Circadian guard: ${kind} job accepted during ${phase} phase.`,
  };
}
