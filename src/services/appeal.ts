import type { StructuredComplaint } from "../lib/schema.js";

export interface AppealContext {
  location?: string;
  contact?: string;
  channelUrl: string;
}

export function generateAppeal(
  structured: StructuredComplaint,
  context: AppealContext,
): string {
  const lines: string[] = [];
  lines.push("Звернення громадянина до органу місцевого самоврядування");
  lines.push("");
  if (context.channelUrl) {
    lines.push(`Канал надсилання (офіційний): ${context.channelUrl}`);
    lines.push("");
  }
  if (structured.topics.length === 0) {
    lines.push("Проблема не визначена.");
    lines.push("");
  } else {
    structured.topics.forEach((topic, idx) => {
      lines.push(`${idx + 1}. Проблема: ${topic.issue || "—"}`);
      if (topic.object) {
        lines.push(`   Об'єкт: ${topic.object}`);
      }
      if (topic.domain) {
        lines.push(`   Сфера: ${topic.domain}`);
      }
      if (topic.requested_action) {
        lines.push(`   Прошу: ${topic.requested_action}`);
      }
      if (Object.keys(topic.attributes).length > 0) {
        lines.push("   Додаткові відомості:");
        for (const [k, v] of Object.entries(topic.attributes)) {
          if (v) {
            lines.push(`     - ${k}: ${v}`);
          }
        }
      }
      lines.push("");
    });
  }
  if (context.location) {
    lines.push(`Місце порушення: ${context.location}`);
    lines.push("");
  }
  if (context.contact) {
    lines.push(`Контактні дані для відповіді: ${context.contact}`);
    lines.push("");
  }
  lines.push(
    "Звернення підготовлене за допомогою асистента. Кожне твердження перевірено користувачем перед надсиланням.",
  );
  return lines.join("\n");
}
